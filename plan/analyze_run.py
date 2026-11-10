"""Read-only analysis of an agent.py run folder (runs/<name>): plans.jsonl, states.jsonl, events.jsonl and,
when present, times.jsonl.

    python -m plan.analyze_run runs/plan_20261103-152013 [--json out.json] [--level 1-11]
    python -m plan.analyze_run latest

Works on a run that is still being written (a malformed line, e.g. a half-written last one, is skipped and
counted). The files are streamed; of each plan only the fields needed here are kept.

Played actions are reconstructed from the plans: each plan plays its actions from its frame until the agent's
next plan (the real replan interval, taken from consecutive plan frames, not assumed); an agent's last plan
in a wave plays until its last alive frame. A plan is followed for h frames when the
actions played on its frames f .. f+h-1 equal its actions[0:h].

Reported (text, and the same numbers with --json): decisions and the airborne share (start not supported);
ms per stage (median / 90%); stage-2 steps rolled / needed; frames each plan was followed, and each committed
plan (a decision that did not keep the previous plan: from_prev false), through the decisions that kept it;
the decisions' triggers by kind (plans.jsonl "trigger", plan/commit.py), the switch rate (decisions that
switched plans / all decisions) and the gains the kept plans were kept against (plans.jsonl "switch"; runs
before these fields report none); the share of decisions whose next 16 actions differ from the previous plan's
remaining ones (all, airborne); the score margin chosen - best rejected, among plans that do not touch the
goal (frames: runs scored by the estimated frames to the goal, score.estimate; older runs V units; successes,
total = SUCCESS_SCORE - touch frame, are compared with the best rejected success by their touch frames instead:
success_margin, in frames); uncertainty factor shares (older runs only); plans ending supported; chosen plans
completed in the air (Plan.completion) and landing (score.landing); the share of decisions whose chosen plan
the collision guard clipped (plans.jsonl "model_clipped"; older runs: none); the position error per
horizon for followed and not followed predictions (alive agents only); per level: decisions, no-progress
share (frames: T(end) >= T(start), the time-to-go did not shrink; older runs: progress <= 0.05 V units), dead-end ends, the most common first-move families, deaths per cause, finishes and
their times (times.jsonl or finish events; for older runs without them, the wave length of a 'quota' wave a
planned agent finished, labelled as an estimate). Still share (overall and per level): decisions whose chosen plan
has no moves other than wait / pad (is_still), and the share of decisions that kept such a plan (from_prev).

Per level also the decisions that replayed the recorded best route (plans.jsonl best_route.used, `replayed`) and
`wave_list`: the level's completed waves in log order (a wave event ends a wave; the wave a level switch or a stop
interrupted has none and is not listed, though its decisions count above), each with its end (why, finished,
frames, agent_frames), its decisions and replayed decisions, whether a recorded demo was replayed (a "demo
check" event), its planned finishes (frame, seconds), its deaths (x, y, frame) and, for a wave that ended without
either, the planned agent's last logged position. Plans and states are matched to the wave events per level in
order by wave number: a plan or state segment without a wave event (interrupted) is skipped. `models`: the world
models the run logged (agent.py's level or start event "model", --learn's "learn model" path); empty when the run
did not record one.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from plan.commit import kind_of                    # noqa: E402
RUNS = os.path.join(HERE, "runs")
HORIZONS = (1, 4, 8, 16, 24, 32, 48)
CHANGE_WINDOW = 16                 # actions compared for "plan changed"
NO_PROGRESS = 0.0                  # frames: no progress when T(start) - T(end) is not above it
NO_PROGRESS_V = 0.05               # older runs (V units)
NEAR_TIE = 0.1                     # V units (older runs); frames: NEAR_TIE_FRAMES
SUCCESS_SCORE = 1e5                # plan.planner.SUCCESS_SCORE: a success's total is SUCCESS_SCORE - touch frame
NEAR_TIE_FRAMES = 3                # successes whose touch frames are this close (plan.planner.SUCCESS_MARGIN)
FRAME_MS = 16.667                  # the locked time step (older runs' finishes have no dt)
TOP_FAMILIES = 5
STILL_MOVES = ("wait",)            # move-name prefixes that do nothing (plan.moves: "wait 8f", "wait 16f")


def resolve_run(arg: str, runs_dir: str = RUNS) -> str:
    """'latest' = the newest runs/plan_* folder (by name); else a folder under runs/ (smbrl/runs, whatever the
    current folder), then under smbrl/, then a path as given."""
    if arg == "latest":
        cands = sorted(d for d in glob.glob(os.path.join(runs_dir, "plan_*")) if os.path.isdir(d))
        if not cands:
            raise SystemExit(f"no plan_* run folder in {runs_dir}")
        return cands[-1]
    for p in (os.path.join(runs_dir, arg), os.path.join(HERE, arg), arg):     # smbrl/runs first, from any folder
        if os.path.isdir(p):
            return os.path.abspath(p)
    raise SystemExit(f"no run folder {arg!r}")


def stream(path: str, bad: Counter):
    """JSON lines of a file; malformed lines are skipped and counted in bad[file name]."""
    if not os.path.exists(path):
        return
    name = os.path.basename(path)
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except ValueError:
                bad[name] += 1


class Segments:
    """(level, wave, occurrence): a wave's lines are contiguous in each log; the same (level, wave) seen again
    later (the level played again) is a new occurrence. Applied to plans and states alike, so keys match."""

    def __init__(self):
        self.last, self.last_frame, self.occ = None, None, Counter()

    def key(self, level, wave, frame) -> tuple:
        k = (level, wave)
        if k != self.last or (frame is not None and self.last_frame is not None and frame < self.last_frame):
            self.occ[k] += 1
            self.last = k
        self.last_frame = frame
        return (level, wave, self.occ[k])


def family(name: str) -> str:
    """A move name without its numbers: 'run-jump R hold 10 after 4f' -> 'run-jump R', 'wait 8f' -> 'wait'."""
    out, skip = [], False
    for t in name.split():
        if skip:
            skip = False
            continue
        if t in ("hold", "after"):
            skip = True
            continue
        if re.fullmatch(r"\d+f?", t):
            continue
        out.append(t)
    return " ".join(out)


def q(a, ps=(50, 90)):
    a = np.asarray([v for v in a if v is not None], float)
    return [round(float(np.percentile(a, p)), 3) for p in ps] if len(a) else [None] * len(ps)


def share(n, d):
    return round(n / d, 4) if d else None


def is_still(chosen) -> bool:
    """A chosen plan that does nothing: no moves other than wait (chosen lists no pads; empty: pad only)."""
    return all(nm.startswith(STILL_MOVES) for nm in chosen)


def slim(d: dict) -> dict:
    """The fields of a plan record this analysis uses."""
    sc = d.get("score") or {}
    alts = [a.get("total") for a in d.get("alternatives") or []]
    alts = [a for a in alts if a is not None and math.isfinite(a)]
    total = sc.get("total")
    succ = total is not None and total > SUCCESS_SCORE / 2
    rivals = [a for a in alts if (a > SUCCESS_SCORE / 2) == succ]      # successes against successes only
    sw = d.get("switch")
    path = d.get("path") or []
    c2 = (d.get("counts") or {}).get("stage2") or {}
    chosen = d.get("chosen") or []
    frames = "estimate" in sc                                          # scored in frames (else older V units)
    if frames:                                     # T at the start minus T at the end (frames): <= 0, no progress
        t0, t1 = sc.get("t_start"), sc.get("t_end")
        progress = None if t0 is None or t1 is None else t0 - t1
    else:
        progress = sc.get("progress")
    return {"frame": int(d["frame"]), "actions": [int(a) for a in d.get("actions") or []],
            "path": {h: path[h] for h in HORIZONS if h < len(path) and path[h] is not None},
            "air": not bool((d.get("state") or {}).get("supported")),
            "end_supported": bool(d.get("end_supported")), "unc": sc.get("uncertainty"),
            "progress": progress, "frames": frames, "total": total, "best_alt": max(rivals) if rivals else None,
            "completed": d.get("completion") is not None, "landing": sc.get("landing"),
            "success": succ, "from_prev": bool(d.get("from_prev")), "trigger": d.get("trigger"),
            "switch": None if not sw else {"kept": bool(sw.get("kept")), "gain": sw.get("gain"),
                                           "unit": sw.get("unit")},
            "dead_end": d.get("v_end") is None, "family": family(chosen[0]) if chosen else "(none)",
            "timing": d.get("timing") or {}, "clipped": d.get("model_clipped"),
            "s2": c2["steps"] / c2["of_steps"] if c2.get("of_steps") else None,
            "mode": d.get("mode") or "normal", "still": is_still(chosen),
            "replay": bool((d.get("best_route") or {}).get("used")),
            "mode_ran": {k: bool(v.get("ran")) for k, v in ((d.get("counts") or {}).get("modes") or {}).items()
                         if isinstance(v, dict)}}


def analyze(run: str, level: str | None = None) -> dict:
    bad = Counter()
    plans = defaultdict(list)                       # (segment, agent) -> slim plans in frame order
    seg = Segments()
    for d in stream(os.path.join(run, "plans.jsonl"), bad):
        if level and d.get("level") != level:
            continue
        k = seg.key(d.get("level"), d.get("wave"), d.get("frame"))
        plans[(k, int(d.get("agent", 0)))].append(slim(d))
    states = defaultdict(dict)                      # (segment, agent) -> {frame: (x, y, status)}
    seg = Segments()
    for d in stream(os.path.join(run, "states.jsonl"), bad):
        if level and d.get("level") != level:
            continue
        k = seg.key(d.get("level"), d.get("wave"), d.get("frame"))
        for a in d.get("agents") or []:
            states[(k, int(a[0]))][int(d["frame"])] = (float(a[1]), float(a[2]), int(a[5]))

    T = defaultdict(list)
    M = defaultdict(lambda: {"chosen": 0, "own_ms": [], "total_ms": []})     # planning mode -> decisions, ms
    err_f, err_u = defaultdict(list), defaultdict(list)
    n = n_air = 0
    changed = changed_air = with_prev = with_prev_air = 0
    margins, unc, s_margins = [], [], []
    committed = []                                  # frames each committed plan (not kept from the previous) was followed
    triggers, n_trig = Counter(), 0
    n_switch = n_kept = 0
    kept_gain = defaultdict(list)                   # unit -> gains the kept plans were kept against
    end_sup = n_completed = n_landing = n_landing_known = 0
    units = Counter()                               # decisions scored in frames / V
    n_clip_known = n_clip = 0                       # decisions with model_clipped (older runs: none)
    n_still = n_still_kept = 0                      # chosen plans that do nothing (is_still); of them kept from_prev
    lv = defaultdict(lambda: {"decisions": 0, "no_progress": 0, "dead_end": 0, "still": 0, "still_kept": 0,
                              "replayed": 0, "families": Counter()})
    seg_dec = defaultdict(lambda: [0, 0])          # segment -> [decisions, replayed]
    for (k, agent), ps in plans.items():
        ps.sort(key=lambda p: p["frame"])
        real = states.get((k, agent), {})
        alive = [f for f, r in real.items() if r[2] == 0]
        last_frame = max(alive) if alive else ps[-1]["frame"]
        played = {}
        for p, nxt in zip(ps, ps[1:] + [None]):
            end = nxt["frame"] if nxt is not None else last_frame + 1
            if nxt is not None:
                T["interval"].append(nxt["frame"] - p["frame"])
            for f in range(p["frame"], min(end, p["frame"] + len(p["actions"]))):
                played[f] = p["actions"][f - p["frame"]]
        prev = None
        for p in ps:
            f, acts = p["frame"], p["actions"]
            n += 1
            n_air += p["air"]
            for key, v in p["timing"].items():
                T["ms_" + key].append(v)
            tm = p["timing"]
            M[p["mode"]]["chosen"] += 1
            M[p["mode"]]["total_ms"].append(tm.get("total"))
            M["normal"]["own_ms"].append((tm.get("stage1") or 0) + (tm.get("stage2") or 0))
            for tk, nm in (("long_safe", "long-safe"), ("reflex", "reflex")):
                if p["mode_ran"].get(tk):
                    M[nm]["own_ms"].append(tm.get(tk))
            if p["s2"] is not None:
                T["s2"].append(p["s2"])
            h = 0
            while h < len(acts) and played.get(f + h) == acts[h]:
                h += 1
            T["follow"].append(h)
            if not p["from_prev"]:
                committed.append(h)
            if p["trigger"] is not None:
                n_trig += 1
                for t in p["trigger"]:
                    triggers[kind_of(t)] += 1
            if p["switch"] is not None:
                if p["switch"]["kept"]:
                    n_kept += 1
                    if p["switch"]["gain"] is not None:
                        kept_gain[p["switch"]["unit"] or "V"].append(p["switch"]["gain"])
                else:
                    n_switch += 1
            for hz, (px, py) in p["path"].items():
                r = real.get(f + hz)
                if r is None or r[2] != 0:
                    continue
                (err_f if h >= hz else err_u)[hz].append(math.hypot(px - r[0], py - r[1]))
            if prev is not None:
                dd = f - prev["frame"]
                old, new = prev["actions"][dd:dd + CHANGE_WINDOW], acts[:CHANGE_WINDOW]
                m = min(len(old), len(new))
                if m:
                    c = old[:m] != new[:m]
                    with_prev += 1
                    changed += c
                    if p["air"]:
                        with_prev_air += 1
                        changed_air += c
            if p["total"] is not None and p["best_alt"] is not None:
                (s_margins if p["success"] else margins).append(p["total"] - p["best_alt"])
            if p["unc"] is not None:
                unc.append(p["unc"])
            end_sup += p["end_supported"]
            units["frames" if p["frames"] else "V"] += 1
            n_completed += p["completed"]
            if p["landing"] is not None:
                n_landing_known += 1
                n_landing += int(p["landing"])
            if p["clipped"] is not None:
                n_clip_known += 1
                n_clip += int(p["clipped"] > 0)
            L = lv[k[0]]
            L["decisions"] += 1
            L["no_progress"] += int(p["progress"] is not None and
                                    p["progress"] <= (NO_PROGRESS if p["frames"] else NO_PROGRESS_V))
            L["dead_end"] += int(p["dead_end"])
            L["still"] += int(p["still"])
            L["still_kept"] += int(p["still"] and p["from_prev"])
            L["replayed"] += int(p["replay"])
            seg_dec[k][0] += 1
            seg_dec[k][1] += int(p["replay"])
            n_still += int(p["still"])
            n_still_kept += int(p["still"] and p["from_prev"])
            L["families"][p["family"]] += 1
            prev = p
    segs = {}                                       # segment -> the lowest agent's last logged (frame, x, y), in log order
    for (k, agent) in sorted(states, key=lambda ka: ka[1]):
        if k not in segs and states[(k, agent)]:
            f = max(states[(k, agent)])
            segs[k] = (f, *states[(k, agent)][f][:2])
    order = list(dict.fromkeys([k for k, _ in states] + [k for k, _ in plans]))
    del plans, states

    # events and times: deaths per cause, finishes
    deaths = defaultdict(Counter)
    finishes = defaultdict(list)                    # level -> [{wave, agent, frames, seconds, source}]
    have_times = os.path.exists(os.path.join(run, "times.jsonl"))
    if have_times:
        for t in stream(os.path.join(run, "times.jsonl"), bad):
            if t.get("source") == "planned" and (not level or t.get("level") == level):
                finishes[t.get("level")].append({"wave": t.get("wave"), "agent": t.get("agent"),
                                                 "frames": t.get("frames"), "seconds": t.get("seconds"),
                                                 "lock": t.get("lock"), "from": "times.jsonl"})
    waves = defaultdict(lambda: {"waves": 0, "finished": 0, "why": Counter()})
    lock = None
    fin_in_wave = 0
    models = []                                     # world models the run logged, in order
    wave_list = defaultdict(list)                   # level -> its completed waves in log order
    pending = defaultdict(lambda: {"finishes": [], "deaths": [], "demo": False})   # (level, wave) -> outcomes
    for e in stream(os.path.join(run, "events.jsonl"), bad):
        kd = e.get("kind")
        if kd == "lock":
            lock = bool(e.get("ok"))
        m = e.get("model") if kd in ("level", "start") else e.get("path") if kd == "learn model" else None
        if m and m not in models:
            models.append(m)
        if level and e.get("level") != level:
            continue
        L = e.get("level")
        if kd == "death":
            deaths[L][e.get("cause") or "?"] += 1
            pending[(L, e.get("wave"))]["deaths"].append([e.get("x"), e.get("y"), e.get("frame")])
        elif kd == "demo check":
            pending[(L, e.get("wave"))]["demo"] = True
        elif kd == "finish":
            pending[(L, e.get("wave"))]["finishes"].append([e.get("frames", e.get("frame")), e.get("seconds")])
            fin_in_wave += 1
            if not have_times:
                frames = e.get("frames", e.get("frame"))     # older events: the message frame = updates played
                secs = e.get("seconds")
                if secs is None and frames is not None:
                    secs = round(frames * FRAME_MS / 1000.0, 3)
                finishes[L].append({"wave": e.get("wave"), "agent": e.get("agent"), "frames": frames,
                                    "seconds": secs, "lock": e.get("lock", lock), "from": "finish event"})
        elif kd == "wave":
            got = pending.pop((L, e.get("wave")), None) or {"finishes": [], "deaths": [], "demo": False}
            for pk in [pk for pk in pending if pk[0] == L]:          # an interrupted wave's outcomes: no wave event
                del pending[pk]
            wave_list[L].append({"wave": e.get("wave"), "why": e.get("why"), "finished": int(e.get("finished") or 0),
                                 "frames": e.get("frames"), "agent_frames": e.get("agent_frames", e.get("frames")),
                                 "decisions": 0, "replayed": 0, "demo": got["demo"], "finishes": got["finishes"],
                                 "deaths": got["deaths"], "end": None})
            W = waves[L]
            W["waves"] += 1
            W["finished"] += int(e.get("finished") or 0)
            W["why"][e.get("why")] += 1
            missing = int(e.get("finished") or 0) - fin_in_wave
            if not have_times and missing > 0 and e.get("why") == "quota" and e.get("frames"):
                fr = int(e["frames"])
                finishes[L].append({"wave": e.get("wave"), "agent": None, "frames": fr,
                                    "seconds": round(fr * FRAME_MS / 1000.0, 3), "lock": lock,
                                    "from": "estimate: wave length"})
            fin_in_wave = 0

    # plans and states segments -> the wave events, per level in order by wave number (interrupted ones skipped)
    by_level = defaultdict(list)
    for k in order:
        by_level[k[0]].append(k)
    for name, ws in wave_list.items():
        ks, i = by_level.get(name, []), 0
        for w in ws:
            j = next((j for j in range(i, len(ks)) if ks[j][1] == w["wave"]), None)
            if j is None:
                continue
            i = j + 1
            w["decisions"], w["replayed"] = seg_dec[ks[j]] if ks[j] in seg_dec else (0, 0)
            if not w["finishes"] and not w["deaths"] and ks[j] in segs:
                w["end"] = [segs[ks[j]][1], segs[ks[j]][2]]

    unit = "frames" if units["frames"] >= units["V"] and units["frames"] else "V"
    err = {}
    for hz in HORIZONS:
        err[str(hz)] = {"followed": q(err_f[hz]) + [len(err_f[hz])], "not_followed": q(err_u[hz]) + [len(err_u[hz])]}
    per_level = {}
    for name in sorted(set(lv) | set(deaths) | set(finishes) | set(waves), key=lambda s: str(s)):
        L = lv.get(name) or {"decisions": 0, "no_progress": 0, "dead_end": 0, "still": 0, "still_kept": 0,
                             "replayed": 0, "families": Counter()}
        fs = finishes.get(name, [])
        secs = [x["seconds"] for x in fs if x["seconds"] is not None]
        per_level[str(name)] = {
            "decisions": L["decisions"], "no_progress": share(L["no_progress"], L["decisions"]),
            "dead_end": share(L["dead_end"], L["decisions"]),
            "still": share(L["still"], L["decisions"]), "still_kept": share(L["still_kept"], L["decisions"]),
            "still_n": L["still"], "replayed": L["replayed"], "wave_list": wave_list.get(name, []),
            "first_moves": [[f, share(c, L["decisions"])] for f, c in L["families"].most_common(TOP_FAMILIES)],
            "deaths": dict(deaths.get(name, Counter()).most_common()),
            "waves": waves[name]["waves"] if name in waves else 0,
            "wave_end": dict(waves[name]["why"]) if name in waves else {},
            "finishes": fs, "best_s": min(secs) if secs else None}
    return {
        "run": run, "level": level, "skipped_lines": dict(bad), "models": models,
        "decisions": n, "airborne": share(n_air, n),
        "replan_interval": q(T["interval"]),
        "timing_ms": {k[3:]: q(v) for k, v in T.items() if k.startswith("ms_")},
        "stage2_steps_share": q(T["s2"], (50, 10, 90)),
        "followed_frames": q(T["follow"], (10, 50, 90)),
        "committed_followed_frames": q(committed, (10, 50, 90)) + [len(committed)],
        "triggers": {"decisions": n_trig, "by_kind": dict(triggers.most_common())},
        "switch": {"switched": n_switch, "kept": n_kept, "rate": share(n_switch, n) if n_switch + n_kept else None,
                   "kept_gain": {u: q(g, (50, 90)) + [len(g)] for u, g in kept_gain.items()}},
        "plan_changed": {"all": share(changed, with_prev), "airborne": share(changed_air, with_prev_air),
                         "n": with_prev, "n_airborne": with_prev_air},
        "margin": {"median": q(margins, (50,))[0], "unit": unit,
                   "near_tie": share(sum(m < (NEAR_TIE_FRAMES if unit == "frames" else NEAR_TIE) for m in margins),
                                     len(margins)), "n": len(margins)},
        "success_margin": {"median": q(s_margins, (50,))[0],
                           "near_tie": share(sum(m < NEAR_TIE_FRAMES for m in s_margins), len(s_margins)),
                           "n": len(s_margins)},
        "uncertainty": {"below_0.9": share(sum(u < 0.9 for u in unc), len(unc)),
                        "below_0.5": share(sum(u < 0.5 for u in unc), len(unc))},
        "ends_supported": share(end_sup, n),
        "still": {"share": share(n_still, n), "kept": share(n_still_kept, n), "n": n_still},
        "completed": share(n_completed, n), "landing": share(n_landing, n_landing_known),
        "clipped": {"share": share(n_clip, n_clip_known), "n": n_clip_known},
        "position_error": err,
        "modes": {nm: {"share": share(v["chosen"], n), "chosen": v["chosen"], "runs": len(v["own_ms"]),
                       "own_ms": q([x for x in v["own_ms"] if x is not None]),
                       "total_ms_when_chosen": q([x for x in v["total_ms"] if x is not None])}
                  for nm, v in M.items() if v["chosen"] or v["own_ms"]},
        "levels": per_level}


def pct(v):
    return "-" if v is None else f"{100 * v:.0f}%"


def num(v, spec=".1f"):
    return "-" if v is None else format(v, spec)


def report(r: dict) -> str:
    out = []
    w = out.append
    w(f"run {r['run']}" + (f" (level {r['level']})" if r["level"] else "")
      + f"; model {', '.join(r['models']) if r['models'] else 'not recorded'}"
      + (f"; skipped malformed lines: {r['skipped_lines']}" if r["skipped_lines"] else ""))
    w(f"decisions {r['decisions']}, airborne (start not supported) {pct(r['airborne'])}; replan interval median / 90% "
      f"{num(r['replan_interval'][0], '.0f')} / {num(r['replan_interval'][1], '.0f')} frames")
    w("ms per decision, median / 90%: " + ", ".join(f"{k} {num(v[0], '.0f')} / {num(v[1], '.0f')}"
                                                     for k, v in r["timing_ms"].items()))
    if r.get("modes"):
        w("planning modes (chosen share; ms of its own search when it ran, median / 90%; total ms when chosen): "
          + "; ".join(f"{nm} {pct(v['share'])} ({v['chosen']}), own {num(v['own_ms'][0], '.0f')} / "
                      f"{num(v['own_ms'][1], '.0f')} in {v['runs']} runs, total {num(v['total_ms_when_chosen'][0], '.0f')}"
                      for nm, v in sorted(r["modes"].items())))
    s2 = r["stage2_steps_share"]
    w(f"stage 2 steps rolled / needed: median {num(s2[0], '.2f')}, 10% {num(s2[1], '.2f')}, 90% {num(s2[2], '.2f')}")
    fl = r["followed_frames"]
    w(f"frames each plan was followed: 10% {num(fl[0], '.0f')}, 50% {num(fl[1], '.0f')}, 90% {num(fl[2], '.0f')}")
    cf = r["committed_followed_frames"]
    w(f"frames each committed plan was followed (through the decisions that kept it): 10% {num(cf[0], '.0f')}, "
      f"50% {num(cf[1], '.0f')}, 90% {num(cf[2], '.0f')} (n {cf[3]})")
    tr = r["triggers"]
    w(f"triggers ({tr['decisions']} decisions with them): "
      + (", ".join(f"{k} {v}" for k, v in tr["by_kind"].items()) or "none logged (an older run)"))
    sw = r["switch"]
    kg = "; ".join(f"{u} median {num(g[0], '.2f')} / 90% {num(g[1], '.2f')} (n {g[2]})" for u, g in sw["kept_gain"].items())
    w(f"switch rate (switched / decisions): {pct(sw['rate'])} (switched {sw['switched']}, kept {sw['kept']})"
      + (f"; kept against a best other of {kg}" if kg else ""))
    pc = r["plan_changed"]
    w(f"plan changed (next {CHANGE_WINDOW} actions differ from the previous plan's remaining): all {pct(pc['all'])}, "
      f"airborne {pct(pc['airborne'])} (n {pc['n']} / {pc['n_airborne']})")
    mg = r["margin"]
    fr = mg.get("unit") == "frames"
    w(f"score margin chosen - best rejected (no goal touch, {mg.get('unit', 'V')}): median "
      f"{num(mg['median'], '.1f' if fr else '.2f')}; < {NEAR_TIE_FRAMES if fr else NEAR_TIE} in "
      f"{pct(mg['near_tie'])} (n {mg['n']})")
    sm = r["success_margin"]
    if sm["n"]:
        w(f"goal touches against the best rejected touch: median {num(sm['median'], '.1f')} frames earlier; < "
          f"{NEAR_TIE_FRAMES} frames in {pct(sm['near_tie'])} (n {sm['n']})")
    w(f"uncertainty factor < 0.9: {pct(r['uncertainty']['below_0.9'])}, < 0.5: {pct(r['uncertainty']['below_0.5'])}; "
      f"plans ending supported {pct(r['ends_supported'])}; completed in the air {pct(r.get('completed'))}, landing "
      f"{pct(r.get('landing'))}")
    cl = r.get("clipped") or {}
    w(f"chosen plans the collision guard clipped (the model's path went into solid tiles): {pct(cl.get('share'))}"
      + (f" (n {cl['n']})" if cl.get("n") else " (not logged: an older run)"))
    st = r["still"]
    w(f"still (chosen plan only wait / pad): {pct(st['share'])} of decisions (n {st['n']}), kept from the previous "
      f"plan {pct(st['kept'])}")
    w("position error (units), median / 90% (n): followed | not followed")
    for hz, e in r["position_error"].items():
        a, b = e["followed"], e["not_followed"]
        w(f"  +{hz:>2}: {num(a[0])} / {num(a[1])} ({a[2]})  |  {num(b[0])} / {num(b[1])} ({b[2]})")
    w("per level:")
    for name, L in r["levels"].items():
        fam = ", ".join(f"{f} {pct(s)}" for f, s in L["first_moves"])
        w(f"  {name}: {L['decisions']} decisions, no progress {pct(L['no_progress'])}, dead-end ends "
          f"{pct(L['dead_end'])}, still {pct(L['still'])} (kept {pct(L['still_kept'])}), replayed the best route "
          f"{L['replayed']}; {L['waves']} waves {L['wave_end']}")
        if fam:
            w(f"    first moves: {fam}")
        if L["deaths"]:
            w("    deaths: " + ", ".join(f"{c} {k}" for k, c in L["deaths"].items()))
        if L["finishes"]:
            est = any(f["from"].startswith("estimate") for f in L["finishes"])
            w(f"    finishes {len(L['finishes'])}: " + ", ".join(
                f"w{f['wave']} {num(f['seconds'], '.2f')} s" + ("*" if f["from"].startswith("estimate") else "")
                for f in L["finishes"][:12]) + (" ..." if len(L["finishes"]) > 12 else "")
              + f"; best {num(L['best_s'], '.2f')} s" + (" (* estimate: wave length of a 'quota' wave)" if est else ""))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run folder (runs/<name>, a name under runs/, or 'latest')")
    ap.add_argument("--json", default=None, help="also write the numbers to this JSON file")
    ap.add_argument("--level", default=None, help="only this level (e.g. 1-11)")
    a = ap.parse_args(argv)
    r = analyze(resolve_run(a.run), a.level)
    print(report(r))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1)
    return r


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
