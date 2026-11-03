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
ms per stage (median / 90%); stage-2 steps rolled / needed; frames each plan was followed; the share of
decisions whose next 16 actions differ from the previous plan's remaining ones (all, airborne); the score
margin chosen - best rejected; uncertainty factor shares; plans ending supported; the position error per
horizon for followed and not followed predictions (alive agents only); per level: decisions, no-progress
share (progress <= 0.05), dead-end ends, the most common first-move families, deaths per cause, finishes and
their times (times.jsonl or finish events; for older runs without them, the wave length of a 'quota' wave a
planned agent finished, labelled as an estimate).
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
RUNS = os.path.join(HERE, "runs")
HORIZONS = (1, 4, 8, 16, 24, 32, 48)
CHANGE_WINDOW = 16                 # actions compared for "plan changed"
NO_PROGRESS = 0.05                 # plan.planner.NO_PROGRESS
NEAR_TIE = 0.1
FRAME_MS = 16.667                  # the locked time step (older runs' finishes have no dt)
TOP_FAMILIES = 5


def resolve_run(arg: str, runs_dir: str = RUNS) -> str:
    """'latest' = the newest runs/plan_* folder (by name); else a path as given or a folder under runs/."""
    if arg == "latest":
        cands = sorted(d for d in glob.glob(os.path.join(runs_dir, "plan_*")) if os.path.isdir(d))
        if not cands:
            raise SystemExit(f"no plan_* run folder in {runs_dir}")
        return cands[-1]
    for p in (arg, os.path.join(runs_dir, arg), os.path.join(HERE, arg)):
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


def slim(d: dict) -> dict:
    """The fields of a plan record this analysis uses."""
    sc = d.get("score") or {}
    alts = [a.get("total") for a in d.get("alternatives") or []]
    alts = [a for a in alts if a is not None and math.isfinite(a)]
    path = d.get("path") or []
    c2 = (d.get("counts") or {}).get("stage2") or {}
    chosen = d.get("chosen") or []
    return {"frame": int(d["frame"]), "actions": [int(a) for a in d.get("actions") or []],
            "path": {h: path[h] for h in HORIZONS if h < len(path) and path[h] is not None},
            "air": not bool((d.get("state") or {}).get("supported")),
            "end_supported": bool(d.get("end_supported")), "unc": sc.get("uncertainty"),
            "progress": sc.get("progress"), "total": sc.get("total"), "best_alt": max(alts) if alts else None,
            "dead_end": d.get("v_end") is None, "family": family(chosen[0]) if chosen else "(none)",
            "timing": d.get("timing") or {},
            "s2": c2["steps"] / c2["of_steps"] if c2.get("of_steps") else None}


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
    err_f, err_u = defaultdict(list), defaultdict(list)
    n = n_air = 0
    changed = changed_air = with_prev = with_prev_air = 0
    margins, unc = [], []
    end_sup = 0
    lv = defaultdict(lambda: {"decisions": 0, "no_progress": 0, "dead_end": 0, "families": Counter()})
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
            if p["s2"] is not None:
                T["s2"].append(p["s2"])
            h = 0
            while h < len(acts) and played.get(f + h) == acts[h]:
                h += 1
            T["follow"].append(h)
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
                margins.append(p["total"] - p["best_alt"])
            if p["unc"] is not None:
                unc.append(p["unc"])
            end_sup += p["end_supported"]
            L = lv[k[0]]
            L["decisions"] += 1
            L["no_progress"] += int(p["progress"] is not None and p["progress"] <= NO_PROGRESS)
            L["dead_end"] += int(p["dead_end"])
            L["families"][p["family"]] += 1
            prev = p
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
    for e in stream(os.path.join(run, "events.jsonl"), bad):
        kd = e.get("kind")
        if kd == "lock":
            lock = bool(e.get("ok"))
        if level and e.get("level") != level:
            continue
        L = e.get("level")
        if kd == "death":
            deaths[L][e.get("cause") or "?"] += 1
        elif kd == "finish":
            fin_in_wave += 1
            if not have_times:
                frames = e.get("frames", e.get("frame"))     # older events: the message frame = updates played
                secs = e.get("seconds")
                if secs is None and frames is not None:
                    secs = round(frames * FRAME_MS / 1000.0, 3)
                finishes[L].append({"wave": e.get("wave"), "agent": e.get("agent"), "frames": frames,
                                    "seconds": secs, "lock": e.get("lock", lock), "from": "finish event"})
        elif kd == "wave":
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

    err = {}
    for hz in HORIZONS:
        err[str(hz)] = {"followed": q(err_f[hz]) + [len(err_f[hz])], "not_followed": q(err_u[hz]) + [len(err_u[hz])]}
    per_level = {}
    for name in sorted(set(lv) | set(deaths) | set(finishes) | set(waves), key=lambda s: str(s)):
        L = lv.get(name) or {"decisions": 0, "no_progress": 0, "dead_end": 0, "families": Counter()}
        fs = finishes.get(name, [])
        secs = [x["seconds"] for x in fs if x["seconds"] is not None]
        per_level[str(name)] = {
            "decisions": L["decisions"], "no_progress": share(L["no_progress"], L["decisions"]),
            "dead_end": share(L["dead_end"], L["decisions"]),
            "first_moves": [[f, share(c, L["decisions"])] for f, c in L["families"].most_common(TOP_FAMILIES)],
            "deaths": dict(deaths.get(name, Counter()).most_common()),
            "waves": waves[name]["waves"] if name in waves else 0,
            "wave_end": dict(waves[name]["why"]) if name in waves else {},
            "finishes": fs, "best_s": min(secs) if secs else None}
    return {
        "run": run, "level": level, "skipped_lines": dict(bad),
        "decisions": n, "airborne": share(n_air, n),
        "replan_interval": q(T["interval"]),
        "timing_ms": {k[3:]: q(v) for k, v in T.items() if k.startswith("ms_")},
        "stage2_steps_share": q(T["s2"], (50, 10, 90)),
        "followed_frames": q(T["follow"], (10, 50, 90)),
        "plan_changed": {"all": share(changed, with_prev), "airborne": share(changed_air, with_prev_air),
                         "n": with_prev, "n_airborne": with_prev_air},
        "margin": {"median": q(margins, (50,))[0], "near_tie": share(sum(m < NEAR_TIE for m in margins), len(margins)),
                   "n": len(margins)},
        "uncertainty": {"below_0.9": share(sum(u < 0.9 for u in unc), len(unc)),
                        "below_0.5": share(sum(u < 0.5 for u in unc), len(unc))},
        "ends_supported": share(end_sup, n),
        "position_error": err,
        "levels": per_level}


def pct(v):
    return "-" if v is None else f"{100 * v:.0f}%"


def num(v, spec=".1f"):
    return "-" if v is None else format(v, spec)


def report(r: dict) -> str:
    out = []
    w = out.append
    w(f"run {r['run']}" + (f" (level {r['level']})" if r["level"] else "")
      + (f"; skipped malformed lines: {r['skipped_lines']}" if r["skipped_lines"] else ""))
    w(f"decisions {r['decisions']}, airborne (start not supported) {pct(r['airborne'])}; replan interval median / 90% "
      f"{num(r['replan_interval'][0], '.0f')} / {num(r['replan_interval'][1], '.0f')} frames")
    w("ms per decision, median / 90%: " + ", ".join(f"{k} {num(v[0], '.0f')} / {num(v[1], '.0f')}"
                                                     for k, v in r["timing_ms"].items()))
    s2 = r["stage2_steps_share"]
    w(f"stage 2 steps rolled / needed: median {num(s2[0], '.2f')}, 10% {num(s2[1], '.2f')}, 90% {num(s2[2], '.2f')}")
    fl = r["followed_frames"]
    w(f"frames each plan was followed: 10% {num(fl[0], '.0f')}, 50% {num(fl[1], '.0f')}, 90% {num(fl[2], '.0f')}")
    pc = r["plan_changed"]
    w(f"plan changed (next {CHANGE_WINDOW} actions differ from the previous plan's remaining): all {pct(pc['all'])}, "
      f"airborne {pct(pc['airborne'])} (n {pc['n']} / {pc['n_airborne']})")
    mg = r["margin"]
    w(f"score margin chosen - best rejected: median {num(mg['median'], '.2f')}; < {NEAR_TIE} in {pct(mg['near_tie'])} "
      f"(n {mg['n']})")
    w(f"uncertainty factor < 0.9: {pct(r['uncertainty']['below_0.9'])}, < 0.5: {pct(r['uncertainty']['below_0.5'])}; "
      f"plans ending supported {pct(r['ends_supported'])}")
    w("position error (units), median / 90% (n): followed | not followed")
    for hz, e in r["position_error"].items():
        a, b = e["followed"], e["not_followed"]
        w(f"  +{hz:>2}: {num(a[0])} / {num(a[1])} ({a[2]})  |  {num(b[0])} / {num(b[1])} ({b[2]})")
    w("per level:")
    for name, L in r["levels"].items():
        fam = ", ".join(f"{f} {pct(s)}" for f, s in L["first_moves"])
        w(f"  {name}: {L['decisions']} decisions, no progress {pct(L['no_progress'])}, dead-end ends "
          f"{pct(L['dead_end'])}; {L['waves']} waves {L['wave_end']}")
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
