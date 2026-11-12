"""Fixed benchmark: the beam planner (agent.py) with a frozen config over a fixed World-1 level set, turned into
one comparable record per run (or set of runs), with before/after deltas against a saved baseline.

    python smbrl/plan/benchmark.py cmd                 # from the folder above smbrl/: the agent.py command to run
    python smbrl/plan/benchmark.py report bench_20261108-120000 [more folders] [--k 20] [--label NAME] [--set-baseline]
    python -m plan.benchmark cmd | report ...          # the same from smbrl/

Run folder names resolve under smbrl/runs from either folder (then as paths). The printed commands are for the
folder above smbrl/ (`python smbrl/agent.py ...`, `python smbrl/plan/benchmark.py report ...`). The agent.py
command draws the overlay (the plans are visible) and passes --wm-collect (runs/<name>/wm_data, for offline
replays: plan/offline.py); --learn must not be added (it changes the model between waves); changing the level
with S is fine. It also gives the run its own death and point-of-no-return memories in the run folder
(--death-memory / --move-memory runs/<name>/...json, created empty), so every benchmark run starts with empty
memories (the failure memory and the best route are per run already) and runs stay comparable.

The spec is smbrl/benchmark.json: levels (disjoint from eval_levels.json's held-out levels, checked at load),
held_out_levels (held-out levels played in the same run for generalization: a subset of eval_levels.json's
held_out, checked at load; agent.py collects no world-model data, records no self-demo and charges no death or
point-of-no-return memory there, so the same command plays them; the report shows them in their own "held-out"
block and overall line, never in the benchmark's overall), waves per level, the frozen agent.py config and
"switches" (on/off agent.py flags the benchmark always passes, not agent.py defaults: --wm-collect).
Held-out levels chosen (2026-11-10): 1-18 and 1-20, World 1 like the trained levels (the world model knows only
World 1) and not much harder than the set: 1-18 five spinners (1-13 has one), 54 x 50 tiles, 696 units spawn to
goal; 1-20 twenty saws (1-8 has 17) and 21 crumble blocks (1-11 has 31), 541 units. Not 1-19 (160 x 30 tiles,
2,259 units spawn to goal, about four times 1-8's) nor the World 2 levels 2-5 / 2-6 / 2-7 (hospital tiles, needles
and wind the world model never saw). The config there is resolved from agent.py (DEFAULT_MODEL,
REPLAN_DEFAULT, TUBE_DEFAULT, HYSTERESIS_FRAMES and build_parser()'s defaults) by freeze(), so a later default
change shows as a diff of benchmark.json when it is frozen again; `cmd` notes any drift.

Run folders are read only through plan/analyze_run.py: analyze(folder, level) per benchmark level.

The first K waves of each level (PLAN_GAME_SEARCH.md section 4; K = 20, --k): a level's completed waves in log
order, the run folders in the order given; waves beyond K are ignored, fewer than K are shown as n/K. Per level:
- first-try: did the level's first wave finish;
- waves to first finish (none within K: "fail") and the game frames to it: every agent's alive frames (the wave
  event's all_agent_frames; older logs: agent_frames, the planned agents' only) summed over the waves up to and
  including the first finishing one;
- fresh and replay completion apart: finished waves / waves over each kind. A wave is a REPLAY wave when any of
  its decisions replayed the recorded best route (plans.jsonl best_route.used), it replayed a recorded demo (a
  "demo check" event) or, in a search run (agent.py --search), it started with a trusted finish recorded (its
  verifier and copies replay it: the wave event's replay_route); a fresh wave used no recorded route at all.
  A search run's finishes are its search agents' (times.jsonl source "search"); Any, not a majority: the logs have waves that
  replay the route from frame 0 for hundreds of frames, then plan the rest (bench_20261110-003407, 1-13 and 1-14:
  26-48% of the decisions), and such a finish is not a fresh one;
- best and median finish seconds over the K waves;
- distinct outcomes, the real sample size next to the wave count: distinct finish frames, distinct death places
  (position rounded to DEATH_CELL units and the frame to DEATH_FRAMES frames) and distinct places where a wave
  ended without a finish or a death (time-outs; the planned agent's last logged position, rounded the same).
  With the time-step lock a repeated wave is the same outcome again;
- the decisions that replayed the best route, over every logged decision of the level (the interrupted last
  wave included, as the decisions column) and over the K waves.
Overall per block: first-try and solved-within-K counts of levels, fresh / replay completion pooled, distinct
outcomes summed. The older columns (completion over every logged wave, deaths per wave, decisions, ms per
decision, replan interval, the still share: decisions whose chosen plan has no moves other than wait / pad,
plan.analyze_run.is_still) stay in a second table over every logged wave, then deaths by cause and the position
error per horizon (followed / not followed). Deltas against the baseline use the K metrics (and ms per
decision, still). Records go to runs/benchmarks/<label>.json; --set-baseline also writes
runs/benchmarks/baseline.json.

The model: the one the run's own logs name (analyze_run "models": agent.py's level or start event "model",
--learn's "learn model" path), else "not recorded"; the spec's model is shown apart as the one the benchmark asked
for (its mtime and size read when the report is made), never as the one used.

Comparability: the K metrics are over the first K waves; the second table is over every logged wave. A level
played in several run folders sums its counts exactly; its medians (ms, replan interval, position error) are then
weighted means of the per-run medians (marked approx). The config recorded is the frozen spec's: run folders do
not log agent.py's arguments, so it is what the benchmark asked for, not verified from the run. The git commit is
read when the report is made.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from plan.analyze_run import HORIZONS, RUNS, analyze, resolve_run      # noqa: E402
from levels_smb import a_plus, a_plus_time                          # noqa: E402

SPEC = os.path.join(HERE, "benchmark.json")
EVAL = os.path.join(HERE, "eval_levels.json")
OUT_DIR = os.path.join(RUNS, "benchmarks")
BASELINE = "baseline"
# config key -> agent.py flag, in the order emitted
FLAGS = (("model", "--model"), ("budget_ms", "--budget-ms"), ("replan", "--replan"), ("tube", "--tube"),
         ("hysteresis", "--hysteresis"), ("modes", "--modes"), ("triggers", "--triggers"), ("agents", "--agents"),
         ("end_after", "--end-after"), ("wave_seconds", "--wave-seconds"), ("seed", "--seed"),
         ("threads", "--threads"))
K_DEFAULT = 20                     # waves per level that count (PLAN_GAME_SEARCH.md section 4)
DEATH_CELL = 20.0                  # units: death / time-out places rounded to a tile
DEATH_FRAMES = 30                  # frames: death times bucketed by half a second
COMPARABILITY = ("first K waves per level (K shown); a replay wave replayed a recorded route (any decision with "
                 "best_route.used, or a demo); 'all logged waves' columns are over every wave played")


def resolved_config() -> dict:
    """agent.py's current defaults for the benchmark's flags (importing agent.py starts nothing)."""
    import agent
    ap = agent.build_parser()
    cfg = {"model": agent.DEFAULT_MODEL.replace(os.sep, "/"), "replan": agent.REPLAN_DEFAULT,
           "tube": agent.TUBE_DEFAULT, "hysteresis": agent.HYSTERESIS_FRAMES}
    for k, _ in FLAGS:
        if k not in cfg:
            cfg[k] = ap.get_default(k)
    return {k: cfg[k] for k, _ in FLAGS}


def freeze(path: str = SPEC) -> dict:
    """(Re)write the spec's config from agent.py's current defaults, keeping levels and waves_per_level."""
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    spec["config"] = resolved_config()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")
    return spec


def load_spec(path: str = SPEC, eval_path: str = EVAL) -> dict:
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    missing = [k for k, _ in FLAGS if k not in (spec.get("config") or {})]
    if not spec.get("levels") or not spec.get("waves_per_level") or missing:
        raise ValueError(f"{path}: needs levels, waves_per_level and config {[k for k, _ in FLAGS]}"
                         + (f" (config missing {missing})" if missing else ""))
    with open(eval_path, encoding="utf-8") as f:
        held = set(json.load(f).get("held_out") or [])
    overlap = sorted(set(spec["levels"]) & held)
    if overlap:
        raise ValueError(f"{path}: benchmark levels {overlap} are held out in {eval_path} (generalization only, "
                         "never benchmarked)")
    spec.setdefault("held_out_levels", [])
    not_held = sorted(set(spec["held_out_levels"]) - held)
    if not_held:
        raise ValueError(f"{path}: held_out_levels {not_held} are not held out in {eval_path} (the agent would "
                         "collect data and charge memories there)")
    return spec


def _arg(v) -> str:
    s = str(v)
    return f'"{s}"' if " " in s else s


def command(spec: dict, now: float | None = None) -> str:
    name = time.strftime("bench_%Y%m%d-%H%M%S", time.localtime(now))
    cfg = spec["config"]
    cmd = "python smbrl/agent.py " + " ".join(f"{fl} {_arg(cfg[k])}" for k, fl in FLAGS)
    cmd += "".join(f" {s}" for s in spec.get("switches") or []) + f" --name {name}"
    cmd += (f" --death-memory smbrl/runs/{name}/death_memory.json"       # the run's own memories, empty at start
            f" --move-memory smbrl/runs/{name}/move_memory.json")
    out = ["run from the folder above smbrl/ (the overlay shows the plans; --wm-collect keeps the world-model data "
           "for offline replays):", "  " + cmd, "",
           f"in that one run, play each level for {spec['waves_per_level']} waves:"]
    out += [f"  {lv}: {spec['waves_per_level']} waves" for lv in spec["levels"]]
    if spec.get("held_out_levels"):
        out += ["then the held-out levels (generalization; agent.py collects no world-model data and charges no "
                "memory there, its level line says so):"]
        out += [f"  {lv}: {spec['waves_per_level']} waves (held out)" for lv in spec["held_out_levels"]]
    out += ["", "do not add --learn (it trains and swaps the model between waves: not the frozen config); changing "
            "the level with S is fine.", "", "then, from the same folder:",
            f"  python smbrl/plan/benchmark.py report {name}"]
    return "\n".join(out)


def config_drift(spec: dict) -> dict:
    now = resolved_config()
    return {k: [spec["config"].get(k), now[k]] for k, _ in FLAGS if spec["config"].get(k) != now[k]}


# ---------------------------------------------------------------------------------------------- the record

def model_info(model: str) -> dict:
    """The model file as agent.py resolves it (as given if it exists, else under smbrl/)."""
    path = model if os.path.exists(model) else os.path.join(HERE, model)
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return {"path": path, "exists": False, "mtime": None, "size": None}
    st = os.stat(path)
    return {"path": path, "exists": True, "mtime": round(st.st_mtime, 3), "size": st.st_size}


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _wmean(pairs):
    """Weighted mean of (value, weight) pairs, None values skipped."""
    pairs = [(v, w) for v, w in pairs if v is not None and w]
    tot = sum(w for _, w in pairs)
    return round(sum(v * w for v, w in pairs) / tot, 3) if tot else None


def _merge_err(errs: list) -> dict:
    """Position error per horizon from per-run [p50, p90, n]: exact for one run, else n-weighted medians."""
    out = {}
    for hz in map(str, HORIZONS):
        out[hz] = {}
        for side in ("followed", "not_followed"):
            xs = [e[hz][side] for e in errs if hz in e]
            if len(xs) == 1:
                out[hz][side] = xs[0]
            else:
                out[hz][side] = [_wmean((x[0], x[2]) for x in xs), _wmean((x[1], x[2]) for x in xs),
                                 sum(x[2] for x in xs)]
    return out


def _ratio(n, d):
    return round(n / d, 4) if d else None


def is_replay(w: dict) -> bool:
    """A wave that replayed a recorded route: any decision with best_route.used, a recorded demo, or a search wave
    that replayed a recorded finish (agent.py --search: the wave event's replay_route)."""
    return bool(w.get("demo")) or bool(w.get("route_replay")) or (w.get("replayed") or 0) > 0


def _place(x, y, f=None):
    p = (round(x / DEATH_CELL), round(y / DEATH_CELL))
    return p if f is None else p + (int(f) // DEATH_FRAMES,)


def k_metrics(waves: list, k: int = K_DEFAULT, level: str | None = None) -> dict:
    """A level's completed waves in log order (analyze_run wave_list, run folders concatenated) -> the section 4
    metrics over the first k; with `level`, the A+ target too (levels_smb.a_plus_time: a_plus_k, the best within
    k at most the A+ time; waves_to_a_plus, the first wave with such a finish)."""
    ws = waves[:k]
    first_a = next((i for i, w in enumerate(ws) if any(a_plus(level, sc) for _, sc in w["finishes"])), None)
    first = next((i for i, w in enumerate(ws) if w["finished"]), None)
    frames = None
    if first is not None:
        af = [w.get("agent_frames") for w in ws[:first + 1]]
        frames = None if any(a is None for a in af) else int(sum(af))
    rep = [w for w in ws if is_replay(w)]
    fresh = [w for w in ws if not is_replay(w)]
    secs = [sc for w in ws for _, sc in w["finishes"] if sc is not None]
    fin_frames = {f for w in ws for f, _ in w["finishes"] if f is not None}
    deaths = {_place(x, y, f) for w in ws for x, y, f in w["deaths"] if None not in (x, y, f)}
    stuck = {_place(*w["end"]) for w in ws if w.get("end") and not w["finished"] and not w["deaths"]}
    fr_fin, rp_fin = sum(w["finished"] > 0 for w in fresh), sum(w["finished"] > 0 for w in rep)
    return {"k": k, "waves_k": len(ws), "waves_logged": len(waves),
            "first_try": None if not ws else bool(ws[0]["finished"]),
            "solved": None if not ws else first is not None,
            "waves_to_first": None if first is None else first + 1, "frames_to_first": frames,
            "finished_waves_k": fr_fin + rp_fin,
            "fresh_waves": len(fresh), "fresh_finishes": fr_fin, "fresh_completion": _ratio(fr_fin, len(fresh)),
            "replay_waves": len(rep), "replay_finishes": rp_fin, "replay_completion": _ratio(rp_fin, len(rep)),
            "best_s_k": min(secs) if secs else None,
            "median_s_k": round(float(statistics.median(secs)), 3) if secs else None,
            "a_plus_k": a_plus(level, min(secs) if secs else None) if secs else (False if ws and a_plus_time(level) else None),
            "waves_to_a_plus": None if first_a is None else first_a + 1,
            "distinct_finish_frames": len(fin_frames), "distinct_death_places": len(deaths),
            "distinct_stuck_places": len(stuck), "distinct_outcomes": len(fin_frames) + len(deaths) + len(stuck),
            "decisions_k": sum(w["decisions"] for w in ws), "replayed_decisions_k": sum(w["replayed"] for w in ws)}


def k_overall(levels: dict, k: int = K_DEFAULT) -> dict:
    """A block's overall K line: level counts, fresh / replay completion pooled, distinct outcomes summed."""
    vs = [v for v in levels.values() if v["waves_k"]]
    fw, ff = sum(v["fresh_waves"] for v in vs), sum(v["fresh_finishes"] for v in vs)
    rw, rf = sum(v["replay_waves"] for v in vs), sum(v["replay_finishes"] for v in vs)
    return {"k": k, "levels_played": len(vs),
            "waves_k": sum(v["waves_k"] for v in vs), "waves_logged": sum(v["waves_logged"] for v in vs),
            "first_try": sum(bool(v["first_try"]) for v in vs), "solved": sum(bool(v["solved"]) for v in vs),
            "waves_to_first": None, "frames_to_first": None,
            "finished_waves_k": ff + rf, "fresh_waves": fw, "fresh_finishes": ff, "fresh_completion": _ratio(ff, fw),
            "replay_waves": rw, "replay_finishes": rf, "replay_completion": _ratio(rf, rw),
            "best_s_k": None, "median_s_k": None,
            "a_plus_k": sum(bool(v.get("a_plus_k")) for v in vs), "waves_to_a_plus": None,
            **{key: sum(v[key] for v in vs) for key in ("distinct_finish_frames", "distinct_death_places",
                                                         "distinct_stuck_places", "distinct_outcomes",
                                                         "decisions_k", "replayed_decisions_k")}}


def aggregate(results: list, spec: dict, which: str = "levels", k: int = K_DEFAULT) -> tuple[dict, dict]:
    """results: [(run folder, {level: analyze(folder, level)})] -> per level and overall, over spec[which]
    ("levels": the benchmark's, "held_out_levels": the held-out block). Each level's "k" holds k_metrics over its
    first k waves (the run folders in the order given), the overall's "k" k_overall."""
    levels = {}
    for lv in spec.get(which) or []:
        parts = []                                  # (run, analyze result, its level entry) that played lv
        for run, per in results:
            r = per[lv]
            L = r["levels"].get(lv)
            if L and (L["waves"] or L["decisions"] or L["finishes"]):
                parts.append((run, r, L))
        waves = sum(L["waves"] for _, _, L in parts)
        fins = sum(len(L["finishes"]) for _, _, L in parts)
        deaths = Counter()
        for _, _, L in parts:
            deaths.update(L["deaths"])
        bests = [L["best_s"] for _, _, L in parts if L["best_s"] is not None]
        secs = [x["seconds"] for _, _, L in parts for x in L["finishes"] if x.get("seconds") is not None]
        dec = sum(r["decisions"] for _, r, _ in parts)
        still = sum(L.get("still_n") or 0 for _, _, L in parts)    # analyze_run.is_still decisions
        replayed = sum(L.get("replayed") or 0 for _, _, L in parts)
        ms = [((r["timing_ms"].get("total") or [None])[0], r["decisions"]) for _, r, _ in parts]
        rp = [(r["replan_interval"][0], r["decisions"]) for _, r, _ in parts]
        levels[lv] = {
            "waves": waves, "waves_planned": spec["waves_per_level"], "finishes": fins,
            "completion": _ratio(fins, waves), "best_s": min(bests) if bests else None,
            "a_plus_s": a_plus_time(lv), "a_plus": a_plus(lv, min(bests) if bests else None),
            "a_plus_finishes": sum(bool(a_plus(lv, x)) for x in secs),
            "median_s": round(float(statistics.median(secs)), 3) if secs else None,
            "deaths": dict(deaths.most_common()), "deaths_total": sum(deaths.values()),
            "deaths_per_wave": _ratio(sum(deaths.values()), waves), "decisions": dec,
            "still_decisions": still, "still": _ratio(still, dec),
            "replayed_decisions": replayed, "replayed": _ratio(replayed, dec),
            "k": k_metrics([w for _, _, L in parts for w in L.get("wave_list") or []], k, lv),
            "ms_per_decision": ms[0][0] if len(parts) == 1 else _wmean(ms),
            "replan_interval": rp[0][0] if len(parts) == 1 else _wmean(rp),
            "position_error": _merge_err([r["position_error"] for _, r, _ in parts]) if parts else None,
            "approx_medians": len(parts) > 1,
            "runs": [{"run": os.path.basename(run), "waves": L["waves"]} for run, _, L in parts]}
    vs = levels.values()
    waves = sum(v["waves"] for v in vs)
    fins = sum(v["finishes"] for v in vs)
    dt = sum(v["deaths_total"] for v in vs)
    overall = {"waves": waves, "waves_planned": spec["waves_per_level"] * len(levels), "finishes": fins,
               "completion": _ratio(fins, waves), "deaths_per_wave": _ratio(dt, waves),
               "decisions": sum(v["decisions"] for v in vs),
               "still_decisions": sum(v["still_decisions"] for v in vs),
               "still": _ratio(sum(v["still_decisions"] for v in vs), sum(v["decisions"] for v in vs)),
               "ms_per_decision": _wmean((v["ms_per_decision"], v["decisions"]) for v in vs),
               "replayed_decisions": sum(v["replayed_decisions"] for v in vs),
               "replayed": _ratio(sum(v["replayed_decisions"] for v in vs), sum(v["decisions"] for v in vs)),
               "levels_finished": sum(v["finishes"] > 0 for v in vs), "levels": len(levels),
               "levels_a_plus": sum(bool(v["a_plus"]) for v in vs),
               "k": k_overall({lv: v["k"] for lv, v in levels.items()}, k)}
    return levels, overall


def build_record(folders: list, spec: dict, label: str | None = None, runs_dir: str = RUNS,
                 k: int = K_DEFAULT) -> dict:
    runs = [resolve_run(f, runs_dir) for f in folders]
    every = list(spec["levels"]) + list(spec.get("held_out_levels") or [])
    results = [(run, {lv: analyze(run, lv) for lv in every}) for run in runs]
    levels, overall = aggregate(results, spec, k=k)
    h_levels, h_overall = aggregate(results, spec, "held_out_levels", k=k)
    return {"label": label or "+".join(os.path.basename(r) for r in runs),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "comparability": COMPARABILITY, "k": k,
            "levels_spec": spec["levels"], "waves_per_level": spec["waves_per_level"],
            "config": spec["config"], "switches": spec.get("switches") or [],
            "model_spec": model_info(spec["config"]["model"]),     # what the benchmark asked for, not what ran
            "model_used": [{"run": os.path.basename(r), "models": next(iter(res.values()))["models"] if res else []}
                           for r, res in results],                 # from the run's own logs ([]: not recorded)
            "git_commit": git_commit(),
            "runs": [{"run": r, "skipped_lines": {k: v for per in res.values() for k, v in per["skipped_lines"].items()}}
                     for r, res in results],
            "wave_counts": {lv: v["waves"] for lv, v in levels.items()},
            "levels": levels, "overall": overall,
            "held_out": {"levels_spec": list(spec.get("held_out_levels") or []), "levels": h_levels,
                         "overall": h_overall}}


K_DELTA_KEYS = ("first_try", "solved", "waves_to_first", "frames_to_first", "fresh_completion", "replay_completion",
                "best_s_k", "median_s_k", "distinct_outcomes")
DELTA_KEYS = K_DELTA_KEYS + ("ms_per_decision", "still")


def deltas(rec: dict, base: dict) -> dict:
    """Per benchmark level and overall: the K metrics (from each record's "k"; a baseline without them: None),
    ms per decision and the still share; the held-out block's under "held_out" (a baseline without one: None)."""
    def d(a, b, key):
        a, b = a or {}, b or {}
        if key in K_DELTA_KEYS:
            a, b = a.get("k") or {}, b.get("k") or {}
        x, y = a.get(key), b.get(key)
        return None if x is None or y is None else round(float(x) - float(y), 4)

    def block(r, b):
        out = {lv: {k: d(r["levels"][lv], (b or {}).get("levels", {}).get(lv), k) for k in DELTA_KEYS}
               for lv in r["levels"]}
        out["overall"] = {k: d(r["overall"], (b or {}).get("overall"), k) for k in DELTA_KEYS}
        return out
    out = block(rec, base)
    if rec.get("held_out"):
        out["held_out"] = block(rec["held_out"], base.get("held_out"))
    return out


# ---------------------------------------------------------------------------------------------- the table

def _f(v, spec=".2f"):
    return "-" if v is None else format(v, spec)


def _p(v):
    return "-" if v is None else f"{100 * v:.0f}%"


def _d(v, spec, pct=False):
    if v is None:
        return "-"
    return f"{100 * v:+.0f}pp" if pct else format(v, "+" + spec)


def _models_used(rec: dict) -> str:
    return "; ".join(f"{u['run']}: {', '.join(u['models']) if u['models'] else 'not recorded'}"
                     for u in rec.get("model_used") or [])


def _yn(v):
    return "-" if v is None else ("yes" if v else "no")


def _frac(n, d, r):
    return "-" if not d else f"{n}/{d} {_p(r)}"


K_HELP = ("1st: wave 1 finished; to 1st: waves to the first finish (fail: none within K) and the game frames to it "
          "(all agents); fresh / replay: finished waves / waves of each kind (replay: any decision replayed the "
          "recorded best route, or a demo); outcomes: distinct finish frames / death places / time-out places; "
          "replayed: decisions that replayed the best route over every logged decision; A+: the game's par time "
          "(levels_smb.a_plus_time), A+ ok: w<N> = the first wave with a finish within it (no: none within K), "
          "A+ fin: logged finishes within it")


def _k_rows(w, levels: dict, overall: dict, k: int):
    w(f"{'level':<8}{'waves':>7}{'logged':>7}{'1st':>5}{'to 1st':>8}{'frames':>8}{'fresh':>12}{'replay':>12}"
      f"{'best_s':>8}{'med_s':>8}{'A+':>6}{'A+ ok':>7}{'outcomes':>9} {'(f/d/t)':<10}{'replayed':>18}")
    for lv, v in list(levels.items()) + [("overall", overall)]:
        x = v["k"]
        if lv == "overall":
            aok, apar = f"{x['a_plus_k']}/{x['levels_played']}", ""
        else:
            aok = "-" if x["a_plus_k"] is None else (f"w{x['waves_to_a_plus']}" if x["a_plus_k"] else "no")
            apar = _f(v.get("a_plus_s"), ".1f")
        if lv == "overall":
            first = f"{x['first_try']}/{x['levels_played']}"
            to1 = f"{x['solved']}/{x['levels_played']}"
        else:
            first = _yn(x["first_try"])
            to1 = "-" if not x["waves_k"] else (str(x["waves_to_first"]) if x["solved"] else "fail")
        of = k * (len(levels) if lv == "overall" else 1)
        w(f"{lv:<8}{str(x['waves_k']) + '/' + str(of):>7}{x['waves_logged']:>7}{first:>5}{to1:>8}"
          f"{_f(x['frames_to_first'], 'd'):>8}"
          f"{_frac(x['fresh_finishes'], x['fresh_waves'], x['fresh_completion']):>12}"
          f"{_frac(x['replay_finishes'], x['replay_waves'], x['replay_completion']):>12}"
          f"{_f(x['best_s_k']):>8}{_f(x['median_s_k']):>8}{apar:>6}{aok:>7}{x['distinct_outcomes']:>9} "
          f"{'(' + str(x['distinct_finish_frames']) + '/' + str(x['distinct_death_places']) + '/' + str(x['distinct_stuck_places']) + ')':<10}"
          f"{_frac(v['replayed_decisions'], v['decisions'], v['replayed']):>18}")


def _delta_rows(w, levels: dict, dlb: dict):
    w(f"{'level':<8}{'d 1st':>7}{'d solved':>9}{'d to1st':>8}{'d frames':>9}{'d fresh':>8}{'d replay':>9}"
      f"{'d best':>8}{'d med':>8}{'d outc':>7}{'d ms':>7}{'d still':>8}")
    for lv in list(levels) + ["overall"]:
        x = dlb[lv]
        w(f"{lv:<8}{_d(x['first_try'], '.0f'):>7}{_d(x['solved'], '.0f'):>9}{_d(x['waves_to_first'], '.0f'):>8}"
          f"{_d(x['frames_to_first'], '.0f'):>9}{_d(x['fresh_completion'], '', True):>8}"
          f"{_d(x['replay_completion'], '', True):>9}{_d(x['best_s_k'], '.2f'):>8}{_d(x['median_s_k'], '.2f'):>8}"
          f"{_d(x['distinct_outcomes'], '.0f'):>7}{_d(x['ms_per_decision'], '.0f'):>7}{_d(x['still'], '', True):>8}")


def _logged_rows(w, levels: dict, overall: dict):
    w(f"{'level':<8}{'waves':>6}{'fin':>5}{'compl':>7}{'best_s':>8}{'med_s':>8}{'A+':>6}{'A+ fin':>7}"
      f"{'d/wave':>8}{'dec':>7}{'ms/dec':>8}{'replan':>7}{'still':>7}")
    for lv, v in list(levels.items()) + [("overall", overall)]:
        if lv == "overall":
            apar, afin = "", f"{v.get('levels_a_plus', 0)}/{v['levels']}"
        else:
            apar = _f(v.get("a_plus_s"), ".1f")
            afin = "-" if v.get("a_plus_s") is None else str(v.get("a_plus_finishes", 0))
        w(f"{lv:<8}{v['waves']:>6}{v['finishes']:>5}{_p(v['completion']):>7}"
          f"{_f(v.get('best_s')):>8}{_f(v.get('median_s')):>8}{apar:>6}{afin:>7}{_f(v['deaths_per_wave']):>8}{v['decisions']:>7}"
          f"{_f(v['ms_per_decision'], '.0f'):>8}{_f(v.get('replan_interval'), '.0f'):>7}{_p(v.get('still')):>7}")


def table(rec: dict, base: dict | None = None) -> str:
    out = []
    w = out.append
    m = rec["model_spec"]
    k = rec["k"]
    w(f"benchmark {rec['label']} ({rec['created']}); git {rec['git_commit'] or 'unknown'}")
    w("runs: " + ", ".join(os.path.basename(r["run"]) + (f" (skipped lines {r['skipped_lines']})"
                                                        if r["skipped_lines"] else "") for r in rec["runs"]))
    w("model used (from the run's logs): " + _models_used(rec))
    w(f"model the spec asks for (not verified as the one used): {m['path']}"
      + (f" (mtime {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(m['mtime']))}, {m['size']} bytes)"
         if m["exists"] else " (missing)"))
    w("config (frozen, not verified from the run): " + " ".join(f"{a}={b}" for a, b in rec["config"].items()))
    w("note: " + rec["comparability"])
    dl = None
    if base is not None:
        dl = deltas(rec, base)
        used = sorted({x for u in rec.get("model_used") or [] for x in u["models"]})
        bused = sorted({x for u in base.get("model_used") or [] for x in u["models"]})
        w(f"baseline {base.get('label')} (git {base.get('git_commit') or 'unknown'}, K {base.get('k', 'none')})"
          + ("; MODEL USED DIFFERS from the baseline's" if used and bused and used != bused else "")
          + ("; models used not recorded in both, not compared" if not (used and bused) else "")
          + ("; K DIFFERS from the baseline" if base.get("k") not in (None, k) else "")
          + ("; CONFIG DIFFERS from the baseline" if base.get("config") != rec["config"] else ""))
    ho = rec.get("held_out") or {}
    blocks = [("trained levels", rec["levels"], rec["overall"], (dl or {}))]
    if ho.get("levels"):
        blocks.append(("held-out levels (generalization: never trained on, no world-model data, no memory charged; "
                       "not in the overall above)", ho["levels"], ho["overall"], (dl or {}).get("held_out")))
    w(K_HELP)
    for title, levels, overall, dlb in blocks:
        w("")
        w(f"{title}, first {k} waves:")
        _k_rows(w, levels, overall, k)
        if dlb:
            w(f"vs baseline {base.get('label')}:")
            _delta_rows(w, levels, dlb)
        w("all logged waves:")
        _logged_rows(w, levels, overall)
    w("")
    every = list(rec["levels"].items()) + list((ho.get("levels") or {}).items())
    approx = [lv for lv, v in every if v["approx_medians"]]
    if approx:
        w(f"(medians of {', '.join(approx)}: weighted means of per-run medians, played in several runs)")
    w("still: decisions whose chosen plan has no moves other than wait / pad (plan.analyze_run.is_still)")
    w("deaths by cause (all logged waves):")
    for lv, v in every:
        if v["deaths"]:
            w(f"  {lv}: " + ", ".join(f"{c} {cause}" for cause, c in v["deaths"].items()))
    w("position error p50 (n), followed | not followed:")
    for lv, v in every:
        pe = v["position_error"]
        if pe:
            w(f"  {lv}: " + "  ".join(f"+{hz} {_f(e['followed'][0], '.1f')} ({e['followed'][2]}) | "
                                      f"{_f(e['not_followed'][0], '.1f')} ({e['not_followed'][2]})"
                                      for hz, e in pe.items()))
    return "\n".join(out)


def write_record(rec: dict, name: str, out_dir: str = OUT_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    return path


def report(folders, label=None, set_baseline=False, spec=None, runs_dir=RUNS, out_dir=OUT_DIR,
           k: int = K_DEFAULT) -> dict:
    spec = spec or load_spec()
    rec = build_record(folders, spec, label, runs_dir, k)
    bp = os.path.join(out_dir, BASELINE + ".json")
    base = None
    if os.path.exists(bp):
        with open(bp, encoding="utf-8") as f:
            base = json.load(f)
        rec["vs_baseline"] = {"label": base.get("label"), "deltas": deltas(rec, base)}
    print(table(rec, base))
    print(f"wrote {write_record(rec, rec['label'], out_dir)}")
    if set_baseline:
        print(f"baseline: wrote {write_record(rec, BASELINE, out_dir)}")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    sub.add_parser("cmd", help="print the agent.py command and the level / wave plan")
    rp = sub.add_parser("report", help="aggregate run folder(s) into one benchmark record")
    rp.add_argument("runs", nargs="+", help="run folders (runs/<name>, a name under runs/, or 'latest')")
    rp.add_argument("--k", type=int, default=K_DEFAULT,
                    help=f"waves per level that count, the first K (default {K_DEFAULT}); later waves are ignored")
    rp.add_argument("--label", default=None, help="record name (default: the run folders' names joined by +)")
    rp.add_argument("--set-baseline", action="store_true", help="also write this record as baseline.json")
    a = ap.parse_args(argv)
    spec = load_spec()
    if a.what == "cmd":
        print(command(spec))
        drift = config_drift(spec)
        if drift:
            print("\nnote: agent.py's defaults changed since benchmark.json was frozen (frozen, now): "
                  + ", ".join(f"{k} {v[0]} now {v[1]}" for k, v in drift.items()))
        return None
    if a.k < 1:
        ap.error("--k must be at least 1")
    return report(a.runs, a.label, a.set_baseline, spec, k=a.k)


if __name__ == "__main__":
    main()
