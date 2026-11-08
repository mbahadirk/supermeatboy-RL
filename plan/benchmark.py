"""Fixed benchmark: the beam planner (agent.py) with a frozen config over a fixed World-1 level set, turned into
one comparable record per run (or set of runs), with before/after deltas against a saved baseline.

    python -m plan.benchmark cmd                       # the agent.py command to run, and the level / wave plan
    python -m plan.benchmark report runs/bench_20261108-120000 [more folders] [--label NAME] [--set-baseline]

Run from smbrl/ (the `plan` package lives there). The printed agent.py command is for the folder above smbrl/
(`python smbrl/agent.py ...`).

The spec is smbrl/benchmark.json: levels (disjoint from eval_levels.json's held-out levels, checked at load),
waves per level and the frozen agent.py config. The config there is resolved from agent.py (DEFAULT_MODEL,
REPLAN_DEFAULT, TUBE_DEFAULT, HYSTERESIS_FRAMES and build_parser()'s defaults) by freeze(), so a later default
change shows as a diff of benchmark.json when it is frozen again; `cmd` notes any drift.

Run folders are read only through plan/analyze_run.py: analyze(folder, level) per benchmark level. Per level:
waves, finishes, completion (finishes / waves), best seconds, deaths by cause (and per wave), decisions, median
total ms per decision, median replan interval, the still share (decisions whose chosen plan has no moves other
than wait / pad: plan.analyze_run.is_still) and the position error per horizon (followed / not followed).
Records go to runs/benchmarks/<label>.json; --set-baseline also writes runs/benchmarks/baseline.json.

Comparability: completion is over the waves actually played per level (shown with every record), not trimmed to
a fixed K; fixed-K trimming is a deliberate later refinement, not silently omitted. A level played in several
run folders sums its counts exactly; its medians (ms, replan interval, position error) are then weighted means
of the per-run medians (marked approx). The config recorded is the frozen spec's: run folders do not log
agent.py's arguments, so it is what the benchmark asked for, not verified from the run. The model's mtime and
size and the git commit are read when the report is made.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from plan.analyze_run import HORIZONS, RUNS, analyze, resolve_run      # noqa: E402

SPEC = os.path.join(HERE, "benchmark.json")
EVAL = os.path.join(HERE, "eval_levels.json")
OUT_DIR = os.path.join(RUNS, "benchmarks")
BASELINE = "baseline"
# config key -> agent.py flag, in the order emitted
FLAGS = (("model", "--model"), ("budget_ms", "--budget-ms"), ("replan", "--replan"), ("tube", "--tube"),
         ("hysteresis", "--hysteresis"), ("modes", "--modes"), ("triggers", "--triggers"), ("agents", "--agents"),
         ("end_after", "--end-after"), ("wave_seconds", "--wave-seconds"), ("seed", "--seed"),
         ("threads", "--threads"))
COMPARABILITY = ("completion is over the waves actually played per level (shown), not trimmed to a fixed K; "
                 "fixed-K trimming is a deliberate later refinement")


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
    return spec


def _arg(v) -> str:
    s = str(v)
    return f'"{s}"' if " " in s else s


def command(spec: dict, now: float | None = None) -> str:
    name = time.strftime("bench_%Y%m%d-%H%M%S", time.localtime(now))
    cfg = spec["config"]
    cmd = "python smbrl/agent.py " + " ".join(f"{fl} {_arg(cfg[k])}" for k, fl in FLAGS)
    cmd += f" --name {name} --no-overlay"
    out = ["run from the folder above smbrl/:", "  " + cmd, "",
           f"in that one run, play each level for {spec['waves_per_level']} waves:"]
    out += [f"  {lv}: {spec['waves_per_level']} waves" for lv in spec["levels"]]
    out += ["", "then, from smbrl/:", f"  python -m plan.benchmark report {name}"]
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


def aggregate(results: list, spec: dict) -> tuple[dict, dict]:
    """results: [(run folder, {level: analyze(folder, level)})] -> per level and overall."""
    levels = {}
    for lv in spec["levels"]:
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
        dec = sum(r["decisions"] for _, r, _ in parts)
        still = sum(L.get("still_n") or 0 for _, _, L in parts)    # analyze_run.is_still decisions
        ms = [((r["timing_ms"].get("total") or [None])[0], r["decisions"]) for _, r, _ in parts]
        rp = [(r["replan_interval"][0], r["decisions"]) for _, r, _ in parts]
        levels[lv] = {
            "waves": waves, "waves_planned": spec["waves_per_level"], "finishes": fins,
            "completion": _ratio(fins, waves), "best_s": min(bests) if bests else None,
            "deaths": dict(deaths.most_common()), "deaths_total": sum(deaths.values()),
            "deaths_per_wave": _ratio(sum(deaths.values()), waves), "decisions": dec,
            "still_decisions": still, "still": _ratio(still, dec),
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
               "levels_finished": sum(v["finishes"] > 0 for v in vs), "levels": len(levels)}
    return levels, overall


def build_record(folders: list, spec: dict, label: str | None = None, runs_dir: str = RUNS) -> dict:
    runs = [resolve_run(f, runs_dir) for f in folders]
    results = [(run, {lv: analyze(run, lv) for lv in spec["levels"]}) for run in runs]
    levels, overall = aggregate(results, spec)
    return {"label": label or "+".join(os.path.basename(r) for r in runs),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "comparability": COMPARABILITY,
            "levels_spec": spec["levels"], "waves_per_level": spec["waves_per_level"],
            "config": spec["config"], "model": model_info(spec["config"]["model"]), "git_commit": git_commit(),
            "runs": [{"run": r, "skipped_lines": {k: v for per in res.values() for k, v in per["skipped_lines"].items()}}
                     for r, res in results],
            "wave_counts": {lv: v["waves"] for lv, v in levels.items()},
            "levels": levels, "overall": overall}


DELTA_KEYS = ("completion", "best_s", "deaths_per_wave", "ms_per_decision", "still")


def deltas(rec: dict, base: dict) -> dict:
    def d(a, b, k):
        x, y = (a or {}).get(k), (b or {}).get(k)
        return None if x is None or y is None else round(x - y, 4)
    out = {lv: {k: d(rec["levels"][lv], base.get("levels", {}).get(lv), k) for k in DELTA_KEYS}
           for lv in rec["levels"]}
    out["overall"] = {k: d(rec["overall"], base.get("overall"), k) for k in DELTA_KEYS}
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


def table(rec: dict, base: dict | None = None) -> str:
    out = []
    w = out.append
    m = rec["model"]
    w(f"benchmark {rec['label']} ({rec['created']}); git {rec['git_commit'] or 'unknown'}")
    w("runs: " + ", ".join(os.path.basename(r["run"]) + (f" (skipped lines {r['skipped_lines']})"
                                                        if r["skipped_lines"] else "") for r in rec["runs"]))
    w(f"model {m['path']}" + (f" (mtime {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(m['mtime']))}, "
                             f"{m['size']} bytes)" if m["exists"] else " (missing)"))
    w("config (frozen, not verified from the run): " + " ".join(f"{k}={v}" for k, v in rec["config"].items()))
    w("note: " + rec["comparability"])
    dl = None
    if base is not None:
        dl = deltas(rec, base)
        bm = base.get("model") or {}
        w(f"baseline {base.get('label')} (git {base.get('git_commit') or 'unknown'})"
          + ("; MODEL CHANGED since the baseline" if (bm.get("mtime"), bm.get("size"), bm.get("path"))
             != (m["mtime"], m["size"], m["path"]) else "")
          + ("; CONFIG DIFFERS from the baseline" if base.get("config") != rec["config"] else ""))
    hdr = (f"{'level':<8}{'waves':>8}{'fin':>5}{'compl':>7}{'best_s':>8}{'d/wave':>8}{'dec':>7}{'ms/dec':>8}{'replan':>7}"
           f"{'still':>7}")
    if dl:
        hdr += f"  | {'d compl':>8}{'d best':>8}{'d d/w':>8}{'d ms':>8}{'d still':>8}"
    w(hdr)
    rows = list(rec["levels"].items()) + [("overall", rec["overall"])]
    for lv, v in rows:
        s = (f"{lv:<8}{str(v['waves']) + '/' + str(v['waves_planned']):>8}{v['finishes']:>5}{_p(v['completion']):>7}"
             f"{_f(v.get('best_s')):>8}{_f(v['deaths_per_wave']):>8}{v['decisions']:>7}"
             f"{_f(v['ms_per_decision'], '.0f'):>8}{_f(v.get('replan_interval'), '.0f'):>7}{_p(v.get('still')):>7}")
        if dl:
            x = dl[lv]
            s += (f"  | {_d(x['completion'], '', True):>8}{_d(x['best_s'], '.2f'):>8}"
                  f"{_d(x['deaths_per_wave'], '.2f'):>8}{_d(x['ms_per_decision'], '.0f'):>8}"
                  f"{_d(x['still'], '', True):>8}")
        w(s)
    approx = [lv for lv, v in rec["levels"].items() if v["approx_medians"]]
    if approx:
        w(f"(medians of {', '.join(approx)}: weighted means of per-run medians, played in several runs)")
    w("still: decisions whose chosen plan has no moves other than wait / pad (plan.analyze_run.is_still)")
    w("deaths by cause:")
    for lv, v in rec["levels"].items():
        if v["deaths"]:
            w(f"  {lv}: " + ", ".join(f"{c} {k}" for k, c in v["deaths"].items()))
    w("position error p50 (n), followed | not followed:")
    for lv, v in rec["levels"].items():
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


def report(folders, label=None, set_baseline=False, spec=None, runs_dir=RUNS, out_dir=OUT_DIR) -> dict:
    spec = spec or load_spec()
    rec = build_record(folders, spec, label, runs_dir)
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
    return report(a.runs, a.label, a.set_baseline, spec)


if __name__ == "__main__":
    main()
