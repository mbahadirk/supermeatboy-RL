"""Finishes under the level's A+ time in recorded play (any source: your own play with agent.py --human, PPO with
realgame.py, the planner), as demos for the world model.

    python -m plan.aplus runs/human_20261114-120000 runs/collect_more_ppo50_02     run folders (or wm_data folders)
    python -m plan.aplus --all                                                       every runs/*/wm_data

Writes runs/demos/aplus.json in recorded.json's form ({wm_data, shard, agent, frames [0, finish row], level,
seconds, run, source}; wm.train --demos runs/demos/aplus.json) and prints a table per level.

A finish counts when every check holds; the others are counted with their reason:
- the agent's first status-2 row is at frame f (realgame.js reports a finish in update f - 1: the time is f frames,
  as agent.py times it) and f x dt / 1000 <= the level's A+ time (levels_smb.a_plus_time: the game's own par);
- the wave ran with the time-step lock: every world record's dt is the fixed step (16.667 ms +- 0.01);
- the agent's rows are frames 0..f without a gap, and it never moved more than JUMP_MAX units in one frame;
- not a held-out level (eval_levels.json) and not a level the user excluded (plan.demos.USER_EXCLUDED);
- not a duplicate: the same level with the same actions over frames 0..f as an earlier kept finish (waves repeat
  exactly under the lock, so a replayed route adds no new transitions)."""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from levels_smb import a_plus_time                       # noqa: E402
from plan.demos import user_excluded                     # noqa: E402
from wm.buffer import eval_levels, shard_files           # noqa: E402

OUT = os.path.join(HERE, "runs", "demos", "aplus.json")
FIXED_MS, DT_TOL = 16.667, 0.01
JUMP_MAX = 60.0                                          # units in one frame (a fall tops out near 50)
STATUS_FINISHED = 2


def wm_dirs(paths: list) -> list:
    out = []
    for p in paths:
        p = os.path.normpath(p)
        d = p if os.path.basename(p) == "wm_data" else os.path.join(p, "wm_data")
        if not os.path.isdir(d):
            raise SystemExit(f"no wm_data folder at {d}")
        out.append(d)
    return out


def finishes(d: str):
    """(record or None, reason, level) per finishing agent of every shard in wm_data folder d."""
    held = eval_levels()
    for p in shard_files(d):
        z = np.load(p)
        meta = json.loads(str(z["meta"]))
        level = meta.get("level")
        rows = z["rows"]
        fin = sorted({int(a) for a in rows["agent"][rows["status"] == STATUS_FINISHED]})
        if not fin:
            continue
        par = a_plus_time(level)
        why_level = ("held-out level" if level in held else "user-excluded level" if user_excluded(level) else
                     "no A+ time known" if par is None else None)
        world = None
        for a in fin:
            r = rows[rows["agent"] == a]
            r = r[np.argsort(r["frame"], kind="stable")]
            f = int(r["frame"][r["status"] == STATUS_FINISHED].min())
            if why_level:
                yield None, why_level, level
                continue
            if world is None:
                wp = p[:-4] + "_world.json.gz"
                world = json.load(gzip.open(wp, "rt", encoding="utf-8")) if os.path.exists(wp) else []
                dts = np.array([w.get("dt") or 0.0 for w in world], np.float64)
                locked = bool(len(dts)) and bool(np.all(np.abs(dts - FIXED_MS) <= DT_TOL))
            seconds = f * FIXED_MS / 1000.0
            if not locked:
                yield None, "not locked", level
                continue
            if seconds > par:
                yield None, "over A+", level
                continue
            pre = r[r["frame"] <= f]
            if len(pre) != f + 1 or int(pre["frame"][0]) != 0 or np.any(np.diff(pre["frame"]) != 1):
                yield None, "rows missing", level
                continue
            if np.hypot(np.diff(pre["x"]), np.diff(pre["y"])).max(initial=0.0) > JUMP_MAX:
                yield None, "teleport", level
                continue
            acts = pre["action"].astype(np.int8).tobytes()
            src = meta.get("source")
            for name, ags in (meta.get("agent_sources") or {}).items():
                if a in ags:
                    src = name
            run = os.path.basename(os.path.dirname(d))
            yield ({"wm_data": os.path.relpath(d, HERE).replace("\\", "/"), "shard": os.path.basename(p),
                    "agent": a, "frames": [0, f], "level": level, "seconds": round(seconds, 3), "a_plus": par,
                    "run": run, "source": src}, acts, level)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run folders or their wm_data folders")
    ap.add_argument("--all", action="store_true", help="every runs/*/wm_data")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    dirs = sorted(glob.glob(os.path.join(HERE, "runs", "*", "wm_data"))) if args.all else wm_dirs(args.runs)
    if not dirs:
        raise SystemExit("give run folders or --all")
    recs, table, seen = [], {}, set()
    for d in dirs:
        for rec, why, level in finishes(d):
            if rec is not None:
                key = (level, why)
                why = "duplicate" if key in seen else "ok"
                seen.add(key)
                if why == "duplicate":
                    rec = None
            t = table.setdefault(level, {})
            t[why] = t.get(why, 0) + 1
            if rec is not None:
                recs.append(rec)
                t.setdefault("best", rec["seconds"])
                t["best"] = min(t["best"], rec["seconds"])
                t.setdefault("sources", set()).add(str(rec["source"]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=1)
    os.replace(tmp, args.out)
    reasons = ["duplicate", "over A+", "not locked", "rows missing", "teleport", "held-out level", "user-excluded level",
               "no A+ time known"]
    print(f"{'level':10s} {'A+ s':>5s} {'kept':>5s} {'best s':>7s}  " + "  ".join(reasons) + "  sources")
    key = lambda s: [int(x) if x.isdigit() else x for x in str(s).replace("-", " ").split()]
    for level in sorted(table, key=key):
        t = table[level]
        par = a_plus_time(level)
        print(f"{str(level):10s} {'-' if par is None else f'{par:g}':>5s} {t.get('ok', 0):5d} "
              f"{t['best'] if 'best' in t else '-':>7}  " + "  ".join(f"{t.get(r, 0):{len(r)}d}" for r in reasons)
              + "  " + ",".join(sorted(t.get("sources", []))))
    print(f"{len(recs)} A+ finishes -> {args.out}")


if __name__ == "__main__":
    main()
