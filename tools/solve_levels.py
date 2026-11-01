"""Search a winning path (policy-free Go-Explore, env.prepare_level) for many imported levels in parallel.

Found paths are cached in runs/demos/ and start each level's backward curriculum in train.py, so a
campaign does not wait for the search. The report says which levels the clone can be finished in.

Run from the project root:
    python smbrl/tools/solve_levels.py --chapter hell                 # 4-1 .. 4-20
    python smbrl/tools/solve_levels.py --all --world both --seconds 120 --jobs 8
"""
from __future__ import annotations

import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _solve(name: str, seconds: float) -> tuple[str, float | None, float, str]:
    import env
    t0 = time.time()
    lines = []
    demo = env.prepare_level("smb:" + name, seconds=seconds, log=lines.append)
    frames = len(demo["states"]) if demo else None
    return name, (frames / 60.0 if frames else None), time.time() - t0, lines[-1] if lines else ""


def main():
    import levels_smb
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default=None, help="comma-separated level names")
    ap.add_argument("--chapter", default=None, help="forest, hospital, saltfactory, hell, rapture, drfetus, bandage, ...")
    ap.add_argument("--world", default="light", choices=["light", "dark", "both", "warp", "all"])
    ap.add_argument("--all", action="store_true", help="every chapter")
    ap.add_argument("--seconds", type=float, default=120.0, help="search budget per level (default 120)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    if args.levels:
        names = [n.strip().split(":", 1)[-1] for n in args.levels.split(",") if n.strip()]
    else:
        names = levels_smb.chapter_levels(None if args.all else args.chapter, args.world)
    pars = {c["file"]: c.get("par") for c in levels_smb.chapters()}
    print(f"{len(names)} levels, up to {args.seconds:g}s each, {args.jobs} at a time", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(_solve, n, args.seconds) for n in names]
        for f in as_completed(futs):
            name, secs, took, line = f.result()
            results[name] = secs
            par = pars.get(name)
            print(f"  {name:12s} " + (f"solved: {secs:5.1f}s run" + (f" (par {par:g})" if par else "")
                                       if secs else "not solved") + f"   [{took:.0f}s search]", flush=True)
    solved = [n for n in names if results.get(n)]
    print(f"\n{len(solved)}/{len(names)} solved in the clone; not solved: "
          + (" ".join(n for n in names if not results.get(n)) or "-"))


if __name__ == "__main__":
    main()
