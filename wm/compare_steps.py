"""Open-loop position error of two world models (e.g. the 1-frame and the step-4 one) on the same held-out waves.

    python -m wm.compare_steps <wm_data folders...> --models runs/wm_locked/wm_model.pt runs/wm_step4/wm_model.pt
                               [--frames 4 8 16 32 48] [--starts 20] [--out report.md]

Waves: every shard of the given folders that BOTH models never trained on (a folder not in a model's info "data",
or a shard wm.train held out for it: every 5th in that model's own folder order). From --starts random frames per
wave, every agent alive there with a recorded action is rolled out (wm.rollout.rollout_batched, the ensemble
members' mean position) with the actions it really took and the recorded hazards / platforms; the error is the
distance to the recorded position --frames later (agents that died or left the recording before are left out
at that horizon). A step-k model rolls ceil(h / k) steps (h a multiple of k: exactly h frames).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from wm.buffer import load_shards, load_level            # noqa: E402
from wm.features import LevelGeom                         # noqa: E402
from wm.model import WorldModel                           # noqa: E402
from wm.rollout import World, rollout_batched             # noqa: E402
from shutdown import immediate                            # noqa: E402


def norm(d: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(os.path.join(HERE, d) if not os.path.isabs(d) else d)))


def held_out(model: WorldModel) -> tuple[set, set]:
    """(the model's training folders, its held-out shards as (folder, index in folder))."""
    dirs = [norm(d) for d in model.info.get("data", [])]
    held, no = set(), 0
    for d in dirs:
        for k, _ in enumerate(sorted(glob.glob(os.path.join(d, "wave_*.npz")))):
            no += 1
            if no % 5 == 0:
                held.add((d, k))
    return set(dirs), held


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--frames", nargs="+", type=int, default=[4, 8, 16, 32, 48])
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    immediate()
    torch.set_num_threads(args.threads)
    models = [WorldModel.load(m).eval() for m in args.models]
    trained = [held_out(m) for m in models]
    H = max(args.frames)
    rng = np.random.default_rng(0)
    err = [{h: [] for h in args.frames} for _ in models]
    waves, levels = 0, set()
    for d in args.data:
        nd = norm(d)
        geoms = {}
        for k, (meta, rows, world) in enumerate(load_shards(d)):
            if any(nd in dirs and (nd, k) not in held for dirs, held in trained):
                continue                                    # a wave one of the models trained on
            lv = meta.get("level")
            wf = {w["f"]: w for w in world}
            frames = sorted(wf)
            if len(frames) < H + 2:
                continue
            if lv not in geoms:
                geoms[lv] = LevelGeom(*load_level(d, lv))
            geom = geoms[lv]
            by = {(int(r["agent"]), int(r["frame"])): r for r in rows}
            starts = sorted(rng.choice(frames[1:-H], min(args.starts, len(frames) - H - 1), replace=False))
            geom.reset()
            keep, fans, upto = None, [], -1
            for f0 in starts:
                for ff in range(upto + 1, f0 + 1):
                    rec = wf.get(ff, {})
                    for idx, word in rec.get("tc", []) or []:
                        geom.set_tile(idx, word)
                    keep = rec.get("keep", keep)
                    if "fx" in rec:
                        fans = rec["fx"].get("fans", [])
                upto = f0
                agents = [a for (a, f) in by if f == f0 and by[(a, f)]["action"] >= 0 and by[(a, f)]["status"] == 0]
                if not agents:
                    continue
                s0 = np.array([by[(a, f0)] for a in agents])
                acts = np.zeros((len(agents), H), np.int64)
                truth = np.full((len(agents), H + 1, 2), np.nan)
                for i, a in enumerate(agents):
                    for j in range(H + 1):
                        r = by.get((a, f0 + j))
                        if r is None:
                            break
                        if r["status"] != 0:
                            break
                        truth[i, j] = (r["x"], r["y"])
                        if j < H:
                            acts[i, j] = max(0, int(r["action"]))
                        if j < H and r["action"] < 0:
                            break
                worlds = [World.from_record(wf.get(f0 + j, wf[frames[-1]]), wf.get(f0 + j - 1), keep, fans)
                          for j in range(H)]
                dts = np.array([wf.get(f0 + j, {}).get("dt") or 16.667 for j in range(H)])
                for mi, m in enumerate(models):
                    out = rollout_batched(m, geom, s0, acts, dts, worlds)
                    st = out.get("step", 1)
                    x, y = out["x"].mean(0), out["y"].mean(0)
                    for h in args.frames:
                        if h % st:
                            continue
                        e = np.hypot(x[:, h // st] - truth[:, h, 0], y[:, h // st] - truth[:, h, 1])
                        err[mi][h] += e[np.isfinite(e)].tolist()
            waves += 1
            levels.add(lv)
            print(f"{d} wave {k} ({lv}) done", flush=True)
    lines = [f"# Open-loop position error, {waves} held-out waves ({', '.join(sorted(levels))})", "",
             "| model | step | " + " | ".join(f"+{h} median / 90% (n)" for h in args.frames) + " |",
             "|---|---|" + "---|" * len(args.frames)]
    for mi, m in enumerate(models):
        cells = []
        for h in args.frames:
            e = np.asarray(err[mi][h])
            cells.append("-" if not len(e) else f"{np.median(e):.2f} / {np.percentile(e, 90):.2f} ({len(e)})")
        lines.append(f"| {args.models[mi]} | {m.step} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fo:
            fo.write(text + "\n")


if __name__ == "__main__":
    main()
