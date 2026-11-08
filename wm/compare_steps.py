"""Open-loop position error of two world models (e.g. the 1-frame and the step-4 one) on the same held-out waves.

    python -m wm.compare_steps <wm_data folders...> --models runs/wm_locked/wm_model.pt runs/wm_step4/wm_model.pt
                               [--frames 4 8 16 32 48] [--starts 20] [--out report.md]

Waves: every shard of the given folders that BOTH models never trained on (a folder not in a model's info "data",
or a shard wm.train held out for it: every 5th in that model's own folder order). From --starts random frames per
wave, every agent alive there with a recorded action is rolled out (wm.rollout.rollout_batched, the ensemble
members' mean position) with the actions it really took and the recorded hazards / platforms; the error is the
distance to the recorded position --frames later (agents that died or left the recording before are left out
at that horizon). A step-k model rolls ceil(h / k) steps (h a multiple of k: exactly h frames).
--demos runs/demos/recorded.json (plan.demos recorded): a second table on the demo agents of those held-out waves
only, rolled out from every DEMO_STRIDE-th frame of each demo.
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

from wm.buffer import load_shards, load_level, shard_files    # noqa: E402
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


DEMO_STRIDE = 4


def demo_index(path: str) -> dict:
    """recorded.json -> {(norm folder, shard index): [(agent, first frame, last frame)]}."""
    import json
    with open(path, encoding="utf-8") as f:
        recs = json.load(f)
    out = {}
    for r in recs:
        d = norm(r["wm_data"])
        names = [os.path.basename(p) for p in shard_files(d)]
        if r["shard"] in names:
            out.setdefault((d, names.index(r["shard"])), []).append((int(r["agent"]), int(r["frames"][0]),
                                                                       int(r["frames"][1])))
    return out


def table(title: str, err: list, names: list, models: list, frames: list) -> list:
    lines = [title, "", "| model | step | " + " | ".join(f"+{h} median / 90% (n)" for h in frames) + " |",
             "|---|---|" + "---|" * len(frames)]
    for mi, m in enumerate(models):
        cells = []
        for h in frames:
            e = np.asarray(err[mi][h])
            cells.append("-" if not len(e) else f"{np.median(e):.2f} / {np.percentile(e, 90):.2f} ({len(e)})")
        lines.append(f"| {names[mi]} | {m.step} | " + " | ".join(cells) + " |")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--frames", nargs="+", type=int, default=[4, 8, 16, 32, 48])
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--demos", default=None, help="recorded.json: also the error on the demo agents' rows")
    args = ap.parse_args()
    immediate()
    torch.set_num_threads(args.threads)
    models = [WorldModel.load(m).eval() for m in args.models]
    trained = [held_out(m) for m in models]
    H = max(args.frames)
    rng = np.random.default_rng(0)
    err = [{h: [] for h in args.frames} for _ in models]
    err_demo = [{h: [] for h in args.frames} for _ in models]
    dem = demo_index(args.demos) if args.demos else {}
    waves, levels, demo_waves = 0, set(), []
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
            passes = [([(int(f0), None) for f0 in starts], err)]
            if (nd, k) in dem:                              # the demo agents, from every DEMO_STRIDE-th frame
                ds = sorted((f0, a) for a, fa, fb in dem[(nd, k)] for f0 in range(max(1, fa), fb, DEMO_STRIDE))
                passes.append((ds, err_demo))
                demo_waves.append(f"{lv} ({os.path.basename(os.path.dirname(nd))} wave {k})")
            for pts, sink in passes:
                roll_starts(pts, sink, geom, wf, frames, by, H, models, args.frames)
            waves += 1
            levels.add(lv)
            print(f"{d} wave {k} ({lv}) done", flush=True)
    lines = table(f"# Open-loop position error, {waves} held-out waves ({', '.join(sorted(levels))})", err,
                  args.models, models, args.frames)
    if args.demos:
        lines += [""] + table(f"## Demo rows of held-out waves: {len(demo_waves)} demos ({', '.join(demo_waves)}), "
                              f"starts every {DEMO_STRIDE} frames", err_demo, args.models, models, args.frames)
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fo:
            fo.write(text + "\n")


def roll_starts(pts, err, geom, wf, frames, by, H, models, horizons):
    """Roll the models out from each (start frame, agent or None = every agent alive there) in frame order."""
    geom.reset()
    keep, fans, upto = None, [], -1
    for f0, only in pts:
        if f0 + H > frames[-1]:
            continue
        for ff in range(upto + 1, f0 + 1):
            rec = wf.get(ff, {})
            for idx, word in rec.get("tc", []) or []:
                geom.set_tile(idx, word)
            keep = rec.get("keep", keep)
            if "fx" in rec:
                fans = rec["fx"].get("fans", [])
        upto = max(upto, f0)
        agents = [a for (a, f) in by if f == f0 and by[(a, f)]["action"] >= 0 and by[(a, f)]["status"] == 0
                  and (only is None or a == only)]
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
            for h in horizons:
                if h % st:
                    continue
                e = np.hypot(x[:, h // st] - truth[:, h, 0], y[:, h // st] - truth[:, h, 1])
                err[mi][h] += e[np.isfinite(e)].tolist()


if __name__ == "__main__":
    main()
