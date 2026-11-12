"""Open-loop position error of two world models (e.g. the 1-frame and the step-4 one) on the same held-out waves.

    python -m wm.compare_steps --help

Choose --manifest FILE or explicitly --legacy-split. Strict comparison requires every checkpoint to record
the identical trace-split-v1 manifest; only validation agents are rolled out. Legacy checkpoints cannot be
silently evaluated against a new split. The following global-modulo selection describes legacy mode only.

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

from wm.buffer import load_shard, load_shards, load_level, shard_files    # noqa: E402
from wm.features import LevelGeom, model_layout           # noqa: E402
from wm.model import WorldModel                           # noqa: E402
from wm.rollout import World, rollout_batched             # noqa: E402
from shutdown import immediate                            # noqa: E402


def norm(d: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(os.path.join(HERE, d) if not os.path.isabs(d) else d)))


def held_out(model: WorldModel) -> tuple[set, set]:
    """(the model's training folders, its held-out shards as (folder, index in folder))."""
    if (model.info or {}).get("dataset", {}).get("protocol") == "trace-split-v1":
        raise ValueError("strict checkpoint membership comes from its manifest, not legacy held_out()")
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


def level_geoms(level, models) -> dict:
    """One level's geometry (words, meta) as each of these models' input encoding: {protocol: LevelGeom}."""
    return {p: LevelGeom(*level, protocol=p) for p in sorted({model_layout(m).protocol for m in models})}


def table(title: str, err: list, names: list, models: list, frames: list) -> list:
    from wm.protocol import feature_usage
    use = [feature_usage(m.info) for m in models]
    lines = [title, "", "Features active: " + ", ".join(u["active"] for u in use) + "; checkpoints trained: " +
             ", ".join(u["checkpoint_trained"] for u in use) + ".", "",
             "| model | step | " + " | ".join(f"+{h} median / 90% (n)" for h in frames) + " |",
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
    ap.add_argument("data", nargs="*")
    from wm.dataset import add_cli, cli_dataset
    add_cli(ap)
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
    dataset = cli_dataset(args, models[0])
    if dataset is not None:
        args.data = dataset.source_list(args.data)
        for model in models:
            dataset.require_model(model)
    elif not args.data:
        ap.error("--legacy-split needs data folders")
    if dataset is None:
        for model in models[1:]:
            cli_dataset(args, model)
    trained = [held_out(m) for m in models] if dataset is None else None
    rng = np.random.default_rng(0)
    err = [{h: [] for h in args.frames} for _ in models]
    err_demo = [{h: [] for h in args.frames} for _ in models]
    dem = demo_index(args.demos) if args.demos else {}
    waves, levels, demo_waves = 0, set(), []
    for d in args.data:
        nd = norm(d)
        geoms = {}
        for k, (meta, rows, world) in enumerate(dataset.iter_shards(d, split="validation")
                                               if dataset is not None else load_shards(d)):
            if dataset is None and any(nd in dirs and (nd, k) not in held for dirs, held in trained):
                continue                                    # a wave one of the models trained on
            lv = meta.get("level")
            if dataset is not None:
                geoms[lv] = level_geoms(dataset.level(d, lv), models)
                k = meta["_dataset_local_index"]
            if not roll_wave(d, lv, rows, world, geoms, rng, args.starts, err, models, args.frames,
                             dem.get((nd, k)), err_demo):
                continue
            if (nd, k) in dem:
                demo_waves.append(f"{lv} ({os.path.basename(os.path.dirname(nd))} wave {k})")
            waves += 1
            levels.add(lv)
            print(f"{d} wave {k} ({lv}) done", flush=True)
    protocol = dataset.reference() if dataset is not None else {"protocol": "legacy-global-mod5", "sha256": "none"}
    lines = table(f"# Open-loop position error, {waves} held-out waves ({', '.join(sorted(levels))})\n\n"
                  f"Dataset: {protocol['protocol']}; manifest {protocol['sha256']}", err,
                  args.models, models, args.frames)
    if args.demos:
        lines += [""] + table(f"## Demo rows of held-out waves: {len(demo_waves)} demos ({', '.join(demo_waves)}), "
                              f"starts every {DEMO_STRIDE} frames", err_demo, args.models, models, args.frames)
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fo:
            fo.write(text + "\n")


def roll_wave(d, lv, rows, world, geoms, rng, n_starts, err, models, horizons, demos=None, err_demo=None) -> bool:
    """One wave: n_starts random start frames (every agent alive there) into err, and the demo agents
    [(agent, first frame, last frame)] from every DEMO_STRIDE-th frame into err_demo. False: the wave is too short."""
    H = max(horizons)
    wf = {w["f"]: w for w in world}
    frames = sorted(wf)
    if len(frames) < H + 2:
        return False
    if lv not in geoms:
        geoms[lv] = level_geoms(load_level(d, lv), models)
    level_geom = geoms[lv]                              # {protocol: LevelGeom}
    by = {(int(r["agent"]), int(r["frame"])): r for r in rows}
    starts = sorted(rng.choice(frames[1:-H], min(n_starts, len(frames) - H - 1), replace=False))
    passes = [([(int(f0), None) for f0 in starts], err)]
    if demos:                                           # the demo agents, from every DEMO_STRIDE-th frame
        passes.append((sorted((f0, a) for a, fa, fb in demos for f0 in range(max(1, fa), fb, DEMO_STRIDE)),
                       err_demo))
    for pts, sink in passes:
        roll_starts(pts, sink, level_geom, wf, frames, by, H, models, horizons)
    return True


def wave_errors(models: list, waves: list, horizons=(16, 48), starts: int = 10, seed: int = 0, dataset=None) -> list:
    """The open-loop errors of each model on these waves [(wm_data folder, shard index)] (wm.learner's fixed
    evaluation): [{horizon: [errors]}] per model, the same metric as main."""
    rng = np.random.default_rng(seed)
    err = [{h: [] for h in horizons} for _ in models]
    if dataset is None and any((m.info or {}).get("dataset", {}).get("protocol") == "trace-split-v1" for m in models):
        raise ValueError("strict checkpoint errors require its manifest")
    if dataset is not None:
        for model in models:
            dataset.require_model(model)
        wanted = set(waves)
        with torch.no_grad():
            for meta, rows, world in dataset.iter_shards(split="validation"):
                if meta["_dataset_entry"] not in wanted:
                    continue
                lv, entry = meta["level"], dataset.entry(meta)
                geoms = {lv: level_geoms(dataset.level(entry["source"], lv), models)}
                roll_wave(entry["source"], lv, rows, world, geoms, rng, starts, err, models, list(horizons))
        return err
    want = {}
    for d, k in waves:
        want.setdefault(d, set()).add(int(k))
    with torch.no_grad():
        for d, ks in want.items():
            geoms = {}
            for k, p in enumerate(shard_files(d)):
                if k not in ks:
                    continue
                meta, rows, world = load_shard(p, k)
                roll_wave(d, meta.get("level"), rows, world, geoms, rng, starts, err, models, list(horizons))
    return err


def roll_starts(pts, err, geoms, wf, frames, by, H, models, horizons):
    """Roll the models out from each (start frame, agent or None = every agent alive there) in frame order.
    geoms: {protocol: LevelGeom} (level_geoms), each model reading its own encoding."""
    for geom in geoms.values():
        geom.reset()
    keep, fans, upto = None, [], -1
    for f0, only in pts:
        if f0 + H > frames[-1]:
            continue
        for ff in range(upto + 1, f0 + 1):
            rec = wf.get(ff, {})
            for idx, word in rec.get("tc", []) or []:
                for geom in geoms.values():
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
            out = rollout_batched(m, geoms[model_layout(m).protocol], s0, acts, dts, worlds)
            st = out.get("step", 1)
            x, y = out["x"].mean(0), out["y"].mean(0)
            for h in horizons:
                if h % st:
                    continue
                e = np.hypot(x[:, h // st] - truth[:, h, 0], y[:, h // st] - truth[:, h, 1])
                err[mi][h] += e[np.isfinite(e)].tolist()


if __name__ == "__main__":
    main()
