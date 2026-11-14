"""Online learning between waves (agent.py --online): the world model learns from the run's own play while the
agent plays, and the agent takes a better model between waves.

    python -m wm.online --init MODEL --replay REPLAY.npz --data DIR [DIR ...] --out OUT

The run records every wave (wm.buffer.Collector: wave_*.npz with its world file). After each new wave this child
process:
1. builds the new wave's transitions with the model's own feature encoding (wm.features.build on that one shard),
   held-out levels included (the run's own play is what is learned from; the base model is never written);
2. takes the newest wave as the test ("fresh") and every earlier wave of the run as the training data;
3. fine-tunes a copy of the published model on the run's training data mixed 1:9 with a replay sample of the base
   model's training data (--replay: runs/experiments/human_v3/make_replay.py), so it learns the new physics
   without forgetting the rest (AdamW, lr LR, OWN_EPOCHS passes over the run's data);
4. publishes the copy as OUT/wm_model.pt (atomic replace; its info "online": version, the losses) only when its
   one-step loss on the fresh wave is FRESH_GAIN lower than the published model's AND its loss on the replay's
   validation rows is at most REPLAY_SLACK higher (a model that learns the new level by forgetting the old ones
   is refused).
The fresh wave is trained on in the next cycle (the test always comes from play the candidate never saw).
Every cycle adds a line to OUT/online.log. Below-normal priority (Windows), --threads 2: the game keeps its speed.

Settings (runs/experiments/online_v1: 5 recorded 3-2 waves of runs/human_v3_check01, human_v3): lr 1e-4 with half
replay learned the new waves (fresh loss 0.007 -> -0.035) and forgot (replay +6%; the agent's own 1-8 waves +48
frames p90 8.5 -> 11.3); lr 3e-5 with 90% replay learned almost as much (-0.027) with the replay loss +0-4% and the
other levels' open-loop errors even or a little lower (1-8 +48 3.6 / 8.5 -> 3.3 / 8.3, 3-1 5.3 / 68.0 -> 5.0 / 67.0).
The replay loss moves 0-4% between equal settings, so the slack is 5%. ANCHOR (a pull toward the start weights)
kept the replay loss flat too but learned half as much (sweep.py); it is off.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

LR = 3e-5
OWN_EPOCHS = 4                     # passes over the run's own training transitions per cycle
BATCH = 512
REPLAY_SHARE = 0.9                 # of each batch: rows of the base training data (the rest: the run's own)
ANCHOR = 0.0                       # weight of sum((w - w_start)^2) in the loss: keeps the copy near its start
MIN_SAMPLES = 8192                 # a cycle trains on at least this many samples (own + replay)
FRESH_GAIN = 0.01                  # publish when the fresh wave's loss falls by at least this share ...
REPLAY_SLACK = 0.05                # ... and the replay validation loss rises by at most this share
SETTLE_S = 1.5                     # a shard is read only when it and its world file are this old (fully written)
POLL_S = 2.0
KEYS = ("X", "yc", "yb", "ys", "yr", "wc")


def log_line(out: str, text: str):
    line = time.strftime("%H:%M:%S ") + text
    print(line, flush=True)
    with open(os.path.join(out, "online.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ready(path: str) -> bool:
    world = path[:-4] + "_world.json.gz"
    now = time.time()
    return os.path.exists(world) and now - os.path.getmtime(path) > SETTLE_S and now - os.path.getmtime(world) > SETTLE_S


def featurize(shard: str, model, tmp: str) -> dict | None:
    """One shard's transitions in the model's encoding (wm.features.build on a folder holding only this shard and
    its level file), with the targets the model trains on (yc in its res_scale units; wc: full chains). None: the
    shard has no usable transitions (its level file missing, or nothing alive)."""
    from wm.features import RES_SCALE, build, model_layout, model_prior
    d = os.path.dirname(shard)
    base = os.path.basename(shard)[:-4]
    meta = json.loads(str(np.load(shard)["meta"]))
    lv = meta.get("level")
    level_file = os.path.join(d, "levels", f"{lv}.npz")
    if not os.path.exists(level_file):
        return None
    work = tempfile.mkdtemp(dir=tmp)
    try:
        os.makedirs(os.path.join(work, "levels"))
        shutil.copy2(shard, os.path.join(work, base + ".npz"))
        shutil.copy2(os.path.join(d, base + "_world.json.gz"), os.path.join(work, base + "_world.json.gz"))
        shutil.copy2(level_file, os.path.join(work, "levels", f"{lv}.npz"))
        b = build([work], log=lambda *a: None, step=model.step, protocol=model_layout(model).protocol,
                  prior_protocol=model_prior(model))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if not len(b["X"]):
        return None
    out = {"X": b["X"], "yb": b["yb"], "ys": b["ys"], "yr": b["yr"], "level": np.full(len(b["X"]), lv)}
    if model.step > 1:
        out["yc"] = (b["yc"] * RES_SCALE / np.asarray(model.res_scale, np.float32)).astype(np.float32)
        out["wc"] = b["full"].astype(np.float32)
    else:
        out["yc"] = b["yc"].astype(np.float32)
    return out


def cat(parts: list) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def loss_of(model, d: dict, bs: int = 4096) -> float:
    """The model's mean one-step training loss on these transitions (wm.model.WorldModel.loss)."""
    import torch
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
    total, n = 0.0, len(d["X"])
    model.eval()
    with torch.no_grad():
        for i in range(0, n, bs):
            s = slice(i, i + bs)
            loss, _ = model.loss(T(d["X"][s]), T(d["yc"][s]), T(d["yb"][s]), T(d["ys"][s], torch.long), T(d["yr"][s]),
                                 T(d["wc"][s]) if "wc" in d else None)
            total += float(loss) * len(d["X"][s])
    return total / max(n, 1)


def fine_tune(model, own: dict, replay: dict, rng) -> int:
    """OWN_EPOCHS passes over `own`, each batch REPLAY_SHARE replay rows (training split), the rest own. Returns
    the steps."""
    import torch
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
    n_own = len(own["X"])
    k_own = max(1, int(round(BATCH * (1.0 - REPLAY_SHARE))))
    steps = max(1, max(OWN_EPOCHS * n_own, int(MIN_SAMPLES * (1.0 - REPLAY_SHARE))) // k_own)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0 if ANCHOR else 1e-5)
    start = [p.detach().clone() for p in model.parameters()] if ANCHOR else None
    rtr = np.nonzero(~replay["val"])[0]
    model.train()
    for _ in range(steps):
        a = rng.integers(0, n_own, k_own)
        b = rtr[rng.integers(0, len(rtr), BATCH - k_own)]
        batch = {k: np.concatenate([own[k][a], replay[k][b]]) for k in KEYS if k in own}
        loss, _ = model.loss(T(batch["X"]), T(batch["yc"]), T(batch["yb"]), T(batch["ys"], torch.long),
                             T(batch["yr"]), T(batch["wc"]) if "wc" in batch else None)
        if start is not None:
            loss = loss + ANCHOR * sum(((p - p0) ** 2).sum() for p, p0 in zip(model.parameters(), start))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    model.eval()
    return steps


def load_replay(path: str, model) -> dict:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    if int(meta["step"]) != model.step:
        raise SystemExit(f"replay {path}: step {meta['step']}, the model's is {model.step}")
    return {k: z[k] for k in z.files if k != "meta"}


def cycle(model, own_parts: list, fresh_parts: list, replay: dict, rval: dict, rng, version: int):
    """One candidate from `model`: (candidate or None, the losses {fresh: [old, new], replay: [old, new]},
    training steps)."""
    own, fresh = cat(own_parts), cat(fresh_parts)
    old_fresh, old_replay = loss_of(model, fresh), loss_of(model, rval)
    cand = copy.deepcopy(model)
    steps = fine_tune(cand, own, replay, rng)
    new_fresh, new_replay = loss_of(cand, fresh), loss_of(cand, rval)
    losses = {"fresh": [round(old_fresh, 4), round(new_fresh, 4)], "replay": [round(old_replay, 4), round(new_replay, 4)]}
    ok = new_fresh <= old_fresh - FRESH_GAIN * abs(old_fresh) and new_replay <= old_replay + REPLAY_SLACK * abs(old_replay)
    return (cand if ok else None), losses, steps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="the model the run started with (never written)")
    ap.add_argument("--replay", required=True, help="replay sample of the base training data (make_replay.py)")
    ap.add_argument("--data", nargs="+", required=True, help="the run's wm_data folders (recorded waves)")
    ap.add_argument("--out", required=True, help="folder for wm_model.pt (published), online.log, online.json")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--once", action="store_true", help="one pass over what is there now, then exit (tests)")
    args = ap.parse_args(argv)
    import torch
    from wm.model import WorldModel
    torch.set_num_threads(args.threads)
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
        except Exception:
            pass
    os.makedirs(args.out, exist_ok=True)
    tmp = os.path.join(args.out, "tmp")
    os.makedirs(tmp, exist_ok=True)
    pub = os.path.join(args.out, "wm_model.pt")
    model = WorldModel.load(pub if os.path.exists(pub) else args.init).eval()
    version = int(((model.info or {}).get("online") or {}).get("version", 0))
    replay = load_replay(args.replay, model)
    rval = {k: v[replay["val"]] for k, v in replay.items() if k in KEYS}
    rng = np.random.default_rng(0)
    parts: dict[str, dict | None] = {}            # shard -> its transitions (None: nothing usable)
    log_line(args.out, f"start: model {pub if os.path.exists(pub) else args.init} (v{version}), replay "
                       f"{len(replay['X'])} rows, data {args.data}")
    while True:
        shards = sorted(p for d in args.data for p in glob.glob(os.path.join(d, "wave_*.npz")))
        new = [p for p in shards if p not in parts and ready(p)]
        for p in new:
            try:
                parts[p] = featurize(p, model, tmp)
            except Exception as e:                # a shard still being written, or broken: tried again later
                log_line(args.out, f"{os.path.basename(p)}: not read ({type(e).__name__}: {e})")
                continue
        usable = [p for p in new if parts.get(p) is not None]
        test = usable[-1:]                        # the newest wave is the test; earlier new ones are trained on
        fresh = [parts[p] for p in test]
        own = [v for p, v in parts.items() if v is not None and p not in test]
        if fresh and own:
            t0 = time.time()
            cand, losses, steps = cycle(model, own, fresh, replay, rval, rng, version)
            n_own = sum(len(p["X"]) for p in own)
            levels = sorted({str(lv) for p in fresh for lv in np.unique(p["level"])})
            if cand is not None:
                version += 1
                cand.info = dict(cand.info or {}, online={"version": version, "init": args.init,
                                                          "own_transitions": n_own, "losses": losses,
                                                          "time": time.strftime("%Y-%m-%d %H:%M:%S")})
                cand.save(pub + ".tmp", cand.info)
                os.replace(pub + ".tmp", pub)
                model = cand
            log_line(args.out, f"cycle: {len(own)} waves ({n_own} transitions) trained, fresh {len(fresh)} wave(s) of "
                               f"{', '.join(levels)}: fresh loss {losses['fresh'][0]} -> {losses['fresh'][1]}, replay "
                               f"{losses['replay'][0]} -> {losses['replay'][1]}, {steps} steps "
                               f"[{time.time() - t0:.1f}s]: " + (f"published v{version}" if cand is not None else "kept"))
        elif fresh:
            log_line(args.out, f"{len(fresh)} first wave(s) read: the next one is the first test")
        if args.once:
            return version
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
