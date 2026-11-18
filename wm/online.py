"""Online learning between waves (agent.py --online): the world model learns from the run's own play while the
agent plays, and the agent takes a better model between waves.

    python -m wm.online --init MODEL --replay REPLAY.npz --data DIR [DIR ...] --out OUT

The run records every wave (wm.buffer.Collector: wave_*.npz with its world file). After each new wave this child
process:
1. builds the new wave's transitions with the model's own feature encoding (wm.features.build on that one shard),
   held-out levels included (the run's own play is what is learned from; the base model is never written);
2. takes the newest wave as the test ("fresh"); the run's other waves are either absorbed (a published model was
   trained on them) or pending (not yet);
3. fine-tunes a copy of the published model on the pending waves, rehearsing the absorbed ones (OLD_SHARE of each
   batch's own rows when there are any), mixed 1:9 with a replay sample of the base model's training data
   (--replay: runs/experiments/human_v3/make_replay.py), so it learns the new physics without forgetting the rest
   (AdamW, lr LR, OWN_EPOCHS passes over the pending rows, at most MAX_STEPS steps: a cycle's cost stays bounded
   however long the run is);
4. publishes the copy as OUT/wm_model.pt (atomic replace; its info "online": version, the losses, the bar) only
   when its one-step loss on the fresh wave is FRESH_GAIN lower than the published model's AND its loss on the
   replay's validation rows is at most REPLAY_SLACK above the base (--init) model's (a model that learns the new
   level by forgetting the old ones is refused); the pending waves are then absorbed. A refused cycle leaves them
   pending: they are trained on again next cycle with the newer ones.
The fresh wave is trained on in the next cycle (the test always comes from play the candidate never saw).
New agent.py shards (finish-aplus-v1) record finish rewards 5 / A+ 15 (targets after scaling: 1 / 3).
A+ trajectories get twice the sampling weight within pending/absorbed own rows; replay remains 90% of each
batch, validation is unweighted, and the publication gates are unchanged. Legacy/PPO shards are not relabelled.
This trains the world model, not a policy: the planner still chooses actions by estimated time and risk.
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

from wm.rewards import REWARD_PROTOCOL, FINISH_REWARD, A_PLUS_BONUS, A_PLUS_SAMPLE_WEIGHT

LR = 3e-5
OWN_EPOCHS = 4                     # passes over the pending own transitions per cycle
MAX_STEPS = 1500                   # a cycle trains at most this many steps (its cost does not grow with the run)
BATCH = 512
REPLAY_SHARE = 0.9                 # of each batch: rows of the base training data (the rest: the run's own)
OLD_SHARE = 0.3                    # of a batch's own rows: rows of absorbed waves (rehearsal), when there are any
ANCHOR = 0.0                       # weight of sum((w - w_start)^2) in the loss: keeps the copy near its start
MIN_SAMPLES = 8192                 # a cycle trains on at least this many samples (own + replay)
FRESH_GAIN = 0.01                  # publish when the fresh wave's loss falls by at least this share ...
REPLAY_SLACK = 0.05                # ... and the replay validation loss is at most this share above the base model's
# The forgetting bar is the base (--init) model's replay loss, not the published model's: the base model's replay
# val loss is 0.0597 (train 0.0065), so the first short fine-tune dropped it to 0.0203, and in runs/online_check01
# all 54 later candidates (replay 0.04-0.08, fresh loss much better, e.g. 2.59 -> 0.27) were refused against it.
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


def featurize(shard: str, model, tmp: str, log=None) -> dict | None:
    """One shard's transitions in the model's encoding (wm.features.build on a folder holding only this shard and
    its level file), with the targets the model trains on (yc in its res_scale units; wc: full chains). None: the
    shard has no usable transitions (its level file missing, or nothing alive). log(text): told how many samples
    went through a game portal and were dropped (wm.features.build), when any."""
    from wm.features import RES_SCALE, build, model_layout, model_prior, demo_key
    d = os.path.dirname(shard)
    base = os.path.basename(shard)[:-4]
    with np.load(shard) as z:
        meta = json.loads(str(z["meta"]))
        rows = z["rows"]
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
        # Reuse build's per-agent/frame selection so weighting stays aligned even
        # when portal-crossing transitions are dropped. Old/PPO rewards are not relabelled.
        examples = aplus_examples(meta, rows)
        demos = {demo_key(work): {int(rows["wave"][0]): examples}} if examples else None
        counts = {}
        b = build([work], log=lambda *a: None, step=model.step, protocol=model_layout(model).protocol,
                  prior_protocol=model_prior(model), counts=counts, demos=demos)
        if counts.get("teleport_dropped") and log is not None:
            log(f"{base}: {counts['teleport_dropped']} samples through a game portal dropped")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if not len(b["X"]):
        return None
    out = {"X": b["X"], "yb": b["yb"], "ys": b["ys"], "yr": b["yr"], "level": np.full(len(b["X"]), lv)}
    out["sample_weight"] = np.where(b["demo"], A_PLUS_SAMPLE_WEIGHT, 1.0).astype(np.float32)
    if examples and log is not None:
        log(f"{base}: {len(examples)} A+ trajectories, {int(b['demo'].sum())} transitions sampled "
            f"at {A_PLUS_SAMPLE_WEIGHT:g}x within own-data batches ({REWARD_PROTOCOL})")
    if model.step > 1:
        out["yc"] = (b["yc"] * RES_SCALE / np.asarray(model.res_scale, np.float32)).astype(np.float32)
        out["wc"] = b["full"].astype(np.float32)
    else:
        out["yc"] = b["yc"].astype(np.float32)
    return out


def aplus_examples(meta, rows):
    """Exact rewarded finish rows identify A+ lives, including partial-wave captures."""
    if meta.get("reward_protocol") != REWARD_PROTOCOL:
        return []
    finished = rows[(rows["status"] == 2) & (rows["reward"] >= FINISH_REWARD + A_PLUS_BONUS)]
    return [(int(row["agent"]), 0, int(row["frame"])) for row in finished]


def own_probabilities(data):
    """A+ lives get twice the sampling weight; replay share and validation stay unchanged."""
    weights = data.get("sample_weight")
    return None if weights is None else weights.astype(np.float64) / weights.sum(dtype=np.float64)


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


def train_steps(n_pending: int) -> int:
    """OWN_EPOCHS passes over n_pending own rows (at least MIN_SAMPLES samples), clipped to MAX_STEPS."""
    k_own = max(1, int(round(BATCH * (1.0 - REPLAY_SHARE))))
    return min(MAX_STEPS, max(1, max(OWN_EPOCHS * n_pending, int(MIN_SAMPLES * (1.0 - REPLAY_SHARE))) // k_own))


def fine_tune(model, pending: dict, replay: dict, rng, old: dict | None = None) -> int:
    """train_steps(len(pending)) steps; each batch REPLAY_SHARE replay rows (training split), the rest own: of those,
    OLD_SHARE from `old` (absorbed waves) when given, the rest from `pending`. Returns the steps."""
    import torch
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
    n_pend = len(pending["X"])
    n_old = len(old["X"]) if old is not None else 0
    k_own = max(1, int(round(BATCH * (1.0 - REPLAY_SHARE))))
    k_old = min(k_own - 1, int(round(k_own * OLD_SHARE))) if n_old else 0
    k_pend = k_own - k_old
    steps = train_steps(n_pend)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0 if ANCHOR else 1e-5)
    start = [p.detach().clone() for p in model.parameters()] if ANCHOR else None
    rtr = np.nonzero(~replay["val"])[0]
    p_pend = own_probabilities(pending)
    p_old = own_probabilities(old) if n_old else None
    # Build cumulative weights once, not an O(all rows) choice table on every step.
    cdf_pend = np.cumsum(p_pend) if p_pend is not None else None
    cdf_old = np.cumsum(p_old) if p_old is not None else None
    if cdf_pend is not None:
        cdf_pend[-1] = 1.0
    if cdf_old is not None:
        cdf_old[-1] = 1.0
    model.train()
    for _ in range(steps):
        a = (rng.integers(0, n_pend, k_pend) if cdf_pend is None else
             np.searchsorted(cdf_pend, rng.random(k_pend), side="right"))
        b = rtr[rng.integers(0, len(rtr), BATCH - k_own)]
        if k_old:
            c = (rng.integers(0, n_old, k_old) if cdf_old is None else
                 np.searchsorted(cdf_old, rng.random(k_old), side="right"))
            batch = {k: np.concatenate([pending[k][a], old[k][c], replay[k][b]]) for k in KEYS if k in pending}
        else:
            batch = {k: np.concatenate([pending[k][a], replay[k][b]]) for k in KEYS if k in pending}
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


def replay_mismatch(meta: dict, model) -> str | None:
    """Why a replay sample (its meta: make_replay.py) cannot be trained with this model, or None when it can: its
    step, feature protocol (wm.features.model_layout) and prior protocol (wm.features.model_prior; a sample that
    records none is gravity-v1) must all be the model's. The one owner of this rule (wm.online and agent.py --online)."""
    from wm.features import model_layout, model_prior
    from wm.protocol import GRAVITY_PRIOR
    have = (int(meta.get("step", -1)), meta.get("protocol"), meta.get("prior_protocol", GRAVITY_PRIOR))
    want = (int(model.step), model_layout(model).protocol, model_prior(model))
    if have == want:
        return None
    return (f"the replay sample is step {have[0]} / {have[1] or 'no recorded feature protocol'} / {have[2]}, the "
            f"model is step {want[0]} / {want[1]} / {want[2]} (step / feature protocol / prior): make a replay sample "
            f"of the model's own training features (runs/experiments/human_v3/make_replay.py: tile-shapes-v1, "
            f"runs/experiments/human_v4/make_replay.py --model <the model>: tile-accel-v1 -> replay.npz, tile-plat-v1 "
            f"-> replay_plat.npz, once that model exists) and pass it with --replay / --online-replay")


def load_replay(path: str, model) -> dict:
    z = np.load(path, allow_pickle=False)
    why = replay_mismatch(json.loads(str(z["meta"])), model)
    if why:
        raise SystemExit(f"replay {path}: {why}")
    return {k: z[k] for k in z.files if k != "meta"}


def replay_bar(base_replay: float) -> float:
    """The highest replay validation loss a candidate may have: REPLAY_SLACK above the base model's."""
    return base_replay + REPLAY_SLACK * abs(base_replay)


def cycle(model, pending_parts: list, fresh_parts: list, replay: dict, rval: dict, rng, version: int,
          base_replay: float, old_parts: list | None = None):
    """One candidate from `model`, trained on the pending waves (rehearsing `old_parts`, the absorbed ones):
    (candidate or None, the losses {fresh: [old, new], replay: [old, new], bar}, training steps). Accepted when the
    fresh loss falls by FRESH_GAIN and the replay loss is at most replay_bar(base_replay)."""
    pending, fresh = cat(pending_parts), cat(fresh_parts)
    old = cat(old_parts) if old_parts else None
    old_fresh, old_replay = loss_of(model, fresh), loss_of(model, rval)
    cand = copy.deepcopy(model)
    steps = fine_tune(cand, pending, replay, rng, old)
    new_fresh, new_replay = loss_of(cand, fresh), loss_of(cand, rval)
    bar = replay_bar(base_replay)
    losses = {"fresh": [round(old_fresh, 4), round(new_fresh, 4)], "replay": [round(old_replay, 4), round(new_replay, 4)],
              "bar": round(bar, 4)}
    ok = new_fresh <= old_fresh - FRESH_GAIN * abs(old_fresh) and new_replay <= bar
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
    base = WorldModel.load(args.init).eval()
    model = WorldModel.load(pub).eval() if os.path.exists(pub) else base
    version = int(((model.info or {}).get("online") or {}).get("version", 0))
    replay = load_replay(args.replay, model)
    rval = {k: v[replay["val"]] for k, v in replay.items() if k in KEYS}
    base_replay = loss_of(base, rval)             # the forgetting bar is the base model's replay loss
    del base
    rng = np.random.default_rng(0)
    parts: dict[str, dict | None] = {}            # shard -> its transitions (None: nothing usable)
    absorbed: set[str] = set()                    # shards a published model was trained on
    log_line(args.out, f"start: model {pub if os.path.exists(pub) else args.init} (v{version}), replay "
                       f"{len(replay['X'])} rows, base replay loss {base_replay:.4f} (bar {replay_bar(base_replay):.4f}), "
                       f"data {args.data}")
    while True:
        shards = sorted(p for d in args.data for p in glob.glob(os.path.join(d, "wave_*.npz")))
        new = [p for p in shards if p not in parts and ready(p)]
        for p in new:
            try:
                parts[p] = featurize(p, model, tmp, log=lambda t: log_line(args.out, t))
            except Exception as e:                # a shard still being written, or broken: tried again later
                log_line(args.out, f"{os.path.basename(p)}: not read ({type(e).__name__}: {e})")
                continue
        usable = [p for p in new if parts.get(p) is not None]
        test = usable[-1:]                        # the newest wave is the test; earlier new ones are trained on
        fresh = [parts[p] for p in test]
        pending = [p for p, v in parts.items() if v is not None and p not in test and p not in absorbed]
        old = [parts[p] for p in sorted(absorbed)]
        if fresh and pending:
            t0 = time.time()
            cand, losses, steps = cycle(model, [parts[p] for p in pending], fresh, replay, rval, rng, version,
                                        base_replay, old)
            n_pend = sum(len(parts[p]["X"]) for p in pending)
            levels = sorted({str(lv) for p in fresh for lv in np.unique(p["level"])})
            if cand is not None:
                version += 1
                absorbed |= set(pending)
                cand.info = dict(cand.info or {}, online={"version": version, "init": args.init,
                                                          "own_transitions": n_pend, "losses": losses,
                                                          "base_replay": round(base_replay, 4),
                                                          "absorbed_waves": len(absorbed),
                                                          "time": time.strftime("%Y-%m-%d %H:%M:%S")})
                cand.save(pub + ".tmp", cand.info)
                os.replace(pub + ".tmp", pub)
                model = cand
            log_line(args.out, f"cycle: {len(pending)} pending waves ({n_pend} transitions) trained, {len(old)} absorbed "
                               f"rehearsed, fresh {len(fresh)} wave(s) of {', '.join(levels)}: fresh loss "
                               f"{losses['fresh'][0]} -> {losses['fresh'][1]}, replay {losses['replay'][0]} -> "
                               f"{losses['replay'][1]} (bar {losses['bar']}), {steps} steps "
                               f"[{time.time() - t0:.1f}s]: " + (f"published v{version}" if cand is not None else "kept"))
        elif fresh:
            log_line(args.out, f"{len(fresh)} first wave(s) read: the next one is the first test")
        if args.once:
            return version
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
