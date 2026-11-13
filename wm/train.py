"""Train the world model on recorded gameplay and report its one-step accuracy.

    python -m wm.train --help

Choose --manifest FILE for trace-split-v1, or explicitly --legacy-split for historical global-modulo
compatibility. A manifest is a frozen, bounded dataset, not proof of unseen-level performance. Fresh strict
training needs a new empty output folder. Strict --init requires the identical manifest and matching --step;
legacy weights and manifest extensions cannot initialize it; a strict parent needs a known trained feature
protocol. Strict self-correction supports step 1 only. Explicit legacy --init with no fit preserves the
parent's trained feature protocol; inference's active protocol is reported separately.
--feature-protocol: fresh training's input encoding (default per-agent-key-tiles-v1, the layout every
existing checkpoint and the frozen experiment caches use; tile-shapes-v1 is an explicit opt-in). --init
always continues in the parent's encoding. The checkpoint records it (feature_protocol /
active_feature_protocol) and every consumer builds that encoding from it.
--prior-protocol: the physics prior the targets correct (wm/protocol.py; default gravity-v1, the prior of
every existing checkpoint; gravity-fan-wind-v1 adds the measured fan wind force: explicit opt-in). --init
keeps the parent's (a different value is an error). The checkpoint records it (prior_protocol) and rollouts
apply it.

Runs in its own process (it does not touch the game or the PPO trainer); --threads limits its CPU use so a
training run in the game next to it keeps its speed. In legacy mode every 5th recorded wave is held out for
the report: position / velocity errors against two simple guesses (same velocity; same velocity with
gravity), per situation (ground / air / wall), how sure the model is versus how wrong it is, and the events
(landing, death, finish, blocked, flags, movement state). Writes wm_model.pt and wm_report.md to --out.

--demos runs/demos/recorded.json (plan.demos recorded) --demo-share 0.2: the PPO agents' best finishes found in
the given wm_data folders are oversampled to about that share of the training samples (the real share is
printed); demo rows of held-out waves stay held out, held-out levels stay out. Replay folders given as data
(agent.py --demo: runs/demo_<level>_*/wm_data) add their shards' "demo" agents (meta agent_sources) as demo rows.
Demos the user excluded (plan.demos.USER_EXCLUDED: 1-16, 4-6) are never used: their replay folders are dropped from
the data and their recorded.json entries skipped (the levels' other rows stay).
Partial shards (a wave cut by S, a level change or a stop; meta "partial") are read like any other.
Train mode (agent.py --learn with an explicit dataset choice) fine-tunes with wm.learner (legacy discovers
new folders; a manifest learner stays on its frozen snapshot). In legacy mode it also uses
agent's own new bests in runs/demos/self.json); this script trains from scratch (--init: self-correction only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from wm.features import build, BIN_NAMES, RES_SCALE, EVENTS   # noqa: E402
from wm.model import WorldModel                           # noqa: E402
from wm.buffer import eval_levels, shard_files            # noqa: E402
from wm.features import demo_key                          # noqa: E402
from plan.demos import excluded_demo_folder, user_excluded  # noqa: E402
from shutdown import immediate                            # noqa: E402

GROUPS = ["ground", "air", "wall"]


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = float(label.sum()), float((~label).sum())     # (float: numpy's int is 32-bit on Windows)
    if pos == 0 or neg == 0:
        return float("nan")
    r = (np.argsort(np.argsort(score)) + 1).astype(np.float64)
    return float((r[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def to_raw(m: np.ndarray, scale=RES_SCALE) -> np.ndarray:
    """Model output (correction / the model's res_scale) -> game units (add the physics prior for a prediction)."""
    return m * scale


def predict_all(model, X, bs=8192):
    outs = {}
    dev = next(model.parameters()).device
    for i in range(0, len(X), bs):
        p = model.predict(torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=dev))
        for k, v in p.items():
            outs.setdefault(k, []).append(v.detach().cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


def report(model, d: dict, idx: np.ndarray) -> str:
    p = predict_all(model, d["X"][idx])
    raw = d["raw"][idx]
    rs = model.res_scale
    pred = d["prior"][idx] + to_raw(p["mean"], rs)
    std = to_raw(p["std"], rs)
    g = d["group"][idx]
    lines = [f"Held-out transitions: {len(idx)} (recorded waves never trained on)" +
             (f"; one step = {model.step} frames" if model.step > 1 else "") + ".", ""]

    def errs(pr):
        pos = np.hypot(pr[:, 0] - raw[:, 0], pr[:, 1] - raw[:, 1])
        vel = np.hypot(pr[:, 2] - raw[:, 2], pr[:, 3] - raw[:, 3])
        return pos, vel

    rows = [("world model", errs(pred)), ("same velocity", errs(d["base_const"][idx])),
            ("velocity + gravity", errs(d["base_grav"][idx]))]
    lines += ["### Position and velocity after one step (game units; a tile is 20)", "",
              "| predictor | situation | n | position error median | 90% | velocity error median | 90% |",
              "|---|---|---|---|---|---|---|"]
    for name, (pe, ve) in rows:
        for gi, gn in [(None, "all")] + list(enumerate(GROUPS)):
            m = np.ones(len(g), bool) if gi is None else g == gi
            if m.sum() == 0:
                continue
            lines.append(f"| {name} | {gn} | {int(m.sum())} | {np.median(pe[m]):.2f} | {np.percentile(pe[m], 90):.2f} | "
                         f"{np.median(ve[m]):.1f} | {np.percentile(ve[m], 90):.1f} |")
    within = (np.abs(pred - raw) <= 2 * np.maximum(std, 1e-6)).mean(0)
    lines += ["", "### How sure versus how right",
              f"Share of true values within 2 predicted standard deviations (about 95% if the model knows its own "
              f"uncertainty): dx {within[0]:.2f}, dy {within[1]:.2f}, vx {within[2]:.2f}, vy {within[3]:.2f}.",
              f"Ensemble spread (median, game units): dx {np.median(to_raw(p['spread'], rs)[:, 0]):.3f}, "
              f"vy {np.median(to_raw(p['spread'], rs)[:, 3]):.2f}.", ""]
    yb = d["yb"][idx] > 0.5
    lines += ["### Events and flags after one step", "", "| event | true rate | accuracy | precision | recall | AUC |",
              "|---|---|---|---|---|---|"]
    for j, name in enumerate(BIN_NAMES):
        y, pr = yb[:, j], p["p"][:, j]
        hat = pr > 0.5
        tp = (hat & y).sum()
        prec = tp / max(1, hat.sum())
        rec = tp / max(1, y.sum())
        lines.append(f"| {name} | {y.mean():.4f} | {(hat == y).mean():.4f} | {prec:.3f} | {rec:.3f} | {auc(pr, y):.3f} |")
    st_acc = (p["state"].argmax(1) == d["ys"][idx]).mean()
    rmae = np.abs(p["reward"] - d["yr"][idx]).mean() * 5
    lines += ["", f"Next movement state: {100 * st_acc:.1f}% right.  Reward of the step: mean error {rmae:.3f}.", ""]
    return "\n".join(lines)


def load_demos(path: str, data: list) -> dict:
    """recorded.json -> {demo_key(folder): {shard index: [(agent, first frame, last frame)]}} for the folders in
    data (demos in other folders are reported and skipped)."""
    import json
    with open(path, encoding="utf-8") as f:
        recs = json.load(f)
    keys = {demo_key(d) for d in data}
    out, skipped, excl = {}, 0, 0
    for r in recs:
        if user_excluded(r.get("level")):
            excl += 1
            continue
        k = demo_key(r["wm_data"])
        if k not in keys:
            skipped += 1
            continue
        names = [os.path.basename(p) for p in shard_files(k)]
        if r["shard"] not in names:
            skipped += 1
            continue
        out.setdefault(k, {}).setdefault(names.index(r["shard"]), []).append(
            (int(r["agent"]), int(r["frames"][0]), int(r["frames"][1])))
    print(f"demos: {len(recs) - skipped - excl} of {len(recs)} in the given folders" +
          (f" ({skipped} skipped: folder not given or shard missing)" if skipped else "") +
          (f" ({excl} user-excluded)" if excl else ""), flush=True)
    return out


def drop_excluded(data: list) -> list:
    """The data folders without the replay folders of user-excluded demos (plan.demos.USER_EXCLUDED)."""
    keep = []
    for d in data:
        why = excluded_demo_folder(d)
        if why:
            print(f"skipped {d}: {why}", flush=True)
        else:
            keep.append(d)
    return keep


def replay_demos(data: list) -> dict:
    """The "demo" agents (meta agent_sources) of every shard in replay folders (agent.py --demo), in load_demos'
    form: {demo_key(folder): {shard index: [(agent, 0, last frame)]}}."""
    import json
    out = {}
    for d in data:
        if excluded_demo_folder(d) or not os.path.basename(os.path.dirname(os.path.normpath(d))).startswith("demo_"):
            continue
        for k, p in enumerate(shard_files(d)):
            z = np.load(p)
            meta = json.loads(str(z["meta"]))
            if meta.get("source") not in ("demo", "demo+demo-noisy"):
                continue
            for ag in (meta.get("agent_sources") or {}).get("demo", [0]):
                out.setdefault(demo_key(d), {}).setdefault(k, []).append((int(ag), 0, 10 ** 9))
    return out


def all_demos(paths, data: list) -> dict:
    """The demos of recorded.json-form files (load_demos; one path or a list: wm.learner adds the agent's own
    runs/demos/self.json) plus the replay folders' demo agents (replay_demos)."""
    out = {}
    for path in ([paths] if isinstance(paths, str) else list(paths or [])):
        for k, sh in load_demos(path, data).items():
            for i, v in sh.items():
                out.setdefault(k, {}).setdefault(i, []).extend(v)
    rep = replay_demos(data)
    for k, sh in rep.items():
        for i, v in sh.items():
            out.setdefault(k, {}).setdefault(i, []).extend(v)
    if rep:
        print(f"demos: {sum(len(v) for v in rep.values())} replay shards in {len(rep)} demo folders", flush=True)
    return out


def demo_oversample(tr_idx: np.ndarray, demo: np.ndarray, share: float) -> tuple[np.ndarray, float]:
    """Training indices with the demo ones repeated so they make about `share` of them, and the real share."""
    d = tr_idx[demo[tr_idx]]
    if not len(d) or share <= 0:
        return tr_idx, len(d) / max(1, len(tr_idx))
    rest = len(tr_idx) - len(d)
    reps = max(1, int(round(share * rest / ((1 - share) * len(d))))) if share < 1 else 1
    idx = np.concatenate([tr_idx[~demo[tr_idx]], np.tile(d, reps)])
    return idx, reps * len(d) / len(idx)


def fit(model, ts: dict, d: dict, va_idx, epochs: int, lr: float, batch: int, t0: float):
    """Train on the arrays in ts; held-out loss on d[va_idx] after each epoch."""
    n = len(ts["X"])
    batch = max(1, min(batch, n))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * max(1, n // batch), pct_start=0.1)
    dev = next(model.parameters()).device
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
    W = lambda src, b: T(src["wc"][b]) if "wc" in src else None     # step-k: cut chains' NLL weight 0
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        perm = rng.permutation(n)
        acc = []
        for i in range(0, n - batch + 1, batch):
            b = perm[i:i + batch]
            loss, parts = model.loss(T(ts["X"][b]), T(ts["yc"][b]), T(ts["yb"][b]), T(ts["ys"][b], torch.long), T(ts["yr"][b]),
                                     W(ts, b))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            acc.append([loss.item()] + list(parts.values()))
        a = np.mean(acc, 0)
        with torch.no_grad():
            vb = va_idx[:20000]
            vl, _ = model.loss(T(d["X"][vb]), T(d["yc"][vb]), T(d["yb"][vb]), T(d["ys"][vb], torch.long), T(d["yr"][vb]),
                               W(d, vb))
        print(f"epoch {ep + 1}/{epochs}: train loss {a[0]:.3f} (nll {a[1]:.3f}, bce {a[2]:.3f}, ce {a[3]:.3f}, "
              f"mse {a[4]:.3f}), held-out loss {vl.item():.3f}  [{time.time() - t0:.0f}s]", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="*", help="wm_data folders (manifest sources when omitted)")
    from wm.dataset import add_cli, cli_dataset, sample_validation, load_model, LEGACY
    add_cli(ap)
    ap.add_argument("--out", default=None, help="output folder (default runs/wm_<time>)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=256, help="units per hidden layer")
    ap.add_argument("--layers", type=int, default=3, help="hidden layers")
    ap.add_argument("--seed", type=int, default=0, help="weights and sampling seed (same seed: comparable runs)")
    ap.add_argument("--max-samples", type=int, default=500000, help="transitions used (all deaths / finishes kept)")
    ap.add_argument("--sample-frac", type=float, default=None,
                    help="keep this share of each wave's transitions before building inputs (long runs)")
    ap.add_argument("--levels", nargs="*", default=None, help="only these levels")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (leave some for the game)")
    ap.add_argument("--step", type=int, default=1,
                    help="frames per model step (default 1; 4: one step predicts 4 frames from the 4 actions)")
    ap.add_argument("--demos", default=None, help="recorded.json of demo trajectories (plan.demos recorded)")
    ap.add_argument("--demo-share", type=float, default=0.2, help="demo rows' share of the training samples")
    ap.add_argument("--init", default=None, help="start from this wm_model.pt (then --epochs is not used)")
    ap.add_argument("--self-correct", type=int, default=0, metavar="K",
                    help="then train on the model's own K-step rollouts (wm/selfcorrect.py)")
    ap.add_argument("--sc-starts", type=int, default=15, help="rollout start frames per wave for --self-correct")
    ap.add_argument("--sc-epochs", type=int, default=4)
    ap.add_argument("--sc-share", type=float, default=0.35,
                    help="self-correction samples as a share of the real ones (more: better far ahead, worse next step)")
    from wm.protocol import (FEATURE_PROTOCOL, INPUT_PROTOCOLS, PRIOR_PROTOCOL, PRIOR_PROTOCOLS,
                             UNTRAINED_FEATURE_PROTOCOL, feature_usage)
    ap.add_argument("--feature-protocol", choices=INPUT_PROTOCOLS, default=None,
                    help=f"input encoding (wm/protocol.py). Fresh training: default {FEATURE_PROTOCOL}; "
                         "tile-shapes-v1 adds the open side of half blocks / slopes (explicit opt-in). --init: the "
                         "parent's; a different value is an error")
    ap.add_argument("--prior-protocol", choices=PRIOR_PROTOCOLS, default=None,
                    help=f"physics prior the targets correct (wm/protocol.py). Fresh training: default {PRIOR_PROTOCOL}; "
                         "gravity-fan-wind-v1 adds the measured fan wind force (explicit opt-in). --init: the parent's; "
                         "a different value is an error")
    ap.add_argument("--save-features", default=None, metavar="NPZ",
                    help="build the training features from --manifest (all the other data options apply), write them "
                         "with the split and their provenance to this .npz and stop: train elsewhere (a GPU) with "
                         "--load-features")
    ap.add_argument("--load-features", default=None, metavar="NPZ",
                    help="train from a --save-features file instead of the recordings (no --manifest, no data "
                         "folders needed: the file holds the features, the split and the manifest's reference)")
    ap.add_argument("--device", default="cpu", help="torch device for the fit: cpu (default) or cuda")
    if "--load-features" in sys.argv:              # the recordings stay home: no manifest to open
        sys.argv += ["--legacy-split"] if "--legacy-split" not in sys.argv and "--manifest" not in sys.argv else []
    args = ap.parse_args()
    if args.load_features:
        return train_from_features(args)
    dataset = cli_dataset(args)
    if dataset is not None:
        args.data = dataset.source_list(args.data)
        if args.self_correct and args.step != 1:
            raise ValueError("strict self-correction supports one-frame models; step-k targets are not implemented")
    elif not args.data:
        ap.error("--legacy-split needs data folders")
    args.data = drop_excluded(args.data) if dataset is None else args.data
    from wm.features import model_layout, model_prior
    parent, model = None, None
    parent_feature_protocol = None
    protocol = args.feature_protocol or FEATURE_PROTOCOL
    prior_protocol = args.prior_protocol or PRIOR_PROTOCOL
    if args.init:
        model, parent = load_model(args.init)
        parent_feature_protocol = feature_usage(model.info)["checkpoint_trained"]
        protocol = model_layout(model).protocol
        if args.feature_protocol and args.feature_protocol != protocol:
            raise ValueError(f"--feature-protocol {args.feature_protocol} differs from the parent checkpoint's "
                             f"{protocol} (its weights expect that encoding)")
        prior_protocol = model_prior(model)
        if args.prior_protocol and args.prior_protocol != prior_protocol:
            raise ValueError(f"--prior-protocol {args.prior_protocol} differs from the parent checkpoint's "
                             f"{prior_protocol} (its outputs correct that prior)")
        if dataset is not None:
            dataset.require_model(model)
            if model.step != args.step:
                raise ValueError("--step must match the strict parent checkpoint's step")
        elif (model.info or {}).get("dataset", {}).get("protocol") != LEGACY and (model.info or {}).get("dataset"):
            raise ValueError("strict checkpoint cannot use --legacy-split")
    immediate()
    torch.set_num_threads(args.threads)
    out = args.out or os.path.join(HERE, "runs", "wm_" + time.strftime("%Y%m%d-%H%M%S"))
    if dataset is not None and os.path.exists(out) and os.listdir(out):
        raise SystemExit("strict training needs a new empty output folder; old checkpoints are never overwritten")
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    held = eval_levels()
    d = build(args.data, levels=args.levels, max_samples=args.max_samples, log=lambda *a: None, seed=args.seed,
              frac=args.sample_frac, exclude=held, step=args.step,
              demos=all_demos(args.demos, args.data) if args.demos else None, dataset=dataset, protocol=protocol,
              prior_protocol=prior_protocol)
    val = sample_validation(d, strict=dataset is not None)
    if args.save_features:
        save_features(args.save_features, d, val, args, dataset, protocol, prior_protocol, held)
        print(f"saved {len(d['X'])} transitions' features to {args.save_features}  [{time.time() - t0:.0f}s]")
        return
    res_scale = RES_SCALE
    if args.step > 1:                        # targets in unit scale: the corrections' spread over full chains
        full = d["full"]
        corr = (d["raw"] - d["prior"])[full & ~val]
        if dataset is not None and not len(corr):
            raise ValueError("manifest has no full training chains for residual scaling")
        res_scale = model.res_scale if model is not None else np.maximum(corr.std(0), 1e-3).astype(np.float32)
        d["yc"] = (d["yc"] * RES_SCALE / res_scale).astype(np.float32)
        d["wc"] = full.astype(np.float32)
        print(f"step {args.step}: res_scale {np.round(res_scale, 2).tolist()}, {int((~full).sum())} chains cut "
              f"by a death / finish", flush=True)
    n = len(d["X"])
    tr_idx, va_idx = np.nonzero(~val)[0], np.nonzero(val)[0]
    args.batch = max(1, min(args.batch, len(tr_idx)))
    print(f"{n} transitions ({len(tr_idx)} train / {len(va_idx)} held out) from "
          f"{len(np.unique(d['shard']))} waves, levels {', '.join(sorted(set(d['level'])))}  [{time.time() - t0:.0f}s]",
          flush=True)

    if args.init:                            # continue from a trained model
        print(f"starting from {args.init}")
    else:
        torch.manual_seed(args.seed)
        model = WorldModel(d["X"].shape[1], members=args.members, hidden=args.hidden, layers=args.layers,
                           step=args.step, res_scale=res_scale)
        # the encoding its inputs use, before any fit (self-correction rebuilds inputs from it)
        model.info = {"feature_protocol": UNTRAINED_FEATURE_PROTOCOL, "active_feature_protocol": protocol,
                      "prior_protocol": prior_protocol}
        model.set_norm(torch.as_tensor(d["X"][tr_idx[:200000]], dtype=torch.float32))
        pos = d["yb"][tr_idx].mean(0)
        pw = np.where(np.isin(np.arange(len(BIN_NAMES)), list(EVENTS.values())),
                      np.clip((1 - pos) / np.maximum(pos, 1e-6), 1, 50), 1)
        model.pos_weight.copy_(torch.as_tensor(pw, dtype=torch.float32))
    fit_idx = tr_idx
    if args.demos:
        fit_idx, real = demo_oversample(tr_idx, d["demo"], args.demo_share)
        print(f"demos: {int(d['demo'][tr_idx].sum())} training / {int(d['demo'][va_idx].sum())} held-out demo "
              f"transitions; oversampled to {real:.1%} of {len(fit_idx)} training samples "
              f"(asked {args.demo_share:.0%})", flush=True)
    train_set = {k: d[k][fit_idx] for k in ("X", "yc", "yb", "ys", "yr") + (("wc",) if "wc" in d else ())}
    base_fit_samples, sc_samples, sc_epochs = (len(fit_idx) if not args.init else 0), 0, 0
    if not args.init:
        model.to(args.device)
        fit(model, train_set, d, va_idx, args.epochs, 1e-3, args.batch, t0)
        model.to("cpu")
    if dataset is not None:
        model.info = dict(getattr(model, "info", {}) or {}, dataset=dataset.reference(),
                          feature_protocol=parent_feature_protocol if args.init else protocol,
                          active_feature_protocol=protocol)
    if args.self_correct:                    # then on its own rollouts (wm/selfcorrect.py)
        from wm.selfcorrect import generate
        sc = generate(args.data, model, K=args.self_correct, starts=args.sc_starts, log=lambda *a: None,
                      dataset=dataset)
        print(f"self-correction: {len(sc.get('X', []))} samples from the model's own {args.self_correct}-step "
              f"rollouts  [{time.time() - t0:.0f}s]", flush=True)
        if sc and args.sc_share > 0 and len(sc["X"]) > args.sc_share * len(train_set["X"]):
            pick = np.random.default_rng(2).choice(len(sc["X"]), int(args.sc_share * len(train_set["X"])), replace=False)
            sc = {k: v[pick] for k, v in sc.items()}
            print(f"self-correction: using {len(pick)} of them ({args.sc_share:.0%} of the real samples)", flush=True)
        if sc:
            sc_samples, sc_epochs = len(sc["X"]), args.sc_epochs
            train_set = {k: np.concatenate([train_set[k], sc[k]]) for k in train_set}
            fit(model, train_set, d, va_idx, args.sc_epochs, 3e-4, args.batch, t0)

    info = {"data": args.data, "levels": sorted(set(d["level"])), "train": int(len(tr_idx)), "held_out": int(len(va_idx)),
            "eval_levels_excluded": sorted(held), "sample_frac": args.sample_frac,
            "demos": args.demos, "demo_share": args.demo_share if args.demos else None,
            "hidden": model.hidden, "layers": model.layers, "step": model.step,
            "dataset": dataset.reference() if dataset is not None else {"protocol": LEGACY},
            "parent_checkpoint": parent,
            "active_feature_protocol": protocol,
            "parent_feature_protocol": parent_feature_protocol,
            "prior_protocol": prior_protocol,
            "feature_protocol": protocol if (base_fit_samples and args.epochs > 0) or
                                (sc_samples and sc_epochs > 0) else
                                (parent_feature_protocol or UNTRAINED_FEATURE_PROTOCOL),
            "training": {"epochs": 0 if args.init else args.epochs, "seed": args.seed,
                         "max_samples": args.max_samples, "sample_frac": args.sample_frac, "batch": args.batch,
                         "train_transitions": len(tr_idx), "validation_transitions": len(va_idx),
                         "base_fit_samples": base_fit_samples,
                         "fit_samples": len(train_set["X"]) if sc_samples else base_fit_samples,
                         "demo_samples": int(d["demo"][fit_idx].sum()), "actual_sc_samples": sc_samples,
                         "self_correction": args.self_correct,
                         "self_correction_epochs": sc_epochs}}
    model.save(os.path.join(out, "wm_model.pt"), info)
    text = "# World model: one-step accuracy\n\n" + \
        f"Features active: {protocol}; checkpoint trained: {info['feature_protocol']}; prior: {prior_protocol}.\n\n" + \
        f"Dataset protocol: {info['dataset']['protocol']}; manifest {info['dataset'].get('sha256', 'none (legacy)')}.\n\n" + \
        f"Data: {', '.join(args.data)}; levels {', '.join(info['levels'])}.\n\n" + report(model, d, va_idx)
    with open(os.path.join(out, "wm_report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"saved {out}")


FEATURE_KEYS_SKIP = ()


def save_features(path: str, d: dict, val, args, dataset, protocol: str, prior_protocol: str, held):
    """--save-features: every array of build()'s result, the validation split and where they came from."""
    from wm.dataset import LEGACY
    arrays = {f"d_{k}": v for k, v in d.items() if isinstance(v, np.ndarray) and v.dtype != object}
    meta = {"dataset": dataset.reference() if dataset is not None else {"protocol": LEGACY},
            "data": list(args.data), "protocol": protocol, "prior_protocol": prior_protocol, "step": args.step,
            "sample_frac": args.sample_frac, "max_samples": args.max_samples, "seed": args.seed,
            "demos": args.demos, "held": sorted(held),
            "keys": sorted(k for k, v in d.items() if isinstance(v, np.ndarray) and v.dtype != object)}
    np.savez_compressed(path, val=np.asarray(val, bool), meta=json.dumps(meta), **arrays)


def train_from_features(args):
    """--load-features: the rest of main() on the saved arrays (fresh weights; the saved split and settings)."""
    from wm.protocol import UNTRAINED_FEATURE_PROTOCOL
    t0 = time.time()
    z = np.load(args.load_features, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    d = {k: z[f"d_{k}"] for k in meta["keys"]}
    val = z["val"]
    out = args.out or os.path.join(HERE, "runs", "wm_" + time.strftime("%Y%m%d-%H%M%S"))
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit("training needs a new empty output folder; old checkpoints are never overwritten")
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(args.threads)
    step = int(meta["step"])
    res_scale = RES_SCALE
    if step > 1:
        full = d["full"]
        corr = (d["raw"] - d["prior"])[full & ~val]
        res_scale = np.maximum(corr.std(0), 1e-3).astype(np.float32)
        d["yc"] = (d["yc"] * RES_SCALE / res_scale).astype(np.float32)
        d["wc"] = full.astype(np.float32)
    tr_idx, va_idx = np.nonzero(~val)[0], np.nonzero(val)[0]
    batch = max(1, min(args.batch, len(tr_idx)))
    print(f"{len(d['X'])} transitions ({len(tr_idx)} train / {len(va_idx)} held out) from {args.load_features}, "
          f"device {args.device}", flush=True)
    torch.manual_seed(args.seed)
    model = WorldModel(d["X"].shape[1], members=args.members, hidden=args.hidden, layers=args.layers, step=step,
                       res_scale=res_scale)
    model.info = {"feature_protocol": UNTRAINED_FEATURE_PROTOCOL, "active_feature_protocol": meta["protocol"],
                  "prior_protocol": meta["prior_protocol"]}
    model.set_norm(torch.as_tensor(d["X"][tr_idx[:200000]], dtype=torch.float32))
    pos = d["yb"][tr_idx].mean(0)
    pw = np.where(np.isin(np.arange(len(BIN_NAMES)), list(EVENTS.values())),
                  np.clip((1 - pos) / np.maximum(pos, 1e-6), 1, 50), 1)
    model.pos_weight.copy_(torch.as_tensor(pw, dtype=torch.float32))
    fit_idx = tr_idx
    if meta.get("demos") and "demo" in d:
        fit_idx, real = demo_oversample(tr_idx, d["demo"], args.demo_share)
        print(f"demos oversampled to {real:.1%} of {len(fit_idx)} training samples", flush=True)
    train_set = {k: d[k][fit_idx] for k in ("X", "yc", "yb", "ys", "yr") + (("wc",) if "wc" in d else ())}
    model.to(args.device)
    fit(model, train_set, d, va_idx, args.epochs, 1e-3, batch, t0)
    model.to("cpu")
    info = {"data": meta["data"], "levels": sorted(set(d["level"].tolist())), "train": int(len(tr_idx)),
            "held_out": int(len(va_idx)), "eval_levels_excluded": meta["held"], "sample_frac": meta["sample_frac"],
            "demos": meta["demos"], "demo_share": args.demo_share if meta.get("demos") else None,
            "hidden": model.hidden, "layers": model.layers, "step": model.step, "dataset": meta["dataset"],
            "parent_checkpoint": None, "active_feature_protocol": meta["protocol"], "parent_feature_protocol": None,
            "prior_protocol": meta["prior_protocol"], "feature_protocol": meta["protocol"],
            "training": {"epochs": args.epochs, "seed": args.seed, "max_samples": meta["max_samples"],
                         "sample_frac": meta["sample_frac"], "batch": batch, "train_transitions": len(tr_idx),
                         "validation_transitions": len(va_idx), "base_fit_samples": len(fit_idx),
                         "fit_samples": len(fit_idx), "demo_samples": int(d["demo"][fit_idx].sum()) if "demo" in d else 0,
                         "actual_sc_samples": 0, "self_correction": 0, "self_correction_epochs": 0,
                         "features_file": os.path.basename(args.load_features), "device": args.device}}
    model.save(os.path.join(out, "wm_model.pt"), info)
    text = "# World model: one-step accuracy\n\n" + \
        f"Features: {meta['protocol']}; prior: {meta['prior_protocol']}; trained from {args.load_features} on " \
        f"{args.device}.\n\nDataset {meta['dataset'].get('protocol')}; manifest {meta['dataset'].get('sha256', 'none')}.\n\n" + \
        report(model, d, va_idx)
    with open(os.path.join(out, "wm_report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
