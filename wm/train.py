"""Train the world model on recorded gameplay and report its one-step accuracy.

    python -m wm.train runs/<run>/wm_data [more wm_data folders] [--out runs/wm_<name>] [--epochs 6]

Runs in its own process (it does not touch the game or the PPO trainer); --threads limits its CPU use so a
training run in the game next to it keeps its speed. Every 5th recorded wave is held out and only used for
the report: position / velocity errors against two simple guesses (same velocity; same velocity with
gravity), per situation (ground / air / wall), how sure the model is versus how wrong it is, and the events
(landing, death, finish, blocked, flags, movement state). Writes wm_model.pt and wm_report.md to --out.

--demos runs/demos/recorded.json (plan.demos recorded) --demo-share 0.2: the PPO agents' best finishes found in
the given wm_data folders are oversampled to about that share of the training samples (the real share is
printed); demo rows of held-out waves stay held out, held-out levels stay out.
"""
from __future__ import annotations

import argparse
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
    for i in range(0, len(X), bs):
        p = model.predict(torch.as_tensor(X[i:i + bs], dtype=torch.float32))
        for k, v in p.items():
            outs.setdefault(k, []).append(v.numpy())
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
    out, skipped = {}, 0
    for r in recs:
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
    print(f"demos: {len(recs) - skipped} of {len(recs)} in the given folders" +
          (f" ({skipped} skipped: folder not given or shard missing)" if skipped else ""), flush=True)
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
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
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
    ap.add_argument("data", nargs="+", help="wm_data folders")
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
    args = ap.parse_args()
    immediate()
    torch.set_num_threads(args.threads)
    out = args.out or os.path.join(HERE, "runs", "wm_" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    held = eval_levels()
    d = build(args.data, levels=args.levels, max_samples=args.max_samples, log=lambda *a: None, seed=args.seed,
              frac=args.sample_frac, exclude=held, step=args.step,
              demos=load_demos(args.demos, args.data) if args.demos else None)
    res_scale = RES_SCALE
    if args.step > 1:                        # targets in unit scale: the corrections' spread over full chains
        full = d["full"]
        corr = (d["raw"] - d["prior"])[full & (d["shard"] % 5 != 0)]
        res_scale = np.maximum(corr.std(0), 1e-3).astype(np.float32)
        d["yc"] = (d["yc"] * RES_SCALE / res_scale).astype(np.float32)
        d["wc"] = full.astype(np.float32)
        print(f"step {args.step}: res_scale {np.round(res_scale, 2).tolist()}, {int((~full).sum())} chains cut "
              f"by a death / finish", flush=True)
    n = len(d["X"])
    val = d["shard"] % 5 == 0
    if not val.any() or val.all():           # too few waves: hold out the last one
        val = d["shard"] == d["shard"].max()
    tr_idx, va_idx = np.nonzero(~val)[0], np.nonzero(val)[0]
    args.batch = max(1, min(args.batch, len(tr_idx)))
    print(f"{n} transitions ({len(tr_idx)} train / {len(va_idx)} held out) from "
          f"{len(np.unique(d['shard']))} waves, levels {', '.join(sorted(set(d['level'])))}  [{time.time() - t0:.0f}s]",
          flush=True)

    if args.init:                            # continue from a trained model
        model = WorldModel.load(args.init)
        print(f"starting from {args.init}")
    else:
        torch.manual_seed(args.seed)
        model = WorldModel(d["X"].shape[1], members=args.members, hidden=args.hidden, layers=args.layers,
                           step=args.step, res_scale=res_scale)
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
    if not args.init:
        fit(model, train_set, d, va_idx, args.epochs, 1e-3, args.batch, t0)
    if args.self_correct:                    # then on its own rollouts (wm/selfcorrect.py)
        from wm.selfcorrect import generate
        sc = generate(args.data, model, K=args.self_correct, starts=args.sc_starts, log=lambda *a: None)
        print(f"self-correction: {len(sc.get('X', []))} samples from the model's own {args.self_correct}-step "
              f"rollouts  [{time.time() - t0:.0f}s]", flush=True)
        if sc and args.sc_share > 0 and len(sc["X"]) > args.sc_share * len(train_set["X"]):
            pick = np.random.default_rng(2).choice(len(sc["X"]), int(args.sc_share * len(train_set["X"])), replace=False)
            sc = {k: v[pick] for k, v in sc.items()}
            print(f"self-correction: using {len(pick)} of them ({args.sc_share:.0%} of the real samples)", flush=True)
        if sc:
            train_set = {k: np.concatenate([train_set[k], sc[k]]) for k in train_set}
            fit(model, train_set, d, va_idx, args.sc_epochs, 3e-4, args.batch, t0)

    info = {"data": args.data, "levels": sorted(set(d["level"])), "train": int(len(tr_idx)), "held_out": int(len(va_idx)),
            "eval_levels_excluded": sorted(held), "sample_frac": args.sample_frac,
            "demos": args.demos, "demo_share": args.demo_share if args.demos else None,
            "hidden": model.hidden, "layers": model.layers, "step": model.step}
    model.save(os.path.join(out, "wm_model.pt"), info)
    text = "# World model: one-step accuracy\n\n" + \
        f"Data: {', '.join(args.data)}; levels {', '.join(info['levels'])}.\n\n" + report(model, d, va_idx)
    with open(os.path.join(out, "wm_report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
