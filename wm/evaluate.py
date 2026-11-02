"""How the world model's predictions hold up over many steps (open loop, the agents' real actions).

    python -m wm.evaluate runs/<run>/wm_data --model runs/wm_<name>/wm_model.pt [--horizon 32] [--starts 40]

From waves the model never trained on (every 5th recorded wave, as wm.train holds out), rollouts start at
--starts frames per wave for every agent alive there and run --horizon steps with the actions the agents
really took. Reported:
  - position / velocity error after 1, 2, 4, 8, 16, 32 steps, with the real time steps ("known dt") and with
    the wave's median step for every step (what a planner has: "median dt")
  - whether the ensemble spread grows with the error (it is the planner's warning sign)
  - deaths within the horizon: predicted death probability vs what happened (AUC, recall)
  - landings: where and when the first landing is predicted vs where it happened
Hazards and platforms are taken at their recorded future positions (a planner has to predict them; see
`repeat` in wm.probes). Writes rollout_report.md next to the model.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from wm.buffer import load_shards, load_level, STATUS_DEAD   # noqa: E402
from wm.features import LevelGeom, AIR                       # noqa: E402
from wm.model import WorldModel                               # noqa: E402
from wm.rollout import rollout, World                         # noqa: E402
from wm.train import auc                                      # noqa: E402

HORIZONS = [1, 2, 4, 8, 16, 32]


def shard_cases(meta, rows, world, geom, model, H, n_starts, rng):
    """Rollouts from several start frames of one wave; per case the truth and both predictions."""
    wf = {w["f"]: w for w in world}
    frames = sorted(wf)
    if len(frames) < H + 2:
        return []
    by = {(int(r["agent"]), int(r["frame"])): r for r in rows}
    dts = np.array([wf[f].get("dt") or 33.0 for f in frames], np.float64)
    med = float(np.median(dts))
    starts = sorted(rng.choice(frames[1:-H], min(n_starts, len(frames) - H - 1), replace=False))
    geom.reset()
    keep, fans, upto, cases = None, [], -1, []
    for f0 in starts:
        for ff in range(upto + 1, f0 + 1):           # the tiles / decoration list / wind as of the start frame
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
        truth = np.full((len(agents), H + 1, 4), np.nan, np.float32)
        status = np.zeros((len(agents), H + 1), np.int8)
        for i, a in enumerate(agents):
            for k in range(H + 1):
                r = by.get((a, f0 + k))
                if r is None:
                    break
                truth[i, k] = (r["x"], r["y"], r["vx"], r["vy"])
                status[i, k] = r["status"]
                air = bool(r["flags"] & AIR)
                if k < H:
                    acts[i, k] = max(0, int(r["action"]))
                if r["status"] != 0:
                    break
        worlds = [World.from_record(wf.get(f0 + k, wf[frames[-1]]), wf.get(f0 + k - 1), keep, fans) for k in range(H)]
        real_dt = np.array([wf.get(f0 + k, {}).get("dt") or med for k in range(H)])
        known = rollout(model, geom, s0, acts, real_dt, worlds)
        median = rollout(model, geom, s0, acts, np.full(H, med), worlds)
        air0 = np.array([bool(r["flags"] & AIR) for r in s0])
        cases.append({"truth": truth, "status": status, "known": known, "median": median, "air0": air0,
                      "true_air": np.array([[bool(by[(a, f0 + k)]["flags"] & AIR) if (a, f0 + k) in by else False
                                             for k in range(H + 1)] for a in agents])})
    return cases


def summarise(cases, H) -> str:
    L = []
    for mode in ("known", "median"):
        L += [f"### Position / velocity error after h steps ({'real time steps' if mode == 'known' else 'the wave median time step for every step'})",
              "", "| h | agents | position median | 75% | 90% | velocity median | 90% | ensemble spread median | spread vs error (rank corr.) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for h in [x for x in HORIZONS if x <= H]:
            pe, ve, sp = [], [], []
            for c in cases:
                t = c["truth"][:, h]
                ok = ~np.isnan(t[:, 0]) & (c["status"][:, h] == 0)
                if not ok.any():
                    continue
                r = c[mode]
                px, py = r["x"][:, ok, h].mean(0), r["y"][:, ok, h].mean(0)
                vx, vy = r["vx"][:, ok, h].mean(0), r["vy"][:, ok, h].mean(0)
                pe.append(np.hypot(px - t[ok, 0], py - t[ok, 1]))
                ve.append(np.hypot(vx - t[ok, 2], vy - t[ok, 3]))
                sp.append(np.hypot(r["x"][:, ok, h].std(0), r["y"][:, ok, h].std(0)))
            if not pe:
                continue
            pe, ve, sp = np.concatenate(pe), np.concatenate(ve), np.concatenate(sp)
            rc = np.corrcoef(np.argsort(np.argsort(sp)), np.argsort(np.argsort(pe)))[0, 1]
            L.append(f"| {h} | {len(pe)} | {np.median(pe):.2f} | {np.percentile(pe, 75):.2f} | {np.percentile(pe, 90):.2f} | "
                     f"{np.median(ve):.1f} | {np.percentile(ve, 90):.1f} | {np.median(sp):.2f} | {rc:.2f} |")
        L.append("")
    # deaths within the horizon
    died, score = [], []
    for c in cases:
        died.append((c["status"] == STATUS_DEAD).any(1))
        score.append(1.0 - c["median"]["alive"][:, :, -1].mean(0))
    died, score = np.concatenate(died), np.concatenate(score)
    L += [f"### Deaths within {H} steps", "",
          f"{int(died.sum())} of {len(died)} rollouts end in a death. Predicted death probability (median time step): "
          f"AUC {auc(score, died):.3f}; with probability > 0.5 counted as 'will die': recall {((score > 0.5) & died).sum() / max(1, died.sum()):.2f}, "
          f"precision {((score > 0.5) & died).sum() / max(1, (score > 0.5).sum()):.2f}.", ""]
    # first landing
    dstep, dpos, n_land, missed = [], [], 0, 0
    for c in cases:
        r = c["median"]
        pa = r["airborne"].mean(0) > 0.5                 # (n, H+1) the members' vote
        for i in np.nonzero(c["air0"])[0]:
            ta = c["true_air"][i]
            tl = next((k for k in range(1, H + 1) if not ta[k] and not np.isnan(c["truth"][i, k, 0])), None)
            if tl is None:
                continue
            n_land += 1
            pl = next((k for k in range(1, H + 1) if not pa[i, k]), None)
            if pl is None:
                missed += 1
                continue
            dstep.append(abs(pl - tl))
            dpos.append(np.hypot(r["x"][:, i, pl].mean() - c["truth"][i, tl, 0], r["y"][:, i, pl].mean() - c["truth"][i, tl, 1]))
    if n_land:
        L += [f"### First landing within {H} steps (agents in the air at the start)", "",
              f"{n_land} landings; predicted no landing in time for {missed}. Of the rest: landing step off by "
              f"median {np.median(dstep):.0f} (90% {np.percentile(dstep, 90):.0f}) steps, landing spot off by median "
              f"{np.median(dpos):.1f} units (90% {np.percentile(dpos, 90):.1f}).", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--starts", type=int, default=40, help="start frames per held-out wave")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    model = WorldModel.load(args.model).eval()
    rng = np.random.default_rng(0)
    cases, shard_no, levels = [], 0, set()
    for d in args.data:
        geoms = {}
        for meta, rows, world in load_shards(d):
            shard_no += 1
            if shard_no % 5:                         # the waves wm.train held out
                continue
            lv = meta.get("level")
            if lv not in geoms:
                geoms[lv] = LevelGeom(*load_level(d, lv))
            c = shard_cases(meta, rows, world, geoms[lv], model, args.horizon, args.starts, rng)
            cases += c
            levels.add(lv)
            print(f"wave {shard_no} ({lv}): {sum(len(x['air0']) for x in c)} rollouts", flush=True)
    text = (f"# World model: multi-step rollouts\n\nModel {args.model}; held-out waves of {', '.join(sorted(levels))}; "
            f"{sum(len(c['air0']) for c in cases)} rollouts of {args.horizon} steps with the agents' real actions. "
            f"Hazards and platforms at their recorded positions.\n\n") + summarise(cases, args.horizon)
    print(text)
    with open(os.path.join(os.path.dirname(args.model), "rollout_report.md"), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
