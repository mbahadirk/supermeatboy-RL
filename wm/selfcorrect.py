"""Training samples from the model's own rollouts, so its errors do not pile up over many steps.

From waves it trains on (never the held-out ones), the model rolls out K steps with the agents' real actions,
time steps and hazards (rollout_mean). At each step it is given the state it predicted, and the target is
where the agent really was one step later: the model learns to steer back toward reality from its own slightly
wrong states (Venkatraman et al. 2015, "data as demonstrator"). Only predicted states within MAX_OFF of the
truth are used, so the corrections stay physically plausible.
"""
from __future__ import annotations

import numpy as np

from wm.buffer import load_shards, load_level, eval_levels
from wm.features import LevelGeom, model_layout, model_prior, targets, prior, group_of
from wm.rollout import rollout_mean, World

MAX_OFF = 20.0                                   # game units: one tile


def generate(dirs, model, K: int = 8, starts: int = 15, seed: int = 1, log=print, dataset=None) -> dict:
    rng = np.random.default_rng(seed)
    X, YC, YB, YS, YR, G = [], [], [], [], [], []
    shard_no = 0
    held = eval_levels()
    if dataset is not None:
        dataset.require_model(model)
        if model.step != 1:
            raise ValueError("self-correction supports one-frame models; step-k target generation is not implemented")
        dirs = dataset.source_list(dirs)
    elif (getattr(model, "info", {}) or {}).get("dataset", {}).get("protocol") == "trace-split-v1":
        raise ValueError("strict self-correction requires the checkpoint's manifest")
    protocol = model_layout(model).protocol          # the model's own input encoding
    pp = model_prior(model)                          # and its physics prior
    for d in dirs:
        geoms = {}
        for meta, rows, world in (dataset.iter_shards(d, split="train") if dataset is not None else load_shards(d)):
            shard_no += 1
            if dataset is None and shard_no % 5 == 0:  # legacy only
                continue
            lv = meta.get("level")
            if lv in held:                       # held-out evaluation level: never trained on
                continue
            if lv not in geoms:
                geoms[lv] = LevelGeom(*(dataset.level(d, lv) if dataset is not None else load_level(d, lv)),
                                      protocol=protocol)
            geom = geoms[lv]
            geom.reset()
            wf = {w["f"]: w for w in world}
            frames = sorted(wf)
            if len(frames) < K + 2:
                continue
            by = {(int(r["agent"]), int(r["frame"])): r for r in rows}
            med = float(np.median([wf[f].get("dt") or 33.0 for f in frames]))
            keep, fans, upto = None, [], -1
            for f0 in sorted(rng.choice(frames[1:-K], min(starts, len(frames) - K - 1), replace=False)):
                for ff in range(upto + 1, int(f0) + 1):
                    rec = wf.get(ff, {})
                    for idx, word in rec.get("tc", []) or []:
                        geom.set_tile(idx, word)
                    keep = rec.get("keep", keep)
                    if "fx" in rec:
                        fans = rec["fx"].get("fans", [])
                upto = int(f0)
                agents = [a for (a, f) in by if f == f0 and by[(a, f)]["action"] >= 0 and by[(a, f)]["status"] == 0]
                agents = [a for a in agents if all((a, f0 + k) in by for k in range(K + 1))]
                if not agents:
                    continue
                truth = [np.array([by[(a, f0 + k)] for a in agents]) for k in range(K + 1)]
                acts = np.stack([np.maximum(0, truth[k]["action"]) for k in range(K)], 1).astype(np.int64)
                dts = np.array([wf.get(f0 + k, {}).get("dt") or med for k in range(K)])
                worlds = [World.from_record(wf.get(f0 + k, wf[frames[-1]]), wf.get(f0 + k - 1), keep, fans)
                          for k in range(K)]
                feats, pred = rollout_mean(model, geom, truth[0], acts, dts, worlds)
                for k in range(1, K):             # step 0 is a real state: those samples exist already
                    p, t_now, t_next = pred[k], truth[k], truth[k + 1]
                    ok = (np.hypot(p["x"] - t_now["x"], p["y"] - t_now["y"]) < MAX_OFF) & (t_now["status"] == 0)
                    if not ok.any():
                        continue
                    pri = prior(p[ok], dts[k], 1, worlds[k].fans, pp)
                    yc, yb, ys, yr = targets(p[ok], t_next[ok], t_next["reward"][ok], pri)
                    X.append(feats[k][ok].astype(np.float16))
                    YC.append(yc); YB.append(yb); YS.append(ys); YR.append(yr); G.append(group_of(p[ok]))
            log(f"[self-correct] wave {shard_no} ({lv}): {sum(len(x) for x in X)} samples so far")
    if not X:
        return {}
    return {"X": np.concatenate(X), "yc": np.concatenate(YC), "yb": np.concatenate(YB), "ys": np.concatenate(YS),
            "yr": np.concatenate(YR), "group": np.concatenate(G)}
