"""Run the planner on recorded states (no game): what it picks, why, and what each decision costs.

    python -m mpc.offline <run>/wm_data --level 1-13 [--policy <run>/model.pt] [--decisions 20]
                          [--candidates 32] [--horizon 16] [--model runs/wm_locked/wm_model.pt]

Starts from agents' real states of one recorded wave; the hazards' future is the wave's own recording
(as the planner would have it from an earlier wave). Routes are built as the trainer builds them.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import realgame as rg                                    # noqa: E402
from realpath import Planner                             # noqa: E402
from mpc.controller import MPC                           # noqa: E402
from mpc.score import Weights                            # noqa: E402
from wm.buffer import load_level                         # noqa: E402


def default_model() -> str:
    for name in ("wm_locked", "wm_phase4mix"):
        p = os.path.join(HERE, "runs", name, "wm_model.pt")
        if os.path.exists(p):
            return p
    raise SystemExit("no world model found under runs/")


def level_from_data(d: str, name: str):
    words, meta = load_level(d, name)
    info = dict(meta)
    info["name"] = name
    return rg.Level(info, words.astype("<u4").tobytes())


def waves_of(d: str, level: str):
    for p in sorted(glob.glob(os.path.join(d, "wave_*.npz"))):
        z = np.load(p)
        if json.loads(str(z["meta"])).get("level") == level:
            yield z["rows"].copy(), json.load(gzip.open(p[:-4] + "_world.json.gz", "rt", encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--level", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--policy", default=None, help="PPO model.pt: its first actions and critic value are used")
    ap.add_argument("--decisions", type=int, default=20)
    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--value", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    lv = level_from_data(args.data, args.level)
    gen = waves_of(args.data, args.level)
    ref_rows, ref_world = next(gen)                       # an earlier wave: the hazards' recording
    rows, world = next(gen, (ref_rows, ref_world))
    static, _ = rg.hazard_boxes(world[0].get("hz", []), lv.hz_keep)
    routes = Planner(lv.words0, lv.ox, lv.oy, lv.spawn, lv.goals, static, [], lv.keys, lv.bandages,
                     list(lv.portals)).find(1)
    route = routes[0] if routes else None
    print(f"{args.level}: route {route.length / rg.TILE:.0f} tiles" if route else f"{args.level}: no route")
    policy = None
    if args.policy:
        policy = rg.Policy()
        rg.load_policy(policy, args.policy)
    mpc = MPC(args.model or default_model(), args.candidates, args.horizon, weights=Weights(value=args.value))
    print(f"model {mpc.model_path}")
    mpc.set_level(lv)
    keep = lv.hz_keep
    as_msg = lambda w: {**w, "frame": w["f"]}            # recorded frames say "f", the game's messages "frame"
    for w in ref_world:                                   # the earlier wave, frame by frame: the recording
        mpc.frame(as_msg(w), keep)
    mpc.new_wave(1)
    wf = {w["f"]: w for w in world}
    frames = sorted(wf)
    picks = np.linspace(5, len(frames) - 1, args.decisions).astype(int)
    by_frame = {}
    for r in rows:
        by_frame.setdefault(int(r["frame"]), []).append(r)
    ms, done = [], 0
    for f in frames:
        mpc.frame(as_msg(wf[f]), keep)
        if f not in set(frames[i] for i in picks):
            continue
        alive = [r for r in by_frame.get(f, []) if r["status"] == 0]
        if not alive:
            continue
        r = alive[0]
        s = [float(r["x"]), float(r["y"]), float(r["vx"]), float(r["vy"]), int(r["flags"]), int(r["state"]), 0,
             int(r["keys"])]
        probs = value_fn = None
        if policy is not None and route is not None:
            boxes, _ = rg.hazard_boxes(wf[f].get("hz", []), lv.hz_keep)
            grid = lv.grid_with_hazards(boxes)

            def obs_of(x, y, vx, vy, flags, state, held):
                prog = route.value(x, y, ()) or 0.0
                return rg.observe(lv, grid, [x, y, vx, vy, flags, state, 0, 0], route, prog, route.offset(x, y),
                                  None, (), (0.0, 0.0), held)
            o = obs_of(*s[:6], int(r["held"]))
            with torch.no_grad():
                probs = policy.dist(torch.as_tensor(o[None])).probs[0].numpy()

            def value_fn(end):
                ob = np.stack([obs_of(float(e["x"]), float(e["y"]), float(e["vx"]), float(e["vy"]), int(e["flags"]),
                                      int(e["state"]), int(e["held"])) for e in end])
                with torch.no_grad():
                    return policy.vf(torch.as_tensor(ob)).squeeze(-1).numpy()
        a, info = mpc.plan(int(r["agent"]), s, int(r["held"]), f, route, (), lv, probs, value_fn)
        ms.append(info["ms"]["total"])
        done += 1
        t = info["top"][0]
        print(f"f{f:5d} ({s[0]:6.0f},{s[1]:6.0f}) v({s[2]:5.0f},{s[3]:5.0f}) -> {rg.ACTIONS[a]} '{info['label']}' "
              f"total {t['total']:+.2f} (progress {t['progress']:+.2f}, death {t['death']:.2f}, land {t['land']:.2f}, "
              f"value {t['value']:+.2f}, unc {t['unc']:.2f}) n={info['n']} steps={info['steps']} "
              f"recorded={info['recorded']} | {info['ms']}", flush=True)
    if ms:
        print(f"{done} decisions: {np.median(ms):.1f} ms median, {np.percentile(ms, 95):.1f} ms 95%, max {max(ms):.1f}")


if __name__ == "__main__":
    main()
