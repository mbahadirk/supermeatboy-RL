"""Run the beam planner (plan/planner.py) on recorded states, no game: what it picks, why, and what a decision
costs.

    python -m plan.offline <run>/wm_data --level 4-6 [--decisions 20] [--model runs/wm_locked/wm_model.pt]
                           [--budget 120] [--threads 2] [--seed 0] [--json plans.jsonl]

The level comes from the recorded level data (game.level.Level); the progress field V from the level's first
recorded wave (its pieces that stay put, the sweeps of the moving ones, fans, platforms, sealed pockets, as
plan/measure_progress.py builds it); the hazards' future from that same earlier wave (mpc.future.FutureWorld's
recording), while the decisions start at agents' real states of the next recorded wave. No PPO anywhere.
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

from mpc.future import FutureWorld                                        # noqa: E402
from plan.measure_progress import level_from_data, planner_inputs          # noqa: E402
from plan.planner import Planner, future_inputs, state_row                 # noqa: E402
from plan.progress import ProgressField                                    # noqa: E402

DEFAULT_MODEL = os.path.join(HERE, "runs", "wm_locked", "wm_model.pt")


def waves_of(d: str, level: str):
    for p in sorted(glob.glob(os.path.join(d, "wave_*.npz"))):
        z = np.load(p)
        if json.loads(str(z["meta"])).get("level") == level:
            yield z["rows"].copy(), json.load(gzip.open(p[:-4] + "_world.json.gz", "rt", encoding="utf-8"))


def build(d: str, level: str, model: str, budget: float):
    """(Level, ProgressField, Planner, reference wave world, current wave rows, current wave world)."""
    gen = waves_of(d, level)
    ref_rows, ref_world = next(gen)
    rows, world = next(gen, (ref_rows, ref_world))
    lv = level_from_data(d, level)
    inp = planner_inputs(level_from_data(d, level), ref_world)
    field = ProgressField(lv.words0, lv.ox, lv.oy, lv.spawn, lv.goals, inp["hazards"], inp["sweeps"], inp["keys"],
                          inp["bandages"], inp["portals"], lift=inp["lift"], blocks=inp["blocks"], doors=inp["doors"])
    planner = Planner(model, lv, field, budget_ms=budget)
    return lv, field, planner, ref_world, rows, world


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--level", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--decisions", type=int, default=20)
    ap.add_argument("--budget", type=float, default=120.0, help="ms per decision (the cheap stages' deadline)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="write every Plan.to_dict() here (one JSON line each)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    lv, field, planner, ref_world, rows, world = build(args.data, args.level, args.model, args.budget)
    print(f"{args.level}: V field {field.summary()}; setup {time.time() - t0:.1f} s; model {args.model}")
    fw = FutureWorld()
    fw.new_level()
    keep, grow = lv.hz_keep, lv.hz_grow
    dt_of = lambda w: float(w.get("dt") or 16.667)

    def feed(w):
        if w.get("keep") is not None:
            nonlocal keep
            keep = w["keep"]
        fans = w["fx"].get("fans", []) if w.get("fx") else None
        fw.observe(w["f"], w.get("hz", []), w.get("pf"), w.get("sh", []), keep, fans, dt_of(w), grow)
    for w in ref_world:                                   # the earlier wave: the hazards' recording
        feed(w)
    fw.new_wave()
    keep = lv.hz_keep
    wf = {w["f"]: w for w in world}
    frames = sorted(wf)
    rng = np.random.default_rng(args.seed)
    picks = set(frames[i] for i in np.linspace(5, len(frames) - 1, args.decisions).astype(int))
    by_frame = {}
    for r in rows:
        by_frame.setdefault(int(r["frame"]), []).append(r)
    out = open(args.json, "w", encoding="utf-8") if args.json else None
    ms, stats = [], []
    for f in frames:
        w = wf[f]
        feed(w)
        if w.get("tc"):
            planner.tiles(w["tc"])
        if w.get("pf"):
            lv.set_platforms(w["pf"])
        if f not in picks:
            continue
        alive = [r for r in by_frame.get(f, []) if r["status"] == 0]
        if not alive:
            continue
        r = alive[int(rng.integers(len(alive)))]
        row = state_row(r["x"], r["y"], r["vx"], r["vy"], r["flags"], r["state"], r["keys"], r["held"])
        H, dt = planner.H, dt_of(w)
        worlds, fboxes, recorded = future_inputs(fw, f, H, dt, lv.name)
        planner.prev = {}                                 # sampled states of different agents: no previous plan
        plan = planner.plan(row, f, worlds, fboxes, dt, agent=int(r["agent"]))
        ms.append(plan.timing["total"])
        c = plan.counts
        stats.append((c["stage1"]["steps"], c["stage1"]["complete"], c["stage2"].get("candidates", 0),
                      c["stage2"].get("steps", 0), c["stage2"].get("complete", 0), c["final"]["candidates"]))
        print(f"f{f:5d} agent {int(r['agent']):3d} ({plan.state['x']:6.0f},{plan.state['y']:6.0f}) "
              f"v({plan.state['vx']:5.0f},{plan.state['vy']:5.0f}) {'supported' if plan.state['supported'] else 'airborne'}"
              f" | {plan.summary()}")
        print(f"        {plan.timing['total']:.0f} ms (stage 1 {plan.timing['stage1']:.0f}, stage 2 "
              f"{plan.timing['stage2']:.0f}, final {plan.timing['final']:.0f}); stage 1 {c['stage1']['candidates']} moves x "
              f"{c['stage1']['steps']} steps ({c['stage1']['vetoed']} vetoed); stage 2 {c['stage2'].get('candidates', 0)} x "
              f"{c['stage2'].get('steps', 0)}/{c['stage2'].get('of_steps', 0)} steps ({c['stage2'].get('complete', 0)} to 48 frames,"
              f" {c['stage2'].get('vetoed', 0)} vetoed); final {c['final']['candidates']} x {c['final']['members']} members "
              f"({c['final']['vetoed']} vetoed); hazards recorded for {recorded}/{H} steps", flush=True)
        if out is not None:
            out.write(json.dumps(plan.to_dict()) + "\n")
    if out is not None:
        out.close()
    if ms:
        st = np.array(stats, np.float64)
        print(f"{len(ms)} decisions: {np.median(ms):.1f} ms median, {np.percentile(ms, 95):.1f} ms 95%, max {max(ms):.1f}; "
              f"mean evaluated: stage 1 {st[:, 0].mean():.1f} steps ({st[:, 1].mean():.1f} of 73 moves complete), "
              f"stage 2 {st[:, 2].mean():.0f} candidates x {st[:, 3].mean():.1f} steps ({st[:, 4].mean():.0f} complete), "
              f"final {st[:, 5].mean():.1f} candidates x 48 steps x all members")


if __name__ == "__main__":
    main()
