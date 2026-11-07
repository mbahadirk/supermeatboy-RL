"""Run the beam planner (plan/planner.py) on recorded states, no game: what it picks, why, and what a decision
costs.

    python -m plan.offline <run>/wm_data --level 4-6 [--decisions 20] [--model runs/wm_locked/wm_model.pt]
                           [--budget 120] [--threads 2] [--seed 0] [--json plans.jsonl]
    python -m plan.offline runs/plan_20261103-152013/wm_data --level 1-12 --closed-loop 400 [--start 0]
                           [--replan 8] [--tube 1.5] [--triggers on] [--hysteresis 0.5] [--json summary.jsonl]

--closed-loop: the world model (member 0) plays itself from a recorded state, the old way (--replan 2,
triggers off, hysteresis 0) and then with the given flags (closed_loop): decisions per 100 frames, switch rate,
frames each committed plan was followed, planning ms per game frame. The model playing itself, not the game.

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
import math
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


def closed_loop(d: str, level: str, model: str, frames: int = 300, replan: int = 8, tube: float = 1.5,
                triggers: bool = True, hysteresis: float = 0.5, budget: float | None = 120.0, start: int = 0,
                agent: int | None = None, log=None) -> dict:
    """The world model plays itself: ensemble member 0 stands in for the game for one level (NOT the game).

    From a recorded agent's state at wave frame `start` of the level's second recorded wave (agent: a planned
    one when the shard says so, else the first alive), each frame: the planner's surprise / failure check on
    the state (Planner.observe), the replan triggers (plan/commit.py, as agent.py with --replan / --tube /
    --triggers), a decision when one fires (Planner.plan with `hysteresis`), the committed plan's action, then
    one step of member 0 with that action; the hazards are the recorded ones of that wave (the time step is
    locked, so they do not depend on the agent), death by the geometry (plan.safety against the next frame's
    recorded boxes and the level's edge), the goal and keys by their contact boxes. Not modelled: tile changes
    (crumbling blocks stay), the model's own death and finish heads. Returns the numbers and the plan dicts."""
    from plan import safety as S
    from plan.commit import Committer, kind_of
    from plan.planner import imagine
    lv, field, planner, ref_world, rows, world = build(d, level, model, budget)
    planner.hysteresis = float(hysteresis)
    com = Committer(replan, tube, triggers)
    fw = FutureWorld()
    fw.new_level()
    keep, grow = lv.hz_keep, lv.hz_grow
    dt_of = lambda w: float(w.get("dt") or 16.667)

    def feed(w):
        nonlocal keep
        if w.get("keep") is not None:
            keep = w["keep"]
        fans = w["fx"].get("fans", []) if w.get("fx") else None
        fw.observe(w["f"], w.get("hz", []), w.get("pf"), w.get("sh", []), keep, fans, dt_of(w), grow)
    for w in ref_world:
        feed(w)
    fw.new_wave()
    keep = lv.hz_keep
    wf = {w["f"]: w for w in world}
    here = [r for r in rows if int(r["frame"]) == start and r["status"] == 0]
    if agent is not None:
        here = [r for r in here if int(r["agent"]) == agent]
    if not here:
        raise SystemExit(f"{level}: no alive recorded agent at frame {start}")
    r0 = min(here, key=lambda r: int(r["agent"]))
    s = state_row(r0["x"], r0["y"], r0["vx"], r0["vy"], r0["flags"], r0["state"], r0["keys"], r0["held"])
    goal_boxes = np.asarray(lv.goal_boxes(), np.float64).reshape(-1, 5)
    key_boxes = list(lv.key_boxes())
    plans, played, plan_ms, check_s = [], {}, 0.0, 0.0
    cur = None
    outcome, f = "alive", start
    for f in range(start, start + frames):
        w = wf.get(f)
        if w is None or f + 1 not in wf:
            outcome = "end of the recording"
            break
        feed(w)
        if w.get("pf"):
            lv.set_platforms(w["pf"])
        dt = dt_of(w)
        x, y, keys = float(s["x"][0]), float(s["y"][0]), int(s["keys"][0])
        planner.observe(0, f, x, y, keys)
        fut = {}

        def fboxes_now():
            if "fb" not in fut:
                fut["w"], fut["fb"], _ = future_inputs(fw, f, planner.H, dt, lv.name)
            return fut["fb"]
        t = time.perf_counter()
        trig = com.triggers(0, f, x, y, keys, False, planner, fboxes_now)
        check_s += time.perf_counter() - t
        if trig:
            fboxes_now()
            cur = planner.plan(s, f, fut["w"], fut["fb"], dt, agent=0)
            com.decided(0, cur, f, fut["fb"], planner)
            plan_ms += cur.timing["total"]
            p = cur.to_dict()
            p["trigger"] = list(trig)
            plans.append(p)
        a = int(cur.actions[f - cur.frame])
        planner.played(0, f, a)
        played[f] = a
        out = imagine(planner.model, slice(0, 1), planner.geom, s, np.array([[a]]), dt, fw.worlds(f, 1, dt)[0],
                      keep_rows=True)
        s2 = out["rows"][0, :, 1].copy()
        x2, y2 = float(s2["x"][0]), float(s2["y"][0])
        nxt = S.record_boxes(wf[f + 1], keep, grow)
        kind, _ = S.segment_contact([x], [y], [x2], [y2], nxt, planner.bounds)
        seg = (np.array([[[x, x2]]]), np.array([[[y, y2]]]))
        if kind[0] != S.KIND_NONE:
            outcome = f"died at frame {f + 1} ({S.KIND_NAMES[int(kind[0])]})"
            break
        if len(goal_boxes) and S.first_contact(*seg, goal_boxes)[0][0, 0] > 0:
            outcome = f"finished at frame {f + 1}"
            break
        for j, b in enumerate(key_boxes):
            if not (int(s2["keys"][0]) >> j) & 1 and S.first_contact(*seg, [b])[0][0, 0] > 0:
                s2["keys"] = int(s2["keys"][0]) | (1 << j)
        s = s2
    n_frames = max(1, f - start + 1)
    follow = []
    for p in plans:
        if p.get("from_prev"):
            continue
        h = 0
        while h < len(p["actions"]) and played.get(p["frame"] + h) == p["actions"][h]:
            h += 1
        follow.append(h)
    fr = [p["frame"] for p in plans]
    sw = [p.get("switch") for p in plans]
    trig = {}
    for p in plans:
        for t in p["trigger"]:
            trig[kind_of(t)] = trig.get(kind_of(t), 0) + 1
    v1 = float(planner.value(np.array([float(s["x"][0])]), np.array([float(s["y"][0])]), int(s["keys"][0]))[0][0])
    res = {"level": level, "start": start, "agent": int(r0["agent"]), "frames": n_frames, "outcome": outcome,
           "mode": {"replan": replan, "tube": tube, "triggers": "on" if triggers else "off", "hysteresis": hysteresis,
                    "budget_ms": budget},
           "decisions": len(plans), "decisions_per_100": round(100.0 * len(plans) / n_frames, 2),
           "switched": sum(1 for x in sw if x and not x["kept"]), "kept": sum(1 for x in sw if x and x["kept"]),
           "switch_rate": round(sum(1 for x in sw if x and not x["kept"]) / len(plans), 4) if plans else None,
           "gap_median": float(np.median(np.diff(fr))) if len(fr) > 1 else None,
           "committed_followed_median": float(np.median(follow)) if follow else None, "committed": len(follow),
           "plan_ms_per_frame": round(plan_ms / n_frames, 2), "check_ms_per_frame": round(1000 * check_s / n_frames, 3),
           "triggers": dict(sorted(trig.items(), key=lambda kv: -kv[1])),
           "v_start": plans[0]["v_start"] if plans else None, "v_end": round(v1, 2) if math.isfinite(v1) else None,
           "plans": plans}
    if log:
        log(closed_loop_text(res))
    return res


def closed_loop_text(r: dict) -> str:
    m = r["mode"]
    f = lambda v, spec: "-" if v is None else format(v, spec)
    return (f"[closed loop: the model plays itself, not the game] {r['level']} from frame {r['start']} (agent "
            f"{r['agent']}), --replan {m['replan']} --triggers {m['triggers']} --tube {m['tube']:g} --hysteresis "
            f"{m['hysteresis']:g}: {r['frames']} frames, {r['outcome']}; {r['decisions']} decisions "
            f"({f(r['decisions_per_100'], '.1f')} per 100 frames, every {f(r['gap_median'], '.0f')} f median), "
            f"switch rate {f(None if r['switch_rate'] is None else 100 * r['switch_rate'], '.0f')}% "
            f"(switched {r['switched']}, kept {r['kept']}), committed plans followed {f(r['committed_followed_median'], '.0f')}"
            f" frames (median, n {r['committed']}), planning {f(r['plan_ms_per_frame'], '.1f')} ms per game frame "
            f"(+checks {f(r['check_ms_per_frame'], '.2f')}); V {f(r['v_start'], '.1f')} -> {f(r['v_end'], '.1f')}; "
            f"triggers {r['triggers']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--level", required=True)
    ap.add_argument("--closed-loop", type=int, default=0, metavar="FRAMES",
                    help="instead: the model (member 0) plays itself for FRAMES frames from a recorded state, twice: "
                         "the old mode (--replan 2, triggers off, hysteresis 0) and the given flags")
    ap.add_argument("--start", type=int, default=0, help="closed loop: the recorded wave frame to start from")
    ap.add_argument("--replan", type=int, default=8, help="closed loop: as agent.py --replan (default 8)")
    ap.add_argument("--tube", type=float, default=1.5, help="closed loop: as agent.py --tube (default 1.5)")
    ap.add_argument("--triggers", choices=("on", "off"), default="on", help="closed loop: as agent.py --triggers")
    ap.add_argument("--hysteresis", type=float, default=0.5, help="closed loop: as agent.py --hysteresis (default 0.5)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--decisions", type=int, default=20)
    ap.add_argument("--budget", type=float, default=120.0, help="ms per decision (the cheap stages' deadline)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="write every Plan.to_dict() here (one JSON line each)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.closed_loop:
        out = []
        for mode in ({"replan": 2, "tube": args.tube, "triggers": False, "hysteresis": 0.0},
                     {"replan": args.replan, "tube": args.tube, "triggers": args.triggers == "on",
                      "hysteresis": args.hysteresis}):
            r = closed_loop(args.data, args.level, args.model, args.closed_loop, budget=args.budget, start=args.start,
                            log=lambda t: print(t, flush=True), **mode)
            out.append(r)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                for r in out:
                    f.write(json.dumps({k: v for k, v in r.items() if k != "plans"}) + "\n")
        return
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
