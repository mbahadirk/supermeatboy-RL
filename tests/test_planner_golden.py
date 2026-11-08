"""Plans that touch no goal and no key score exactly as recorded (plan/planner.py): the decisions below were
re-recorded when the score became estimated frames to the goal (step 4: time field, airborne completion,
point-of-no-return memory; tests/data/planner_golden.json, with the world model's digest) and are compared
field by field, floats exactly; goal / key contact must not change them. Skipped without the model, or
when the model file is not the one the record was made with.

The collision guard (plan/planner.py collide): with the guard off every case is the record ("cases"); with
it on, the cases in which no candidate of any stage needed clipping are the record too, and the ones that did
were re-recorded with the guard ("guarded": only those).

    python tests/test_planner_golden.py --write      # re-record (only when the scoring is meant to change)
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

MODEL = os.path.join(HERE, "runs", "wm_locked", "wm_model.pt")
DATA_113 = os.path.join(HERE, "runs", "real_20261102-165351", "wm_data")
GOLDEN = os.path.join(HERE, "tests", "data", "planner_golden.json")
T = 20.0
NO_DEADLINE = None             # every stage runs complete (the record was made with a 1e9 ms budget: the same)
FIELDS = ("chosen", "actions", "segments", "pad", "from_prev", "score", "v_start", "v_end", "dead_end_members",
          "leg", "veto", "soft_risk", "soft_frame", "end_supported", "first_move_frames")
ALT_FIELDS = ("moves", "total", "progress", "stage", "reason", "vetoed")


def model_digest() -> str:
    h = hashlib.sha256()
    with open(MODEL, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def record(plan) -> dict:
    out = {k: getattr(plan, k) for k in FIELDS}
    out["alternatives"] = [{k: a.get(k) for k in ALT_FIELDS} for a in plan.alternatives]
    out["counts"] = {k: {kk: vv for kk, vv in v.items()} for k, v in plan.counts.items()}
    out["contact"] = [getattr(plan, "goal_touch", None), getattr(plan, "key_touches", [])]
    return json.loads(json.dumps(out))            # plain types (floats keep every digit)


def _flat(keys=(), tol="default"):
    from game.level import Level
    from plan.planner import Planner
    from plan.progress import ProgressField
    H, W = 20, 60
    w = np.zeros((H, W), np.uint32)
    w[0:2, :] = 1
    info = {"name": "test-flat", "W": W, "H": H, "ox": 0.0, "oy": 0.0, "spawn": [10 * T + 10, 46.0],
            "goals": [[50 * T + 10, 50.0]], "objs": [[0, float(x), float(y), 0x40] for x, y in keys],
            "lethalTypes": []}
    lv = Level(info, w.astype("<u4").tobytes())
    field = ProgressField(w, 0.0, 0.0, lv.spawn, lv.goals, [], [], lv.keys, [], [])
    kw = {} if tol == "default" else {"collide_tol": tol}
    return lv, Planner(MODEL, lv, field, budget_ms=NO_DEADLINE, hysteresis=0.0, **kw)   # no switch rule: as before


def _inputs(H, boxes=()):
    from plan import safety as S
    from wm.rollout import World
    fb = S.FrameBoxes(np.array(boxes, float).reshape(-1, 5), np.zeros(len(boxes), np.int8),
                      [f"h{k}" for k in range(len(boxes))])
    return [World() for _ in range(H)], [None] + [fb] * H


def cases(tol="default"):
    """[(name, recorded decision)] on fixed inputs; none of them reaches the goal or a key within 48 frames.
    tol: the collision guard's tolerance (default: the planner's; None: no guard)."""
    import torch
    from plan.planner import state_row
    torch.set_num_threads(1)
    out = []
    lv, pl = _flat(tol=tol)
    worlds, frames = _inputs(pl.future_frames)               # inputs as decisions get them: H + the completion
    st = state_row(210.0, 46.0, 0.0, 0.0, 0, 5, 0, 0)
    out.append(("flat standing", record(pl.plan(st, 100, worlds, frames))))
    out.append(("flat standing, previous plan", record(pl.plan(st, 102, worlds, frames))))
    pl.new_wave()
    worlds, frames = _inputs(pl.future_frames, [(240.0, 150.0, 4.0, 150.0, 0.0)])
    out.append(("flat hazard strip", record(pl.plan(st, 200, worlds, frames))))
    pl.new_wave()
    worlds, frames = _inputs(pl.future_frames, [(330.0, 60.0, 10.0, 10.0, 0.0)])
    air = state_row(260.0, 120.0, 240.0, -100.0, 0, 7, 0, 0)
    out.append(("flat airborne, hazard ahead", record(pl.plan(air, 50, worlds, frames))))
    lvk, plk = _flat(keys=[(800.0, 50.0)], tol=tol)                        # a key far away: two legs
    worlds, frames = _inputs(plk.future_frames)
    out.append(("flat with a far key", record(plk.plan(st, 100, worlds, frames))))
    if os.path.isdir(DATA_113):
        from mpc.future import FutureWorld
        from plan.offline import build
        from plan.planner import future_inputs
        lv3, field, pl3, ref_world, rows, world = build(DATA_113, "1-13", MODEL, NO_DEADLINE, tol)
        fw = FutureWorld()
        for w in ref_world:
            fw.observe(w["f"], w.get("hz", []), w.get("pf"), w.get("sh", []), lv3.hz_keep, None, 16.667)
        fw.new_wave()
        fs = sorted(w["f"] for w in world)
        for f in (fs[200], fs[400]):
            for w in world:
                if w["f"] <= f and w["f"] not in fw.cur:
                    fw.observe(w["f"], w.get("hz", []), w.get("pf"), w.get("sh", []), lv3.hz_keep, None, 16.667)
            r = [r for r in rows if r["frame"] == f and r["status"] == 0][0]
            worlds, frames, _ = future_inputs(fw, f, pl3.future_frames, 16.667, lv3.name)
            pl3.prev = {}
            plan = pl3.plan(state_row(r["x"], r["y"], r["vx"], r["vy"], r["flags"], r["state"], r["keys"], r["held"]),
                            f, worlds, frames)
            out.append((f"1-13 recorded f{f}", record(plan)))
    return out


def _same(a, b, path=""):
    """Exact comparison of recorded values; keys the record does not have (added later) are not compared."""
    if isinstance(b, dict):
        assert isinstance(a, dict), path
        for k in b:
            assert k in a, f"{path}.{k} missing"
            _same(a[k], b[k], f"{path}.{k}")
    elif isinstance(b, list):
        assert isinstance(a, list) and len(a) == len(b), f"{path}: {a} != {b}"
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def clipped(rec) -> int:
    """Candidates the collision guard clipped in any stage of a recorded decision."""
    return sum(int(rec["counts"][s].get("clipped") or 0) for s in ("stage1", "stage2", "final")) +         int(rec["counts"]["final"].get("completion_clipped") or 0)               # completion rollouts too


def _check(now, want):
    for name, rec in want.items():
        _same(now[name], rec, name)
        # none of them touches the goal or a key (in any stage), so contact must not have changed anything
        assert now[name]["contact"] == [None, []], name
        assert all(now[name]["counts"][s].get("goal", 0) == 0 for s in ("stage1", "stage2", "final")), name


def _load():
    with open(GOLDEN, encoding="utf-8") as f:
        gold = json.load(f)
    if gold["model_sha256"] != model_digest():
        pytest.skip("the world model changed since the record was made")
    return gold


def _cases(tol):
    import torch
    threads = torch.get_num_threads()
    try:
        return dict(cases(tol))
    finally:
        torch.set_num_threads(threads)


def _wanted(gold, keys):
    want = dict(gold.get(keys) or [])
    if not os.path.isdir(DATA_113):
        want = {k: v for k, v in want.items() if not k.startswith("1-13")}
    return want


@pytest.mark.skipif(not os.path.exists(MODEL) or not os.path.exists(GOLDEN), reason="no world model or record")
def test_no_contact_plans_score_as_before():
    """Without the collision guard: every case exactly as recorded before goal / key contact."""
    gold = _load()
    now = _cases(None)
    want = _wanted(gold, "cases")
    assert set(now) == set(want)
    _check(now, want)


@pytest.mark.skipif(not os.path.exists(MODEL) or not os.path.exists(GOLDEN), reason="no world model or record")
def test_collision_guard_changes_only_clipped_decisions():
    """With the guard: a case no candidate needed clipping in is the old record; the others the re-record."""
    gold = _load()
    now = _cases("default")
    old, guarded = _wanted(gold, "cases"), _wanted(gold, "guarded")
    assert set(now) == set(old)
    for name in now:
        if name in guarded:
            assert clipped(now[name]) > 0, name
        else:
            assert clipped(now[name]) == 0, name
    _check(now, {k: (guarded[k] if k in guarded else v) for k, v in old.items()})


if __name__ == "__main__" and "--write" in sys.argv:
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    off = cases(None)
    on = dict(cases("default"))
    data = {"about": "planner decisions scored in estimated frames to the goal (step 4; tests/test_planner_golden.py); "
                     "guarded: the cases recorded with the collision guard because it clipped a candidate in them",
            "model_sha256": model_digest(), "cases": off,
            "guarded": [[k, v] for k, v in on.items() if clipped(v) > 0]}
    with open(GOLDEN, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {GOLDEN}: {len(data['cases'])} cases, {len(data['guarded'])} re-recorded with the guard")
