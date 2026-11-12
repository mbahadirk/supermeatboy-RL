"""agent.py --async-plan: Committer.due, deferred plan effects, and the planner process (plan/worker.py)."""
import os
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from plan import safety as S                                               # noqa: E402
from plan.commit import Committer, SHORT_FRAMES                            # noqa: E402

MODEL = os.path.join(HERE, "runs", "wm_step4", "wm_model.pt")
needs_model = pytest.mark.skipif(not os.path.exists(MODEL), reason="no world model (runs/wm_step4)")
T = 20.0


def _plan(frame=0, n=48, move_end=12, trial=0):
    return types.SimpleNamespace(frame=frame, actions=[0] * n, first_move_frames=move_end, trial_frames=trial,
                                 goal_touch=None, key_touches=[])


def test_due_is_the_first_timed_trigger():
    cm = Committer(replan=8, on=True)
    assert cm.due(0) is None
    cm.decided(0, _plan(frame=100, move_end=12), 100)
    assert cm.due(0) == 108                                           # interval before the move's end
    cm.decided(0, _plan(frame=100, move_end=4), 100)
    assert cm.due(0) == 104                                           # the move ends first
    cm.decided(0, _plan(frame=100, n=20, move_end=0), 100)
    assert cm.due(0) == 100 + 20 - SHORT_FRAMES + 1                   # short before the interval
    cm.decided(0, _plan(frame=100, move_end=4, trial=30), 100)
    assert cm.due(0) == 130                                           # a route trial plays out first
    assert not any(cm.triggers(0, 100 + k, 0.0, 0.0) for k in range(1, 30))   # the triggers agree with due
    assert "interval" in cm.triggers(0, 130, 0.0, 0.0)
    cm.decided(0, _plan(frame=100, move_end=12), 100)
    assert not any(cm.triggers(0, 100 + k, 0.0, 0.0) for k in range(1, 8))
    assert cm.triggers(0, 108, 0.0, 0.0) == ["interval"]


def _flat():
    from game.level import Level
    from plan.progress import ProgressField
    H, W = 20, 60
    w = np.zeros((H, W), np.uint32)
    w[0:2, :] = 1
    info = {"name": "test-flat", "W": W, "H": H, "ox": 0.0, "oy": 0.0, "spawn": [10 * T + 10, 46.0],
            "goals": [[50 * T + 10, 50.0]], "objs": [], "lethalTypes": []}
    lv = Level(info, w.astype("<u4").tobytes())
    return lv, ProgressField(w, 0.0, 0.0, lv.spawn, lv.goals, [], [], [], [], [])


def _inputs(H):
    from wm.rollout import World
    fb = S.FrameBoxes(np.zeros((0, 5)), np.zeros(0, np.int8), [])
    return [World() for _ in range(H)], [None] + [fb] * H


def _standing():
    from plan.planner import state_row
    return state_row(10 * T + 10, 46.0, 0.0, 0.0, 0, 5, 0, 0)


@needs_model
def test_deferred_plan_leaves_the_planner_untouched_until_committed():
    from plan.planner import Planner
    lv, field = _flat()
    pl = Planner(MODEL, lv, field, budget_ms=None)
    worlds, frames = _inputs(pl.future_frames)
    p = pl.plan(_standing(), 0, worlds, frames, defer=True)
    assert 0 not in pl.prev and not pl.surprise.plans
    p.commit_fn()
    assert pl.prev[0] is p and pl.surprise.plans
    pred, spread = pl.predict_start(_standing(), np.asarray(p.actions[:8]), worlds[:8])
    assert pred.shape == (1,) and pred["x"][0] > 10 * T and spread >= 0.0     # it runs right
    with pytest.raises(ValueError):
        pl.predict_start(_standing(), np.asarray(p.actions[:6]), worlds[:6])  # not whole model steps


@needs_model
def test_planner_process_plans_ahead_in_order_and_cancels():
    from plan.worker import PlannerProcess, TubeFn
    from plan.commit import ErrorTube
    lv, field = _flat()
    proc = PlannerProcess(threads=1)
    try:
        proc.call("init", MODEL, lv, field, {"budget_ms": None}, None, TubeFn(ErrorTube(), 1.5, 3.0))
        H = 48 + 45 + 8
        worlds, frames = _inputs(H)
        p0, trial = proc.call("plan_now", _standing(), 0, worlds, frames, 16.667, agent=0)
        assert len(p0.actions) == 48 and trial is None and not hasattr(p0, "commit_fn")
        row = _standing()
        mid = proc.send("plan_ahead", row, np.asarray(p0.actions[:8], np.int64), worlds, frames, 8, 8, 16.667, 0, 0,
                        keep=True)
        proc.send("observe", 0, 1, float(row["x"][0]), 46.0, 0, False)  # waits behind the plan, in order
        ok, res = proc.result(mid)
        assert ok and res[0].frame == 8 and res[1] > float(row["x"][0])
        assert proc.call("commit", mid) is None
        assert [n for n, ok_, _ in proc.take_events()] == ["observe"]
        mid = proc.send("plan_ahead", row, np.asarray(p0.actions[:8], np.int64), worlds, frames, 8, 8, 16.667, 0, 0,
                        keep=True)
        proc.cancel(mid)                                               # stopped (or already done): dropped
        p1, _ = proc.call("plan_now", _standing(), 0, worlds, frames, 16.667, agent=0)   # not cancelled after
        assert len(p1.actions) == 48
    finally:
        proc.close()
    assert not proc.proc.is_alive()
