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


@needs_model
def test_ahead_start_on_the_best_route_is_the_recorded_state():
    """A decision made ahead on the best route starts from the route's recorded state (the game repeats it),
    not the model's guess: live (online_check01 3-1 wave 2 f152) the model's start, 0.92 units and 0.39 u/f off
    the exact route state, priced a pass the route had survived at soft risk 0.92 and the agent left the route.
    Off the route (another state, frame or prefix) the model's prediction stays."""
    from mpc.candidates import act
    from plan.planner import Planner, state_row
    lv, field = _flat()
    pl = Planner(MODEL, lv, field, budget_ms=None)
    pl.trials.gain = 0.0                                              # no route trial here (test_route_trials.py)
    x0, n, d = 10 * T + 10, 120, 8
    pl.route.best = {"finish": 400, "f": np.arange(92, 92 + n), "x": np.full(n, x0), "y": np.full(n, 46.0),
                     "vx": np.zeros(n), "vy": np.zeros(n), "a": np.full(n, act(0))}
    H = pl.future_frames
    worlds, frames = _inputs(H + d)
    prefix = np.full(d, act(0), np.int64)
    best = pl.route.best

    def both(row, acts, frame):                                       # (with the route, the model's alone)
        pl.route.best = None
        alone = pl.predict_start(row, acts, worlds[:d], frame=frame)
        pl.route.best = best
        return pl.predict_start(row, acts, worlds[:d], frame=frame), alone
    (pred, spread), (model_pred, _) = both(_standing(), prefix, 92)
    assert spread == 0.0 and pred["x"][0] == x0 and pred["y"][0] == 46.0 and pred["vx"][0] == 0.0
    assert abs(model_pred["x"][0] - x0) > 0.05                        # the model's guess is off (the stub's point)
    assert pred["state"][0] == model_pred["state"][0]                 # the fields not recorded: the model's
    for row, acts, frame in ((state_row(x0 + 1.0, 46.0, 0.0, 0.0, 0, 5, 0, 0), prefix, 92),   # another state
                             (_standing(), np.full(d, act(1), np.int64), 92),                  # another prefix
                             (_standing(), prefix, 91)):                         # a frame the route has not
        (p, s), (m, ms) = both(row, acts, frame)
        assert p["x"][0] == m["x"][0] and p["vx"][0] == m["vx"][0] and s == ms
    # the decision at frame 100: a box 3 units beside the route from frame 20 on (the route stood there and lived)
    fb = S.FrameBoxes(np.array([[x0 - 13.0, 46.0, 4.0, 4.0, 0.0]]), np.zeros(1, np.int8), ["h0"])
    assert fb.clearance(np.array([x0]), np.array([46.0]))[0, 0] == 3.0
    fr = [frames[k] if k < 20 else fb for k in range(H + 1)]
    pl.new_wave()
    mis = state_row(x0 + 0.92, 46.0, 0.39 * 60.0, 0.0, 0, 5, 0, 0)    # the model's start as live: 0.92 / 0.39 off
    p_mis = pl.plan(mis, 100, worlds[d:], fr, recorded=H)
    br = p_mis.best_route
    assert br["judged"] == "real" and not br["used"] and br["why"].startswith("soft risk"), br
    pl.new_wave()
    p_rec = pl.plan(pred, 100, worlds[d:], fr, recorded=H)
    br = p_rec.best_route
    assert br["judged"] == "real" and br["used"] and p_rec.mode == "best-route", br
    assert br["real"]["soft"] == 0.0 and br["real"]["margin"] == [0.0, 0.0]
