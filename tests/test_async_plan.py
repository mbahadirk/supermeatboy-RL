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


class _FakeProc:
    """The planner process as agent.Runner.pl_sync / ahead_take use it: calls recorded, nothing planned.
    send returns call ids from 100 on; ready(): `is_ready`; result(): (ok, res), the ids taken in `taken`."""

    def __init__(self, result=None):
        self.calls = []
        self.res = result
        self.ok = True
        self.is_ready = True
        self.taken = []
        self.next_id = 100

    def cancel(self, mid):
        self.calls.append(("cancel", mid))

    def call(self, name, *a, **kw):
        self.calls.append((name,) + a)

    def send(self, name, *a, **kw):
        self.calls.append((name,) + a)
        self.next_id += 1
        return self.next_id - 1

    def ready(self, mid):
        return self.is_ready

    def result(self, mid):
        self.taken.append(mid)
        return self.ok, self.res

    def take_events(self):
        return []


def _pending(mode: str, target: int = 108):
    """A Runner stub with --async-plan --async-wait `mode`: agent 0 has a committed plan and a decision for frame
    `target` being made ahead (started at 100)."""
    import agent
    r = object.__new__(agent.Runner)
    r.async_on = True
    r.wait_kinds = agent.WAIT_KINDS.get(mode, ())
    r.nowait = mode in agent.WAIT_KINDS
    r.cur_plan = {0: object()}
    r.commit = Committer(replan=8, on=True)
    r.ahead_miss = 0
    r.async_lead = 8
    r.st = {}
    r.proc = _FakeProc()
    r.ajob = agent.AheadJob(7, 0, 100, target, None, ())
    r.events = []
    r.event = lambda kind, frame, **kw: r.events.append(dict(kw, kind=kind, frame=frame))
    return r


def test_async_wait_defaults_to_events():
    import agent
    assert agent.build_parser().parse_args([]).async_wait == "events"
    assert set(agent.WAIT_KINDS["events"]) == {"danger", "start", "surprise", "support"}
    assert set(agent.WAIT_KINDS["danger"]) == {"danger", "start"}


@pytest.mark.parametrize("trigger", ["surprise +4 err 7.1 > 6.0", "support: on a wall (predicted ground)"])
def test_events_drops_the_plan_being_made_and_decides_now(trigger):
    """--async-wait events: a surprise or support trigger before the ahead job's frame drops the job (logged,
    the trigger named) and is returned for a decision at this frame, from the real state."""
    r = _pending("events")
    s = [0.0] * 8
    tr = r.route_triggers(0, 103, s, ["interval", trigger], 0, 16.667)
    assert tr == ["interval", trigger]                              # decided now (agent.py: need[i])
    assert r.ajob is None and ("cancel", 7) in r.proc.calls
    ev = [e for e in r.events if e["kind"] == "ahead dropped"]
    assert len(ev) == 1 and ev[0]["reason"].startswith(trigger.split()[0]) and "at 103" in ev[0]["reason"]
    assert r.st.get("ahead_dropped") == 1 and not r.st.get("ahead_absorbed")


def test_events_at_the_jobs_frame_a_surprise_is_urgent():
    r = _pending("events")
    r.proc.res = ("plan", 0.0, 0.0, 0.0, 0.0, 0.0)
    tr = r.route_triggers(0, 108, [0.0] * 8, ["surprise +8 err 9.0 > 6.6"], 0, 16.667)
    assert tr == ["surprise +8 err 9.0 > 6.6"] and r.ajob is None
    assert ("discard", 7) in r.proc.calls and r.events[-1]["reason"].startswith("surprise")
    assert r.ahead_miss == 0                                         # urgent: not a miss of the prediction


@pytest.mark.parametrize("mode,trigger", [("danger", "surprise +4 err 7.1 > 6.0"),
                                          ("danger", "support: on a wall (predicted ground)"),
                                          ("events", "interval")])
def test_other_triggers_before_the_jobs_frame_are_absorbed(mode, trigger):
    """--async-wait danger (the old rule): a surprise or support trigger is answered by the plan being made; with
    events, so is a timed one. The job stays, nothing is decided now."""
    r = _pending(mode)
    assert r.route_triggers(0, 103, [0.0] * 8, [trigger], 0, 16.667) == []
    assert r.ajob is not None and r.st["ahead_absorbed"] == 1 and not r.events and not r.proc.calls


def test_danger_still_drops_the_job_in_both_modes():
    for mode in ("danger", "events"):
        for triggers in (["danger: h3 at +6"], ["interval", "danger: h3 at +6"]):
            r = _pending(mode)
            assert r.route_triggers(0, 103, [0.0] * 8, triggers, 0, 16.667) == triggers
            assert r.ajob is None


@needs_model
def test_deferred_plan_leaves_the_planner_untouched_until_committed():
    from plan.planner import Planner
    lv, field = _flat()
    pl = Planner(MODEL, lv, field, budget_ms=None)
    worlds, frames = _inputs(pl.future_frames)
    p = pl.plan(_standing(), 0, worlds, frames, defer=True)
    assert p.path_step == 4 and p.to_dict()["path_step"] == 4
    assert set(p.predictions) == {4, 8, 16}       # +1 is only display interpolation
    assert 0 not in pl.prev and not pl.surprise.plans
    p.commit_fn()
    assert pl.prev[0] is p and pl.surprise.plans
    for k in range(4):
        pl.surprise.played(0, k, p.actions[k])
    assert pl.surprise.observe(0, 1, p.path[1][0] + 100, p.path[1][1]) == []
    assert pl.surprise.err[1] == []
    events = pl.surprise.observe(0, 4, p.path[4][0] + 100, p.path[4][1])
    assert len(events) == 1 and events[0]["horizon"] == 4
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
    assert p_rec.path_step == 1 and set(p_rec.predictions) == {1, 4, 8, 16}
    assert br["real"]["soft"] == 0.0 and br["real"]["margin"] == [0.0, 0.0]


# ---- --async-wait react: the game never waits for a plan made ahead (agent.Runner.route_react) ---------------
SURPRISE = "surprise +4 err 7.1 > 6.0"


def _committed(frame=96, n=48, trial=0):
    """A committed plan: actions 0, 1, .., 4, 0, 1, .. (told apart by frame), no move end; due at frame + 8."""
    p = _plan(frame=frame, n=n, move_end=0, trial=trial)
    p.actions = [k % 5 for k in range(n)]
    return p


def _react(monkeypatch, job=True, start=100, target=112, reactive=False, lat=(), plan=None):
    """A Runner stub with --async-plan --async-wait react: agent 0 plays a committed plan (made at 96); with
    job, a decision for frame `target` is being made ahead (started at `start`, call id 7)."""
    from collections import deque
    import agent
    monkeypatch.setattr(agent, "future_inputs",
                        lambda fw, frame, n, dt, name, deaths, **kw: ([None] * n, [None] * (n + 1), 0))
    r = object.__new__(agent.Runner)
    r.async_on = r.react = r.nowait = True
    r.wait_kinds = agent.WAIT_KINDS["react"]
    r.react_lat = deque(lat, maxlen=agent.REACT_WINDOW)
    r.planner = types.SimpleNamespace(f=4, future_frames=60, key_pickup=False,
                                      support=lambda xs, ys, keys=None: (None, np.ones(len(xs), bool),
                                                                         np.zeros(len(xs))))
    r.level = types.SimpleNamespace(name="test-flat")
    r.fw = r.deaths = None
    r.dt_ms = 1000.0 / 60.0
    r.commit = Committer(replan=8, on=True)
    r.commit.danger_always = True
    p = plan or _committed()
    r.cur_plan = {0: p}
    r.commit.decided(0, p, p.frame)
    r.ahead_miss = 0
    r.async_lead = 8
    r.st = {}
    r.proc = _FakeProc()
    r.ajob = agent.AheadJob(7, 0, start, target, None, (), reactive) if job else None
    r.events, r.decisions = [], []
    r.event = lambda kind, frame, **kw: r.events.append(dict(kw, kind=kind, frame=frame))
    r.on_plan = lambda plan, s, trig, ahead=None: r.decisions.append((plan, list(trig), ahead))
    return r


def _state(x=0.0, y=50.0, vx=0.0, vy=0.0):
    return [x, y, vx, vy, 0, 5, 0, 0]


def _started(r):
    """The plan_ahead calls sent: (d, target) each."""
    return [(c[5], c[6]) for c in r.proc.calls if c[0] == "plan_ahead"]


def _ahead_plan(actions, x0=100.0, n=60):
    """A plan made ahead for frame 112: runs right 1 unit / frame from x0 at y 50."""
    return types.SimpleNamespace(frame=112, agent=0, actions=list(actions) + [0] * (48 - len(actions)),
                                 path=[[x0 + k, 50.0] for k in range(n)], predictions={4: [x0 + 4, 50.0]},
                                 first_move_frames=0, trial_frames=0, goal_touch=None, key_touches=[],
                                 state={"keys": 0})


def test_react_parses_and_waits_only_for_danger_and_start():
    import agent
    ap = agent.build_parser()
    assert ap.parse_args(["--async-wait", "react"]).async_wait == "react"
    assert ap.parse_args([]).async_wait == "events"
    assert agent.WAIT_KINDS["react"] == ("danger", "start")
    assert "react (new, not the default)" in " ".join(ap.format_help().split())


def test_react_surprise_restarts_a_routine_job_then_absorbs(monkeypatch):
    """A surprise before the target of a job made for a timed decision cancels it and starts a reactive one from
    the real state now (prefix: the committed plan's next L actions); nothing is decided now. The surprise firing
    again the next frame is absorbed by the reactive job."""
    r = _react(monkeypatch)
    r.proc.is_ready = False
    assert r.route_triggers(0, 103, _state(), [SURPRISE], 0, 16.667) == []
    assert ("cancel", 7) in r.proc.calls and _started(r) == [(12, 115)]
    pa = next(c for c in r.proc.calls if c[0] == "plan_ahead")
    assert list(pa[2]) == r.cur_plan[0].actions[103 - 96:103 - 96 + 12]   # the committed plan's next 12 actions
    j = r.ajob
    assert j.reactive and j.start == 103 and j.target == 115 and j.why == [SURPRISE]
    assert r.st["ahead_restarted"] == 1 and not r.st.get("ahead_dropped")
    ev = [e for e in r.events if e["kind"] == "ahead dropped"]
    assert len(ev) == 1 and ev[0]["reason"].startswith("surprise") and "restarted" in ev[0]["reason"]
    assert r.route_triggers(0, 104, _state(), [SURPRISE, "missed: princess f3"], 0, 16.667) == []
    assert r.ajob is j and r.st["ahead_absorbed"] == 1 and len(_started(r)) == 1
    assert not r.proc.taken                                            # nothing waited for


def test_react_ready_routine_job_at_its_frame_is_judged_not_restarted(monkeypatch):
    """A routine job whose reply is here at its target is judged on the real state even when a surprise fires
    that frame (the tube check decides); not ready, the surprise restarts it."""
    r = _react(monkeypatch)
    r.proc.res = (_ahead_plan([]), 100.0, 50.0, 0.3, 0.0, 0.0)
    assert r.route_triggers(0, 112, _state(100.5, 50.0), [SURPRISE], 0, 16.667) == []
    assert r.st["ahead_used"] == 1 and not r.st.get("ahead_restarted") and r.decisions[0][1] == [SURPRISE]
    r = _react(monkeypatch)
    r.proc.is_ready = False
    assert r.route_triggers(0, 112, _state(100.5, 50.0), [SURPRISE], 0, 16.667) == []
    assert r.st["ahead_restarted"] == 1 and r.ajob.reactive and "not ready" in r.events[-1]["reason"]


def test_react_trigger_with_no_job_starts_one(monkeypatch):
    r = _react(monkeypatch, job=False)
    assert r.route_triggers(0, 101, _state(), ["support: landed early +5"], 0, 16.667) == []
    assert r.ajob.reactive and _started(r) == [(12, 113)]
    r = _react(monkeypatch, job=False)
    assert r.route_triggers(0, 104, _state(), ["interval"], 0, 16.667) == []
    assert not r.ajob.reactive and _started(r) == [(12, 116)]


@pytest.mark.parametrize("triggers", [["danger h3 f6"], [SURPRISE, "danger: risk 0.10 -> 0.50"], ["start: tiles"]])
def test_react_danger_and_start_are_decided_now(monkeypatch, triggers):
    r = _react(monkeypatch)
    assert r.route_triggers(0, 103, _state(), triggers, 0, 16.667) == triggers
    assert r.ajob is None and ("cancel", 7) in r.proc.calls and not _started(r)
    assert r.st["ahead_dropped"] == 1


def test_react_no_plan_is_decided_now(monkeypatch):
    r = _react(monkeypatch, job=False)
    r.cur_plan = {}
    assert r.route_triggers(0, 103, _state(), ["start"], 0, 16.667) == ["start"]


def test_react_never_waits_for_a_late_job(monkeypatch):
    """At its target, not ready: the committed plan plays on (no result() call), the job stays; up to
    REACT_LATE_MAX frames late it is still kept, one more and it is cancelled and restarted."""
    import agent
    r = _react(monkeypatch)
    r.proc.is_ready = False
    for f in range(112, 112 + agent.REACT_LATE_MAX + 1):
        assert r.route_triggers(0, f, _state(), ["interval"] if f > 112 else [], 0, 16.667) == []
        assert r.ajob is not None and r.ajob.mid == 7
    assert not r.proc.taken and not r.proc.calls
    f = 112 + agent.REACT_LATE_MAX + 1
    assert r.route_triggers(0, f, _state(), ["interval"], 0, 16.667) == []
    assert ("cancel", 7) in r.proc.calls and r.st["ahead_late_dropped"] == 1
    assert _started(r) == [(20, f + 20)] and r.ajob.start == f and not r.ajob.reactive   # L 24: 23 actions left
    assert r.events[-1]["reason"].startswith("late") and not r.proc.taken
    assert list(r.react_lat) == [f - 100] and r.st["react_latency"] == [f - 100]   # its age: a lower bound


def test_react_late_cancel_raises_the_lead(monkeypatch):
    """A job never ready gives its age at the late cancel as a latency sample, so L grows (else it never could)."""
    import agent
    r = _react(monkeypatch)
    r.proc.is_ready = False
    assert r.lead_now() == 12
    f = 112 + agent.REACT_LATE_MAX + 1                                 # 121: 21 frames after its start
    assert r.route_triggers(0, f, _state(), [], 0, 16.667) == []
    assert list(r.react_lat) == [21] and r.lead_now() == 24
    assert _started(r) == [(20, f + 20)]           # the new lead 24, shrunk to the 23 committed actions left


def test_react_late_result_with_the_played_prefix_is_used(monkeypatch):
    """Ready 2 frames after its target: the actions played on frames 112, 113 (the committed plan's) are its
    own first 2 and the real state is on its path at +2 (speed from the path, 1 unit / frame): it is committed,
    moved onto the real state, its frame stays 112 (action 2 plays now)."""
    r = _react(monkeypatch)
    played = r.cur_plan[0].actions[112 - 96:114 - 96]
    plan = _ahead_plan(played)
    r.proc.res = (plan, 100.0, 50.0, 0.3, 60.0, 0.0)
    assert r.route_triggers(0, 114, _state(102.5, 50.0, 60.0 * 1.1, 0.0), [], 0, 16.667) == []
    assert r.ajob is None and r.proc.taken == [7] and ("commit", 7) in r.proc.calls
    assert r.cur_plan[0] is plan and plan.frame == 112 and plan.path[2] == [102.5, 50.0]
    assert plan.predictions[4] == [104.5, 50.0]
    assert r.st["ahead_used"] == 1 and r.st["ahead_late_used"] == 1 and not r.st.get("ahead_dropped")
    (p, trig, ahead), = r.decisions
    assert p is plan and ahead["late"] == 2 and ahead["lead"] == 12 and ahead["start_off"] == 0.5
    assert r.commit.cur[0].plan is plan
    assert list(r.react_lat) == [14]                                   # ready seen at 114, started at 100


def test_react_late_result_with_other_actions_is_dropped_and_restarted(monkeypatch):
    r = _react(monkeypatch)
    played = r.cur_plan[0].actions[112 - 96:114 - 96]
    plan = _ahead_plan([(a + 1) % 5 for a in played])
    r.proc.res = (plan, 100.0, 50.0, 0.3, 60.0, 0.0)
    assert r.route_triggers(0, 114, _state(102.0, 50.0, 60.0, 0.0), ["interval"], 0, 16.667) == []
    assert ("discard", 7) in r.proc.calls and ("commit", 7) not in r.proc.calls
    assert r.st["ahead_dropped"] == 1 and r.ahead_miss == 1 and not r.decisions
    assert r.ajob.reactive and r.ajob.start == 114 and _started(r) == [(16, 130)]   # L from the sample 14
    assert "not its first 2" in r.events[-1]["reason"]


def test_react_on_time_result_off_the_tube_is_dropped_and_restarted(monkeypatch):
    r = _react(monkeypatch)
    r.proc.res = (_ahead_plan([]), 100.0, 50.0, 0.3, 0.0, 0.0)
    assert r.route_triggers(0, 112, _state(130.0, 50.0), [], 0, 16.667) == []
    assert r.st["ahead_dropped"] == 1 and r.ajob.reactive and r.events[-1]["reason"].startswith("start off by")
    r = _react(monkeypatch)
    r.proc.res = (_ahead_plan([]), 100.0, 50.0, 0.3, 0.0, 0.0)
    assert r.route_triggers(0, 112, _state(100.5, 50.0), [], 0, 16.667) == []      # on time, on the start
    assert r.st["ahead_used"] == 1 and not r.st.get("ahead_late_used") and r.decisions[0][2]["late"] == 0


def test_react_lead_follows_the_latency():
    import agent
    assert agent.react_lead([]) == agent.REACT_LEAD_INIT == 12
    assert agent.react_lead([10] * 16) == 12
    assert agent.react_lead([9, 10, 11, 10, 9, 10, 11, 10]) == 12
    assert agent.react_lead([15] * 16) == 16
    assert agent.react_lead([14, 15, 16, 15]) == 20                      # 90% 15.7: +1 -> 20
    assert agent.react_lead([40] * 4) == agent.REACT_LEAD_MAX == 24
    assert agent.react_lead([1, 2, 3]) == agent.REACT_LEAD_MIN == 8
    assert agent.react_lead([10] * 3, f=1) == 11
    assert agent.react_span(12, 40, 4) == 12 and agent.react_span(12, 10, 4) == 8 and agent.react_span(12, 3, 4) == 0


def test_react_latency_samples_set_the_lead(monkeypatch):
    """Polled once per frame while a job is in flight: the first frame its reply is seen gives the latency."""
    r = _react(monkeypatch, lat=[10, 10], target=120)
    r.proc.is_ready = False
    r.route_triggers(0, 110, _state(), [], 0, 16.667)
    assert list(r.react_lat) == [10, 10] and r.lead_now() == 12
    r.proc.is_ready = True
    r.route_triggers(0, 115, _state(), [], 0, 16.667)
    r.route_triggers(0, 116, _state(), [], 0, 16.667)                    # seen once only
    assert list(r.react_lat) == [10, 10, 15] and r.st["react_latency"] == [15]
    assert r.lead_now() == 16                                            # 90% 14 + 1 -> 16


def test_react_timed_start_uses_the_lead(monkeypatch):
    """No job in flight: the next timed decision (due 104) starts once due within L, for L frames on; with too
    few committed actions left, the largest whole model steps that fit, none: decided now."""
    r = _react(monkeypatch, job=False)
    assert r.ahead_start(0, 96, _state(), 0, 16.667)
    assert _started(r) == [(12, 108)] and not r.ajob.reactive
    r = _react(monkeypatch, job=False, lat=[18] * 4)                    # L = 20
    assert r.ahead_start(0, 96, _state(), 0, 16.667)
    assert _started(r) == [(20, 116)] and r.st["react_leads"] == [20]
    r = _react(monkeypatch, job=False, plan=_committed(trial=30))       # due 126: not within 12 at 96
    assert not r.ahead_start(0, 96, _state(), 0, 16.667) and r.ajob is None
    assert r.ahead_start(0, 114, _state(), 0, 16.667) and _started(r) == [(12, 126)]
    r = _react(monkeypatch, job=False, plan=_committed(n=20))           # actions up to frame 115
    assert r.route_triggers(0, 106, _state(), ["short"], 0, 16.667) == [] and _started(r) == [(8, 114)]
    r = _react(monkeypatch, job=False, plan=_committed(n=20))
    assert r.route_triggers(0, 113, _state(), ["short"], 0, 16.667) == ["short"] and r.ajob is None


def test_frame_hist_counts_python_plus_wait(monkeypatch):
    import agent
    clock = [0.0]
    monkeypatch.setattr(agent.time, "perf_counter", lambda: clock[0])
    c = agent.FrameCost()
    # (python ms, then the wait until the next frame's message); the last frame has no wait yet
    for py, wait in ((5, 10), (4, 12.5), (5, 25), (5, 40), (5, 60), (5, None)):
        c.begin()
        c.go("planning")
        clock[0] += py / 1000.0
        c.end()
        if wait is not None:
            clock[0] += wait / 1000.0
    s = c.summary(6, 16.667)
    assert s["frame_hist"] == {"<=17": 2, "<=34": 1, "<=50": 1, ">50": 1}
    assert set(s["cost_ms"]) == set(agent.FrameCost.CATS) | {"python", "wait"}
    assert "<=17 2 <=34 1 <=50 1 >50 1" in c.text(s)
    c.reset()
    assert c.frame_hist() == {"<=17": 0, "<=34": 0, "<=50": 0, ">50": 0}


def test_react_wave_summary():
    import agent
    st = {"react_leads": [12, 12, 16], "react_latency": [8, 10, 12, 20], "sync_decisions": 1, "sync_ms": 210.0}
    s = agent.react_summary(st)
    assert s["lead"] == {"median": 12.0, "max": 16} and s["latency_frames"]["p50"] == 11.0
    assert agent.react_summary({}) == {"lead": None, "latency_frames": None}
    assert "waited for 1 decisions (210 ms)" in agent.react_text(st, s)
