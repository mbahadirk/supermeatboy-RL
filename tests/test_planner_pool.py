"""agent.py --async-wait auto and --planners N: the per-level wait mode, plan.worker.PlannerPool (broadcast,
primary-only replies, adopt), Planner.adopt (the deferred effects as data) and the react scheduling of several
plans made ahead at once (agent.Runner.route_pool / pool_fill / pool_reconcile)."""
import os
import pickle
import sys
import types
from collections import deque

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plan.commit import Committer                                          # noqa: E402
from test_async_plan import (MODEL, SURPRISE, T, _FakeProc, _ahead_plan, _committed, _flat, _inputs,  # noqa: E402
                             _standing, _state, needs_model)


# ---- --async-wait auto ---------------------------------------------------------------------------------------
def _auto(name="4-6"):
    """A Runner stub with --async-plan --async-wait auto on level `name`: react, a job in flight (id 7)."""
    import agent
    from mpc.future import FutureWorld
    r = object.__new__(agent.Runner)
    r.async_on = r.auto_wait = True
    r.dyn_levels = {}
    r.commit = Committer(replan=8, on=True)
    r.fw = FutureWorld()
    r.level = types.SimpleNamespace(name=name, plats=[])
    r.st, r.wave_no = {}, 3
    r.proc = _FakeProc()
    r.ajob = agent.AheadJob(7, 0, 100, 112, None, ())
    r.jobs = []
    r.events = []
    r.event = lambda kind, frame, **kw: r.events.append(dict(kw, kind=kind, frame=frame))
    r.level_wait(name)
    return r


def _hz(x, y=50.0):
    return [x, y, 24.0, 24.0, 1.0, 1.0, 0.0]


def test_auto_parses_and_the_default_stays_events():
    import agent
    ap = agent.build_parser()
    assert ap.parse_args(["--async-wait", "auto"]).async_wait == "auto"
    assert ap.parse_args([]).async_wait == "events" and ap.parse_args([]).planners == 1
    text = " ".join(ap.format_help().split())
    assert "auto (new, not the default)" in text and "--planners" in text


def test_auto_starts_in_react_and_a_static_level_stays_there():
    import agent
    r = _auto("3-1")
    assert r.react and r.wait_mode == "react" and r.wait_kinds == agent.WAIT_KINDS["react"]
    assert r.commit.danger_always
    for f in range(1, 40):                                            # still hazards, no platform, no shot
        r.fw.observe(f, [_hz(100.0), _hz(300.0, 80.0)], [], [], None, [], 16.667)
        r.auto_check(f, {"hz": [], "pf": [], "sh": []}, r.level)
    assert r.react and not r.events and r.ajob is not None and not r.proc.calls


@pytest.mark.parametrize("what", ["platform", "hazard", "shot"])
def test_auto_switches_to_events_once_on_the_first_moving_part(what):
    """A moving platform, a kill box moving more than 0.5 units in a frame, or a shot: the job in flight is
    dropped (pl_sync), the level decides as events from this frame on, logged once; a later visit of the level
    starts in events, another level in react."""
    import agent
    r = _auto("4-6")
    r.fw.observe(10, [_hz(100.0)], [], [], None, [], 16.667)
    m = {"pf": [], "sh": []}
    if what == "platform":
        m["pf"] = [[200.0, 60.0, 1]]
    elif what == "shot":
        m["sh"] = [[3, 150.0, 70.0, 4.0]]
    x1 = 100.8 if what == "hazard" else 100.0
    r.fw.observe(11, [_hz(x1)], m["pf"], m["sh"], None, [], 16.667)
    r.auto_check(11, m, r.level)
    assert not r.react and r.wait_mode == "events" and r.wait_kinds == agent.WAIT_KINDS["events"]
    assert not r.commit.danger_always and r.nowait
    assert r.ajob is None and ("cancel", 7) in r.proc.calls and ("ping",) in r.proc.calls
    ev = [e for e in r.events if e["kind"] == "wait mode"]
    assert len(ev) == 1 and ev[0]["level"] == "4-6" and ev[0]["mode"] == "events" and ev[0]["frame"] == 11
    assert {"platform": "moving platform", "hazard": "hazard h0 moved 0.80", "shot": "1 shot"}[what] in ev[0]["why"]
    assert r.st["wait_mode_switch"] == 11
    r.fw.observe(12, [_hz(x1 + 1.0)], m["pf"], m["sh"], None, [], 16.667)
    r.auto_check(12, m, r.level)
    assert len([e for e in r.events if e["kind"] == "wait mode"]) == 1     # once
    r.level_wait("4-6")                                               # a later visit
    assert r.wait_mode == "events" and not r.react
    r.level_wait("3-1")
    assert r.wait_mode == "react" and r.react


def test_hazard_motion_matches_boxes_by_id_and_skips_changed_pieces():
    from mpc.future import FutureWorld
    fw = FutureWorld()
    fw.observe(1, [_hz(100.0), _hz(200.0)], [], [], None, [], 16.667)
    assert fw.hazard_motion() == (0.0, None)                          # nothing before
    fw.observe(2, [_hz(100.0), _hz(200.3)], [], [], None, [], 16.667)
    d, hid = fw.hazard_motion()
    assert hid == "h1" and abs(d - 0.3) < 1e-9
    fw.observe(3, [_hz(150.0)], [], [], None, [], 16.667)              # a piece went: not comparable
    assert fw.hazard_motion() == (0.0, None)
    fw.observe(4, [_hz(150.0), _hz(400.0)], [], [], [True, False], [], 16.667)   # another keep: not comparable
    assert fw.hazard_motion() == (0.0, None)
    fw.new_wave()
    fw.observe(1, [_hz(999.0)], [], [], [True, False], [], 16.667)    # a new wave's first frame
    assert fw.hazard_motion() == (0.0, None)


# ---- PlannerPool ---------------------------------------------------------------------------------------------
class _Child:
    """A PlannerProcess stand-in: every send recorded as (name, args, keep, kw); replies at once."""

    def __init__(self, threads=2):
        self.sent, self.next_id, self.replies, self.events = [], 0, {}, []
        self.proc = types.SimpleNamespace(pid=1000 + id(self) % 1000)
        self.cancels = []

    def send(self, name, *a, keep=False, **kw):
        mid = self.next_id
        self.next_id += 1
        self.sent.append((name, a, keep, kw))
        res = {"plan_ahead": ("plan@" + str(mid), 1.0, 2.0, 0.0, 0.0, 0.0),
               "plan_now": ("now@" + str(mid), None)}.get(name)
        if keep:
            self.replies[mid] = (True, res)
        else:
            self.events.append((name, True, res))
        return mid

    def ready(self, mid):
        return mid in self.replies

    def result(self, mid):
        return self.replies.pop(mid)

    def cancel_begin(self):
        self.cancels.append("begin")

    def cancel_end(self, mid):
        self.cancels.append(("end", mid))
        self.replies.pop(mid, None)
        self.send("discard", mid)

    def take_events(self):
        ev, self.events = self.events, []
        return ev

    def close(self):
        self.sent.append(("close", (), False, {}))


def _pool(n=3):
    from plan.worker import PlannerPool
    return PlannerPool(n, threads=2, make=_Child)


def _names(child):
    return [s[0] for s in child.sent]


def test_pool_broadcasts_state_calls_in_order_and_uses_the_primarys_replies():
    pool = _pool(3)
    a, b, c = pool.procs
    pool.call("init", "model", "lv", "field", {}, None, "tube")
    pool.call("set_level", "lv2", "field2", "tube")
    pool.send("frame_level", {"tc": [1]})
    pool.send("observe", 0, 5, 1.0, 2.0, 0, False, cause=None, held=0)
    pool.send("played", 0, 5, 3)
    pool.call("moves_learn", True)
    pool.call("moves_transient", ["1-1"])
    pool.call("set_model", "m2")
    pool.call("set_field", "f3", "bounds")
    pool.call("finished", 0, 99)
    pool.call("new_wave")
    pool.call("ping")
    order = ["init", "set_level", "frame_level", "observe", "played", "moves_learn", "moves_transient",
             "set_model", "set_field", "finished", "new_wave", "ping"]
    assert _names(a) == _names(b) == _names(c) == order
    assert [s[1] for s in a.sent] == [s[1] for s in b.sent] == [s[1] for s in c.sent]
    assert b.sent[3][3] == {"cause": None, "held": 0}                 # keywords too
    assert all(not s[2] for s in b.sent + c.sent)                     # the others' replies are never waited for
    assert [s[2] for s in a.sent] == [True, True, False, False, False] + [True] * 7
    assert pool.take_events() == [("frame_level", True, None), ("observe", True, None), ("played", True, None)]
    assert not b.events and not c.events                              # drained, dropped


def test_pool_primary_only_calls_and_errors_of_the_others():
    pool = _pool(2)
    a, b = pool.procs
    pool.call("moves_save")
    pool.call("surprise")
    assert _names(a) == ["moves_save", "surprise"] and not b.sent
    b.events.append(("observe", False, "ValueError: boom"))
    b.events.append(("observe", True, [{"kind": "position"}]))       # an event of the other: dropped
    assert pool.take_events() == [("observe (planner 1)", False, "ValueError: boom")]
    assert pool.proc is a.proc and pool.n == 2


def test_pool_plans_and_adopts_where_they_are_committed():
    """plan_now runs on the primary and is adopted by the others; plan_ahead runs on `to`, its commit goes there
    and every other process adopts the plan (at that place in its call order); a discard adopts nothing."""
    pool = _pool(3)
    a, b, c = pool.procs
    plan, trial = pool.call("plan_now", "row", 8, agent=0)
    assert plan == "now@0" and _names(a) == ["plan_now"]
    assert b.sent == c.sent == [("adopt", ("now@0",), False, {})]
    mid = pool.send("plan_ahead", "row", "prefix", keep=True, to=1)
    assert pool.owner(mid) == 1 and _names(b)[-1] == "plan_ahead" and _names(a) == ["plan_now"]
    assert pool.ready(mid)
    ok, res = pool.result(mid)
    assert ok and res[0] == "plan@1"
    pool.send("played", 0, 9, 2)
    pool.call("commit", mid)
    assert b.sent[-1][:3] == ("commit", (1,), True)                   # its own id there
    assert _names(a) == ["plan_now", "played", "adopt"] and a.sent[-1][1] == ("plan@1",)
    assert _names(c)[-2:] == ["played", "adopt"] and c.sent[-1][1] == ("plan@1",)
    mid = pool.send("plan_ahead", "row", "prefix", keep=True, to=2)
    pool.result(mid)
    pool.send("discard", mid)
    assert _names(c)[-1] == "discard" and _names(a)[-1] == "adopt" and _names(b)[-1] == "commit"
    with pytest.raises(KeyError):
        pool.send("commit", mid)                                      # gone


def test_pool_cancel_many_asks_every_process_before_waiting():
    pool = _pool(3)
    a, b, c = pool.procs
    m1 = pool.send("plan_ahead", "r", "p", keep=True, to=1)
    m2 = pool.send("plan_ahead", "r", "p", keep=True, to=2)
    order = []
    for ch in (b, c):
        ch.cancel_begin = (lambda ch=ch: order.append(("begin", ch)))
        ch.cancel_end = (lambda mid, ch=ch: order.append(("end", ch)))
    pool.cancel_many([m1, m2])
    assert order == [("begin", b), ("begin", c), ("end", b), ("end", c)]
    assert not pool.route and not pool.plans


# ---- Planner.adopt: the same deferred effects in another planner -----------------------------------------------
def _norm(x, depth=0):
    """Planner state as plain data, for equality across planners."""
    import dataclasses
    from plan.explain import Plan
    if depth > 12:
        return repr(x)
    if isinstance(x, Plan):
        return {"plan": x.to_dict(), "route_trial": _norm(getattr(x, "route_trial", None), depth + 1)}
    if isinstance(x, dict):
        return {repr(k): _norm(v, depth + 1) for k, v in sorted(x.items(), key=lambda kv: repr(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_norm(v, depth + 1) for v in x]
    if isinstance(x, set):
        return sorted(repr(v) for v in x)
    if isinstance(x, np.ndarray):
        return ["nd", x.tolist()]
    if isinstance(x, np.generic):
        return x.item()
    if dataclasses.is_dataclass(x):
        return _norm(dataclasses.asdict(x), depth + 1)
    return x


def _state_of(pl):
    return {"prev": _norm(pl.prev), "surprise": _norm(vars(pl.surprise)), "using": _norm(pl.route.using),
            "trials": _norm(vars(pl.trials)), "moves": _norm({k: v for k, v in vars(pl.moves).items() if k != "path"}),
            "memory": _norm(vars(pl.memory)), "long": _norm(pl.long), "stalls": _norm(pl.stalls)}


def _sent(plan):
    """The plan as another process gets it (plan.worker.plain_plan through a pipe)."""
    from plan.worker import plain_plan
    return pickle.loads(pickle.dumps(plain_plan(plan)))


@needs_model
def test_adopt_in_another_planner_gives_the_same_state():
    """A plan committed in planner A (commit_fn) and adopted in B from its pickled copy leaves both with the same
    previous plan, surprise predictions, best-route use, route trials, move memory, failure memory and planning
    caches; and the other way round. Covered: a route trial (route_using drop, trials.start), the best route
    (route_using set), a kept plan (no surprise.add) and the move memory's charges."""
    from mpc.candidates import act
    from plan.planner import MoveMemory, Planner
    lv, field = _flat()
    pls = [Planner(MODEL, lv, field, budget_ms=None, moves=MoveMemory()) for _ in range(2)]
    x0, n = 10 * T + 10, 120
    for pl in pls:
        pl.route.best = {"finish": 400, "f": np.arange(100, 100 + n), "x": np.full(n, x0), "y": np.full(n, 46.0),
                         "vx": np.zeros(n), "vy": np.zeros(n), "a": np.full(n, act(0))}
        pl.new_wave()
    worlds, frames = _inputs(pls[0].future_frames)
    assert _state_of(pls[0]) == _state_of(pls[1])
    modes = []
    for k, (src, frame) in enumerate([(0, 100), (0, 108), (1, 116), (1, 124), (0, 132)]):
        a, b = pls[src], pls[1 - src]
        p = a.plan(_standing(), frame, worlds, frames, defer=True)
        modes.append((p.mode, p.from_prev))
        q = _sent(p)
        assert not hasattr(q, "commit_fn") and q._effects is not None
        p.commit_fn()
        b.adopt(q)
        sa, sb = _state_of(a), _state_of(b)
        for key in sa:
            assert sa[key] == sb[key], (k, key)
    assert modes[0] == ("route-trial", False) and modes[1] == ("best-route", False) and modes[2][1], modes
    assert pls[0].moves.wave and pls[0].prev[0].frame == pls[1].prev[0].frame == 132


@needs_model
def test_planner_pool_adopts_across_real_processes():
    """Two real planner processes: a plan made now on the primary is adopted by process 1 (its plan made ahead has
    a previous plan to keep: switch is not None); a plan made ahead in process 1 and committed there is adopted by
    the primary: both then make the same next decision from the same state; no errors."""
    from plan.commit import ErrorTube
    from plan.worker import PlannerPool, TubeFn
    lv, field = _flat()
    pool = PlannerPool(2, threads=1)
    try:
        pool.call("init", MODEL, lv, field, {"budget_ms": None}, None, TubeFn(ErrorTube(), 1.5, 3.0))
        H = 48 + 45 + 8
        worlds, frames = _inputs(H)
        row = _standing()
        p0, _ = pool.call("plan_now", row, 0, worlds, frames, 16.667, agent=0)   # adopted by process 1
        mid = pool.send("plan_ahead", row, np.asarray(p0.actions[:8], np.int64), worlds, frames, 8, 8, 16.667, 0, 0,
                        keep=True, to=1)
        assert pool.owner(mid) == 1
        ok, res = pool.result(mid)
        assert ok and res[0].frame == 8 and res[0].switch is not None    # process 1 had p0 as its previous plan
        pool.call("commit", mid)
        q1, _ = pool.procs[1].call("plan_now", row, 16, worlds, frames, 16.667, agent=0)   # process 1 alone
        p1, _ = pool.call("plan_now", row, 16, worlds, frames, 16.667, agent=0)
        for key in ("actions", "chosen", "from_prev", "switch", "score", "mode"):
            assert getattr(p1, key) == getattr(q1, key), key          # the primary adopted process 1's plan
        pool.call("ping")
        assert [e for e in pool.take_events() if not e[1]] == []
    finally:
        pool.close()
    assert not any(p.proc.is_alive() for p in pool.procs)


# ---- react with --planners 2: several plans made ahead (agent.Runner.route_pool) -------------------------------
class _FakePool(_FakeProc):
    """A pool of 2 as agent.Runner uses it: calls recorded with the process (`to`), per-id readiness and results."""

    def __init__(self):
        super().__init__()
        self.to, self.ready_ids, self.res_by, self.cancelled = {}, set(), {}, []

    def send(self, name, *a, to=None, **kw):
        mid = super().send(name, *a)
        if name == "plan_ahead":
            self.to[mid] = to
        return mid

    def ready(self, mid):
        return mid in self.ready_ids

    def result(self, mid):
        self.taken.append(mid)
        return True, self.res_by[mid]

    def cancel_many(self, mids):
        self.cancelled.append(list(mids))
        self.calls.append(("cancel_many", list(mids)))


def _pool_react(monkeypatch, plan=None):
    """A Runner stub with react and --planners 2: agent 0 plays a committed plan made at 96 (actions k % 5, due
    104), no job in flight."""
    import agent
    class NoBoxes:                                  # the deadly boxes: none known (a job's fboxes[d:] is None)
        def __getitem__(self, k):
            return None
    monkeypatch.setattr(agent, "future_inputs",
                        lambda fw, frame, n, dt, name, deaths, **kw: ([None] * n, NoBoxes(), 0))
    r = object.__new__(agent.Runner)
    r.async_on = r.react = r.nowait = True
    r.wait_kinds = agent.WAIT_KINDS["react"]
    r.react_lat = deque(maxlen=agent.REACT_WINDOW)
    r.planner = types.SimpleNamespace(f=4, future_frames=60, key_pickup=False,
                                      support=lambda xs, ys, keys=None: (None, np.ones(len(xs), bool),
                                                                         np.zeros(len(xs))))
    r.level = types.SimpleNamespace(name="test")
    r.fw = r.deaths = None
    r.dt_ms = 1000.0 / 60.0
    r.commit = Committer(replan=8, on=True)
    r.commit.danger_always = True
    p = plan or _committed()
    r.cur_plan = {0: p}
    r.commit.decided(0, p, p.frame)
    r.ahead_miss, r.async_lead, r.st = 0, 8, {}
    r.n_planners = 2
    r.proc = _FakePool()
    r.ajob, r.jobs = None, []
    r.events, r.decisions = [], []
    r.event = lambda kind, frame, **kw: r.events.append(dict(kw, kind=kind, frame=frame))
    r.on_plan = lambda plan, s, trig, ahead=None: r.decisions.append((plan, list(trig), ahead))
    return r


def _starts(r):
    """The plan_ahead calls: (d, target, process, prefix) each."""
    return [(c[5], c[6], r.proc.to[mid], list(c[2])) for mid, c in
            zip(sorted(r.proc.to), [c for c in r.proc.calls if c[0] == "plan_ahead"])]


def _two_jobs(monkeypatch):
    """pool_fill at 100: the timed job (due 104) for 112 on the primary, a speculative one for 116 on 1."""
    r = _pool_react(monkeypatch)
    r.pool_fill(0, 100, _state(), 0, 16.667)
    return r


def test_pool_first_job_then_a_speculative_one(monkeypatch):
    r = _two_jobs(monkeypatch)
    acts = r.cur_plan[0].actions
    st = _starts(r)
    assert [(d, t, k) for d, t, k, _ in st] == [(12, 112, 0), (16, 116, 1)]   # S = max(4, 12 // 2 -> 4) = 4
    assert st[0][3] == acts[4:16] and st[1][3] == acts[4:20]          # the committed plan's actions up to each
    j0, j1 = r.jobs
    assert not j0.spec and j1.spec and j1.target == 116 and r.st["ahead_spec_started"] == 1
    assert r.st["react_leads"] == [12]                                # a speculative job is not a lead
    r.pool_fill(0, 101, _state(), 0, 16.667)
    assert len(_starts(r)) == 2                                       # no free process: nothing more
    assert r.ajob is None


def _take_first(r, actions, x=100.5):
    """j0 (id 100) ready at its frame 112 with a plan whose first actions are `actions`; route at 112."""
    j0 = r.jobs[0]
    r.proc.ready_ids.add(j0.mid)
    r.proc.res_by[j0.mid] = (_ahead_plan(actions), 100.0, 50.0, 0.3, 0.0, 0.0)
    return r.route_triggers(0, 112, _state(x, 50.0), [], 0, 16.667)


def test_pool_commit_with_the_same_prefix_keeps_the_later_job(monkeypatch):
    r = _two_jobs(monkeypatch)
    acts = r.cur_plan[0].actions
    j1 = r.jobs[1]
    assert _take_first(r, acts[16:20]) == []                          # frames 112..115 as the old plan's
    assert r.st["ahead_used"] == 1 and r.jobs == [j1] and not r.proc.cancelled
    assert ("commit", 100) in r.proc.calls and r.decisions[0][2]["planner"] == 0
    assert list(r.react_lat) == [12] and r.lead_now() == 16           # its latency: L 16, S 8
    r.pool_fill(0, 112, _state(), 0, 16.667)                          # the primary is free: speculative after 116
    assert [(d, t, k) for d, t, k, _ in _starts(r)][-1] == (16, 128, 0)   # max(116 + 8, 112 + 16)


def test_pool_commit_with_another_prefix_restarts_the_later_job(monkeypatch):
    r = _two_jobs(monkeypatch)
    acts = r.cur_plan[0].actions
    new = [(a + 1) % 5 for a in acts[16:20]]
    j1 = r.jobs[1]
    assert _take_first(r, new) == []
    assert r.proc.cancelled == [[j1.mid]] and r.st["ahead_spec_cancelled"] == 1
    d, t, k, prefix = _starts(r)[-1]
    assert (d, t, k) == (4, 116, 1) and prefix == new                 # same target, the new plan's actions
    (j,) = r.jobs
    assert j.proc == 1 and j.target == 116 and not j.spec and j.base is r.cur_plan[0]
    ev = [e for e in r.events if e["kind"] == "ahead dropped"]
    assert len(ev) == 1 and "not those of the plan committed for 112" in ev[0]["reason"]


def test_pool_route_trial_takes_no_decision_while_it_plays(monkeypatch):
    """j0's plan is a route trial of 12 frames (to 124): j1 (for 116, inside it) is cancelled, not restarted; the
    next job waits for the trial's end (Committer.due) and the speculative one after it is not before it either."""
    r = _two_jobs(monkeypatch)
    acts = r.cur_plan[0].actions
    j1 = r.jobs[1]
    j0 = r.jobs[0]
    r.proc.ready_ids.add(j0.mid)
    trial = _ahead_plan(acts[16:20])
    trial.trial_frames = 12
    r.proc.res_by[j0.mid] = (trial, 100.0, 50.0, 0.3, 0.0, 0.0)
    assert r.route_triggers(0, 112, _state(100.5, 50.0), [], 0, 16.667) == []
    assert r.cur_plan[0] is trial and not r.jobs and r.proc.cancelled == [[j1.mid]]
    assert r.st["ahead_spec_cancelled"] == 1 and not r.st.get("ahead_restarted")
    assert "within the route trial" in [e for e in r.events if e["kind"] == "ahead dropped"][-1]["reason"]
    n = len(_starts(r))
    for f in range(112, 116):                                        # due 124: not within L (16) yet at 112..
        r.pool_fill(0, f, _state(), 0, 16.667)
        if r.jobs:
            break
    assert all(t >= 124 for _, t, _, _ in _starts(r)[n:])
    assert len(r.jobs) == 2 and r.jobs[0].target >= 124 and r.jobs[1].target > r.jobs[0].target


def test_pool_drop_keeps_the_later_jobs(monkeypatch):
    r = _two_jobs(monkeypatch)
    j1 = r.jobs[1]
    assert _take_first(r, [], x=130.0) == []                          # off the tube: dropped
    assert r.st["ahead_dropped"] == 1 and r.jobs == [j1] and not r.proc.cancelled
    assert len(_starts(r)) == 2                                       # nothing restarted: j1 answers
    assert r.events[-1]["reason"].startswith("start off by")


def test_pool_kept_job_keeping_the_replaced_plan_is_dropped(monkeypatch):
    """j1 kept (its prefix matched), but its plan keeps the plan before j0's (from_prev) with other actions than
    j0's plan from 116 on: dropped when taken (it would play the replaced plan again)."""
    r = _two_jobs(monkeypatch)
    acts = r.cur_plan[0].actions
    p0 = acts[16:20] + [4] * 44                                       # j0's plan: 4 from frame 116 on
    assert _take_first(r, p0) == []
    j1 = r.jobs[0]
    r.proc.ready_ids.add(j1.mid)
    keep = _ahead_plan(acts[20:40])
    keep.frame, keep.from_prev = 116, True
    r.proc.res_by[j1.mid] = (keep, 104.0, 50.0, 0.3, 0.0, 0.0)
    r.route_triggers(0, 116, _state(104.5, 50.0), [], 0, 16.667)
    assert r.st["ahead_used"] == 1 and r.st["ahead_dropped"] == 1
    assert "keeps the plan replaced" in [e for e in r.events if e["kind"] == "ahead dropped"][-1]["reason"]


def test_pool_event_cancels_every_job_then_reactive_and_staggered(monkeypatch):
    r = _two_jobs(monkeypatch)
    ids = [j.mid for j in r.jobs]
    assert r.route_triggers(0, 103, _state(), [SURPRISE], 0, 16.667) == []
    assert r.proc.cancelled == [ids] and r.st["ahead_restarted"] == 1 and r.st["ahead_spec_cancelled"] == 1
    (j,) = r.jobs
    assert j.reactive and j.proc == 0 and j.target == 115 and j.why == [SURPRISE] and not j.spec
    r.pool_fill(0, 103, _state(), 0, 16.667)
    d, t, k, prefix = _starts(r)[-1]
    assert (d, t, k) == (16, 119, 1) and prefix == r.cur_plan[0].actions[7:23]
    assert r.route_triggers(0, 104, _state(), [SURPRISE], 0, 16.667) == []      # a reactive job: absorbed
    assert r.st["ahead_absorbed"] == 1 and len(r.proc.cancelled) == 1


def test_pool_danger_drops_every_job_and_decides_now(monkeypatch):
    r = _two_jobs(monkeypatch)
    ids = [j.mid for j in r.jobs]
    assert r.route_triggers(0, 103, _state(), ["danger h3 f6"], 0, 16.667) == ["danger h3 f6"]
    assert not r.jobs and r.proc.cancelled == [ids] and r.st["ahead_dropped"] == 2
    assert r.st["ahead_spec_cancelled"] == 1 and ("ping",) in r.proc.calls


def test_pool_late_job_is_cancelled_and_the_next_taken(monkeypatch):
    """j0 never ready: REACT_LATE_MAX + 1 frames after 112 it is cancelled (its age a latency sample); j1, ready
    and with the played actions its own, is taken late in the same frame."""
    import agent
    r = _two_jobs(monkeypatch)
    acts = r.cur_plan[0].actions
    j0, j1 = r.jobs
    f = 112 + agent.REACT_LATE_MAX + 1
    r.proc.ready_ids.add(j1.mid)
    k = f - 116
    plan = _ahead_plan(acts[20:20 + k])
    plan.frame = 116
    r.proc.res_by[j1.mid] = (plan, 100.0, 50.0, 0.3, 60.0, 0.0)
    assert r.route_triggers(0, f, _state(100.0 + k + 0.5, 50.0, 60.0, 0.0), [], 0, 16.667) == []
    assert r.st["ahead_late_dropped"] == 1 and r.proc.cancelled == [[j0.mid]]
    assert r.st["ahead_used"] == 1 and r.st["ahead_late_used"] == 1 and r.st["ahead_spec_used"] == 1
    assert list(r.react_lat)[0] == f - 100 and r.cur_plan[0] is plan and not r.jobs


def test_one_planner_stays_route_react(monkeypatch):
    """--planners 1: the old single job (self.ajob, no process named), never self.jobs."""
    r = _pool_react(monkeypatch)
    r.n_planners = 1
    assert not r.pool_react()
    assert r.route_triggers(0, 104, _state(), ["interval"], 0, 16.667) == []
    assert r.ajob is not None and not r.jobs and list(r.proc.to.values()) == [None]
