"""Commitment to the chosen plan (agent.py replan triggers, plan/commit.py; the planner's switch rule):
the old replan mode reproduces the old decision schedule and decisions, the triggers fire when they should,
and a kept plan is registered once.

    python tests/test_commit.py --write      # re-record the old runner's schedule (only with the old code)
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(HERE, "tests")
sys.path.insert(0, HERE)
sys.path.insert(0, TESTS)

DATA = os.path.join(HERE, "runs", "real_20261102-165351", "wm_data")
MODEL = os.path.join(HERE, "runs", "wm_locked", "wm_model.pt")
OLD = os.path.join(HERE, "tests", "data", "runner_old_schedule.json")
need_data = pytest.mark.skipif(not (os.path.isdir(DATA) and os.path.exists(MODEL)), reason="recorded data not here")


def old_mode_run(tmp: str, extra=(), tol="default") -> dict:
    """The recorded 1-13 wave of tests/test_agent.py through the runner with --replan 2 (no planner deadline,
    1 thread: the result does not depend on the machine's load). extra: more flags. Returns the decisions
    [agent, frame, chosen, actions] and the replies per frame. tol: the planner's collision guard tolerance
    (default: the planner's; None: no guard)."""
    import agent
    import test_agent as TA
    from plan.planner import Planner

    class NoDeadline(Planner):
        def __init__(self, *a, **kw):
            kw["budget_ms"] = None
            if tol != "default":
                kw["collide_tol"] = tol
            super().__init__(*a, **kw)
    args = agent.build_parser().parse_args(
        ["--agents", "2", "--replan", "2", "--budget-ms", "60", "--model", MODEL, "--name", os.path.join(tmp, "run"),
         "--death-memory", os.path.join(tmp, "deaths.json"), "--move-memory", os.path.join(tmp, "moves.json"), "--telemetry", "off", "--threads", "1",
         "--best-times", os.path.join(tmp, "best_times.json"), *extra])
    old_planner = agent.Planner
    agent.Planner = NoDeadline
    try:
        r = agent.Runner(args, telemetry=None)
        r.script = TA._FakeScript()
        acts, _, _, _ = TA._drive_wave(r)
        r.close()
    finally:
        agent.Planner = old_planner
    plans = [json.loads(l) for l in open(os.path.join(tmp, "run", "plans.jsonl"), encoding="utf-8")]
    return {"decisions": [[p["agent"], p["frame"], p["chosen"], p["actions"]] for p in plans],
            "replies": [a["a"] for a in acts], "plans": plans}


@need_data
@pytest.mark.skipif(not os.path.exists(OLD), reason="no record of the old runner")
def test_old_replan_mode_reproduces_the_old_schedule(tmp_path):
    from test_planner_golden import model_digest
    gold = json.load(open(OLD, encoding="utf-8"))
    if gold["model_sha256"] != model_digest():
        pytest.skip("the world model changed since the record was made")
    now = old_mode_run(str(tmp_path), ["--triggers", "off", "--hysteresis", "0"], tol=None)   # no collision guard
    assert [d[:2] for d in now["decisions"]] == [d[:2] for d in gold["decisions"]]       # the schedule
    assert now["decisions"] == gold["decisions"]                                          # and the decisions
    assert now["replies"] == gold["replies"]
    assert all(p["trigger"] in (["start"], ["interval"]) for p in now["plans"])


def _clipped(p) -> int:
    return sum(int((p["counts"].get(s) or {}).get("clipped") or 0) for s in ("stage1", "stage2", "final")) +         int((p["counts"].get("final") or {}).get("completion_clipped") or 0)


@need_data
@pytest.mark.skipif(not os.path.exists(OLD), reason="no record of the old runner")
def test_collision_guard_changes_only_clipped_decisions(tmp_path):
    """With the collision guard (plan/planner.py collide) the old mode's decisions are the old record except
    where it clipped a candidate of that decision, or a decision of the same agent before it changed (the
    previous plan is a candidate). The guarded run is recorded too ("guarded")."""
    from test_planner_golden import model_digest
    gold = json.load(open(OLD, encoding="utf-8"))
    if gold["model_sha256"] != model_digest() or "guarded" not in gold:
        pytest.skip("the world model changed since the record was made, or no guarded record")
    now = old_mode_run(str(tmp_path), ["--triggers", "off", "--hysteresis", "0"])
    assert now["decisions"] == gold["guarded"]["decisions"]
    assert now["replies"] == gold["guarded"]["replies"]
    changed_before = set()
    for d, old, p in zip(now["decisions"], gold["decisions"], now["plans"]):
        if d != old:
            assert _clipped(p) > 0 or d[0] in changed_before, d[:3]
            changed_before.add(d[0])


# ---- the triggers (plan/commit.py) on stub plans ---------------------------------------------------------------
from types import SimpleNamespace                       # noqa: E402

from plan import commit as CM                           # noqa: E402
from plan import safety as S                            # noqa: E402
from plan.explain import ERR90_AT                       # noqa: E402


class _Terrain:
    """Ground below y = 0.5, a wall right of x = 1000."""

    def support(self, x, y, plats=None):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return y < 0.5, np.where(x > 1000, 1, 0).astype(np.int8)


def _stub_planner():
    return SimpleNamespace(terrain=_Terrain(), lv=SimpleNamespace(plats=[]), bounds=None)


def _stub_plan(frame=100, n=48, first=12, path=None, goal=None, keys=()):
    path = path if path is not None else [[2.0 * k, 0.0] for k in range(n + 1)]      # runs right on the ground
    return SimpleNamespace(frame=frame, actions=[2] * n, path=path, first_move_frames=first, goal_touch=goal,
                           key_touches=list(keys))


def _boxes(n, box=None, at=()):
    """n + 1 frames of deadly boxes: `box` present at the frames in `at`."""
    out = [None]
    for k in range(1, n + 1):
        b = [box] if box is not None and k in at else []
        out.append(S.FrameBoxes(np.array(b, float).reshape(-1, 5), np.zeros(len(b), np.int8),
                                [f"h{j}" for j in range(len(b))]))
    return out


def test_triggers_start_interval_move_end_short():
    pl = _stub_planner()
    c = CM.Committer(replan=8, tube=1.5, on=True)
    assert c.triggers(0, 100, 0.0, 0.0, planner=pl) == ["start"]
    c.decided(0, _stub_plan(first=6), 100, _boxes(48), pl)
    on = lambda f, **kw: c.triggers(0, f, 2.0 * (f - 100), 0.0, planner=pl, fboxes_fn=lambda: _boxes(48), **kw)
    assert on(101) == [] and on(105) == []
    assert on(106) == ["move end"]                                  # the first move's 6 frames were played
    assert on(102, tiles=True) == ["start: tiles"]
    c.decided(0, _stub_plan(first=0), 100, _boxes(48), pl)          # only padding: no move end
    assert on(107) == [] and on(108) == ["interval"]
    c.decided(0, _stub_plan(n=20, first=0), 100, _boxes(20), pl)
    assert on(104) == [] and on(105) == ["short"]                   # 15 of 20 frames left
    off = CM.Committer(replan=2, on=False)                         # the old rule: the interval only
    off.decided(0, _stub_plan(first=1), 100)
    assert off.triggers(0, 101, 999.0, 999.0) == [] and off.triggers(0, 102, 0.0, 0.0) == ["interval"]


def test_triggers_surprise_support_missed():
    pl = _stub_planner()
    c = CM.Committer(replan=8, tube=1.5, on=True)
    c.decided(0, _stub_plan(first=0, goal={"frame": 2, "x": 4.0, "y": 0.0, "members": 5},
                            keys=[{"frame": 1, "key": 1, "x": 2.0, "y": 0.0}]), 100, None, pl)
    thr = max(1.5 * ERR90_AT[4], CM.MIN_TUBE)
    assert c.tube.at(4) == ERR90_AT[4] and c.tube.at(6) == pytest.approx((ERR90_AT[4] + ERR90_AT[8]) / 2)
    tr = c.triggers(0, 104, 8.0 + thr + 0.5, 0.0, keys=0b10, planner=pl)
    assert tr[0] == f"surprise +4 err {thr + 0.5:.1f} > {thr:.1f}" and "missed: princess f2" in tr
    assert not any(t.startswith("missed: key") for t in tr)          # key 1 is held
    assert "missed: key 1 f1" in c.triggers(0, 104, 8.0, 0.0, keys=0, planner=pl)
    c.decided(0, _stub_plan(first=0), 100, None, pl)
    assert c.triggers(0, 103, 6.0, 0.0, planner=pl) == []
    assert c.triggers(0, 103, 6.0, 1.0, planner=pl) == ["support: not on the ground +3"]     # 1 unit up: in the air
    # predicted: rises from the ground at frame 5 and lands at frame 20; landing early by more than a frame fires
    path = [[0.0, 0.0 if k < 5 or k >= 20 else 30.0] for k in range(49)]
    c = CM.Committer(replan=48, tube=1e9, on=True)
    c.decided(0, _stub_plan(path=path, first=0), 100, None, pl)
    assert c.triggers(0, 119, 0.0, 0.0, planner=pl) == []            # one frame early: within the tolerance
    assert c.triggers(0, 117, 0.0, 0.0, planner=pl) == ["support: landed early +17"]
    path = [[1001.0, 0.0 if k < 5 else 30.0] for k in range(49)]     # predicted on the right wall from frame 5
    c.decided(0, _stub_plan(path=path, first=0), 100, None, pl)
    assert c.triggers(0, 110, 990.0, 30.0, planner=pl) == ["support: lost the wall +10"]


def test_surprise_tube_has_a_minimum_width():
    pl = _stub_planner()
    c = CM.Committer(replan=48, tube=1.5, on=True)
    c.decided(0, _stub_plan(first=0), 100, None, pl)                  # path x = 2k on the ground
    assert 1.5 * c.tube.at(1) < CM.MIN_TUBE
    assert c.triggers(0, 101, 2.0 + 2.0, 0.0, planner=pl) == []      # 2 units off at +1: inside the minimum tube
    assert c.triggers(0, 101, 2.0 + 3.5, 0.0, planner=pl) == ["surprise +1 err 3.5 > 3.0"]
    c.tube.add({1: [0.1] * CM.MIN_SAMPLES})                          # the run's own +1 error: tiny, still the minimum
    assert c.tube.table[1][1] == "run" and c.triggers(0, 101, 2.0 + 2.0, 0.0, planner=pl) == []
    thr8 = 1.5 * c.tube.at(8)                                         # wider than the minimum: unchanged
    assert thr8 > CM.MIN_TUBE
    assert c.triggers(0, 108, 16.0 + thr8 + 0.5, 0.0, planner=pl) == [f"surprise +8 err {thr8 + 0.5:.1f} > {thr8:.1f}"]


def test_triggers_danger():
    pl = _stub_planner()
    c = CM.Committer(replan=48, tube=1e9, on=True)
    box = (40.0, 0.0, 2.0, 2.0, 0.0)                                 # on the path at frame 20 (x = 40)
    c.decided(0, _stub_plan(first=0), 100, _boxes(48), pl)            # nothing near at the decision
    on = lambda f, fb: c.triggers(0, f, 2.0 * (f - 100), 0.0, planner=pl, fboxes_fn=lambda: fb)
    assert on(105, _boxes(48)) == []
    # the hazard appears where the path passes 15 frames from now (absolute frame 120)
    assert on(105, _boxes(48, box, at=range(13, 18))) == ["danger h0 f13"]
    # far away (40 frames from now): soft risk only
    far = (2.0 * 45, 0.0, 2.0, 2.0, 0.0)
    assert on(105, _boxes(48, far, at=range(38, 43))) == ["danger: risk 0.00 -> 1.00"]
    # a plan chosen while already vetoed by that contact does not fire again for it, only when it comes earlier
    box2 = (30.0, 0.0, 2.0, 2.0, 0.0)                                # reached at frame 12 (absolute 112)
    c.decided(0, _stub_plan(first=0), 100, _boxes(48, box2, at=range(10, 16)), pl)
    assert c.cur[0].base_contact == 112
    assert on(105, _boxes(48, box2, at=range(5, 11))) == []            # the same contact, still at 112
    assert on(105, _boxes(48, (20.0, 0.0, 2.0, 2.0, 0.0), at=range(1, 6))) == ["danger h0 f2"]


def test_path_check_matches_the_planners_rule():
    rng = np.random.default_rng(1)
    for _ in range(200):
        K = int(rng.integers(5, 49))
        xs, ys = np.cumsum(rng.normal(0, 6, K + 1)), np.cumsum(rng.normal(0, 6, K + 1))
        frames = [None]
        for k in range(K):
            m = int(rng.integers(0, 5))
            b = np.c_[rng.normal(0, 40, (m, 2)), rng.uniform(2, 10, (m, 2)), rng.uniform(0, 3, m)]
            frames.append(S.FrameBoxes(b, np.zeros(m, np.int8), [f"h{j}" for j in range(m)]))
        bounds = (-60.0, 60.0, -60.0, 60.0)
        s = S.check(xs[None, None], ys[None, None], frames, bounds)
        cf, hid, risk = CM.path_check(xs, ys, frames, bounds)
        sw = s.sweep
        assert cf == int(sw.contact_frame[0, 0]) and risk == pytest.approx(float(s.soft_risk[0]), abs=1e-12)
        if cf > 0:
            assert hid == (S.OUT_ID if sw.contact_kind[0, 0] == S.KIND_OUT else sw.ids[sw.contact_box[0, 0]])


def test_error_tube_takes_the_runs_errors():
    t = CM.ErrorTube()
    assert t.summary()["8"] == [ERR90_AT[8], "ERR90_AT", 0]
    t.refresh({8: list(np.arange(CM.MIN_SAMPLES))})                  # the current wave alone: enough
    assert t.table[8][1] == "run" and t.at(8) == pytest.approx(np.percentile(np.arange(CM.MIN_SAMPLES), 90))
    t.refresh()                                                      # not kept without add()
    assert t.table[8][1] == "ERR90_AT"
    t.add({1: [1.0] * CM.MIN_SAMPLES})
    assert t.at(1) == 1.0 and t.at(32) == dict(S.ERR90)[32] and t.at(100) == dict(S.ERR90)[48]
    assert CM.kind_of("surprise +8 err 14.2 > 11.0") == "surprise" and CM.kind_of("start: tiles") == "start"


@need_data
def test_runner_with_triggers_decides_less(tmp_path):
    out = old_mode_run(str(tmp_path), ["--replan", "8"])            # the new defaults, triggers on
    plans = out["plans"]
    assert plans and all(p["trigger"] for p in plans)
    assert all(CM.kind_of(t) in CM.KINDS for p in plans for t in p["trigger"])
    if os.path.exists(OLD):
        assert len(plans) < len(json.load(open(OLD, encoding="utf-8"))["decisions"])
    for i in (0, 1):
        fr = [p["frame"] for p in plans if p["agent"] == i]
        assert all(0 < b - a <= 8 for a, b in zip(fr, fr[1:]))      # at most --replan frames apart
    events = [json.loads(l) for l in open(tmp_path / "run" / "events.jsonl", encoding="utf-8")]
    wave = [e for e in events if e["kind"] == "wave"][-1]
    assert sum(wave["triggers"].values()) >= wave["decisions"] and wave["agent_frames"] > 0
    # the wave's commitment numbers
    assert wave["decisions_per_100"] == pytest.approx(100 * wave["decisions"] / wave["agent_frames"], abs=0.01)
    assert wave["switched"] + wave["kept"] <= wave["decisions"] and 0 <= wave["switch_rate"] <= 1
    assert 1 <= wave["gap_median"] <= 8 and wave["plan_ms_per_frame"] > 0 and wave["check_ms_per_frame"] >= 0
    assert set(wave["tube"]) == {"1", "4", "8", "16"}
    assert all(("switch" in p) for p in plans) and any(p["switch"] for p in plans)


# ---- the planner's switch rule ------------------------------------------------------------------------------
def _rule(h, total, succ=None, tg=None, veto=None, risk=None, best=None, p=0):
    from plan.planner import Planner
    n = len(total)
    total = np.asarray(total, float)
    saf = SimpleNamespace(veto=np.asarray(veto if veto is not None else [False] * n),
                          veto_id=["h3"] * n, veto_frame=np.full(n, 9),
                          soft_risk=np.asarray(risk if risk is not None else [0.0] * n, float))
    succ = np.asarray(succ if succ is not None else [False] * n)
    tg = np.asarray(tg if tg is not None else [-1] * n)
    best = int(np.argmax(total)) if best is None else best
    return Planner.switch_rule(SimpleNamespace(hysteresis=h), best, p, total, succ, tg, saf)


def test_switch_rule():
    # totals are minus the estimated frames to the goal: the previous plan (index 0) against a slightly better
    # one: kept within the margin (frames), switched over it
    c, r = _rule(3.0, [-300.0, -298.0, -400.0])
    assert c == 0 and r["kept"] and r["reason"] == "kept: best other +2 frames < 3" and r["unit"] == "frames"
    c, r = _rule(3.0, [-300.0, -296.0])
    assert c == 1 and not r["kept"] and r["reason"] == "switched: +4 frames" and r["gain"] == pytest.approx(4.0)
    assert r["margin"] == 3.0
    c, r = _rule(3.0, [-300.0, -305.0])
    assert c == 0 and r["kept"] and r["reason"] == "kept: best other -5 frames"
    # hysteresis 0: always the best total (only described)
    assert _rule(0.0, [-300.0, -299.9])[0] == 1 and _rule(0.0, [-300.0, -300.0])[0] == 0
    # a vetoed previous plan never blocks a switch; a riskier one is dropped
    c, r = _rule(3.0, [-1e6 + 9, -1e6 + 12], veto=[True, True])
    assert c == 1 and r["reason"] == "switched: previous plan vetoed (h3 f9)"
    c, r = _rule(3.0, [-300.0, -299.0], risk=[0.5, 0.2])
    assert c == 1 and r["reason"].startswith("switched: previous plan's risk 0.50")
    # successes: by the touch frame, 3 frames margin; a success always beats a non-success
    S5 = 1e5
    c, r = _rule(3.0, [S5 - 20, S5 - 18], succ=[True, True], tg=[20, 18])
    assert c == 0 and r["unit"] == "frames" and r["reason"] == "kept: best other +2 frames < 3"
    c, r = _rule(3.0, [S5 - 20, S5 - 16], succ=[True, True], tg=[20, 16])
    assert c == 1 and r["reason"] == "switched: +4 frames"
    c, r = _rule(50.0, [-300.0, S5 - 40], succ=[False, True], tg=[-1, 40])
    assert c == 1 and r["reason"] == "switched: reaches the princess, the previous plan does not"
    assert _rule(3.0, [1.0], p=0) == (0, {"kept": True, "gain": None, "margin": None, "unit": "frames",
                                          "reason": "kept: no other candidate"})
    assert _rule(3.0, [1.0, 2.0], p=-1) == (1, None)


@pytest.mark.skipif(not os.path.exists(MODEL), reason="no world model")
def test_kept_plan_is_registered_once():
    import torch
    from test_planner_golden import _flat, _inputs
    from plan.planner import state_row
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        lv, pl = _flat()
        worlds, frames = _inputs(pl.H)
        st = state_row(210.0, 46.0, 0.0, 0.0, 0, 5, 0, 0)
        p0 = pl.plan(st, 100, worlds, frames)
        assert p0.switch is None and len(pl.surprise.plans[0]) == 1
        pend = pl.memory.pending.get(0)
        pl.hysteresis = 1e9                                          # nothing beats the committed plan
        for k, a in enumerate(p0.actions[:2]):
            pl.played(0, 100 + k, a)
        p1 = pl.plan(st, 102, worlds, frames)
        assert p1.from_prev and p1.switch["kept"] and p1.switch["reason"].startswith("kept: best other")
        assert p1.actions[:46] == p0.actions[2:]                     # the same committed actions
        assert [e[0] for e in pl.surprise.plans[0]] == [100]          # still only its adoption's predictions
        if pend is not None:                                         # the move's expectation is the original one
            assert pl.memory.pending[0] is pend
        assert all("within the switch margin" in a["reason"] for a in p1.alternatives
                   if a.get("total") is not None and a["total"] > p1.score["total"])
        pl.hysteresis = 0.0
        p2 = pl.plan(st, 104, worlds, frames)                        # the best total again
        assert (p2.switch["kept"]) == p2.from_prev
        if not p2.from_prev:
            assert [e[0] for e in pl.surprise.plans[0]] == [100, 104]
    finally:
        torch.set_num_threads(threads)


@need_data
def test_closed_loop_harness(tmp_path):
    """plan/offline.py closed_loop on recorded 1-13: the model plays itself; the old mode decides every 2 frames."""
    import torch
    from plan.offline import closed_loop, closed_loop_text
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        old = closed_loop(DATA, "1-13", MODEL, frames=12, replan=2, triggers=False, hysteresis=0.0, budget=None)
        new = closed_loop(DATA, "1-13", MODEL, frames=12, budget=None)
    finally:
        torch.set_num_threads(threads)
    assert old["frames"] == 12 and old["decisions"] == 6 and [p["frame"] for p in old["plans"]] == list(range(0, 12, 2))
    assert old["triggers"] == {"start": 1, "interval": 5}
    assert 1 <= new["decisions"] < old["decisions"] and new["plans"][0]["trigger"] == ["start"]
    assert new["mode"]["hysteresis"] == 3.0 and new["plan_ms_per_frame"] > 0
    assert "the model plays itself, not the game" in closed_loop_text(new)


if __name__ == "__main__" and "--write" in sys.argv:
    from test_planner_golden import model_digest
    import torch
    flags = ["--triggers", "off", "--hysteresis", "0"]           # the old replan mode (as the tests run it)
    with tempfile.TemporaryDirectory() as tmp:
        out = old_mode_run(tmp, flags, tol=None)
    with tempfile.TemporaryDirectory() as tmp:
        on = old_mode_run(tmp, flags)
    out.pop("plans")
    on.pop("plans")
    out["guarded"] = on
    torch.set_num_threads(2)
    os.makedirs(os.path.dirname(OLD), exist_ok=True)
    with open(OLD, "w", encoding="utf-8") as f:
        json.dump({"about": "agent.py decisions with --replan 2 --triggers off --hysteresis 0 (the old replan mode; "
                            "tests/test_commit.py), re-recorded with the step-4 score in frames",
                   "model_sha256": model_digest(), **out}, f)
    print(f"wrote {OLD}: {len(out['decisions'])} decisions")
