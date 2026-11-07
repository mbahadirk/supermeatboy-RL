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


def old_mode_run(tmp: str, extra=()) -> dict:
    """The recorded 1-13 wave of tests/test_agent.py through the runner with --replan 2 (no planner deadline,
    1 thread: the result does not depend on the machine's load). extra: more flags. Returns the decisions
    [agent, frame, chosen, actions] and the replies per frame."""
    import agent
    import test_agent as TA
    from plan.planner import Planner

    class NoDeadline(Planner):
        def __init__(self, *a, **kw):
            kw["budget_ms"] = None
            super().__init__(*a, **kw)
    args = agent.build_parser().parse_args(
        ["--agents", "2", "--replan", "2", "--budget-ms", "60", "--model", MODEL, "--name", os.path.join(tmp, "run"),
         "--death-memory", os.path.join(tmp, "deaths.json"), "--telemetry", "off", "--threads", "1",
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
    now = old_mode_run(str(tmp_path), ["--triggers", "off"])
    assert [d[:2] for d in now["decisions"]] == [d[:2] for d in gold["decisions"]]       # the schedule
    assert now["decisions"] == gold["decisions"]                                          # and the decisions
    assert now["replies"] == gold["replies"]
    assert all(p["trigger"] in (["start"], ["interval"]) for p in now["plans"])


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
    thr = 1.5 * ERR90_AT[4]
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


if __name__ == "__main__" and "--write" in sys.argv:
    from test_planner_golden import model_digest
    import torch
    with tempfile.TemporaryDirectory() as tmp:
        out = old_mode_run(tmp)
    out.pop("plans")
    torch.set_num_threads(2)
    os.makedirs(os.path.dirname(OLD), exist_ok=True)
    with open(OLD, "w", encoding="utf-8") as f:
        json.dump({"about": "agent.py decisions with --replan 2 before the replan triggers (tests/test_commit.py)",
                   "model_sha256": model_digest(), **out}, f)
    print(f"wrote {OLD}: {len(out['decisions'])} decisions")
