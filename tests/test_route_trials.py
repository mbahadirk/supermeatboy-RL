"""Route trials (plan.planner.RouteTrials): a plan much better than the best route is played once, exactly."""
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plan.commit import Committer                                          # noqa: E402
from plan.planner import (TRIAL_RISK_SLACK, TRIAL_SPOT_RADIUS, RouteTrials, move_family,  # noqa: E402
                          route_trial_choice)


def test_choice_takes_the_best_unvetoed_untried_candidate_beyond_the_gain():
    est = np.array([200.0, 170.0, 150.0, 185.0, 140.0])
    veto = np.array([False, False, False, False, True])
    keys = ["route", "a", "b", "c", "d"]
    none = {"spot": 0, "family": 0, "risk": 0}
    assert route_trial_choice(est, veto, 0, 20.0, keys, {}) == (2, none)          # 140 is vetoed
    assert route_trial_choice(est, veto, 0, 20.0, keys, {"b": {}})[0] == 1   # b tried: the next one (30 > 20)
    assert route_trial_choice(est, veto, 0, 20.0, keys, {"a": {}, "b": {}})[0] is None   # c beats it by 15 only
    assert route_trial_choice(est, veto, 0, 60.0, keys, {})[0] is None
    assert route_trial_choice(est, veto, 0, 0.0, keys, {})[0] is None        # off


@pytest.mark.parametrize("name, family", [
    ("walk L 8f", "walk L"), ("walk L 24f", "walk L"), ("run-jump R hold 6 after 4f", "run-jump R"),
    ("run-jump R hold max after 8f", "run-jump R"), ("jump up hold max", "jump up"), ("jump up hold 22", "jump up"),
    ("run wall-jump R hold 8", "run wall-jump R"), ("best route", "best route"), ("wait 16f", "wait"),
    ("fall drift L 12f", "fall drift L"), ("jump L hold 6 turn", "jump L turn")])
def test_move_family_drops_every_number_bearing_part(name, family):
    assert move_family(name) == family


# the spot memory: candidates 1..3 beat the route (0) by more than the gain, the best first
EST = np.array([200.0, 150.0, 160.0, 170.0])
VETO = np.zeros(4, bool)
KEYS = ["route", "a", "b", "c"]
FAMS = ["best route", "walk L", "jump up", "walk L"]


def _choose(fails, here=(100.0, 50.0, 0), soft=None):
    return route_trial_choice(EST, VETO, 0, 20.0, KEYS, {}, soft=soft, families=FAMS, here=here, fails=fails)


def test_a_failed_family_is_not_tried_again_near_its_spot():
    failed = [(130.0, 50.0, 0, "walk L")]                                # 30 units away, same mask
    assert _choose(failed) == (2, {"spot": 0, "family": 1, "risk": 0})  # walk L skipped: jump up
    assert _choose([(130.0, 50.0, 1, "walk L")])[0] == 1                # another key mask: allowed
    assert _choose([(100.0, 50.0 + TRIAL_SPOT_RADIUS + 1.0, 0, "walk L")])[0] == 1    # far: allowed
    assert _choose([(130.0, 50.0, 0, "run R")])[0] == 1                 # another family: allowed


def test_two_failures_close_the_spot():
    failed = [(130.0, 50.0, 0, "run R"), (90.0, 40.0, 0, "jump L")]
    assert _choose(failed) == (None, {"spot": 3, "family": 0, "risk": 0})
    assert _choose(failed, here=(300.0, 50.0, 0))[0] == 1               # elsewhere: open


def test_a_trial_riskier_than_the_route_is_skipped():
    soft = np.array([0.10, 0.16, 0.15, 0.0])
    assert _choose([], soft=soft) == (2, {"spot": 0, "family": 0, "risk": 1})       # 0.16 > 0.10 + 0.05
    assert _choose([], soft=np.array([0.10, 0.15, 0.0, 0.0]))[0] == 1               # at the slack: allowed
    assert TRIAL_RISK_SLACK == 0.05


def test_failed_outcomes_are_remembered_a_faster_one_is_not():
    tr = RouteTrials(20.0)
    spot = (100.0, 50.0, 0, "walk L")
    tr.start(0, 10, "k1", _path(), 12, {}, spot=spot)
    assert tr.finished(0, 300, faster=True)["outcome"] == "faster" and tr.fails == []
    tr.new_wave()
    tr.start(0, 10, "k2", _path(), 12, {}, spot=spot)
    ev = tr.finished(0, 400, faster=False)
    assert ev["outcome"] == "slower" and "_spot" not in ev and tr.fails == [spot]
    tr.new_wave()
    tr.start(0, 10, "k3", _path(), 12, {}, spot=(110.0, 50.0, 0, "jump up"))
    tr.observe(0, 12, 0.0, 0.0, dead=True, tube=lambda k: 3.0)
    tr.start(1, 10, "k4", _path(), 12, {}, spot=(500.0, 50.0, 0, "run R"))
    tr.new_wave()                                                         # no finish
    assert [f[3] for f in tr.fails] == ["walk L", "jump up", "run R"]
    assert _choose(tr.fails) == (None, {"spot": 3, "family": 0, "risk": 0})


def _path(n=12):
    return np.stack([np.arange(n + 1) * 2.0, np.zeros(n + 1)], 1)


def test_held_prediction_then_a_faster_finish_and_never_again():
    tr = RouteTrials(20.0)
    assert tr.allowed(0)
    ev = tr.start(0, 100, ("k", ("run R",)), _path(), 12, {"gain": 30.0})
    assert ev["kind"] == "route trial start" and ev["frames"] == 12 and "_path" not in ev
    assert not tr.allowed(1)                                              # one trial per wave
    out = []
    for k in range(1, 13):
        out += tr.observe(0, 100 + k, 2.0 * k + 0.5, 0.0, tube=lambda k: 3.0)
    assert [e["kind"] for e in out] == ["route trial window"] and out[0]["held"] and out[0]["deviation"] is None
    fin = tr.finished(0, 300, faster=True)
    assert fin["outcome"] == "faster" and fin["end_frame"] == 300 and fin["window"]["held"]
    assert tr.new_wave() == [] and tr.allowed(0)
    assert ("k", ("run R",)) in tr.tried                                 # never tried again


def test_deviation_is_recorded_where_the_prediction_broke():
    tr = RouteTrials(20.0)
    tr.start(0, 0, "key", _path(), 12, {})
    for k in range(1, 13):
        y = 0.0 if k < 5 else 10.0                                        # off the predicted path from frame 5
        tr.observe(0, k, 2.0 * k, y, tube=lambda k: 3.0)
    ev = tr.new_wave()[0]                                                 # the wave ended without a finish
    assert ev["outcome"] == "no finish"
    assert ev["deviation"]["k"] == 5 and ev["deviation"]["real"] == [10.0, 10.0]
    assert ev["window"]["held"] is False and ev["max_error"] == 10.0


def test_death_closes_the_trial_with_its_cause():
    tr = RouteTrials(20.0)
    tr.start(3, 50, "key", _path(), 12, {})
    out = tr.observe(3, 54, 0.0, 0.0, dead=True, cause="hazard h2", tube=lambda k: 3.0)
    assert out[-1]["kind"] == "route trial" and out[-1]["outcome"] == "died" and out[-1]["cause"] == "hazard h2"
    assert out[-1]["window"] == {"held": False, "died_at": 4} and out[-1]["deviation"]["k"] == 4
    assert tr.finished(3, 60, faster=True) is None                        # already closed


def test_committer_plays_a_trial_without_new_decisions():
    cm = Committer(replan=8, on=True)
    p = types.SimpleNamespace(frame=0, actions=[0] * 48, trial_frames=12, first_move_frames=12, goal_touch=None,
                              key_touches=[])
    cm.decided(0, p, 0)
    assert cm.triggers(0, 10, 0.0, 0.0, tiles=True) == []                 # inside the trial: nothing fires
    assert "interval" in cm.triggers(0, 12, 0.0, 0.0)                    # after it: the usual rules
    p.trial_frames = 0
    assert "interval" in cm.triggers(0, 10, 0.0, 0.0)
