"""Route trials (plan.planner.RouteTrials): a plan much better than the best route is played once, exactly."""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plan.commit import Committer                                          # noqa: E402
from plan.planner import RouteTrials, route_trial_choice                   # noqa: E402


def test_choice_takes_the_best_unvetoed_untried_candidate_beyond_the_gain():
    est = np.array([200.0, 170.0, 150.0, 185.0, 140.0])
    veto = np.array([False, False, False, False, True])
    keys = ["route", "a", "b", "c", "d"]
    assert route_trial_choice(est, veto, 0, 20.0, keys, {}) == 2          # 140 is vetoed
    assert route_trial_choice(est, veto, 0, 20.0, keys, {"b": {}}) == 1   # b tried: the next one (30 > 20)
    assert route_trial_choice(est, veto, 0, 20.0, keys, {"a": {}, "b": {}}) is None   # c beats it by 15 only
    assert route_trial_choice(est, veto, 0, 60.0, keys, {}) is None
    assert route_trial_choice(est, veto, 0, 0.0, keys, {}) is None        # off


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
