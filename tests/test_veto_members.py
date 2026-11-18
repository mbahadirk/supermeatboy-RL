"""plan.planner._verdict's quorum (agent.py --veto-members): how many world-model members must touch a hazard
within the veto window for a veto; fewer are soft risk (their share)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from plan import safety as S                                               # noqa: E402
from plan.planner import _verdict                                          # noqa: E402


def _sweep(contact_frames):
    """5 members x candidates x 48 one-frame steps; contact_frames (E, n): each member's first contact (-1 none)."""
    cf = np.asarray(contact_frames, np.int64)
    E, n = cf.shape
    clear = np.full((E, n, 48), 100.0)
    for e in range(E):
        for c in range(n):
            if cf[e, c] > 0:
                clear[e, c, cf[e, c] - 1:] = -1.0
    return S.Sweep(cf, np.where(cf > 0, S.KIND_HAZARD, S.KIND_NONE), np.where(cf > 0, 0, -1), ["h0"],
                   np.array([S.KIND_HAZARD]), np.zeros((E, n, 1)), np.zeros((E, n, 1), np.int64),
                   np.full((E, n), 100.0), np.full((E, n), -1, np.int64), clear)


def test_any_member_vetoes_by_default_and_a_majority_with_quorum_3():
    # candidates: nobody touches | 1 of 5 at frame 8 | 3 of 5 at frames 6, 8, 10 | all 5 at frame 4
    cf = np.array([[-1, 8, 6, 4], [-1, -1, 8, 4], [-1, -1, 10, 4], [-1, -1, -1, 4], [-1, -1, -1, 4]])
    upto = np.full(4, 48)
    veto, vf, _, vid, soft, _ = _verdict(_sweep(cf), upto)
    assert veto.tolist() == [False, True, True, True] and vf.tolist() == [-1, 8, 6, 4] and vid[1] == "h0"
    veto3, vf3, _, _, soft3, sf3 = _verdict(_sweep(cf), upto, quorum=3)
    assert veto3.tolist() == [False, False, True, True] and vf3.tolist() == [-1, -1, 6, 4]
    assert soft3[0] == 0.0 and soft3[1] >= 0.2                 # 1 of 5 touched: at least its share is risk
    assert sf3[1] in (8, 17, 18, 19, 20) and sf3[1] > 0        # its frame: the touch (or the later margin rule's)
    # a quorum above the member count is the member count (stage 1 and 2 run one member: any touch still vetoes)
    one = np.array([[-1, 8]])
    v1 = _verdict(_sweep(one), np.full(2, 48), quorum=3)[0]
    assert v1.tolist() == [False, True]


def test_soft_risk_of_the_second_worst_member_by_default():
    """One member grazing a hazard after the veto window: the worst rule costs a whole death (risk 1), the second
    worst member's rule (the default) what the second member sees, the mean a fifth."""
    cf = np.array([[30], [-1], [-1], [-1], [-1]])                  # member 0 touches at frame 30, nobody else
    upto = np.full(1, 48)
    worst = _verdict(_sweep(cf), upto)[4]
    second = _verdict(_sweep(cf), upto, members="second")[4]
    mean = _verdict(_sweep(cf), upto, members="mean")[4]
    assert worst[0] == 1.0 and second[0] == 0.0 and mean[0] == 0.2
    cf2 = np.array([[30], [34], [-1], [-1], [-1]])                 # two members agree: a whole death again
    assert _verdict(_sweep(cf2), upto, members="second")[4][0] == 1.0


def test_corner_risk_is_soft_risk_by_the_same_members_rule():
    cf = np.full((5, 1), -1)
    upto = np.full(1, 48)
    one = (np.array([[0.8], [0.0], [0.0], [0.0], [0.0]]), np.array([[30], [-1], [-1], [-1], [-1]]))
    two = (np.array([[0.8], [0.6], [0.0], [0.0], [0.0]]), np.array([[30], [28], [-1], [-1], [-1]]))
    v = _verdict(_sweep(cf), upto, corner=one)
    assert not v[0][0] and v[4][0] == 0.8 and v[5][0] == 30          # worst: the member's risk, never a veto
    assert _verdict(_sweep(cf), upto, members="second", corner=one)[4][0] == 0.0
    v2 = _verdict(_sweep(cf), upto, members="second", corner=two)
    assert v2[4][0] == 0.6 and v2[5][0] == 30
    assert abs(_verdict(_sweep(cf), upto, members="mean", corner=two)[4][0] - 0.28) < 1e-9


def test_a_near_pass_in_the_first_frames_is_soft_risk():
    """A hazard within the margin at frame 8 (no touch): soft risk 1 - clearance / margin(8) in the estimate (the
    reflex move no longer overrides such a plan); early_margin False is the rule before: nothing until frame 17."""
    cf = np.full((1, 1), -1)
    sw = _sweep(cf)
    sw.step_clear[0, 0, 7] = 0.5 * S.margin(8)                     # frame 8: half the margin away
    upto = np.full(1, 48)
    veto, _, _, _, soft, sf = _verdict(sw, upto)
    assert not veto[0] and abs(soft[0] - 0.5) < 1e-9 and sf[0] == 8
    old = _verdict(sw, upto, early_margin=False)
    assert not old[0][0] and old[4][0] == 0.0


def test_the_stall_rule_charges_plans_that_do_not_get_on():
    """Planner.stalled counts the frames since the agent last got STALL_STEP below its best T (beyond the free
    ones); short executable progress earns proportional credit without resetting observed stall history."""
    from plan.planner import Planner, STALL_STEP, stall_penalty
    pl = Planner.__new__(Planner)
    pl.stall_free, pl.stalls = 60.0, {}
    assert pl.stalled(0, 100, 300.0) == (0.0, 300.0)               # the first decision: the best T, no stall
    assert pl.stalled(0, 150, 296.0) == (0.0, 300.0)               # 4 lower: not progress, still free frames
    assert pl.stalled(0, 260, 303.0) == (100.0, 300.0)             # 160 frames without it: 100 beyond the free 60
    pen = stall_penalty(100.0, 300.0, np.array([305.0, 300.0, 293.0, 300.0 - STALL_STEP, 200.0, np.inf]))
    np.testing.assert_allclose(pen, [100.0, 100.0, 30.0, 0.0, 0.0, 100.0])
    assert pl.stalled(0, 270, 289.0) == (0.0, 289.0)               # real progress: the count starts again
    assert pl.stalled(1, 270, 500.0) == (0.0, 500.0)               # each agent its own
    pl.stall_free = -1.0                                           # off
    assert pl.stalled(0, 900, 289.0) == (0.0, float("inf"))
    assert stall_penalty(0.0, 300.0, np.array([300.0])).tolist() == [0.0]
