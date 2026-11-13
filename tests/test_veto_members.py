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
