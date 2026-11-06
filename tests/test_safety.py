"""Checks for the geometric safety check (plan/safety.py), no game or data needed."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from plan import safety as S                     # noqa: E402

LV = SimpleNamespace(ox=0.0, oy=0.0, W=50, H=20)   # map 0..1000 x 0..400


def fb_of(*boxes):
    return S.frame_boxes([], sh=(), memory=None) if not boxes else S.FrameBoxes(
        np.array(boxes, float), np.zeros(len(boxes), np.int8), [f"h{k}" for k in range(len(boxes))])


def traj(points):
    """One member, one candidate: [(x, y), ...] -> (1, 1, K+1) arrays."""
    p = np.array(points, float)
    return p[None, None, :, 0], p[None, None, :, 1]


def test_swept_check_catches_a_thin_box_the_end_points_miss():
    thin = fb_of((100.0, 200.0, 20.0, 1.0, 0.0))            # a thin strip at y = 200
    xs, ys = traj([(100, 250), (100, 150)])                  # falls 100 units straight through it
    sw = S.sweep(xs, ys, [None, thin])
    assert sw.contact_frame[0, 0] == 1 and sw.contact_kind[0, 0] == S.KIND_HAZARD
    assert sw.ids[sw.contact_box[0, 0]] == "h0"
    # an end-points-only check: both ends are far from the strip
    ends, _ = S.segment_clearance([100, 100], [250, 150], [100, 100], [250, 150], thin)
    assert (ends > 0).all()


def test_clearance_for_a_known_box():
    box = fb_of((0.0, 0.0, 10.0, 5.0, 0.0))
    clear, oc = S.segment_clearance([30.0, 0.0], [0.0, 20.0], [30.0, 0.0], [0.0, 20.0], box)
    assert clear[0, 0] == pytest.approx(30 - 10 - S.BODY)
    assert clear[1, 0] == pytest.approx(20 - 5 - S.BODY)
    assert np.isinf(oc).all()
    # rotated by 90 degrees: the half sizes swap
    rot = fb_of((0.0, 0.0, 10.0, 5.0, np.pi / 2))
    clear, _ = S.segment_clearance([30.0], [0.0], [30.0], [0.0], rot)
    assert clear[0, 0] == pytest.approx(30 - 5 - S.BODY)
    # min over a trajectory, with its frame
    xs, ys = traj([(60, 0), (40, 0), (25, 0), (50, 0)])
    sw = S.sweep(xs, ys, [None, box, box, box])
    assert sw.clear_min[0, 0, 0] == pytest.approx(25 - 10 - S.BODY)
    assert sw.clear_frame[0, 0, 0] == 2 and sw.contact_frame[0, 0] == -1
    near = S.assess(sw).nearest[0]
    assert near[0]["id"] == "h0" and near[0]["frame"] == 2


@pytest.mark.parametrize("x,y,out", [
    (-10.0, 200, False), (-10.1, 200, True),              # left: x < left - 10
    (1010.0, 200, False), (1010.1, 200, True),            # right: x > right + 10
    (500, 410.0, False), (500, 410.1, True),              # top: y > top + 10
    (500, -60.0, False), (500, -60.1, True),              # bottom: y < bottom - 60
])
def test_out_of_level_each_side(x, y, out):
    assert bool(S.out_of_level(LV, [x], [y])[0]) is out
    kind, _ = S.segment_contact([500.0], [200.0], [x], [y], fb_of(), S.level_bounds(LV))
    assert bool(kind[0] == S.KIND_OUT) is out


def test_veto_before_frame_16_soft_risk_after():
    box = (500.0, 200.0, 10.0, 10.0, 0.0)
    K = 24
    # candidate 0 walks into the box at frame 10, candidate 1 at frame 20, candidate 2 passes 5 units over it
    xs = np.full((2, 3, K + 1), 300.0)
    ys = np.full((2, 3, K + 1), 200.0)
    xs[:, 0, 10:] = 500.0
    xs[:, 1, 20:] = 500.0
    xs[:, 2, 20:] = 500.0
    ys[:, 2, :] = 200.0 + 10 + S.BODY + 5.0              # clearance 5
    ys[1, 2, :] = 200.0 + 10 + S.BODY + 8.0              # the other member: clearance 8 (max risk is taken)
    frames = [fb_of(box)] * (K + 1)
    sf = S.check(xs, ys, frames, S.level_bounds(LV))
    assert sf.veto.tolist() == [True, False, False]
    assert sf.veto_frame.tolist() == [10, -1, -1] and sf.veto_id[0] == "h0"
    assert sf.soft_risk[1] == 1.0 and sf.soft_frame[1] == 20
    # clearance 5 from frame 20 on: the risk grows with the horizon's margin, highest at the last frame
    assert sf.soft_risk[2] == pytest.approx(1 - 5.0 / float(S.margin(K)))
    assert sf.soft_frame[2] == K
    assert sf.soft_risk[0] == 1.0                          # still inside the box after frame 16
    # the same contact at frame 16 is a veto, at 17 it is not
    for f, v in ((16, True), (17, False)):
        xs = np.full((1, 1, K + 1), 300.0)
        xs[:, :, f:] = 500.0
        sf = S.check(xs, np.full_like(xs, 200.0), frames)
        assert bool(sf.veto[0]) is v
        assert sf.veto_frame[0] == (16 if v else -1)
        assert sf.soft_risk[0] == 1.0 and sf.soft_frame[0] == 17      # (still inside after frame 16)


def test_margin_interpolates_the_measured_error():
    assert float(S.margin(16)) == pytest.approx(13.9)
    assert float(S.margin(24)) == pytest.approx((13.9 + 38.1) / 2)
    assert float(S.margin(60)) == pytest.approx(60.0)


def test_death_memory_window_and_round_trip(tmp_path):
    m = S.DeathMemory()
    m.add("4-6", 100, 50.0, 60.0)
    m.add("1-1", 100, 0.0, 0.0)
    assert len(m.boxes("4-6", 100 - S.WINDOW)) == 1
    assert len(m.boxes("4-6", 100 + S.WINDOW)) == 1
    assert len(m.boxes("4-6", 100 + S.WINDOW + 1)) == 0
    assert len(m.boxes("4-5", 100)) == 0
    b = m.boxes("4-6", 90)[0]
    assert b.tolist() == [50.0, 60.0, S.DEATH_HALF, S.DEATH_HALF, 0.0]
    fb = S.frame_boxes([], memory=m.boxes("4-6", 90))
    kind, _ = S.segment_contact([50.0], [60.0 + S.DEATH_HALF + S.BODY - 1], [50.0], [60.0 + S.DEATH_HALF + S.BODY - 1], fb)
    assert kind[0] == S.KIND_MEMORY
    p = str(tmp_path / "deaths.json")
    m.save(p)
    m2 = S.DeathMemory.load(p)
    assert m2.entries == m.entries
    assert np.array_equal(m2.boxes("4-6", 120), m.boxes("4-6", 120))
    assert len(S.DeathMemory.load(str(tmp_path / "none.json"))) == 0


def test_explain_death_uses_hazards_shots_and_bounds():
    rec = {"hz": [[200.0, 200.0, 20.0, 20.0, 1.0, 1.0, 0.0]], "sh": [[7, 400.0, 200.0, 5.0]]}
    keep, grow = [True], [(0.0, 0.0)]
    assert S.explain_death(LV, rec, keep, grow, 150, 200, 185, 200)          # into the piece
    assert S.explain_death(LV, rec, keep, grow, 380, 200, 392, 200)          # into the shot
    assert S.explain_death(LV, rec, keep, grow, 5, 200, -15, 200)            # out of the level
    assert not S.explain_death(LV, rec, keep, grow, 600, 200, 610, 200)
    assert not S.explain_death(LV, rec, [False], grow, 150, 200, 185, 200)   # decoration does not kill
    assert S.explain_death(LV, rec, keep, [(20.0, 0.0)], 150, 200, 165, 200)  # kill reach grown by 20


def test_first_contact_goal_and_key_boxes():
    goal = (100.0, 200.0, 20.0, 1.0, 0.0)                    # touched within BODY of it, like a hazard
    key = (300.0, 50.0, 5.0, 5.0, 0.0)
    pts = [(100, 260), (100, 240), (100, 150), (300, 70), (300, 50)]
    xs, ys = traj(pts)
    f, w = S.first_contact(xs, ys, [goal, key])
    assert f[0, 0] == 2 and w[0, 0] == 0                     # the fall 240 -> 150 sweeps through the thin goal
    f, w = S.first_contact(xs, ys, [goal, key], first=3)
    assert f[0, 0] == 4 and w[0, 0] == 1                     # from frame 3 on: the key at frame 4
    f, _ = S.first_contact(xs, ys, [goal, key], upto=1)
    assert f[0, 0] == -1                                     # not within the judged frames
    # a path that starts in contact with a box does not count it (the game said it was not touched)
    xs, ys = traj([(100, 204), (100, 203), (100, 150)])
    f, _ = S.first_contact(xs, ys, [goal])
    assert f[0, 0] == -1
    f, _ = S.first_contact(xs, ys, np.zeros((0, 5)))
    assert f[0, 0] == -1
    # per member limits broadcast over (E, n)
    xs = np.array([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
    ys = np.array([[[100.0, 50.0, 0.0]], [[100.0, 90.0, 0.0]]])
    f, _ = S.first_contact(xs, ys, [(0.0, 0.0, 1.0, 1.0, 0.0)], first=np.array([[1], [3]]))
    assert f[:, 0].tolist() == [2, -1]
