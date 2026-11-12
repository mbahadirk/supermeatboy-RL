"""Flying shots' futures (mpc.future.shot_future / FutureWorld.shot_track): each shot from its own flight."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from mpc import future as F                              # noqa: E402
from mpc.future import FutureWorld, shot_future          # noqa: E402
from plan.geometry import Terrain                        # noqa: E402

DT = 16.667


def wall_terrain():
    """A wall column at col 10 (x 200..220), rows 0-11 (y 0..240)."""
    w = np.zeros((12, 30), np.uint32)
    w[:, 10] = 1
    return Terrain(w, 0.0, 0.0)


def lava_ball(n, x0=50.0, y0=0.0, vy=7.5, ay=-0.15):
    """A lava ball as the game moves it (v += a; y += v, recorded to 0.01): frames 0..n-1."""
    y, v, out = y0, vy, []
    for _ in range(n):
        out.append((x0, round(y, 2)))
        v += ay
        y += v
    return np.array(out)


def feed(fw, sid, path, start=0, rad=9.44, terrain=None):
    for j, (x, y) in enumerate(path):
        fw.observe(start + j, [], None, [[sid, float(x), float(y), rad]], [], None, DT, terrain=terrain)


def test_ballistic_lava_ball_is_followed_within_a_small_margin():
    path = lava_ball(120)
    for i in (10, 45, 70):                               # rising, near the top, falling
        centre, margin, alive = shot_future(path[i - F.SHOT_FIT + 1:i + 1], 48)
        err = np.abs(centre - path[i + 1:i + 49]).max(1)
        assert alive == 48 and (err <= margin).all()
        assert margin[47] < 15.0                         # the old rule: 2 + 0.5 x speed x t, up to ~190 at +48


def test_straight_bullet_is_exact():
    path = np.array([(100.0 + 6 * j, -110.61) for j in range(60)])
    centre, margin, alive = shot_future(path[:8], 48)
    assert np.abs(centre - path[8:56]).max() < 1e-6 and margin[47] < 8.0 and alive == 48


def test_shot_stops_where_it_would_be_inside_a_wall():
    ter = wall_terrain()
    path = np.array([(150.0 + 6 * j, 100.0) for j in range(8)])     # x 150..192, the wall's face at x 200
    centre, margin, alive = shot_future(path, 20, ter)
    first_solid = int(np.argmax(centre[:, 0] >= 200.0))              # index of the first centre in the wall
    assert first_solid == 1 and alive == 1                           # x 198 still flies, x 204 is in the wall
    fw = FutureWorld()                                               # the planner's boxes drop it from then on
    feed(fw, 3, path, terrain=ter)
    _, check, _ = fw.worlds(7, 20, DT)
    with_shot = [k for k, (_, _, ids) in enumerate(check, 1) if "shot3" in ids]
    assert with_shot == list(range(1, alive + 1))
    b, r, ids = check[0]
    j = ids.index("shot3")
    assert tuple(b[j][2:]) == (9.44, 9.44, 0.0) and r[j] == pytest.approx(margin[0])
    fw2 = FutureWorld()                                              # no terrain known: never stopped
    feed(fw2, 3, path)
    assert all("shot3" in ids for _, _, ids in fw2.worlds(7, 20, DT)[1])


def test_respawned_shot_reusing_its_id_starts_a_new_flight():
    fw = FutureWorld()
    first = [(100.0 + 6 * j, 0.0) for j in range(10)]
    feed(fw, 0, first)                                               # frames 0..9
    fw.observe(10, [], None, [], [], None, DT)                       # gone for a frame
    feed(fw, 0, [(100.0, 0.0)], start=11)                            # back at the launcher
    assert len(fw.shot_track(11, 0)) == 1
    fw = FutureWorld()                                               # respawn without an empty frame
    feed(fw, 0, first + [(100.0, 0.0), (106.0, 0.0)])                # frames 0..11
    assert fw.shot_track(10, 0).tolist() == [[100.0, 0.0]]
    assert fw.shot_track(11, 0).tolist() == [[100.0, 0.0], [106.0, 0.0]]
    fw = FutureWorld()                                               # a jump smaller than a step: second difference
    near = [(100.0 + 6 * j, 0.0) for j in range(5)] + [(118.0, 3.0), (124.0, 3.0), (130.0, 3.0)]
    feed(fw, 0, near)
    assert fw.shot_track(7, 0).tolist() == [[118.0, 3.0], [124.0, 3.0], [130.0, 3.0]]


def test_short_history_is_bounded_and_covers_the_truth():
    path = lava_ball(60)
    c1, m1, _ = shot_future(path[10:11], 48)                         # seen once: where it is, speed bound
    assert (c1 == path[10]).all() and m1[47] == pytest.approx(F.SHOT_BASE + F.SHOT_V_MAX * 48)
    assert (np.abs(c1 - path[11:59]).max(1) <= m1).all()
    c2, m2, _ = shot_future(path[9:11], 48)                          # seen twice: velocity, acceleration bound
    assert (np.abs(c2 - path[11:59]).max(1) <= m2).all() and (m2 < m1).all()
    c3, m3, _ = shot_future(path[8:11], 48)                          # three frames: the fit
    assert (np.abs(c3 - path[11:59]).max(1) <= m3).all() and m3[47] < m2[47]


def test_a_flight_that_does_not_fit_constant_acceleration_falls_back_to_the_bound():
    track = np.array([(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)])   # turning corners
    centre, margin, _ = shot_future(track, 10)
    assert np.allclose(centre[0], (-5.0, 5.0)) and margin[9] == pytest.approx(
        F.SHOT_BASE + 0.5 * F.SHOT_A_MAX * 10 * 11 + 2 * F.SHOT_NOISE * 10)


def _saw(x, y=100.0):
    """A lethal piece whose kill shape is a 20-unit circle at (x, y) (realgame.js hazards row with shapes)."""
    return [x, y, 50.0, 50.0, 1.0, 1.0, 0.0, [[1, x, y, 20.0, 20.0, 0]]]


def _ping_pong(f):
    """x of a saw going 0 -> 90 -> 0 at 1.5 units a frame: period 120 frames."""
    p = (1.5 * f) % 180.0
    return p if p <= 90.0 else 180.0 - p


def test_moving_piece_repeating_in_this_wave_is_looked_up_one_period_back():
    fw = FutureWorld()
    still = [0.0, 300.0, 40.0, 40.0, 1.0, 1.0, 0.0, [[1, 0.0, 300.0, 20.0, 20.0, 0]]]
    for f in range(300):
        fw.observe(f, [_saw(_ping_pong(f)), still], None, [], [True, True], None, DT)
        if f == 200:                                  # under two periods seen: the linear guess, growing
            _, check, recorded = fw.worlds(f, 48, DT)
            b, r, ids = check[47]
            assert recorded == 0 and r[0] > 2.0 and r[1] == 0.0
    _, check, _ = fw.worlds(299, 48, DT)
    for k in (8, 16, 32, 48):
        b, r, ids = check[k - 1]
        assert ids[:2] == ["h0", "h1"] and r[0] == F.REC_RADIUS and r[1] == 0.0
        assert abs(b[0][0] - _ping_pong(299 + k)) <= 1e-9 and tuple(b[1][:2]) == (0.0, 300.0)
    assert fw.runs["h0"].P == 120 and fw.runs["h1"].period() is None   # a box at rest has no period
    fw.observe(300, [_saw(50.0), still], None, [], [True, True], None, DT)   # off its period: guessed again
    assert fw.runs["h0"].P is None
    _, check, _ = fw.worlds(300, 8, DT)
    assert check[7][1][0] != F.REC_RADIUS


def _launcher_frame(f, period=30, life=10, speed=3.0):
    """A launcher at (0, 0) firing a straight shot (id 7) every `period` frames from frame 0, flying `life` frames."""
    age = f % period
    return [[7, speed * age, 0.0, 4.0]] if age < life else []


def test_launcher_predicts_shots_not_fired_yet_after_seeing_its_period():
    fw = FutureWorld()
    for f in range(0, 66):
        fw.observe(f, [], [], _launcher_frame(f), None, None, 16.667)
        if f == 25:                                    # one birth seen (frame 0 is the wave's first: not a birth)
            assert fw.launcher_shots(f, 48) == []
    shots = fw.launcher_shots(65, 48)                 # births at 30 and 60 seen: period 30, next at 90
    assert len(shots) == 1
    sid, rad, centre, margin, k0, k1 = shots[0]
    assert (k0, k1) == (25, 34) and rad == 4.0          # alive at ages 0..9: frames 90..99
    assert centre[24].tolist() == [0.0, 0.0] and centre[29].tolist() == [15.0, 0.0]
    assert 1.0 <= margin[29] <= 1.0 + 3.0 + 1e-9         # SHOT_BASE + one frame of motion
    _, check, _ = fw.worlds(65, 48, 16.667)
    assert any(str(i).startswith("shot7+90") for i in check[29][2])
    assert not any(str(i).startswith("shot7+") for i in check[10][2])


def test_trusted_recording_gives_the_recorded_shots_and_aimed_shots_are_not_trusted():
    fw = FutureWorld()
    for f in range(0, 100):                            # wave 1: the recording
        fw.observe(f, [], [], _launcher_frame(f), None, None, 16.667)
    fw.new_wave()
    for f in range(0, 41):
        fw.observe(f, [], [], _launcher_frame(f), None, None, 16.667)
    _, check, rec = fw.worlds(40, 48, 16.667)
    assert rec == 48
    b, r, ids = check[20]                              # step 21, frame 61: the recorded shot fired at 60, age 1
    j = ids.index("shot7")
    assert b[j][:2].tolist() == [3.0, 0.0] and r[j] == F.REC_RADIUS
    fw.new_wave()
    for f in range(0, 5):                              # an aimed shot: not where the recording has it
        fw.observe(f, [], [], [[7, 3.0 * f + 5.0, 0.0, 4.0]] if f < 10 else [], None, None, 16.667)
    assert not fw.shots_trusted


def test_deadly_only_gives_the_same_boxes_on_a_recorded_wave():
    """The per-frame danger check's boxes (future_inputs deadly_only) equal the decision's, frame by frame."""
    import glob
    import gzip
    import json
    from plan.planner import future_inputs
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs",
                     "capture_2-14_route_check01", "wm_data")
    paths = sorted(glob.glob(os.path.join(d, "wave_*_world.json.gz")))
    if not paths:
        pytest.skip("no recorded wave here")
    world = json.load(gzip.open(paths[-1], "rt", encoding="utf-8"))[:120]
    fw = FutureWorld()
    keep = None
    for w in world:
        keep = w.get("keep", keep)
        fw.observe(w["f"], w.get("hz") or [], w.get("pf"), w.get("sh") or [], keep, None, 16.667)
        if w["f"] % 7:
            continue
        wa, fa, ra = future_inputs(fw, w["f"], 48, 16.667)
        wb, fb, rb = future_inputs(fw, w["f"], 48, 16.667, deadly_only=True)
        assert wb is None and len(wa) == 48 and ra == rb
        for x, y in zip(fa[1:], fb[1:]):
            assert np.array_equal(x.boxes, y.boxes) and list(x.ids) == list(y.ids)
