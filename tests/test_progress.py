"""Checks for the progress field V (plan/progress.py): hand-made maps, and one recorded level against
realpath.Planner's own routes (skipped when the data is missing)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from realpath import JUMP, Planner                            # noqa: E402
from plan.progress import ProgressField, path_cost           # noqa: E402

T = 20.0
H, W = 22, 40
DATA = os.path.join(HERE, "runs", "real_20261103-012157", "wm_data")
LEVEL = "4-6"


def at(c, r):
    """The centre of cell (c, r) in game units (origin 0, 0)."""
    return ((c + 0.5) * T, (r + 0.5) * T)


def terrain():
    """Floor rows 0-1; a plateau (cols 10-39) whose top row is 8, so its standing row is 9 and its face at
    col 10 is 7 tiles high (more than JUMP's rise of 5); a low wall (2 tiles) on the plateau at col 25. No
    border walls and no ceiling: the map's edges are not walls to climb."""
    w = np.zeros((H, W), np.uint32)
    w[0:2, :] = 1
    w[2:9, 10:] = 1
    w[9:11, 25] = 1
    return w


FACE = (199.0, 110.0, 5.0, 70.0, 0.0)          # a hazard strip covering the plateau's face (rows 2-8)


def field(hazards=(), keys=(), spawn=(15, 9), goal=(35, 9)):
    return ProgressField(terrain(), 0.0, 0.0, at(*spawn), [at(*goal)], list(hazards), [], list(keys), [], [])


def test_v_falls_toward_the_goal_and_matches_the_route():
    pf = field()
    assert pf.found_route and pf.jumps == JUMP and pf.n_legs == 1
    v = pf.V_support[0, 9]
    row = [v[c] for c in range(11, 25)] + [v[c] for c in range(26, 36)]   # the plateau, both sides of the low wall
    assert np.isfinite(row).all()
    assert all(a > b for a, b in zip(row, row[1:]))
    assert v[35] == 0.0
    assert pf.V_support[0, 9, 15] == pytest.approx(pf.route_cost)
    lay = pf._layer(pf.leg_locks[0])
    assert pf.route_cost == pytest.approx(path_cost(pf.route_cells, lay["cost"], lay["support"]))
    assert np.isinf(pf.V_support[0][pf.pl.wall]).all()
    # airborne: never better than standing in the same cell (a fresh jump can do everything a used one can)
    open_ = np.isfinite(pf.V_air[0]) & np.isfinite(pf.V_support[0])
    assert (pf.V_air[0][open_] >= pf.V_support[0][open_] - 1e-9).all()


def test_a_ledge_higher_than_a_jump_is_climbed_only_by_its_wall():
    # below the plateau (cols 0-9): the goal on the plateau is 7 tiles up. The face is a wall to climb ...
    pf = field()
    assert np.isfinite(pf.V_support[0, 2, 0:10]).all()
    # ... but with a hazard on the face (the cells beside it closed) the goal cannot be reached from there:
    # a dead end, +inf, while the plateau keeps its values. Standing on the floor (row 2) or a row up, a jump
    # rises to row 8 at most; no airborne state below row 8 there can get up either
    pf = field(hazards=[FACE])
    assert pf.label == "clear"
    assert np.isinf(pf.V_support[0, 2:4, 0:9]).all()
    assert np.isinf(pf.V_air[0, 2:8, 0:9]).all()
    assert np.isfinite(pf.V_support[0, 4, 0:9]).all()      # a jump starting at row 4 would reach row 9
    assert np.isfinite(pf.V_support[0, 9, 12:25]).all() and np.isfinite(pf.V_support[0, 9, 26:36]).all()
    # the lookup: a dead end, a point in a wall (its open neighbour above plus the way there), outside the map
    x, y = at(4, 2)
    assert np.isinf(pf.values([x], [y])[0])
    inside = pf.values([at(30, 8)[0]], [at(30, 8)[1] + 5.0])[0]           # 5 units below the top of the plateau
    v9 = pf.V_support[0, 9]                                               # its open neighbours: row 9, cols 29-31
    assert inside == pytest.approx(min(v9[29] + 25.0 / T, v9[30] + 15.0 / T, v9[31] + 25.0 / T))
    assert inside == pytest.approx(v9[31] + 25.0 / T)                     # the diagonal toward the goal
    assert np.isinf(pf.values([-50.0, 100.0], [100.0, -50.0])).all()


def test_key_legs_follow_the_route_order():
    k_left, k_right = at(12, 9), at(28, 12)      # spawn at col 20: the left key is nearer; the right one is in the air
    pf = field(keys=[k_right, k_left], spawn=(20, 9), goal=(35, 9))
    assert pf.n_legs == 3
    assert pf.key_order == [[1], [0]]             # left key (index 1) first, then the right one
    s = (9, 20)
    v0, v1, v2 = (pf.V_support[j][s] for j in range(3))
    assert v0 > v1 > v2 > 0
    assert v0 == pytest.approx(pf.route_cost)
    # leg 0 falls toward the left key, leg 2 toward the goal
    assert pf.V_support[0, 9, 13] < pf.V_support[0, 9, 18]
    assert pf.V_support[2, 9, 30] < pf.V_support[2, 9, 20]
    # at a key, a leg's value is the next leg's value there
    assert pf.V_support[0, 9, 12] == pytest.approx(pf.V_support[1, 9, 12])
    assert pf.V_support[1, 12, 28] == pytest.approx(pf.V_support[2, 12, 28])
    # legs from the keys held: none, the second one only (out of order: still leg 0), the first, both
    assert pf.leg_of([0, 1, 2, 3]).tolist() == [0, 0, 1, 2]
    x, y = at(20, 9)
    assert pf.values([x, x, x], [y, y, y], leg=[0, 1, 2]).tolist() == pytest.approx([v0, v1, v2])


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "levels", f"{LEVEL}.npz")),
                    reason="recorded level data not found")
def test_recorded_level_spawn_value_is_the_planner_route_cost():
    from wm.buffer import load_level
    words, meta = load_level(DATA, LEVEL)
    args = (words, meta["ox"], meta["oy"], tuple(meta["spawn"]), [tuple(g) for g in meta["goals"]], [], [],
            [tuple(k) for k in meta["keys"]], [tuple(b) for b in meta["bandages"]], [])
    pf = ProgressField(*args)
    pl = Planner(*args)
    route = pl.find(1)[0]
    assert route.mode.split(",")[0] == pf.label.split(",")[0]
    assert pf.route_cells == route.cells                      # the same search: the same first route
    lay = pf._layer(pf.leg_locks[-1])
    sc, sr = pf.start
    v = pf.V_support[0, sr, sc]
    assert np.isfinite(v)
    assert v == pytest.approx(path_cost(route.cells, lay["cost"], lay["support"]), abs=1e-6)
    assert v == pytest.approx(float(pf.values([meta["spawn"][0]], [meta["spawn"][1]])[0]), abs=1e-9)
    # and from other cells: V_support equals the cost of the Planner's own search started there
    rng = np.random.default_rng(0)
    rr, cc = np.nonzero(~pf.pl.wall)
    for i in rng.choice(len(rr), 25, replace=False):
        st = (int(cc[i]), int(rr[i]))
        cells = pf.pl._leg(st, pf.goal_cells, pf.mode, None, pf.jumps, pf.leg_locks[-1])
        want = path_cost(cells, lay["cost"], lay["support"]) if cells else np.inf
        got = pf.V_support[0, st[1], st[0]]
        assert (np.isinf(want) and np.isinf(got)) or got == pytest.approx(want, abs=1e-6)


def test_a_point_on_a_half_block_floor_takes_its_open_side():
    """A floor of lower-half blocks (row 5, cols 5-30) over a room, with a hole at col 4; the goal below. A
    body standing on that floor (centre in the cells' open upper half) is valued by the cell above it, never by
    the cell below the floor; a point inside a full block keeps the best open neighbour of its 3 x 3."""
    w = np.zeros((H, W), np.uint32)
    w[0:2, :] = 1
    w[5, 5:31] = 1 | (1 << 18) | (2 << 12)              # half blocks, rotated 180: the lower half is solid
    w[2:12, 31] = 1                                      # a wall right of the floor
    pf = ProgressField(w, 0.0, 0.0, at(10, 6), [at(20, 2)], [], [], [], [], [])
    assert pf.found_route
    cols = np.arange(5, 31)
    on = pf.values((cols + 0.5) * T, np.full(len(cols), 5 * T + 16.0))           # standing: centre 6 above
    up = pf.V_support[0, 6]                             # the open side: row 6, the cell above and its diagonals
    want = np.minimum.reduce([up[cols] + 0.7, up[cols - 1] + np.hypot(1, 0.7), up[cols + 1] + np.hypot(1, 0.7)])
    assert np.allclose(on, want)
    below = pf.V_support[0, 4, cols]
    assert (on > below + 1).all()                       # the old rule took the cell below the floor
    assert np.all(np.diff(on) > 0)                      # rising away from the hole at col 4
    # inside a full block (the wall at col 31): the best open neighbour, as before
    v = pf.values([31.5 * T], [3.5 * T])[0]
    nb = min(pf.V_support[0, r, c] + np.hypot(31.5 - c - 0.5, 3.5 - r - 0.5) for r in (2, 3, 4) for c in (30, 32)
             if np.isfinite(pf.V_support[0, r, c]))
    assert v == pytest.approx(nb)


# ---- the time field T (frames) on the same graph ---------------------------------------------------------
from plan.progress import FX_GROUND, FY_UP, FY_DOWN          # noqa: E402


def test_time_on_a_flat_corridor_is_distance_times_frames_per_tile():
    pf = field()
    t = pf.T_support[0, 9]
    for c in range(26, 36):                       # the plateau right of the low wall: a straight walk to the goal
        assert t[c] == pytest.approx((35 - c) * FX_GROUND)
    # the lookup at a cell centre gives the table's value; a point in the plateau's top row is valued from the
    # open cell above plus the way there, in frames (FX_GROUND per tile)
    assert pf.times([at(30, 9)[0]], [at(30, 9)[1]], 0, True)[0] == pytest.approx(5 * FX_GROUND)
    inside = pf.times([at(30, 8)[0]], [at(30, 8)[1] + 5.0], 0, True)[0]
    assert inside == pytest.approx(t[31] + FX_GROUND * 25.0 / T)
    # crossing the low wall (2 tiles up and down again) costs its rise and fall frames, more than walking
    assert t[24] - t[26] >= 2 * FX_GROUND
    assert np.isinf(pf.T_support[0][pf.pl.wall]).all()


def test_rising_and_falling_airborne_times_differ_below_a_ledge():
    # below the plateau's face (col 10, rows 2-8, a wall to climb): a body rising at (6, 6) can still reach the
    # face beside it and climb; a falling one only goes down to the floor (row 2) and walks back to the face
    pf = field()
    c, r = 6, 6
    rise, fall = pf.T_rise[0, r, c], pf.T_fall[0, r, c]
    assert np.isfinite(rise) and np.isfinite(fall) and rise < fall
    x, y = at(c, r)
    up = pf.times([x], [y], 0, False, vy=[2.0])[0]
    down = pf.times([x], [y], 0, False, vy=[-2.0])[0]
    assert up == pytest.approx(rise) and down == pytest.approx(fall)
    assert pf.times([x], [y], 0, False)[0] == pytest.approx(min(rise, fall))          # vy unknown: the better one
    # standing in a cell is never slower than being airborne there
    ok = np.isfinite(pf.T_support[0]) & np.isfinite(pf.T_fall[0])
    assert (pf.T_fall[0][ok] >= pf.T_support[0][ok] - 1e-9).all()
    # a step up costs FY_UP, a step down FY_DOWN: the face's wall cells (col 9) get slower further down
    face = pf.T_support[0, 2:9, 9]
    assert all(a > b for a, b in zip(face, face[1:]))
    assert face[0] - face[1] == pytest.approx(FY_UP)
    assert FY_DOWN > FY_UP


def test_time_key_legs_add_up_like_v():
    k_left, k_right = at(12, 9), at(28, 12)
    pf = field(keys=[k_right, k_left], spawn=(20, 9), goal=(35, 9))
    s = (9, 20)
    t0, t1, t2 = (pf.T_support[j][s] for j in range(3))
    assert t0 > t1 > t2 > 0
    assert pf.T_support[0, 9, 12] == pytest.approx(pf.T_support[1, 9, 12])    # at a key: the next leg's value
    assert pf.T_support[1, 12, 28] == pytest.approx(pf.T_support[2, 12, 28])
    # leg 0 is the way to the left key (8 tiles of floor) plus leg 1 from there
    assert t0 == pytest.approx(8 * FX_GROUND + pf.T_support[1, 9, 12])
