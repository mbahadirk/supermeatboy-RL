"""Checks for the camera-bounds kill box (realgame.js killBox -> info["kb"] -> game.level.Level.kill_box): where
the level says a centre is dead off screen, the agents' map, the sealed pockets, the routes and the safety
check. No game needed; the identity check on a recorded level is skipped when the data is missing."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from game.level import C_GOAL, C_HAZARD, C_SOLID, Level, alive_box     # noqa: E402
from plan import safety as S                                    # noqa: E402
from realpath import Planner                                    # noqa: E402

T = 20.0
DATA = os.path.join(HERE, "runs", "real_20261103-012157", "wm_data")
LEVEL = "4-6"


def at(c, r):
    return ((c + 0.5) * T, (r + 0.5) * T)


def make_level(words, kb=None, spawn=(3, 6), goal=(37, 6)):
    H, W = words.shape
    info = {"name": "synthetic-kill-box", "W": W, "H": H, "ox": 0.0, "oy": 0.0, "goals": [list(at(*goal))],
            "spawn": list(at(*spawn))}
    if kb is not None:
        info.update(kb=kb, cam_bounds=[0.0, 0.0, 1.0, 1.0], view_half=[100.0, 60.0])
    return Level(info, words.astype("<u4").tobytes())


def flat(H=20, W=50):
    w = np.zeros((H, W), np.uint32)
    w[0:2, :] = 1
    return w


def test_out_of_level_and_outside_cells_follow_the_kill_box():
    kb = [100.0, 700.0, 50.0, 300.0]                  # map 0..1000 x 0..400
    lv = make_level(flat(), kb)
    assert lv.kill_box == tuple(kb)
    xs = np.array([99.9, 100.0, 700.0, 700.1, 400.0, 400.0, 400.0, 400.0])
    ys = np.array([200.0, 200.0, 200.0, 200.0, 49.9, 50.0, 300.0, 300.1])
    assert lv.out_of_level(xs, ys).tolist() == [True, False, False, True, True, False, False, True]
    # vectorised (broadcast) and the same rule as the safety check
    assert lv.out_of_level(xs[:, None], ys[None, :]).shape == (8, 8)
    assert (S.out_of_level(lv, xs, ys) == lv.out_of_level(xs, ys)).all()
    assert S.level_bounds(lv) == tuple(kb)
    # outside_cells: cell centres (c + 0.5, r + 0.5) * 20 outside the box
    cx = (np.arange(lv.W) + 0.5) * T
    cy = (np.arange(lv.H) + 0.5) * T
    want = (cx[None, :] < 100) | (cx[None, :] > 700) | (cy[:, None] < 50) | (cy[:, None] > 300)
    assert lv.outside_cells.shape == (lv.H, lv.W)
    assert (lv.outside_cells == want).all()
    assert lv.outside_cells[:, :5].all() and not lv.outside_cells[5, 5] and lv.outside_cells[15, 20]
    assert not lv.outside_cells[2:15, 5:35].any()


def test_no_kill_box_keeps_the_map_edge_rule():
    lv = make_level(flat())
    assert lv.kill_box is None and not lv.outside_cells.any()
    assert lv.alive_box() == (-10.0, 1010.0, -60.0, 410.0)
    assert lv.out_of_level([-10.0, -10.1, 1010.1, 500.0, 500.0, 500.0], [200, 200, 200, 410.1, -60.0, -60.1]).tolist() \
        == [False, True, True, True, False, True]
    # a bad kb from the game: the older rule
    for bad in ([1.0, 1.0, 0.0, 5.0], [0.0, 5.0, float("nan"), 1.0], [1.0, 2.0], "x"):
        assert make_level(flat(), bad).kill_box is None


def test_kill_box_changes_from_a_frame_message():
    lv = make_level(flat())
    ver = lv.tile_ver
    assert lv.set_kill_box([100.0, 700.0, 50.0, 300.0]) and lv.tile_ver > ver and lv.outside_cells.any()
    assert not lv.set_kill_box([100.0, 700.0, 50.0, 300.0])           # the same box: nothing to do
    assert lv.set_kill_box(None) and not lv.outside_cells.any()


def test_agents_map_marks_outside_cells_as_hazard():
    lv = make_level(flat(), [100.0, 700.0, 50.0, 300.0], goal=(37, 2))   # goal cell x 750: off screen
    g = lv.grid_with_hazards([])
    open_out = lv.outside_cells & ~lv.solid
    open_out[2, 37] = False
    assert (g[open_out] == C_HAZARD).all()
    assert g[2, 37] == C_GOAL                                         # a goal stays a goal
    assert (g[lv.solid & lv.outside_cells] == C_SOLID).all()          # walls stay walls
    inside = ~lv.outside_cells & ~lv.solid
    assert (g[inside] != C_HAZARD).all()
    assert (make_level(flat()).grid_with_hazards([]) != C_HAZARD).all()


def tunnel_map():
    """Ground rows 0-5 with a tunnel through it (rows 2-3, cols 8-32, shafts at cols 8-9 and 31-32 up to the
    surface) and a wall 12 tiles high on the surface (cols 19-21, rows 6-17). Spawn (3, 6), goal (37, 6): the
    way under the wall (the tunnel) is shorter than climbing over it."""
    w = np.zeros((24, 40), np.uint32)
    w[0:6, :] = 1
    w[2:4, 8:33] = 0
    w[2:6, 8:10] = 0
    w[2:6, 31:33] = 0
    w[6:18, 19:22] = 1
    return w


def route_of(lv):
    pl = Planner(lv.words0, lv.ox, lv.oy, lv.spawn, lv.goals, [], [], [], [], [],
                 outside=lv.outside_cells if lv.kill_box is not None else None)
    routes = pl.find(1)
    assert routes
    return routes[0]


def test_routes_do_not_pass_outside_the_kill_box():
    words = tunnel_map()
    free = route_of(make_level(words))
    assert any(r <= 3 for _, r in free.cells)                         # without a kill box: through the tunnel
    kb = [-10.0, 810.0, 80.0, 490.0]                                  # rows 0-3 (centres below y 80) are off screen
    lv = make_level(words, kb)
    assert lv.outside_cells[2:4, 8:33].all() and not lv.outside_cells[4:, :].any()
    rt = route_of(lv)
    assert not any(lv.outside_cells[r, c] for c, r in rt.cells)
    assert max(r for _, r in rt.cells) >= 18                          # over the wall instead
    # the sealed pockets treat off-screen cells as closed (like a hazard): the shafts' dead ends left in the
    # map are reachable from the surface, so nothing inside the box is sealed, and nothing outside is marked
    lv.set_sealed([], [lv.spawn] + list(lv.goals))
    assert not (lv.sealed_haz & lv.outside_cells).any() and not (lv.sealed_wall & lv.outside_cells).any()
    assert not lv.sealed_haz[4:6, 8:10].any()


def test_a_pocket_reached_only_through_off_screen_cells_is_sealed():
    w = np.zeros((24, 40), np.uint32)
    w[0:6, :] = 1
    w[6:12, 30:40] = 1
    w[7:10, 32:38] = 0                                 # a closed room in the block ...
    w[7, 38:40] = 0                                    # ... opening only to the right edge (col 39)
    lv = make_level(w, spawn=(3, 6), goal=(10, 6))
    lv.set_sealed([], [lv.spawn] + list(lv.goals))
    assert lv.sealed_wall[7:10, 32:38].all()           # closed in by walls (no way in from the seeds)
    lv = make_level(w, [-10.0, 770.0, -60.0, 490.0], spawn=(3, 6), goal=(10, 6))   # col 38-39 off screen
    lv.set_sealed([], [lv.spawn] + list(lv.goals))
    assert lv.sealed_haz[7:10, 32:38].all()            # touches the off-screen cells: hazard, like a lava pit


def test_safety_check_uses_the_kill_box():
    plain = SimpleNamespace(ox=0.0, oy=0.0, W=50, H=20)
    boxed = SimpleNamespace(ox=0.0, oy=0.0, W=50, H=20, kill_box=(100.0, 700.0, 50.0, 300.0))
    assert S.level_bounds(plain) == (-10.0, 1010.0, -60.0, 410.0)
    assert S.level_bounds(boxed) == (100.0, 700.0, 50.0, 300.0)
    assert not S.out_of_level(plain, [90.0], [200.0])[0] and S.out_of_level(boxed, [90.0], [200.0])[0]
    assert S.out_clearance(S.level_bounds(boxed), [400.0], [290.0])[0] == pytest.approx(10.0)
    kind, _ = S.segment_contact([400.0], [200.0], [400.0], [320.0], S.frame_boxes([]), S.level_bounds(boxed))
    assert kind[0] == S.KIND_OUT
    kind, _ = S.segment_contact([400.0], [200.0], [400.0], [320.0], S.frame_boxes([]), S.level_bounds(plain))
    assert kind[0] == S.KIND_NONE
    # a fall out of the bottom of the box within the veto horizon is vetoed
    ys = np.array([200.0, 120.0, 60.0, 30.0])
    xs = np.full_like(ys, 400.0)
    sf = S.check(xs[None, None], ys[None, None], [None] + [S.frame_boxes([])] * 3, S.level_bounds(boxed))
    assert sf.veto[0] and sf.veto_id[0] == S.OUT_ID


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "levels", f"{LEVEL}.npz")),
                    reason="recorded level data not found")
def test_recorded_level_without_kill_box_is_unchanged():
    """Identity: a recorded level without a kill box, and with one equal to the older rule's alive area (every
    cell centre inside), gives the same map, sealed pockets, routes and out-of-level answers as before."""
    from wm.buffer import load_level
    words, meta = load_level(DATA, LEVEL)
    info = dict(meta, name=LEVEL)
    info.pop("kb", None)
    a = Level(info, words.astype("<u4").tobytes())
    edges = alive_box(a.ox, a.oy, a.W, a.H)
    b = Level(dict(info, kb=list(edges)), words.astype("<u4").tobytes())
    assert a.kill_box is None and b.kill_box == edges
    assert not a.outside_cells.any() and not b.outside_cells.any()
    seeds = [a.spawn] + list(a.goals) + list(a.keys) + list(a.bandages) + [p for pr in a.portals for p in pr]
    assert a.set_sealed([], seeds) == b.set_sealed([], seeds)
    assert (a.sealed_haz == b.sealed_haz).all() and (a.sealed_wall == b.sealed_wall).all()
    boxes = [(a.spawn[0] + 60.0, a.spawn[1], 12.0, 12.0, 0.3)]
    assert (a.grid_with_hazards(boxes) == b.grid_with_hazards(boxes)).all()
    rng = np.random.default_rng(1)
    xs = rng.uniform(a.ox - 100, a.ox + a.W * T + 100, 500)
    ys = rng.uniform(a.oy - 100, a.oy + a.H * T + 100, 500)
    assert (a.out_of_level(xs, ys) == b.out_of_level(xs, ys)).all()
    assert (a.out_of_level(xs, ys) == S.out_of_level(SimpleNamespace(ox=a.ox, oy=a.oy, W=a.W, H=a.H), xs, ys)).all()
    args = (a.words0, a.ox, a.oy, a.spawn, a.goals, [], [], a.keys, a.bandages, list(a.portals))
    kw = {"blocks": a.sealed_haz | a.sealed_wall}
    p0 = Planner(*args, **kw)
    p1 = Planner(*args, **kw, outside=b.outside_cells)
    assert (p0.pit == p1.pit).all()
    r0, r1 = p0.find(2), p1.find(2)
    assert [r.cells for r in r0] == [r.cells for r in r1] and r0
