"""Moving platforms at each imagined step (plan/planner.py, module docstring): the support rule, the map-bottom
rule and the collision guard see a bar where the step's world has it (bars_at), not where it was at the
decision. Stubs and a synthetic fixed-output model, no learned checkpoint."""
import types

import numpy as np
import torch

from game.level import platform_boxes
from plan import planner as P
from plan import safety as S
from plan.geometry import Terrain
from realpath import BODY
from wm import features as F
from wm import rollout as R
from wm.buffer import ROW
from wm.model import WorldModel
from wm.protocol import KEY_TILES_PROTOCOL


def moving(x0=150.0, y=0.0, vx=2.0, n=40, upright=0):
    """Worlds of one bar moving right by vx units per frame (worlds[k]: frame k)."""
    return [R.World(pf=[[x0 + vx * k, y, upright]]) for k in range(n)]


def stub(words, plats=()):
    ter = Terrain(words, 0.0, 0.0)
    return types.SimpleNamespace(terrain=ter, support_terrain=lambda keys=0: ter, lv=types.SimpleNamespace(plats=list(plats)),
                                 floor=object())


def test_bars_at_reads_the_steps_world_and_clamps():
    w = moving()
    assert P.bars_at(w, 0) == platform_boxes([[150.0, 0.0, 0]])
    assert P.bars_at(w, 16) == platform_boxes([[182.0, 0.0, 0]])
    assert P.bars_at(w, 500) == P.bars_at(w, 39) and P.bars_at(w, -3) == P.bars_at(w, 0)
    assert P.bars_at([R.World()] * 3, 1) == [] and P.bars_at(None, 0) == []
    assert P.has_bars(w) and not P.has_bars([R.World()] * 3) and not P.has_bars(None)


def test_a_landing_on_a_bar_that_moved_away_is_not_ground():
    """The 4-4 f440 phantom in small: a body standing where the bar was at the decision (frame 0) is on the
    ground by the live bars, and at frame 0 of the worlds; at frame 16 the bar has moved 32 units right and the
    same point has nothing under it."""
    pl = stub(np.zeros((10, 30), np.uint32), plats=P.bars_at(moving(), 0))
    x, y = np.array([130.0]), np.array([10.0 + BODY])              # feet on the bar's top (y 10), its left part
    assert P.Planner.support(pl, x, y)[1].tolist() == [True]                         # live bars: ground
    assert P.Planner.support(pl, x, y, at=0, worlds=moving())[1].tolist() == [True]
    sup, g, side = P.Planner.support(pl, x, y, at=16, worlds=moving())
    assert g.tolist() == [False] and sup.tolist() == [False]
    # one call, positions at different frames: each with its own bars
    xs, ys = np.full(3, 130.0), np.full(3, 10.0 + BODY)
    assert P.Planner.support(pl, xs, ys, at=np.array([0, 4, 16]), worlds=moving())[1].tolist() == [True, True, False]
    # the bar where it is at frame 16 does carry a body there
    assert P.Planner.support(pl, np.array([182.0]), y, at=16, worlds=moving())[1].tolist() == [True]


def test_without_bars_support_is_the_tiles_alone():
    w = np.zeros((10, 30), np.uint32)
    w[0, :] = 1                                                     # floor, top y 20
    pl = stub(w)
    x, y = np.array([100.0, 100.0]), np.array([20.0 + BODY, 60.0])
    worlds = [R.World()] * 10
    a = P.Planner.support(pl, x, y)
    b = P.Planner.support(pl, x, y, at=5, worlds=worlds)
    for u, v in zip(a, b):
        assert np.array_equal(u, v)
    assert a[1].tolist() == [True, False]
    # worlds without bars never fall back to the live bars (stale between frames)
    pl2 = stub(w, plats=P.bars_at(moving(x0=100.0, y=40.0), 0))
    assert P.Planner.support(pl2, np.array([100.0]), np.array([50.0 + BODY]), at=0, worlds=worlds)[1].tolist() == [False]
    fl = P.Planner.floor_for(pl, worlds)
    assert not hasattr(fl, "at") and fl.plats is None


def test_the_map_bottom_rule_sees_the_bar_of_each_column():
    """A body at the map's bottom (the model's false floor) is on a bar while the bar is under it; once the
    column's bar has moved away it stands on the padding: out of the level from that column on."""
    pl = stub(np.zeros((10, 30), np.uint32))
    worlds = moving(x0=100.0, y=-8.0, vx=10.0)                      # top y 2; x 100, 110, 120, ...
    xs, ys = np.full((1, 1, 6), 100.0), np.full((1, 1, 6), BODY)  # standing still at y 6
    sw = S.sweep(xs, ys, [None] * 6, None, P.Planner.floor_for(pl, worlds), 1)
    # the bar covers x 70..130 (feet 94.5..105.5) up to frame 3 (x 130: 100..160), not at frame 4 (140: 110..170)
    assert int(sw.contact_frame[0, 0]) == 4 and int(sw.contact_kind[0, 0]) == S.KIND_FLOOR
    sw2 = S.sweep(xs, ys, [None] * 6, None, P.Planner.floor_for(pl, worlds, 2), 1)   # columns from frame 2
    assert int(sw2.contact_frame[0, 0]) == 2


def _rows(xy, v=(0.0, 0.0)):
    r = np.zeros(len(xy), ROW)
    r["x"], r["y"] = [p[0] for p in xy], [p[1] for p in xy]
    r["vx"], r["vy"] = v
    return r


def test_collide_bars_puts_a_body_falling_onto_the_bar_back_on_its_top():
    top = 10.0
    bar = P.bars_at(moving(), 0)                                    # x 120..180, y -10..10
    # falling through the top within the bar's x extent: back on top, vy 0, x kept
    prev, s = _rows([(150.0, 20.0)]), _rows([(153.0, 8.0)], (180.0, -400.0))
    assert P.collide_bars(prev, s, bar, bar).tolist() == [True]
    assert s["x"][0] == 153.0 and abs(s["y"][0] - (top + BODY - P.CONTACT_EPS)) < 1e-4
    assert s["vy"][0] == 0.0 and s["vx"][0] == 180.0
    # through the whole bar in one step: still caught by the top
    prev, s = _rows([(150.0, 20.0)]), _rows([(150.0, -30.0)], (0.0, -900.0))
    assert P.collide_bars(prev, s, bar, bar).tolist() == [True] and abs(s["y"][0] - (top + BODY - P.CONTACT_EPS)) < 1e-4
    # within the tolerance, beside the bar, or coming from below: left alone (sides and bottom are open)
    prev = _rows([(150.0, 17.0), (115.0, 20.0), (150.0, -20.0)])
    s = _rows([(150.0, 15.5), (110.0, 8.0), (150.0, 5.0)])
    s0 = s.copy()
    assert P.collide_bars(prev, s, bar, bar).tolist() == [False, False, False] and np.array_equal(s, s0)
    # the bar at the step's end decides: it moved away under the body
    gone = P.bars_at(moving(), 40)                                  # x 198 at the clamp (frame 39)
    prev, s = _rows([(130.0, 17.0)]), _rows([(130.0, 8.0)])
    assert P.collide_bars(prev, s, bar, gone).tolist() == [False]
    assert P.collide_bars(_rows([(0.0, 0.0)]), _rows([(0.0, 0.0)]), [], []).tolist() == [False]


def _falling_model(dy=-5.0):
    """A one-member model with zero weights: every step moves dy (plus its gravity prior), whatever the inputs."""
    model = WorldModel(n_in=F.layout(KEY_TILES_PROTOCOL).n_in(1), members=1, layers=0, step=1, res_scale=np.ones(4))
    model.info = {"feature_protocol": KEY_TILES_PROTOCOL}
    with torch.no_grad():
        m = model.members[0][0]
        m.weight.zero_()
        m.bias.zero_()
        m.bias[1] = dy
        m.bias[8:8 + len(F.BIN_NAMES)] = -100
    return model


def _floor_level():
    words = np.zeros((10, 20), np.uint32)
    words[0, :] = 1                                                 # floor, top y 20
    return words, F.LevelGeom(words, {"ox": 0, "oy": 0, "keys": []}), Terrain(words, 0, 0, slopes=False)


def test_the_guard_puts_each_row_on_its_own_steps_bar():
    """imagine(): rows starting at different frames (stage 2's offsets) see the bar of their own steps: the row
    from frame 0 lands on the bar under it (top y 110); the row from frame 20, whose bar has moved 60 units
    right by then, falls through where the bar was at frame 0."""
    words, geom, ter = _floor_level()
    model = _falling_model()
    s0 = np.zeros(2, ROW)
    s0["x"], s0["y"] = 100.0, 120.0
    worlds = moving(x0=100.0, y=100.0, vx=3.0)
    out = P.imagine(model, slice(None), geom, s0, np.zeros((2, 8), int), 16.667, worlds, offsets=[0, 20],
                    terrain=ter)
    y0, y1 = out["y"][0, 0], out["y"][0, 1]
    assert abs(y0[-1] - (110.0 + BODY - P.CONTACT_EPS)) < 1e-4 and out["clip"][0, 0, -1]
    assert y1[-1] < 100.0 and not out["clip"][0, 1].any()       # through where the bar was at frame 0
    # the guard off: nothing caught by the bar either
    off = P.imagine(model, slice(None), geom, s0, np.zeros((2, 8), int), 16.667, worlds, offsets=[0, 20])
    assert off["clip"] is None and off["y"][0, 0, -1] < 100.0


def test_without_bars_the_guard_is_the_tiles_alone():
    """Bars that no row comes near change nothing; worlds without bars run the tile guard only (the fast path)."""
    words, geom, ter = _floor_level()
    model = _falling_model()
    s0 = np.zeros(3, ROW)
    s0["x"], s0["y"] = [60.0, 100.0, 300.0], [120.0, 60.0, 40.0]
    acts = np.zeros((3, 12), int)
    none = P.imagine(model, slice(None), geom, s0, acts, 16.667, [R.World()] * 20, offsets=[0, 4, 8], terrain=ter)
    far = P.imagine(model, slice(None), geom, s0, acts, 16.667, moving(x0=2000.0, y=500.0), offsets=[0, 4, 8],
                    terrain=ter)
    for k in ("x", "y", "clip"):
        assert np.array_equal(none[k], far[k]), k
    assert none["clip"].any()                                       # the tile floor did catch them
