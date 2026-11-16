"""Crumble blocks in the planner's imagined world (plan.planner.Planner.crumble): a touched tile is expected gone
CRUMBLE_DELAY frames later, and imagined steps from then on see it open (the guard's tiles and the model's tile
inputs). Synthetic fixed model outputs, no learned checkpoint."""
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from plan import planner as P                                              # noqa: E402
from plan.geometry import Terrain                                          # noqa: E402
from wm import features as F                                               # noqa: E402
from wm.buffer import ROW                                                  # noqa: E402
from wm.model import WorldModel                                            # noqa: E402
from wm.protocol import KEY_TILES_PROTOCOL                                 # noqa: E402

CRUMBLE = 0x3000164                         # a crumble block's word (3-1's wall)
WALL = [(1, 4), (2, 4)]                     # cells (row, col): x 80..100, y 20..60


def level():
    words = np.zeros((7, 12), np.uint32)
    for r, c in WALL:
        words[r, c] = CRUMBLE
    return words, {"ox": 0, "oy": 0, "keys": []}


def planner():
    """A Planner with only what the crumble clock and imagine_options read."""
    words, meta = level()
    pl = P.Planner.__new__(P.Planner)
    pl.lv = types.SimpleNamespace(ox=0.0, oy=0.0, words0=words, lock_groups=None)
    pl.geom = F.LevelGeom(words, meta)
    pl.words, pl._words0 = words.copy(), words
    pl.terrain, pl.guard = Terrain(words, 0, 0), Terrain(words, 0, 0, slopes=False)
    pl._guard_keys, pl._support_keys = {}, {}
    pl.collide_tol, pl.key_pickup = P.COLLIDE_TOL, False
    pl._crumble = P.tile_classes(words)["crumble"]
    pl.crumble, pl._stages, pl._now = {}, {}, 0
    pl.route = P.RouteBook(pl.lv)                               # Planner.tiles keeps its tile signature
    return pl


def walker(speed=4.0):
    """One member that moves `speed` units right per frame whatever it sees."""
    model = WorldModel(n_in=F.layout(KEY_TILES_PROTOCOL).n_in(1), members=1, layers=0, step=1, res_scale=np.ones(4))
    model.info = {"feature_protocol": KEY_TILES_PROTOCOL}
    with torch.no_grad():
        m = model.members[0]
        m[0].weight.zero_()
        m[0].bias.zero_()
        m[0].bias[0] = speed
        m[0].bias[8:8 + len(F.BIN_NAMES)] = -100
    return model.eval()


def start(x=60.0, y=30.0):
    s = np.zeros(1, ROW)
    s["x"], s["y"] = x, y
    return s


WORLDS = [P.World(np.zeros((0, 5)), None, None, 1 / 60, [])] * 40


def test_a_touch_schedules_the_tile_and_the_game_or_time_clears_it():
    pl = planner()
    pl.touch(100, 60.0, 30.0)                               # the body's box ends at x 66: 14 units from the wall
    assert pl.crumble == {}
    pl.touch(101, 73.5, 30.0)                               # box to 79.5, within CRUMBLE_REACH of x 80; rows 1..1
    assert pl.crumble == {(1, 4): 101 + P.CRUMBLE_DELAY}
    pl.touch(105, 73.5, 38.0)                               # sliding up: row 2's tile too, the first keeps its time
    assert pl.crumble == {(1, 4): 131, (2, 4): 135}
    st = pl.crumble_stages(0, 120)
    assert [t for t, *_ in st] == [11, 15]                   # frames from the decision's frame
    assert st[0][1].cells == ((1, 4),) and st[1][1].cells == ((1, 4), (2, 4))
    assert st[0][2].point_solid(np.array([90.0]), np.array([50.0]))[0]         # row 2 still there in stage 1
    assert not st[1][2].point_solid(np.array([90.0]), np.array([50.0]))[0]
    pl.tiles([(1 * 12 + 4, 0x100)])                          # the game broke row 1's tile: no expectation left
    assert pl.crumble == {(2, 4): 135} and not pl.terrain.point_solid(np.array([90.0]), np.array([30.0]))[0]
    pl.touch(135 + P.CRUMBLE_LATE + 1, 20.0, 30.0)          # row 2's never broke: forgotten
    assert pl.crumble == {} and pl.crumble_stages(0, 200) is None


def test_imagined_steps_pass_the_wall_once_it_is_due():
    pl = planner()
    model = walker()
    acts = np.zeros((1, 16), np.int64)
    shut = P.imagine(model, slice(None), pl.geom, start(), acts, 16.667, WORLDS, **pl.imagine_options(0))
    assert shut["x"][0, 0].max() < 80.0 - P.BODY + 1.0                          # stopped at the wall's face
    pl.touch(0, 73.5, 30.0)                                                    # touched at frame 0: due at 30
    pl._now = 22                                                                # a decision at frame 22: due in 8
    out = P.imagine(model, slice(None), pl.geom, start(), acts, 16.667, WORLDS, **pl.imagine_options(0))
    x = out["x"][0, 0]
    assert x[8] < 80.0 - P.BODY + 1.0 and x[9] > x[8] + 3.9 and x[16] > 100.0   # through from the 9th step
    late = P.imagine(model, slice(None), pl.geom, start(), acts, 16.667, WORLDS, offsets=np.array([8]),
                     **pl.imagine_options(0))                                   # rows that start 8 frames later
    assert late["x"][0, 0, 16] > 120.0                                          # see it open from their first step


def test_the_models_tile_inputs_show_the_due_tile_empty():
    words, meta = level()
    geom = F.LevelGeom(words, meta)
    x, y = np.array([70.0]), np.array([30.0])
    full = geom.patches(x, y, np.zeros(1, np.uint32)).copy()
    gone = geom.opened([(1, 4)]).patches(x, y, np.zeros(1, np.uint32))
    assert full.sum() > gone.sum() > 0                                          # one tile less, row 2's still there
    assert np.array_equal(geom.patches(x, y, np.zeros(1, np.uint32)), full)     # the level's own inputs untouched
    geom.set_tile(1 * 12 + 4, 0x100)                                            # the game's own change: the same
    assert np.array_equal(geom.patches(x, y, np.zeros(1, np.uint32)), gone)
