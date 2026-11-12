"""Tile shapes: one owner (realpath.tile_shape) and its consumers (monitor drawing, world-model protocols, PPO
--shape-inputs). Offscreen and offline only: no window, game, training run or live agent."""
import hashlib
import os
import sys
import types

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from plan.geometry import Terrain                                                  # noqa: E402
from realpath import (CORNER_SIGN, SHAPE_EMPTY, SHAPE_FULL, SHAPE_HALF, SHAPE_SLOPE, SHAPE_SPECIAL,   # noqa: E402
                      half_normal, slope_open_corner, solid_polygon, tile_shape)
from wm import features as F                                                       # noqa: E402
from wm.protocol import (FEATURE_PROTOCOL, KEY_TILES_PROTOCOL, LEGACY_FEATURE_PROTOCOL, TILE_SHAPES_PROTOCOL,   # noqa: E402
                         UNTRAINED_FEATURE_PROTOCOL, feature_usage, input_protocol, require_feature_protocol)

T = 20.0
LOCK = 0xc00163                       # a lock block word (tests/test_key_features.py)


def word(shape, rot, flip, tid=1):
    return tid | (rot << 12) | (flip << 14) | (shape << 18)


def oriented(shape):
    """Every rotation / flip of one shape (1 half block, 2 slope) as a 1 x 16 row of tile words."""
    return np.array([[word(shape, rot, flip) for rot in range(4) for flip in range(4)]], np.uint32)


# ---- the owner ---------------------------------------------------------------------------------------------
def test_slopes_open_toward_their_open_corner_all_four():
    w = oriented(2)
    t = tile_shape(w)
    oc = slope_open_corner(w)
    assert (t.kind == SHAPE_SLOPE).all() and set(oc.ravel().tolist()) == {0, 1, 2, 3}
    for c in range(w.shape[1]):
        assert (t.ox[0, c], t.oy[0, c]) == CORNER_SIGN[oc[0, c]]


def test_half_blocks_open_away_from_their_solid_half_all_four():
    w = oriented(1)
    t = tile_shape(w)
    nx, ny = half_normal(w)
    assert (t.kind == SHAPE_HALF).all()
    assert {(int(a), int(b)) for a, b in zip(t.ox.ravel(), t.oy.ravel())} == {(1, 0), (-1, 0), (0, 1), (0, -1)}
    assert np.array_equal(t.ox, -nx) and np.array_equal(t.oy, -ny)


def test_other_kinds_have_no_open_direction():
    w = np.array([[0, 1, LOCK, word(2, 1, 0) | 0x100, word(1, 0, 0, tid=150)]], np.uint32)
    t = tile_shape(w)
    assert t.kind.tolist() == [[SHAPE_EMPTY, SHAPE_FULL, SHAPE_SPECIAL, SHAPE_SPECIAL, SHAPE_EMPTY]]
    assert not t.ox.any() and not t.oy.any()


@pytest.mark.parametrize("shape", [1, 2])
def test_open_direction_agrees_with_point_solid(shape):
    """In each cell: a point toward the open direction is not solid, the point opposite it is."""
    w = oriented(shape)
    t = tile_shape(w)
    ter = Terrain(w, 0.0, 0.0)
    c = np.arange(w.shape[1])
    ox, oy = t.ox[0].astype(float), t.oy[0].astype(float)
    cx, cy = (c + 0.5) * T, 0.5 * T
    d = 0.4 * T if shape == 2 else 0.3 * T                 # slopes: near the corner; halves: inside the half
    assert not ter.point_solid(cx + d * ox, cy + d * oy).any()
    assert ter.point_solid(cx - d * ox, cy - d * oy).all()


def area(poly):
    return 0.5 * abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1])))


@pytest.mark.parametrize("shape", [1, 2])
def test_solid_polygon_is_the_point_solid_part(shape):
    w = oriented(shape)
    t = tile_shape(w)
    ter = Terrain(w, 0.0, 0.0)
    for c in range(w.shape[1]):
        poly = solid_polygon(t.kind[0, c], t.ox[0, c], t.oy[0, c])
        assert area(poly) == 0.5
        u = sum(p[0] for p in poly) / len(poly)
        v = sum(p[1] for p in poly) / len(poly)
        assert ter.point_solid((c + u) * T, v * T)                 # its centroid is solid
        assert ((1 + t.ox[0, c]) / 2, (1 + t.oy[0, c]) / 2) not in poly or shape == 1
    assert area(solid_polygon(SHAPE_FULL, 0, 0)) == area(solid_polygon(SHAPE_SPECIAL, 0, 0)) == 1.0
    assert solid_polygon(SHAPE_EMPTY, 0, 0) == []


# ---- monitor -----------------------------------------------------------------------------------------------
def synthetic_level(words):
    from game.level import Level
    H, W = words.shape
    info = {"name": "synthetic-shapes", "W": W, "H": H, "ox": 0.0, "oy": 0.0, "goals": [[(W - 1.5) * T, 2.5 * T]],
            "spawn": [1.5 * T, 2.5 * T]}
    return Level(info, words.astype("<u4").tobytes())


def shape_words():
    w = np.zeros((6, 8), np.uint32)
    w[0, :] = 1
    w[1, 2] = word(2, 0, 0)                 # slope
    w[1, 4] = word(1, 0, 0)                 # half block
    return w


def test_monitor_draws_only_the_solid_part_offscreen():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pytest.importorskip("pygame")
    import monitor as M
    lv = synthetic_level(shape_words())
    fake = types.SimpleNamespace(t=types.SimpleNamespace(level=lv))
    fake.pt = lambda x, y: M.Monitor.pt(fake, x, y)
    surf = M.Monitor._level_bg(fake, lv)
    ts = T * fake.sc
    t = tile_shape(lv.words)

    def px(c, r, u, v):                    # screen pixel at (u, v) inside cell (c, r), v up
        return tuple(surf.get_at((int(fake.ox0 + (c + u) * ts), int(fake.oy0 + (lv.H - 1 - r + 1 - v) * ts))))[:3]
    for c in (2, 4):
        ox, oy = int(t.ox[1, c]), int(t.oy[1, c])
        assert px(c, 1, 0.5 + 0.35 * ox, 0.5 + 0.35 * oy) == M.BG
        assert px(c, 1, 0.5 - 0.35 * ox, 0.5 - 0.35 * oy) == M.SOLID
    assert px(0, 0, 0.5, 0.5) == M.SOLID
    pts = M.cell_polygon(10.0, 20.0, 4.0, SHAPE_SLOPE, 1, 1)          # open top-right: screen y down
    assert pts == [(10.0, 24.0), (14.0, 24.0), (10.0, 20.0)]


def test_agent_viewer_draws_only_the_solid_part_offscreen():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pg = pytest.importorskip("pygame")
    from viz import viewer as V
    w = shape_words()
    lvl = synthetic_level(w)
    lv = V.LevelView("synthetic", lvl.W, lvl.H, 0.0, 0.0, lvl.solid.astype(np.uint8), w, None, (0, 0, 0, 0), [],
                     None, [], [], [], 1, 0, None, ("synthetic", 1))
    assert [(r, c) for r, c, *_ in lv.shape_cells()] == [(1, 2), (1, 4)]
    R = V.Renderer.__new__(V.Renderer)
    R.pg, R._small, R._tiles = pg, None, None
    rect = (0, 0, 400, 300)
    m = V.Mapper(rect, lv.bounds())
    surf = pg.Surface((400, 300))
    surf.fill(V.BG)
    R.draw_tiles(surf, lv, m, rect)
    t = tile_shape(w)

    def px(c, r, u, v):                    # screen pixel at (u, v) inside cell (c, r), v up
        x, y = m.pt((c + u) * T, (r + v) * T)
        return tuple(surf.get_at((int(x), int(y))))[:3]
    for c in (2, 4):
        ox, oy = int(t.ox[1, c]), int(t.oy[1, c])
        assert px(c, 1, 0.5 + 0.35 * ox, 0.5 + 0.35 * oy) == V.BG
        assert px(c, 1, 0.5 - 0.35 * ox, 0.5 - 0.35 * oy) == V.SOLID
    assert px(0, 0, 0.5, 0.5) == V.SOLID


# ---- world-model feature protocols ----------------------------------------------------------------------------
def test_protocol_selection_is_recorded_never_guessed():
    assert input_protocol({}) == input_protocol(None) == KEY_TILES_PROTOCOL == FEATURE_PROTOCOL
    assert input_protocol({"feature_protocol": KEY_TILES_PROTOCOL}) == KEY_TILES_PROTOCOL
    assert input_protocol({"feature_protocol": TILE_SHAPES_PROTOCOL}) == TILE_SHAPES_PROTOCOL
    assert input_protocol({"feature_protocol": UNTRAINED_FEATURE_PROTOCOL,
                           "active_feature_protocol": TILE_SHAPES_PROTOCOL}) == TILE_SHAPES_PROTOCOL
    assert feature_usage({"feature_protocol": LEGACY_FEATURE_PROTOCOL, "active_feature_protocol": KEY_TILES_PROTOCOL}) \
        == {"active": KEY_TILES_PROTOCOL, "checkpoint_trained": LEGACY_FEATURE_PROTOCOL}
    for bad in ({"feature_protocol": "tile-shapes-v9"}, {"feature_protocol": UNTRAINED_FEATURE_PROTOCOL},
                {"feature_protocol": TILE_SHAPES_PROTOCOL, "active_feature_protocol": KEY_TILES_PROTOCOL},
                {"active_feature_protocol": TILE_SHAPES_PROTOCOL}):
        with pytest.raises(ValueError, match="protocol"):
            input_protocol(bad)
    require_feature_protocol({"feature_protocol": TILE_SHAPES_PROTOCOL})
    with pytest.raises(ValueError, match="feature protocol"):
        require_feature_protocol({})


def test_model_layout_checks_inputs_against_the_recorded_protocol():
    from wm.model import WorldModel
    old = WorldModel(members=1, hidden=8, layers=1)
    old.info = {}
    new = WorldModel(F.layout(TILE_SHAPES_PROTOCOL).n_in(4), members=1, hidden=8, layers=1, step=4)
    new.info = {"feature_protocol": TILE_SHAPES_PROTOCOL}
    assert F.model_layout(old).feat_dim == F.FEAT_DIM == 291
    assert F.model_layout(new).feat_dim == 389 and new.mu.numel() == 419
    unlabelled = WorldModel(F.layout(TILE_SHAPES_PROTOCOL).feat_dim, members=1, hidden=8, layers=1)
    unlabelled.info = {}
    with pytest.raises(ValueError, match="inputs"):
        F.model_layout(unlabelled)                   # a wider model without a recorded protocol: not guessed
    g_old = F.LevelGeom(shape_words(), {"ox": 0, "oy": 0})
    with pytest.raises(ValueError, match="match"):
        F.require_geom(new, g_old)
    from wm.rollout import rollout_batched
    with pytest.raises(ValueError, match="match"):
        rollout_batched(new, g_old, np.zeros(1, F_ROW()), np.zeros((1, 4), int), np.full(4, 16.667), [None] * 4)


def F_ROW():
    from wm.buffer import ROW
    return ROW


def test_tile_shapes_channels_and_key_masks():
    w = shape_words()
    w[3, 4] = LOCK
    meta = {"ox": 0, "oy": 0, "keys": [[30, 70]]}
    old = F.LevelGeom(w, meta)
    new = F.LevelGeom(w, meta, protocol=TILE_SHAPES_PROTOCOL)
    t = tile_shape(w)
    assert new.ch.shape == (6,) + w.shape
    assert np.array_equal(new.ch[4], t.ox) and np.array_equal(new.ch[5], t.oy)
    assert np.array_equal(new.ch[:2], old.ch[:2]) and np.array_equal(new.ch[3], old.ch[3])
    assert np.array_equal(new.ch[2] > 0, old.ch[2] > 0) and set(np.unique(new.ch[2])) <= {0.0, 1.0}
    assert (new.pad[0, :3].all() and not new.pad[1:, :3].any())          # outside the map: full, nothing open
    s = np.zeros(2, F_ROW())
    s["x"], s["y"], s["keys"] = 4.5 * T, 2.5 * T, [0, 1]
    X = F.featurize(s, np.zeros(2, int), 16.667, new, np.empty((0, 5)), None, None, .016667, [])
    lay = new.layout
    assert X.shape == (2, 389)
    patch = X[:, lay.patch:lay.haz].reshape(2, 6, 7, 7)
    assert patch[0, 3, 4, 3] == 1 and not patch[1, :, 4, 3].any()        # the lock opens for key mask 1 only
    assert patch[0, 4, 2, 3] == t.ox[1, 4] and patch[0, 5, 2, 3] == t.oy[1, 4]   # the half block below
    assert np.array_equal(X[:, lay.haz:], F.featurize(s, np.zeros(2, int), 16.667, old, np.empty((0, 5)), None,
                                                       None, .016667, [])[:, F._C_HAZ:])
    Xk = F.featurize_k(s, np.zeros((2, 4), int), 66.668, new, np.empty((0, 5)), None, None, .016667, [])
    assert Xk.shape == (2, lay.n_in(4))
    new.set_tile(3 * w.shape[1] + 4, 0)                    # the lock is gone for everyone
    assert not new.patches(s["x"][:1], s["y"][:1], [0]).reshape(6, 7, 7)[:, 4, 3].any()


DATA_2_10 = os.path.join(HERE, "runs", "collect_2-10_explore_01", "wm_data")
DATA_1_3 = os.path.join(HERE, "runs", "demo_1-3_20261108-171442", "wm_data")


@pytest.mark.parametrize("data, step, rows, digest", [
    (DATA_2_10, 4, 5474, "6befca3a27c224f62c99ec9df7fb01ad986b5f9cddd245e3b2179d5d0ba68344"),
    (DATA_1_3, 1, 7128, "75e09e7b84280378ec9e8aa09905e442b29b91b82080a83fc6dcffc9c8465ccb"),
    (DATA_1_3, 4, 7056, "4c6f85d62bea09b016ec584eafd433126e5bb6dd659229ba5630a70bb5e8c4a0"),
])
def test_old_protocol_features_are_byte_identical_on_recorded_data(data, step, rows, digest):
    """X of the old (per-agent-key-tiles-v1 / legacy) encoding, hashed before the tile-shapes change (1-3 has
    19 slope cells in all four orientations; 2-10's hash is the key-feature audit's)."""
    if not os.path.isdir(data):
        pytest.skip("recorded data not present")
    d = F.build([data], step=step, log=lambda *a: None)
    assert d["X"].shape[0] == rows
    assert hashlib.sha256(np.ascontiguousarray(d["X"]).tobytes()).hexdigest() == digest
    if data == DATA_1_3:
        new = F.build([data], step=step, log=lambda *a: None, protocol=TILE_SHAPES_PROTOCOL)
        lay = F.layout(TILE_SHAPES_PROTOCOL)
        assert new["X"].shape == (rows, lay.n_in(step))
        assert np.array_equal(new["X"][:, :lay.patch], d["X"][:, :lay.patch])
        assert np.array_equal(new["X"][:, lay.haz:], d["X"][:, F._C_HAZ:])
        assert np.abs(new["X"][:, lay.patch:lay.haz]).reshape(rows, 6, 49)[:, 4:].any()


def test_fresh_training_records_opted_in_protocol_and_init_keeps_it(tmp_path, monkeypatch):
    from test_key_features import recording
    from wm import train as T_
    from wm.model import WorldModel
    data = recording(tmp_path / "data")
    monkeypatch.setattr(T_, "immediate", lambda: None)
    out = tmp_path / "shapes"
    monkeypatch.setattr(sys, "argv", ["wm.train", str(data), "--legacy-split", "--out", str(out), "--epochs", "1",
                                      "--members", "1", "--hidden", "8", "--layers", "1", "--sample-frac", "1",
                                      "--feature-protocol", TILE_SHAPES_PROTOCOL])
    T_.main()
    m = WorldModel.load(out / "wm_model.pt")
    assert m.info["feature_protocol"] == m.info["active_feature_protocol"] == TILE_SHAPES_PROTOCOL
    assert F.model_layout(m).protocol == TILE_SHAPES_PROTOCOL and m.mu.numel() == 389
    monkeypatch.setattr(sys, "argv", ["wm.train", str(data), "--legacy-split", "--init", str(out / "wm_model.pt"),
                                      "--out", str(tmp_path / "x"), "--feature-protocol", KEY_TILES_PROTOCOL])
    with pytest.raises(ValueError, match="differs"):
        T_.main()


# ---- PPO --shape-inputs ------------------------------------------------------------------------------------
def test_ppo_shape_inputs_are_appended_and_off_by_default():
    import realgame as rg
    lv = synthetic_level(shape_words())
    grid = lv.grid_with_hazards([])
    s = [4.5 * T, 2.5 * T, 0.0, 0.0, 0, 0, 0, 0]
    off = rg.observe(lv, grid, s, None, 0.0, 0.0)
    on = rg.observe(lv, grid, s, None, 0.0, 0.0, shape_inputs=True)
    assert off.shape == (rg.OBS,) == (rg.obs_size(),) and on.shape == (rg.obs_size(True),) == (rg.OBS + 242,)
    assert np.array_equal(on[:rg.OBS], off)
    n = 2 * rg.K + 1
    ox, oy = on[rg.OBS:rg.OBS + n * n].reshape(n, n), on[rg.OBS + n * n:].reshape(n, n)
    t = tile_shape(lv.words)
    # rows top-down: the agent's cell (r 2, c 4) is at [K, K]; the half block (r 1, c 4) one row below it
    assert (ox[rg.K + 1, rg.K], oy[rg.K + 1, rg.K]) == (t.ox[1, 4], t.oy[1, 4]) != (0, 0)
    assert (ox[rg.K + 1, rg.K - 2], oy[rg.K + 1, rg.K - 2]) == (t.ox[1, 2], t.oy[1, 2])
    assert not ox[:rg.K - 2].any()                          # above the map: nothing
    lv.apply_tiles([(1 * lv.W + 4, 1)])                    # the half block becomes a full block live
    on2 = rg.observe(lv, grid, s, None, 0.0, 0.0, shape_inputs=True)
    assert on2[rg.OBS:].reshape(2, n, n)[:, rg.K + 1, rg.K].tolist() == [0, 0]


def test_ppo_old_checkpoint_widens_only_with_the_flag(tmp_path):
    import realgame as rg
    old = rg.Policy()
    p = tmp_path / "model.pt"
    torch.save({"policy": old.state_dict(), "obs": rg.OBS, "actions": rg.ACTIONS}, p)
    same = rg.Policy()
    assert rg.load_policy(same, str(p)) is False
    assert all(torch.equal(a, b) for a, b in zip(same.state_dict().values(), old.state_dict().values()))
    wide = rg.Policy(rg.obs_size(True))
    assert rg.load_policy(wide, str(p)) is True
    assert torch.equal(wide.pi[0].weight[:, :rg.OBS], old.pi[0].weight)
    assert torch.equal(wide.vf[0].weight[:, :rg.OBS], old.vf[0].weight)
    q = tmp_path / "shapes.pt"
    torch.save({"policy": wide.state_dict(), "obs": rg.obs_size(True), "actions": rg.ACTIONS}, q)
    with pytest.raises(ValueError, match="shape-inputs"):
        rg.load_policy(rg.Policy(), str(q))
