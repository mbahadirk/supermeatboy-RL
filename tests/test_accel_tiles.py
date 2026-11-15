"""Accelerator tiles: one owner (realpath.accel_bits), game.level's mask from it, and the accelerator channels of
the tile-accel-v1 and tile-plat-v1 input protocols (tile-shapes-v1's channels plus the accelerator and its raw
orientation bits; tile-plat-v1's moving platforms: tests/test_plat_tiles.py). tile-accel-v1 never paints
platforms. Offline only."""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from realpath import ACCEL_TILE, accel_bits, tile_shape, SHAPE_FULL                     # noqa: E402
from wm import features as F                                                         # noqa: E402
from wm.buffer import ROW                                                            # noqa: E402
from wm.protocol import (ACCEL_TILES_PROTOCOL, PLAT_TILES_PROTOCOL, FEATURE_PROTOCOL, INPUT_PROTOCOLS, KEY_TILES_PROTOCOL,   # noqa: E402
                         TILE_SHAPES_PROTOCOL, input_protocol)

T = 20.0
LOCK = 0xc00163
FLOOR_R, FLOOR_R2, FLOOR_L, FLOOR_L2 = 0x800062, 0xc00062, 0x802062, 0xc02062
WALL_B, WALL_A, WALL_AB = 0x2001062, 0x1001062, 0x3001062
DATA_1_3 = os.path.join(HERE, "runs", "demo_1-3_20261108-171442", "wm_data")
LEVEL_3_6 = os.path.join(HERE, "runs", "online_check01", "wm_data", "levels", "3-6.npz")


def bits(word):
    return tuple(int(np.asarray(b).ravel()[0]) for b in accel_bits(np.array([[word]], np.uint32)))


# ---- the owner ---------------------------------------------------------------------------------------------
def test_accel_bits_of_the_measured_and_unmeasured_words():
    # (accel, flip, wall, face_a, face_b)
    assert bits(FLOOR_R) == bits(FLOOR_R2) == (1, 0, 0, 0, 0)
    assert bits(FLOOR_L) == bits(FLOOR_L2) == (1, 1, 0, 0, 0)
    assert bits(WALL_B) == (1, 0, 1, 0, 1)
    assert bits(WALL_A) == (1, 0, 1, 1, 0)
    assert bits(WALL_AB) == (1, 0, 1, 1, 1)
    assert bits(0x100f) == (0, 0, 0, 0, 0)                      # a plain block: nothing
    assert bits(0x2001063) == (0, 0, 0, 0, 0)                   # the bits only count on an accelerator
    assert bits(0x162) == bits(0x2001162) == (0, 0, 0, 0, 0)    # id 98 with the special bit: a switch block
    a = accel_bits(np.array([[FLOOR_R, 0], [WALL_B, 1]], np.uint32))
    assert all(b.shape == (2, 2) and b.dtype == bool for b in a)
    assert ACCEL_TILE == 98


def test_accelerators_stay_full_blocks_in_tile_shape():
    w = np.array([[FLOOR_R, FLOOR_L, WALL_A, WALL_B, WALL_AB]], np.uint32)
    t = tile_shape(w)
    assert (t.kind == SHAPE_FULL).all() and not t.ox.any() and not t.oy.any()


def synthetic_level(words):
    from game.level import Level
    H, W = words.shape
    info = {"name": "synthetic-accel", "W": W, "H": H, "ox": 0.0, "oy": 0.0, "goals": [[(W - 1.5) * T, 2.5 * T]],
            "spawn": [1.5 * T, 2.5 * T]}
    return Level(info, words.astype("<u4").tobytes())


def accel_words():
    w = np.zeros((6, 8), np.uint32)
    w[0, :] = 1
    w[0, 2], w[0, 3] = FLOOR_R, FLOOR_L
    w[1:4, 6] = WALL_B
    w[3, 1] = 0x162                                             # a switch block (id 98, special)
    w[1, 4] = 1 | (2 << 18)                                     # a slope
    w[4, 3] = LOCK
    return w


def test_game_level_uses_the_owner():
    import game.level as GL
    assert GL.ACCEL_TILE == ACCEL_TILE                          # still importable from game.level (realgame.py)
    w = accel_words()
    lv = synthetic_level(w)
    assert lv.accel.dtype == bool and np.array_equal(lv.accel, accel_bits(w).accel)
    assert int(lv.accel.sum()) == 5 and lv.base[0, 2] == GL.C_ACCEL and lv.base[0, 0] == GL.C_SOLID


def test_game_level_accel_mask_on_3_6():
    if not os.path.exists(LEVEL_3_6):
        pytest.skip("3-6 level file not present")
    z = np.load(LEVEL_3_6, allow_pickle=False)
    w = z["words"].astype(np.uint32)
    old = ((w & 0xff) == 98) & ~(((w >> 8) & 1).astype(bool))  # game.level's former rule
    lv = synthetic_level(w)
    assert lv.accel.any() and np.array_equal(lv.accel, accel_bits(w).accel) and np.array_equal(lv.accel, old)


# ---- the input protocol --------------------------------------------------------------------------------------
def test_protocol_is_an_explicit_opt_in():
    assert PLAT_TILES_PROTOCOL == "tile-plat-v1" and PLAT_TILES_PROTOCOL in INPUT_PROTOCOLS
    assert FEATURE_PROTOCOL == KEY_TILES_PROTOCOL
    assert input_protocol({"feature_protocol": PLAT_TILES_PROTOCOL}) == PLAT_TILES_PROTOCOL


def test_tile_accel_stays_loadable_and_never_paints():
    """tile-accel-v1 (runs/experiments/human_v4_gpu): the same columns as tile-plat-v1, no platforms painted."""
    assert ACCEL_TILES_PROTOCOL == "tile-accel-v1" and ACCEL_TILES_PROTOCOL in INPUT_PROTOCOLS
    assert input_protocol({"feature_protocol": ACCEL_TILES_PROTOCOL}) == ACCEL_TILES_PROTOCOL
    la, lp = F.layout(ACCEL_TILES_PROTOCOL), F.layout(PLAT_TILES_PROTOCOL)
    assert (la.channels, la.feat_dim, la.n_in(4)) == (lp.channels, lp.feat_dim, lp.n_in(4)) == (11, 634, 664)
    assert la.accel and lp.accel and not la.platforms and lp.platforms
    w = accel_words()
    acc, plat = F.LevelGeom(w, {"ox": 0, "oy": 0}, protocol=ACCEL_TILES_PROTOCOL), \
        F.LevelGeom(w, {"ox": 0, "oy": 0}, protocol=PLAT_TILES_PROTOCOL)
    s = np.zeros(2, ROW)
    s["x"], s["y"] = 2.5 * T, 3.5 * T
    pf = [[2.5 * T, 2.5 * T, 0]]                                     # a bar right under the agent
    X = lambda g, p: F.featurize(s, np.zeros(2, int), 16.667, g, np.empty((0, 5)), p, p, .016667, [])
    lay = la
    assert X(acc, pf)[:, :lay.plat].tobytes() == X(acc, None)[:, :lay.plat].tobytes()     # never painted
    assert X(acc, None).tobytes() == X(plat, None).tobytes()                              # no bars: the same
    assert not np.array_equal(X(plat, pf)[:, lay.patch:lay.haz], X(plat, None)[:, lay.patch:lay.haz])
    assert np.array_equal(X(acc, pf)[:, lay.haz:], X(plat, pf)[:, lay.haz:])


def test_layout_dims():
    lay = F.layout(PLAT_TILES_PROTOCOL)
    assert lay.channels == 11 and lay.haz - lay.patch == 11 * 49 == 539
    assert lay.feat_dim == 634 and lay.n_in(4) == 664
    assert F.layout(TILE_SHAPES_PROTOCOL).feat_dim == 389 and F.layout(KEY_TILES_PROTOCOL).feat_dim == 291


def test_channels_extend_tile_shapes_unchanged():
    w = accel_words()
    meta = {"ox": 0, "oy": 0, "keys": [[30, 70]]}
    old = F.LevelGeom(w, meta, protocol=TILE_SHAPES_PROTOCOL)
    new = F.LevelGeom(w, meta, protocol=PLAT_TILES_PROTOCOL)
    assert new.ch.shape == (11,) + w.shape
    assert np.array_equal(new.ch[:6], old.ch)
    a = accel_bits(w)
    for k, b in enumerate(a):
        assert np.array_equal(new.ch[6 + k], b.astype(np.float32))
    assert new.ch[6, 0, 2] == 1 and new.ch[7, 0, 3] == 1 and new.ch[10, 2, 6] == 1 and new.ch[6, 3, 1] == 0
    # outside the map: a plain full block (accelerator channels 0)
    assert new.pad[0, :3].all() and new.pad[0, :, :3].all() and not new.pad[1:, :3].any()
    assert not new.pad[1:, :, -3:].any()
    s = np.zeros(2, ROW)
    s["x"], s["y"] = 2.5 * T, 1.5 * T
    X = F.featurize(s, np.zeros(2, int), 16.667, new, np.empty((0, 5)), None, None, .016667, [])
    Xo = F.featurize(s, np.zeros(2, int), 16.667, old, np.empty((0, 5)), None, None, .016667, [])
    lay, lo = new.layout, old.layout
    assert X.shape == (2, 634)
    pn = X[:, lay.patch:lay.haz].reshape(2, 11, 7, 7)
    po = Xo[:, lo.patch:lo.haz].reshape(2, 6, 7, 7)
    assert np.array_equal(pn[:, :6], po)
    assert np.array_equal(X[:, :lay.patch], Xo[:, :lo.patch]) and np.array_equal(X[:, lay.haz:], Xo[:, lo.haz:])
    assert pn[0, 6, 2, 3] == 1 and pn[0, 7, 2, 4] == 1               # the floor accelerators below
    assert F.featurize_k(s, np.zeros((2, 4), int), 66.668, new, np.empty((0, 5)), None, None, .016667,
                         []).shape == (2, lay.n_in(4))


def test_live_tile_changes_and_opened_cells_clear_accel_channels():
    w = accel_words()
    g = F.LevelGeom(w, {"ox": 0, "oy": 0}, protocol=PLAT_TILES_PROTOCOL)
    x, y = np.array([2.5 * T]), np.array([1.5 * T])
    patch = lambda geom: geom.patches(x, y).reshape(11, 7, 7)
    assert patch(g)[6, 2, 3] == 1
    opened = g.opened([(0, 2)])
    assert not patch(opened)[:, 2, 3].any() and patch(opened)[6, 2, 4] == 1      # every channel of the cell 0
    assert patch(g)[6, 2, 3] == 1                                                 # the geometry itself unchanged
    g.set_tile(0 * w.shape[1] + 2, 1)                       # the accelerator becomes a plain block
    assert patch(g)[0, 2, 3] == 1 and not patch(g)[6:, 2, 3].any()
    g.set_tile(0 * w.shape[1] + 2, WALL_A)                  # and an (unmeasured) wall accelerator
    assert patch(g)[6:, 2, 3].tolist() == [1, 0, 1, 1, 0]
    g.reset()
    assert patch(g)[6:, 2, 3].tolist() == [1, 0, 0, 0, 0]


@pytest.mark.parametrize("step, rows, digest", [
    (1, 7128, "20437633e61041d94a941b6c4d2f5a1d49743a196442010ac2530878739eaf88"),
    (4, 7056, "46b1fdacc133755a3c0b3302ff4a04e37d52717cab52638cee3b90f557432eef"),
])
def test_tile_shapes_features_are_byte_identical_on_recorded_data(step, rows, digest):
    """tile-shapes-v1's X hashed before tile-accel-v1 existed (commit a6cc655); tile-accel-v1's and tile-plat-v1's
    first six channels equal it, and they equal each other (1-3 has no moving platforms: nothing painted)."""
    if not os.path.isdir(DATA_1_3):
        pytest.skip("recorded data not present")
    d = F.build([DATA_1_3], step=step, log=lambda *a: None, protocol=TILE_SHAPES_PROTOCOL)
    assert d["X"].shape[0] == rows
    assert hashlib.sha256(np.ascontiguousarray(d["X"]).tobytes()).hexdigest() == digest
    a = F.build([DATA_1_3], step=step, log=lambda *a: None, protocol=PLAT_TILES_PROTOCOL)
    lay, lo = F.layout(PLAT_TILES_PROTOCOL), F.layout(TILE_SHAPES_PROTOCOL)
    assert a["X"].shape == (rows, lay.n_in(step))
    assert np.array_equal(a["X"][:, lay.patch:lay.haz].reshape(rows, 11, 49)[:, :6],
                          d["X"][:, lo.patch:lo.haz].reshape(rows, 6, 49))
    assert np.array_equal(a["X"][:, :lay.patch], d["X"][:, :lo.patch])
    assert np.array_equal(a["X"][:, lay.haz:], d["X"][:, lo.haz:])
    assert not a["X"][:, lay.patch:lay.haz].reshape(rows, 11, 49)[:, 6:].any()     # 1-3 has no accelerators
    acc = F.build([DATA_1_3], step=step, log=lambda *a: None, protocol=ACCEL_TILES_PROTOCOL)
    assert acc["X"].tobytes() == a["X"].tobytes()


# ---- training CLI and the planner's loading path ----------------------------------------------------------------
def accel_recording(folder):
    from test_key_features import recording
    recording(folder)
    p = folder / "levels" / "test-key.npz"
    z = np.load(p, allow_pickle=False)
    words, meta = z["words"].copy(), str(z["meta"])
    words[3, 3], words[3, 5], words[4, 6] = FLOOR_R, FLOOR_L, WALL_B
    np.savez_compressed(p, words=words, meta=meta)
    return folder


def test_training_save_and_load_features_with_tile_plat(tmp_path, monkeypatch):
    from wm import train as T_
    from wm.model import WorldModel
    from wm.rollout import World, rollout_batched
    data = accel_recording(tmp_path / "data")
    monkeypatch.setattr(T_, "immediate", lambda: None)
    small = ["--members", "1", "--hidden", "8", "--layers", "1", "--epochs", "1"]
    out = tmp_path / "fresh"
    monkeypatch.setattr(sys, "argv", ["wm.train", str(data), "--legacy-split", "--out", str(out), "--sample-frac", "1",
                                      "--feature-protocol", PLAT_TILES_PROTOCOL] + small)
    T_.main()
    m = WorldModel.load(out / "wm_model.pt")
    assert m.info["feature_protocol"] == m.info["active_feature_protocol"] == PLAT_TILES_PROTOCOL
    assert F.model_layout(m).protocol == PLAT_TILES_PROTOCOL and m.mu.numel() == 634

    npz = tmp_path / "features.npz"
    monkeypatch.setattr(sys, "argv", ["wm.train", str(data), "--legacy-split", "--out", str(tmp_path / "unused"),
                                      "--sample-frac", "1", "--step", "4", "--max-samples", "0",
                                      "--feature-protocol", PLAT_TILES_PROTOCOL, "--save-features", str(npz)])
    T_.main()
    z = np.load(npz, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    assert meta["protocol"] == PLAT_TILES_PROTOCOL and z["d_X"].shape[1] == 664
    lay = F.layout(PLAT_TILES_PROTOCOL)
    assert z["d_X"][:, lay.patch:lay.haz].reshape(len(z["d_X"]), 11, 49)[:, 6].any()   # accelerators in the patches
    unknown = tmp_path / "unknown.npz"              # a features file of no known protocol (same width)
    np.savez(unknown, **{k: z[k] for k in z.files if k != "meta"}, meta=json.dumps({**meta, "protocol": "tile-x-v0"}))
    monkeypatch.setattr(sys, "argv", ["wm.train", "--load-features", str(unknown), "--out", str(tmp_path / "u")] + small)
    with pytest.raises(SystemExit, match="tile-x-v0"):
        T_.main()
    assert not (tmp_path / "u").exists()
    monkeypatch.setattr(sys, "argv", ["wm.train", "--load-features", str(npz), "--out", str(tmp_path / "r"),
                                      "--feature-protocol", ACCEL_TILES_PROTOCOL] + small)
    with pytest.raises(SystemExit, match="not --feature-protocol tile-accel-v1"):
        T_.main()                                   # the file's recorded protocol differs from the requested one
    assert not (tmp_path / "r").exists()
    gpu = tmp_path / "gpu"
    monkeypatch.setattr(sys, "argv", ["wm.train", "--load-features", str(npz), "--out", str(gpu)] + small)
    T_.main()
    g = WorldModel.load(gpu / "wm_model.pt").eval()
    assert g.step == 4 and F.model_layout(g) == lay and g.mu.numel() == 664
    words, lmeta = np.load(data / "levels" / "test-key.npz")["words"], {"ox": 0, "oy": 0}
    geom = F.LevelGeom(words, lmeta, protocol=F.model_layout(g).protocol)     # as plan.planner.Planner.set_level
    assert F.require_geom(g, geom) == lay
    with pytest.raises(ValueError, match="match"):
        F.require_geom(g, F.LevelGeom(words, lmeta, protocol=TILE_SHAPES_PROTOCOL))
    s = np.zeros(1, ROW)
    s["x"], s["y"] = 90.0, 70.0
    out = rollout_batched(g, geom, s, np.zeros((1, 4), int), np.full(4, 16.667), [World()] * 4)
    assert out["steps"] == 1 and np.isfinite(out["x"]).all()
