"""tile-plat-v1: moving platforms painted into the tile patch as plain full tiles (wm.features.paint_platforms, the
one owner, through tile_patch): which cells (coverage threshold), a real tile wins, the fast path, OpenedGeom then
painting, the other protocols untouched, and every featurize path (featurize, featurize_k, build on a recorded 4-4
shard, the planner's grouped features_into). Offline only."""
import gzip
import json
import math
import os
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from game.level import platform_boxes                                                  # noqa: E402
from wm import features as F                                                         # noqa: E402
from wm.buffer import ROW, transitions                                               # noqa: E402
from wm.protocol import KEY_TILES_PROTOCOL, PLAT_TILES_PROTOCOL, TILE_SHAPES_PROTOCOL  # noqa: E402

T = 20.0
C = 11
HALF_UP = 1 | (1 << 18)                    # a half block (realpath.tile_shape), any orientation: a real tile
DATA_44 = os.path.join(HERE, "runs", "online_check01", "wm_data")
SHARD_44 = "wave_00555"                    # 4-4; frame 360: the agent at (97, -94) beside the bar at (152.7, -110.4)


def empty_words(H=12, W=14):
    w = np.zeros((H, W), np.uint32)
    w[0, :] = 1                                                     # a floor far below the test cells
    return w


def geom(words=None, protocol=PLAT_TILES_PROTOCOL):
    return F.LevelGeom(empty_words() if words is None else words, {"ox": 0, "oy": 0}, protocol=protocol)


def at(x, y, n=1):
    s = np.zeros(n, ROW)
    s["x"], s["y"] = x, y
    s["flags"] = 2
    return s


def inputs(g, s, pf):
    return F.featurize(s, np.zeros(len(s), int), 16.667, g, np.empty((0, 5)), pf, pf, .016667, [])


def patch_of(g, X, row=0):
    lay = g.layout
    return X[row, lay.patch:lay.haz].reshape(lay.channels, 7, 7)


# the agent at (5.5 T, 5.5 T): its cell is tile (row 5, column 5); patch cell (i, j) is tile (5 - 3 + i, 5 - 3 + j)
AGENT = (5.5 * T, 5.5 * T)


def painted(g, s, pf):
    """The patch cells (row i, column j) painted: tile-plat-v1 with pf against the same without platforms."""
    a, b = patch_of(g, inputs(g, s, pf)), patch_of(g, inputs(g, s, None))
    diff = np.nonzero((a != b).any(0))
    return sorted(zip(diff[0].tolist(), diff[1].tolist())), a


def test_horizontal_bar_paints_the_cells_it_covers_by_half_or_more():
    g = geom()
    # a 60 x 20 bar from x 55..115, y 56..76: tile columns 2 (0.25), 3, 4 (1), 5 (0.75); rows 2 (0.2), 3 (0.8)
    cells, p = painted(g, at(*AGENT), [[85.0, 66.0, 0]])
    assert cells == [(1, 1), (1, 2), (1, 3)]                         # tile row 3, columns 3..5 (0.8, 0.8, 0.6)
    for i, j in cells:                                               # exactly a plain full block
        assert p[0, i, j] == 1 and not p[1:, i, j].any()
    # coverage exactly 0.5 is painted: y 50..70 halves rows 2 and 3 (columns 3..5 fully: x 60..120)
    assert painted(g, at(*AGENT), [[90.0, 60.0, 0]])[0] == [(i, j) for i in (0, 1) for j in (1, 2, 3)]
    # 0.45 is not: y 49..69 covers row 2 by 0.55, row 3 by 0.45
    assert painted(g, at(*AGENT), [[90.0, 59.0, 0]])[0] == [(0, 1), (0, 2), (0, 3)]


def test_upright_bar_paints_a_column():
    g = geom()
    # a 20 x 60 upright bar from x 137..157 (column 6: 0.15, column 7: 0.85), y 73..133 (row 3: 0.35, rows 4, 5,
    # row 6: 0.65)
    cells, p = painted(g, at(*AGENT), [[147.0, 103.0, 1]])
    assert cells == [(2, 5), (3, 5), (4, 5)]
    assert all(p[0, i, j] == 1 and not p[1:, i, j].any() for i, j in cells)
    assert platform_boxes([[147.0, 103.0, 1]])[0][2:4] == (10.0, 30.0)  # game.level's rule, reused


def test_two_bars_add_up_in_a_cell():
    g = geom()
    # two bars meeting at x 70 inside tile column 3 (60..80): 0.5 each, summed 1.0 (rows: y 60..80, row 3 only)
    pf = [[40.0, 70.0, 0], [100.0, 70.0, 0]]                          # x 10..70 and 70..130
    cells = painted(g, at(*AGENT), pf)[0]
    assert cells == [(1, j) for j in range(5)]                       # columns 2..6 (6: 120..130, half)
    assert painted(g, at(*AGENT), pf[:1])[0] == [(1, 0), (1, 1)]     # one of them: column 3 is half covered


def test_a_real_tile_wins():
    w = empty_words()
    w[3, 4] = HALF_UP                                                 # under the bar of the first test
    w[3, 3] = 0x800062                                                # an accelerator floor (accel channel)
    g = geom(w)
    s = at(*AGENT)
    a, b = patch_of(g, inputs(g, s, [[85.0, 66.0, 0]])), patch_of(g, inputs(g, s, None))
    assert np.array_equal(a[:, 1, 2], b[:, 1, 2]) and b[1, 1, 2] == 1 and a[0, 1, 2] == 0   # the half block kept
    assert np.array_equal(a[:, 1, 1], b[:, 1, 1]) and b[6, 1, 1] == 1                       # the accelerator kept
    assert a[0, 1, 3] == 1 and not a[1:, 1, 3].any()                                         # the empty one painted


def test_fast_path_is_byte_identical_and_the_tables_are_untouched():
    g = geom()
    s = np.concatenate([at(*AGENT), at(2.5 * T, 8.5 * T)])
    base = inputs(g, s, None)
    table = g.patches(s["x"].astype(float), s["y"].astype(float)).copy()
    pad = g.pad.copy()
    far = [[400.0, 200.0, 0], [-300.0, 100.0, 1]]                     # no bar reaches either patch
    assert inputs(g, s, far)[:, :g.layout.plat].tobytes() == base[:, :g.layout.plat].tobytes()
    assert inputs(g, s, []).tobytes() == base.tobytes()
    near = inputs(g, s, [[85.0, 66.0, 0]])[:, :g.layout.plat]
    assert not np.array_equal(near[0], base[0, :g.layout.plat])                         # only the reached row
    assert near[1].tobytes() == base[1, :g.layout.plat].tobytes()
    # the geometry's own cached patches and channels never change
    assert g.patches(s["x"].astype(float), s["y"].astype(float)).tobytes() == table.tobytes()
    assert g.pad.tobytes() == pad.tobytes() and g._table is not None


def test_cells_just_out_of_reach_are_not_painted():
    g = geom()
    # the patch spans x 40..180: a bar ending at x 40 touches nothing; at 50, half of column 2's cell (40..60)
    assert painted(g, at(*AGENT), [[10.0, 70.0, 0]])[0] == []
    assert painted(g, at(*AGENT), [[20.0, 70.0, 0]])[0] == [(1, 0)]


@pytest.mark.parametrize("protocol", [TILE_SHAPES_PROTOCOL, KEY_TILES_PROTOCOL])
def test_other_protocols_never_paint(protocol):
    g = geom(protocol=protocol)
    s = at(*AGENT)
    pf = [[85.0, 66.0, 0]]
    a, b = inputs(g, s, pf), inputs(g, s, None)
    lay = g.layout
    assert a[:, :lay.plat].tobytes() == b[:, :lay.plat].tobytes()   # the platform columns differ (they see the bar)


def test_opened_cells_then_painting():
    w = empty_words()
    w[3, 4] = 1                                                       # a crumble the planner expects gone: opened
    w[3, 3] = 1                                                       # a block that stays
    g = geom(w)
    s = at(*AGENT)
    pf = [[85.0, 66.0, 0]]
    opened = g.opened([(3, 4)])
    p = patch_of(g, inputs(opened, s, pf))
    assert p[0, 1, 2] == 1 and not p[1:, 1, 2].any()                  # opened (empty), then painted solid
    q = patch_of(g, inputs(opened, s, None))
    assert not q[:, 1, 2].any()                                       # opened, no bar: empty
    assert patch_of(g, inputs(g, s, None))[0, 1, 2] == 1              # the level itself unchanged
    assert p[0, 1, 1] == 1 and p[0, 1, 3] == 1


def test_featurize_k_paints():
    g = geom()
    s = at(*AGENT, n=2)
    pf = [[85.0, 66.0, 0]]
    Xk = F.featurize_k(s, np.zeros((2, 4), int), 66.668, g, np.empty((0, 5)), pf, pf, .016667, [])
    X1 = inputs(g, s, pf)
    fd = g.layout.feat_dim
    assert Xk.shape == (2, g.layout.n_in(4))
    assert np.array_equal(Xk[:, g.layout.patch:g.layout.haz], X1[:, g.layout.patch:g.layout.haz])
    assert patch_of(g, Xk)[0, 1, 2] == 1 and not np.array_equal(Xk[:, :fd], inputs(g, s, None)[:, :fd])


def test_grouped_features_paint_each_group_with_its_own_platforms():
    from plan.planner import features_into
    from wm.rollout import World
    g = geom()
    s = np.concatenate([at(*AGENT), at(AGENT[0] + 3, AGENT[1]), at(*AGENT), at(AGENT[0] + 3, AGENT[1])])
    s["keys"] = [0, 0, 0, 0]
    a = np.array([0, 2, 7, 5])
    w0 = World(np.zeros((0, 5)), [[85.0, 66.0, 0]], None)            # a bar under the agent
    w1 = World(np.zeros((0, 5)), [[400.0, 200.0, 0]], None)          # far away: nothing painted
    w2 = World(np.zeros((0, 5)), [[147.0, 103.0, 1]], None)          # an upright bar beside it
    groups = [(0, np.array([0, 1])), (5, np.array([2])), (9, np.array([3]))]
    X = np.empty((4, g.layout.feat_dim), np.float32)
    features_into(X, groups, s, a, 16.667, g, [w0, w1, w2])
    for (o, idx), w in zip(groups, [w0, w1, w2]):
        ref = F.featurize(s[idx], a[idx], 16.667, g, w.boxes, w.pf, w.pf_prev, w.dt_prev_s, w.fans)
        assert X[idx].tobytes() == ref.tobytes()
    plain = inputs(g, s, None)
    lay = g.layout
    assert X[2, lay.patch:lay.haz].tobytes() == plain[2, lay.patch:lay.haz].tobytes()     # w1: nothing painted
    assert patch_of(g, X, 0)[0, 1, 2] == 1 and patch_of(g, X, 3)[0, 3, 5] == 1
    assert patch_of(g, X, 3)[0, 1, 2] == 0                                                  # not w0's bar
    # a non-painting protocol keeps the old path (the patch of the first call for every group)
    gs = geom(protocol=TILE_SHAPES_PROTOCOL)
    Xs = np.empty((4, gs.layout.feat_dim), np.float32)
    features_into(Xs, groups, s, a, 16.667, gs, [w0, w1, w2])
    for (o, idx), w in zip(groups, [w0, w1, w2]):
        ref = F.featurize(s[idx], a[idx], 16.667, gs, w.boxes, w.pf, w.pf_prev, w.dt_prev_s, w.fans)
        assert Xs[idx].tobytes() == ref.tobytes()


# ---- build() on a recorded 4-4 shard -----------------------------------------------------------------------------
def expected_cells(x, y, pf, ox, oy):
    """An independent per-cell rule: the (i, j) patch cells of a point at (x, y) at least half covered by the bars."""
    tc, tr = math.floor((x - ox) / T), math.floor((y - oy) / T)
    out, edge = set(), []
    for i in range(7):
        for j in range(7):
            cx0, cy0 = ox + T * (tc - 3 + j), oy + T * (tr - 3 + i)
            area = 0.0
            for bx, by, hw, hh, _ in platform_boxes(pf):
                area += max(0.0, min(cx0 + T, bx + hw) - max(cx0, bx - hw)) * \
                    max(0.0, min(cy0 + T, by + hh) - max(cy0, by - hh))
            if abs(area / (T * T) - 0.5) < 1e-6:
                edge.append((i, j))
            if area / (T * T) >= 0.5:
                out.add((i, j))
    return out, edge


def test_build_paints_the_recorded_bars_on_4_4(tmp_path, monkeypatch):
    src = os.path.join(DATA_44, SHARD_44 + ".npz")
    if not os.path.exists(src) or not os.path.exists(os.path.join(DATA_44, "levels", "4-4.npz")):
        pytest.skip("recorded 4-4 data not present")
    d = tmp_path / "wm_data"
    (d / "levels").mkdir(parents=True)
    for name in (SHARD_44 + ".npz", SHARD_44 + "_world.json.gz"):
        shutil.copy(os.path.join(DATA_44, name), d / name)
    shutil.copy(os.path.join(DATA_44, "levels", "4-4.npz"), d / "levels" / "4-4.npz")
    out = F.build([str(d)], log=lambda *a: None, protocol=PLAT_TILES_PROTOCOL)
    monkeypatch.setattr(F, "paint_platforms", lambda *a, **k: None)
    base = F.build([str(d)], log=lambda *a: None, protocol=PLAT_TILES_PROTOCOL)
    lay = F.layout(PLAT_TILES_PROTOCOL)
    pn = out["X"][:, lay.patch:lay.haz].reshape(-1, C, 7, 7)
    pb = base["X"][:, lay.patch:lay.haz].reshape(-1, C, 7, 7)
    assert np.array_equal(out["X"][:, :lay.patch], base["X"][:, :lay.patch])
    assert np.array_equal(out["X"][:, lay.haz:], base["X"][:, lay.haz:])
    assert not (pn[:, 1:] != pb[:, 1:]).any() and ((pn[:, 0] != pb[:, 0]) <= (pn[:, 0] == 1)).all()
    # every row against the independent rule, with its own frame's bars (build's row order: frame by frame)
    z = np.load(d / (SHARD_44 + ".npz"))
    world = {w["f"]: w for w in json.load(gzip.open(d / (SHARD_44 + "_world.json.gz"), "rt", encoding="utf-8"))}
    lmeta = json.loads(str(np.load(d / "levels" / "4-4.npz")["meta"]))
    s = transitions(z["rows"])["s"]
    order = np.concatenate([np.nonzero(s["frame"] == f)[0] for f in np.unique(s["frame"])])
    assert len(order) == len(pn)
    n_painted, beside = 0, None
    for k, t in enumerate(order):
        x, y, f = float(s["x"][t]), float(s["y"][t]), int(s["frame"][t])
        want, edge = expected_cells(x, y, world.get(f, {}).get("pf"), float(lmeta["ox"]), float(lmeta["oy"]))
        empty = ~pb[k].any(0)
        got = {(int(i), int(j)) for i, j in zip(*np.nonzero(pn[k, 0] != pb[k, 0]))}
        want = {c for c in want if empty[c]}
        assert got - set(edge) == want - set(edge), (f, x, y)
        n_painted += bool(got)
        if f == 360 and int(s["agent"][t]) == 0:
            beside = (x, y, got)
    assert n_painted > 0
    x, y, got = beside                                                # (97, -94), the bar at (152.7, -110.4)
    assert abs(x - 97) < 1 and abs(y + 94) < 1 and got
    ox, oy = float(lmeta["ox"]), float(lmeta["oy"])
    tc, tr = math.floor((x - ox) / T), math.floor((y - oy) / T)
    for i, j in got:                                                  # each painted cell's centre is inside the bar
        cx, cy = ox + T * (tc - 3 + j + 0.5), oy + T * (tr - 3 + i + 0.5)
        assert 152.7 - 30 - 1e-6 <= cx <= 152.7 + 30 + 1e-6 and -110.4 - 10 - 1e-6 <= cy <= -110.4 + 10 + 1e-6
