"""Known private key masks change tile inputs, without fitting, prediction, or gameplay."""
import gzip
import json
import sys
import types

import numpy as np
import pytest

from game.level import Level, LockContext, LOCK_CONTEXT_PROTOCOL, key_lock_groups
from wm.buffer import Collector, ROW, load_level
from wm import features as F
from wm.protocol import FEATURE_PROTOCOL, LEGACY_FEATURE_PROTOCOL, feature_usage


@pytest.mark.parametrize("marker", [None, ""])
def test_missing_trained_protocol_marker_displays_legacy_and_strict_rejects(marker):
    from wm.protocol import require_feature_protocol
    info = {"feature_protocol": marker}
    assert feature_usage(info)["checkpoint_trained"] == LEGACY_FEATURE_PROTOCOL
    with pytest.raises(ValueError, match="feature protocol"):
        require_feature_protocol(info)


def geometry(keys=None, applied=False):
    words = np.zeros((9, 9), np.uint32)
    words[4, 4] = 0xc00163
    meta = {"ox": 0, "oy": 0, "keys": [[110, 90]] if keys is None else keys}
    if applied:
        meta["lock_context"] = {"protocol": LOCK_CONTEXT_PROTOCOL, "source": "python-go-payload",
                                "keys": meta["keys"], "groups": [[40]]}
    return words, meta


def rows(masks):
    s = np.zeros(len(masks), ROW)
    s["x"], s["y"], s["keys"] = 90, 90, masks
    return s


def inputs(geom, s):
    return F.featurize(s, np.zeros(len(s), int), 16.667, geom, np.empty((0, 5)), None, None, .016667, [])


def test_mixed_masks_are_private_and_zero_mask_is_bitexact():
    words, meta = geometry()
    g = F.LevelGeom(words, meta)
    p = g.patches(np.array([90., 90., 90.]), np.array([90., 90., 90.]))
    original = [a.copy() for a in (g.words, g.pad, g._table)]
    q = g.patches(np.array([90., 90., 90.]), np.array([90., 90., 90.]), [0, 1, 0])
    assert p[0].tobytes() == q[0].tobytes() == q[2].tobytes()
    assert q[1].reshape(4, 7, 7)[:, 3, 3].tolist() == [0, 0, 0, 0]
    assert p[1].reshape(4, 7, 7)[:, 3, 3].tolist() == [0, 0, 0, 1]
    for before, after in zip(original, (g.words, g.pad, g._table)):
        assert before.tobytes() == after.tobytes()
    # Neither processing order nor changing the returned buffer changes another agent's map.
    q[:] = 123
    assert g.patches(np.array([90.]), np.array([90.]), [0]).tobytes() == p[:1].tobytes()


def test_key_patches_off_centre_and_after_tile_updates_reset():
    words, meta = geometry()
    g = F.LevelGeom(words, meta)
    # lock is one cell to the right and below this agent; neighbouring solid remains.
    g.set_tile(49, 1)
    s = rows([0, 1])
    s["x"], s["y"] = 70, 110
    p = g.patches(s["x"], s["y"], s["keys"]).reshape(2, 4, 7, 7)
    assert p[0, 3, 2, 4] == 1 and p[1, 3, 2, 4] == 0
    assert p[:, 0, 3, 4].tolist() == [1, 1]
    g.set_tile(40, 0)
    assert not g.patches(s["x"], s["y"], [0, 0]).reshape(2, 4, 7, 7)[:, :, 2, 4].any()
    g.reset()
    assert g.patches(s["x"], s["y"], [0, 1]).reshape(2, 4, 7, 7)[:, 3, 2, 4].tolist() == [1, 0]


def test_common_features_apply_mask_without_changing_dimension_or_other_columns():
    g = F.LevelGeom(*geometry())
    s = rows([0, 1])
    X = inputs(g, s)
    assert X.shape == (2, 291)
    assert np.array_equal(X[0, :F._C_PATCH], X[1, :F._C_PATCH])
    assert np.array_equal(X[0, F._C_HAZ:], X[1, F._C_HAZ:])
    assert not np.array_equal(X[0, F._C_PATCH:F._C_HAZ], X[1, F._C_PATCH:F._C_HAZ])
    Xk = F.featurize_k(s, np.zeros((2, 4), int), 66.668, g, np.empty((0, 5)), None, None, .016667, [])
    assert Xk.shape == (2, 321)
    out = np.empty_like(X)
    assert F.featurize(s, [0, 0], 16.667, g, np.empty((0, 5)), None, None, .016667, [], out) is out
    assert out.tobytes() == X.tobytes()


def test_planner_grouped_feature_buffer_keeps_each_trajectory_key_mask():
    from plan.planner import features_into
    from wm.rollout import World, advance
    g = F.LevelGeom(*geometry())
    s = rows([0, 1, 1, 0])
    X = np.empty((4, F.FEAT_DIM), np.float32)
    features_into(X, [(0, np.array([0, 1])), (4, np.array([2, 3]))], s, np.zeros(4, int), 16.667,
                  g, [World(), World()])
    assert X.tobytes() == inputs(g, s).tobytes()
    # This component does not invent pickups when motion is advanced deterministically.
    nxt = advance(s, np.zeros(4, int), np.zeros((4, 4)), np.zeros((4, len(F.BIN_NAMES))), np.zeros(4, int))
    assert nxt["keys"].tobytes() == s["keys"].tobytes()


def test_applied_mapping_is_authoritative_and_legacy_multi_key_owned_is_unknown():
    words, meta = geometry(keys=[[10, 10], [90, 90]])
    assert key_lock_groups(words, 0, 0, meta["keys"]) == [[], [40]]
    meta["lock_context"] = {"protocol": LOCK_CONTEXT_PROTOCOL, "source": "python-go-payload",
                            "keys": meta["keys"], "groups": [[40], []]}
    g = F.LevelGeom(words, meta)
    p = g.patches(np.array([90., 90.]), np.array([90., 90.]), [1, 2]).reshape(2, 4, 7, 7)
    assert p[:, 3, 3, 3].tolist() == [0, 1]
    del meta["lock_context"]
    old = F.LevelGeom(words, meta)
    assert old.lock_context.source == "unknown-legacy-multi-key"
    old.patches(np.array([90.]), np.array([90.]), [0])
    with pytest.raises(ValueError, match="multi-key"):
        old.patches(np.array([90.]), np.array([90.]), [1])


@pytest.mark.parametrize("change", ["protocol", "keys", "duplicate", "range"])
def test_invalid_applied_context_is_rejected(change):
    words, meta = geometry(applied=True)
    c = meta["lock_context"]
    if change == "protocol":
        c["protocol"] = "unknown"
    elif change == "keys":
        c["keys"] = []
    elif change == "duplicate":
        c["groups"] = [[40, 40]]
    else:
        c["groups"] = [[81]]
    with pytest.raises(ValueError):
        LockContext(words, meta)


def test_single_key_domain_owner_and_collector_record_same_applied_mapping(tmp_path):
    words, meta = geometry()
    lv = Level.__new__(Level)
    lv.name, lv.W, lv.H, lv.ox, lv.oy = "test-key", 9, 9, 0, 0
    lv.words0, lv.keys = words, meta["keys"]
    lv.lock_groups = lv._lock_groups()
    lv.spawn, lv.goals, lv.bandages, lv.portals = (10, 10), [(150, 150)], [], []
    assert lv.lock_groups == [[40]] and lv.opened_cells(1) == [(4, 4)]
    applied = lv.wm_geometry_metadata()
    assert LockContext(words, meta).opened_cells(1) == LockContext(words, applied).opened_cells(1)
    c = Collector(str(tmp_path / "data"))
    c.level(lv, {})
    _, saved = load_level(c.dir, lv.name)
    assert saved["lock_context"] == applied["lock_context"]
    lv.lock_groups[0].clear()
    c.level(lv, {})                     # never overwrite old static metadata
    assert load_level(c.dir, lv.name)[1]["lock_context"]["groups"] == [[40]]


def recording(folder):
    folder.mkdir()
    (folder / "levels").mkdir()
    words, meta = geometry(applied=True)
    meta.update(name="test-key", W=9, H=9, spawn=[10, 10], goals=[[150, 150]])
    np.savez_compressed(folder / "levels" / "test-key.npz", words=words, meta=json.dumps(meta))
    for i, mask in enumerate([0, 1]):
        s = rows([mask] * 12)
        s["frame"], s["wave"], s["action"] = np.arange(12), i, 0
        s["action"][-1] = -1
        np.savez_compressed(folder / f"wave_{i:05d}.npz", rows=s, meta=json.dumps({"level": "test-key"}))
        with gzip.open(folder / f"wave_{i:05d}_world.json.gz", "wt", encoding="utf-8") as f:
            json.dump([{"f": j, "dt": 16.667, "hz": []} for j in range(12)], f)
    return folder


def test_offline_build_uses_row_keys_and_static_applied_mapping(tmp_path):
    d = F.build([str(recording(tmp_path / "data"))], frac=1, log=lambda *a: None)
    assert len(d["X"]) == 22
    a, b = d["X"][d["shard"] == 1], d["X"][d["shard"] == 2]
    assert a.shape == b.shape == (11, F.FEAT_DIM)
    assert np.array_equal(a[:, :F._C_PATCH], b[:, :F._C_PATCH])
    assert a[:, F._C_PATCH:F._C_HAZ].sum() == 11 and b[:, F._C_PATCH:F._C_HAZ].sum() == 0


def test_matching_manifest_still_rejects_old_feature_weights(tmp_path):
    from wm.dataset import Dataset
    ds = Dataset.create([str(recording(tmp_path / "data"))], 2)
    m = types.SimpleNamespace(info={"dataset": ds.reference()})
    assert feature_usage(m.info) == {"active": FEATURE_PROTOCOL, "checkpoint_trained": LEGACY_FEATURE_PROTOCOL}
    with pytest.raises(ValueError, match="feature protocol"):
        ds.require_model(m)
    m.info["feature_protocol"] = FEATURE_PROTOCOL
    ds.require_model(m)


def test_strict_train_init_rejects_feature_mismatch_before_build(tmp_path, monkeypatch):
    from wm import train as T
    from wm.dataset import Dataset
    from wm.model import WorldModel
    ds = Dataset.create([str(recording(tmp_path / "data"))], 2)
    manifest = tmp_path / "manifest.json"
    ds.save(manifest)
    parent = tmp_path / "parent.pt"
    WorldModel(members=1, hidden=8, layers=1).save(parent, {"dataset": ds.reference()})
    monkeypatch.setattr(T, "build", lambda *a, **kw: pytest.fail("protocol mismatch must precede feature scan"))
    monkeypatch.setattr(sys, "argv", ["wm.train", "--manifest", str(manifest), "--init", str(parent)])
    with pytest.raises(ValueError, match="feature protocol"):
        T.main()


def test_legacy_init_without_fit_preserves_parent_trained_protocol(tmp_path, monkeypatch):
    from wm import train as T
    from wm.model import WorldModel
    data = recording(tmp_path / "data")
    parent = tmp_path / "parent.pt"
    WorldModel(members=1, hidden=8, layers=1).save(parent, {})
    before = parent.read_bytes()
    monkeypatch.setattr(T, "fit", lambda *a, **kw: pytest.fail("--init must not fit without self-correction"))
    monkeypatch.setattr(T, "report", lambda *a: "no prediction in contract test")
    monkeypatch.setattr(T, "immediate", lambda: None)
    out = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["wm.train", str(data), "--legacy-split", "--init", str(parent),
                                      "--out", str(out), "--sample-frac", "1"])
    T.main()
    info = WorldModel.load(out / "wm_model.pt").info
    assert info["active_feature_protocol"] == FEATURE_PROTOCOL
    assert info["feature_protocol"] == info["parent_feature_protocol"] == LEGACY_FEATURE_PROTOCOL
    assert info["training"]["epochs"] == info["training"]["fit_samples"] == 0
    assert parent.read_bytes() == before
