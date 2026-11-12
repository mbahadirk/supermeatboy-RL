"""Experiment identity, val isolation and metrics; fake predictions/rollouts, no fitting or gameplay."""
import gzip
import json
import types

import numpy as np
import pytest
import torch

from wm import experiment as E
from wm import features as F
from wm.buffer import ROW
from wm.dataset import Dataset, file_hash
from wm.model import WorldModel
from wm.protocol import FEATURE_PROTOCOL, KEY_TILES_PROTOCOL, TILE_SHAPES_PROTOCOL


def recording(folder, level):
    folder.mkdir(parents=True)
    (folder / "levels").mkdir()
    words = np.zeros((8, 96), np.uint32)
    words[0] = 1
    meta = {"name": level, "ox": 0, "oy": 0, "keys": [], "spawn": [40, 30], "goals": [[1000, 30]]}
    np.savez_compressed(folder / "levels" / (level + ".npz"), words=words, meta=json.dumps(meta))
    agents = []
    for agent in range(32):
        r = np.zeros(65, ROW)
        r["agent"], r["frame"], r["action"] = agent, np.arange(65), 6
        r["x"], r["y"], r["vx"] = 40 + np.arange(65), 30 + agent * 5, 10 + agent
        r["action"][-1], r["status"][-1] = -1, 1
        agents.append(r)
    p = folder / "wave_00000.npz"
    np.savez_compressed(p, rows=np.concatenate(agents), meta=json.dumps({"level": level}))
    world = [{"f": f, "dt": 16.667, "hz": [],
              **({"fx": {"fans": [[0, 30, 1, 0, 20, 200, 5]]}} if f == 0 else {})}
             for f in range(65)]
    with gzip.open(folder / "wave_00000_world.json.gz", "wt", encoding="utf-8") as f:
        json.dump(world, f)
    return folder


@pytest.fixture
def experiment(tmp_path):
    source = recording(tmp_path / E.HISTORICAL_RUN / "wm_data", "2-10")
    old = recording(tmp_path / "base" / "wm_data", "1-1")
    ds = Dataset.create([str(source), str(old)], 2)
    ds.save(tmp_path / "manifest.json")
    data = F.build(ds.sources, dataset=ds, step=4, frac=1, max_samples=0, provenance=True, log=lambda *a: None)
    roster = E.prepare_roster(ds, data, historical_source=str(source))
    return ds, data, roster


def checkpoint(tmp_path, ds, strict=True, scale=(2, 3, 4, 5), protocol=FEATURE_PROTOCOL):
    m = WorldModel(n_in=F.layout(protocol).n_in(4), members=1, hidden=8, layers=1, step=4, res_scale=scale)
    info = {"dataset": ds.reference(), "feature_protocol": protocol} if strict else {"data": ["old-evidence"]}
    p = tmp_path / ("strict.pt" if strict else "reference.pt")
    m.save(p, info)
    return p


@pytest.fixture
def fake_predictions(monkeypatch):
    calls = []
    def predict(model, X):
        assert X.shape[1] == F.model_layout(model).n_in(4)
        calls.append(("predict", len(X)))
        n = len(X)
        p = torch.zeros((n, len(F.BIN_NAMES)))
        p[:, F.EVENTS["death"]] = .75
        return {"mean": torch.ones((n, 4)), "p": p}
    def rollout(model, geom, s0, actions, dts, worlds, deadline=None):
        assert geom.layout == F.model_layout(model)
        calls.append(("rollout", actions.copy(), [w.fans for w in worlds]))
        n, K = actions.shape
        assert n == 1 and K % 4 == 0 and len(dts) == len(worlds) == K
        t = np.arange(K // 4 + 1) * 4
        x = s0["x"][None, :, None] + t[None, None, :]
        y = np.broadcast_to(s0["y"][None, :, None], x.shape)
        return {"x": x, "y": y, "step": 4, "steps": K // 4}
    monkeypatch.setattr(WorldModel, "predict", predict)
    monkeypatch.setattr(E, "rollout_batched", rollout)
    return calls


def test_opt_in_origin_matches_raw_ids_and_does_not_change_default_features(experiment):
    ds, data, _ = experiment
    assert data["origin"].dtype == np.int64 and data["origin"].shape == (len(data["X"]), 4)
    assert set(data["origin"][:, 0]) == {0, 1}
    assert np.max(data["origin"][:, 2]) == 63 and np.all(data["origin"][:, 3] == 0)
    plain = F.build(ds.sources, dataset=ds, step=4, frac=1, log=lambda *a: None)
    assert "origin" not in plain
    for key in plain:
        assert plain[key].dtype == data[key].dtype
        assert plain[key].tobytes() == data[key].tobytes()
    sample = F.build(ds.sources, dataset=ds, step=4, frac=.2, seed=8, max_samples=400,
                     provenance=True, log=lambda *a: None)
    lookup = {tuple(o): i for i, o in enumerate(data["origin"])}
    for key in ("X", "raw", "full", "validation"):
        ids = [lookup[tuple(o)] for o in sample["origin"]]
        assert sample[key].tobytes() == data[key][ids].tobytes()
    with pytest.raises(ValueError, match="manifest"):
        F.build([], provenance=True)


def test_roster_is_deterministic_val_only_with_fixed_wind_quota(experiment, tmp_path):
    ds, d, r = experiment
    assert E.prepare_roster(ds, d, r["historical_source"]) == r
    assert set(r["one_step_indices"]) == set(np.flatnonzero(d["validation"]))
    assert len(r["starts"]) <= 128
    for c in r["starts"]:
        assert d["validation"][c["sample"]]
        assert ds.entries[c["entry"]]["agents"][str(c["agent"])]["split"] == "validation"
    assert r["selected_strata"]["2-10"]["wind_nonzero"] == 4
    assert r["selected_strata"]["2-10"]["wind_zero"] == 4
    p = tmp_path / "roster.json"
    E.write_json_exclusive(p, r)
    assert E.read_roster(p) == r
    with pytest.raises(FileExistsError):
        E.write_json_exclusive(p, r)
    with pytest.raises(ValueError, match="budgets"):
        E.prepare_roster(ds, d, max_starts=129)


def test_role_comparison_same_cases_own_scale_and_censor_counts(experiment, tmp_path, fake_predictions):
    ds, d, r = experiment
    strict, old = checkpoint(tmp_path, ds), checkpoint(tmp_path, ds, strict=False)
    old_bytes = old.read_bytes()
    new = E.evaluate(ds, d, r, strict, out=tmp_path / "strict.json", batch_size=37,
                     expected_checkpoint_sha256=file_hash(strict))
    ref = E.evaluate(ds, d, r, old, role="historical-reference", out=tmp_path / "reference.json", batch_size=37)
    assert new["matched_historical"]["one_step"] == ref["one_step"]
    assert new["matched_historical"]["open_loop"] == ref["open_loop"]
    assert new["role"] == "strict-model" and ref["role"] == "historical-reference"
    ids = np.array(r["historical_indices"])
    ids = ids[d["full"][ids]]
    residual = d["prior"][ids] + np.array([2, 3, 4, 5]) - d["raw"][ids]
    expected = np.median(np.hypot(residual[:, 0], residual[:, 1]))
    assert ref["one_step"]["primary_full_four_frames"]["overall"]["position"]["median"] == pytest.approx(expected)
    assert ref["one_step"]["events_terminal_short"]["death"]["actual_positive"] > 0
    assert ref["one_step"]["events_full"]["survivor_death_false_alarm"]["rate"] == 1
    assert ref["open_loop"]["48"]["position"]["n"] < ref["open_loop"]["48"]["selected_starts"]
    for h, metric in ref["open_loop"].items():
        assert metric["position"]["median"] in (0, None)
        assert metric["survivors_scored"] + sum(metric["censored"].values()) == metric["selected_starts"]
    assert max(n for kind, n, *rest in fake_predictions if kind == "predict") <= 37
    assert old.read_bytes() == old_bytes
    with pytest.raises(FileExistsError):
        E.evaluate(ds, d, r, strict, out=tmp_path / "strict.json")


@pytest.mark.parametrize("bad", ["cache", "roster", "manifest", "checkpoint", "membership", "role"])
def test_invalid_identity_is_rejected_before_prediction(experiment, tmp_path, monkeypatch, bad):
    ds, d, r = experiment
    p = checkpoint(tmp_path, ds)
    monkeypatch.setattr(WorldModel, "predict", lambda *a: pytest.fail("invalid evidence must precede prediction"))
    kw = {}
    if bad == "cache":
        d["X"][0, 0] += 1
    elif bad == "roster":
        r["one_step_indices"][0] = int(np.flatnonzero(~d["validation"])[0])
        r = E._sealed({k: v for k, v in r.items() if k != "sha256"})
    elif bad == "manifest":
        r["manifest_sha256"] = "wrong"
        r = E._sealed({k: v for k, v in r.items() if k != "sha256"})
    elif bad == "checkpoint":
        kw["expected_checkpoint_sha256"] = "wrong"
    elif bad == "membership":
        d["validation"][0] = ~d["validation"][0]
        r = E._sealed(dict({k: v for k, v in r.items() if k != "sha256"}, data_sha256=E.data_hash(d)))
    else:
        kw["role"] = "skip-validator"
    with pytest.raises(ValueError):
        E.evaluate(ds, d, r, p, **kw)


def test_strict_boundary_is_not_weakened_for_historical_role(experiment, tmp_path, monkeypatch):
    ds, d, r = experiment
    strict, old = checkpoint(tmp_path, ds), checkpoint(tmp_path, ds, strict=False)
    monkeypatch.setattr(WorldModel, "predict", lambda *a: pytest.fail("role rejection must precede prediction"))
    with pytest.raises(ValueError, match="legacy"):
        E.evaluate(ds, d, r, old)
    with pytest.raises(ValueError, match="disguised"):
        E.evaluate(ds, d, r, strict, role="historical-reference")
    with pytest.raises(ValueError, match="explicit"):
        E.evaluate(ds, d, E.prepare_roster(ds, d), old, role="historical-reference")
    with pytest.raises(ValueError, match="source"):
        E.prepare_roster(ds, d, ds.sources[0] + "wrong")


def test_duplicate_group_needs_explicit_selection_and_failure_writes_no_report(experiment, tmp_path, monkeypatch):
    ds, d, r = experiment
    p = checkpoint(tmp_path, ds)
    monkeypatch.setattr(WorldModel, "predict", lambda *a: {"mean": torch.full((len(a[1]), 4), float("nan")),
                                                         "p": torch.zeros((len(a[1]), len(F.BIN_NAMES)))})
    out = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="non-finite"):
        E.evaluate(ds, d, r, p, out=out)
    assert not out.exists()
    owners = list(ds.entries[0]["agents"].values())
    owners[1]["group"] = owners[0]["group"]
    with pytest.raises(ValueError, match="replay"):
        E.prepare_roster(ds, d)


def test_expired_budget_never_publishes_partial_metrics(experiment, tmp_path, monkeypatch):
    ds, d, r = experiment
    p = checkpoint(tmp_path, ds)
    tick = iter([0, 1])
    monkeypatch.setattr(E.time, "perf_counter", lambda: next(tick, 1))
    monkeypatch.setattr(WorldModel, "predict", lambda *a: pytest.fail("expired stage must not predict"))
    out = tmp_path / "timeout.json"
    with pytest.raises(TimeoutError):
        E.evaluate(ds, d, r, p, out=out, max_seconds=.5)
    assert not out.exists()


def test_missing_context_and_terminal_endpoint_are_censored_without_padding():
    s = np.zeros(10, ROW)
    s["frame"], s["action"] = np.arange(10), 6
    world = [{"f": i, "dt": 16.667} for i in range(10)]
    g = F.LevelGeom(np.zeros((2, 2), np.uint32), {"ox": 0, "oy": 0})
    case = {"frame": 0, "agent": 0, "keys": 0}
    s["status"][8] = 1
    *_, valid, reason, _ = E._case_inputs(case, s, world, g)
    assert valid == [] and reason == "terminal_censored"
    s["status"][:] = 0
    world[3]["dt"] = None
    *_, valid, reason, _ = E._case_inputs(case, s, world, g)
    assert valid == [] and reason == "missing_world_or_dt"


@pytest.fixture
def shape_experiment(tmp_path):
    # Actual synthetic captured key changes, before manifest creation; user recordings are never edited.
    folders = [recording(tmp_path / name / "wm_data", level) for name, level in
               (("shape_a", "2-10"), ("shape_b", "1-1"))]
    for folder in folders:
        p = folder / "wave_00000.npz"
        with np.load(p) as z:
            captured, meta = z["rows"].copy(), str(z["meta"])
        captured["keys"][captured["frame"] >= 24] = 1
        np.savez_compressed(p, rows=captured, meta=meta)
        lp = next((folder / "levels").glob("*.npz"))
        with np.load(lp) as z:
            words, lm = z["words"].copy(), json.loads(str(z["meta"]))
        lm["keys"] = [[80, 30]]
        np.savez_compressed(lp, words=words, meta=json.dumps(lm))
    ds = Dataset.create([str(p) for p in folders], 2)
    ds.save(tmp_path / "shape_manifest.json")
    data = F.build(ds.sources, dataset=ds, protocol=TILE_SHAPES_PROTOCOL, step=4, frac=1,
                   provenance=True, log=lambda *a: None)
    roster = E.prepare_roster(ds, data, feature_protocol=TILE_SHAPES_PROTOCOL, max_shards=160, max_rows=1500000)
    return ds, data, roster


def test_v1_default_contract_is_identical_and_extended_limits_are_explicit(experiment):
    ds, d, old = experiment
    assert old == E.prepare_roster(ds, d, old["historical_source"], feature_protocol=KEY_TILES_PROTOCOL,
                                  max_shards=100, max_rows=410000)
    assert old["protocol"] == E.PROTOCOL and old["feature_protocol"] == KEY_TILES_PROTOCOL
    assert set(old["budgets"]) == {"max_starts", "per_level"}
    assert "encoding" not in old and "open_loop_key_policy" not in old
    extended = E.prepare_roster(ds, d, old["historical_source"], max_shards=160, max_rows=1500000)
    assert extended["protocol"] == E.PROTOCOL_V2 and extended["sha256"] != old["sha256"]
    # Apart from the new versioned encoding/resource/key-transition contract, old selection is untouched.
    body = {k: v for k, v in extended.items() if k not in ("sha256", "encoding", "open_loop_key_policy")}
    body["protocol"] = E.PROTOCOL
    body["budgets"] = {k: v for k, v in body["budgets"].items() if k not in ("max_shards", "max_rows")}
    assert E._sealed(body) == old


def test_shapes_opt_in_freezes_layout_limits_and_key_policy(shape_experiment, tmp_path, fake_predictions):
    ds, d, r = shape_experiment
    assert d["X"].shape[1] == F.layout(TILE_SHAPES_PROTOCOL).n_in(4) == 419
    assert r["protocol"] == E.PROTOCOL_V2 and r["feature_protocol"] == TILE_SHAPES_PROTOCOL
    assert r["encoding"] == {"protocol": TILE_SHAPES_PROTOCOL, "step": 4, "n_in": 419}
    assert r["budgets"]["max_shards"] == 160 and r["budgets"]["max_rows"] == 1500000
    assert r["open_loop_key_policy"] == "censor_recorded_key_change"
    assert r["historical_source"] is None and r["historical_indices"] == []
    path = tmp_path / "shape_roster.json"
    E.write_json_exclusive(path, r)
    assert E.read_roster(path) == r
    p = checkpoint(tmp_path, ds, protocol=TILE_SHAPES_PROTOCOL)
    result = E.evaluate(ds, d, r, p, batch_size=40)
    assert result["protocol"] == E.PROTOCOL_V2 and result["encoding"] == r["encoding"]
    assert result["budgets"] == r["budgets"] and result["matched_historical"] is None
    one = result["one_step"]
    counts = one["sample_counts"]
    assert counts["keys_zero"] + counts["keys_owned"] == counts["all"]
    assert counts["keys_zero_full"] + counts["keys_owned_full"] == counts["full"]
    for label in ("keys_zero", "keys_owned"):
        assert one["primary_full_four_frames"][label]["position"]["n"] == counts[label + "_full"] > 0
    assert sum(m["censored"].get("key_mask_changed", 0) for m in result["open_loop"].values()) > 0


@pytest.mark.parametrize("direction", ["old_cache_new_request", "new_cache_default", "old_model_new_roster", "new_model_old_roster"])
def test_cross_encoding_is_rejected_before_prediction(experiment, shape_experiment, tmp_path, monkeypatch, direction):
    ds, old, roster = experiment
    sds, new, sr = shape_experiment
    monkeypatch.setattr(WorldModel, "predict", lambda *a: pytest.fail("mixed protocols cannot predict"))
    with pytest.raises(ValueError):
        if direction == "old_cache_new_request":
            E.prepare_roster(ds, old, feature_protocol=TILE_SHAPES_PROTOCOL)
        elif direction == "new_cache_default":
            E.prepare_roster(sds, new)
        elif direction == "old_model_new_roster":
            E.evaluate(sds, new, sr, checkpoint(tmp_path, sds))
        else:
            E.evaluate(ds, old, roster, checkpoint(tmp_path, ds, protocol=TILE_SHAPES_PROTOCOL))


def test_shapes_reference_and_resealed_encoding_tampering_are_rejected(shape_experiment, tmp_path, monkeypatch):
    ds, d, r = shape_experiment
    monkeypatch.setattr(WorldModel, "predict", lambda *a: pytest.fail("invalid role/encoding cannot predict"))
    with pytest.raises(ValueError, match="cross-encoding"):
        E.prepare_roster(ds, d, historical_source=ds.sources[0], feature_protocol=TILE_SHAPES_PROTOCOL)
    with pytest.raises(ValueError, match="cross-encoding"):
        E.evaluate(ds, d, r, "not-loaded.pt", role="historical-reference")
    bad = dict(r, encoding={"protocol": TILE_SHAPES_PROTOCOL, "step": 4, "n_in": 321})
    bad = E._sealed({k: v for k, v in bad.items() if k != "sha256"})
    with pytest.raises(ValueError, match="selection differs"):
        E.evaluate(ds, d, bad, "not-loaded.pt")


@pytest.mark.parametrize("limits", [{"max_shards": 161}, {"max_rows": 2000001}, {"max_rows": 0},
                                  {"max_rows": 1500000.0}, {"max_shards": True},
                                  {"max_rows": 10}, {"max_shards": 1}])
def test_shape_resource_limits_are_explicit_frozen_and_bounded(shape_experiment, limits):
    ds, d, _ = shape_experiment
    with pytest.raises(ValueError):
        E.prepare_roster(ds, d, feature_protocol=TILE_SHAPES_PROTOCOL, **limits)


def test_v2_key_change_censors_prefix_but_v1_semantics_remain_unchanged():
    captured = np.zeros(65, ROW)
    captured["frame"], captured["action"] = np.arange(65), 6
    captured["keys"][12:] = 1
    world = [{"f": i, "dt": 16.667} for i in range(65)]
    geom = F.LevelGeom(np.zeros((2, 2), np.uint32), {"ox": 0, "oy": 0, "keys": [[1, 1]]})
    case = {"frame": 0, "agent": 0, "keys": 0}
    *_, old_valid, old_reason, _ = E._case_inputs(case, captured, world, geom)
    *_, new_valid, new_reason, _ = E._case_inputs(case, captured, world, geom, censor_key_change=True)
    assert old_valid == list(E.HORIZONS) and old_reason == "horizon_limit"
    assert new_valid == [8] and new_reason == "key_mask_changed"
    assert np.all(captured["keys"][12:] == 1)  # no repair, pickup prediction or mutation
