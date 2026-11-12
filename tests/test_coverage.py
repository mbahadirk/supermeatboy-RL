"""Coverage and metadata inventory from tiny recordings, without features, training or rollout."""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from wm import coverage as C
from wm import dataset as D
from wm.buffer import ROW
from test_dataset import source, shard


def recording(folder, rows, meta, world, number=0):
    p = folder / f"wave_{number:05d}.npz"
    np.savez_compressed(p, rows=rows, meta=json.dumps(dict(level="test-level", **meta)))
    with gzip.open(str(p)[:-4] + "_world.json.gz", "wt", encoding="utf-8") as f:
        json.dump(world, f)
    return p


@pytest.fixture
def repeated(tmp_path):
    a, b = source(tmp_path / "a"), source(tmp_path / "b")
    shard(a, 0, [(0, 0)])
    shard(b, 0, [(99, 0)], wave=99, reward=float("nan"))
    ds = D.Dataset.create([str(a), str(b)], 2)
    ds.save(tmp_path / "manifest.json")
    return ds


def test_raw_and_global_trace_dedup_denominators(repeated):
    r = C.report(repeated, max_shards=2, max_rows=24, step=4)
    assert r["measurement_complete_manifest"] and r["manifest_complete_listed_sources"]
    assert r["budget"]["decoded_rows"] == 24 and r["training_ready"] == "not established"
    total = r["totals"]
    assert total["raw_core_counts"]["agent_episodes"] == 2
    assert total["raw_core_counts"]["transitions"] == 22
    assert total["trace_dedup_core_counts"]["transitions"] == 11
    assert total["raw_core_counts"]["chains"] == 16 and total["trace_dedup_core_counts"]["chains"] == 8
    assert total["unique_exact_trace_groups"] == 1 and total["repeat_factor"] == 2
    assert sum(v["unique_exact_trace_groups"] for v in r["source_level_split"]) == 2
    assert total["proxy_counts"]["wind_context_unknown"] == 22
    assert total["proxy_counts"]["platform_context_known"] == 22
    assert total["wind_nonzero_rate"] is None and total["platform_near_200_rate"] == 0
    assert r["world_timeline_counts"] == {"world_frames": 24, "dt_known": 24, "dt_outside_locked_16_667ms": 0}


def test_flags_labels_chains_and_known_context_proxies(tmp_path):
    d = source(tmp_path / "data")
    rows = np.zeros(6, ROW)
    rows["frame"], rows["x"], rows["y"], rows["action"] = np.arange(6), 40, 30, 6
    rows["flags"] = [0, 2, 2 | 512, 0, 0, 0]
    rows["vx"] = [100, 100, 100, 10, 10, 0]
    rows["status"][-1], rows["action"][-1] = 1, -1
    world = [{"f": 0, "dt": 16.667, "pf": [[40, 30, 1]]}, {"f": 1, "dt": 33},
             {"f": 2, "dt": None, "fx": {"fans": [[0, 30, 1, 0, 20, 200, 5]]}},
             {"f": 3, "dt": 16.667}, {"f": 5, "dt": "invalid"}]
    recording(d, rows, {"status": [1], "why": "done"}, world)
    ds = D.Dataset.create([str(d)], 1)
    r = C.report(ds, 1, 6, step=4)
    raw, proxy, outcome = [r["totals"][k] for k in ("raw_core_counts", "proxy_counts", "metadata_outcome_counts")]
    assert [raw[k] for k in ("flag_ground", "flag_air", "flag_wall")] == [3, 1, 1]
    assert raw["landing_label"] == raw["blocked_label"] == raw["death_label"] == 1
    assert raw["finish_label"] == 0 and raw["last_row_death"] == 1
    assert raw["chains"] == 5 and raw["terminal_short_death"] == 3 and raw["full_chains"] == 2
    assert proxy["platform_context_known"] == 4 and proxy["platform_context_unknown"] == 1
    assert proxy["platform_logged_present"] == proxy["platform_near_200"] == 1
    assert proxy["wind_context_known"] == proxy["wind_nonzero"] == 2 and proxy["wind_context_unknown"] == 3
    assert outcome["metadata_death"] == 1 and outcome["terminal_outcome_without_terminal_row"] == 0
    assert r["world_timeline_counts"] == {"world_frames": 5, "dt_known": 3, "dt_outside_locked_16_667ms": 1,
                                           "dt_missing": 1, "dt_invalid": 1}
    assert r["totals"]["platform_near_200_rate"] == .25
    assert r["totals"]["wind_nonzero_rate"] == 1


def test_censored_terminal_metadata_is_not_invented_as_a_transition(tmp_path):
    a, b, partial = [source(tmp_path / n) for n in ("a", "b", "partial")]
    rows = np.zeros(4, ROW)
    rows["frame"], rows["x"], rows["y"], rows["action"] = np.arange(4), 40, 30, 6
    rows["action"][-1] = -1
    world = [{"f": i, "dt": 16.667} for i in range(4)]
    recording(a, rows, {"status": [2], "finish": [3], "why": "quota"}, world)
    recording(b, rows, {"status": [1], "finish": [-1], "why": "done"}, world)
    recording(partial, rows, {"partial": True, "why": "cut"}, world)
    r = C.report(D.Dataset.create([str(a), str(b), str(partial)], 3), 3, 12, step=4)
    total = r["totals"]
    assert total["raw_core_counts"]["death_label"] == total["raw_core_counts"]["finish_label"] == 0
    assert total["raw_core_counts"]["alive_at_record_end"] == 3
    assert total["raw_core_counts"]["chains"] == 0     # missing K-step tails are not terminal-short chains
    outcomes = total["metadata_outcome_counts"]
    assert outcomes["metadata_finish"] == outcomes["metadata_death"] == outcomes["metadata_unknown"] == 1
    assert outcomes["terminal_outcome_without_terminal_row"] == 2 and outcomes["partial_episodes"] == 1
    assert outcomes["metadata_timeout_alive"] == 0
    assert total["unique_exact_trace_groups"] == 1
    assert total["groups_with_conflicting_known_metadata_outcomes"] == 1


def test_whole_shard_row_budget_is_checked_before_state_decode(repeated, monkeypatch):
    monkeypatch.setattr(D, "_read_npz", lambda *a: pytest.fail("state array decoded beyond budget"))
    result = C.report(repeated, 2, 11)
    assert result["budget"]["decoded_rows"] == 0 and result["budget"]["next_rows"] == 12
    assert result["budget"]["stop_reason"] == "row_budget"
    assert not result["measurement_complete_manifest"] and len(result["unprocessed_ids"]) == 2


def test_shard_and_soft_time_budget(repeated, monkeypatch):
    r = C.report(repeated, 1, 100)
    assert r["budget"]["processed_shards"] == 1 and r["budget"]["stop_reason"] == "shard_budget"
    ticks = iter(k * .1 for k in range(100))
    monkeypatch.setattr(C.time, "monotonic", lambda: next(ticks))
    r = C.report(repeated, 2, 100, max_seconds=.15)
    assert r["budget"]["stop_reason"] == "soft_time_budget" and r["budget"]["processed_shards"] == 1


def test_catalog_is_bounded_stratified_order_independent_and_does_not_decode_rows(tmp_path, monkeypatch):
    a, b = source(tmp_path / "a"), source(tmp_path / "b")
    for k in range(20):
        shard(a, k, [(0, k)])
    for k in range(3):
        shard(b, k, [(0, k)])
    monkeypatch.setattr(D, "_read_npz", lambda *a: pytest.fail("catalog must not decode states"))
    x = D.Dataset.catalog([str(a), str(b)], 6, seed=7, per_source_cap=3)
    y = D.Dataset.catalog([str(b), str(a)], 6, seed=7, per_source_cap=3)
    assert x["entries"] == y["entries"] and x["rows_decoded"] == 0
    assert x["files_listed"] == 23 and x["metadata_inspected"] == 6 and not x["complete_listed_metadata"]
    assert all(v["metadata_inspected"] == 3 for v in x["source_counts"].values())
    assert max(int(Path(e["id"]).stem.split("_")[1]) for e in x["entries"] if e["source"] == D.norm(a)) >= 13


def test_explicit_builder_selection_and_row_caps_preserve_trace_membership(tmp_path, monkeypatch):
    d = source(tmp_path / "data")
    paths = [shard(d, k, [(0, k)]) for k in range(5)]
    all_data = D.Dataset.create([str(d)], 5)
    selected = D.Dataset.create([str(d)], 2, selected_paths=[str(paths[4]), str(paths[2])], max_raw_rows=24)
    assert {e["id"] for e in selected.entries} == {D.norm(paths[2]), D.norm(paths[4])}
    assert selected.document["selection"]["new_raw_rows"] == 24
    for e in selected.entries:
        assert e["agents"] == next(a["agents"] for a in all_data.entries if a["id"] == e["id"])
    monkeypatch.setattr(D, "_read_npz", lambda *a: pytest.fail("state decode beyond selected row cap"))
    with pytest.raises(ValueError, match="max_raw_rows"):
        D.Dataset.create([str(d)], 1, selected_paths=[str(paths[4])], max_raw_rows=11)
    with pytest.raises(ValueError, match="max_shards"):
        D.Dataset.create([str(d)], 1, selected_paths=[str(paths[4]), str(paths[2])])
    with pytest.raises(ValueError, match="declared sources"):
        D.Dataset.create([str(d)], 1, selected_paths=[str(tmp_path / "elsewhere" / "wave_00000.npz")])


def test_cli_outputs_are_new_and_label_incomplete_coverage(repeated, tmp_path):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"sources": repeated.sources, "model_sha256": "declared evidence"}))
    catalog = tmp_path / "catalog.json"
    C.main(["inventory", "--source-request", str(request), "--max-candidates", "1", "--per-source-cap", "1",
            "--out", str(catalog)])
    inventory = json.loads(catalog.read_text())
    assert inventory["rows_decoded"] == 0 and inventory["metadata_inspected"] == 1
    assert inventory["source_request"]["sha256"] == D.file_hash(request)
    out = tmp_path / "coverage.json"
    argv = ["report", "--manifest", repeated.path, "--max-shards", "1", "--max-rows", "100", "--out", str(out)]
    C.main(argv)
    data = json.loads(out.read_text())
    assert not data["measurement_complete_manifest"] and data["training_ready"] == "not established"
    old = out.read_bytes()
    with pytest.raises(SystemExit, match="already exists"):
        C.main(argv)
    assert out.read_bytes() == old


def test_invalid_agent_trace_names_the_source_and_agent_without_repair(tmp_path):
    d = source(tmp_path / "data")
    rows = np.zeros(3, ROW)
    rows["agent"], rows["frame"], rows["x"] = 7, [0, 1, 1], [40, 41, 99]
    p = recording(d, rows, {}, [{"f": 0, "dt": 16.667}])
    before = p.read_bytes()
    with pytest.raises(ValueError) as error:
        D.Dataset.create([str(d)], 1, selected_paths=[str(p)], max_raw_rows=3)
    assert D.norm(p) in str(error.value) and "agent 7" in str(error.value) and "duplicate frame" in str(error.value)
    assert p.read_bytes() == before
