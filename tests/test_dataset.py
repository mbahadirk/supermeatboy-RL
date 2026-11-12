"""Bounded synthetic recordings exercise split ownership without real-game data, training, or model rollout."""
import gzip
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from wm import dataset as D
from wm.buffer import ROW
from wm.model import WorldModel


def source(folder, level="test-level"):
    folder.mkdir(parents=True)
    (folder / "levels").mkdir()
    words = np.zeros((8, 96), np.uint32)
    words[0] = 1
    meta = {"name": level, "W": 96, "H": 8, "ox": 0, "oy": 0,
            "spawn": [40, 30], "goals": [[1000, 30]]}
    np.savez_compressed(folder / "levels" / (level + ".npz"), words=words, meta=json.dumps(meta))
    return folder


def shard(folder, number, agents, level="test-level", wave=1, reward=1):
    rows = []
    for agent, kind in agents:
        r = np.zeros(12, ROW)
        r["wave"], r["frame"], r["agent"] = wave, np.arange(12), agent
        r["x"], r["y"], r["vx"] = 40 + kind * 2 + np.arange(12), 30, 10 + kind
        r["flags"], r["action"], r["reward"] = 0x50081, 6, reward
        r["action"][-1] = -1
        rows.append(r)
    p = folder / f"wave_{number:05d}.npz"
    np.savez_compressed(p, rows=np.concatenate(rows), meta=json.dumps({"level": level, "wave": wave}))
    world = [{"f": f, "dt": 16.667, "hz": [], "pf": []} for f in range(12)]
    with gzip.open(str(p)[:-4] + "_world.json.gz", "wt", encoding="utf-8") as f:
        json.dump(world, f)
    return p


@pytest.fixture
def snapshot(tmp_path):
    a, b = source(tmp_path / "a"), source(tmp_path / "b")
    shard(a, 0, [(k, k) for k in range(32)])
    shard(b, 8, [(k + 100, k) for k in range(32)], wave=999, reward=float("nan"))
    ds = D.Dataset.create([str(b), str(a)], 2)
    ds.save(tmp_path / "manifest.json")
    return ds


def model_for(ds, path=None, step=1):
    from wm.features import FEAT_DIM
    from wm.protocol import FEATURE_PROTOCOL
    m = WorldModel(n_in=FEAT_DIM + (step - 1) * 10, members=1, hidden=8, layers=1, step=step)
    m.info = {"data": ds.sources, "dataset": ds.reference(), "feature_protocol": FEATURE_PROTOCOL}
    if path:
        m.save(path, m.info)
    return m


def test_per_agent_replays_ignore_wave_agent_reward_and_source_order(snapshot):
    ds = snapshot
    a, b = ds.entries
    assert {a["agents"][str(k)]["group"] for k in range(32)} == {
        b["agents"][str(k + 100)]["group"] for k in range(32)}
    for k in range(32):
        assert a["agents"][str(k)] == b["agents"][str(k + 100)]
    again = D.Dataset.create(list(reversed(ds.sources)), 2)
    assert again.hash == ds.hash
    assert {v["split"] for e in ds.entries for v in e["agents"].values()} == {"train", "validation"}


def test_partial_manifest_keeps_declared_sources_and_extension_membership(tmp_path):
    a, b = source(tmp_path / "a"), source(tmp_path / "b")
    shard(a, 0, [(0, 1)])
    shard(b, 0, [(0, 2)])
    ds = D.Dataset.create([str(b), str(a)], 1)
    assert ds.source_list([str(a), str(b)]) == sorted([D.norm(a), D.norm(b)])
    assert ds.document["coverage"] == {"selected": 1, "candidates": 2, "complete": False,
                                      "new_shard_cap": 1, "scope": "listed sources at creation; no whole-archive claim"}
    old = ds.entries[0]
    shard(a, 1, [(0, 3)])
    assert len(list(ds.iter_shards())) == 1        # newly appearing files never enter the old snapshot
    extended = D.Dataset.create([str(b), str(a)], 2, parent=ds)
    assert old == extended.entries[0] and extended.document["parent_sha256"] == ds.hash
    with pytest.raises(ValueError, match="different manifest"):
        extended.require_model(model_for(ds))
    p = tmp_path / "immutable.json"
    ds.save(p)
    with pytest.raises(FileExistsError):
        extended.save(p)


@pytest.mark.parametrize("which", ["shard", "world", "level"])
def test_modified_content_is_rejected_at_owner_boundary(snapshot, which):
    e = snapshot.entries[0]
    p = Path(e["id"] if which == "shard" else e[which + "_path"])
    p.write_bytes(p.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="content changed"):
        list(snapshot.iter_shards())


def test_missing_world_and_manifest_tampering_fail_closed(snapshot):
    body = json.loads(json.dumps(snapshot.document))
    body["entries"][0]["agents"]["0"]["split"] = "validation" if body["entries"][0]["agents"]["0"]["split"] == "train" else "train"
    body["sha256"] = D.digest(D.canonical({k: v for k, v in body.items() if k != "sha256"}))
    with pytest.raises(ValueError, match="split was changed"):
        D.Dataset(body)
    Path(snapshot.entries[0]["world_path"]).unlink()
    with pytest.raises(FileNotFoundError):
        list(snapshot.iter_shards())


def test_signed_zero_and_invalid_physics_trace():
    r = np.zeros(3, ROW)
    r["frame"] = range(3)
    z = r.copy()
    z["vx"] = -0.0
    assert D.trace_id(r, "geometry") == D.trace_id(z, "geometry")
    z["x"][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        D.trace_id(z, "geometry")


def test_heldout_levels_never_become_train_or_validation(tmp_path):
    s = source(tmp_path / "held", "1-18")
    shard(s, 0, [(0, 1)], level="1-18")
    ds = D.Dataset.create([str(s)], 1, excluded={"1-18"})
    assert ds.entries[0]["agents"]["0"]["split"] == "excluded"
    assert list(ds.iter_shards(split="train")) == list(ds.iter_shards(split="validation")) == []


def test_user_excluded_demo_sources_are_rejected_but_regular_play_is_allowed(tmp_path):
    for level in ("1-16", "4-6"):
        excluded = tmp_path / ("demo_" + level + "_20261108-171806") / "wm_data"
        with pytest.raises(ValueError, match="user-excluded demo"):
            D.Dataset.create([str(excluded)], 1)
        regular = source(tmp_path / ("regular_" + level), level)
        shard(regular, 0, [(0, 0)], level=level)
        assert len(D.Dataset.create([str(regular)], 1).entries) == 1


def test_features_derive_validation_from_trace_not_shard_counter(snapshot):
    from wm.features import build
    data = build(snapshot.sources, frac=1, dataset=snapshot, log=lambda *a: None)
    val = D.sample_validation(data, strict=True)
    by_kind = {10 + k: snapshot.entries[0]["agents"][str(k)]["split"] == "validation" for k in range(32)}
    assert val.tolist() == [by_kind[int(round(v * 400))] for v in data["X"][:, 0]]
    assert val.any() and (~val).any()
    with pytest.raises(ValueError, match="both training and validation"):
        D.sample_validation({"validation": np.zeros(4, bool)}, strict=True)


def test_selfcorrect_only_passes_training_agents_to_rollouts(snapshot, monkeypatch):
    from wm import selfcorrect as S
    class Probe(Exception):
        pass
    def rollout(model, geom, rows, *rest):
        expected = snapshot.entries[0]["agents"]
        assert all(expected[str(int(a))]["split"] == "train" for a in rows["agent"])
        raise Probe
    monkeypatch.setattr(S, "rollout_mean", rollout)
    with pytest.raises(Probe):
        S.generate(snapshot.sources, model_for(snapshot), K=3, starts=1, dataset=snapshot)


def test_compare_error_stream_only_passes_validation_agents(snapshot, monkeypatch):
    from wm import compare_steps as C
    seen = []
    def no_rollout(d, lv, rows, world, *rest):
        e = next(e for e in snapshot.entries if e["source"] == d)
        assert all(e["agents"][str(int(a))]["split"] == "validation" for a in rows["agent"])
        seen.append(d)
        return True
    monkeypatch.setattr(C, "roll_wave", no_rollout)
    C.wave_errors([model_for(snapshot)], [e["id"] for e in snapshot.entries], dataset=snapshot)
    assert set(seen) == set(snapshot.sources)
    with pytest.raises(ValueError, match="legacy"):
        snapshot.require_model(types.SimpleNamespace(info={}))
    with pytest.raises(ValueError, match="manifest"):
        C.held_out(model_for(snapshot))


@pytest.mark.parametrize("step", [1, 4])
def test_training_masks_and_metadata_without_training(snapshot, tmp_path, monkeypatch, step):
    from wm import train as T
    seen = []
    def no_training(model, ts, data, va_idx, *rest):
        # No transition from a validation agent, even a demo, reaches the optimizer.
        va_vel = set(data["X"][va_idx, 0].tolist())
        assert not va_vel.intersection(ts["X"][:, 0].tolist())
        seen.append(len(ts["X"]))
    monkeypatch.setattr(T, "fit", no_training)
    monkeypatch.setattr(T, "report", lambda *a: "no model rollout in this test")
    monkeypatch.setattr(T, "immediate", lambda: None)
    from wm import selfcorrect
    monkeypatch.setattr(selfcorrect, "generate", lambda *a, **kw: {})
    out = tmp_path / "trained-contract"
    monkeypatch.setattr(sys, "argv", ["wm.train", "--manifest", snapshot.path, "--out", str(out),
                                      "--epochs", "1", "--members", "1", "--hidden", "8", "--layers", "1",
                                      "--sample-frac", "1", "--batch", "8", "--step", str(step),
                                      "--self-correct", "3" if step == 1 else "0"])
    T.main()
    m = WorldModel.load(out / "wm_model.pt")
    assert seen and m.info["dataset"]["sha256"] == snapshot.hash
    stats = m.info["training"]
    assert stats["fit_samples"] == seen[0] and stats["actual_sc_samples"] == stats["self_correction_epochs"] == 0
    assert m.info["train"] + m.info["held_out"] == 2 * 32 * (12 - step)
    if step == 4:
        from wm.features import build
        data = build(snapshot.sources, frac=1, dataset=snapshot, step=4, log=lambda *a: None)
        corr = (data["raw"] - data["prior"])[data["full"] & ~data["validation"]]
        np.testing.assert_allclose(m.res_scale, np.maximum(corr.std(0), 1e-3), rtol=1e-5)


@pytest.mark.parametrize("module,extra", [("wm.train", []), ("wm.learner", ["--init", "not-read.pt"]),
                                        ("wm.evaluate", ["--model", "not-read.pt"]),
                                        ("wm.compare_steps", ["--models", "not-read.pt"])])
def test_clis_require_explicit_protocol(module, extra, monkeypatch):
    import importlib
    m = importlib.import_module(module)
    monkeypatch.setattr(sys, "argv", [module, *extra])
    with pytest.raises(SystemExit) as e:
        m.main()
    assert e.value.code == 2


def test_strict_evaluate_rejects_step_four_before_any_rollout(snapshot, tmp_path, monkeypatch):
    from wm import evaluate as E
    p = tmp_path / "step4.pt"
    model_for(snapshot, p, step=4)
    monkeypatch.setattr(E, "shard_cases", lambda *a: pytest.fail("unexpected rollout"))
    monkeypatch.setattr(sys, "argv", ["wm.evaluate", "--manifest", snapshot.path, "--model", str(p)])
    with pytest.raises(ValueError, match="compare_steps"):
        E.main()


def test_second_strict_model_cannot_hide_in_legacy_comparison(snapshot, tmp_path, monkeypatch):
    from wm import compare_steps as C
    old, strict = tmp_path / "old.pt", tmp_path / "strict.pt"
    WorldModel(members=1, hidden=8, layers=1).save(old, {})
    model_for(snapshot, strict)
    monkeypatch.setattr(C, "immediate", lambda: None)
    monkeypatch.setattr(sys, "argv", ["wm.compare_steps", *snapshot.sources, "--legacy-split", "--models", str(old), str(strict)])
    with pytest.raises(ValueError, match="strict checkpoint"):
        C.main()


def test_parent_step_mismatch_rejected_before_feature_scan(snapshot, tmp_path, monkeypatch):
    from wm import train as T
    p = tmp_path / "parent4.pt"
    model_for(snapshot, p, step=4)
    monkeypatch.setattr(T, "build", lambda *a, **kw: pytest.fail("unexpected scan"))
    monkeypatch.setattr(sys, "argv", ["wm.train", "--manifest", snapshot.path, "--init", str(p)])
    with pytest.raises(ValueError, match="--step"):
        T.main()


def test_learner_stats_and_exact_parent_identity(snapshot, tmp_path, monkeypatch):
    from wm import learner as L
    p = tmp_path / "parent.pt"
    model_for(snapshot, p)
    out = tmp_path / "live"
    out.mkdir()
    args = types.SimpleNamespace(out=str(out), dataset=snapshot, demos=None, self_demos=None, demo_share=0)
    stats = {"train_transitions": 40, "validation_transitions": 10, "fit_samples": 40, "demo_samples": 0,
             "epochs": 1, "levels": ["test-level"], "sample_frac": 1}
    def no_training(model, *rest):
        model.cycle_stats = stats
        return 50
    monkeypatch.setattr(L, "fine_tune", no_training)
    monkeypatch.setattr(L, "errors", lambda *a, **kw: {"16": [1, 2, 1], "48": [1, 2, 1]})
    state = {"base": str(p), "version": 0, "cycles": 0, "eval_waves": [],
             "errors": {"16": [2, 3, 1], "48": [2, 3, 1]}, "dataset": snapshot.reference()}
    L.cycle(args, state, snapshot.sources, {}, 0)
    info = WorldModel.load(out / "wm_model.pt").info
    assert info["parent_checkpoint"]["sha256"] == D.file_hash(p)
    assert info["train"] == 40 and info["held_out"] == 10 and info["training"]["demo_samples"] == 0
    assert info["dataset"]["sha256"] == snapshot.hash and info["levels"] == ["test-level"]


@pytest.mark.parametrize("wrapper_log", [False, True])
def test_strict_learner_startup_preserves_old_model_and_allows_only_wrapper_log(snapshot, tmp_path, monkeypatch, wrapper_log):
    from wm import learner as L
    from shutdown import immediate
    import shutdown
    monkeypatch.setattr(shutdown, "immediate", lambda: None)
    init = tmp_path / "initial.pt"
    model_for(snapshot, init)
    out = tmp_path / "live"
    out.mkdir()
    if wrapper_log:
        (out / "learner_output.txt").write_text("wrapper log")
    else:
        (out / "wm_model.pt").write_bytes(b"OLD CHECKPOINT")
    monkeypatch.setattr(L, "errors", lambda *a, **kw: {"16": [1, 2, 1], "48": [1, 2, 1]})
    calls = []
    def no_cycle(args, state, folders, counts, n_self):
        calls.append(state)
        return dict(state, cycles=1)
    monkeypatch.setattr(L, "cycle", no_cycle)
    argv = ["--manifest", snapshot.path, "--init", str(init), "--out", str(out), "--once"]
    if wrapper_log:
        L.main(argv)
        assert len(calls) == 1 and (out / "learner_output.txt").read_text() == "wrapper log"
    else:
        with pytest.raises(ValueError, match="empty output folder"):
            L.main(argv)
        assert not calls and (out / "wm_model.pt").read_bytes() == b"OLD CHECKPOINT"


def test_agent_requires_protocol_and_forwards_it_before_attachment(tmp_path, monkeypatch):
    import agent
    args = agent.build_parser().parse_args(["--learn"])
    with pytest.raises(SystemExit, match="before attaching"):
        agent.validate_learning_args(args)
    monkeypatch.setattr(sys, "argv", ["agent.py", "--learn"])
    monkeypatch.setattr(agent, "Runner", lambda *a: pytest.fail("must reject before loading a model or attaching"))
    with pytest.raises(SystemExit, match="--learn requires"):
        agent.main()
    args = agent.build_parser().parse_args(["--learn", "--legacy-split", "--live-dir", str(tmp_path / "live")])
    agent.validate_learning_args(args)


def test_evaluate_cli_only_supplies_validation_rows(snapshot, tmp_path, monkeypatch):
    from wm import evaluate as E
    p = tmp_path / "step1.pt"
    model_for(snapshot, p)
    seen = []
    def no_rollout(meta, rows, *rest):
        e = snapshot.entry(meta)
        assert all(e["agents"][str(int(a))]["split"] == "validation" for a in rows["agent"])
        seen.append(e["id"])
        return []
    monkeypatch.setattr(E, "shard_cases", no_rollout)
    monkeypatch.setattr(E, "summarise", lambda *a: "no rollout")
    monkeypatch.setattr(sys, "argv", ["wm.evaluate", "--manifest", snapshot.path, "--model", str(p)])
    E.main()
    assert len(seen) == 2


def test_snapshot_learner_does_not_train_again_when_new_files_appear(snapshot, tmp_path, monkeypatch):
    from wm import learner as L
    import shutdown
    monkeypatch.setattr(shutdown, "immediate", lambda: None)
    init = tmp_path / "initial.pt"
    model_for(snapshot, init)
    monkeypatch.setattr(L, "errors", lambda *a, **kw: {"16": [1, 2, 1], "48": [1, 2, 1]})
    monkeypatch.setattr(L, "discover", lambda *a: pytest.fail("snapshot learner must not discover training folders"))
    cycles, sleeps = [], []
    def no_cycle(args, state, folders, counts, n_self):
        cycles.append(1)
        return dict(state, cycles=1)
    monkeypatch.setattr(L, "cycle", no_cycle)
    class EndProbe(Exception):
        pass
    def no_sleep(seconds):
        sleeps.append(1)
        if len(sleeps) == 1:
            shard(Path(snapshot.sources[0]), 99, [(0, 99)])
        else:
            raise EndProbe
    monkeypatch.setattr(L.time, "sleep", no_sleep)
    with pytest.raises(EndProbe):
        L.main(["--manifest", snapshot.path, "--init", str(init), "--out", str(tmp_path / "live")])
    assert len(cycles) == 1 and len(sleeps) == 2
    assert len(snapshot.entries) == 2
    assert len(L.eval_waves(model_for(snapshot), n=1, dataset=snapshot)) == 1


def test_agent_forwards_manifest_and_rejects_wrong_live_checkpoint(snapshot, tmp_path, monkeypatch):
    import agent
    import subprocess
    import os
    checkpoint = tmp_path / "initial.pt"
    model_for(snapshot, checkpoint)
    args = agent.build_parser().parse_args([
        "--learn", "--dataset-manifest", snapshot.path, "--model", str(checkpoint),
        "--name", str(tmp_path / "run"), "--live-dir", str(tmp_path / "live"), "--telemetry", "off",
        "--death-memory", str(tmp_path / "deaths.json"), "--move-memory", "none",
        "--best-times", str(tmp_path / "best.json")])
    r = agent.Runner(args)
    commands = []
    class Child:
        pid = 123
        def terminate(self): pass
        def wait(self, *args): pass
    def fake_child(argv, **kw):
        commands.append(argv)
        return Child()
    monkeypatch.setattr(subprocess, "Popen", fake_child)
    try:
        r.start_learner()
        assert commands[0][-2:] == ["--manifest", snapshot.path]
        old, mt = r.model, r.live_mtime
        p = Path(args.live_dir) / "wm_model.pt"
        WorldModel(members=1, hidden=8, layers=1).save(p, {})
        os.utime(p, (mt + 10, mt + 10))
        r.load_live()
        assert r.model is old and r.live_mtime == mt
        model_for(snapshot, p)
        os.utime(p, (mt + 11, mt + 11))
        r.load_live()
        assert r.model is not old
    finally:
        r.close()


def test_agent_forwards_explicit_legacy_flag(tmp_path, monkeypatch):
    import agent
    import subprocess
    checkpoint = tmp_path / "legacy.pt"
    WorldModel(members=1, hidden=8, layers=1).save(checkpoint, {})
    args = agent.build_parser().parse_args([
        "--learn", "--legacy-split", "--model", str(checkpoint), "--name", str(tmp_path / "run"),
        "--live-dir", str(tmp_path / "live"), "--telemetry", "off", "--move-memory", "none",
        "--death-memory", str(tmp_path / "deaths.json"), "--best-times", str(tmp_path / "best.json")])
    r = agent.Runner(args)
    commands = []
    class Child:
        pid = 123
        def terminate(self): pass
        def wait(self, *args): pass
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kw: commands.append(argv) or Child())
    try:
        r.start_learner()
        assert commands[0][-1] == "--legacy-split" and "--manifest" not in commands[0]
    finally:
        r.close()
