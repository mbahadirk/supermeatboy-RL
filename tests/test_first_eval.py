"""First-life protocol and isolation through the real runner's fake-script boundary; no game or recordings."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import agent
from plan import first_eval as E
from wm.model import WorldModel


class Script:
    def __init__(self):
        self.posts = []

    def post(self, value):
        self.posts.append(value)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "model.pt"
    model = WorldModel(members=1, hidden=8, layers=1)
    model.save(path, {"levels": ["seen-level"]})
    return path


def make_runner(tmp_path, checkpoint, monkeypatch, *extra):
    # Deliberately unusable persistent files: evaluation must not even try their loaders.
    paths = [tmp_path / (name + ".json") for name in ("deaths", "moves", "best")]
    for p in paths:
        p.write_text("USER DATA: leave unchanged", encoding="utf-8")
    def forbidden(*a, **kw):
        raise AssertionError("persistent memory accessed")
    monkeypatch.setattr(agent.S.DeathMemory, "load", forbidden)
    monkeypatch.setattr(agent.S.DeathMemory, "save", forbidden)
    monkeypatch.setattr(agent.MoveMemory, "load", forbidden)
    monkeypatch.setattr(agent.MoveMemory, "save", forbidden)
    old = agent.BestTimes._load
    def best_load(self):
        assert self.path is None
        return old(self)
    monkeypatch.setattr(agent.BestTimes, "_load", best_load)
    args = agent.build_parser().parse_args([
        "--eval-first", "test-level", "--model", str(checkpoint), "--name", str(tmp_path / "run"),
        "--telemetry", "off", "--modes", "off", "--wave-seconds", str(2 / 60),
        "--death-memory", str(paths[0]), "--move-memory", str(paths[1]), "--best-times", str(paths[2]), *extra])
    r = agent.Runner(args)
    r.script = Script()
    return r, paths


def send(r, ev, data=None, **kw):
    r.on_message({"type": "send", "payload": {"ev": ev, **kw}}, data)


def info(r, name="test-level", agents=1):
    words = np.zeros((6, 8), dtype="<u4")
    words[0, :] = 1
    send(r, "debug", msg="time step locked to the fixed step")
    send(r, "info", words.tobytes(), name=name, agents=agents, W=8, H=6, ox=0, oy=0,
         spawn=[30, 30], goals=[[130, 30]], hz=[], lethalTypes=[])


def frame(r, f, *, wave=1, dt=16.667, x=30):
    send(r, "frame", frame=f, wave=wave, dt=dt, hz=[], pf=[[80 + f, 60, 1]],
         s=[[x, 30, 0, 0, 0x50081, 0, 0, 0]])
    return r.script.posts[-1]["payload"]


def end(r, status=2, why="quota", *, wave=1):
    send(r, "wave_end", wave=wave, frame=2, why=why, status=[status], finish=[1 if status == 2 else -1],
         s=[[32, 30, 0, 0, 0x50081, 0, status, 0]], hz=[], sh=[])


def read(r, name="evaluation.json"):
    return json.loads((Path(r.run_dir) / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("extra", [["--learn"], ["--search"], ["--demo", "x.json"],
                                  ["--demo-level", "1-1"], ["--wm-collect"], ["--explore", "1"],
                                  ["--agents", "2"], ["--no-lock-step"],
                                  ["--wave-seconds", "nan"], ["--wave-seconds", "0.001"]])
def test_conflicts_fail_before_model_or_run_access(tmp_path, extra):
    args = agent.build_parser().parse_args(["--eval-first", "1-18", "--name", str(tmp_path / "run"), *extra])
    with pytest.raises(SystemExit, match="--eval-first"):
        agent.Runner(args)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("status,why,outcome", [(2, "quota", "success"), (1, "done", "death"),
                                               (0, "time", "timeout")])
def test_single_life_outcomes_isolation_and_stop(tmp_path, checkpoint, monkeypatch, capsys, status, why, outcome):
    r, paths = make_runner(tmp_path, checkpoint, monkeypatch)
    try:
        info(r)
        frame(r, 0)
        frame(r, 1)
        # The previous/current observations of THIS life are retained, but never become a prior-wave recording.
        assert r.fw.prev[0] == 0 and r.fw.last[0] == 1 and not r.fw.ref
        worlds, boxes, recorded = r.fw.worlds(1, 2, 16.667)
        assert recorded == 0
        assert worlds[1].pf[0][0] > worlds[0].pf[0][0]
        end(r, status, why)
        assert r.stopping and not r.fw.ref
        result = read(r)
        assert result["validity"] == "valid" and result["outcome"] == outcome
        assert result["frames"] == 2 and result["game_seconds"] == pytest.approx(0.033334)
        assert result["attempt_wall_seconds"] >= 0
        assert "planning_ms_median" in result["metrics"]
        # endWave waits for 'go'; the next frame's stop is handled by realgame.js BEFORE any input application.
        assert r.script.posts[-1] == {"type": "go", "payload": {"agents": 1}}
        before = r.st["decisions"]
        reply = frame(r, 0, wave=2)
        assert reply["stop"] is True and reply["a"] == [[0, 0, 0, 0]]
        assert r.st["decisions"] == before
        assert read(r) == result
        manifest = read(r, "run.json")
        assert manifest["model"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert manifest["training_overlap"] == "unknown"
        assert manifest["effective_config"]["death_memory"] is None
        assert r.moves is None and len(r.deaths) == 0
        E.main(["report", r.run_dir])
        assert f"valid | {outcome} |" in capsys.readouterr().out
    finally:
        r.close()
    assert all(p.read_text(encoding="utf-8") == "USER DATA: leave unchanged" for p in paths)
    assert not (Path(r.run_dir) / "wm_data").exists()


@pytest.mark.parametrize("case", ["mid_frame", "wrong_spawn", "wrong_level", "missing_lock", "nan_dt",
                                  "inf_dt", "wrong_wave", "skipped_frame", "terminal_wave", "script_error"])
def test_invalid_evidence_never_becomes_a_success(tmp_path, checkpoint, monkeypatch, case):
    r, _ = make_runner(tmp_path, checkpoint, monkeypatch)
    try:
        info(r, name="other-level" if case == "wrong_level" else "test-level")
        if case == "missing_lock":
            r.lock = None
        frame(r, 7 if case == "mid_frame" else 0, x=60 if case == "wrong_spawn" else 30,
              dt=float("nan") if case == "nan_dt" else float("inf") if case == "inf_dt" else 16.667)
        if case in ("wrong_wave", "skipped_frame", "terminal_wave", "script_error"):
            if case == "script_error":
                r.on_message({"type": "error", "description": "broken"}, None)
            elif case == "terminal_wave":
                frame(r, 1)
                end(r, wave=2)
            else:
                frame(r, 2 if case == "skipped_frame" else 1, wave=2 if case == "wrong_wave" else 1)
        assert r.stopping and read(r)["validity"] == "invalid"
        end(r)
        assert read(r)["outcome"] is None
    finally:
        r.close()


@pytest.mark.parametrize("case", ["stop", "level_change", "manual_finish", "reset"])
def test_interruption_is_incomplete(tmp_path, checkpoint, monkeypatch, case):
    r, _ = make_runner(tmp_path, checkpoint, monkeypatch)
    info(r)
    frame(r, 0)
    if case == "level_change":
        send(r, "level", name="other-level")
    elif case == "manual_finish":
        monkeypatch.setattr(r, "on_save", lambda *a: pytest.fail("evaluation must not modify game saves"))
        send(r, "save", level="test-level")
    elif case == "reset":
        frame(r, 1)
        end(r, status=0, why="reset")
    r.close()
    assert read(r)["validity"] == "incomplete" and read(r)["outcome"] is None


def test_planner_error_is_invalid_and_stops_before_action(tmp_path, checkpoint, monkeypatch):
    r, _ = make_runner(tmp_path, checkpoint, monkeypatch)
    info(r)
    def broken(*a, **kw):
        raise RuntimeError("test planning failure")
    monkeypatch.setattr(r.planner, "plan", broken)
    reply = frame(r, 0)
    r.close()
    assert reply["stop"] and read(r)["validity"] == "invalid"


def test_existing_run_is_never_overwritten(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    evidence = folder / "events.jsonl"
    evidence.write_text("old evidence", encoding="utf-8")
    args = agent.build_parser().parse_args(["--eval-first", "1-18", "--name", str(folder)])
    with pytest.raises(SystemExit, match="empty run folder"):
        agent.Runner(args)
    assert evidence.read_text(encoding="utf-8") == "old evidence"


def test_snapshot_hashes_the_loaded_bytes(tmp_path, checkpoint, monkeypatch):
    original = checkpoint.read_bytes()
    old_load = WorldModel.load
    def replace_source(source):
        checkpoint.write_bytes(b"source changed after snapshot")
        return old_load(source)
    monkeypatch.setattr(WorldModel, "load", replace_source)
    model, identity = E.load_checkpoint(checkpoint)
    assert identity["sha256"] == hashlib.sha256(original).hexdigest()
    assert model.info["levels"] == ["seen-level"]


@pytest.mark.parametrize("listed", [False, True])
def test_declared_holdout_is_not_proof_of_unseen_training(tmp_path, checkpoint, monkeypatch, capsys, listed):
    import wm.buffer
    monkeypatch.setattr(wm.buffer, "eval_levels", lambda: {"test-level"})
    if listed:
        WorldModel(members=1, hidden=8, layers=1).save(checkpoint, {"levels": ["test-level"]})
    r, _ = make_runner(tmp_path, checkpoint, monkeypatch)
    r.close()
    E.main(["report", r.run_dir])
    out = capsys.readouterr().out
    overlap = "seen" if listed else "unknown"
    assert "attempt wall seconds" in out and overlap in out and "incomplete" in out
    manifest = read(r, "run.json")
    assert manifest["declared_holdout"] is True and manifest["training_overlap"] == overlap
    assert manifest["unseen_evidence"].startswith("checkpoint lists" if listed else "not established")
