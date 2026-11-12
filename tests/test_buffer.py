"""Exact wave-end targets through the collector and PPO caller; no game, fitting or model rollout."""
import gzip
import json
import types

import numpy as np
import pytest

from wm.buffer import CAPTURE_PROTOCOL, Collector, chains, load_shards, transitions


def state(x=40, status=0):
    return [float(x), 30.5, 123.25, -45.5, 0x80050082, 7, status, 3]


def collector(tmp_path, source="planner"):
    c = Collector(str(tmp_path / "wm_data"), source=source)
    c.level_name = "test-level"
    return c


def observe(c, frame, states, alive=None, held=None, actions=None, rewards=None, wave=7):
    n = len(states)
    c.frame({"frame": frame, "dt": 16.667, "hz": [], "sh": []}, [], [])
    c.agents(wave, frame, states, [True] * n if alive is None else alive,
             [4] * n if held is None else held, [7] * n if actions is None else actions,
             [1.25] * n if rewards is None else rewards)


def end(states, frame=2, wave=7, why="done"):
    return {"frame": frame, "wave": wave, "why": why, "s": states,
            "status": [s[6] for s in states], "finish": [frame - 1 if s[6] == 2 else -1 for s in states],
            "hz": [[99.25, 55.5, 10, 10, 0]], "sh": [[10, 11.25, 12.5, 3]]}


def saved(c):
    return next(load_shards(c.dir))


@pytest.mark.parametrize("status,why", [(1, "done"), (2, "quota"), (0, "time")])
def test_exact_final_target_and_reward_alignment(tmp_path, status, why):
    c = collector(tmp_path, source="ppo")
    observe(c, 0, [state(40)], held=[3])
    observe(c, 1, [state(41)], held=[4], rewards=[1.25])
    final = state(43.25, status)
    message = end([final], why=why)
    c.wave_end(7, message, held=[5], rewards=[9.875])
    meta, rows, world = saved(c)
    assert rows["frame"].tolist() == [0, 1, 2]
    r = rows[-1]
    assert [float(r[k]) for k in ("x", "y", "vx", "vy")] == final[:4]
    assert [int(r[k]) for k in ("flags", "state", "status", "keys")] == final[4:]
    assert r["held"] == 5 and r["action"] == -1 and r["reward"] == 9.875
    tr = transitions(rows)
    assert tr["r"].tolist() == [1.25, 9.875]             # never copy the previous row's reward
    assert tr["terminated"].tolist() == [False, status != 0]
    assert tr["truncated_rows"]["frame"].tolist() == ([2] if status == 0 else [])
    assert [w["f"] for w in world] == [0, 1]             # target F is not a fabricated input-world frame
    assert all(w["dt"] == 16.667 for w in world)
    assert meta["capture_protocol"] == CAPTURE_PROTOCOL
    assert meta["final_capture"]["appended"] == [
        {"agent": 0, "held_source": "caller_post_action_mirror", "reward_source": "caller_final_step"}]
    context = meta["final_observation"]
    assert context["target_only"] and not context["input_context_complete"]
    assert context["hz"] == message["hz"] and context["sh"] == message["sh"]
    assert context["available_fields"] == ["hz", "sh"] and "dt" not in context
    assert context["missing_input_fields"] == ["dt", "pf", "fx", "tc"]
    c.wave_end(7, message, held=[5], rewards=[9.875])     # duplicate callback cannot write a second shard
    from pathlib import Path
    assert len(list(Path(c.dir).glob("wave_*.npz"))) == 1


def test_quota_mixed_early_terminal_final_terminal_and_survivor(tmp_path):
    c = collector(tmp_path, source="ppo")
    observe(c, 0, [state(10), state(20), state(30), state(40)])
    early = [state(11, 1), state(21, 2), state(31), state(41)]
    observe(c, 1, early, actions=[-1, -1, 7, 6], rewards=[-15, 70, .5, .5])
    observe(c, 2, [*early[:2], state(32), state(42)], alive=[False, False, True, True],
            actions=[-1, -1, 7, 6], held=[0, 0, 4, 0], rewards=[float("nan"), float("nan"), .75, .75])
    message = end([*early[:2], state(33, 2), state(43)], frame=3, why="quota")
    c.wave_end(7, message, held=[0, 0, 5, 0], rewards=[float("nan"), float("nan"), 99.75, -.01])
    meta, rows, _ = saved(c)
    assert rows[rows["agent"] == 0]["frame"].tolist() == [0, 1]
    assert rows[rows["agent"] == 1]["frame"].tolist() == [0, 1]
    assert rows[rows["agent"] == 2]["frame"].tolist() == [0, 1, 2, 3]
    assert rows[rows["agent"] == 3]["frame"].tolist() == [0, 1, 2, 3]
    assert meta["final_capture"]["skipped"] == [{"agent": 0, "reason": "already_terminal"},
                                                  {"agent": 1, "reason": "already_terminal"}]
    tr = transitions(rows)
    assert (tr["s2"]["status"] == 1).sum() == 1 and (tr["s2"]["status"] == 2).sum() == 2
    assert tr["truncated_rows"]["agent"].tolist() == [3]
    k = chains(rows, 4)
    assert len(k["s"]) == 5 and not k["full"].any()
    assert (k["s2"]["status"] != 0).all() and 3 not in k["s"]["agent"]


@pytest.mark.parametrize("case,reason", [
    ("no_s", "missing_exact_final_state"), ("short_s", "missing_final_state_fields"),
    ("no_held", "missing_exact_held_mirror"), ("gap", "nonconsecutive_final_frame"),
    ("wave", "wave_mismatch"), ("message_wave", "wave_mismatch"),
    ("nan", "invalid_final_physical_state"), ("status", "final_status_mismatch"),
    ("no_status", "missing_final_status"), ("no_action", "no_source_action"),
    ("same_frame", "already_observed_frame"), ("negative_frame", "invalid_final_frame")])
def test_missing_or_ambiguous_final_evidence_is_not_guessed(tmp_path, case, reason):
    c = collector(tmp_path)
    observe(c, 0, [state(40)], actions=[-1] if case == "no_action" else [7])
    message = end([state(41, 1)], frame=1)
    held, wave = [5], 7
    if case == "no_s": message.pop("s")
    if case == "short_s": message["s"][0].pop()
    if case == "no_held": held = None
    if case == "gap": message["frame"] = 2
    if case == "wave": wave = 8
    if case == "message_wave": message["wave"] = 8
    if case == "nan": message["s"][0][0] = float("nan")
    if case == "status": message["status"] = [2]
    if case == "no_status": message.pop("status")
    if case == "same_frame": message["frame"] = 0
    if case == "negative_frame": message["frame"] = -1
    c.wave_end(wave, message, held=held)
    meta, rows, _ = saved(c)
    assert len(rows) == 1 and not meta["final_capture"]["appended"]
    assert meta["final_capture"]["skipped"] == [{"agent": 0, "reason": reason}]
    assert rows[0]["x"] == 40 and rows[0]["status"] == 0


@pytest.mark.parametrize("why", ["stop", "S", "level change"])
def test_partial_cut_does_not_invent_post_update_state(tmp_path, why):
    c = collector(tmp_path)
    observe(c, 0, [state(40)])
    c.cut(why)
    meta, rows, world = saved(c)
    assert meta["partial"] and meta["partial_reason"] == why and len(rows) == 1
    assert meta["final_capture"]["skipped"][0]["reason"] == "partial_cut_without_post_update_observation"
    assert "final_observation" not in meta and len(world) == 1


def test_undefined_planner_reward_stays_nan_and_target_can_be_manifested(tmp_path):
    from test_dataset import source
    from wm.dataset import Dataset
    d = source(tmp_path / "data")
    c = Collector(str(d), source="planner")
    c.level_name = "test-level"
    observe(c, 0, [state(40)])
    c.wave_end(7, end([state(41, 2)], frame=1), held=[5])
    meta, rows, _ = saved(c)
    assert np.isnan(rows[-1]["reward"])
    assert meta["final_capture"]["appended"][0]["reward_source"] == "unknown_or_undefined"
    ds = Dataset.create([str(d)], 1)
    assert ds.entries[0]["agents"]["0"]["group"]


def test_ambiguous_duplicate_agent_frame_is_preserved_but_not_extended(tmp_path):
    c = collector(tmp_path)
    observe(c, 0, [state(40)])
    observe(c, 0, [state(99)])                       # conflicting duplicate; not a benign repeated row
    before = np.concatenate(c.rows).copy()
    c.wave_end(7, end([state(100, 1)], frame=1), held=[5], rewards=[-15])
    meta, rows, _ = saved(c)
    np.testing.assert_array_equal(rows, before)
    assert not meta["final_capture"]["appended"]
    assert meta["final_capture"]["skipped"] == [{"agent": 0, "reason": "ambiguous_duplicate_agent_frame"}]


def test_level_change_cuts_old_level_without_synthetic_next_state(tmp_path):
    c = collector(tmp_path)
    observe(c, 0, [state(40)])
    lv = types.SimpleNamespace(name="next-level", W=2, H=2, ox=0, oy=0, spawn=(10, 10), goals=[(30, 30)],
                               keys=[], bandages=[], portals=[], words0=np.zeros((2, 2), np.uint32))
    c.level(lv, {})
    meta, rows, _ = saved(c)
    assert meta["level"] == "test-level" and meta["partial_reason"] == "level change"
    assert len(rows) == 1 and rows["frame"].tolist() == [0]
    assert c.level_name == "next-level" and not c.rows


@pytest.mark.parametrize("phase", ["finish", "speed"])
def test_ppo_passes_the_same_computed_final_rewards_after_finishing_trajectories(tmp_path, phase):
    import torch
    from realgame import Trainer, R_STEP, R_STEP_SPEED, R_DEATH
    t = Trainer.__new__(Trainer)
    t.level, t.waves, t.phase, t.protected = types.SimpleNamespace(name="test-level"), 0, phase, False
    t.best_time, t.args = 30.0, types.SimpleNamespace(wave_seconds=20)
    t.traj = [{"a": [7], "r": [], "o": [np.zeros(2)], "done": False} for _ in range(4)]
    t.traj[3]["r"], t.traj[3]["done"] = [-3.0], True
    t.alive, t.fit, t.jump_held = [True, True, True, False], [None] * 4, [5, 8, 0, 3]
    t.agent_fitness = lambda *a, **kw: 0
    t.policy = types.SimpleNamespace(vf=lambda obs: torch.tensor(2.5))
    finished = []
    t.to_ppo = lambda tr, value: finished.append((list(tr["r"]), value))
    message = {"wave": 7, "frame": 5, "status": [1, 2, 0, 1], "finish": [-1, 4, -1, -1]}
    expected_step = R_STEP_SPEED if phase == "speed" else R_STEP
    expected = [expected_step + R_DEATH, expected_step + t.finish_reward(4), expected_step]
    class Captured(Exception): pass
    def capture(wave, m, *, held, rewards):
        assert wave == 7 and m is message and held == [5, 8, 0, 3]
        assert rewards[:3] == expected and np.isnan(rewards[3])
        assert [tr["r"][-1] for tr in t.traj[:3]] == expected
        assert t.traj[3]["r"] == [-3.0]                 # already-finished lives do not get another reward
        assert len(finished) == 3 and [v for _, v in finished] == [0.0, 0.0, 2.5]
        raise Captured
    t.wm = types.SimpleNamespace(wave_end=capture)
    with pytest.raises(Captured):
        t.on_wave_end(message)
