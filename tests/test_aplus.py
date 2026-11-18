"""A+ rewards through recorded transitions/online features, and native finish requests without a game."""
from types import SimpleNamespace

import numpy as np
import pytest

import agent
from wm import rewards as R
from wm.buffer import Collector, transitions, load_shards


def state(status=0):
    return [40.0, 40.0, 0.0, 0.0, 0, 0, status, 0]


@pytest.mark.parametrize("locked,frames,level,reward,grade", [
    (True, 100, "known", 15, True), (True, 101, "known", 5, False),
    (False, 50, "known", 5, None), (None, 50, "known", 5, None),
    (True, 50, "unknown", 5, None),
    (True, 0, "known", 5, None),
])
def test_finish_reward_boundary(monkeypatch, locked, frames, level, reward, grade):
    monkeypatch.setattr(R, "a_plus_time", lambda lv: 1.0 if lv == "known" else None)
    out = R.finish_outcome(level, frames, 10.0, locked)
    assert out["reward"] == reward and out["a_plus"] is grade
    assert out["seconds"] == frames / 100


@pytest.mark.parametrize("step", [1, 4])
@pytest.mark.parametrize("terminal_in_frame", [False, True])
def test_reward_survives_capture_and_online_features(tmp_path, monkeypatch, step, terminal_in_frame):
    from wm.online import featurize, aplus_examples, own_probabilities
    from wm.features import layout, EVENTS
    monkeypatch.setattr(R, "a_plus_time", lambda lv: .05)
    c = Collector(str(tmp_path / "data"), source="planner")
    c.meta_extra["reward_protocol"] = R.REWARD_PROTOCOL
    lv = SimpleNamespace(name="test", W=8, H=8, ox=0, oy=0, spawn=(40, 40), goals=[(80, 40)],
                         keys=[], bandages=[], portals=[], words0=np.zeros((8, 8), np.uint32))
    c.level(lv, {})
    # Agent 0 finishes at 3 frames (A+); agent 1 at 6 (ordinary finish).
    for frame in range(6 + int(terminal_in_frame)):
        states = [state(2 if frame >= 3 else 0), state(2 if frame >= 6 else 0)]
        c.frame({"frame": frame, "dt": 16.667, "hz": []}, [])
        c.agents(1, frame, states, [frame <= 3, True], [0, 0],
                 [-1 if s[6] else 0 for s in states], R.outcome_rewards("test", frame, states, 16.667, True))
    final = {"wave": 1, "frame": 6, "s": [state(2), state(2)], "status": [2, 2], "finish": [2, 5]}
    c.wave_end(1, final, held=[0, 0], rewards=R.outcome_rewards("test", 6, final["s"], 16.667, True))
    meta, rows, _ = next(load_shards(c.dir))
    tr = transitions(rows)
    assert tr["r"][tr["s2"]["status"] == 2].tolist() == [15.0, 5.0]
    assert aplus_examples(meta, rows) == [(0, 0, 3)]
    assert aplus_examples({}, rows) == []             # legacy/PPO rewards are never guessed to be A+
    model = SimpleNamespace(step=step, info={"feature_protocol": "tile-shapes-v1"},
                            mu=SimpleNamespace(numel=lambda: layout("tile-shapes-v1").n_in(step)),
                            res_scale=np.array([20, 20, 100, 100]))
    b = featurize(str(tmp_path / "data" / "wave_00000.npz"), model, str(tmp_path))
    terminal = b["yb"][:, EVENTS["finish"]] > 0
    assert set(b["yr"][terminal]) == {1.0, 3.0}      # scaled once, bonus counted once per chain
    assert set(b["sample_weight"]) == {1.0, 2.0}
    p = own_probabilities(b)
    assert p[b["sample_weight"] == 2][0] == pytest.approx(2 * p[b["sample_weight"] == 1][0])


def runner(tmp_path):
    # Exercise the real runner/collector callbacks without loading a planning network.
    args = agent.build_parser().parse_args([
        "--agents", "0", "--explore", "2", "--wm-collect", "--telemetry", "off", "--no-overlay",
        "--name", str(tmp_path / "run"), "--best-times", str(tmp_path / "best.json"),
        "--death-memory", str(tmp_path / "deaths.json"), "--move-memory", "none"])
    r = agent.Runner(args)
    r.args.realtest, r.n_plan, r.n_explore = True, 1, 1
    r.level, r.lock, r.dt_ms = SimpleNamespace(name="1-1"), True, 16.667
    r.new_wave(2)
    r.field_dirty = False
    r.script = SimpleNamespace(posts=[])
    r.script.post = r.script.posts.append
    r.post_best = lambda lv: None
    r.rec.level_name = "1-1"
    r.backup_root = str(tmp_path / "backups")
    save = tmp_path / "savegame.dat"
    save.write_bytes(b"original game save")
    r.find_save = lambda: (str(save), None)
    return r


def test_realtest_flushes_reward_then_saves_current_finish_once(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "a_plus_time", lambda lv: 1.0)
    r = runner(tmp_path)
    try:
        r.best.offer("1-1", "planned", {"seconds": .01})  # historical best must not replace current result
        r.rec.frame({"frame": 0, "dt": 16.667, "hz": []}, [])
        r.rec.agents(1, 0, [state(), state()], [True, True], [0, 0], [0, 0], [np.nan, np.nan])
        posted = r.script.post
        def post(message):
            if message["type"] == "finish":
                _, rows, _ = next(load_shards(r.wm.dir))
                assert rows[rows["status"] == 2]["reward"].tolist() == [15.0]
                assert (tmp_path / "backups").exists()
            posted(message)
        r.script.post = post
        r.on_message({"type": "send", "payload": {"ev": "wave_end", "wave": 1, "frame": 1,
                     "s": [state(2), state()], "status": [2, 0], "finish": [0, -1], "why": "quota"}}, None)
        assert [p for p in r.script.posts if p["type"] == "finish"] == [
            {"type": "finish", "payload": {"time": .017}}]
        assert r.script.posts[-1]["type"] == "go"  # release Frida's wave-end wait
        r.on_finish(0, 0, 40, 40)
        r.complete_realtest()
        assert len([p for p in r.script.posts if p["type"] == "finish"]) == 1
        backups = list((tmp_path / "backups").glob("*/savegame.dat"))
        assert len(backups) == 1 and backups[0].read_bytes() == b"original game save"
    finally:
        r.close()


@pytest.mark.parametrize("case", ["ordinary", "explore", "unknown", "unlocked", "disabled", "backup_failure", "stopping"])
def test_realtest_does_not_save_other_outcomes(tmp_path, monkeypatch, case):
    monkeypatch.setattr(R, "a_plus_time", lambda lv: None if case == "unknown" else 1.0)
    r = runner(tmp_path)
    try:
        if case == "unlocked": r.lock = False
        if case == "disabled": r.args.realtest = False
        if case == "stopping": r.stopping = True
        if case == "backup_failure":
            monkeypatch.setattr(agent, "backup_save", lambda *a: (_ for _ in ()).throw(OSError("read-only")))
        r.on_finish(1 if case == "explore" else 0, 100 if case == "ordinary" else 0, 40, 40)
        r.complete_realtest()
        assert not [p for p in r.script.posts if p["type"] == "finish"]
    finally:
        r.close()


@pytest.mark.parametrize("extra", [["--eval-first", "1-1"], ["--human"], ["--search"],
                                  ["--demo", "x.json"], ["--demo-level", "1-1"],
                                  ["--no-lock-step"], ["--agents", "0"]])
def test_realtest_rejects_incompatible_modes(extra):
    args = agent.build_parser().parse_args(["--realtest", *extra])
    with pytest.raises(SystemExit, match="realtest"):
        agent.validate_realtest_args(args)


def test_online_trains_bonus_targets_with_aplus_priority_and_fixed_replay_share(monkeypatch):
    import torch
    import wm.online as W
    from test_online import toy_parts
    monkeypatch.setattr(W, "MAX_STEPS", 12)

    class RewardLearner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.reward = torch.nn.Parameter(torch.zeros(()))
            self.seen = []

        def loss(self, X, yc, yb, ys, yr, wc):
            self.seen.append(yr.numpy().copy())
            return ((self.reward - yr) ** 2).mean(), None

    own = toy_parts(100, 0)
    own["yr"][:50], own["yr"][50:] = 3, 1
    own["sample_weight"] = np.where(own["yr"] == 3, 2.0, 1.0)
    replay = dict(toy_parts(100, 1), val=np.zeros(100, bool))
    model = RewardLearner()
    W.fine_tune(model, own, replay, np.random.default_rng(17), old=own)
    seen = np.stack(model.seen)
    assert np.all((seen == 0).sum(1) == W.BATCH - round(W.BATCH * (1 - W.REPLAY_SHARE)))
    ratio = (seen == 3).sum() / (seen == 1).sum()
    assert 1.6 < ratio < 2.4
    assert model.reward.item() > 0                  # the bonus targets reach an actual gradient update
