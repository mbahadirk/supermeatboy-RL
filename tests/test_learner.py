"""Train mode (wm/learner.py, agent.py --learn) with fakes, no game: self-demos on new bests only, the learner
publishes only models that are not worse, and the agent loads a published model only between waves."""
import json
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wm import learner as L                             # noqa: E402

MODEL = os.path.join(HERE, "runs", "wm_locked", "wm_model.pt")
need_model = pytest.mark.skipif(not os.path.exists(MODEL), reason="trained model not here")


def _args(tmp_path, *extra):
    import agent
    return agent.build_parser().parse_args(
        ["--model", MODEL, "--name", str(tmp_path / "run"), "--wm-collect", "--telemetry", "off",
         "--death-memory", str(tmp_path / "deaths.json"), "--move-memory", str(tmp_path / "moves.json"),
         "--best-times", str(tmp_path / "best_times.json"), "--self-demos", str(tmp_path / "self.json"),
         "--live-dir", str(tmp_path / "live"), *extra])


@need_model
def test_self_demo_on_a_new_best_only(tmp_path):
    import agent
    r = agent.Runner(_args(tmp_path, "--agents", "1", "--explore", "1", "--learn"))
    r.post_best = lambda lv: None
    r.level, r.lock, r.protected = types.SimpleNamespace(name="1-12"), True, False
    sd = lambda: json.load(open(tmp_path / "self.json")) if os.path.exists(tmp_path / "self.json") else []
    r.on_finish(0, 100, 0.0, 0.0)                                  # a first best: recorded
    assert [(e["level"], e["shard"], e["agent"], e["frames"]) for e in sd()] == [("1-12", "wave_00000.npz", 0, [0, 101])]
    assert sd()[0]["wm_data"].endswith("run/wm_data") and sd()[0]["seconds"] == pytest.approx(1.683, abs=1e-3)
    r.on_finish(0, 120, 0.0, 0.0)                                  # slower: not a best
    r.on_finish(1, 50, 0.0, 0.0)                                   # an explore agent
    r.lock = False
    r.on_finish(0, 60, 0.0, 0.0)                                   # unlocked: no best at all
    r.lock, r.protected, r.level = True, True, types.SimpleNamespace(name="1-18")
    r.on_finish(0, 60, 0.0, 0.0)                                   # a held-out level
    assert len(sd()) == 1
    r.protected, r.level = False, types.SimpleNamespace(name="1-12")
    r.on_finish(0, 90, 0.0, 0.0)                                   # faster: recorded
    assert [e["frames"] for e in sd()] == [[0, 101], [0, 91]]
    r.close()
    r2 = agent.Runner(_args(tmp_path / "b", "--agents", "1"))      # without --learn: never
    r2.post_best = lambda lv: None
    r2.level, r2.lock, r2.protected = types.SimpleNamespace(name="1-13"), True, False
    r2.on_finish(0, 30, 0.0, 0.0)
    assert not os.path.exists(tmp_path / "b" / "self.json")
    r2.close()


def test_learner_publishes_only_when_not_worse(tmp_path, monkeypatch):
    from wm.model import WorldModel
    init = str(tmp_path / "init.pt")
    WorldModel(6, members=1, hidden=8, layers=1).save(init, {"data": []})
    out = tmp_path / "live"
    out.mkdir()
    errs = {"16": [4.0, 20.0, 100], "32": [10.0, 40.0, 90], "48": [20.0, 70.0, 80]}
    state = {"init": init, "base": init, "version": 0, "eval_waves": [["x", 0]], "errors": errs, "shards": {}}
    args = types.SimpleNamespace(out=str(out), init=init, demos=None, self_demos=None)
    seen = []
    monkeypatch.setattr(L, "fine_tune", lambda model, folders, demos, a: seen.append(folders) or 1000)
    nxt = iter([{"16": [3.5, 18.0, 100], "32": [11.0, 41.0, 90], "48": [20.0, 69.0, 80]},   # not worse: published
                {"16": [3.4, 17.0, 100], "32": [9.0, 30.0, 90], "48": [20.5, 60.0, 80]}])  # +48 worse: kept
    monkeypatch.setattr(L, "errors", lambda model, waves: next(nxt))
    s1 = L.cycle(args, state, ["a/wm_data"], {"a/wm_data": 25}, 0)
    pub = out / "wm_model.pt"
    assert s1["version"] == 1 and pub.exists() and s1["errors"]["16"][0] == 3.5 and s1["shards"] == {"a/wm_data": 25}
    info = WorldModel.load(str(pub)).info
    assert info["learner"]["version"] == 1 and info["learner"]["parent_errors"]["16"][0] == 4.0
    assert info["data"] == ["a/wm_data"] and not (out / "wm_model.pt.tmp").exists()
    mt = os.path.getmtime(pub)
    s2 = L.cycle(args, s1, ["a/wm_data"], {"a/wm_data": 50}, 1)
    assert s2["version"] == 1 and s2["errors"] == s1["errors"] and os.path.getmtime(pub) == mt
    assert s2["cycles"] == 2 and s2["self_demos"] == 1 and s2["base"].endswith("wm_model.pt")
    log = (out / "learner.log").read_text().splitlines()
    assert "published v1" in log[0] and "kept v1" in log[1]
    # when a cycle is due
    assert L.due({"shards": {"a": 10}, "self_demos": 1}, {"a": 29}, 1, 20) == (False, 19, 0)
    assert L.due({"shards": {"a": 10}, "self_demos": 1}, {"a": 30}, 1, 20)[0]
    assert L.due({"shards": {"a": 10}, "self_demos": 1}, {"a": 10, "b": 0}, 2, 20) == (True, 0, 1)
    assert L.not_worse(errs, errs) and not L.not_worse({"16": [None, None, 0], "48": [1, 1, 1]}, errs)


def test_add_self_demo_is_atomic_and_appends(tmp_path):
    p = str(tmp_path / "d" / "self.json")
    assert L.add_self_demo({"level": "1-1"}, p) == 1 and L.add_self_demo({"level": "1-2"}, p) == 2
    assert [e["level"] for e in json.load(open(p))] == ["1-1", "1-2"] and not os.path.exists(p + ".tmp")


@need_model
def test_agent_loads_a_published_model_between_waves_only(tmp_path, capsys):
    import agent
    from test_agent import _drive_wave, _FakeScript, need_data
    if need_data.args[0]:
        pytest.skip("recorded data not here")
    from wm.model import WorldModel
    r = agent.Runner(_args(tmp_path, "--agents", "2", "--replan", "2", "--budget-ms", "60", "--learn"))
    r.script = _FakeScript()
    first = r.model
    m = WorldModel.load(MODEL)
    os.makedirs(tmp_path / "live")
    m.save(str(tmp_path / "live" / "wm_model.pt"),
           {"learner": {"version": 3, "errors": {"16": [3.5, 1, 1]}, "parent_errors": {"16": [4.0, 1, 1]}}})
    os.utime(tmp_path / "live" / "wm_model.pt", (r.live_mtime + 10, r.live_mtime + 10))
    during = []
    on_frame = r.on_frame

    def spy(msg, data=None):
        during.append((r.model is first, r.planner is None or r.planner.model is first))
        return on_frame(msg, data)
    r.on_frame = spy
    _drive_wave(r)
    assert during and all(a and b for a, b in during)               # never mid-wave
    assert r.model is not first and r.planner.model is r.model       # at the wave end
    assert "[learn] model v3 loaded (+16 error 4.0 -> 3.5)" in capsys.readouterr().out
    loaded = r.model
    r.on_message({"type": "send", "payload": {"ev": "wave_end", "wave": 2, "status": [1, 1], "finish": [-1, -1],
                                              "frame": 3, "why": "done"}}, None)
    assert r.model is loaded                                         # not again: nothing newer
    r.close()
