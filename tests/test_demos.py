"""Demo replay (plan/demos.py, agent.py --demo): action schedule, seeded and bounded perturbations, refusals, and
one fake-game wave through agent.Runner (no game, no model)."""
import glob
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from mpc.candidates import ACTIONS                      # noqa: E402
from plan import demos as D                             # noqa: E402

DATA = os.path.join(HERE, "runs", "real_20261102-165351", "wm_data")
LEVEL = "1-13"
need_data = pytest.mark.skipif(not os.path.isdir(DATA), reason="recorded data not here")
# a demo-like sequence: run right, jumps held for a while, a pause
ACTS = ([6] * 20 + [7] * 15 + [6] * 10 + [9] * 8 + [0] * 5 + [7] * 30 + [3] * 6 + [6] * 26)


def _entries(tmp_path, level="1-4"):
    out = []
    for name, sec, verdict, lv in [("a", 3.0, "VALID_RECORDED", level), ("b", 2.5, "REPLAYABLE", level),
                                   ("c", 1.0, "SUSPICIOUS", level), ("d", 4.0, "HELD_OUT", "1-18")]:
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({"level": lv, "seconds": sec, "agent": 0, "wave": 1, "actions": ACTS}))
        out.append({"level": lv, "best": str(p), "seconds": sec, "verdict": verdict, "rows": False,
                    "reasons": ["finish row away from the goal box"] if verdict == "SUSPICIOUS" else []})
    return out


def test_schedule_plays_the_demo_then_waits():
    play = D.DemoPlay({"actions": ACTS, "best": "x", "level": "1-4"}, copies=3, seed=1)
    assert play.n == 4 and len(play.seqs) == 4
    assert [play.action(0, f) for f in range(len(ACTS))] == ACTS
    assert play.action(0, len(ACTS)) == 0 and play.action(2, 10 ** 6) == 0 and play.action(9, 0) == 0
    assert play.finish_frame == len(ACTS) - 1
    assert play.check(len(ACTS) - 1)[0] and play.check(len(ACTS))[0] and play.check(len(ACTS) - 2)[0]
    assert not play.check(len(ACTS) + 1)[0] and not play.check(None)[0]


def test_perturbations_are_seeded_and_bounded():
    _, off, has, n_act = D._jump_ids()
    seen = set()
    for seed in range(200):
        a1, c1 = D.perturb(ACTS, np.random.default_rng([seed, 0, 1]))
        a2, c2 = D.perturb(ACTS, np.random.default_rng([seed, 0, 1]))
        assert a1 == a2 and c1 == c2                                    # seeded
        assert len(a1) == len(ACTS) and 1 <= len(c1) <= 3
        assert all(0 <= x < n_act for x in a1)
        assert sum(x != y for x, y in zip(a1, ACTS)) <= 3 * 12          # small
        for c in c1:
            seen.add(c["kind"])
            assert 0 <= c["frame"] < len(ACTS)
            if c["kind"] == "shift":
                assert 1 <= abs(c["by"]) <= 4
            elif c["kind"].startswith("jump"):
                assert 1 <= c["by"] <= 4
            else:
                assert 4 <= c["frames"] <= 12
        if len(c1) == 1 and c1[0]["kind"] == "jump early":              # only the jump key changed
            diff = [i for i, (x, y) in enumerate(zip(a1, ACTS)) if x != y]
            assert all(not has[a1[i]] and off[ACTS[i]] == a1[i] for i in diff)
            assert 1 <= len(diff) <= 4
    assert seen == {"shift", "jump early", "jump late", "random"}
    play = D.DemoPlay({"actions": ACTS, "best": "x", "level": "1-4"}, copies=2, seed=5)
    first = [list(s) for s in play.seqs]
    play.new_wave()
    assert play.seqs == first                                           # same wave: same copies
    play.wave_done()
    play.new_wave()
    assert play.seqs[0] == ACTS and play.seqs[1:] != first[1:]          # next wave: new copies, exact one kept


def test_refusals_and_level_pick(tmp_path):
    es = _entries(tmp_path)
    assert D.resolve_demo(level="1-4", entries=es)["seconds"] == 2.5    # fastest non-suspicious
    assert D.resolve_demo(best=es[0]["best"], entries=es)["actions"] == ACTS
    with pytest.raises(SystemExit, match="SUSPICIOUS"):
        D.resolve_demo(best=es[2]["best"], entries=es)
    with pytest.raises(SystemExit, match="held-out"):
        D.resolve_demo(best=es[3]["best"], entries=es)
    with pytest.raises(SystemExit, match="held-out"):
        D.resolve_demo(level="1-18", entries=es)
    with pytest.raises(SystemExit, match="not in the demo inventory"):
        D.resolve_demo(best=str(tmp_path / "other.json"), entries=es)
    with pytest.raises(SystemExit, match="no non-suspicious demo"):
        D.resolve_demo(level="2-1", entries=es)


def _runner(tmp_path, entries, copies=2):
    import agent
    args = agent.build_parser().parse_args(
        ["--demo", entries[0]["best"], "--demo-copies", str(copies), "--name", str(tmp_path / "run"), "--wm-collect",
         "--death-memory", str(tmp_path / "deaths.json"), "--move-memory", str(tmp_path / "moves.json"),
         "--telemetry", "off", "--best-times", str(tmp_path / "best_times.json"), "--seed", "3"])
    agent.setup_demo(args, entries=entries)
    assert args.end_after == 0 and args.wave_seconds >= 3.0 + 2.0
    r = agent.Runner(args)

    class _Script:
        posts = []

        def post(self, msg):
            self.posts.append(msg)
    r.script = _Script()
    return r


@need_data
def test_runner_replays_collects_and_checks(tmp_path):
    from wm.buffer import load_level
    es = _entries(tmp_path, level=LEVEL)
    r = _runner(tmp_path, es)
    assert r.n_plan == 0 and r.model is None and r.n_total == 3
    words, meta = load_level(DATA, LEVEL)
    r.on_message({"type": "send", "payload": dict(meta, name=LEVEL, agents=3, ev="info")}, words.astype("<u4").tobytes())
    assert r.script.posts[-1] == {"type": "go", "payload": {"agents": 3}} and not r.stopping
    sx, sy = meta["spawn"]
    n = len(ACTS)
    for f in range(n):
        st = [[sx + f, sy, 0.0, 0.0, 0, 0, 0, 0] for _ in range(3)]
        r.on_message({"type": "send", "payload": {"ev": "frame", "frame": f, "wave": 1, "dt": 16.667, "s": st}}, None)
        act = [p for p in r.script.posts if p["type"] == "act"][-1]["payload"]["a"]
        assert [ACTIONS.index(tuple(a)) for a in act] == [r.demo.seqs[i][f] for i in range(3)]
    assert act == [list(ACTIONS[r.demo.seqs[i][n - 1]]) for i in range(3)]
    paths = [p for p in r.script.posts if p["type"] == "path"]
    assert paths and all(seg["col"] == 0xE0FFFFFF for p in paths for seg in p["payload"])   # white trail
    copies = [list(s) for s in r.demo.seqs]
    r.on_message({"type": "send", "payload": {"ev": "wave_end", "wave": 1, "status": [2, 1, 1],
                                              "finish": [n - 1, -1, -1], "frame": n, "why": "done",
                                              "s": [[sx + n, sy, 0, 0, 0, 0, 2, 0]] * 3}}, None)
    r.flush()
    ev = [json.loads(x) for x in open(os.path.join(r.run_dir, "events.jsonl"), encoding="utf-8")]
    chk = [e for e in ev if e["kind"] == "demo check"]
    assert len(chk) == 1 and chk[0]["ok"] and chk[0]["finish_frame"] == n - 1 and len(chk[0]["perturbations"]) == 3
    shard = sorted(glob.glob(os.path.join(r.run_dir, "wm_data", "wave_*.npz")))
    m = json.loads(str(np.load(shard[0])["meta"]))
    assert m["source"] == "demo+demo-noisy" and m["agent_sources"] == {"demo": [0], "demo-noisy": [1, 2]}
    assert m["perturbations"] == chk[0]["perturbations"] and m["demo"] == es[0]["best"]
    assert r.demo.seqs[0] == ACTS and r.demo.seqs[1:] != copies[1:]   # the next wave has new copies
    assert not os.path.exists(tmp_path / "best_times.json")              # demos are not offered as best times
    # a finish elsewhere warns; another level is refused
    r.on_message({"type": "send", "payload": {"ev": "wave_end", "wave": 2, "status": [1, 1, 1],
                                              "finish": [-1, -1, -1], "frame": 30, "why": "done"}}, None)
    r.flush()
    ev = [json.loads(x) for x in open(os.path.join(r.run_dir, "events.jsonl"), encoding="utf-8")]
    assert [e["ok"] for e in ev if e["kind"] == "demo check"] == [True, False]
    r.on_message({"type": "send", "payload": dict(meta, name="1-12", agents=3, ev="info")}, words.astype("<u4").tobytes())
    assert r.stopping and r.level is None
    r.close()
