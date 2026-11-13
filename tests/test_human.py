"""agent.py --human (plan/human.py): agent 0 plays the buttons read each frame; the recording has source "human"
and the actions actually sent; a finish is timed and judged against the level's A+ time. No game: the runner is
driven with a recorded 1-13 wave through a fake Frida script (tests/test_agent.py's harness)."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))

from mpc.candidates import ACTIONS                                         # noqa: E402
from plan import human as Hm                                               # noqa: E402
from test_agent import LEVEL, _FakeScript, _recorded, need_data           # noqa: E402


def test_buttons_map_onto_the_action_table():
    A = lambda *b: ACTIONS[Hm.to_action(*map(bool, b))]
    assert A(0, 0, 0, 0) == (0, 0, 0, 0) and A(1, 0, 0, 0) == (1, 0, 0, 0) and A(0, 1, 1, 1) == (0, 1, 1, 1)
    assert A(1, 1, 0, 0) == (0, 0, 0, 0) and A(1, 1, 1, 1) == (0, 0, 1, 0)     # both directions: neither
    assert A(0, 0, 0, 1) == (0, 0, 0, 0) and A(0, 0, 1, 1) == (0, 0, 1, 0)     # run needs a direction


@need_data
def test_human_plays_agent_0_and_is_recorded(tmp_path, capsys, monkeypatch):
    import agent
    seq = [(False, True, False, True)] * 5 + [(False, True, True, True)] * 3 + [(True, False, False, False)] * 4
    calls = iter(seq)
    monkeypatch.setattr(Hm.HumanInput, "buttons", lambda self: next(calls))
    args = agent.build_parser().parse_args(
        ["--human", "--wm-collect", "--name", str(tmp_path / "run"), "--telemetry", "off",
         "--death-memory", str(tmp_path / "d.json"), "--move-memory", "none", "--best-times", str(tmp_path / "b.json")])
    r = agent.Runner(args, telemetry=None)
    r.script = _FakeScript()
    assert r.n_total == 1 and r.n_plan == 0 and r.model is None and r.human is not None
    words, meta, rows, world = _recorded()
    a0 = sorted(set(rows["agent"].tolist()))[0]
    r.on_message({"type": "send", "payload": {"ev": "debug", "msg": "time step locked to the game's fixed step"}}, None)
    r.on_message({"type": "send", "payload": dict(meta, name=LEVEL, agents=1, ev="info")}, words.astype("<u4").tobytes())
    sent, last = [], None
    for k, w in enumerate(world[:len(seq)]):
        q = rows[(rows["agent"] == a0) & (rows["frame"] == w["f"])][0]
        last = [float(q["x"]), float(q["y"]), float(q["vx"]), float(q["vy"]), int(q["flags"]), int(q["state"]), 0,
                int(q["keys"])]
        msg = dict(w, ev="frame", frame=k, wave=1, s=[last], dt=16.667)
        msg.pop("f")
        n = len(r.script.posts)
        r.on_message({"type": "send", "payload": msg}, None)
        sent.append([p for p in r.script.posts[n:] if p["type"] == "act"][0]["payload"]["a"][0])
    assert [tuple(a) for a in sent] == [(0, 1, 0, 1)] * 5 + [(0, 1, 1, 1)] * 3 + [(1, 0, 0, 0)] * 4
    final = [last[:6] + [2, last[7]]]
    r.on_message({"type": "send", "payload": {"ev": "wave_end", "wave": 1, "status": [2], "finish": [len(seq) - 1],
                                              "frame": len(seq), "why": "quota", "s": final, "hz": [], "sh": []}}, None)
    r.close()
    out = capsys.readouterr().out
    assert f"[human] {LEVEL} wave 1: finished in 0.20 s (A+ 7 s: UNDER A+" in out
    run = tmp_path / "run"
    t = [json.loads(l) for l in open(run / "times.jsonl", encoding="utf-8")]
    assert [(x["source"], x["frames"]) for x in t] == [("human", 12)]
    assert not os.path.exists(tmp_path / "b.json") or LEVEL not in json.load(open(tmp_path / "b.json"))
    from wm.buffer import load_shards
    shards = list(load_shards(str(run / "wm_data")))
    assert len(shards) == 1
    m, rr, _ = shards[0]
    assert m["source"] == "human" and m["agent_sources"] == {"human": [0]} and m["level"] == LEVEL
    acts = [int(a) for a in rr[rr["frame"] < len(seq)]["action"]]
    assert [ACTIONS[a] for a in acts] == [tuple(a) for a in sent]
    ev = [json.loads(l) for l in open(run / "events.jsonl", encoding="utf-8")]
    fin = [e for e in ev if e["kind"] == "finish"]
    assert len(fin) == 1 and fin[0]["a_plus"] is True and fin[0]["source"] == "human"
    # the A+ extractor keeps it (locked, rows 0..12 without a gap, 0.2 s under 1-13's A+), once
    from plan import aplus
    out = tmp_path / "aplus.json"
    aplus.main([str(run), str(run), "--out", str(out)])
    recs = json.load(open(out, encoding="utf-8"))
    assert [(x["level"], x["agent"], x["frames"], x["source"], x["seconds"]) for x in recs] == \
        [(LEVEL, 0, [0, len(seq)], "human", 0.2)]                    # the same folder twice: the second a duplicate
    assert "duplicate" in capsys.readouterr().out
