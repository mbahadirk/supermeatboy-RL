"""Checks for the goal / key contact reach (contact_reach.py) and Level.goal_boxes / key_boxes. No game."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import contact_reach as C                                   # noqa: E402
from wm.features import box_distances                       # noqa: E402


def test_inside_is_safety_convention():
    """inside(reach) == centre within BODY of the box (x, y, reach - BODY), plan/safety.py's contact rule."""
    rng = np.random.default_rng(0)
    p = rng.uniform(-25, 25, (4000, 2))
    rx, ry = 12.0, 11.5
    d = box_distances(p[:, 0], p[:, 1], [(0.0, 0.0, rx - C.BODY, ry - C.BODY, 0.0)])[:, 0]
    assert (C.inside(p[:, 0], p[:, 1], rx, ry) == (d < C.BODY)).all()


def _free(pts, v=(200.0, 0.0)):
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    return np.concatenate([pts, np.tile(v, (len(pts), 1))], 1)


def test_choose_keeps_resting_agents_outside():
    rng = np.random.default_rng(1)
    touch = np.stack([rng.uniform(-10, 10, 200), rng.uniform(0, 10, 200)], 1)
    touch[0] = (13.0, 0.0)                                   # one far touch (moving)
    rest = _free([[12.5, 0.3]] * 5, v=(0.0, 0.0))           # an agent standing 12.5 units off, never touching
    moving = _free(rng.uniform(15, 30, (100, 2)))
    pre = np.array([[12.8, 0.0]])
    row = C.choose(touch, np.concatenate([rest, moving]), pre)
    assert row["percentile"] != 100 and row["reach_x"] < 12.5     # the largest offset would hold the standing one
    c100 = row["candidates"][0]
    assert c100["percentile"] == 100 and c100["rest_inside"] == 5 and c100["pre_inside"] == 1
    assert row["half_w"] == pytest.approx(row["reach_x"] - C.BODY)


def test_write_refuses_an_incomplete_table(tmp_path):
    path = str(tmp_path / "contact_reach.json")
    with open(path, "w") as f:
        f.write('{"old": 1}')
    for bad in ({}, {"goal": None}, {"goal": {"reach_x": 10.0, "reach_y": None, "half_w": 4.0, "half_h": 4.0,
                                              "touches": 100}},
                {"goal": {"reach_x": 10.0, "reach_y": 10.0, "half_w": 4.0, "half_h": 4.0, "touches": 3}}):
        with pytest.raises(ValueError):
            C.write(bad, [], path)
        assert open(path).read() == '{"old": 1}'
    good = {"goal": {"reach_x": 10.0, "reach_y": 9.0, "half_w": 4.0, "half_h": 3.0, "touches": 100}, "key": None}
    out = C.write(good, [], path)
    assert out["key"]["measured"] is False and out["key"]["reach_x"] < C.KEY_R
    # the fallback key box lies inside realgame.js's pickup circle
    h = out["key"]["half_w"]
    assert h * np.sqrt(2) + C.BODY <= C.KEY_R + 1e-6


def test_table_and_level_boxes():
    """The committed table: a measured goal row from enough touches; Level boxes in the safety convention."""
    with open(C.TABLE, encoding="utf-8") as f:
        t = json.load(f)
    g = t["goal"]
    assert g["measured"] and g["touches"] >= C.MIN_TOUCHES
    chosen = [c for c in g["candidates"] if c["percentile"] == g["percentile"]][0]
    assert chosen["rest_inside"] == 0                        # nobody at rest without a touch inside the box
    from game.level import Level
    w = np.zeros((10, 20), np.uint32)
    w[0, :] = 1
    info = {"name": "test-contact", "W": 20, "H": 10, "ox": 0.0, "oy": 0.0, "spawn": [30.0, 26.0],
            "goals": [[300.0, 26.0]], "objs": [[0, 150.0, 26.0, 0x40]], "lethalTypes": []}
    lv = Level(info, w.astype("<u4").tobytes())
    (gb,) = lv.goal_boxes()
    assert gb[:2] == (300.0, 26.0) and gb[2] == pytest.approx(g["reach_x"] - C.BODY)
    assert gb[3] == pytest.approx(g["reach_y"] - C.BODY)
    assert lv.keys == [(150.0, 26.0)]
    (kb,) = lv.key_boxes()
    assert kb[:2] == (150.0, 26.0) and kb[2] == pytest.approx(t["key"]["reach_x"] - C.BODY)
