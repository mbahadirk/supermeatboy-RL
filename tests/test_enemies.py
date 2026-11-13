"""Enemies (realgame.js enemyRows: the game's SMBAnimal crawlers and fliers) arrive as extra hazard rows with lethal
type ENEMY_TYPE + kind and id "enemy<address>": kept (not decoration), a circle while they can kill, nothing while
they cannot."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from game.level import Level, box_ids, hazard_boxes                       # noqa: E402

ENEMY_TYPE = 1000


def _level(types, ids):
    H, W = 10, 20
    w = np.zeros((H, W), np.uint32)
    w[0, :] = 1
    info = {"name": "test-enemies", "W": W, "H": H, "ox": 0.0, "oy": 0.0, "spawn": [30.0, 30.0], "goals": [[370.0, 30.0]],
            "objs": [], "lethalTypes": types, "lethalIds": ids}
    return Level(info, w.astype("<u4").tobytes())


def test_enemy_rows_are_kept_circles_while_they_can_kill():
    saw = [100.0, 50.0, 20.0, 20.0, 1.0, 1.0, 0.0, [[1, 100.0, 50.0, 10.0, 10.0, 0.0]]]
    flier = [200.0, 80.0, 20.0, 20.0, 1, 1, 0, [[1, 200.0, 80.0, 10.0, 10.0, 0]]]
    gone = [0.0, 0.0, 20.0, 20.0, 1, 1, 0, []]
    lv = _level([9, ENEMY_TYPE + 12, ENEMY_TYPE + 8], ["0x1", "enemy0x2", "enemy0x3"])
    assert lv.hz_keep == [True, True, True]
    boxes, kept = hazard_boxes([saw, flier, gone], lv.hz_keep)
    assert kept == [0, 1] and boxes[1] == (200.0, 80.0, 10.0, 10.0, 0.0)
    assert box_ids(kept, lv.hz_ids) == ["0x1", "enemy0x2"]
