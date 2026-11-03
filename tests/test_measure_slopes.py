"""plan/measure_slopes.py on a synthetic recording: a full block, a half block and a slope, an agent sliding
on each and pressing jump once, and one agent in free fall."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from plan import measure_slopes as M       # noqa: E402
from wm.buffer import ROW                   # noqa: E402

HALF, SLOPE = 1 << 18, 2 << 18


def _level():
    w = np.zeros((10, 10), np.uint32)
    w[5, 2] = 1                              # full block: x 40..60, y 100..120
    w[5, 6] = 1 | HALF | (3 << 12)           # half block, solid right half: x 130..140
    w[2, 2] = 1 | SLOPE                      # slope, open corner bottom-right (an overhang facing down-right)
    return w


def _rows():
    spots = [(66.0, 110.0), (146.0, 110.0), (66.0, 59.5)]   # right of each, touching it (body half width 6)
    out = []
    for a, (x, y) in enumerate(spots):
        for f in range(6):
            press = f == 3
            vy, vx = (100.0, 50.0) if f == 4 else (-60.0, 0.0)
            out.append((0, f, a, x, y, vx, vy, 0, 9, 0, 0, 0, 5 if press else 0, 0.0))
    for f in range(6):                                          # free fall, far from everything
        out.append((0, f, 3, 170.0, 170.0 - 3 * f, 0.0, -150.0, 0, 7, 0, 0, 0, 0, 0.0))
    return np.array(out, ROW)


def test_slope_measurement_classes(tmp_path):
    d = tmp_path / "runs" / "real_x" / "wm_data"
    os.makedirs(d / "levels")
    np.savez_compressed(d / "levels" / "T-1.npz", words=_level(),
                        meta=json.dumps({"name": "T-1", "W": 10, "H": 10, "ox": 0.0, "oy": 0.0}))
    np.savez_compressed(d / "wave_00000.npz", rows=_rows(), meta=json.dumps({"level": "T-1", "wave": 1}))
    os.makedirs(tmp_path / "runs" / "fake_wm" / "wm_data")                  # skipped
    r = M.measure(str(tmp_path / "runs"), game_levels=str(tmp_path / "none"))
    assert r["shards"] == 1 and r["slide_state"] == 9
    c = r["classes"]
    for name in ("full", "half outer face", "slope diagonal down"):
        t = c[name]
        assert t["frames"] == 6 and t["slide_state_share"] == 1.0, name
        assert t["jump_presses"] == 1 and t["wall_jumps"] == 1 and t["slide_wall_jump_share"] == 1.0, name
        assert t["vy_desc_mean"] == -60.0
    ff = c["free fall (no support)"]
    assert ff["frames"] == 6 and ff["top_states"] == [[7, 1.0]] and ff["jump_presses"] == 0
    assert r["levels"]["T-1"] == {"slope_cells": 1, "half_cells": 1, "slope_contact_frames": 6,
                                  "half_contact_frames": 6}
    assert not r["enough_data"]["slope diagonal down"] and r["game_levels_with_slopes"] is None
    assert "too few recorded frames" in M.report(r)
