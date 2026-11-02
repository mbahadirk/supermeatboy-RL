"""Checks for the MPC pieces on recorded data (no game needed):  python -m pytest smbrl/tests -q"""
import glob
import gzip
import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from wm.buffer import load_level                       # noqa: E402
from wm.features import LevelGeom                      # noqa: E402
from wm.model import WorldModel                        # noqa: E402
from wm.rollout import rollout, rollout_batched, World  # noqa: E402

DATA = os.path.join(HERE, "runs", "real_20261102-165351", "wm_data")
MODEL = os.path.join(HERE, "runs", "wm_phase4mix", "wm_model.pt")
LEVEL = "1-13"                                          # moving hazards
need_data = pytest.mark.skipif(not (os.path.isdir(DATA) and os.path.exists(MODEL)), reason="recorded data not here")


def one_wave(level=LEVEL):
    for p in sorted(glob.glob(os.path.join(DATA, "wave_*.npz"))):
        z = np.load(p)
        if json.loads(str(z["meta"])).get("level") == level:
            world = json.load(gzip.open(p[:-4] + "_world.json.gz", "rt", encoding="utf-8"))
            return z["rows"].copy(), world
    raise FileNotFoundError(level)


@need_data
def test_batched_rollout_matches_rollout():
    torch.manual_seed(0)
    model = WorldModel.load(MODEL).eval()
    geom = LevelGeom(*load_level(DATA, LEVEL))
    rows, world = one_wave()
    wf = {w["f"]: w for w in world}
    f0 = sorted(wf)[80]
    s0 = rows[(rows["frame"] == f0) & (rows["status"] == 0)][:6]
    H = 12
    acts = np.random.default_rng(0).integers(0, 10, (len(s0), H))
    worlds = [World.from_record(wf[f0 + k], wf.get(f0 + k - 1)) for k in range(H)]
    a = rollout(model, geom, s0, acts, np.full(H, 16.7), worlds)
    b = rollout_batched(model, geom, s0, acts, np.full(H, 16.7), worlds, keep_rows=True)
    assert b["steps"] == H
    for k in ("x", "y", "vx", "vy", "alive", "airborne", "p", "std"):
        np.testing.assert_allclose(a[k], b[k], rtol=1e-4, atol=1e-3, err_msg=k)
    assert b["rows"].shape == (len(model.members), len(s0), H + 1)


@need_data
def test_batched_rollout_stops_at_deadline():
    model = WorldModel.load(MODEL).eval()
    geom = LevelGeom(*load_level(DATA, LEVEL))
    rows, world = one_wave()
    s0 = rows[rows["status"] == 0][:2]
    H = 20
    out = rollout_batched(model, geom, s0, np.zeros((2, H), np.int64), np.full(H, 16.7), [World()] * H, deadline=0.0)
    assert out["steps"] == 1                   # one step, then the deadline (already past) stops it


@need_data
def test_route_values_match_value():
    from realpath import Planner
    words, meta = load_level(DATA, LEVEL)
    pl = Planner(words, meta["ox"], meta["oy"], tuple(meta["spawn"]), [tuple(g) for g in meta["goals"]], [], [],
                 [tuple(k) for k in meta["keys"]], [tuple(b) for b in meta["bandages"]], [])
    routes = pl.find(1)
    assert routes
    r = routes[0]
    rng = np.random.default_rng(1)
    pts = r.points[rng.integers(0, len(r.points), 300)] + rng.normal(0, 40, (300, 2))
    vec = r.values(pts[:, 0], pts[:, 1], r.leg(()))
    one = [r.value(x, y, ()) for x, y in pts]
    for v, o in zip(vec, one):
        if o is None:
            assert np.isnan(v)
        else:
            assert abs(v - o) < 1e-3
