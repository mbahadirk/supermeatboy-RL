"""The versioned physics prior (wm.protocol prior protocols, wm.features.prior / fan_force): old checkpoints'
gravity-v1 prior byte-identical on recorded data, the measured fan wind law on synthetic and recorded cases,
the prior selected from the checkpoint in training, rollouts, the planner and the frozen-experiment evaluator.
Offline only: no game, training run on real data or live agent."""
import hashlib
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wm import features as F                                                       # noqa: E402
from wm.buffer import ROW                                                          # noqa: E402
from wm.model import WorldModel, N_CONT                                            # noqa: E402
from wm.protocol import (FAN_WIND_PRIOR, GRAVITY_PRIOR, KEY_TILES_PROTOCOL, PRIOR_PROTOCOL,   # noqa: E402
                         prior_protocol)

DT = 16.667
FAN = [0.0, 0.0, 0.0, 1.0, 40.0, 350.0, 370.0]        # at the origin, blowing up, half width 40, range 350
DATA_2_10 = os.path.join(HERE, "runs", "collect_2-10_explore_01", "wm_data")
DATA_1_3 = os.path.join(HERE, "runs", "demo_1-3_20261108-171442", "wm_data")


def rows(x, y, vx=0.0, vy=0.0, air=True):
    s = np.zeros(len(x), ROW)
    s["x"], s["y"], s["vx"], s["vy"] = x, y, vx, vy
    s["flags"] = F.AIR if air else 0
    return s


# ---- protocol ---------------------------------------------------------------------------------------------
def test_prior_protocol_is_recorded_never_guessed():
    assert PRIOR_PROTOCOL == GRAVITY_PRIOR                     # fresh training's default is unchanged
    assert prior_protocol({}) == prior_protocol(None) == GRAVITY_PRIOR    # every checkpoint before the marker
    assert prior_protocol({"prior_protocol": FAN_WIND_PRIOR}) == FAN_WIND_PRIOR
    for bad in ("gravity-fan-wind-v9", None, ""):
        with pytest.raises(ValueError, match="prior protocol"):
            prior_protocol({"prior_protocol": bad})
    with pytest.raises(ValueError, match="prior protocol"):
        F.prior(rows([0], [0]), DT, 1, [], "gravity-v0")
    with pytest.raises(ValueError, match="fans"):
        F.prior(rows([0], [0]), DT, 1, None, FAN_WIND_PRIOR)   # the wind prior never assumes "no fans"
    with pytest.raises(ValueError, match="prior protocol"):
        F.build([], prior_protocol="gravity-v0")


# ---- old prior: byte-identical ------------------------------------------------------------------------------
@pytest.mark.parametrize("data, step, rows_, prior_digest, yc_digest", [
    (DATA_2_10, 4, 5474, "784194d0f96d0b767899184ee2997fd2a55650885e347a03087bfed21496698a",
     "e47027296ebbe93256224046a35a449cb72af501d128ae809ae8d1b12b287117"),
    (DATA_1_3, 1, 7128, "a41b278c2fc3137d93b3b5d7793b67c3765d9d819a6164bdf11f865259edf242",
     "c94ae06393c79950c38d1986924fc5270a8dd449bf04ad88319af2a2c6b696b6"),
    (DATA_1_3, 4, 7056, "3d3a04ada056d890d824dcf5d480a04925758ebcc6992d33a21fcf5543f82bc8",
     "16d88641fc6bea926d8fdc814b7526f5da1fd585504a7d4ddf4ae7290c81c25b"),
])
def test_gravity_prior_and_targets_are_byte_identical_on_recorded_data(data, step, rows_, prior_digest, yc_digest):
    """prior / yc of the default (gravity-v1) prior, hashed before prior protocols existed."""
    if not os.path.isdir(data):
        pytest.skip("recorded data not present")
    for d in (F.build([data], step=step, log=lambda *a: None),
              F.build([data], step=step, log=lambda *a: None, prior_protocol=GRAVITY_PRIOR)):
        assert len(d["X"]) == rows_
        assert hashlib.sha256(np.ascontiguousarray(d["prior"]).tobytes()).hexdigest() == prior_digest
        assert hashlib.sha256(np.ascontiguousarray(d["yc"]).tobytes()).hexdigest() == yc_digest


# ---- the measured law ---------------------------------------------------------------------------------------
def test_fan_force_column_strength_and_range():
    x = np.array([0.0, 0.0, 40.0, 40.01, -30.0, 40.0, 0.0, 0.0, 0.0])
    y = np.array([175.0, 100.0, 0.0, 100.0, 345.0, 349.0, 351.0, -1.0, 175.01])
    fx, fy = F.fan_force(x, y, [FAN])
    assert np.all(fx == 0)
    assert fy[0] == pytest.approx(925.0)                         # c = 0.5: the near half's x5
    assert fy[1] == pytest.approx(370 * 5 * (1 - 100 / 350))
    assert fy[2] == pytest.approx(370 * 5 * (1 - 40 / 350))      # strength by distance from the fan point
    assert fy[3] == 0                                            # |lateral| > half width (no +6 band)
    assert fy[4] == pytest.approx(370 * (1 - np.hypot(345, 30) / 350))
    assert fy[5] == 0 and fy[6] == 0 and fy[7] == 0              # beyond the range (distance) / behind the fan
    assert fy[8] == pytest.approx(370 * (1 - 175.01 / 350))      # the far half: no x5
    # the input formula is frozen with the feature protocols and differs from the force (the +6 band)
    assert F.wind(x[3:4], y[3:4], [FAN])[1][0] > 0
    # the force is along the fan's axis (its form; sideways fans were never recorded)
    side = [0.0, 0.0, 1.0, 0.0, 40.0, 350.0, 370.0]
    sx, sy = F.fan_force(y[:3], -x[:3], [side])
    assert np.allclose(sx, fy[:3]) and np.all(sy == 0)


def test_fan_wind_prior_per_frame_airborne_only():
    h = DT / 1000.0
    s = np.concatenate([rows([0.0], [100.0], 50.0, -100.0), rows([0.0], [100.0], 50.0, -100.0, air=False),
                        rows([500.0], [100.0], 50.0, -100.0)])
    p = F.prior(s, DT, 1, [FAN], FAN_WIND_PRIOR)
    vy = -100.0 + (370 * 5 * (1 - 100 / 350) - F.GRAVITY) * h
    assert p[0] == pytest.approx([50.0 * h, vy * h, 50.0, vy], rel=1e-5)
    g = F.prior(s, DT, 1)
    assert np.array_equal(p[1], g[1])                            # grounded: never observed in wind; no force
    assert np.allclose(p[2], g[2], atol=1e-5)                    # outside the column: gravity only
    assert np.allclose(F.prior(s, DT, 1, [], FAN_WIND_PRIOR), g, atol=1e-5)
    # a step-k prior: k frames, each with the force at that frame's start position, velocity first
    s4 = rows([10.0], [160.0], 30.0, 20.0)
    x, y, vx, vy = 10.0, 160.0, 30.0, 20.0
    for _ in range(4):
        wx, wy = F.fan_force(np.array([x]), np.array([y]), [FAN])
        vx, vy = vx + wx[0] * h, vy + (wy[0] - F.GRAVITY) * h
        x, y = x + vx * h, y + vy * h
    assert F.prior(s4, 4 * DT, 4, [FAN], FAN_WIND_PRIOR)[0] == pytest.approx([x - 10.0, y - 160.0, vx, vy], rel=1e-5)


def test_fan_wind_prior_matches_recorded_2_10_velocities():
    """On the recorded 2-10 exploration (every recorded fan is vertical), airborne one-frame steps the wind
    prior changes get the recorded vertical velocity exactly (collisions aside); gravity-v1 does not."""
    if not os.path.isdir(DATA_2_10):
        pytest.skip("recorded data not present")
    g = F.build([DATA_2_10], log=lambda *a: None)
    w = F.build([DATA_2_10], log=lambda *a: None, prior_protocol=FAN_WIND_PRIOR)
    assert np.array_equal(g["raw"], w["raw"]) and np.array_equal(g["X"], w["X"])
    m = np.any(g["prior"] != w["prior"], 1) & (g["group"] == 1) & (g["yb"][:, F.EVENTS["death"]] == 0)
    assert m.sum() > 1000
    assert np.mean(np.abs(w["raw"][m, 3] - w["prior"][m, 3]) < 0.5) > 0.99
    assert np.mean(np.abs(g["raw"][m, 3] - g["prior"][m, 3]) < 0.5) < 0.1


# ---- the checkpoint's prior in training, rollouts and the planner ------------------------------------------
def zero_model(prior=None, step=1):
    """A model whose corrections are zero (rollouts follow the prior) and who stays airborne."""
    m = WorldModel(F.layout(KEY_TILES_PROTOCOL).n_in(step), members=2, hidden=8, layers=1, step=step)
    with torch.no_grad():
        for mem in m.members:
            mem[-1].weight.zero_()
            mem[-1].bias.zero_()
            mem[-1].bias[2 * N_CONT + F.BIN_NAMES.index("airborne")] = 10.0
    m.info = {"feature_protocol": KEY_TILES_PROTOCOL}
    if prior is not None:
        m.info["prior_protocol"] = prior
    return m


@pytest.mark.parametrize("step", [1, 4])
def test_rollouts_and_planner_apply_the_checkpoint_prior(step):
    from wm.rollout import World, rollout, rollout_batched
    from plan.planner import imagine
    geom = F.LevelGeom(np.zeros((40, 40), np.uint32), {"ox": -400, "oy": -400})
    s0 = rows([0.0, 300.0], [20.0, 20.0], 0.0, 0.0)
    K = 8
    acts = np.zeros((2, K), np.int64)
    worlds = [World(fans=[FAN]) for _ in range(K)]
    out = {}
    for pp in (None, GRAVITY_PRIOR, FAN_WIND_PRIOR):
        m = zero_model(pp, step)
        out[pp] = rollout_batched(m, geom, s0, acts, np.full(K, DT), worlds)
        # the expected path: the prior chained step by step
        s = s0.copy()
        for k in range(K // step):
            p = F.prior(s, DT * step, step, [FAN], pp or GRAVITY_PRIOR)
            s["x"], s["y"], s["vx"], s["vy"] = s["x"] + p[:, 0], s["y"] + p[:, 1], p[:, 2], p[:, 3]
        assert np.allclose(out[pp]["y"][:, :, -1], s["y"][None], atol=1e-3)
        im = imagine(m, slice(0, 2), geom, s0, acts, DT, worlds)
        assert np.allclose(im["y"], out[pp]["y"], atol=1e-3) and np.allclose(im["x"], out[pp]["x"], atol=1e-3)
        if step == 1:
            assert np.allclose(rollout(m, geom, s0, acts, np.full(K, DT), worlds)["y"], out[pp]["y"], atol=1e-3)
    assert np.array_equal(out[None]["y"], out[GRAVITY_PRIOR]["y"])
    lifted = out[FAN_WIND_PRIOR]["y"][:, 0, -1] - out[GRAVITY_PRIOR]["y"][:, 0, -1]
    assert np.all(lifted > 1.0)                                  # in the fan: lifted
    assert np.allclose(out[FAN_WIND_PRIOR]["y"][:, 1], out[GRAVITY_PRIOR]["y"][:, 1], atol=1e-4)   # outside
    bad = zero_model("gravity-v9", step)
    with pytest.raises(ValueError, match="prior protocol"):
        rollout_batched(bad, geom, s0, acts, np.full(K, DT), worlds)


def test_training_records_the_opted_in_prior_and_init_keeps_it(tmp_path, monkeypatch):
    from test_key_features import recording
    from wm import train as T_
    data = recording(tmp_path / "data")
    monkeypatch.setattr(T_, "immediate", lambda: None)
    base = ["wm.train", str(data), "--legacy-split", "--epochs", "1", "--members", "1", "--hidden", "8",
            "--layers", "1", "--sample-frac", "1"]
    monkeypatch.setattr(sys, "argv", base + ["--out", str(tmp_path / "plain")])
    T_.main()
    assert WorldModel.load(tmp_path / "plain" / "wm_model.pt").info["prior_protocol"] == GRAVITY_PRIOR
    monkeypatch.setattr(sys, "argv", base + ["--out", str(tmp_path / "wind"), "--prior-protocol", FAN_WIND_PRIOR])
    T_.main()
    m = WorldModel.load(tmp_path / "wind" / "wm_model.pt")
    assert m.info["prior_protocol"] == FAN_WIND_PRIOR and F.model_prior(m) == FAN_WIND_PRIOR
    monkeypatch.setattr(sys, "argv", ["wm.train", str(data), "--legacy-split", "--init",
                                      str(tmp_path / "wind" / "wm_model.pt"), "--out", str(tmp_path / "x"),
                                      "--prior-protocol", GRAVITY_PRIOR])
    with pytest.raises(ValueError, match="differs"):
        T_.main()


# ---- the frozen-experiment evaluator --------------------------------------------------------------------------
def test_experiment_rejects_a_prior_mismatch_and_uses_the_rebuilt_prior(tmp_path, monkeypatch):
    from test_experiment import recording, checkpoint
    from wm import experiment as E
    from wm.dataset import Dataset
    source = recording(tmp_path / E.HISTORICAL_RUN / "wm_data", "2-10")
    old = recording(tmp_path / "base" / "wm_data", "1-1")
    ds = Dataset.create([str(source), str(old)], 2)
    ds.save(tmp_path / "manifest.json")
    d = F.build(ds.sources, dataset=ds, step=4, frac=1, max_samples=0, provenance=True, log=lambda *a: None)
    r = E.prepare_roster(ds, d, historical_source=str(source))
    p = checkpoint(tmp_path, ds)
    wind = WorldModel.load(p)
    wind.save(tmp_path / "wind.pt", dict(wind.info, prior_protocol=FAN_WIND_PRIOR))
    with pytest.raises(ValueError, match="prior"):
        E.evaluate(ds, d, r, tmp_path / "wind.pt")
    rebuilt = E.checkpoint_prior(ds, d, FAN_WIND_PRIOR)
    with pytest.raises(ValueError, match="prior"):
        E.evaluate(ds, d, r, p, prior=rebuilt)                  # the cache's own prior: no replacement
    with pytest.raises(ValueError, match="prior"):
        E.evaluate(ds, d, r, tmp_path / "wind.pt", prior=dict(rebuilt, protocol=GRAVITY_PRIOR))
    res = E.evaluate(ds, d, r, tmp_path / "wind.pt", prior=rebuilt, out=tmp_path / "wind.json")
    assert res["prior"] == {"protocol": FAN_WIND_PRIOR, "cache_protocol": GRAVITY_PRIOR,
                            "values_sha256": hashlib.sha256(np.ascontiguousarray(rebuilt["values"])).hexdigest()}
    plain = E.evaluate(ds, d, r, p)
    assert plain["prior"]["protocol"] == GRAVITY_PRIOR
    changed = dict(d, raw=d["raw"] + 1)
    with pytest.raises(ValueError, match="differs"):
        E.checkpoint_prior(ds, changed, FAN_WIND_PRIOR)
