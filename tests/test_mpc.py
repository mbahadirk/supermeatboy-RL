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


def test_candidate_actions_match_realgame():
    import realgame
    from mpc import candidates
    assert candidates.ACTIONS == [tuple(a) for a in realgame.ACTIONS]


class _FakeScript:
    def __init__(self):
        self.posts = []

    def post(self, msg):
        self.posts.append(msg)


def _args(run_dir, control):
    import argparse
    return argparse.Namespace(name=str(run_dir), load=None, agents=2, paths=1, wave_seconds=20.0, end_after=5,
                              batch=16384, test=False, debug_spawn=False, no_lock_step=False, wm_collect=True,
                              probe_char=0, no_monitor=True, control=control, mpc_model=MODEL, mpc_candidates=12,
                              mpc_horizon=8, mpc_every=1, mpc_budget_ms=0.0, mpc_agents=1, mpc_value=0.5,
                              mpc_kappa=1.0, mpc_seed=0)


def _drive(trainer, n_frames=40, name=LEVEL):
    """The trainer's message handler with a recorded 1-13 wave (2 agents) as the game (named `name`)."""
    words, meta = load_level(DATA, LEVEL)
    info = dict(meta, name=name, agents=2)
    rows, world = one_wave()
    agents = sorted(set(rows["agent"].tolist()))[:2]
    trainer.script = _FakeScript()
    trainer.on_message({"type": "send", "payload": dict(info, ev="info")}, words.astype("<u4").tobytes())
    replies = []
    for w in world[:n_frames]:
        f = w["f"]
        st = []
        for a in agents:
            r = rows[(rows["agent"] == a) & (rows["frame"] == f)]
            r = r[0] if len(r) else rows[rows["agent"] == a][-1]
            st.append([float(r["x"]), float(r["y"]), float(r["vx"]), float(r["vy"]), int(r["flags"]), int(r["state"]),
                       0, int(r["keys"])])
        msg = dict(w, ev="frame", frame=f, wave=0, s=st)
        msg.pop("f")
        trainer.on_message({"type": "send", "payload": msg}, None)
        replies.append(trainer.script.posts[-1]["payload"]["a"])
    trainer.on_message({"type": "send", "payload": {"ev": "wave_end", "wave": 0, "status": [1, 0], "finish": [-1, -1],
                                                    "frame": world[n_frames - 1]["f"], "why": "test"}}, None)
    return replies


@need_data
def test_trainer_ppo_only_unchanged(tmp_path):
    import realgame
    t = realgame.Trainer(_args(tmp_path / "ppo", "ppo"))
    assert t.mpc is None
    _drive(t)
    assert len(t.ppo.batch) == 2 and t.mpc_skipped == 0          # both agents' steps went to PPO
    assert (tmp_path / "ppo" / "wm_data").exists()


@need_data
def test_trainer_planner_kept_out_of_ppo(tmp_path):
    import json as js
    import realgame
    t = realgame.Trainer(_args(tmp_path / "mpc", "mpc"))
    assert t.mpc is not None
    _drive(t)
    t.mpc.close()
    assert len(t.ppo.batch) == 1                                  # agent 1 (PPO) trained, agent 0 (planner) not
    assert t.mpc_skipped > 0
    plans = [js.loads(l) for l in open(tmp_path / "mpc" / "mpc_plans.jsonl", encoding="utf-8")]
    assert len(plans) >= 30 and all(p["agent"] == 0 and p["steps"] == 8 for p in plans)
    shard = np.load(next((tmp_path / "mpc" / "wm_data_mpc").glob("wave_*.npz")))
    assert js.loads(str(shard["meta"]))["source"] == "mpc"


@need_data
def test_held_out_level_gives_no_data(tmp_path):
    import realgame
    from wm.buffer import eval_levels
    name = sorted(eval_levels())[0]
    t = realgame.Trainer(_args(tmp_path / "held", "ppo"))
    t.args.batch = 1                                              # would update PPO after this wave anywhere else
    _drive(t, name=name)
    assert t.protected and not t.updates
    assert not list((tmp_path / "held" / "wm_data").glob("wave_*.npz"))


@need_data
def test_hazard_filter_does_not_change_predictions():
    """Leaving far hazards out of the model's inputs (mpc.controller) gives the same rollout."""
    from dataclasses import replace
    from mpc.controller import _relevant, REACH, HAZ_FAR
    model = WorldModel.load(MODEL).eval()
    geom = LevelGeom(*load_level(DATA, LEVEL))
    rows, world = one_wave()
    wf = {w["f"]: w for w in world}
    f0 = sorted(wf)[200]
    s0 = rows[(rows["frame"] == f0) & (rows["status"] == 0)][:1]
    H = 16
    worlds = [World.from_record(wf[f0 + k], wf.get(f0 + k - 1)) for k in range(H)]
    assert max(len(w.boxes) for w in worlds) > 3
    reach = REACH * H * 16.7 / 1000.0 + HAZ_FAR
    few = [replace(w, boxes=w.boxes[_relevant(w.boxes, float(s0["x"][0]), float(s0["y"][0]), reach)]) for w in worlds]
    acts = np.random.default_rng(3).integers(0, 10, (24, H))
    s = np.repeat(s0, 24)
    a = rollout_batched(model, geom, s, acts, np.full(H, 16.7), worlds)
    b = rollout_batched(model, geom, s, acts, np.full(H, 16.7), few)
    np.testing.assert_allclose(a["x"], b["x"], atol=1e-3)
    np.testing.assert_allclose(a["alive"], b["alive"], atol=1e-4)


def test_box_distances_match_box_distance():
    from realpath import box_distance
    from wm.features import box_distances
    rng = np.random.default_rng(4)
    boxes = np.c_[rng.normal(0, 100, (7, 2)), rng.uniform(2, 40, (7, 2)), rng.uniform(-3, 3, 7)]
    x, y = rng.normal(0, 150, 500), rng.normal(0, 150, 500)
    one = np.stack([box_distance(x, y, b) for b in boxes], 1)
    np.testing.assert_allclose(box_distances(x, y, boxes), one, rtol=1e-12, atol=1e-9)


def _fake_out(E, alive_curves, K):
    n = len(alive_curves)
    out = {k: np.zeros((E, n, K + 1), np.float32) for k in ("x", "y", "vx", "vy", "airborne")}
    out["alive"] = np.repeat(np.array(alive_curves, np.float32)[None], E, 0)
    out["p"] = np.zeros((E, n, K, 12), np.float32)
    out["steps"] = K
    return out


def test_later_death_beats_sooner_death():
    """Every candidate dies (a model that does not know the level): the one that dies latest wins, and a
    safe one beats both."""
    from mpc.score import score, Weights
    K = 16
    soon = [1.0] * 3 + [0.0] * (K - 2)
    late = [1.0] * 14 + [0.0] * (K - 13)
    safe = [1.0] * (K + 1)
    rem = lambda x, y: np.zeros_like(x)
    sc = score(_fake_out(5, [soon, late], K), rem, Weights())
    assert sc["death"].min() > 0.99 and sc["total"][1] > sc["total"][0] + 3.0
    assert 2.5 < sc["death_step"][0] < 3.5 and 13.5 < sc["death_step"][1] < 14.5
    sc = score(_fake_out(5, [late, safe], K), rem, Weights())
    assert sc["total"][1] > sc["total"][0] + 5.0 and sc["death_step"][1] == -1


def test_hazard_boxes_rows_without_kill_shapes_use_the_size_box():
    import realgame
    hz = [[0.0, 0.0, 20.0, 10.0, 1.0, 1.0, 0.0], [50.0, 0.0, 20.0, 10.0, 0.5, 2.0, 0.3, None]]
    boxes, kept = realgame.hazard_boxes(hz, [True, True])
    assert boxes[0] == (0.0, 0.0, 10.0, 5.0, 0.0) and boxes[1] == (50.0, 0.0, 5.0, 10.0, 0.3) and kept == [0, 1]
    boxes, kept = realgame.hazard_boxes(hz, [False, True])          # decoration (by name) only without shapes
    assert kept == [1]


@need_data
def test_outside_the_level_is_deadly_in_the_agents_map():
    import realgame as rg
    import mpc.offline as off
    lv = off.level_from_data(DATA, LEVEL)
    grid = lv.grid_with_hazards([])
    x, y = lv.ox + 10.0, lv.oy + lv.H * 10.0                 # the leftmost column, half way up
    o = rg.observe(lv, grid, [x, y, 0.0, 0.0, 0, 0, 0, 0], None, 0.0, 0.0)
    K = rg.K
    patch = o[10:10 + (2 * K + 1) ** 2].reshape(2 * K + 1, 2 * K + 1)
    assert (patch[:, :K] == rg.C_HAZARD).all()               # every column left of the map


def _saw_row(cy):
    """A piece that stays at (0, 100) while its circle (r 20) moves: realgame.js hazards row with kill shapes."""
    return [0.0, 100.0, 50.0, 50.0, 0.6, 0.6, 0.0, [[1, 0.0, cy, 20.0, 20.0, 0]]]


def test_future_follows_the_kill_shapes_not_the_piece():
    from mpc.future import FutureWorld, REC_RADIUS
    fw = FutureWorld()
    for f in range(10):                                  # the recording: the circle falls 2 units a frame
        fw.observe(f, [_saw_row(100.0 - 2 * f)], None, [], [True], None, 16.667)
    fw.new_wave()
    fw.observe(0, [_saw_row(100.0)], None, [], [True], None, 16.667)
    fw.observe(1, [_saw_row(98.0)], None, [], [True], None, 16.667)
    worlds, check, recorded = fw.worlds(1, 3, 16.667)
    assert recorded == 3 and fw.trusted
    b, r, ids = check[0]                                 # frame 2: the recorded circle, as its bounding square
    assert tuple(b[0]) == (0.0, 96.0, 20.0, 20.0, 0.0) and r[0] == REC_RADIUS and ids == ["h0"]
    assert tuple(worlds[0].boxes[0][:2]) == (0.0, 100.0)  # the model inputs keep the piece's own box
    fw.observe(2, [_saw_row(80.0)], None, [], [True], None, 16.667)   # the piece is where recorded, its shape not
    assert not fw.trusted
    fw2 = FutureWorld()                                  # no recording: the shape's own velocity
    fw2.observe(0, [_saw_row(100.0)], None, [], [True], None, 10.0)
    fw2.observe(1, [_saw_row(99.0)], None, [], [True], None, 10.0)
    _, check, recorded = fw2.worlds(1, 2, 10.0)
    assert recorded == 0 and check[1][0][0][1] == pytest.approx(97.0) and check[1][1][0] > 0


def test_future_starts_a_new_recording_after_a_level_switch_mid_wave():
    """The user switches level mid-wave (no wave end): the old level's unfinished wave must not become the new
    level's recording. The new level's first wave records (extrapolated meanwhile), its second wave is trusted
    and looks the future up in the recording; new_wave says what each wave did."""
    from mpc.future import FutureWorld
    fw = FutureWorld()
    for f in range(30):                                  # level A, first wave: recorded
        fw.observe(f, [_saw_row(100.0 - f)], None, [], [True], None, 16.667)
    info = fw.new_wave()
    assert info == {"frames": 30, "recording": 0, "trusted": None, "mismatch": None, "extended": 30}
    for f in range(12):                                  # level A, second wave, cut by the level switch
        fw.observe(f, [_saw_row(100.0 - f)], None, [], [True], None, 16.667)
    fw.new_level()                                       # level B: its saw falls 3 units a frame from 50
    assert fw.ref == {} and fw.cur == {}
    for f in range(20):
        fw.observe(f, [_saw_row(50.0 - 3 * f)], None, [], [True], None, 16.667)
        if f == 5:
            assert fw.worlds(5, 4, 16.667)[2] == 0          # first wave of B: extrapolated
    info = fw.new_wave()
    assert info == {"frames": 20, "recording": 0, "trusted": None, "mismatch": None, "extended": 20}
    assert sorted(fw.ref) == list(range(20))
    for f in range(6):                                   # second wave of B: matches, uses the recording
        fw.observe(f, [_saw_row(50.0 - 3 * f)], None, [], [True], None, 16.667)
    _, check, recorded = fw.worlds(5, 4, 16.667)
    assert fw.trusted and recorded == 4 and check[0][0][0][1] == pytest.approx(50.0 - 3 * 6)
    info = fw.new_wave()
    assert info == {"frames": 6, "recording": 20, "trusted": True, "mismatch": None, "extended": 0}


def test_step_chains_of_a_wave_shorter_than_the_step():
    """A wave cut after 1-3 rows (a level change) gives the step-4 trainer no full chain, and no crash."""
    from wm.buffer import ROW, chains
    for n in (1, 2, 3, 5):
        rows = np.zeros(n, ROW)
        rows["frame"] = np.arange(n)
        rows["action"] = 2
        c = chains(rows, 4)
        assert len(c["s"]) == max(0, n - 4) and len(c["s2"]) == len(c["s"])
