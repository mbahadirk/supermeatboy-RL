"""Learning from the run's own play (agent.py --online): the planner's curiosity (Planner.novelty) and daring under
the stall rule, the recording of held-out levels to their own folder, and wm.online's acceptance rule."""
import glob
import os
import sys
import tempfile
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from plan import planner as P                                              # noqa: E402
from plan.geometry import Terrain                                          # noqa: E402

MODEL = os.path.join(HERE, "runs", "experiments", "human_v3_gpu", "wm_model.pt")
REPLAY = os.path.join(HERE, "runs", "experiments", "human_v3", "replay.npz")
WAVES = os.path.join(HERE, "runs", "experiments", "online_v1", "try1", "wm_data")


def bare_planner():
    pl = P.Planner.__new__(P.Planner)
    pl.lv = types.SimpleNamespace(name="t", ox=0.0, oy=0.0)
    pl.terrain = Terrain(np.zeros((5, 10), np.uint32), 0.0, 0.0)
    return pl


def test_novelty_is_the_share_of_a_path_in_tiles_the_run_has_not_been_in():
    pl = bare_planner()
    for x in (10.0, 30.0, 50.0):                     # the run has been in tiles 0..2 of row 0
        pl.visit(x, 10.0)
    xs = np.array([[10.0, 30.0, 50.0, 70.0, 90.0],   # 2 of its 4 later positions in new tiles
                   [10.0, 10.0, 30.0, 30.0, 50.0],   # none
                   [10.0, -50.0, -70.0, 70.0, 70.0]])  # outside the map is not new; tile 3 twice
    ys = np.full_like(xs, 10.0)
    assert pl.novelty(xs, ys).tolist() == [0.5, 0.0, 0.5]
    pl.lv = types.SimpleNamespace(name="u", ox=0.0, oy=0.0)       # another level: its own counts
    assert pl.novelty(xs[1], ys[1]).tolist() == [1.0]
    pl.lv = types.SimpleNamespace(name="t", ox=0.0, oy=0.0)       # and back: kept for the run
    assert pl.novelty(xs[1], ys[1]).tolist() == [0.0]


def test_the_runner_records_held_out_levels_only_with_online_learning():
    import agent
    r = agent.Runner.__new__(agent.Runner)
    r.wm, r.wm_held, r.evaluation = "wm_data", "wm_heldout", None
    assert r.rec_for(False) == "wm_data" and r.rec_for(True) == "wm_heldout"
    r.wm_held = None                                 # without --online: a held-out level is not recorded
    assert r.rec_for(True) is None
    r.evaluation = object()                          # a first-life evaluation records nothing
    assert r.rec_for(False) is None


@pytest.mark.skipif(not (os.path.exists(MODEL) and os.path.exists(REPLAY) and glob.glob(os.path.join(WAVES, "wave_*.npz"))),
                    reason="no model, replay sample or recorded waves")
def test_online_cycle_learns_the_new_wave_and_refuses_forgetting(monkeypatch):
    """wm.online: a candidate trained on earlier waves of the run is better on its newest wave (published when the
    replay slack allows what the fine-tune costs on the replay rows); with no slack it is refused."""
    import wm.online as W
    from wm.model import WorldModel
    monkeypatch.setattr(W, "OWN_EPOCHS", 2)
    m = WorldModel.load(MODEL).eval()
    replay = W.load_replay(REPLAY, m)
    keep = np.r_[np.nonzero(~replay["val"])[0][:20000], np.nonzero(replay["val"])[0][:4000]]
    replay = {k: v[keep] for k, v in replay.items()}
    rval = {k: v[replay["val"]] for k, v in replay.items() if k in W.KEYS}
    tmp = tempfile.mkdtemp()
    parts = [W.featurize(p, m, tmp) for p in sorted(glob.glob(os.path.join(WAVES, "wave_*.npz")))[:3]]
    parts = [p for p in parts if p is not None]
    monkeypatch.setattr(W, "REPLAY_SLACK", 1.0)                      # forgetting allowed: the fresh gain decides
    cand, losses, steps = W.cycle(m, parts[:-1], parts[-1:], replay, rval, np.random.default_rng(0), 0)
    assert steps > 0 and losses["fresh"][1] < losses["fresh"][0]
    assert cand is not None and cand is not m                       # published (the base object untouched)
    monkeypatch.setattr(W, "REPLAY_SLACK", -1.0)                     # no loss on the replay may rise: refused
    cand2, losses2, _ = W.cycle(m, parts[:-1], parts[-1:], replay, rval, np.random.default_rng(0), 0)
    assert cand2 is None
