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
    base_replay = W.loss_of(m, rval)                                 # m is the base model
    monkeypatch.setattr(W, "REPLAY_SLACK", 1.0)                      # forgetting allowed: the fresh gain decides
    cand, losses, steps = W.cycle(m, parts[:-1], parts[-1:], replay, rval, np.random.default_rng(0), 0, base_replay)
    assert steps > 0 and losses["fresh"][1] < losses["fresh"][0]
    assert cand is not None and cand is not m                       # published (the base object untouched)
    monkeypatch.setattr(W, "REPLAY_SLACK", -1.0)                     # no loss on the replay may rise: refused
    cand2, losses2, _ = W.cycle(m, parts[:-1], parts[-1:], replay, rval, np.random.default_rng(0), 0, base_replay)
    assert cand2 is None


def toy_parts(n: int, seed: int) -> dict:
    r = np.random.default_rng(seed)
    return {"X": r.normal(size=(n, 3)).astype(np.float32), "yc": r.normal(size=(n, 1)).astype(np.float32),
            "yb": np.zeros(n, np.float32), "ys": np.zeros(n, np.int64), "yr": np.zeros(n, np.float32),
            "level": np.full(n, "t")}


def test_online_training_steps_are_clipped_to_max_steps(monkeypatch):
    """wm.online.fine_tune: OWN_EPOCHS passes over the pending rows (rehearsing absorbed ones), at most MAX_STEPS."""
    torch = pytest.importorskip("torch")
    import wm.online as W

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(3, 1)

        def loss(self, X, yc, yb, ys, yr, wc):
            return ((self.w(X) - yc) ** 2).mean(), None

    replay = dict(toy_parts(3000, 1), val=np.arange(3000) % 5 == 0)
    pending, old = toy_parts(2000, 2), toy_parts(500, 3)
    assert W.train_steps(2000) == W.OWN_EPOCHS * 2000 // round(W.BATCH * (1 - W.REPLAY_SHARE)) <= W.MAX_STEPS
    assert W.train_steps(10 ** 7) == W.MAX_STEPS
    monkeypatch.setattr(W, "MAX_STEPS", 40)                         # above the MIN_SAMPLES floor, below 4 passes
    assert W.fine_tune(Toy(), pending, replay, np.random.default_rng(0), old) == 40
    assert W.fine_tune(Toy(), pending, replay, np.random.default_rng(0)) == 40


def test_online_accepts_against_the_base_models_replay_loss(monkeypatch):
    """wm.online.cycle: the forgetting bar is the base model's replay loss, not the published model's own."""
    import wm.online as W
    rval = toy_parts(10, 4)

    def fake_fine_tune(model, pending, replay, rng, old=None):
        model.trained = True
        return 7

    def fake_loss_of(model, d):                                     # replay 0.02 -> 0.04, fresh 1.0 -> 0.5
        trained = getattr(model, "trained", False)
        return (0.04 if trained else 0.02) if d is rval else (0.5 if trained else 1.0)

    monkeypatch.setattr(W, "fine_tune", fake_fine_tune)
    monkeypatch.setattr(W, "loss_of", fake_loss_of)
    m = types.SimpleNamespace()
    args = (m, [toy_parts(10, 5)], [toy_parts(10, 6)], {}, rval, np.random.default_rng(0), 0)
    cand, losses, steps = W.cycle(*args, base_replay=1.0)           # the replay loss rose vs the model's own: accepted
    assert cand is not None and steps == 7 and losses["replay"] == [0.02, 0.04] and losses["bar"] == 1.05
    cand, losses, _ = W.cycle(*args, base_replay=0.001, old_parts=[toy_parts(10, 7)])
    assert cand is None
