"""The planning modes (plan/planner.py, Planner(modes=True)): long-safe (member 0), normal (member 1), reflex
(member 2), explore (members 3 and 4). No game; skipped without the world model."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from plan import planner as P                                     # noqa: E402
from plan import safety as S                                      # noqa: E402

MODEL = os.path.join(HERE, "runs", "wm_locked", "wm_model.pt")
T = 20.0
needs_model = pytest.mark.skipif(not os.path.exists(MODEL), reason="no world model (runs/wm_locked)")


@pytest.fixture(scope="module")
def flat():
    """A flat level (floor rows 0-1, spawn left, goal right) and a planner with modes, no deadlines."""
    from game.level import Level
    from plan.progress import ProgressField
    H, W = 20, 60
    w = np.zeros((H, W), np.uint32)
    w[0:2, :] = 1
    info = {"name": "test-flat", "W": W, "H": H, "ox": 0.0, "oy": 0.0, "spawn": [10 * T + 10, 46.0],
            "goals": [[50 * T + 10, 50.0]], "objs": [], "lethalTypes": []}
    lv = Level(info, w.astype("<u4").tobytes())
    field = ProgressField(w, 0.0, 0.0, lv.spawn, lv.goals, [], [], [], [], [])
    return lv, P.Planner(MODEL, lv, field, budget_ms=None, modes=True)


def inputs(n):
    from wm.rollout import World
    fb = S.FrameBoxes(np.zeros((0, 5)), np.zeros(0, np.int8), [])
    return [World() for _ in range(n)], [None] + [fb] * n


def standing():
    return P.state_row(10 * T + 10, 46.0, 0.0, 0.0, 0, 5, 0, 0)


@needs_model
def test_modes_use_their_members_and_record_the_mode(flat):
    lv, pl = flat
    pl.new_wave()
    assert pl.modes and pl.member == P.MEMBERS["normal"]
    assert pl.future_frames >= P.H_LONG
    worlds, frames = inputs(pl.future_frames)
    plan = pl.plan(standing(), 100, worlds, frames)
    assert plan.mode in P.MODES and plan.mode != "explore"
    m = plan.counts["modes"]
    assert m["long_safe"]["ran"]                     # supported, no long-safe plan yet
    assert m["long_safe"]["candidate"] and "long-safe" in m["candidates"]
    assert not m["reflex"]["ran"]                    # nothing near
    assert plan.to_dict()["mode"] == plan.mode
    # a valid long-safe plan younger than LONG_EVERY frames is reused, not searched again
    plan2 = pl.plan(standing(), 108, worlds, frames)
    assert not plan2.counts["modes"]["long_safe"]["ran"] and plan2.counts["modes"]["long_safe"]["age"] == 8


@needs_model
def test_long_safe_excludes_plans_with_soft_risk_over_the_limit(flat, monkeypatch):
    lv, pl = flat
    pl.new_wave()
    worlds, frames = inputs(pl.future_frames)
    real = pl._judge
    ck = {"terrain": pl.guard_terrain(0), "tol": pl.collide_tol}
    pen = np.zeros(len(pl.lib))

    def risky(soft):
        def judge(out, frames_, upto, leg, keys):
            j = real(out, frames_, upto, leg, keys)
            j["veto"][:] = False
            j["soft"] = np.full(len(j["soft"]), soft)
            return j
        return judge
    monkeypatch.setattr(pl, "_judge", risky(P.LONG_RISK + 0.01))
    res, cnt = pl._long_safe(standing(), worlds, frames, 16.667, ck, 0, 0, pen, 100.0)
    assert res is None and cnt["stage1_qualified"] == 0
    monkeypatch.setattr(pl, "_judge", risky(P.LONG_RISK))
    res, cnt = pl._long_safe(standing(), worlds, frames, 16.667, ck, 0, 0, pen, 100.0)
    assert res is not None and len(res[1]) == P.H_LONG and res[3] <= P.LONG_RISK and cnt["qualified"] > 0


@needs_model
def test_reflex_overrides_a_chosen_plan_with_early_risk(flat, monkeypatch):
    lv, pl = flat
    worlds, frames = inputs(pl.future_frames)
    k = next(i for i, m in enumerate(pl.lib) if m.name == "wait 8f")
    monkeypatch.setattr(pl, "_reflex", lambda *a, **kw: (k, 0.0, 10.0))
    pl.new_wave()
    monkeypatch.setattr(P, "early_risk", lambda sw, upto=None: np.zeros(sw.step_clear.shape[1]))
    calm = pl.plan(standing(), 100, worlds, frames, danger=True)          # danger, but the chosen plan is safe
    assert calm.counts["modes"]["reflex"]["ran"] and not calm.counts["modes"]["reflex"]["override"]
    assert calm.mode != "reflex"
    pl.new_wave()
    monkeypatch.setattr(P, "early_risk", lambda sw, upto=None: np.full(sw.step_clear.shape[1], 0.3))
    plan = pl.plan(standing(), 100, worlds, frames, danger=True)
    assert plan.mode == "reflex" and plan.chosen == ["wait 8f"]
    assert plan.switch["reason"].startswith("reflex override")
    pl.new_wave()
    plan = pl.plan(standing(), 100, worlds, frames)                       # no danger: reflex does not run
    assert not plan.counts["modes"]["reflex"]["ran"] and plan.mode != "reflex"


@needs_model
def test_reflex_runs_when_a_hazard_is_near_the_current_path(flat):
    lv, pl = flat
    pl.new_wave()
    x0 = 10 * T + 10
    worlds, far = inputs(pl.future_frames)
    assert not pl._danger_near(0, 100, x0, 46.0, far)
    near = [None] + [S.FrameBoxes(np.array([[x0 + 40.0, 46.0, 5.0, 5.0, 0.0]]), np.zeros(1, np.int8), ["h0"])] * 20
    assert pl._danger_near(0, 100, x0, 46.0, near)


@needs_model
def test_explore_picks_the_safe_plan_the_members_disagree_on_most(flat, monkeypatch):
    lv, pl = flat
    n = len(pl.lib)
    Hc = pl.H // pl.f

    def fake_imagine(model, members, geom, s0, actions, dt_ms, worlds, **kw):
        assert (members.start, members.stop) == (3, 5)
        x = np.zeros((2, len(s0), Hc + 1))
        x[1, :, -1] = np.arange(len(s0), dtype=float)       # candidate i: the members disagree by i units
        return {"x": x, "y": np.zeros_like(x), "vy": np.zeros_like(x)}

    def fake_judge(out, frames, upto, leg, keys):
        veto = np.zeros(n, bool)
        veto[n - 1] = True                                   # the most disagreeing one is vetoed
        return {"veto": veto, "veto_frame": np.where(veto, 5, -1), "soft": np.zeros(n), "succ": np.zeros(n, bool),
                "tg": np.full(n, -1), "tend": np.zeros(n), "sweep": None}
    monkeypatch.setattr(P, "imagine", fake_imagine)
    monkeypatch.setattr(pl, "_judge", fake_judge)
    worlds, frames = inputs(pl.H)
    p = pl.explore(standing(), 50, worlds, frames, agent=3)
    assert p.mode == "explore" and p.chosen == [pl.lib[n - 2].name] and p.veto is None
    assert p.score["disagreement"] == n - 2 and len(p.actions) == pl.H and p.frame == 50
