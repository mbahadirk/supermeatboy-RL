"""Checks for the beam planner (plan/planner.py), its move library, support geometry and explanations. No game;
the world model and recorded data are used when present (skipped otherwise)."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from mpc.candidates import ACTIONS                                    # noqa: E402
from plan import safety as S                                          # noqa: E402
from plan.explain import Plan, SurpriseTracker, HORIZONS, ERR90_AT, SURPRISE_FACTOR   # noqa: E402
from plan.geometry import Terrain                                     # noqa: E402
from plan.moves import LIBRARY, BY_NAME, CORE_SECOND, MIN_LEN, MAX_LEN, padded       # noqa: E402
from plan.planner import FAIL_COST, FailureMemory                     # noqa: E402

MODEL = os.path.join(HERE, "runs", "wm_locked", "wm_model.pt")
DATA_113 = os.path.join(HERE, "runs", "real_20261102-165351", "wm_data")
T = 20.0


# ---- move library -------------------------------------------------------------------------------------------
def test_library_names_unique_and_sequences_valid():
    names = [m.name for m in LIBRARY]
    assert len(names) == len(set(names)) and 60 <= len(names) <= 80
    assert all(c in BY_NAME for c in CORE_SECOND)
    for m in LIBRARY:
        a = m.array
        assert MIN_LEN <= len(a) <= MAX_LEN, m.name
        assert ((a >= 0) & (a < len(ACTIONS))).all(), m.name
        jump = np.array([ACTIONS[k][2] for k in a], bool)
        if jump.any():                         # the jump key is released at least 2 frames before the end
            assert not jump[-2:].any(), m.name
            assert "jump" in m.name or "climb" in m.name, m.name
        else:
            assert "jump" not in m.name.replace("jump up", "") or m.family == "walk", m.name
    p = padded(BY_NAME["jump R hold 10"].actions, 48)
    assert len(p) == 48 and not ACTIONS[p[-1]][2] and ACTIONS[p[-1]][1]      # padding: right, jump released


def test_wall_jump_moves_press_away_with_a_new_press():
    a = BY_NAME["wall-jump L hold 8"].array
    assert not ACTIONS[a[0]][2] and ACTIONS[a[0]][0]          # released first
    assert all(ACTIONS[k][2] and ACTIONS[k][0] for k in a[1:9])
    c = BY_NAME["wall-climb R hold 10"].array
    assert all(ACTIONS[k][2] and ACTIONS[k][1] for k in c[1:11])


# ---- support geometry ---------------------------------------------------------------------------------------
def terrain():
    """Floor rows 0-1 (top at y = 40); a wall column at col 10 (x 200..220) rows 2-9; a half block (lower half)
    at (15, 2), a slope at (20, 2) with its open corner top-left (solid u >= v: rising to the right)."""
    w = np.zeros((12, 30), np.uint32)
    w[0:2, :] = 1
    w[2:10, 10] = 1
    w[2, 15] = 1 | (1 << 18) | (2 << 12)                  # half block, rotated 180: the lower half
    w[2, 20] = 1 | (2 << 18) | (2 << 12)                  # slope, rot 2, flip 0: open corner 2 (top-left)
    return Terrain(w, 0.0, 0.0)


def test_support_ground_wall_half_and_slope():
    t = terrain()
    g, s = t.support(np.array([100.0, 100.0, 100.0]), np.array([46.0, 47.0, 49.0]))
    assert g.tolist() == [True, True, False]            # standing 6 above the floor; 3 units above it: airborne
    g, s = t.support(np.array([193.5, 192.0, 226.5, 150.0]), np.array([100.0, 100.0, 100.0, 100.0]))
    assert s.tolist() == [1, 0, -1, 0] and not g.any()   # wall on the right within 1 unit / not / on the left
    g, _ = t.support(np.array([310.0, 310.0]), np.array([56.0, 66.0]))
    assert g.tolist() == [True, False]                  # the half block's top is y = 50
    # slope surface y = 40 + u * 20 at x = 400 + u * 20: a body resting with its bottom-right corner on it
    x = 405.0
    y = 40.0 + (x + 6 - 400.0) + 6
    g, _ = t.support(np.array([x, x]), np.array([y, y + 4]))
    assert g.tolist() == [True, False]


# ---- failure memory and surprises (no model) ----------------------------------------------------------------
def test_failure_memory_counts_a_move_that_did_not_deliver():
    m = FailureMemory()
    acts = BY_NAME["run R 16f"].array
    m.expect(0, 100, "run R 16f", 16, padded(acts, 48), (5, 2), 40.0, 36.0)       # expects a drop of 4
    assert m.update(0, 110, 39.0) == []                                            # not finished yet
    ev = m.update(0, 116, 39.0)                                                    # dropped 1 < 4 / 2
    assert len(ev) == 1 and ev[0]["move"] == "run R 16f" and ev[0]["count"] == 1
    assert m.penalty((5, 2), "run R 16f") == pytest.approx(FAIL_COST)
    assert m.reason((5, 2), "run R 16f") == "run R 16f from (5,2) failed 1 times"
    m.expect(0, 200, "run R 16f", 16, padded(acts, 48), (5, 2), 40.0, 36.0)
    m.expect(0, 202, "wait 8f", 8, padded(BY_NAME["wait 8f"].array, 48), (5, 2), 40.0, 39.8)   # changed move:
    assert m.update(0, 216, 40.0) == []                                            # abandoned, expects nothing
    m.expect(0, 300, "run R 16f", 16, padded(acts, 48), (5, 2), 40.0, 36.0)
    m.expect(0, 302, "run R 16f", 16, padded(acts, 48)[2:], (5, 2), 39.5, 36.0)   # the same move continued
    assert m.update(0, 316, 37.5) == []                                            # dropped 2.5 >= 2: fine
    assert m.count((5, 2), "run R 16f") == 1


def fake_plan(frame=10, preds=None, veto=None, risk=0.0):
    preds = preds or {h: [100.0 + h, 50.0] for h in HORIZONS}
    return Plan(frame=frame, agent=0, state={}, chosen=["run R 8f"], actions=[2] * 48, pad=40, from_prev=False,
                path=[], spread=[], landings=[], wall_jumps=[], nearest=[], veto=veto, soft_risk=risk, soft_frame=-1,
                score={}, v_start=10.0, v_end=9.0, dead_end_members=0, leg=0, end_supported=True, alternatives=[],
                predictions=preds, death_model=0.0, timing={}, counts={})


def _play(st, f0, f1, a=2, agent=0):
    for f in range(f0, f1):
        st.played(agent, f, a)


def test_surprise_tracker_events():
    st = SurpriseTracker()
    st.add(fake_plan())
    _play(st, 10, 18)                                                             # the plan's own actions
    assert st.observe(0, 11, 101.1, 50.0) == []                                   # within 1.5 x 0.43
    ev = st.observe(0, 14, 104.0 + 10.0, 50.0)                                    # 10 units off at +4
    assert len(ev) == 1 and ev[0]["kind"] == "position" and ev[0]["horizon"] == 4
    assert ev[0]["threshold"] == pytest.approx(SURPRISE_FACTOR * ERR90_AT[4])
    ev = st.observe(0, 18, 108.0, 50.0, dead=True)
    assert [e["kind"] for e in ev] == ["unpredicted death"]
    st.add(fake_plan(frame=30, veto={"frame": 3, "id": "h1", "kind": "hazard"}))
    assert st.observe(0, 33, 0.0, 0.0, dead=True) == []                            # the plan saw it coming


def test_surprise_counts_only_followed_plans():
    st = SurpriseTracker()
    st.add(fake_plan(frame=10))
    _play(st, 10, 12)                                                             # followed for 2 frames,
    _play(st, 12, 30, a=1)                                                        # then something else played
    assert st.observe(0, 11, 101.0 + 3.0, 50.0) and st.err[1] == [pytest.approx(3.0)]   # +1: followed, counted
    assert st.observe(0, 14, 0.0, 0.0) == []                                      # +4: replaced, not a surprise
    assert st.observe(0, 18, 0.0, 0.0) == [] and st.err[8] == []
    # an unpredicted death is still judged by the latest plan covering the frame, followed or not
    st.add(fake_plan(frame=20))
    ev = st.observe(0, 25, 0.0, 0.0, dead=True)
    assert [e["kind"] for e in ev] == ["unpredicted death"] and ev[0]["plan_frame"] == 20
    # errors per horizon: median, 90%, n; cleared by new_wave
    st.add(fake_plan(frame=40))
    _play(st, 40, 57)
    for h in HORIZONS:
        st.observe(0, 40 + h, 100.0 + h + 1.0, 50.0)                              # 1 unit off at every horizon
    e = st.errors()
    assert set(e) == {str(h) for h in HORIZONS}
    assert e["4"] == [1.0, 1.0, 1] and e["1"][2] == 2 and e["16"][0] == pytest.approx(1.0)
    st.new_wave()
    assert st.errors()["1"] == [None, None, 0] and not st.plans and not st.played_at


# ---- the planner with the world model -----------------------------------------------------------------------
needs_model = pytest.mark.skipif(not os.path.exists(MODEL), reason="no world model (runs/wm_locked)")


@pytest.fixture(scope="module")
def flat():
    """A flat level: floor rows 0-1, spawn left, goal right; and a planner on it without deadlines (the
    assertions below need the complete search, whatever the machine's load)."""
    from game.level import Level
    from plan.planner import Planner
    from plan.progress import ProgressField
    H, W = 20, 60
    w = np.zeros((H, W), np.uint32)
    w[0:2, :] = 1
    info = {"name": "test-flat", "W": W, "H": H, "ox": 0.0, "oy": 0.0, "spawn": [10 * T + 10, 46.0],
            "goals": [[50 * T + 10, 50.0]], "objs": [], "lethalTypes": []}
    lv = Level(info, w.astype("<u4").tobytes())
    field = ProgressField(w, 0.0, 0.0, lv.spawn, lv.goals, [], [], [], [], [])
    return lv, Planner(MODEL, lv, field, budget_ms=None)


def empty_inputs(H, boxes=None):
    from wm.rollout import World
    fb = S.FrameBoxes(np.zeros((0, 5)), np.zeros(0, np.int8), []) if boxes is None else \
        S.FrameBoxes(np.array(boxes, float), np.zeros(len(boxes), np.int8), [f"h{k}" for k in range(len(boxes))])
    return [World() for _ in range(H)], [None] + [fb] * H


def standing(x=10 * T + 10):
    from plan.planner import state_row
    return state_row(x, 46.0, 0.0, 0.0, 0, 5, 0, 0)


@needs_model
def test_plan_has_full_horizon_predictions_and_serialises(flat):
    lv, pl = flat
    pl.new_wave()
    worlds, frames = empty_inputs(pl.H)
    plan = pl.plan(standing(), 100, worlds, frames)
    assert len(plan.actions) == pl.H and len(plan.path) == pl.H + 1
    assert set(plan.predictions) == set(HORIZONS)
    assert plan.chosen and all(n in BY_NAME for n in plan.chosen)
    assert plan.state["ground"] and plan.v_start is not None and plan.veto is None
    assert plan.counts["stage1"]["candidates"] == len(LIBRARY) and plan.counts["final"]["members"] == 5
    assert plan.score["progress"] > 0                                   # the goal is to the right, open floor
    d = json.loads(json.dumps(plan.to_dict()))
    assert d["chosen"] == plan.chosen and "alternatives" in d and "1" in d["predictions"]
    # the next decision keeps the previous plan, shifted, as a candidate
    plan2 = pl.plan(standing(), 102, worlds, frames)
    assert plan2.counts["final"]["prev_included"]


@needs_model
def test_no_budget_runs_every_stage_complete(flat):
    from plan.planner import BUDGET_MS, Planner
    lv, pl = flat
    assert pl.budget is None and BUDGET_MS == 120.0
    import inspect
    assert inspect.signature(Planner).parameters["budget_ms"].default == BUDGET_MS     # default unchanged
    pl.new_wave()
    worlds, frames = empty_inputs(pl.H)
    c = pl.plan(standing(), 100, worlds, frames).counts
    assert c["stage1"]["steps"] == MAX_LEN and c["stage1"]["complete"] == len(LIBRARY)
    assert c["stage2"]["steps"] == c["stage2"]["of_steps"] and "skipped" not in c["stage2"]


@needs_model
def test_a_move_into_a_hazard_is_vetoed_and_explained(flat):
    lv, pl = flat
    pl.new_wave()
    x0 = 10 * T + 10
    wall = (x0 + 30.0, 150.0, 4.0, 150.0, 0.0)          # a tall hazard strip just right of the agent
    worlds, frames = empty_inputs(pl.H, [wall])
    plan = pl.plan(standing(x0), 200, worlds, frames)
    assert plan.veto is None
    assert all(p[0] < x0 + 30.0 for p in plan.path)                     # never passes the strip
    hits = [a for a in plan.alternatives if a["reason"].startswith("hits h0 at frame")]
    assert hits, plan.alternatives
    assert plan.counts["stage1"]["vetoed"] > 0
    assert plan.nearest and plan.nearest[0]["id"] == "h0"


@needs_model
def test_failure_memory_penalty_changes_the_choice(flat):
    lv, pl = flat
    pl.new_wave()
    worlds, frames = empty_inputs(pl.H)
    p1 = pl.plan(standing(), 300, worlds, frames)
    cell = tuple(p1.state["cell"])
    pl.memory.counts[(cell, p1.chosen[0])] = 50
    pl.prev = {}
    p2 = pl.plan(standing(), 300, worlds, frames)
    assert p2.chosen[0] != p1.chosen[0]
    assert pl.memory.penalty(cell, p1.chosen[0]) == pytest.approx(50 * FAIL_COST)
    pl.memory.counts.clear()


@needs_model
def test_grouped_features_match_featurize(flat):
    from plan.planner import features_into
    from wm.features import FEAT_DIM, featurize
    from wm.rollout import World
    lv, pl = flat
    s = np.concatenate([standing(x) for x in (150.0, 210.0, 400.0, 620.0)])
    a = np.array([0, 2, 7, 5])
    w0 = World(np.array([[260.0, 60.0, 10, 10, 0.0], [700.0, 80.0, 5, 30, 0.3]]), [[300.0, 120.0, 0]], None)
    w1 = World(np.array([[200.0, 50.0, 8, 8, 0.0]]), [[320.0, 120.0, 0]], [[300.0, 120.0, 0]])
    groups = [(0, np.array([0, 2])), (5, np.array([1, 3]))]
    X = np.empty((4, FEAT_DIM), np.float32)
    features_into(X, groups, s, a, 16.667, pl.geom, [w0, w1])
    for (o, idx), w in zip(groups, [w0, w1]):
        ref = featurize(s[idx], a[idx], 16.667, pl.geom, w.boxes, w.pf, w.pf_prev, w.dt_prev_s, w.fans)
        np.testing.assert_allclose(X[idx], ref, atol=1e-6)


@needs_model
@pytest.mark.skipif(not os.path.isdir(DATA_113), reason="no recorded 1-13 data")
def test_plan_on_a_recorded_state():
    from mpc.future import FutureWorld
    from plan.offline import build
    from plan.planner import future_inputs, state_row
    lv, field, pl, ref_world, rows, world = build(DATA_113, "1-13", MODEL, 120.0)
    fw = FutureWorld()
    for w in ref_world:
        fw.observe(w["f"], w.get("hz", []), w.get("pf"), w.get("sh", []), lv.hz_keep, None, 16.667, lv.hz_grow)
    fw.new_wave()
    f = sorted(w["f"] for w in world)[200]
    for w in world:
        if w["f"] <= f:
            fw.observe(w["f"], w.get("hz", []), w.get("pf"), w.get("sh", []), lv.hz_keep, None, 16.667, lv.hz_grow)
    r = [r for r in rows if r["frame"] == f and r["status"] == 0][0]
    worlds, frames, recorded = future_inputs(fw, f, pl.H, 16.667, lv.name)
    assert len(frames) == pl.H + 1 and recorded > 0
    plan = pl.plan(state_row(r["x"], r["y"], r["vx"], r["vy"], r["flags"], r["state"], r["keys"], r["held"]), f,
                   worlds, frames)
    assert len(plan.actions) == pl.H and plan.timing["total"] > 0
    json.dumps(plan.to_dict())
    assert pl.observe(int(r["agent"]), f + 1, float(r["x"]) + 500.0, float(r["y"])) == []   # nothing played yet
    pl.played(int(r["agent"]), f, plan.actions[0])
    ev = pl.observe(int(r["agent"]), f + 1, float(r["x"]) + 500.0, float(r["y"]))
    assert any(e["kind"] == "position" and e["horizon"] == 1 for e in ev)
