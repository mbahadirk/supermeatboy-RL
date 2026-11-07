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


# ---- the collision guard ------------------------------------------------------------------------------------
def test_box_overlap_follows_the_exact_shapes():
    t = terrain()
    h = 5.0                                              # BODY - tolerance 1
    # standing on the floor (bottom at 40) / 1 unit in / 2 units in
    assert t.box_overlap(np.array([100.0] * 3), np.array([46.0, 45.0, 44.0]), h).tolist() == [False, False, True]
    # the lower-half block (15, 2): solid 40..50; its upper half is open
    assert t.box_overlap(np.array([310.0, 310.0]), np.array([56.0, 54.0]), h).tolist() == [False, True]
    # the slope (20, 2) rises to the right (surface y = 40 + (x - 400)): under the box's right edge x = 408 it
    # is at 48, so a box bottom at 49 is clear and one at 46 is not
    assert t.box_overlap(np.array([403.0, 403.0]), np.array([54.0, 51.0]), h).tolist() == [False, True]
    assert not t.box_overlap(np.array([0.0]), np.array([-30.0]), h)[0]     # outside the map: not solid
    o = t.opened([(2, 15)])                              # an opened lock: open
    assert not o.box_overlap(np.array([310.0]), np.array([50.0]), h)[0] and t.box_overlap(np.array([310.0]),
                                                                                            np.array([50.0]), h)[0]
    assert not Terrain(t.words0, 0.0, 0.0, slopes=False).box_overlap(np.array([417.0]), np.array([49.0]), h)[0]


def _rows(xy, v=(0.0, 0.0)):
    from wm.buffer import ROW
    r = np.zeros(len(xy), ROW)
    r["x"], r["y"] = [p[0] for p in xy], [p[1] for p in xy]
    r["vx"], r["vy"] = v
    return r


def test_collide_lands_on_the_half_floor_and_keeps_the_free_axis():
    from plan.planner import collide
    t = terrain()
    # falling into the lower-half block (top y = 50) while moving right: x kept, y back to contact, vy 0
    prev, s = _rows([(305.0, 60.0)]), _rows([(308.0, 47.0)], (180.0, -400.0))
    s2 = s.copy()
    assert collide(t, prev, s2).tolist() == [True]
    assert s2["x"][0] == 308.0 and abs(s2["y"][0] - 56.0) < 0.05 and s2["y"][0] >= 56.0
    assert s2["vy"][0] == 0.0 and s2["vx"][0] == 180.0
    # running into the wall column (x 200..220) from the left: y kept, x back to contact (194), vx 0
    prev, s = _rows([(190.0, 100.0)]), _rows([(197.0, 98.0)], (390.0, -50.0))
    assert collide(t, prev, s).tolist() == [True]
    assert abs(s["x"][0] - 194.0) < 0.05 and s["x"][0] <= 194.0 and s["y"][0] == 98.0
    assert s["vx"][0] == 0.0 and s["vy"][0] == -50.0
    # into the corner of floor and wall: both axes cut
    prev, s = _rows([(190.0, 50.0)]), _rows([(197.0, 42.0)], (390.0, -400.0))
    assert collide(t, prev, s).tolist() == [True]
    assert abs(s["x"][0] - 194.0) < 0.05 and abs(s["y"][0] - 46.0) < 0.05 and s["vx"][0] == 0 and s["vy"][0] == 0
    # within the tolerance (0.5 into the floor), clear, or starting already deep inside: left alone
    prev, s = _rows([(100.0, 47.0), (100.0, 60.0), (100.0, 30.0)]), _rows([(100.0, 45.5), (104.0, 55.0), (100.0, 25.0)])
    s0 = s.copy()
    assert collide(t, prev, s).tolist() == [False, False, False] and np.array_equal(s, s0)
    # through the open cell next to the half block (the hole): nothing to clip
    prev, s = _rows([(290.0, 70.0)]), _rows([(290.0, 60.0)])
    assert collide(t, prev, s).tolist() == [False]


@pytest.mark.skipif(not os.path.exists(MODEL), reason="no world model")
def test_imagined_paths_stay_out_of_the_floor(flat):
    """With the guard, no imagined position of any stage overlaps the floor by more than the tolerance; the
    chosen plan records how many of its frames were clipped (None with the guard off)."""
    import torch
    from plan.planner import COLLIDE_TOL, imagine, state_row
    from realpath import BODY
    lv, pl = flat
    torch.set_num_threads(1)
    worlds, frames = empty_inputs(pl.H)
    row = state_row(210.0, 80.0, 0.0, -300.0, 0, 8, 0, 0)               # falling onto the floor
    acts = pl.lib_actions[:8]
    off = imagine(pl.model, slice(None), pl.geom, np.repeat(row, 8), acts, 16.667, worlds)
    on = imagine(pl.model, slice(None), pl.geom, np.repeat(row, 8), acts, 16.667, worlds,
                 terrain=pl.guard_terrain(0))
    assert off["clip"] is None and on["clip"].shape == (len(pl.model.members), 8, MAX_LEN + 1)
    inside = pl.guard.box_overlap(on["x"].ravel(), on["y"].ravel(), BODY - COLLIDE_TOL)
    assert not inside.any()
    plan = pl.plan(row, 0, worlds, frames)
    assert plan.model_clipped is not None and 0 <= plan.model_clipped <= pl.H
    assert "clipped" in plan.counts["final"]
    pl.collide_tol = None
    try:
        assert pl.plan(row, 0, worlds, frames).model_clipped is None
    finally:
        pl.collide_tol = COLLIDE_TOL
        pl.new_wave()


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


DATA_114 = os.path.join(HERE, "runs", "plan_20261107-174219", "wm_data")


@pytest.mark.skipif(not (os.path.exists(MODEL) and os.path.isdir(DATA_114)), reason="no model or 1-14 recording")
def test_lookups_on_the_1_14_half_floor_stay_out_of_solid():
    """Item b of the guard round: with the collision guard, no point the planner's score looks up in the
    progress field (Planner.value: every stage's end points, per member) is inside solid geometry on a recorded
    1-14 state standing on the half-block floor (so the field's "best open neighbour" rule for points inside
    walls is not reached through a floor); without the guard many are (the model goes through that floor)."""
    import torch
    from plan.offline import build, waves_of
    from plan.planner import state_row
    torch.set_num_threads(1)
    inside = {}
    for tol in (None, "default"):
        lv, field, pl, _, _, _ = build(DATA_114, "1-14", MODEL, None, tol)
        r = next(r for rows, _ in waves_of(DATA_114, "1-14") for r in rows
                 if r["status"] == 0 and 60 <= r["x"] <= 120 and abs(r["y"] - 36.0) < 0.5)
        got = []
        orig = pl.value

        def logged(x, y, keys=0, leg=None):
            got.append((np.ravel(x).astype(float), np.ravel(y).astype(float)))
            return orig(x, y, keys, leg)
        pl.value = logged
        worlds, frames = empty_inputs(pl.H)
        pl.plan(state_row(r["x"], r["y"], r["vx"], r["vy"], r["flags"], r["state"], r["keys"], r["held"]), 10,
                worlds, frames)
        xs, ys = np.concatenate([g[0] for g in got[1:]]), np.concatenate([g[1] for g in got[1:]])   # [0]: start
        inside[tol] = int(pl.terrain.point_solid(xs, ys).sum())
    assert inside["default"] == 0 and inside[None] > 10, inside


# ---- goal and key contact -----------------------------------------------------------------------------------
def test_touches_needs_a_majority_of_members():
    from plan.planner import Planner
    xs = np.tile(np.arange(49, dtype=float), (5, 1))
    ys = np.zeros((5, 49))
    g, _ = Planner._touches(xs, ys, np.array([10, 12, -1, -1, -1]), [], 5)
    assert g is None                                                         # 2 of 5: not a success
    g, k = Planner._touches(xs, ys, np.array([10, 12, 14, 20, -1]), [(0, np.array([3, 4, 5, -1, -1]),
                                                                       np.array([1, 1, 1, -1, -1]))], 5)
    assert g == {"frame": 12, "x": pytest.approx(14.0), "y": 0.0, "members": 4}    # lower median of 10 12 14 20
    assert k == [{"frame": 4, "key": 1, "x": pytest.approx(4.0), "y": 0.0}]


def contact_planner(goal_x, keys=()):
    """A flat level (as `flat`) with the goal at goal_x and keys [(x, y)], planner without deadlines."""
    from game.level import Level
    from plan.planner import Planner
    from plan.progress import ProgressField
    H, W = 20, 60
    w = np.zeros((H, W), np.uint32)
    w[0:2, :] = 1
    info = {"name": "test-contact", "W": W, "H": H, "ox": 0.0, "oy": 0.0, "spawn": [10 * T + 10, 46.0],
            "goals": [[goal_x, 50.0]], "objs": [[0, float(x), float(y), 0x40] for x, y in keys], "lethalTypes": []}
    lv = Level(info, w.astype("<u4").tobytes())
    field = ProgressField(w, 0.0, 0.0, lv.spawn, lv.goals, [], [], lv.keys, [], [])
    return lv, Planner(MODEL, lv, field, budget_ms=None)


@needs_model
def test_touching_the_goal_ranks_first_and_earliest():
    from plan.planner import SUCCESS_SCORE
    lv, pl = contact_planner(300.0)                   # 90 units right of the agent
    assert len(pl.goal_boxes) == 1
    worlds, frames = empty_inputs(pl.H)
    plan = pl.plan(standing(), 10, worlds, frames)
    gt = plan.goal_touch
    assert gt is not None and 1 <= gt["frame"] <= 30 and gt["members"] >= 3
    assert abs(gt["x"] - 300.0) < 20.0
    assert plan.score["total"] == pytest.approx(SUCCESS_SCORE - gt["frame"], abs=0.5)
    assert f"touches the princess at f{gt['frame']}" in plan.summary()
    assert plan.counts["final"]["goal"] >= 1 and 0.0 <= plan.finish_model <= 1.0
    for a in plan.alternatives:
        if a["stage"] != "final":
            continue
        if a["reason"].startswith("reaches the princess"):
            assert a["total"] <= plan.score["total"]
        else:
            assert a["total"] < SUCCESS_SCORE - pl.H                         # below every success
    d = json.loads(json.dumps(plan.to_dict()))
    assert d["goal_touch"]["frame"] == gt["frame"] and d["key_touches"] == []


@needs_model
def test_frames_after_the_goal_touch_are_ignored():
    lv, pl = contact_planner(300.0)
    worlds, frames = empty_inputs(pl.H, [(322.0, 150.0, 2.0, 150.0, 0.0)])   # a strip right behind the goal
    plan = pl.plan(standing(), 10, worlds, frames)
    assert plan.goal_touch is not None and plan.veto is None
    assert plan.soft_frame <= plan.goal_touch["frame"]                       # risk only up to the touch


@needs_model
def test_starting_inside_the_goal_box_is_not_a_touch():
    lv, pl = contact_planner(10 * T + 10 + 4.0)       # the agent stands 4 units from the goal piece
    worlds, frames = empty_inputs(pl.H)
    plan = pl.plan(standing(), 10, worlds, frames)
    assert plan.counts["final"]["start_in_goal"] and plan.goal_touch is None


@needs_model
def test_a_key_pickup_continues_on_the_next_leg():
    lv, pl = contact_planner(600.0, keys=[(330.0, 50.0)])
    assert pl.field.n_legs == 2 and len(pl.key_boxes) == 1
    worlds, frames = empty_inputs(pl.H)
    plan = pl.plan(standing(), 10, worlds, frames)
    assert plan.goal_touch is None and plan.leg == 0
    assert plan.key_touches and plan.key_touches[0]["key"] == 0
    assert "takes key 0 at f" in plan.summary()
    ex, ey = plan.path[-1]
    v_leg0 = float(pl.field.values(np.array([ex]), np.array([ey]), 0, True)[0])
    assert plan.v_end is not None and plan.v_end < v_leg0 - 1.0              # V of the leg after the key
