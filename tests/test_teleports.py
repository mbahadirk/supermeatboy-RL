"""Game portals (game.level Teleports / Jumps): the trigger, rollouts through a portal, paths stepping at the
teleport frame, safety sweeps that never test the line across one, and training samples through one dropped.
The world-model rollouts use runs/wm_step4 when present (skipped otherwise)."""
import gzip
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from game.level import (TELEPORT_COOLDOWN, TELEPORT_JUMP, Jumps, Teleports, frame_path, member_mean,  # noqa: E402
                        teleport_inside, teleport_jumps)
from plan import safety as S                                               # noqa: E402
from plan.commit import path_check                                         # noqa: E402

MODEL = os.path.join(HERE, "runs", "wm_step4", "wm_model.pt")
needs_model = pytest.mark.skipif(not os.path.exists(MODEL), reason="no world model (runs/wm_step4)")
T = 20.0


# ---- the trigger and one step ---------------------------------------------------------------------------
def test_trigger_is_a_rounded_box():
    assert teleport_inside(0.0, 0.0) and teleport_inside(15.2, -0.5) and teleport_inside(-15.2, -0.5)
    assert not teleport_inside(15.4, -0.5) and not teleport_inside(0.0, 15.0)     # flat sides at 15.3
    assert teleport_inside(12.0, 12.0) and not teleport_inside(13.0, 13.0)        # rounded corners


def test_step_finds_the_first_frame_inside():
    tp = Teleports([((0.0, 0.0), (300.0, 0.0))])
    end, frame, ex, ey = tp.step(np.array([-40.0]), np.array([0.0]), np.array([-4.0]), np.array([0.0]), 4,
                                 np.array([0]), np.array([-1]))
    assert end.tolist() == [0] and frame.tolist() == [3]                   # frames at -31, -22, -13, -4
    assert ex[0] == pytest.approx(-13.0) and ey[0] == 0.0
    # still cooling down until frame 4; kept off end 0; far away: nothing
    end, frame, _, _ = tp.step(np.array([-40.0] * 3), np.zeros(3), np.array([-4.0, -4.0, 500.0]), np.zeros(3), 4,
                               np.array([4, 0, 0]), np.array([-1, 0, -1]))
    assert end.tolist() == [0, -1, -1] and frame.tolist() == [4, 0, 0]
    assert tp.start_off(np.array([2.0, 300.0, 100.0]), np.zeros(3)).tolist() == [0, 1, -1]
    assert tp.left(np.array([2.0, 100.0]), np.zeros(2), np.array([0, 1])).tolist() == [0, -1]
    assert Teleports.of(type("L", (), {"portals": []})()) is None


# ---- rollouts -----------------------------------------------------------------------------------------------
def _flat_portal(portals):
    from game.level import Level
    from plan.planner import Planner
    from plan.progress import ProgressField
    H, W = 20, 60
    w = np.zeros((H, W), np.uint32)
    w[0:2, :] = 1
    info = {"name": "test-portal", "W": W, "H": H, "ox": 0.0, "oy": 0.0, "spawn": [5 * T, 46.0],
            "goals": [[55 * T, 50.0]], "objs": [], "lethalTypes": [], "portals": portals}
    lv = Level(info, w.astype("<u4").tobytes())
    field = ProgressField(w, 0.0, 0.0, lv.spawn, lv.goals, [], [], [], [], list(lv.portals))
    return lv, Planner(MODEL, lv, field, budget_ms=None)


def _run_right(pl, n, x=5 * T, keys=0):
    from mpc.candidates import ACTIONS
    from plan.planner import imagine, state_row
    from wm.rollout import World
    run = next(i for i, a in enumerate(ACTIONS) if a[1] and not a[0] and not a[2] and a[3])      # right + run
    row = state_row(x, 46.0, 0.0, 0.0, 0, 5, keys, 0)
    acts = np.full((1, n), run, np.int64)
    return imagine(pl.model, slice(None), pl.geom, row, acts, 16.667, [World()] * n, **pl.imagine_options(0, frame=0))


@needs_model
def test_rollout_runs_through_a_portal_and_rearms_after_the_cooldown():
    A, B = (15 * T, 46.0), (8 * T, 46.0)        # running right into A comes out at B, left of it: and again
    lv, pl = _flat_portal([[list(A), list(B)]])
    assert pl.teleports is not None and "teleports" in pl.imagine_options(0)
    out = _run_right(pl, 96, x=10 * T)                                     # between B and A
    J, f = out["jumps"], out["step"]
    assert J is not None
    for e in range(J.at.shape[0]):
        cols = np.nonzero(J.at[e, 0])[0]
        assert len(cols) >= 2                                              # it came back to A and went again
        frames = (cols - 1) * f + J.frame[e, 0, cols]
        assert (np.diff(frames) >= TELEPORT_COOLDOWN).all()                # never sooner than the cooldown
        for k in cols:
            assert tuple(J.exit[e, 0, k]) == B                             # always out of B: never back from B
            ent = J.entry[e, 0, k]
            assert teleport_inside(ent[0] - A[0], ent[1] - A[1])
            rest = f - J.frame[e, 0, k]
            # exit + the row's velocity (units / frame, kept) x the frames left in the step
            assert out["x"][e, 0, k] == pytest.approx(B[0] + out["vx"][e, 0, k] * rest, abs=1e-3)
            assert 2.0 < out["vx"][e, 0, k] < 8.0                         # the velocity, not the jump (-140 / f)
            assert out["x"][e, 0, k - 1] < A[0] < out["x"][e, 0, k - 1] + 40.0
    assert out["tp_ready"].shape == out["x"].shape and out["tp_off"].shape == out["x"].shape


@needs_model
def test_no_teleport_again_within_the_cooldown():
    A, B = (15 * T, 46.0), (13 * T, 46.0)       # 40 units apart: back inside A well before 18 frames
    _, pl = _flat_portal([[list(A), list(B)]])
    out = _run_right(pl, 48, x=13.5 * T)
    J, f = out["jumps"], out["step"]
    assert J is not None
    for e in range(J.at.shape[0]):
        cols = np.nonzero(J.at[e, 0])[0]
        assert len(cols) == 1                                              # one teleport only
        t = (cols[0] - 1) * f + J.frame[e, 0, cols[0]]
        back = [k for k in range(cols[0] + 1, J.at.shape[2]) if k * f - t < TELEPORT_COOLDOWN
                and teleport_inside(out["x"][e, 0, k] - A[0], out["y"][e, 0, k] - A[1])]
        assert back                                                        # inside A again, still cooling down
        assert out["x"][e, 0, -1] > A[0] + 15.3                            # and through it to the right
    # cooling down from a real teleport (Planner.tp_ready): no teleport into A at once
    pl._tp_last[0] = 0
    assert pl.imagine_options(0, frame=4, agent=0)["tp_ready"] == TELEPORT_COOLDOWN - 4
    assert pl.imagine_options(0, frame=30, agent=0)["tp_ready"] == 0


@needs_model
def test_a_start_inside_an_exit_is_kept_off_it_until_it_leaves():
    A, B = (15 * T, 46.0), (8 * T, 46.0)
    _, pl = _flat_portal([[list(A), list(B)]])
    out = _run_right(pl, 16, x=B[0])                                       # standing on B: just came out
    assert out["jumps"] is None                                            # not sent back to A
    off = out["tp_off"][0, 0]
    assert off[0] == 1 and (off[-1] == -1)                                 # kept off end 1 (B) until it left it


@needs_model
def test_levels_without_portals_roll_out_as_before():
    from plan.planner import imagine, state_row
    from wm.rollout import World
    _, pl = _flat_portal([])
    _, far = _flat_portal([[[50 * T, 300.0], [55 * T, 300.0]]])           # a portal it never comes near
    assert pl.teleports is None and "teleports" not in pl.imagine_options(0)
    row = state_row(5 * T, 46.0, 0.0, 0.0, 0, 5, 0, 0)
    acts = np.random.default_rng(3).integers(0, 10, (6, 48))
    a = imagine(pl.model, slice(None), pl.geom, np.repeat(row, 6), acts, 16.667, [World()] * 48,
                **pl.imagine_options(0, frame=0))
    b = imagine(far.model, slice(None), far.geom, np.repeat(row, 6), acts, 16.667, [World()] * 48,
                **far.imagine_options(0, frame=0))
    assert a["jumps"] is None and b["jumps"] is None and "tp_ready" not in a
    for k in ("x", "y", "vx", "vy", "alive", "finish"):
        assert np.array_equal(a[k], b[k]), k
    assert np.array_equal(a["last"], b["last"])


@needs_model
def test_plan_through_a_portal_steps_at_the_teleport():
    from wm.rollout import World
    from plan.planner import state_row
    A, B = (15 * T, 46.0), (50 * T, 46.0)       # the goal (55 T) is just past B: the plan goes through A
    _, pl = _flat_portal([[list(A), list(B)]])
    pl.new_wave()
    fb = S.FrameBoxes(np.zeros((0, 5)), np.zeros(0, np.int8), [])
    H = pl.future_frames
    plan = pl.plan(state_row(13 * T, 46.0, 390.0, 0.0, 0, 5, 0, 0), 100, [World()] * H, [None] + [fb] * H)
    p = np.asarray(plan.path)
    d = np.hypot(*np.diff(p, axis=0).T)
    k = np.nonzero(d > TELEPORT_JUMP)[0]
    assert len(k) == 1                                                     # one teleport, on one frame
    k = int(k[0]) + 1
    assert np.hypot(*(p[k] - np.array(B))) < 12.0                          # at the exit on its frame
    assert p[k - 1][0] < A[0] and np.hypot(*(p[k - 1] - np.array(A))) < 25.0
    assert (d[d <= TELEPORT_JUMP] < 12.0).all()                            # no frame on the line across
    assert plan.veto is None


def test_commit_surprise_follows_a_predicted_teleport():
    from types import SimpleNamespace
    from plan.commit import Committer

    class Stub:                                  # what Committer.decided / triggers ask of the planner
        key_pickup, f, bounds, teleports = False, 1, None, None

        def support(self, xs, ys, keys=0):
            n = np.size(xs)
            return np.ones(n, bool), np.ones(n, bool), np.zeros(n, np.int64)

    path = [[100.0 + 5 * k, 46.0] for k in range(10)] + [[600.0 + 5 * k, 46.0] for k in range(39)]
    straight = [[100.0 + 5 * k, 46.0] for k in range(49)]
    plan = lambda pth: SimpleNamespace(frame=0, actions=[0] * 48, first_move_frames=48, trial_frames=0,
                                       goal_touch=None, key_touches=[], path=pth, state={"keys": 0})
    surprise = lambda cm, k, x, y: any(t.startswith("surprise") for t in cm.triggers(0, k, x, y, planner=Stub()))
    cm = Committer(replan=100)
    cm.decided(0, plan(path), 0, None, Stub())
    assert not any(surprise(cm, k, *path[k]) for k in range(1, 30))       # the game teleports as predicted
    assert surprise(cm, 10, 150.0, 46.0)                                  # it did not teleport: surprise
    cm.decided(0, plan(straight), 0, None, Stub())
    assert surprise(cm, 10, 600.0, 46.0)                                  # an unpredicted teleport: surprise


# ---- paths ----------------------------------------------------------------------------------------------
def test_frame_path_steps_at_the_teleport_frame():
    J = Jumps.empty((2,))
    J.at[1], J.frame[1] = True, 2
    J.entry[1], J.exit[1] = (6.0, 0.0), (300.0, 0.0)
    x, y = frame_path(np.array([0.0, 310.0]), np.array([0.0, 0.0]), 4, J)
    assert x.tolist() == pytest.approx([0.0, 3.0, 300.0, 305.0, 310.0]) and not y.any()
    x, _ = frame_path(np.array([0.0, 8.0]), np.array([0.0, 0.0]), 4)
    assert x.tolist() == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0])           # no jump: linear


def test_member_mean_takes_the_side_most_members_are_on():
    xs = np.array([[[0.0, 300.0, 310.0]], [[0.0, 300.0, 312.0]], [[0.0, 20.0, 40.0]]])    # 2 of 3 went through
    J = Jumps.empty(xs.shape)
    J.at[:2, 0, 1], J.frame[:2, 0, 1] = True, 2
    J.exit[:2, 0, 1] = (300.0, 0.0)
    mx, my, jm = member_mean(xs, np.zeros_like(xs), J)
    assert mx[0].tolist() == pytest.approx([0.0, 300.0, 311.0])
    assert jm.at[0].tolist() == [False, True, False] and jm.frame[0, 1] == 2
    assert tuple(jm.exit[0, 1]) == (300.0, 0.0)
    mx, _, jm = member_mean(xs, np.zeros_like(xs), None)
    assert jm is None and mx[0, 1] == pytest.approx(620.0 / 3)


# ---- safety ---------------------------------------------------------------------------------------------
def _fb(*boxes):
    return S.FrameBoxes(np.array(boxes, float), np.zeros(len(boxes), np.int8), [f"h{k}" for k in range(len(boxes))])


def _jump_at_1(entry=(10.0, 0.0), exit_=(300.0, 0.0), frame=2):
    J = Jumps.empty((1, 1, 2))
    J.at[0, 0, 1], J.frame[0, 0, 1] = True, frame
    J.entry[0, 0, 1], J.exit[0, 0, 1] = entry, exit_
    return J


def test_sweep_never_tests_the_line_across_a_teleport():
    xs, ys = np.array([[[0.0, 310.0]]]), np.zeros((1, 1, 2))
    between = _fb((150.0, 0.0, 5.0, 5.0, 0.0))
    assert S.sweep(xs, ys, [None, between]).contact_frame[0, 0] == 1       # the straight line hits it
    sw = S.sweep(xs, ys, [None, between], jumps=_jump_at_1())
    assert sw.contact_frame[0, 0] == -1 and sw.step_clear[0, 0, 0] > 100.0
    at_exit = _fb((305.0, 0.0, 2.0, 2.0, 0.0))                             # a real hazard after the exit counts
    assert S.sweep(xs, ys, [None, at_exit], jumps=_jump_at_1()).contact_frame[0, 0] == 1
    before = _fb((5.0, 0.0, 1.0, 1.0, 0.0))                                # ... and on the way into the portal
    assert S.sweep(xs, ys, [None, before], jumps=_jump_at_1()).contact_frame[0, 0] == 1
    # a recorded one-frame jump (entry / exit unknown): its end only
    J = teleport_jumps(xs, ys)
    assert J is not None and J.at[0, 0, 1] and np.isnan(J.exit).all()
    assert S.sweep(xs, ys, [None, between], jumps=J).contact_frame[0, 0] == -1
    assert S.check(xs, ys, [None, between], jumps=J).veto.tolist() == [False]


def test_goal_contact_is_not_found_across_a_teleport():
    xs, ys = np.array([[[0.0, 310.0]]]), np.zeros((1, 1, 2))
    goal = np.array([[150.0, 0.0, 5.0, 5.0, 0.0]])
    assert S.first_contact(xs, ys, goal)[0][0, 0] == 1
    assert S.first_contact(xs, ys, goal, jumps=_jump_at_1())[0][0, 0] == -1
    assert S.first_contact(xs, ys, np.array([[305.0, 0.0, 2.0, 2.0, 0.0]]), jumps=_jump_at_1())[0][0, 0] == 1


def test_commit_path_check_skips_a_recorded_teleport_on_portal_levels():
    xs = np.r_[np.arange(0.0, 20.0, 2.0), np.arange(290.0, 310.0, 2.0)]     # one frame of 272 units
    ys = np.zeros_like(xs)
    frames = [None] + [_fb((150.0, 0.0, 5.0, 5.0, 0.0))] * (len(xs) - 1)
    assert path_check(xs, ys, frames)[0] == 10                             # the line across hits it
    assert path_check(xs, ys, frames, portals=True)[0] == -1
    assert teleport_jumps(np.arange(0.0, 50.0, 5.0), np.zeros(10)) is None   # legal moves are not teleports


# ---- training samples -----------------------------------------------------------------------------------
def _source(folder, portals):
    from wm.buffer import ROW
    folder.mkdir(parents=True)
    (folder / "levels").mkdir()
    words = np.zeros((8, 96), np.uint32)
    words[0] = 1
    meta = {"name": "tp-level", "W": 96, "H": 8, "ox": 0, "oy": 0, "spawn": [40, 30], "goals": [[1000, 30]],
            "portals": portals}
    np.savez_compressed(folder / "levels" / "tp-level.npz", words=words, meta=json.dumps(meta))
    r = np.zeros(24, ROW)
    r["wave"], r["frame"], r["agent"] = 1, np.arange(24), 0
    r["x"], r["y"], r["vx"] = 40.0 + 2 * np.arange(24), 30.0, 120.0
    r["x"][12:] += 270.0                                                    # a teleport into frame 12
    r["flags"], r["action"] = 0x50081, 6
    r["action"][-1] = -1
    p = folder / "wave_00000.npz"
    np.savez_compressed(p, rows=r, meta=json.dumps({"level": "tp-level", "wave": 1}))
    with gzip.open(str(p)[:-4] + "_world.json.gz", "wt", encoding="utf-8") as f:
        json.dump([{"f": k, "dt": 16.667, "hz": [], "pf": []} for k in range(24)], f)
    return str(folder)


@pytest.mark.parametrize("step", [1, 4])
def test_training_drops_samples_through_a_portal(tmp_path, step):
    from wm.features import build
    with_p = _source(tmp_path / "with", [[[62.0, 30.0], [332.0, 30.0]]])
    without = _source(tmp_path / "without", [])
    counts, lines = {}, []
    a = build([with_p], step=step, log=lines.append, counts=counts)
    b = build([without], step=step, log=lambda *x: None)
    dropped = 1 if step == 1 else 4                                         # the samples that contain frame 11 -> 12
    assert len(b["X"]) - len(a["X"]) == dropped and counts == {"teleport_dropped": dropped}
    assert any("game portal" in t for t in lines)
    assert (np.abs(a["raw"][:, 0]) < TELEPORT_JUMP).all() and (np.abs(b["raw"][:, 0]) > 200).any()
