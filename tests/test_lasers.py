"""Homing laser eyes (plan/lasers.py) and the laser / shot records of realgame.js (game.level parse_*), no game.

The eye's aim is checked against the scalar rule of analysis/reactive_hazards_20261117/verify.py on logged frames
(runs/online_check11 4-16 wave 3, agent 0, frames 1-40, embedded below)."""
import gzip
import json
import math
import os
import sys
import shutil
import subprocess

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "analysis", "reactive_hazards_20261117"))

import verify as V                                            # noqa: E402
from game.level import SHAPE_SEGMENT, Level, parse_lz, parse_lzp, parse_shx   # noqa: E402
from mpc.future import FutureWorld                              # noqa: E402
from plan import lasers as L                                    # noqa: E402
from plan import safety as S                                    # noqa: E402
from plan.commit import path_check                              # noqa: E402
from plan.planner import future_inputs                          # noqa: E402

# 4-16 (runs/online_check11 wave_00003, agent 0): positions of frames 1..40, the eyes' placed angles and centres
LOGGED = [(-330.2167, -90.0), (-330.0, -90.1528), (-329.5667, -90.4583), (-328.9167, -90.9167), (-328.05, -91.5278),
          (-326.9667, -92.2917), (-325.6667, -93.2083), (-324.15, -93.99), (-322.4167, -93.99), (-320.4667, -93.99),
          (-318.3, -93.99), (-315.9167, -93.99), (-313.3167, -93.99), (-310.5, -94.1428), (-307.4667, -94.4483),
          (-304.2167, -94.9067), (-300.75, -95.5178), (-297.0667, -96.2817), (-293.1667, -97.1983),
          (-289.05, -98.2678), (-284.7167, -99.49), (-280.1667, -100.865), (-275.4, -102.3928), (-270.4167, -104.0733),
          (-265.2167, -105.9067), (-259.8, -107.8928), (-254.1667, -110.0317), (-248.3167, -112.3233),
          (-242.25, -114.7678), (-235.9667, -117.365), (-229.4667, -120.115), (-222.9667, -123.0178),
          (-216.4667, -126.0733), (-209.9667, -129.2817), (-203.4667, -132.6428), (-196.9667, -136.1567),
          (-190.4667, -139.8233), (-183.9667, -143.6428), (-177.4667, -147.615), (-170.9667, -151.74)]
EYES_416 = [((-169.63058471679688, -50.80848693847656), 116.50115203857422),
            ((150.12606811523438, -52.60374069213867), 40.92625045776367)]


# ---- Part 1: realgame.js records ------------------------------------------------------------------------------
LZP = {"wait_len": 1.1, "lasers": [
    {"id": "laser0x100", "piece": "0xa0", "kind": 1, "x": 100.0, "y": 100.0, "flip": 0, "angle": 3.5,
     "params": [1.0, 10.3, 0.0, 999.0, 0, 18]},
    {"id": "laser0x124", "piece": "0xb0", "kind": 0, "x": 300.0, "y": 100.0, "flip": 0, "angle": 1.0,
     "params": [1.0, 0.0, 2.0, 3.0, 7, 0]}]}


def test_parse_lasers_and_shots():
    p = parse_lzp(LZP)
    assert p["wait_len"] == pytest.approx(1.1)
    a, b = p["lasers"]
    assert (a["id"], a["piece"], a["kind"], a["flip"]) == ("laser0x100", "0xa0", 1, 0)
    assert a["params"] == [1.0, 10.3, 0.0, 999.0, 0, 18] and isinstance(a["params"][5], int)
    assert b["params"][4] == 7 and isinstance(b["params"][2], float)
    assert parse_lzp(None) is None and parse_lzp({}) is None
    short = parse_lzp({"lasers": [{"id": "laser0x1", "kind": 1, "x": 1, "y": 2, "params": [1.0]}]})
    assert short["lasers"][0]["params"] == [1.0, None, None, None, None, None] and short["wait_len"] is None
    assert parse_lz([[1, 0.5, 116.25, 0.0], [2, None, 3.0]]) == [(1, 0.5, 116.25, 0.0), (2, None, 3.0, None)]
    sh = [[17, 1.0, 2.0, 1.0], [9000, 5.0, 6.0, 10.0]]
    assert parse_shx(sh, [[5, 100.0, -50.0], [16, 300.0, 0.0]]) == [(5, 100.0, -50.0), (16, 300.0, 0.0)]
    assert parse_shx(sh, None) == [(None, None, None)] * 2 and parse_shx(sh, [[5, 1, 1]]) == [(None, None, None)] * 2


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_native_laser_and_missile_producers_with_fake_memory():
    proc = subprocess.run(["node", os.path.join(HERE, "tests", "native_lasers.cjs")],
                          capture_output=True, text=True, cwd=HERE)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _level(name="test-lasers", lzp=LZP, W=40, H=20, walls=()):
    w = np.zeros((H, W), np.uint32)
    w[0, :] = 1
    for r, c in walls:
        w[r, c] = 1
    info = {"name": name, "W": W, "H": H, "ox": -W * 10.0, "oy": -H * 10.0, "spawn": [0.0, 0.0], "goals": [],
            "objs": [], "lethalTypes": [35, 2000, 36, 2000], "lethalIds": ["0xa0", "laser0x100", "0xb0", "laser0x124"]}
    if lzp is not None:
        info["lzp"] = lzp
    return Level(info, w.astype("<u4").tobytes())


def test_level_keeps_the_lasers_parameters():
    lv = _level()
    assert lv.lzp == parse_lzp(LZP) and lv.hz_types == [35, 2000, 36, 2000]
    assert not lv.set_lasers(LZP) and not lv.set_lasers(None)
    changed = json.loads(json.dumps(LZP))
    changed["lasers"][0]["params"][5] = 9
    assert lv.set_lasers(changed) and lv.lzp["lasers"][0]["params"][5] == 9
    assert _level(lzp=None).lzp is None


def test_collector_records_lasers_and_shot_kinds(tmp_path):
    from wm.buffer import Collector
    c = Collector(str(tmp_path / "wm_data"))
    lv = _level()
    c.level(lv, {"lzp": LZP})
    meta = json.loads(str(np.load(tmp_path / "wm_data" / "levels" / "test-lasers.npz")["meta"]))
    assert meta["lzp"] == LZP
    m = {"frame": 3, "dt": 16.667, "hz": [], "sh": [[17, 1.23456, 2.0, 1.0]], "shx": [[5, 100.123456, -3.0]],
         "lz": [[1, 0.0500001, 116.501152, 0.25], [0, 0.0, 1.0, 0.0]], "lzp": LZP}
    c.frame(m, [], [])
    rec = c.world[-1]
    assert rec["lz"][0] == [1, 0.05, 116.501152, 0.25]          # 6 decimals (hz rows keep 2)
    assert rec["shx"] == [[5, 100.12, -3.0]] and rec["lzp"] == LZP
    c.frame({"frame": 4, "hz": [], "sh": []}, [], [])
    assert "lz" not in c.world[-1] and "shx" not in c.world[-1]


# ---- the eye's rules ----------------------------------------------------------------------------------------
def test_aim_step_matches_verify_rule_on_logged_frames():
    k = np.float32(1.0) / np.float32(18.0)
    for (ex, ey), th0 in EYES_416:
        th_v = V.f32(th0)
        th_m = np.array([th0], np.float32)
        for (x, y) in LOGGED:
            th_v = V.aim_step(th_v, (ex, ey), (x, y), V.f32(1.0) / V.f32(18))
            th_m = L.aim_step(th_m, np.float32(ex), np.float32(ey), np.float32(x), np.float32(y), k)
            assert float(th_m[0]) == float(th_v)                 # bit for bit (float32)
        assert abs(float(th_m[0]) - th0) > 0.01                  # it did turn


def test_aim_step_turns_toward_the_target_by_k():
    th = L.aim_step(np.float32(0.0), 0.0, 0.0, 0.0, 100.0, np.float32(0.1))      # target straight up
    assert float(th) == pytest.approx(0.1 * math.pi / 2, abs=1e-6)
    th = L.aim_step(np.float32(0.0), 0.0, 0.0, 0.0, -100.0, np.float32(0.1))     # straight down
    assert float(th) == pytest.approx(-0.1 * math.pi / 2, abs=1e-6)


def test_ray_cast_stops_at_the_first_solid_sample():
    H, W = 10, 20
    solid = np.zeros((H, W), bool)
    solid[:, 10] = True                                           # a wall: cells x in [100, 120)
    L0 = L.level_length(W, H)
    n = math.ceil(L0 / 10)
    s = L0 / n
    got = L.cast_lengths(5.0, 105.0, [0.0], solid, -100.0, 0.0, L0)[0]
    first = math.ceil((100.0 - 5.0) / s)                          # first sample inside the wall
    assert got == pytest.approx(first * s, abs=1e-3) and 95.0 <= got < 95.0 + 10.0
    got_back = L.cast_lengths(5.0, 105.0, [math.pi], solid, -100.0, 0.0, L0)[0]
    assert got_back == pytest.approx(L0, abs=1e-3)                # nothing that way: the whole length
    # the same against verify.py's float32 ray cast
    vx, vy = V.ray_cast((5.0, 105.0), V.f32(0.0), (W, H, -100.0, 0.0, solid))
    assert got == pytest.approx(math.hypot(vx - 5.0, vy - 105.0), abs=1e-3)


def test_kill_test_hits_and_misses():
    # a beam from (0, 0) along +x, 100 long; the character crosses it in front of the eye
    c, h = L.beam_clearance(0.0, 0.0, 0.0, 100.0, 50.0, 3.0, 50.0, 3.0)        # still, 3 units above: static cross
    assert h and c < 0
    c, h = L.beam_clearance(0.0, 0.0, 0.0, 100.0, 50.0, 7.0, 50.0, 7.0)        # 7 above: beyond the cross (6)
    assert not h and c == pytest.approx(1.0)
    c, h = L.beam_clearance(0.0, 0.0, 0.0, 100.0, 50.0, -10.0, 50.0, 10.0)     # fell through it in one frame: swept
    assert h and c < 0
    c, h = L.beam_clearance(0.0, 0.0, 0.0, 100.0, -20.0, 0.0, -20.0, 0.0)      # behind the eye
    assert not h and c == pytest.approx(20.0)
    c, h = L.beam_clearance(0.0, 0.0, 0.0, 100.0, 110.0, 0.0, 110.0, 0.0)      # past the beam's end (a wall)
    assert not h and c == pytest.approx(10.0)
    # the same as verify.py's scalar test on a grid of cases
    rng = np.random.default_rng(1)
    for _ in range(300):
        phi, ln = rng.uniform(-math.pi, math.pi), rng.uniform(20, 200)
        ax, ay = rng.uniform(-60, 220, 2)
        bx, by = ax - rng.uniform(-8, 8), ay - rng.uniform(-8, 8)
        _, h = L.beam_clearance(0.0, 0.0, phi, ln, ax, ay, bx, by)
        want = V.beam_kills((0.0, 0.0), (ln * math.cos(phi), ln * math.sin(phi)), (ax, ay), (bx, by), (ax - bx, ay - by))
        assert bool(h) == bool(want)


def _beams(theta=(math.pi,), lethal=True, ex=(200.0,), ey=(100.0,), H=60, lv=None):
    lv = lv or _level()
    tables = np.stack([L.BeamTable(x, y, lv).len for x, y in zip(ex, ey)])
    n = len(ex)
    return L.Beams(ids=[f"laser{k}" for k in range(n)], ex=np.array(ex, np.float32), ey=np.array(ey, np.float32),
                   theta=np.array(theta, np.float32), k=np.full(n, 1 / 18, np.float32), flip=np.zeros(n, bool),
                   tables=tables, lethal=np.full((n, H), lethal), frame=0, start=None)


def test_each_candidate_gets_its_own_beam():
    bm = _beams(theta=(math.pi,))                                 # the eye at (200, 100) looks left
    K = 30
    t = np.arange(K + 1)
    X = np.stack([np.full(K + 1, 100.0), np.full(K + 1, 100.0)])
    Y = np.stack([100.0 + 3.0 * t, 100.0 - 3.0 * t])              # one goes up, one down
    clear, th = bm.columns(X, Y, 1, 0, want_theta=True)
    assert clear.shape == (2, K, 1)
    assert th[0, 0] < math.pi < th[1, 0] or th[1, 0] < math.pi < th[0, 0]   # turned opposite ways
    assert th[0, 0] != th[1, 0]
    assert (clear[:, 0, 0] < 0).all()                             # both start on the beam: contact at step 1
    # a path that runs from the beam keeps ahead of it (the eye turns 1/18 of the angle a frame)
    assert (clear[0, 5:, 0] > 0).all()
    # step-4 columns: each column the least clearance of its 4 frames (the same frames as step 1)
    Xc, Yc = X[:, ::4][:, :8], Y[:, ::4][:, :8]
    c4, _ = bm.columns(Xc, Yc, 4, 0)
    per_frame = lambda C: np.stack([np.interp(np.arange(29), np.arange(8) * 4, C[r]) for r in range(2)])
    c1, _ = bm.columns(per_frame(Xc), per_frame(Yc), 1, 0)
    assert np.allclose(c4, c1.reshape(2, 7, 4, 1).min(2))


def test_beam_blocked_by_a_wall_hides_the_path():
    lv = _level(walls=[(r, 25) for r in range(1, 20)])             # a wall column at x in [100, 120)
    bm = _beams(theta=(0.0,), ex=(0.0,), ey=(100.0,), lv=lv)      # the eye at (0, 100) looks right at the wall
    X = np.array([[150.0, 150.0, 150.0]])                         # behind the wall, on the beam's line
    Y = np.array([[100.0, 100.0, 100.0]])
    clear, _ = bm.columns(X, Y, 1, 0)
    assert (clear > 0).all()
    lv2 = _level()
    bm2 = _beams(theta=(0.0,), ex=(0.0,), ey=(100.0,), lv=lv2)    # no wall: the same place is on the beam
    clear2, _ = bm2.columns(X, Y, 1, 0)
    assert (clear2 < 0).all()


def test_teleport_does_not_sweep_a_beam_across_the_gap():
    from game.level import Jumps
    bm = _beams(theta=(0.,), ex=(0.,), ey=(100.,))
    X, Y = np.array([[100., 100.]]), np.array([[50., 150.]])
    direct, _ = bm.columns(X, Y)
    assert direct[0, 0, 0] < 0
    jumps = Jumps.empty((1, 2))
    jumps.at[0, 1], jumps.frame[0, 1] = True, 1
    jumps.entry[0, 1], jumps.exit[0, 1] = (100., 50.), (100., 150.)
    jumped, _ = bm.columns(X, Y, jumps=jumps)
    assert jumped[0, 0, 0] > 0
    jumps.entry[0, 1] = (100., 100.)               # motion up to the portal still crosses the beam
    touched, _ = bm.columns(X, Y, jumps=jumps)
    assert touched[0, 0, 0] < 0


def test_occlusion_edge_is_cast_instead_of_choosing_a_long_neighbour():
    bm = _beams(theta=(0.,), ex=(0.,), ey=(100.,))
    bm.solid = np.zeros((20, 40), bool)
    bm.solid[:, 25] = True
    bm.origin, bm.length = (-400., -200.), L.level_length(40, 20)
    bm.tables[0, 0], bm.tables[0, 1] = 100., 400.
    got = bm.lengths(np.array([[[.00001]]]), np.array([0]))
    assert got[0, 0, 0] < 120                    # do not extend the beam hundreds of units through a wall


def test_lethal_schedule_from_the_recorded_state():
    lv = _level()
    tr = L.EyeTracker(lv)
    hz = [[100.0, 100.0, 16.85, 24.41, 1, 1, 3.5, []], [130.0, 100.0, 60.0, 0, 1, 1, 0.0, []],
          [300.0, 100.0, 16.85, 24.41, 1, 1, 1.0, []], [330.0, 100.0, 60.0, 0, 1, 1, 0.0, []]]
    tr.observe(10, hz, [[2, 5 / 60, 3.5, 0.0], [0, 0.0, 1.0, 0.0]], (0.0, 0.0))
    e = tr.eyes[0]
    assert [x.id for x in tr.eyes] == ["laser0x100"] and tr.beam_rows() == frozenset({1})   # plain laser kept
    s = tr.schedule(e, 10)
    assert not s.any()                                           # recorded animation length: 1.1 s still to wait
    tr.animation[e.id] = None                                    # legacy capture without animation clock
    tr.wait[e.id] = 20                                            # measured: 20 frames in wait
    s = tr.schedule(e, 30)
    assert not s[:15].any() and s[15:].all()                      # 5 frames into the wait: 15 more
    tr.observe(11, hz, [[1, 0.0, 3.5, 0.0], [0, 0.0, 1.0, 0.0]], (0.0, 0.0))
    assert tr.schedule(e, 50).all()                               # on, on time 999 s


def test_wait_schedule_uses_animation_clock_and_parameter_multiplier():
    params = json.loads(json.dumps(LZP))
    params["lasers"][0]["params"][0] = 2.0
    tr = L.EyeTracker(_level(lzp=params))
    tr.observe(100, _hz(False, False), [[2, .7, 3.5, 2.15], [0, 0, 1, 0]], (0, 0))
    s = tr.schedule(tr.eyes[0], 10)
    assert not s[:3].any() and s[4:].all()


def test_unknown_angle_keeps_the_recorded_beam_until_a_model_can_be_built():
    params = json.loads(json.dumps(LZP))
    params["lasers"][0]["piece"] = None
    tr = L.EyeTracker(_level(lzp=params))
    tr.observe(0, _hz(), None, (0, 0))
    assert not tr.beam_rows() and tr.context(0, 10) is None


def test_modelled_legacy_beam_box_is_also_removed():
    from mpc.future import _kill
    old = [[0., 0., 100., 0., 1., 1., 0.]]
    assert len(_kill(old, None)[0]) == 1
    assert not len(_kill(old, None, frozenset({0}))[0])


def test_lethal_schedule_without_state_uses_the_first_lethal_wave_frame():
    lv = _level(lzp=None)
    lv.names = ["x"] * 40
    lv.names[35], lv.names[36] = "hominglazeron", "lazeron"
    assert L.EyeTracker.candidate(lv)
    tr = L.EyeTracker(lv)
    seg = [[3, 100.0, 100.0, 160.0, 100.0, 0]]
    hz0 = [[100.0, 100.0, 16.85, 24.41, 1, 1, 0.0, []], [130.0, 100.0, 60.0, 0, 1, 1, 0.0, []],
           [300.0, 100.0, 16.85, 24.41, 1, 1, 0.0, []], [330.0, 100.0, 60.0, 0, 1, 1, 0.0, []]]
    for f in range(0, 70):
        hz = json.loads(json.dumps(hz0))
        if f >= 67:
            hz[1][7] = seg
        tr.observe(f, hz, None, (0.0, 0.0))
    e = tr.eyes[0]
    assert e.k == pytest.approx(1 / 18) and e.beam_row == 1 and tr.first_on[e.id] == 67
    tr.new_wave()
    tr.observe(0, hz0, None, (0.0, 0.0))
    s = tr.schedule(e, 80)
    assert not s[:67].any() and s[67:].all()


# ---- in the future and the sweep ------------------------------------------------------------------------------
def _hz(seg_a=True, seg_b=True):
    sa = [[3, 100.0, 100.0, 160.0, 100.0, 0]] if seg_a else []
    sb = [[3, 300.0, 100.0, 300.0, 160.0, 0]] if seg_b else []
    return [[100.0, 100.0, 16.85, 24.41, 1, 1, 3.5, []], [130.0, 100.0, 60.0, 0, 1, 1, 0.0, sa],
            [300.0, 100.0, 16.85, 24.41, 1, 1, 1.0, []], [300.0, 130.0, 60.0, 0, 1, 1, math.pi / 2, sb]]


def test_recorded_homing_segments_leave_the_deadly_boxes():
    lv = _level()
    fw = FutureWorld()
    fw.new_level()
    fw.set_lasers(L.EyeTracker.for_level(lv))
    keep = [True] * 4
    for f in range(0, 6):
        fw.observe(f, _hz(), [], [], keep, [], 16.667, lz=[[1, 1.0, 3.5, 0.0], [1, 1.0, 1.0, 0.0]], target=(0.0, 0.0))
    _, frames, _ = future_inputs(fw, 5, 8, 16.667)
    for fb in frames[1:]:
        assert fb.ids == ["h3"]                                   # the plain laser's segment only
        assert fb.beams[0].ids == ["laser0x100"]
    assert [fb.beams[1] for fb in frames[1:]] == list(range(1, 9))
    # without the tracker: both segments, no beams (as before)
    fw2 = FutureWorld()
    fw2.new_level()
    for f in range(0, 6):
        fw2.observe(f, _hz(), [], [], keep, [], 16.667)
    _, frames2, _ = future_inputs(fw2, 5, 8, 16.667)
    assert all(fb.ids == ["h1", "h3"] and fb.beams is None for fb in frames2[1:])


def test_homing_beams_do_not_break_the_recordings_match():
    lv = _level()
    fw = FutureWorld()
    fw.new_level()
    fw.set_lasers(L.EyeTracker.for_level(lv))
    keep = [True] * 4
    for f in range(0, 10):
        fw.observe(f, _hz(), [], [], keep, [], 16.667, lz=[[1, 1.0, 3.5, 0.0], [1, 1.0, 1.0, 0.0]])
    fw.new_wave()
    for f in range(0, 5):                                         # the homing beam points elsewhere this wave
        hz = _hz()
        hz[1] = [100.0, 130.0, 60.0, 0, 1, 1, math.pi / 2, [[3, 100.0, 100.0, 100.0, 160.0, 0]]]
        fw.observe(f, hz, [], [], keep, [], 16.667, lz=[[1, 1.0, 3.5, 0.0], [1, 1.0, 1.0, 0.0]])
    assert fw.trusted
    _, _, rec = fw.worlds(4, 4, 16.667, deadly_only=True)
    assert rec == 4


def test_levels_without_eyes_are_unchanged():
    lv = _level(lzp=None)                                         # no palette names: no homing eye known
    assert L.EyeTracker.for_level(lv) is None
    xs, ys = np.array([[[0.0, 50.0, 120.0]]]), np.array([[[100.0, 100.0, 100.0]]])
    fb = S.FrameBoxes(np.array([[130.0, 100.0, 30.0, 0.0, 0.0]]), np.zeros(1, np.int8), ["h1"],
                      np.array([SHAPE_SEGMENT], np.int8))
    a = S.sweep(xs, ys, [None, fb, fb])
    b = S.sweep(xs, ys, [None, fb, fb], beams=None)
    assert a.beam_theta is None and a.ids == ["h1"]
    for k in ("contact_frame", "contact_kind", "contact_box", "clear_min", "clear_frame", "step_clear"):
        assert np.array_equal(getattr(a, k), getattr(b, k))


def test_sweep_reports_laser_contacts():
    bm = _beams(theta=(math.pi,))                                 # eye (200, 100) looking left
    empty = S.FrameBoxes(np.zeros((0, 5)), np.zeros(0, np.int8), [])
    frames = [None] + [S.FrameBoxes(empty.boxes, empty.kind, [], beams=(bm, k)) for k in range(1, 21)]
    xs = np.zeros((2, 1, 21))
    ys = np.zeros((2, 1, 21))
    xs[:], ys[:] = 100.0, 100.0                                   # member 0: standing on the beam
    ys[1, 0, :] = 96.0 + 8.0 * np.arange(21)                      # member 1: crosses it, then runs away up
    xs = np.concatenate([xs, xs], 1)                              # 2 members x 2 candidates
    ys = np.concatenate([ys, ys[::-1]], 1)
    sw = S.sweep(xs, ys, frames)
    assert sw.ids == ["laser0"] and list(sw.kinds) == [S.KIND_LASER]
    assert sw.contact_frame[0, 0] == 1 and sw.contact_kind[0, 0] == S.KIND_LASER
    assert sw.contact_frame[1, 0] == 1 and sw.contact_frame[1, 1] == 1
    assert sw.beam_theta.shape == (2, 2, 1)
    saf = S.assess(sw)
    assert saf.veto.all() and saf.veto_id == ["laser0", "laser0"] and saf.veto_kind[0] == S.KIND_LASER
    assert saf.nearest[0][0]["kind"] == "laser"
    # explicit start (a later column 0): the given angles, not the frames'
    sw2 = S.sweep(xs[:, :1], ys[:, :1], frames, beams=(bm, 0, np.full((2, 1, 1), math.pi / 2, np.float32)))
    assert sw2.contact_frame[0, 0] == -1                          # the beam points up, away from (100, 100)
    # taken frames keep the eyes
    assert frames[3].take([]).beams == (bm, 3)


def test_commit_path_check_sees_the_beam():
    bm = _beams(theta=(math.pi,))
    empty = S.FrameBoxes(np.zeros((0, 5)), np.zeros(0, np.int8), [])
    frames = [None] + [S.FrameBoxes(empty.boxes, empty.kind, [], beams=(bm, k)) for k in range(1, 30)]
    cf, hid, _ = path_check(np.full(25, 100.0), np.full(25, 100.0), frames)
    assert cf == 1 and hid == "laser0"
    cf, hid, risk = path_check(np.full(25, 100.0), 96.0 + 8.0 * np.arange(25), frames)
    assert cf == 1                                                 # frame 0 -> 1 crosses the beam
    cf, hid, risk = path_check(np.full(25, 100.0), 140.0 + 8.0 * np.arange(25), frames)
    assert cf == -1 and hid is None


# ---- the planner with an eye (the world model of tests/test_planner.py) ---------------------------------------
MODEL = os.path.join(HERE, "runs", "wm_locked", "wm_model.pt")


@pytest.mark.skipif(not os.path.exists(MODEL), reason="no world model (runs/wm_locked)")
def test_planner_judges_candidates_by_their_own_beams():
    from dataclasses import replace
    from game.level import Level
    from plan.planner import Planner, state_row
    from plan.progress import ProgressField
    from wm.rollout import World
    T = 20.0
    H, W = 20, 60
    w = np.zeros((H, W), np.uint32)
    w[0:2, :] = 1
    info = {"name": "test-flat", "W": W, "H": H, "ox": 0.0, "oy": 0.0, "spawn": [10 * T + 10, 46.0],
            "goals": [[50 * T + 10, 50.0]], "objs": [], "lethalTypes": []}
    lv = Level(info, w.astype("<u4").tobytes())
    field = ProgressField(w, 0.0, 0.0, lv.spawn, lv.goals, [], [], [], [], [])
    pl = Planner(MODEL, lv, field, budget_ms=None)
    x0 = 10 * T + 10
    bm = _beams(theta=(math.pi / 2,), ex=(x0 + 100.0,), ey=(60.0,), H=200, lv=lv)   # up, right of the agent
    empty = S.FrameBoxes(np.zeros((0, 5)), np.zeros(0, np.int8), [])
    n = pl.future_frames
    frames = [None] + [replace(empty, beams=(bm, k)) for k in range(1, n + 1)]
    row = state_row(x0, 46.0, 0.0, 0.0, 0, 0)
    plan = pl.plan(row, 200, [World() for _ in range(n)], frames)
    near = [d for d in plan.nearest if d["kind"] == "laser"]
    assert near and near[0]["id"] == "laser0"
    assert plan.veto is None
    # every path the planner keeps stays off this eye's own beam in its first frames
    xs = np.array([p[0] for p in plan.path])[None]
    ys = np.array([p[1] for p in plan.path])[None]
    c, _ = bm.columns(xs[:, :17], ys[:, :17], 1, 0)
    assert (c > 0).all()
