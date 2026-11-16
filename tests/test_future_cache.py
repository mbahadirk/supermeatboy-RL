"""mpc.future's frame cache (FRAME_CACHE: each recorded wave frame worked out once per wave) changes no output:
future_inputs with the cache equals future_inputs recomputing everything, value for value and dtype for dtype, on
every frame of synthetic waves that hit each case where a recorded frame's output depends on more than its wave
frame (the asking frame's platforms, kill shape types and pieces kept, dt, fans, the death memory)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from mpc import future as F                              # noqa: E402
from plan import safety as S                             # noqa: E402
from plan.planner import future_inputs                   # noqa: E402

LEVEL = "T-1"
SAW_P = 24                                               # frames: the saw's own period (>= F.PERIOD_MIN)


def rows(f: int, wave: int):
    """One frame's (hazard rows, platforms, flying shots) (realgame.js rows) of a synthetic level:
    h0 a saw on a 24-frame circuit (a circle kill shape: its own period, looked up after a mismatch), h1 a block
    without kill shapes (shifted 5 units from frame 40 of wave 2: the mismatch), h2 decoration (dropped unless
    keep says otherwise), h3 an animated piece with one or two kill boxes (before frame 25: kill_shape changes, h3#1
    comes and goes), h4 a block drifting at 0.5 units a frame (no period: extrapolated with its velocity);
    one platform (5 units off the recording in wave 2: not part of the trust check, so the step-1 World differs);
    a launcher firing every 20 frames (in wave 3 it stops at frame 30: the shots are no longer trusted, and its
    predicted next shot, at 40, is the last: the recorded ones after it must not come back)."""
    a = 2 * np.pi * (f % SAW_P) / SAW_P
    saw = [100.0 + 40 * np.cos(a), 60.0 + 20 * np.sin(a)]
    hz = [[saw[0], saw[1], 30.0, 30.0, 1.0, 1.0, 0.0, [[1, saw[0], saw[1], 12.0, 12.0, 0]]],
          [200.0 + (5.0 if wave == 2 and f >= 40 else 0.0), 20.0, 40.0, 20.0, 1.0, 1.0, 0.0],
          [300.0, 80.0, 10.0, 10.0, 1.0, 1.0, 0.0]]
    shapes = [[2, 400.0, 40.0 + f % 7, 8.0, 4.0, 0.1 * (f % 5)]]
    if f % 10 < 5 or f >= 25:                           # from frame 25 kill_shape stays (until keep changes)
        shapes.append([2, 410.0, 40.0, 6.0, 6.0, 0.0])
    hz.append([405.0, 40.0, 20.0, 20.0, 1.0, 1.0, 0.0, shapes])
    hz.append([600.0 + 0.5 * f, 30.0, 16.0, 16.0, 1.0, 1.0, 0.0])
    pf = [[500.0 + 2.0 * f + (5.0 if wave == 2 else 0.0), 100.0, 0]]
    sh = []
    for b in range(0, f + 1, 20):
        age = f - b
        if age < 12:
            sh.append([7 + b // 20, 50.0 + 3.0 * age, 150.0, 4.0])
    if wave == 3 and f >= 30:
        sh = []
    return hz, pf, sh


WAVES = {1: 60, 2: 80, 3: 70}                            # frames per wave: 2 and 3 run past the recording (60)


def keep_of(f: int, wave: int):
    return [True, True, False, True, True] if not (wave == 3 and f >= 52) else [True] * 5


def dt_of(f: int, wave: int) -> float:
    return 17.0 if wave == 3 and 20 <= f < 25 else 16.667


def fans_of(f: int, wave: int):
    return [[600.0, 50.0, 1]] if wave == 3 and f >= 45 else None


def same_arr(a, b, what):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), what


def same(a, b, what):
    """future_inputs outputs equal: recorded steps, every FrameBoxes field, every World field."""
    (wa, fa, ra), (wb, fb, rb) = a, b
    assert ra == rb and len(fa) == len(fb) and (wa is None) == (wb is None), what
    for k, (x, y) in enumerate(zip(fa[1:], fb[1:]), start=1):
        for name in ("boxes", "kind", "shape", "grow"):
            same_arr(getattr(x, name), getattr(y, name), f"{what} k{k} {name}")
        assert list(x.ids) == list(y.ids), f"{what} k{k} ids"
    for k, (x, y) in enumerate(zip(wa or [], wb or [])):
        same_arr(x.boxes, y.boxes, f"{what} world {k} boxes")
        assert (x.pf, x.pf_prev, x.dt_prev_s, x.fans) == (y.pf, y.pf_prev, y.dt_prev_s, y.fans), f"{what} world {k}"


def call(fw, cached: bool, *args, **kw):
    old = F.FRAME_CACHE
    F.FRAME_CACHE = cached
    try:
        return future_inputs(fw, *args, **kw)
    finally:
        F.FRAME_CACHE = old


def deaths():
    """Two deaths 2 units and 90 frames apart: a static place (on at every frame) until an agent is seen alive
    there (wave 2, frame 55: more than WINDOW from both), then each death's own window."""
    return S.DeathMemory([(LEVEL, 10, 100.0, 50.0), (LEVEL, 100, 102.0, 50.0)])


def test_cached_future_inputs_equal_the_uncached_ones_on_every_frame():
    fws = {True: F.FutureWorld(), False: F.FutureWorld()}
    mem = {True: deaths(), False: deaths()}
    seen = {"recorded": 0, "extrapolated": 0, "mismatch": None, "shots_untrusted": False, "refuted": False}
    for wave, n in WAVES.items():
        if wave > 1:
            assert fws[True].new_wave() == fws[False].new_wave()
        for f in range(n):
            hz, pf, sh = rows(f, wave)
            dt = dt_of(f, wave)
            for c in (True, False):
                fws[c].observe(f, hz, pf, sh, keep_of(f, wave), fans_of(f, wave), dt)
                if wave == 2 and f == 25:                    # a death added mid-wave
                    mem[c].add(LEVEL, 32, 150.0, 30.0)
                if wave == 2 and f == 55:                    # an agent alive at the static place: refuted
                    seen["refuted"] |= mem[c].observe_alive(LEVEL, f, [101.0], [50.0])
            for H, deadly in ((12, True), (20, False), (5, False), (1, False), (24, True)):   # agent.py's calls
                a = call(fws[True], True, f, H, dt, LEVEL, mem[True], deadly_only=deadly)
                b = call(fws[False], False, f, H, dt, LEVEL, mem[False], deadly_only=deadly)
                same(a, b, f"wave {wave} frame {f} H {H} deadly {deadly}")
            rec = a[2]
            seen["recorded" if rec else "extrapolated"] += 1
            seen["mismatch"] = seen["mismatch"] or fws[True].mismatch
            seen["shots_untrusted"] |= bool(rec) and not fws[True].shots_trusted   # the hazards still recorded
            if wave == 3 and f == 35:                # recorded hazards, the launcher's next shot (born at 40) predicted
                fb = a[1][5]
                assert rec == 24 and any(i.startswith("shot") and i.endswith("+40") for i in fb.ids)
                assert "h3#1" in fb.ids and fb.shape[fb.ids.index("h3#1")] == 2      # SHAPE_GAMEBOX, recorded
    no_memory = call(fws[True], True, f, 20, 16.667), call(fws[False], False, f, 20, 16.667)
    same(*no_memory, "without a death memory")
    assert seen["recorded"] and seen["extrapolated"] and seen["mismatch"] and seen["shots_untrusted"]
    assert seen["refuted"]
    assert not fws[False]._rec                               # the uncached one kept nothing


def test_recorded_frames_are_shared_and_read_only():
    """Two windows asking for the same recorded frame get the same FrameBoxes (read-only); an added death makes a
    new one with its box."""
    fw, mem = F.FutureWorld(), S.DeathMemory()
    for f in range(60):
        fw.observe(f, *rows(f, 1), keep_of(f, 1), None, 16.667)
    fw.new_wave()
    for f in range(2):
        fw.observe(f, *rows(f, 1), keep_of(f, 1), None, 16.667)
    _, fa, ra = future_inputs(fw, 1, 12, 16.667, LEVEL, mem, deadly_only=True)
    wa, _, _ = fw.worlds(1, 12, 16.667)
    for f in (2, 3):
        fw.observe(f, *rows(f, 1), keep_of(f, 1), None, 16.667)
    _, fb, rb = future_inputs(fw, 3, 12, 16.667, LEVEL, mem, deadly_only=True)
    assert ra == rb == 12 and fb[1] is fa[3] and fb[2] is fa[4]       # wave frame 4 and 5, asked from 1 and 3
    with pytest.raises(ValueError):
        fb[1].boxes[0, 0] = 0.0
    v = mem.version
    mem.add(LEVEL, 6, 10.0, 10.0)
    assert mem.version == v + 1
    _, fc, _ = future_inputs(fw, 3, 12, 16.667, LEVEL, mem, deadly_only=True)
    assert fc[1] is not fb[1] and "death@10,10" in fc[1].ids
    wb, _, _ = fw.worlds(3, 12, 16.667)
    assert wb[3] is wa[5] and wb[1] is not wa[3]           # step 1 holds the live platforms: never kept
    assert len(fw._rec) <= 24                              # frames at or before the asking frame are dropped


def test_death_memory_version_changes_with_its_boxes():
    m = S.DeathMemory()
    v0 = m.version
    m.add(LEVEL, 10, 100.0, 50.0)
    m.add(LEVEL, 100, 102.0, 50.0)                         # a static place
    assert m.version == v0 + 2 and len(m.boxes(LEVEL, 60)) == 1
    assert not m.observe_alive(LEVEL, 20, [101.0], [50.0]) and m.version == v0 + 2   # within WINDOW: boxes unchanged
    assert m.observe_alive(LEVEL, 60, [101.0], [50.0]) and m.version == v0 + 3       # refuted
    assert len(m.boxes(LEVEL, 60)) == 0
