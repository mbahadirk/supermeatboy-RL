"""The viewer's progress line (P): plan.progress.descent_path on a hand-made field, the level message's "progress"
entry through telemetry, and an off-screen render of all three P states, live and from a run's fields folder."""
import os
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from plan.progress import ProgressField, best_times, descent_path, leg_of_keys   # noqa: E402
from viz import telemetry as T                                                  # noqa: E402
from viz import viewer as V                                                     # noqa: E402

TL = 20.0
H, W = 22, 40


def at(c, r):
    return ((c + 0.5) * TL, (r + 0.5) * TL)


def terrain():
    w = np.zeros((H, W), np.uint32)
    w[0:2, :] = 1
    w[2:9, 10:] = 1          # a plateau, its face 7 tiles high
    w[9:11, 25] = 1          # a low wall on it
    return w


def test_descent_falls_to_the_goal_and_through_the_keys():
    pf = ProgressField(terrain(), 0.0, 0.0, at(15, 9), [at(35, 9)], [], [], [], [], [])
    tb = best_times(pf)
    line = descent_path(tb, 0.0, 0.0, *at(15, 9))
    ts = [t for _, _, t in line]
    assert all(a > b for a, b in zip(ts, ts[1:])) and ts[-1] == 0.0
    assert line[-1][:2] == at(35, 9)
    assert any(abs(x - at(25, 9)[0]) < 1 and y > at(25, 10)[1] for x, y, _ in line)   # over the low wall
    assert descent_path(tb, 0.0, 0.0, *at(15, 8)) != []          # in the plateau's top row: from a neighbour
    assert descent_path(tb, 0.0, 0.0, *at(15, 5)) == []          # deep in it: no open neighbour
    assert descent_path(tb, 0.0, 0.0, -50.0, 10.0) == []         # outside the map
    # a key: the line goes to it first, then on to the goal
    pk = ProgressField(terrain(), 0.0, 0.0, at(15, 9), [at(35, 9)], [], [], [at(5, 2)], [], [])
    tk = best_times(pk)
    assert tk.shape[0] == 2 and leg_of_keys(pk.key_order, 0) == 0 and leg_of_keys(pk.key_order, 1) == 1
    lk = descent_path(tk, 0.0, 0.0, *at(15, 9), leg=0)
    pts = [p[:2] for p in lk]
    assert at(5, 2) in pts and pts[-1] == at(35, 9) and pts.index(at(5, 2)) < len(pts) - 1
    l1 = descent_path(tk, 0.0, 0.0, *at(15, 9), leg=1)              # the key held: straight to the goal
    assert at(5, 2) not in [p[:2] for p in l1] and l1[-1][:2] == at(35, 9)


def _level_msg(pf, tb, big=False):
    from types import SimpleNamespace
    lv = SimpleNamespace(name="P-1", W=W, H=H, ox=0.0, oy=0.0, kill_box=None, alive_box=lambda: [-10, W * TL + 10, -60, H * TL + 10],
                         words=terrain(), solid=(terrain() > 0).astype(np.uint8), goals=[at(35, 9)], spawn=at(15, 9),
                         keys=[], bandages=[], portals=[])
    return T.level_message(lv, 1, 1, 0, pf.summary(), progress=T.progress_payload(tb, pf.route_cells, pf.key_order, 3))


def _render(src_scene, mode):
    import pygame
    r = V.Renderer()
    r.show_progress = mode
    surf = pygame.Surface((1200, 700))
    r.render(surf, src_scene)
    return r, surf


def test_live_progress_line_renders_in_all_p_states():
    pf = ProgressField(terrain(), 0.0, 0.0, at(15, 9), [at(35, 9)], [], [], [], [], [])
    tb = best_times(pf)
    msg = _level_msg(pf, tb)
    data = T.Telemetry(None).fit(msg)
    src = V.LiveSource(bind=False)
    src.datagram(data)
    x, y = at(15, 9)
    src.handle(T.decode(T.Telemetry(None).fit(T.frame_message("P-1", 1, 0, [
        {"i": 0, "role": "planner", "x": x, "y": y, "vx": 0.0, "vy": 0.0, "status": 0, "keys": 0}], [], {}, []))))
    sc = src.scene()
    pr = sc.level.progress
    assert pr is not None and pr["id"] == 3 and pr["T"].shape == (1, H, W)
    fin = np.isfinite(tb)
    assert (np.isfinite(pr["T"]) == fin).all() and np.allclose(pr["T"][fin], tb[fin], rtol=1e-3, atol=0.05)
    for mode in (0, 1, 2):
        r, _ = _render(sc, mode)
        if mode:
            assert r.progress_note.startswith("progress: ") and "frames to the goal" in r.progress_note
    # a too big message drops "progress" after "words"
    shrunk = list(T._shrink(msg))
    assert shrunk[-1]["progress"] is None and shrunk[-1]["words"] is None


def test_replay_reads_the_runs_fields_folder(tmp_path):
    pf = ProgressField(terrain(), 0.0, 0.0, at(15, 9), [at(35, 9)], [], [], [], [], [])
    tb = best_times(pf)
    d = tmp_path / "fields"
    d.mkdir()
    import json
    np.savez_compressed(d / "P-1.npz", T=tb.astype(np.float16), route=np.asarray(pf.route_cells).reshape(-1, 2),
                        key_order=json.dumps(pf.key_order), ox=0.0, oy=0.0)
    pr = V.load_progress(str(d / "P-1.npz"), H, W)
    assert pr is not None and pr["route"][0] == tuple(pf.route_cells[0])
    assert V.load_progress(str(d / "P-1.npz"), H + 1, W) is None        # another map: not used
