"""Checks for the planner viewer (viz/viewer.py), headless: a replay of a tiny synthetic run folder rendered to an
off-screen surface, telemetry datagrams from the sender's encoder parsed by the live listener, and the
prediction-vs-reality matcher."""
import gzip
import json
import math
import os
import socket
import sys
import time
from types import SimpleNamespace

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from viz import telemetry as T                              # noqa: E402
from viz import viewer as V                                 # noqa: E402

LEVEL = "T-1"
W, H, OX, OY = 12, 8, -100.0, -60.0


def _plan(frame, agent=0, x0=-60.0, y0=-30.0):
    path = [[x0 + 2.0 * k, y0 + (0.5 * k if k < 10 else 5.0)] for k in range(49)]
    return {"frame": frame, "agent": agent, "state": {"x": x0, "y": y0}, "chosen": ["run R 24f", "wall-jump L hold 8"],
            "actions": [6] * 48, "pad": 2, "from_prev": False, "path": path,
            "spread": [0.5 * k for k in range(49)],                    # over 20 units from frame 41: dashed
            "landings": [{"frame": 10, "x": path[10][0], "y": path[10][1], "on": "ground"}],
            "wall_jumps": [{"frame": 30, "x": path[30][0], "y": path[30][1], "side": "R"}],
            "nearest": [{"id": "h0", "kind": "hazard", "clearance": 12.5, "frame": 5},
                        {"id": "out", "kind": "out of level", "clearance": 40.0, "frame": 20},
                        {"id": "death@-20,-20", "kind": "remembered death", "clearance": 30.0, "frame": 8}],
            "veto": None, "soft_risk": 0.1, "soft_frame": 12,
            "score": {"progress": 3.2, "progress_credit": 3.0, "risk": 0.1, "uncertainty": 0.9, "spread_end": 24.0,
                      "failure": 0.0, "total": 2.9}, "v_start": 50.0, "v_end": 46.8, "dead_end_members": 0, "leg": 0,
            "end_supported": True,
            "alternatives": [{"moves": ["wait 8f"], "total": -1.0, "progress": 0.0, "stage": "final",
                              "reason": "no progress", "path": [[x0, y0], [x0, y0 + 1]], "vetoed": False},
                             {"moves": ["run L 16f"], "total": -100.0, "progress": -1.0, "stage": "final",
                              "reason": "vetoed: hazard h0 at frame 3", "path": [[x0, y0], [x0 - 20, y0]],
                              "vetoed": True}],
            "predictions": {str(h): path[h] for h in (1, 4, 8, 16)}, "death_model": 0.1,
            "timing": {"stage1": 20.0, "stage2": 30.0, "final": 50.0, "total": 100.0 + frame},
            "counts": {"stage1": {"candidates": 73, "steps": 24, "complete": 73, "vetoed": 0},
                       "stage2": {"candidates": 128, "steps": 10, "of_steps": 28, "complete": 0, "vetoed": 0},
                       "final": {"candidates": 17, "members": 5, "steps": 48, "vetoed": 1}},
            "segments": [["run R 24f", 0, 24], ["wall-jump L hold 8", 24, 46], ["pad", 46, 48]],
            "first_move_frames": 24, "level": LEVEL, "wave": 1,
            "real": {"x": x0, "y": y0, "vx": 0.0, "vy": 0.0, "flags": 0, "state": 0, "status": 0, "keys": 0}}


def _words():
    w = np.zeros((H, W), "<u4")
    w[0, :] = 1                      # floor
    w[:, 0] = 1                      # left wall
    w[3, 5] = 1 | (1 << 8)           # a crumble block
    return w


def _write_run(d):
    """A run folder: 2 decisions (frames 0 and 2) of agent 0, 20 logged frames, a surprise and an unexplained death;
    wm_data with the level map and one wave's world records (a hazard)."""
    os.makedirs(os.path.join(d, "wm_data", "levels"))
    meta = {"name": LEVEL, "W": W, "H": H, "ox": OX, "oy": OY, "spawn": [-60.0, -30.0], "goals": [[90.0, -30.0]],
            "keys": [], "bandages": [], "portals": [], "objs": [], "lethalTypes": [3], "kb": None}
    np.savez_compressed(os.path.join(d, "wm_data", "levels", f"{LEVEL}.npz"), words=_words().astype(np.uint32),
                        meta=json.dumps(meta))
    np.savez_compressed(os.path.join(d, "wm_data", "wave_00000.npz"), rows=np.zeros(0),
                        meta=json.dumps({"level": LEVEL, "wave": 1, "frames": 20}))
    world = [{"f": f, "dt": 16.667, "hz": [[-20.0 + f, -40.0, 20.0, 20.0, 1.0, 1.0, 0.3]], "sh": []} for f in range(40)]
    with gzip.open(os.path.join(d, "wm_data", "wave_00000_world.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(world, f)
    real = {f: (-60.0 + 2.0 * f + (3.0 if f >= 4 else 0.0), -30.0 + 0.5 * f) for f in range(20)}
    with open(os.path.join(d, "states.jsonl"), "w", encoding="utf-8") as f:
        for fr in range(20):
            st = 1 if fr == 19 else 0
            f.write(json.dumps({"frame": fr, "wave": 1, "level": LEVEL,
                                "agents": [[0, real[fr][0], real[fr][1], 120.0, 30.0, st]]}) + "\n")
    with open(os.path.join(d, "plans.jsonl"), "w", encoding="utf-8") as f:
        for fr in (0, 2):
            f.write(json.dumps(_plan(fr, x0=real[fr][0], y0=real[fr][1])) + "\n")
    events = [{"time": 1.0, "level": LEVEL, "wave": 0, "kind": "level", "frame": None, "name": LEVEL, "W": W, "H": H,
               "kill_box": [-110.0, 150.0, -100.0, 120.0], "protected": False, "agents": 1},
              {"time": 1.1, "level": LEVEL, "wave": 1, "kind": "position", "frame": 6, "agent": 0, "plan_frame": 2,
               "horizon": 4, "predicted": [-48.0, -27.0], "real": [-45.0, -27.0], "error": 3.0, "threshold": 3.0,
               "plan": "run R 24f > wall-jump L hold 8 (+2f)"},
              {"time": 1.2, "level": LEVEL, "wave": 1, "kind": "death", "frame": 19, "agent": 0, "x": real[19][0],
               "y": real[19][1], "explained": False, "cause": "none", "estimated": False, "memory_added": True},
              {"time": 1.3, "level": LEVEL, "wave": 1, "kind": "wave", "frame": 20, "why": "done", "finished": 0,
               "died": 1}]
    with open(os.path.join(d, "events.jsonl"), "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        f.write('{"half written')                     # a run still writing: the last line is cut
    return real


def _surface():
    import pygame
    pygame.init()
    return pygame.Surface((1300, 800))


def test_replay_of_a_synthetic_run_renders(tmp_path):
    run = tmp_path / "runs" / "plan_20261103-120000"
    real = _write_run(str(run))
    src = V.ReplaySource(str(run), runs_dir=str(tmp_path / "runs"))
    assert src.n == 20 and len(src.plans) == 2 and src.planned == [0]
    assert src.plan_step == [0, 2]                                   # each decision at its frame's step
    assert src.event_step[1] == 6 and src.event_step[2] == 19 and src.event_step[3] == 19
    surf = _surface()
    r = V.Renderer()
    for idx in (0, 3, 6, 10, 19):
        sc = src.scene(idx)
        assert sc.level is not None and sc.level.name == LEVEL and sc.level.kill_box == (-110.0, 150.0, -100.0, 120.0)
        assert sc.frame == idx and sc.plans[0]["frame"] == (0 if idx < 2 else 2)
        assert len(sc.hazards) == 1 and sc.hazards[0][5] == V.KIND_HAZARD
        assert set(sc.future) == {"8", "16"}
        r.render(surf, sc)
    # the remembered death becomes a box once it happened, and its wave frames are +-30 around it
    assert not src.scene(10).memory and len(src.scene(19).memory) == 1
    # alternatives drawn (and hover hit-tested), zoom / follow, help overlay, toggles: all render
    sc = src.scene(10)
    r.render(surf, sc)
    assert len(r.alt_hits) == 2
    pts, alt = r.alt_hits[1]
    r.mouse = (int(pts[1][0]), int(pts[1][1]))
    r.show_help, r.follow, r.zoom = True, True, 4.0
    r.render(surf, sc)
    r.show_alts = r.show_future = False
    r.render(surf, sc)
    # navigation: next event, next surprise / unexplained death
    assert src.next_event(0) == 6 and src.next_event(6) == 19
    assert src.next_event(0, V.is_surprise_or_unexplained) == 6 and src.next_event(19) is None
    # running stats at the end: 2 decisions, 1 surprise, 1 unexplained death
    st = src.scene(19).stats
    assert st["decisions"] == 2 and st["surprises"] == 1 and st["unexplained"] == 1
    assert st["ms_mean"] == pytest.approx(101.0) and st["surprises_per_100"] == pytest.approx(50.0)
    # the matcher with the replay's real positions: plan 2's +4 is frame 6, real is off by 3 units in x there
    m = V.matured_predictions(sc.history[0], lambda f: sc.real_at(0, f), 6)
    p4 = next(x for x in m if x["h"] == 4)
    assert p4["plan_frame"] == 2 and p4["frame"] == 6 and p4["error"] == pytest.approx(3.0, abs=1e-6)
    # latest: the newest plan_* folder
    os.makedirs(tmp_path / "runs" / "plan_20261102-090000")
    assert V.resolve_run("latest", str(tmp_path / "runs")) == str(run)


def test_replay_without_map_or_world_still_renders(tmp_path):
    run = tmp_path / "plan_x"
    _write_run(str(run))
    import shutil
    shutil.rmtree(run / "wm_data")
    os.remove(run / "states.jsonl")                                  # no state log: one step per decision
    src = V.ReplaySource(str(run), runs_dir=str(tmp_path / "none"))
    assert src.n == 2
    sc = src.scene(1)
    assert sc.level is None and sc.hazards == [] and any("no map" in s for s in sc.status)
    V.Renderer().render(_surface(), sc)


class _Lv(SimpleNamespace):
    def alive_box(self):
        return (OX - 10, OX + W * 20 + 10, OY - 60, OY + H * 20 + 10)


def _datagrams():
    from game.level import tile_solid
    words = _words()
    lv = _Lv(name=LEVEL, W=W, H=H, ox=OX, oy=OY, kill_box=None, words=words, solid=tile_solid(words),
             goals=[(90.0, -30.0)], spawn=(-60.0, -30.0), keys=[(0.0, 0.0)], bandages=[], portals=[])
    tel = T.Telemetry(None)                                   # the sender's encoder and shrink rules
    msgs = [T.level_message(lv, 1, 1, 2, "clear")]
    for f in range(10):
        agents = [{"i": 0, "role": "planner", "x": -60.0 + 2 * f + (5.0 if f == 8 else 0.0), "y": -30.0,
                   "vx": 120.0, "vy": 0.0, "status": 0, "flags": 0, "state": 0, "keys": 0, "action": 6}] + \
                 [{"i": i, "role": "explore", "x": -50.0 + i, "y": -30.0, "vx": -60.0, "vy": 0.0, "status": 0,
                   "flags": 0, "state": 0, "keys": 0, "action": 1} for i in (1, 2)]
        hz = T.box_rows([(-20.0 + f, -40.0, 10.0, 10.0, 0.3)], 0)
        msgs.append(T.frame_message(LEVEL, 1, f, agents, hz, {"8": hz, "16": hz}, T.box_rows([(0, 0, 8, 8, 0)], 3)))
        if f % 2 == 0:
            p = _plan(f, x0=-60.0 + 2 * f, y0=-30.0)
            p["path"] = [[x, -30.0] for x, _ in p["path"]]
            p["predictions"] = {str(h): p["path"][h] for h in (1, 4, 8, 16)}
            msgs.append(T.plan_message(LEVEL, 1, p))
    msgs.append(T.event_message(LEVEL, 1, {"kind": "death", "frame": 9, "agent": 1, "x": 0.0, "y": 0.0,
                                           "explained": False}))
    return [tel.fit(m) for m in msgs]


def test_live_parses_telemetry_datagrams():
    src = V.LiveSource(bind=False)
    data = _datagrams()
    assert all(d is not None for d in data)
    for d in data:
        src.datagram(d)
    src.datagram(b"Xgarbage")
    assert src.bad == 1 and src.received == len(data)
    sc = src.scene()
    assert sc.level is not None and sc.level.W == W and sc.level.words is not None
    assert sc.level.solid[3, 5] == 1 and sc.level.solid[0, 3] == 1 and sc.level.solid[5, 5] == 0
    assert sc.frame == 9 and len(sc.agents) == 3 and sc.planned == [0]
    assert sc.plans[0]["frame"] == 8 and len(sc.history[0]) == 5
    assert len(sc.hazards) == 1 and set(sc.future) == {"8", "16"} and sc.memory[0][5] == 3
    assert sc.stats["decisions"] == 5 and sc.stats["unexplained"] == 1
    # prediction vs reality from the live history: plan at frame 4 predicted x = -52 + 8 = -44 at frame 8,
    # real x there was -60 + 16 + 5 = -39: off by 5
    m = V.matured_predictions(sc.history[0], lambda f: sc.real_at(0, f), 9)
    p4 = next(x for x in m if x["h"] == 4)
    assert p4["plan_frame"] == 4 and p4["frame"] == 8 and p4["error"] == pytest.approx(5.0)
    assert sc.boxes_at(9 + 8) == sc.future["8"]
    surf = _surface()
    r = V.Renderer()
    r.render(surf, sc)
    r.next_agent(sc)
    assert r.sel == 0                                            # only one planned agent
    # a new wave: the per-wave histories start again
    src.handle(T.decode(T.Telemetry(None).fit(T.frame_message(LEVEL, 2, 0, [], [], {}, []))))
    assert src.scene().plans == {} and src.scene().stats["decisions"] == 5


def test_live_udp_never_blocks_and_receives():
    src = V.LiveSource("127.0.0.1", 0)
    try:
        t = time.perf_counter()
        for _ in range(50):
            assert src.poll() == 0                              # no sender: returns at once
        assert time.perf_counter() - t < 1.0
        V.Renderer().render(_surface(), src.scene())            # renders "waiting" without data
        tel = T.Telemetry(src.addr)
        for d in _datagrams():
            tel.sock.sendto(d, src.addr)
        got, t = 0, time.time()
        while got < len(_datagrams()) and time.time() - t < 3.0:
            got += src.poll()
            time.sleep(0.01)
        assert got == len(_datagrams()) and src.scene().frame == 9
        tel.close()
    finally:
        src.close()


def test_matcher():
    plans = [{"frame": 0, "predictions": {"1": [1.0, 0.0], "4": [4.0, 0.0], "8": [8.0, 0.0], "16": [16.0, 0.0]}},
             {"frame": 2, "predictions": {"1": [3.0, 0.0], "4": [6.0, 0.0], "8": None, "16": [18.0, 0.0]}}]
    real = {f: (float(f), 3.0 if f == 6 else 0.0) for f in range(30)}
    m = {x["h"]: x for x in V.matured_predictions(plans, real.get, 6)}
    assert set(m) == {1, 4}                                    # +8 / +16 have not arrived yet
    assert m[1]["plan_frame"] == 2 and m[1]["error"] == pytest.approx(0.0)
    assert m[4]["plan_frame"] == 2 and m[4]["frame"] == 6 and m[4]["error"] == pytest.approx(3.0)
    assert m[4]["threshold"] == pytest.approx(1.5 * 2.0)
    m = {x["h"]: x for x in V.matured_predictions(plans, real.get, 10)}
    assert m[8]["plan_frame"] == 0 and m[8]["error"] == pytest.approx(0.0)     # plan 2 has no +8: plan 0's
    assert m[1]["frame"] == 3 and m[4]["frame"] == 6           # +1 matured 7 frames ago: still shown (<= 8)
    m = {x["h"]: x for x in V.matured_predictions(plans, lambda f: None, 18)}
    assert m[16]["real"] is None and m[16]["error"] is None    # unknown reality: no error, never a guess
    assert 1 not in m                                          # older than MATURED_MAX_AGE
    assert math.isclose(V.box_closest(10.0, 0.0, [0, 0, 2, 1, 0.0])[0], 2.0)
    x, y = V.box_closest(0.0, 10.0, [0, 0, 2, 1, math.pi / 2])  # rotated: the long side is vertical
    assert math.isclose(x, 0.0, abs_tol=1e-9) and math.isclose(y, 2.0)


def test_finish_times_in_the_panel(tmp_path):
    st = V.Stats()
    e = {"t": "event", "level": "1-6", "wave": 3, "kind": "finish", "frame": 254, "agent": 0, "frames": 255,
         "seconds": 4.25, "lock": True, "best": 4.25, "new_best": True}
    st.event(e)
    lines = V.finish_lines(st.summary(), "1-6")
    assert lines[0].startswith("last finish: 4.25 s (255 frames) 1-6 wave 3 a0, NEW BEST")
    assert lines[1] == "level best: 4.25 s"
    assert "4.25 s" in V.event_text(e) and "NEW BEST" in V.event_text(e)
    # no finish on this level yet: the best from best_times.json, labelled
    p = tmp_path / "best_times.json"
    p.write_text(json.dumps({"1-7": {"planned": {"seconds": 3.5}}}), encoding="utf-8")
    assert V.file_best("1-7", str(p)) == 3.5 and V.file_best("1-8", str(p)) is None
    # live: the event message reaches the running stats
    src = V.LiveSource(bind=False)
    src.handle(e)
    assert src.scene().stats["last_finish"]["seconds"] == 4.25


def test_goal_contact_boxes_and_touch_markers():
    """The level message carries the planner's goal / key contact boxes; a plan's goal and key touches are drawn
    (map markers and the panel line) and sent to the in-game overlay as dots."""
    from game.level import tile_solid
    words = _words()
    gb, kb = [90.0, -30.0, 6.0, 5.7, 0.0], [0.0, 0.0, 11.3, 11.3, 0.0]
    lv = _Lv(name=LEVEL, W=W, H=H, ox=OX, oy=OY, kill_box=None, words=words, solid=tile_solid(words),
             goals=[(90.0, -30.0)], spawn=(-60.0, -30.0), keys=[(0.0, 0.0)], bandages=[], portals=[],
             goal_boxes=lambda: [tuple(gb)], key_boxes=lambda: [tuple(kb)])
    m = T.decode(T.Telemetry(None).fit(T.level_message(lv, 1, 1, 0, "clear")))
    assert m["goal_boxes"] == [gb] and m["key_boxes"] == [kb]
    view = V.LevelView.from_message(m)
    assert view.goal_boxes == [gb] and view.key_boxes == [kb]
    src = V.LiveSource(bind=False)
    src.handle(m)
    p = _plan(4)
    p["goal_touch"] = {"frame": 12, "x": 84.0, "y": -30.0, "members": 4}
    p["key_touches"] = [{"frame": 6, "key": 0, "x": 0.0, "y": -2.0}]
    p["finish_model"] = 0.97
    src.handle(T.decode(T.Telemetry(None).fit(T.plan_message(LEVEL, 1, p))))
    sc = src.scene()
    assert sc.plans[0]["goal_touch"]["frame"] == 12
    V.Renderer().render(_surface(), sc)
    import agent
    plan = SimpleNamespace(alternatives=[], path=p["path"], landings=[], wall_jumps=[],
                           goal_touch=p["goal_touch"], key_touches=p["key_touches"])
    dots = [d for d in agent.overlay_payload(plan) if d["col"] in (agent.COL_GOAL, agent.COL_KEY)]
    assert [d["pts"] for d in dots] == [[[0.0, -2.0]], [[84.0, -30.0]]]
