"""Checks for the planning agent's runner (agent.py) and its telemetry (viz/telemetry.py), no game: the runner's
message handler is driven with a recorded 1-13 wave (2 planned agents) through a fake Frida script."""
import json
import os
import socket
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from viz import telemetry as T                          # noqa: E402

DATA = os.path.join(HERE, "runs", "real_20261102-165351", "wm_data")
MODEL = os.path.join(HERE, "runs", "wm_locked", "wm_model.pt")
LEVEL = "1-13"
need_data = pytest.mark.skipif(not (os.path.isdir(DATA) and os.path.exists(MODEL)), reason="recorded data not here")
N_FRAMES = 40
DIE_AT = 20                                             # agent 1 is made to die here (no hazard near: unexplained)


class _FakeScript:
    def __init__(self):
        self.posts = []

    def post(self, msg):
        self.posts.append(msg)


class _CaptureTel(T.Telemetry):
    """The real encoder and shrink rules; datagrams kept instead of sent."""

    def __init__(self):
        super().__init__(None)
        self.out = []

    def send(self, msg):
        data = self.fit(msg)
        assert data is not None, msg.get("t")
        self.out.append(data)
        return True


def _runner(tmp_path, tel):
    import agent
    args = agent.build_parser().parse_args(
        ["--agents", "2", "--replan", "2", "--budget-ms", "60", "--model", MODEL, "--name", str(tmp_path / "run"),
         "--wm-collect", "--death-memory", str(tmp_path / "deaths.json"), "--move-memory", str(tmp_path / "moves.json"), "--telemetry", "off",
         "--best-times", str(tmp_path / "best_times.json")])
    r = agent.Runner(args, telemetry=tel)
    r.script = _FakeScript()
    return r


def _drive_wave(r, exact: bool = True):
    """info, N_FRAMES recorded frames (agent 1 dies at DIE_AT), wave end with agent 0 dead on the last update.
    exact: the wave_end carries the final states `s` and the hazards / shots `hz` / `sh` (realgame.js now);
    otherwise it is the older message without them. Returns (acts, dead_xy, died_frame, end_xy): end_xy is
    agent 0's final position sent with the wave end (None without `s`)."""
    from wm.buffer import load_level
    import glob
    import gzip
    words, meta = load_level(DATA, LEVEL)
    info = dict(meta, name=LEVEL, agents=2)
    rows = world = None
    for p in sorted(glob.glob(os.path.join(DATA, "wave_*.npz"))):
        z = np.load(p)
        if json.loads(str(z["meta"])).get("level") == LEVEL:
            rows = z["rows"].copy()
            world = json.load(gzip.open(p[:-4] + "_world.json.gz", "rt", encoding="utf-8"))
            break
    assert rows is not None
    agents = sorted(set(rows["agent"].tolist()))[:2]
    r.on_message({"type": "send", "payload": dict(info, ev="info")}, words.astype("<u4").tobytes())
    assert r.script.posts[-1]["type"] == "go" and r.script.posts[-1]["payload"]["agents"] == 2
    acts, dead_xy, died_frame = [], None, None
    for k, w in enumerate(world[:N_FRAMES]):
        f = w["f"]
        st = []
        for j, a in enumerate(agents):
            q = rows[(rows["agent"] == a) & (rows["frame"] == f)]
            q = q[0] if len(q) else rows[rows["agent"] == a][-1]
            s = [float(q["x"]), float(q["y"]), float(q["vx"]), float(q["vy"]), int(q["flags"]), int(q["state"]), 0,
                 int(q["keys"])]
            if j == 1 and k >= DIE_AT:
                if dead_xy is None:
                    dead_xy, died_frame = (s[0], s[1]), f
                s[0], s[1], s[2], s[3], s[6] = dead_xy[0], dead_xy[1], 0.0, 0.0, 1
            st.append(s)
        msg = dict(w, ev="frame", frame=f, wave=1, s=st)
        msg.pop("f")
        n_posts = len(r.script.posts)
        r.on_message({"type": "send", "payload": msg}, None)
        new = r.script.posts[n_posts:]
        act = [p for p in new if p["type"] == "act"]
        assert len(act) == 1, new                                   # exactly one answer per frame
        acts.append(act[0]["payload"])
    w_last = world[N_FRAMES - 1]
    end = {"ev": "wave_end", "wave": 1, "status": [1, 1], "finish": [-1, -1], "frame": w_last["f"] + 1, "why": "done"}
    end_xy = None
    if exact:                                                   # agent 0 moved a little on the last update, died
        s0 = list(st[0])
        s0[0], s0[1], s0[6] = s0[0] + 3.25, s0[1] - 1.5, 1
        end_xy = (s0[0], s0[1])
        end.update(s=[s0, st[1]], hz=w_last.get("hz", []), sh=w_last.get("sh", []))
    r.on_message({"type": "send", "payload": end}, None)
    assert r.script.posts[-1]["type"] == "go"
    return acts, dead_xy, died_frame, end_xy


@need_data
def test_held_out_level_is_played_without_data_or_memories(tmp_path, monkeypatch):
    """On a held-out level (eval_levels.json; here 1-13 made one) the runner plays as anywhere (the benchmark's
    held-out block uses the same command), but writes no world-model shard and charges no death or
    point-of-no-return memory."""
    import wm.buffer
    monkeypatch.setattr(wm.buffer, "eval_levels", lambda: {LEVEL})
    r = _runner(tmp_path, None)
    acts, dead_xy, died_frame, end_xy = _drive_wave(r)
    r.close()
    assert r.protected and len(acts) == N_FRAMES
    run = tmp_path / "run"
    events = [json.loads(l) for l in open(run / "events.jsonl", encoding="utf-8")]
    assert [e["protected"] for e in events if e["kind"] == "level"] == [True]
    deaths = [e for e in events if e["kind"] == "death"]
    assert len(deaths) == 2 and not any(e["memory_added"] for e in deaths)       # agent 1's unexplained one too
    assert not any(e.get("charged") for e in events if e["kind"] == "point of no return")
    assert not (tmp_path / "deaths.json").exists() and not (tmp_path / "moves.json").exists()
    assert not list((run / "wm_data").glob("wave_*.npz"))


@need_data
def test_runner_plays_a_recorded_wave(tmp_path):
    from plan import safety as S
    tel = _CaptureTel()
    r = _runner(tmp_path, tel)
    acts, dead_xy, died_frame, end_xy = _drive_wave(r)
    r.close()
    # replies: one per frame, 2 agents, 4 keys of 0/1 each, a leader index
    assert len(acts) == N_FRAMES
    for a in acts:
        assert set(a) >= {"a", "leader"} and len(a["a"]) == 2 and 0 <= a["leader"] < 2
        assert all(len(k) == 4 and all(v in (0, 1) for v in k) for k in a["a"])
    assert all(a["a"][1] == [0, 0, 0, 0] for a in acts[DIE_AT:])          # the dead agent gets no input
    run = tmp_path / "run"
    plans = [json.loads(l) for l in open(run / "plans.jsonl", encoding="utf-8")]
    events = [json.loads(l) for l in open(run / "events.jsonl", encoding="utf-8")]
    states = [json.loads(l) for l in open(run / "states.jsonl", encoding="utf-8")]
    assert len(states) == N_FRAMES and all(len(s["agents"]) == 2 and len(s["agents"][0]) == 6 for s in states)
    assert len(plans) >= N_FRAMES // 2                                  # agent 0 every 2nd frame, agent 1 until it died
    assert {p["agent"] for p in plans} == {0, 1}
    p = plans[0]
    assert p["level"] == LEVEL and "real" in p and len(p["actions"]) == 48 and len(p["path"]) == 49
    assert any(a.get("path") and "vetoed" in a for q in plans for a in q["alternatives"])
    # the injected death: not explained by the geometry, remembered and saved
    deaths = [e for e in events if e["kind"] == "death"]
    inj = [e for e in deaths if e["agent"] == 1]
    assert len(inj) == 1 and inj[0]["frame"] == died_frame and not inj[0]["explained"] and inj[0]["memory_added"]
    assert not inj[0]["estimated"]
    saved = json.load(open(tmp_path / "deaths.json", encoding="utf-8"))
    assert any(d["level"] == LEVEL and d["frame"] == died_frame and abs(d["x"] - dead_xy[0]) < 1e-3
               for d in saved["deaths"])
    mem = S.DeathMemory.load(str(tmp_path / "deaths.json"))
    assert len(mem.boxes(LEVEL, died_frame + 5)) >= 1
    # agent 0 died on the wave's last update: its exact final position from the wave_end message
    end = [e for e in deaths if e["agent"] == 0]
    assert len(end) == 1 and not end[0]["estimated"]
    assert abs(end[0]["x"] - end_xy[0]) < 1e-2 and abs(end[0]["y"] - end_xy[1]) < 1e-2   # (events: 3 decimals)
    assert end[0]["frame"] == states[-1]["frame"] + 1
    wave = [e for e in events if e["kind"] == "wave"]
    assert wave and wave[-1]["estimated"] == 0
    # the share of decisions whose chosen plan the collision guard clipped (plans: model_clipped)
    n_clip = sum((p.get("model_clipped") or 0) > 0 for p in plans)
    assert all(p.get("model_clipped") is not None for p in plans)
    assert wave[-1]["clipped"] == n_clip and wave[-1]["clipped_share"] == pytest.approx(n_clip / len(plans), abs=1e-3)
    # per-frame cost: every category, the python total and the wait for the game, the game speed
    cm = wave[-1]["cost_ms"]
    assert set(cm) == set(agent_mod().FrameCost.CATS) | {"python", "wait"}
    assert all(len(v) == 3 and v[0] >= 0 for v in cm.values()) and cm["planning"][0] > 0
    assert cm["python"][0] == pytest.approx(sum(cm[c][0] for c in agent_mod().FrameCost.CATS), abs=0.1)
    assert wave[-1]["game_speed"] > 0 and wave[-1]["game_s"] == pytest.approx(N_FRAMES * 0.016667, abs=0.01)
    # followed predictions' error per horizon; a position surprise only for a plan played that far
    ef = wave[-1]["err_followed"]
    assert set(ef) == {"1", "4", "8", "16"} and ef["1"][2] > 0
    played = {}
    for i in (0, 1):
        mine = sorted((q for q in plans if q["agent"] == i), key=lambda q: q["frame"])
        for q, nxt in zip(mine, mine[1:] + [None]):
            for f in range(q["frame"], nxt["frame"] if nxt else q["frame"] + 2):
                played[(i, f)] = q["actions"][f - q["frame"]]
    for e in (e for e in events if e["kind"] == "position"):
        q = next(q for q in plans if q["agent"] == e["agent"] and q["frame"] == e["plan_frame"])
        assert all(played.get((e["agent"], e["plan_frame"] + k)) == q["actions"][k] for k in range(e["horizon"]))
    assert any(e["kind"] == "wave" for e in events) and any(e["kind"] == "field" for e in events)
    # world-model data: one shard, labelled
    shard = np.load(next((run / "wm_data").glob("wave_*.npz")))
    meta = json.loads(str(shard["meta"]))
    assert meta["source"] == "planner" and meta["agent_sources"] == {"planner": [0, 1], "explore": []}
    # overlay: realgame.js 'path' payloads, small, well-formed
    over = [q["payload"] for q in r.script.posts if q["type"] == "path"]
    assert over and any(over)
    for pay in over:
        assert isinstance(pay, list)
        assert sum(len(e["pts"]) for e in pay) <= 120
        for e in pay:
            assert set(e) - {"dash"} == {"pts", "col", "w"} and e.get("dash", True) is True and 0 <= e["col"] < 2 ** 32 and e["w"] > 0
            assert all(len(pt) == 2 and all(isinstance(v, float) for v in pt) for pt in e["pts"])
    assert any(e["col"] == 0xE020E040 and len(e["pts"]) > 2 for pay in over for e in pay)   # the chosen path
    # telemetry: every type, each datagram under 60 KB, decodable; the remembered death reaches the future boxes
    msgs = [T.decode(d) for d in tel.out]
    assert all(len(d) <= T.MAX_BYTES for d in tel.out)
    kinds = {m["t"] for m in msgs}
    assert kinds == {"level", "frame", "plan", "event"}
    lv = next(m for m in msgs if m["t"] == "level")
    assert lv["name"] == LEVEL and lv["W"] > 0 and lv["solid"] and lv["words"]
    import base64
    import zlib
    solid = np.frombuffer(zlib.decompress(base64.b64decode(lv["solid"])), np.uint8)
    assert solid.size == lv["W"] * lv["H"]
    frames = [m for m in msgs if m["t"] == "frame"]
    assert len(frames) == N_FRAMES
    fm = frames[-1]
    assert len(fm["agents"]) == 2 and set(fm["future"]) == {"8", "16"}
    assert all(len(b) == 6 for b in fm["hazards"])
    after = [m for m in frames if m["frame"] > died_frame]
    assert all(m["memory"] for m in after)
    assert any(b[5] == S.KIND_MEMORY for m in after for b in m["future"]["8"])


@need_data
def test_old_wave_end_without_final_states(tmp_path):
    """An older realgame.js wave_end without `s` / `hz` / `sh`: the last-update death still counts, its position
    predicted from the plan (estimated)."""
    r = _runner(tmp_path, None)
    _, _, _, end_xy = _drive_wave(r, exact=False)
    r.close()
    assert end_xy is None
    events = [json.loads(l) for l in open(tmp_path / "run" / "events.jsonl", encoding="utf-8")]
    end = [e for e in events if e["kind"] == "death" and e["agent"] == 0]
    assert len(end) == 1 and end[0]["estimated"]
    assert [e for e in events if e["kind"] == "wave"][-1]["estimated"] == 1


def agent_mod():
    import agent
    return agent


def test_frame_cost_accounting(monkeypatch):
    """Exclusive time per category (a nested part goes back to its caller's category), the wait between the
    reply and the next frame, nothing outside frames, and the game speed."""
    import agent
    clock = [0.0]
    monkeypatch.setattr(agent.time, "perf_counter", lambda: clock[0])
    c = agent.FrameCost()

    def tick(ms):
        clock[0] += ms / 1000.0
    for _ in range(4):
        c.begin()
        c.go("planning"); tick(10)
        prev = c.go("logging"); tick(2); c.go(prev); tick(5)          # planning 15, logging 2
        c.go("telemetry"); tick(1)
        c.end()
        tick(3)                                                        # the game's turn
    assert c.go("planning") is None                                    # outside a frame: nothing charged
    tick(5)
    s = c.summary(4, 16.667)
    cm = s["cost_ms"]
    assert cm["planning"] == [15.0, 15.0, 15.0] and cm["logging"][0] == 2.0 and cm["telemetry"][0] == 1.0
    assert cm["python"][0] == 18.0 and cm["wait"] == [3.0, 3.0, 3.0]
    assert s["wall_s"] == pytest.approx(0.089, abs=0.006) and s["game_speed"] == pytest.approx(4 * 0.016667 / 0.089, abs=1e-3)
    assert "speed" in c.text(s)
    c.reset()
    assert c.summary(1, 16.667) is None


def _recorded():
    """The first recorded 1-13 wave: (level words, meta, rows, world records)."""
    from wm.buffer import load_level
    import glob
    import gzip
    words, meta = load_level(DATA, LEVEL)
    for p in sorted(glob.glob(os.path.join(DATA, "wave_*.npz"))):
        z = np.load(p)
        if json.loads(str(z["meta"])).get("level") == LEVEL:
            world = json.load(gzip.open(p[:-4] + "_world.json.gz", "rt", encoding="utf-8"))
            return words, meta, z["rows"].copy(), world
    raise AssertionError("no recorded 1-13 wave")


@need_data
def test_finish_times_and_best_times(tmp_path, capsys):
    """Finishes seen in a frame message and on the wave's last update: the exact finishing update, the time
    (frames x dt), times.jsonl for planned and explore agents, best_times.json (lock on only), the finish
    events (also over telemetry) and the wave summary line."""
    import agent
    best = tmp_path / "best_times.json"
    args = agent.build_parser().parse_args(
        ["--agents", "1", "--explore", "1", "--replan", "2", "--budget-ms", "40", "--model", MODEL,
         "--name", str(tmp_path / "run"), "--death-memory", str(tmp_path / "deaths.json"), "--move-memory", str(tmp_path / "moves.json"), "--telemetry", "off",
         "--best-times", str(best)])
    tel = _CaptureTel()
    r = agent.Runner(args, telemetry=tel)
    r.script = _FakeScript()
    words, meta, rows, world = _recorded()
    a0 = sorted(set(rows["agent"].tolist()))[0]
    r.on_message({"type": "send", "payload": {"ev": "debug", "msg": "time step locked to the game's fixed step"}},
                 None)
    r.on_message({"type": "send", "payload": dict(meta, name=LEVEL, agents=2, ev="info")},
                 words.astype("<u4").tobytes())

    def wave(no, n_frames, status_at, end_status, end_finish):
        """status_at(agent, frame) -> status in the frame messages; then the wave end after n_frames."""
        last = None
        for k, w in enumerate(world[:n_frames]):
            q = rows[(rows["agent"] == a0) & (rows["frame"] == w["f"])][0]
            s = [float(q["x"]), float(q["y"]), float(q["vx"]), float(q["vy"]), int(q["flags"]), int(q["state"]), 0,
                 int(q["keys"])]
            st = [list(s[:6]) + [status_at(i, k), s[7]] for i in range(2)]
            msg = dict(w, ev="frame", frame=k, wave=no, s=st, dt=16.667)
            msg.pop("f")
            r.on_message({"type": "send", "payload": msg}, None)
            last = st
        final = [list(x[:6]) + [end_status[i], x[7]] for i, x in enumerate(last)]
        r.on_message({"type": "send", "payload": {"ev": "wave_end", "wave": no, "status": end_status,
                                                  "finish": end_finish, "frame": n_frames, "why": "quota",
                                                  "s": final, "hz": [], "sh": []}}, None)

    # wave 1: the explore agent's finish shows in frame 6's message (update 5); the planned agent finishes on
    # the last update (9 of 0..9)
    wave(1, 10, lambda i, k: 2 if i == 1 and k >= 6 else 0, [2, 2], [9, 5])
    # wave 2: planned, slower (update 14 of 0..14): no new best
    wave(2, 15, lambda i, k: 0, [2, 1], [14, -1])
    # wave 3: planned finish seen in frame 8's message (update 7): new best; the wave end repeats it
    wave(3, 12, lambda i, k: 2 if i == 0 and k >= 8 else 0, [2, 1], [7, -1])
    r.close()
    out = capsys.readouterr().out
    run = tmp_path / "run"
    times = [json.loads(l) for l in open(run / "times.jsonl", encoding="utf-8")]
    assert [(t["wave"], t["source"], t["frame"], t["frames"]) for t in times] == \
        [(1, "explore", 5, 6), (1, "planned", 9, 10), (2, "planned", 14, 15), (3, "planned", 7, 8)]
    assert all(t["lock"] is True and t["level"] == LEVEL and t["seconds"] == round(t["frames"] * 0.016667, 3)
               for t in times)
    b = json.load(open(best, encoding="utf-8"))
    assert b[LEVEL]["planned"]["frames"] == 8 and b[LEVEL]["planned"]["wave"] == 3
    assert b[LEVEL]["planned"]["seconds"] == pytest.approx(0.133) and b[LEVEL]["explore"]["frames"] == 6
    events = [json.loads(l) for l in open(run / "events.jsonl", encoding="utf-8")]
    fin = [e for e in events if e["kind"] == "finish"]
    assert [(e["wave"], e["frame"], e["frames"], e["new_best"]) for e in fin] == \
        [(1, 9, 10, True), (2, 14, 15, False), (3, 7, 8, True)]
    assert [e["best"] for e in fin] == [pytest.approx(0.167), pytest.approx(0.167), pytest.approx(0.133)]
    waves = [e for e in events if e["kind"] == "wave"]
    assert [w["finished"] for w in waves] == [1, 1, 1] and waves[0]["ex_finished"] == 1
    assert waves[1]["finish_s"] == pytest.approx(0.25) and waves[1]["best_s"] == pytest.approx(0.167)
    assert "finished in 0.17 s (best 0.17 s, new best)" in out and "finished in 0.25 s (best 0.17 s)" in out
    assert "fastest 0.10 s" in out
    sent = [T.decode(d) for d in tel.out]
    assert [m["seconds"] for m in sent if m["t"] == "event" and m["kind"] == "finish"] == [0.167, 0.25, 0.133]
    assert f"[level] {LEVEL}: " in out and "no agent best yet" in out
    # the timer sign: nothing at the level start (no best yet), then each new planned best
    assert [p["payload"] for p in r.script.posts if p["type"] == "best"] == [pytest.approx(0.167), pytest.approx(0.133)]
    # the viewer, live: the runner's own datagrams (a fake telemetry stream) reach its info panel
    from viz import viewer as V
    src = V.LiveSource(bind=False)
    shown = []
    for d in tel.out:
        src.datagram(d)
        m = T.decode(d)
        if m["t"] == "event" and m["kind"] == "finish":
            shown.append(V.finish_lines(src.scene().stats, LEVEL))
    assert shown[0][0].startswith(f"last finish: 0.17 s (10 frames) {LEVEL} wave 1 a0, NEW BEST")
    assert shown[1][0].startswith(f"last finish: 0.25 s (15 frames) {LEVEL} wave 2 a0") and "NEW" not in shown[1][0]
    assert [s[1] for s in shown] == ["level best: 0.17 s", "level best: 0.17 s", "level best: 0.13 s"]
    import pygame
    pygame.init()
    V.Renderer().render(pygame.Surface((1280, 800)), src.scene())          # the panel draws them
    # the next level start prints the agents' best
    r = agent.Runner(args, telemetry=None)
    r.script = _FakeScript()
    r.on_message({"type": "send", "payload": dict(meta, name=LEVEL, agents=2, ev="info")},
                 words.astype("<u4").tobytes())
    r.close()
    assert f"agent best {b[LEVEL]['planned']['seconds']:.2f} s;" in capsys.readouterr().out
    assert [p for p in r.script.posts if p["type"] == "best"] == [{"type": "best", "payload": pytest.approx(0.133)}]


def test_best_times_file(tmp_path, capsys):
    import agent
    p = tmp_path / "best_times.json"
    bt = agent.BestTimes(str(p))
    rec = {"seconds": 4.25, "frames": 255, "run": "r", "wave": 1, "agent": 0, "date": "d"}
    assert bt.offer("1-1", "planned", rec) and not bt.offer("1-1", "planned", dict(rec, seconds=4.3))
    assert bt.offer("1-1", "explore", dict(rec, seconds=5.0))
    assert json.load(open(p))["1-1"]["planned"]["seconds"] == 4.25 and not os.path.exists(str(p) + ".tmp")
    assert agent.best_text(4.25, bt.best("1-1", "planned"), True, True) == "4.25 s (best 4.25 s, new best)"
    assert "not a best time" in agent.best_text(4.25, None, False, None)
    # an unreadable file is never overwritten
    p.write_text("{broken", encoding="utf-8")
    bt = agent.BestTimes(str(p))
    assert not bt.offer("1-1", "planned", dict(rec, seconds=1.0)) and p.read_text(encoding="utf-8") == "{broken"


def test_s_writes_the_agents_best_after_one_save_backup(tmp_path, capsys):
    """S in the game: the finish request carries the planned agents' best for that level (999.99 in
    realgame.js without one); the save file is copied once per run before the first such write; a save that
    cannot be found means no time is written."""
    import agent
    best = tmp_path / "best_times.json"
    best.write_text(json.dumps({"1-14": {"planned": {"seconds": 6.55, "frames": 393}},
                                "1-13": {"explore": {"seconds": 9.0}}}), encoding="utf-8")
    game = tmp_path / "game"
    (game / "UserData").mkdir(parents=True)
    (game / "UserData" / "savegame.dat").write_bytes(b"\x01\x02save")
    args = agent.build_parser().parse_args(
        ["--agents", "0", "--explore", "1", "--name", str(tmp_path / "run"), "--telemetry", "off",
         "--death-memory", str(tmp_path / "deaths.json"), "--move-memory", str(tmp_path / "moves.json"), "--best-times", str(best)])
    r = agent.Runner(args, telemetry=None)
    r.script = _FakeScript()
    r.backup_root = str(tmp_path / "backups")
    r.find_save = lambda: agent.locate_save(str(game / "SuperMeatBoy.exe"))
    save = lambda lv: r.on_message({"type": "send", "payload": {"ev": "save", "level": lv}}, None)
    save("1-14")
    out = capsys.readouterr().out
    backups = list((tmp_path / "backups").iterdir())
    assert len(backups) == 1 and (backups[0] / "savegame.dat").read_bytes() == b"\x01\x02save"
    assert backups[0].name.startswith("save_")
    assert f"[S] finishing 1-14 the game's way with the agent's best 6.55 s (save backed up to {backups[0]})" in out
    assert r.script.posts[-1] == {"type": "finish", "payload": {"time": 6.55}}
    save("1-14")                                         # again: no second backup
    assert len(list((tmp_path / "backups").iterdir())) == 1 and r.script.posts[-1]["payload"]["time"] == 6.55
    save("1-13")                                         # only an explore agent's best: not the planned one
    assert "[S] finishing 1-13 the game's way, no agent best for 1-13: 999.99" in capsys.readouterr().out
    assert r.script.posts[-1] == {"type": "finish", "payload": {"time": None}}
    # a new run whose save cannot be found: the best is not written
    r2 = agent.Runner(args, telemetry=None)
    r2.script = _FakeScript()
    r2.backup_root = str(tmp_path / "backups2")
    r2.find_save = lambda: agent.locate_save(str(tmp_path / "nowhere" / "SuperMeatBoy.exe"))
    r2.on_message({"type": "send", "payload": {"ev": "save", "level": "1-14"}}, None)
    out = capsys.readouterr().out
    assert "the agent's best 6.55 s is not written (no save file at" in out and out.rstrip().endswith(": 999.99")
    assert r2.script.posts[-1]["payload"]["time"] is None and not (tmp_path / "backups2").exists()
    r.close()
    r2.close()
    ev = [json.loads(l) for l in open(tmp_path / "run" / "events.jsonl", encoding="utf-8")]
    assert [e["seconds"] for e in ev if e["kind"] == "save"] == [6.55, 6.55, None, None]


def test_realgame_js_writes_the_sent_time():
    """realgame.js: the finish request's time replaces 999.99 as the level time written while finishing."""
    src = open(os.path.join(HERE, "realgame.js"), encoding="utf-8").read()
    assert "hud.add(0x384).writeFloat(finishTime)" in src and "writeFloat(999.99)" not in src
    assert "finishTime = typeof t === 'number' && t > 0 && t < 999.99 ? t : 999.99;" in src
    # the 'best' message only changes the timer sign's text: its hook rewrites the format and the number of the
    # set-text call for the sign's text object, and writes nothing else (no record, no hud field)
    hook = src[src.index("Interceptor.attach(img(0x45c220)"):src.index("let completeFrames")]
    assert "TIMER_TEXT" in hook and "a[1] = fmtBest" in hook and "writeDouble(best)" in hook
    assert "hud" not in hook and "0x384" not in hook and hook.count("write") == 1


def test_telemetry_udp_roundtrip_and_never_blocks():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    tel = T.Telemetry(rx.getsockname())
    small = T.event_message("1-1", 1, {"kind": "finish", "frame": 10, "agent": 0})
    big = {"t": "plan", "time": 0.0, "level": "1-1", "wave": 1, "path": [[float(i), float(i)] for i in range(49)],
           "spread": [0.5] * 49, "alternatives": [{"moves": ["wait 8f"], "reason": "x",
                                                    "path": [[i * 1.234567, i * 7.654321] for i in range(49)]}] * 5}
    assert tel.send(small) and tel.send(big)
    got = [T.decode(rx.recv(65536)) for _ in range(2)]
    assert got[0]["kind"] == "finish" and got[0]["t"] == "event"
    assert got[1]["t"] == "plan" and len(got[1]["alternatives"]) == 5
    rx.close()
    # too big even after shrinking: dropped and counted, never raised
    huge = {"t": "event", "kind": "x", "blob": os.urandom(80000).hex()}
    assert not tel.send(huge) and tel.too_big == 1
    # nobody listening: sends never raise
    tel2 = T.Telemetry(("127.0.0.1", 9))
    for _ in range(5):
        tel2.send(small)
    assert tel2.sent + tel2.dropped == 5
    tel.close()
    tel2.close()
    assert T.parse_addr("off") is None and T.parse_addr("127.0.0.1:47800") == ("127.0.0.1", 47800)


def test_frame_message_shrinks_under_the_limit():
    agents = [{"i": i, "role": "explore", "x": 1.0, "y": 2.0} for i in range(200)]
    rng = np.random.default_rng(0)
    boxes = T.box_rows(rng.normal(0, 1000, (3000, 5)), 0)
    m = T.frame_message("1-1", 1, 5, agents, boxes[:200], {"8": boxes, "16": boxes}, [])
    data = T.Telemetry(None).fit(m)
    assert data is not None and len(data) <= T.MAX_BYTES
    assert T.decode(data)["hazards"]


def test_overlay_draws_the_completion_dashed_with_its_landing():
    from types import SimpleNamespace
    import agent
    path = [[float(k), 0.0] for k in range(49)]
    comp = {"continuation": "keep", "frames": 6, "x": 60.0, "y": -10.0, "on": "ground", "died": [],
            "path": [[48.0 + 2 * k, -k * 1.5] for k in range(7)]}
    p = SimpleNamespace(alternatives=[], path=path, landings=[], wall_jumps=[], key_touches=[], goal_touch=None,
                        completion=comp)
    out = agent.overlay_payload(p)
    dashed = [o for o in out if o.get("dash")]
    assert len(dashed) == 1 and dashed[0]["col"] == agent.COL_COMP and dashed[0]["pts"][0] == [48.0, 0.0]
    assert any(o["col"] == agent.COL_LAND and o["pts"] == [[60.0, -10.0]] for o in out)
    p.completion = dict(comp, on=None, frames=None)                  # every continuation died: no landing dot
    assert not any(o["col"] == agent.COL_LAND for o in agent.overlay_payload(p))
