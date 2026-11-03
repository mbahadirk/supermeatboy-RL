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
         "--wm-collect", "--death-memory", str(tmp_path / "deaths.json"), "--telemetry", "off"])
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
            assert set(e) == {"pts", "col", "w"} and 0 <= e["col"] < 2 ** 32 and e["w"] > 0
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
