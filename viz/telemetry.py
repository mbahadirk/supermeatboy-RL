"""Live telemetry of the planning agent (smbrl/agent.py) for a separate viewer: JSON messages over UDP.

Transport
---------
- UDP datagrams to host:port (agent.py --telemetry, default 127.0.0.1:47800). Fire and forget: the sender
  never blocks and never retries. A datagram that cannot be sent right now (socket buffer full, nobody
  listening, any OS error) is dropped and counted (Telemetry.dropped). Messages are not acknowledged and may
  be lost or (rarely) reordered; every message is self-contained.
- Each datagram is one message: 1 byte format tag + body.
    b"J" + UTF-8 JSON              (bodies of at most COMPRESS_OVER = 1200 bytes)
    b"Z" + zlib.compress(UTF-8 JSON) (larger bodies)
  A datagram is never larger than MAX_BYTES = 60000 bytes. A message that would be larger is first shrunk
  (see each type below); if it still does not fit it is dropped and counted (Telemetry.too_big).
- decode(datagram) -> dict is the reference reader.

Coordinates and units (all messages)
------------------------------------
- Game units, x to the right, y UP (the game's world coordinates, as realgame.js reports them). One tile is
  20 units. Tile (column c, row r) covers x in [ox + 20 c, ox + 20 (c + 1)), y in [oy + 20 r, oy + 20 (r + 1));
  row 0 is the BOTTOM row.
- Speeds in units per second. Frames are the game's wave frames (60 per second, the time step is locked),
  counted from 0 at each wave start. Angles in radians.
- A box is [x, y, hw, hh, angle, kind]: centre, half width, half height (a kill shape the game's narrow phase
  tests, game.level.shape_box, and for future boxes grown by the future's uncertainty radius), the
  rotation (a point p is inside when |u| <= hw and |v| <= hh with u = dx cos a + dy sin a,
  v = -dx sin a + dy cos a, d = p - centre), and what it is: 0 lethal piece, 1 flying shot, 3 remembered death
  (plan/safety.py DeathMemory). The agent's centre dies when it comes within BODY = 6 units of a box
  (realpath.BODY), and when it leaves the level's alive box.
- Every message has "t" (type), "time" (sender's wall clock, seconds since the epoch, float), "level" (level
  name, e.g. "1-13") and "wave" (the game's wave number on this level, from 1; 0 before the first wave).

Message types
-------------
{"t": "level", "time", "level", "wave",
 "name": str (= level), "W": int, "H": int (map size in tiles), "ox": float, "oy": float (map origin: the
 bottom-left corner of tile (0, 0)), "tile": 20.0,
 "kill_box": [left, right, bottom, top] or null (the camera-bounds kill box; null: the map-edge rule),
 "alive_box": [left, right, bottom, top] (the lines a centre must stay within: the kill box, or the map edges
 +10 left / right / top and -60 bottom),
 "words": str or null: base64(zlib(H x W uint32 little-endian, row-major, row 0 = bottom)) of the game's live
 collision tile words (low byte = tile id, 0 empty, 1..99 solid; bit 8 = special: crumble / lock / switch;
 bits 12+ = shape and rotation, see plan/geometry.py). Null when the message would be too big with it.
 "solid": str: base64(zlib(H x W uint8, row-major, row 0 = bottom)): 1 where the live map is solid for the
 agents (game.level.Level.solid: crumbled / opened tiles are 0), else 0,
 "goals": [[x, y], ...], "spawn": [x, y], "keys": [[x, y], ...], "bandages": [[x, y], ...],
 "portals": [[[x, y], [x, y]], ...],
 "goal_boxes": [[x, y, hw, hh, angle], ...], "key_boxes": [...] (one per key, in the order of "keys"): what the
 planner counts as touching the goal / a key (game.level.Level.goal_boxes / key_boxes, contact_reach.json):
 the agent's centre within BODY of the box, as for hazards; [] when unknown,
 "agents": {"planned": n, "explore": k} (agent indices 0..n-1 are planned, n..n+k-1 explore),
 "field": str (the progress field's summary line) or null}
 Sent when a level's data is ready and then every 2 seconds (a viewer started late still gets it).

{"t": "frame", "time", "level", "wave", "frame": int,
 "agents": [{"i": int (agent index), "role": "planner" | "explore", "x", "y", "vx", "vy" (floats),
             "status": 0 alive | 1 dead | 2 finished, "flags": int (the game's character flag bits: 0x2
             airborne-ish, 0x200 on a wall), "state": int (movement state), "keys": int (bit k = key k),
             "action": int (index into ACTIONS (left, right, jump, run): 0 none, 1 L, 2 R, 3 jump, 4 L+jump,
             5 R+jump, 6 R+run, 7 R+jump+run, 8 L+run, 9 L+jump+run) or -1 (none: dead / finished)}, ...],
 "hazards": [box, ...] (the deadly boxes now: the lethal pieces' kill shapes and shots),
 "future": {"8": [box, ...], "16": [box, ...]} (the deadly boxes the planner expects 8 and 16 frames ahead,
           including remembered deaths active then; {} when there is no prediction),
 "memory": [box, ...] (remembered deaths active at this frame, kind 3)}
 Sent every frame. Shrunk when too big: "future" dropped, then the agent list cut to the first 60.

{"t": "plan", "time", "level", "wave", "real": {x, y, vx, vy, flags, state, status, keys} (the agent's reported
 state at the decision frame), ... every field of plan.explain.Plan.to_dict() ...}
 One per decision of a planned agent (the same record as a line of runs/<name>/plans.jsonl). The fields (see
 plan/explain.py Plan) include:
 "frame" (the decision's wave frame), "agent", "state" {x, y, vx, vy, flags, state, held, keys, ground,
 wall_side, supported, cell}, "chosen" [move names], "actions" [H ACTIONS indices, index k = frame + k],
 "path" [[x, y] for frames 0..H] (predicted, ensemble mean), "spread" [units per frame], "landings"
 [{frame, x, y, on: ground | wall}] (frame = frames after the decision), "wall_jumps" [{frame, x, y, side}],
 "nearest" [{id, kind, clearance, frame}], "veto" null | {frame, id, kind}, "soft_risk", "soft_frame",
 "score" {progress, progress_credit, risk, uncertainty, spread_end, failure, total}, "v_start", "v_end",
 "alternatives" [{moves, total, progress, stage, reason, path [[x, y], ...], vetoed: bool}],
 "predictions" {"1": [x, y], "4": ..., "8": ..., "16": ...}, "timing" {stage1, stage2, final, total} (ms),
 "counts", "segments" [[move, from frame, to frame]],
 "trigger" [str] (why this decision was made, plan/commit.py: "start", "start: tiles", "interval", "move end",
 "surprise +8 err 14.2 > 11.0", "support: landed early +5", "danger h12 f10", "danger: risk 0.10 -> 0.60",
 "short", "missed: princess f12", "missed: key 0 f9"),
 "switch" null (no previous plan to keep) | {kept: bool, gain: float (the best other candidate's advantage over
 the previous plan), margin: float, unit: "V" | "frames", reason: str} (plan/planner.py switch rule).
 Older senders have neither field.
 Shrunk when too big: alternatives' paths dropped, then "spread", then the alternatives.

{"t": "event", "time", "level", "wave", "kind": str, "frame": int or null, ... kind-specific fields}
 kinds: "position" (surprise: {agent, horizon, predicted [x, y], real [x, y], error, threshold, plan_frame,
 plan}), "unpredicted death" (surprise: {agent, x, y, plan_frame, plan}), "failed move" ({agent, move, cell,
 expected_drop, real_drop, dead, count, reason}), "death" ({agent, x, y, explained: bool, cause: str,
 estimated: bool (the position was predicted, not reported: deaths on a wave's last frame), memory_added:
 bool}), "finish" (a planned agent touched the goal: {agent, x, y, frame: the update it touched it in,
 frames: frame + 1, seconds: frames x dt_ms / 1000, dt_ms, lock: the time-step lock was on (best times only
 then), best: the level's best planned time (s) after this finish or null, new_best: bool}), "level",
 "kill box", "hazards", "field" (progress field built / rebuilt), "plan error", "wave"
 (the per-wave summary; err_followed {"1": [median, 90%, n], "4", "8", "16"}: position error of predictions
 whose plan was followed that far), "lock", "stop".
"""
from __future__ import annotations

import base64
import json
import math
import socket
import time
import zlib

import numpy as np

MAX_BYTES = 60000
COMPRESS_OVER = 1200
DEFAULT_ADDR = ("127.0.0.1", 47800)
FUTURE_AT = (8, 16)


def _plain(v):
    """numpy / tuples -> plain JSON values; non-finite floats -> None."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return round(f, 3) if math.isfinite(f) else None
    return v


def encode(msg: dict) -> bytes:
    body = json.dumps(_plain(msg), separators=(",", ":")).encode("utf-8")
    if len(body) > COMPRESS_OVER:
        return b"Z" + zlib.compress(body, 6)
    return b"J" + body


def decode(data: bytes) -> dict:
    tag, body = data[:1], data[1:]
    if tag == b"Z":
        body = zlib.decompress(body)
    elif tag != b"J":
        raise ValueError(f"unknown telemetry format tag {tag!r}")
    return json.loads(body.decode("utf-8"))


def parse_addr(s: str | None):
    """'host:port' -> (host, port); 'off' / '' / None -> None."""
    if s is None or str(s).strip().lower() in ("", "off", "none", "0"):
        return None
    host, _, port = str(s).rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"--telemetry wants host:port or off, got {s!r}")
    return host, int(port)


def _pack(a: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(np.ascontiguousarray(a).tobytes(), 6)).decode("ascii")


def box_rows(boxes, kind) -> list:
    """(M, 5) boxes and their kinds -> [[x, y, hw, hh, angle, kind], ...] rounded."""
    b = np.asarray(boxes, np.float64).reshape(-1, 5)
    k = np.broadcast_to(np.asarray(kind, np.int64), (len(b),))
    return [[round(float(r[0]), 1), round(float(r[1]), 1), round(float(r[2]), 1), round(float(r[3]), 1),
             round(float(r[4]), 3), int(kk)] for r, kk in zip(b, k)]


# ---- message builders (the schema above) --------------------------------------------------------------
def level_message(lv, wave: int, planned: int, explore: int, field_summary: str | None = None,
                  with_words: bool = True) -> dict:
    return {"t": "level", "time": time.time(), "level": lv.name, "wave": int(wave), "name": lv.name,
            "W": int(lv.W), "H": int(lv.H), "ox": float(lv.ox), "oy": float(lv.oy), "tile": 20.0,
            "kill_box": list(lv.kill_box) if lv.kill_box is not None else None,
            "alive_box": [float(v) for v in lv.alive_box()],
            "words": _pack(np.asarray(lv.words, "<u4")) if with_words else None,
            "solid": _pack(np.asarray(lv.solid, np.uint8)),
            "goals": [list(g) for g in lv.goals], "spawn": list(lv.spawn), "keys": [list(k) for k in lv.keys],
            "bandages": [list(b) for b in lv.bandages], "portals": [[list(a), list(b)] for a, b in lv.portals],
            "goal_boxes": [[float(v) for v in b] for b in (lv.goal_boxes() if hasattr(lv, "goal_boxes") else [])],
            "key_boxes": [[float(v) for v in b] for b in (lv.key_boxes() if hasattr(lv, "key_boxes") else [])],
            "agents": {"planned": int(planned), "explore": int(explore)}, "field": field_summary}


def frame_message(level: str, wave: int, frame: int, agents: list, hazards: list, future: dict,
                  memory: list) -> dict:
    return {"t": "frame", "time": time.time(), "level": level, "wave": int(wave), "frame": int(frame),
            "agents": agents, "hazards": hazards, "future": future, "memory": memory}


def plan_message(level: str, wave: int, plan_dict: dict) -> dict:
    return {**plan_dict, "t": "plan", "time": time.time(), "level": level, "wave": int(wave)}


def event_message(level: str | None, wave: int, ev: dict) -> dict:
    out = {"t": "event", "time": time.time(), "level": level, "wave": int(wave), "frame": ev.get("frame")}
    out.update(ev)
    out["t"] = "event"
    return out


def _shrink(msg: dict):
    """Smaller versions of a message, in order (the schema's shrink rules)."""
    t = msg.get("t")
    if t == "level" and msg.get("words") is not None:
        yield {**msg, "words": None}
    elif t == "frame":
        m = {**msg, "future": {}}
        yield m
        yield {**m, "agents": m["agents"][:60]}
    elif t == "plan":
        alts = [{k: v for k, v in a.items() if k != "path"} for a in msg.get("alternatives", [])]
        m = {**msg, "alternatives": alts}
        yield m
        m = {**m, "spread": []}
        yield m
        yield {**m, "alternatives": []}


class Telemetry:
    """Non-blocking UDP sender. send() never raises and never blocks; counts sent, dropped (could not send now)
    and too_big (no version of the message fits in MAX_BYTES)."""

    def __init__(self, addr=DEFAULT_ADDR, max_bytes: int = MAX_BYTES):
        self.addr = addr
        self.max_bytes = max_bytes
        self.sent = self.dropped = self.too_big = 0
        self.bytes = 0
        self.sock = None
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setblocking(False)
            if hasattr(socket, "SIO_UDP_CONNRESET"):    # Windows: no "connection reset" after a send to nobody
                try:
                    self.sock.ioctl(socket.SIO_UDP_CONNRESET, False)
                except (OSError, ValueError):
                    pass
        except OSError:
            self.sock = None

    def fit(self, msg: dict) -> bytes | None:
        """The datagram for msg (shrunk if needed), or None when nothing fits."""
        data = encode(msg)
        if len(data) <= self.max_bytes:
            return data
        for m in _shrink(msg):
            data = encode(m)
            if len(data) <= self.max_bytes:
                return data
        return None

    def send(self, msg: dict) -> bool:
        try:
            data = self.fit(msg)
        except Exception:
            self.dropped += 1
            return False
        if data is None:
            self.too_big += 1
            return False
        if self.sock is None:
            self.dropped += 1
            return False
        try:
            self.sock.sendto(data, self.addr)
        except Exception:          # buffer full (BlockingIOError), nobody listening, network errors: drop
            self.dropped += 1
            return False
        self.sent += 1
        self.bytes += len(data)
        return True

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
