"""The planner's move library: named primitives, each a short sequence of realgame ACTIONS indices (one per
frame; the (left, right, jump, run) table in mpc.candidates.ACTIONS).

Families (73 moves, 8 to 24 frames):
- wait 8f / 16f: no input;
- walk / run L / R for 8, 16, 24 frames;
- jump up hold H: jump straight up, the key held H frames (3, 6, 10, 15, 22), then released;
- jump L / R hold H: a walking jump (no run);
- run-jump L / R hold H: a running jump; with "after Nf": running N frames before the press (after 4f: holds 6,
  10, 15; after 8f: 6, 10, so the move stays within 24 frames);
- wall-jump L / R hold H: one frame with the jump key released (so the press is a new press), then jump with
  the direction AWAY from the wall (a wall on the right -> "wall-jump L"), held 6, 8, 10 or 15; "run wall-jump"
  with run held (8, 15);
- wall-climb L / R hold H: the same press TOWARD the wall (L = a wall on the left): the short climbing hop of
  wall-jump chains (hold 6, 10, 15);
- fall drift L / R 12f: no jump, run in that direction (leave a ledge or a wall, steer while falling);
- jump L / R hold H turn: jump that way, then steer back the other way once the key is released (hold 6, 12).

Every jump move ends with at least 2 frames without the jump key, so a following move's press is a new press,
and the move is padded (its last action held) to the planner's horizon with no jump held.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mpc.candidates import ACTIONS, act

HOLDS = (3, 6, 10, 15, 22)
MIN_LEN, MAX_LEN = 8, 24
TAIL = 6                      # frames after a jump's release (steering), within MIN_LEN .. MAX_LEN


@dataclass(frozen=True)
class Move:
    name: str
    actions: tuple            # ACTIONS indices, one per frame
    family: str

    def __len__(self):
        return len(self.actions)

    @property
    def array(self) -> np.ndarray:
        return np.asarray(self.actions, np.int64)


def _side(d: int) -> str:
    return "R" if d > 0 else "L"


def _jump(d: int, hold: int, delay: int = 0, run: bool = False, after: int | None = None) -> tuple:
    """`delay` frames moving (d, run), then jump held `hold` frames, then moving `after` direction (default d)
    to the move's length (hold + delay + TAIL, within MIN_LEN .. MAX_LEN, at least 2 frames released)."""
    n = int(np.clip(delay + hold + TAIL, MIN_LEN, MAX_LEN))
    n = max(n, delay + hold + 2)
    after = d if after is None else after
    a = [act(d, run=run)] * delay + [act(d, True, run)] * hold
    a += [act(after, run=run and after != 0)] * (n - len(a))
    return tuple(a)


def library() -> list[Move]:
    out = [Move(f"wait {n}f", (act(0),) * n, "wait") for n in (8, 16)]
    for d in (-1, 1):
        s = _side(d)
        for run in (False, True):
            for n in (8, 16, 24):
                out.append(Move(f"{'run' if run else 'walk'} {s} {n}f", (act(d, run=run),) * n, "run" if run else "walk"))
    for h in HOLDS:
        out.append(Move(f"jump up hold {h}", _jump(0, h), "jump up"))
    for d in (-1, 1):
        s = _side(d)
        for h in HOLDS:
            out.append(Move(f"jump {s} hold {h}", _jump(d, h), "jump"))
        for h in HOLDS:
            out.append(Move(f"run-jump {s} hold {h}", _jump(d, h, run=True), "run-jump"))
        for delay, holds in ((4, (6, 10, 15)), (8, (6, 10))):
            for h in holds:
                out.append(Move(f"run-jump {s} hold {h} after {delay}f", _jump(d, h, delay, run=True), "run-jump"))
        for h in (6, 8, 10, 15):            # away from a wall on the other side
            out.append(Move(f"wall-jump {s} hold {h}", (act(d),) + _jump(d, h)[:-1], "wall-jump"))
        for h in (8, 15):
            out.append(Move(f"run wall-jump {s} hold {h}", (act(d, run=True),) + _jump(d, h, run=True)[:-1],
                            "wall-jump"))
        for h in (6, 10, 15):               # toward the wall (climbing chains)
            out.append(Move(f"wall-climb {s} hold {h}", (act(d),) + _jump(d, h)[:-1], "wall-climb"))
        out.append(Move(f"fall drift {s} 12f", (act(d, run=True),) * 12, "fall"))
        for h in (6, 12):
            out.append(Move(f"jump {s} hold {h} turn", _jump(d, h, after=-d), "turn"))
    return out


LIBRARY = library()
BY_NAME = {m.name: m for m in LIBRARY}
# stage 2's fixed core of second moves (the rest of its 16 are the best-scoring first moves of the decision)
CORE_SECOND = ("wait 8f", "run L 16f", "run R 16f", "run-jump L hold 15", "run-jump R hold 15",
               "wall-jump L hold 8", "wall-jump R hold 8", "jump up hold 15")


def padded(actions, n: int) -> np.ndarray:
    """Actions trimmed / padded to n frames (the last action held; a held jump is released in the padding)."""
    a = np.asarray(actions, np.int64)
    if len(a) >= n:
        return a[:n].copy()
    last = int(a[-1]) if len(a) else act(0)
    l, r, j, run = ACTIONS[last]
    fill = act(r - l, False, run)
    return np.concatenate([a, np.full(n - len(a), fill, np.int64)])


def describe(actions) -> str:
    """A compact reading of an action sequence: runs of the same action, e.g. 'R+run x4, R+run+jump x10'."""
    a = list(np.asarray(actions, np.int64))
    parts, i = [], 0
    while i < len(a):
        j = i
        while j < len(a) and a[j] == a[i]:
            j += 1
        l, r, jump, run = ACTIONS[a[i]]
        tag = ("L" if l else "R" if r else "-") + ("+run" if run else "") + ("+jump" if jump else "")
        parts.append(f"{tag} x{j - i}")
        i = j
    return ", ".join(parts)
