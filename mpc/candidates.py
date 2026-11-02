"""Action sequences for the planner to try, each labelled with where it came from.

  prev   the previous best plan, shifted to now (warm start), and mutations of it (one block of frames
         changed, the jump released earlier / later, the direction changed from some frame on)
  ppo    the policy's most likely actions now, held; and each of them followed by the previous plan
  prim   moves from a library: wait, walk / run left or right, a jump held 3-22 frames after a delay of
         0-10 frames, two jumps in a row (wall jumps: the same or the other way), a jump and then turning
         back in the air. A random subset each decision, so different combinations get tried.

A sequence is an array of realgame.ACTIONS indices, one per frame.
"""
from __future__ import annotations

import numpy as np

# realgame.ACTIONS: (left, right, jump, run); tests/test_mpc.py checks they match
ACTIONS = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (1, 0, 1, 0), (0, 1, 1, 0),
           (0, 1, 0, 1), (0, 1, 1, 1), (1, 0, 0, 1), (1, 0, 1, 1)]
_IDX = {a: i for i, a in enumerate(ACTIONS)}


def act(d: int, jump: bool = False, run: bool = False) -> int:
    """d: -1 left, 0 none, 1 right (no run without a direction)."""
    return _IDX[(int(d < 0), int(d > 0), int(bool(jump)), int(bool(run) and d != 0))]


def library(H: int) -> list[tuple[str, np.ndarray]]:
    out = [("wait", np.full(H, act(0)))]
    for d in (-1, 1):
        side = "R" if d > 0 else "L"
        for run in (False, True):
            tag = side + ("run" if run else "walk")
            out.append((tag, np.full(H, act(d, run=run))))
            for j in (3, 6, 10, 15, 22):
                for delay in (0, 3, 6, 10):
                    if delay >= H:
                        continue
                    s = np.full(H, act(d, run=run))
                    s[delay:delay + j] = act(d, True, run)
                    out.append((f"{tag} jump{j}@{delay}", s))
            for j1 in (4, 8):                 # two jumps (wall jumps): again the same way or the other way
                for gap in (1, 3):
                    for d2 in (d, -d):
                        s = np.full(H, act(d2, run=run))
                        s[:j1] = act(d, True, run)
                        s[j1:j1 + gap] = act(d, False, run)
                        s[j1 + gap:j1 + gap + 12] = act(d2, True, run)
                        out.append((f"{tag} jump{j1} gap{gap} jump{'same' if d2 == d else 'back'}", s))
            for j in (6, 12):                 # jump, then turn back in the air
                s = np.full(H, act(-d, run=run))
                s[:j] = act(d, True, run)
                out.append((f"{tag} jump{j} turn", s))
    for j in (6, 15):                         # straight up
        s = np.full(H, act(0))
        s[:j] = act(0, True)
        out.append((f"up jump{j}", s))
    return out


_LIB: dict[int, list] = {}
BASIC = ("wait", "Lrun", "Rrun", "Lrun jump15@0", "Rrun jump15@0")


def make(n: int, H: int, rng: np.random.Generator, prev: np.ndarray | None, probs: np.ndarray | None,
         n_mut: int = 6, n_ppo: int = 3, block: int = 3) -> tuple[np.ndarray, list[str]]:
    """About n sequences of H frames (fewer if some are the same), with their labels."""
    if H not in _LIB:
        _LIB[H] = library(H)
    lib = _LIB[H]
    seqs, labels = [], []

    def add(s, lab):
        seqs.append(np.asarray(s, np.int64)[:H])
        labels.append(lab)
    if prev is not None:
        p = np.concatenate([prev, np.full(max(0, H - len(prev)), prev[-1] if len(prev) else 0)])[:H]
        add(p, "prev")
        for m in range(n_mut):
            q = p.copy()
            kind = m % 3
            t = int(rng.integers(0, max(1, H - block)))
            if kind == 0:                     # one block of frames changed to another action
                q[t:t + block] = rng.integers(0, len(ACTIONS))
                add(q, "prev:block")
            elif kind == 1:                   # jump pressed / released from some frame on, for a while
                L = int(rng.integers(2, 8))
                for f in range(t, min(H, t + L)):
                    l, r, j, run = ACTIONS[q[f]]
                    q[f] = act(r - l, not j, run)
                add(q, "prev:jump")
            else:                             # the other direction from some frame on
                for f in range(t, H):
                    l, r, j, run = ACTIONS[q[f]]
                    q[f] = act(l - r, j, run)
                add(q, "prev:turn")
    if probs is not None:
        top = np.argsort(probs)[::-1][:n_ppo]
        for a in top:
            add(np.full(H, a), f"ppo:{int(a)} held")
        if prev is not None:
            for a in top[:2]:
                q = np.concatenate([[a], p[1:]])
                add(q, f"ppo:{int(a)} then prev")
    names = {name: s for name, s in lib}
    for name in BASIC:
        add(names[name], name)
    rest = n - len(seqs)
    if rest > 0:
        pick = rng.choice(len(lib), min(rest, len(lib)), replace=False)
        for k in pick:
            add(lib[k][1], lib[k][0])
    # drop repeats, keep the first label
    arr = np.stack(seqs)
    _, first = np.unique(arr, axis=0, return_index=True)
    first = np.sort(first)
    return arr[first], [labels[i] for i in first]
