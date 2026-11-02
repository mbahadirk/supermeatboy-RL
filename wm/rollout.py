"""Multi-step rollouts with the world model: many agents (or candidate action sequences) at once, K steps.

Each step feeds the model's own prediction back in: position, velocity, the flag bits that change during
play (features.FLAG_BITS; the others are kept), the movement state and the jump key (held as realgame.js
holds it, re-pressed on walls). The tiles are read at the predicted position; hazards, platforms and wind
come from `worlds` (one per future step). Every ensemble member rolls out its own trajectory, so the spread
between them grows where the model is unsure (trajectory sampling, as in PETS).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from wm.features import (featurize, prior, hazard_boxes, RES_SCALE, FLAG_BITS, EVENTS, BIN_NAMES, AIR, WALL,
                         DT_DEFAULT)

WALL_REPRESS = 6                                   # realgame.js / realgame.py
ACT_JUMP = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1], bool)   # realgame.ACTIONS: which actions hold jump


@dataclass
class World:
    """The level's moving parts at one frame."""
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))
    pf: list | None = None
    pf_prev: list | None = None
    dt_prev_s: float = DT_DEFAULT / 1000.0
    fans: list = field(default_factory=list)

    @classmethod
    def from_record(cls, rec: dict, prev: dict | None, keep=None, fans=()):
        return cls(hazard_boxes(rec.get("hz", []), keep), rec.get("pf"), (prev or {}).get("pf"),
                   ((prev or {}).get("dt") or DT_DEFAULT) / 1000.0, list(fans))


def next_held(held, action, flags):
    """The jump key after this action, as realgame.js applies it."""
    jump = ACT_JUMP[np.clip(action, 0, len(ACT_JUMP) - 1)]
    repress = jump & (held >= WALL_REPRESS) & ((flags & WALL) != 0) & ((flags & AIR) != 0)
    return np.where(~jump | repress, 0, held + 1).astype(np.int16)


def advance(s: np.ndarray, a, move: np.ndarray, pb: np.ndarray, state_cls: np.ndarray) -> np.ndarray:
    """The next agent rows from one predicted step: move [dx, dy, vx', vy'] (game units), probabilities of
    the binary heads (the flag bits are set where > 0.5, other bits kept), the next movement state."""
    s2 = s.copy()
    s2["x"] = s["x"] + move[:, 0]
    s2["y"] = s["y"] + move[:, 1]
    s2["vx"], s2["vy"] = move[:, 2], move[:, 3]
    flags = s["flags"].astype(np.uint64)
    for j, bit in enumerate(FLAG_BITS):
        flags = np.where(pb[:, j] > 0.5, flags | np.uint64(bit), flags & ~np.uint64(bit))
    s2["flags"] = flags.astype(np.uint32)
    s2["state"] = state_cls
    s2["held"] = next_held(s["held"], a, s["flags"])
    return s2


@torch.no_grad()
def rollout_mean(model, geom, s0, actions, dts, worlds):
    """One rollout per agent with the ensemble's mean prediction; returns the inputs the model saw at each
    step and the rows it predicted ([K] features, [K+1] rows). For training on its own predictions."""
    n, K = actions.shape
    dts = np.asarray(dts, np.float64)
    s, feats, states = s0.copy(), [], [s0.copy()]
    rs = torch.as_tensor(RES_SCALE)
    for k in range(K):
        a, dt, w = actions[:, k], (dts[k] if dts.ndim == 1 else dts[:, k]), worlds[k]
        X = featurize(s, a, dt, geom, w.boxes, w.pf, w.pf_prev, w.dt_prev_s, w.fans)
        p = model.predict(torch.as_tensor(X))
        move = (p["mean"] * rs).numpy() + prior(s, dt)
        s = advance(s, a, move, p["p"].numpy(), p["state"].argmax(-1).numpy())
        feats.append(X)
        states.append(s.copy())
    return feats, states


@torch.no_grad()
def rollout(model, geom, s0: np.ndarray, actions: np.ndarray, dts, worlds: list) -> dict:
    """s0: agent rows (wm.buffer.ROW, n); actions (n, K); dts: ms, (K,) or (n, K); worlds: K World.
    Returns per member e, agent i, step k (index 0 = start):
      x, y, vx, vy   (E, n, K+1) game units
      alive          (E, n, K+1) probability of not having died yet
      p              (E, n, K, BIN_NAMES) probabilities after each step (flags, death, finish, landing, blocked)
      std            (E, n, K, 4) the member's own uncertainty of each step (dx, dy, vx, vy)
      airborne       (E, n, K+1) predicted flag"""
    E = len(model.members)
    n, K = actions.shape
    dts = np.asarray(dts, np.float64)
    out = {k: np.zeros((E, n, K + 1), np.float32) for k in ("x", "y", "vx", "vy", "alive", "airborne")}
    out["p"] = np.zeros((E, n, K, len(BIN_NAMES)), np.float32)
    out["std"] = np.zeros((E, n, K, 4), np.float32)
    rs = torch.as_tensor(RES_SCALE)
    for e, member in enumerate(model.members):
        s = s0.copy()
        alive = np.ones(n, np.float32)
        for name in ("x", "y", "vx", "vy"):
            out[name][e, :, 0] = s[name]
        out["alive"][e, :, 0] = 1.0
        out["airborne"][e, :, 0] = (s["flags"] & AIR) != 0
        for k in range(K):
            a = actions[:, k]
            dt = dts[k] if dts.ndim == 1 else dts[:, k]
            w = worlds[k]
            X = torch.as_tensor(featurize(s, a, dt, geom, w.boxes, w.pf, w.pf_prev, w.dt_prev_s, w.fans))
            h = model.split(member((X - model.mu) / model.sd))
            move = (h["mean"] * rs).numpy() + prior(s, dt)
            pb = torch.sigmoid(h["bin"]).numpy()
            s2 = advance(s, a, move, pb, h["state"].argmax(-1).numpy())
            alive = alive * (1.0 - pb[:, EVENTS["death"]])
            for name in ("x", "y", "vx", "vy"):
                out[name][e, :, k + 1] = s2[name]
            out["alive"][e, :, k + 1] = alive
            out["airborne"][e, :, k + 1] = (s2["flags"] & AIR) != 0
            out["p"][e, :, k] = pb
            out["std"][e, :, k] = (torch.exp(0.5 * h["logvar"]) * rs).numpy()
            s = s2
    return out
