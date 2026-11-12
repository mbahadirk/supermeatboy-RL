"""Multi-step rollouts with the world model: many agents (or candidate action sequences) at once, K steps.

Each step feeds the model's own prediction back in: position, velocity, the flag bits that change during
play (features.FLAG_BITS; the others are kept), the movement state and the jump key (held as realgame.js
holds it, re-pressed on walls). The tiles are read at the predicted position; hazards, platforms and wind
come from `worlds` (one per future step). Each step adds the physics prior the checkpoint was trained with
(wm.features.model_prior; its fans from that step's world). Every ensemble member rolls out its own
trajectory, so the spread between them grows where the model is unsure (trajectory sampling, as in PETS).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from wm.features import (featurize, featurize_k, prior, hazard_boxes, RES_SCALE, FLAG_BITS, EVENTS, BIN_NAMES, AIR,
                         WALL, DT_DEFAULT, model_prior, require_geom)
from game.level import key_pickup_policy

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


_BITS = np.array(FLAG_BITS, np.uint64)
_KEEP = ~np.uint64(int(sum(FLAG_BITS)))


def advance(s: np.ndarray, a, move: np.ndarray, pb: np.ndarray, state_cls: np.ndarray) -> np.ndarray:
    """The next agent rows from one predicted step: move [dx, dy, vx', vy'] (game units), probabilities of
    the binary heads (the flag bits are set where > 0.5, other bits kept), the next movement state. a: the
    step's action (n,), or its k actions (n, k) for a step-k model (the jump key then follows all k, with
    the start's flags)."""
    s2 = s.copy()
    s2["x"] = s["x"] + move[:, 0]
    s2["y"] = s["y"] + move[:, 1]
    s2["vx"], s2["vy"] = move[:, 2], move[:, 3]
    nb = len(FLAG_BITS)                        # each predicted bit set where > 0.5, cleared otherwise (one pass)
    on = (pb[:, :nb] > 0.5) @ _BITS                     # the bits to set (distinct powers of two: OR = sum)
    s2["flags"] = ((s["flags"].astype(np.uint64) & _KEEP) | on).astype(np.uint32)
    s2["state"] = state_cls
    if np.ndim(a) == 2:
        held = s["held"]
        for j in range(a.shape[1]):
            held = next_held(held, a[:, j], s["flags"])
        s2["held"] = held
    else:
        s2["held"] = next_held(s["held"], a, s["flags"])
    return s2


def step_actions(actions: np.ndarray, k: int) -> np.ndarray:
    """Per-frame actions (n, K) as a step-k model's steps (n, ceil(K / k), k): the last action repeats to
    fill the last step."""
    n, K = actions.shape
    Ks = -(-K // k)
    if Ks * k > K:
        actions = np.concatenate([actions, np.repeat(actions[:, -1:], Ks * k - K, 1)], 1)
    return actions.reshape(n, Ks, k)


@torch.no_grad()
def rollout_mean(model, geom, s0, actions, dts, worlds, key_pickup=False):
    """One rollout per agent with the ensemble's mean prediction; returns the inputs the model saw at each
    step and the rows it predicted ([K] features, [K+1] rows). For training on its own predictions."""
    require_geom(model, geom)
    if key_pickup and getattr(model, "step", 1) != 1:
        raise ValueError("rollout_mean supports step1; use rollout_batched for coarse key pickup")
    n, K = actions.shape
    dts = np.asarray(dts, np.float64)
    pp = model_prior(model)
    s, feats, states = s0.copy(), [], [s0.copy()]
    rs = torch.as_tensor(RES_SCALE)
    for k in range(K):
        a, dt, w = actions[:, k], (dts[k] if dts.ndim == 1 else dts[:, k]), worlds[k]
        X = featurize(s, a, dt, geom, w.boxes, w.pf, w.pf_prev, w.dt_prev_s, w.fans)
        p = model.predict(torch.as_tensor(X))
        move = (p["mean"] * rs).numpy() + prior(s, dt, 1, w.fans, pp)
        s2 = advance(s, a, move, p["p"].numpy(), p["state"].argmax(-1).numpy())
        if key_pickup:
            geom.lock_context.pickup_after_step(s, s2)
        s = s2
        feats.append(X)
        states.append(s.copy())
    return feats, states


@torch.no_grad()
def rollout(model, geom, s0: np.ndarray, actions: np.ndarray, dts, worlds: list, key_pickup=False) -> dict:
    """s0: agent rows (wm.buffer.ROW, n); actions (n, K); dts: ms, (K,) or (n, K); worlds: K World.
    Returns per member e, agent i, step k (index 0 = start):
      x, y, vx, vy   (E, n, K+1) game units
      alive          (E, n, K+1) probability of not having died yet
      p              (E, n, K, BIN_NAMES) probabilities after each step (flags, death, finish, landing, blocked)
      std            (E, n, K, 4) the member's own uncertainty of each step (dx, dy, vx, vy)
      airborne       (E, n, K+1) predicted flag"""
    require_geom(model, geom)
    if key_pickup and getattr(model, "step", 1) != 1:
        raise ValueError("rollout supports step1; use rollout_batched for coarse key pickup")
    E = len(model.members)
    n, K = actions.shape
    dts = np.asarray(dts, np.float64)
    pp = model_prior(model)
    out = {k: np.zeros((E, n, K + 1), np.float32) for k in ("x", "y", "vx", "vy", "alive", "airborne")}
    out["p"] = np.zeros((E, n, K, len(BIN_NAMES)), np.float32)
    out["std"] = np.zeros((E, n, K, 4), np.float32)
    if key_pickup:
        out["keys"] = np.zeros((E, n, K + 1), np.uint32)
        out["keys"][:, :, 0] = s0["keys"]
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
            move = (h["mean"] * rs).numpy() + prior(s, dt, 1, w.fans, pp)
            pb = torch.sigmoid(h["bin"]).numpy()
            s2 = advance(s, a, move, pb, h["state"].argmax(-1).numpy())
            if key_pickup:
                geom.lock_context.pickup_after_step(s, s2)
                out["keys"][e, :, k + 1] = s2["keys"]
            alive = alive * (1.0 - pb[:, EVENTS["death"]])
            for name in ("x", "y", "vx", "vy"):
                out[name][e, :, k + 1] = s2[name]
            out["alive"][e, :, k + 1] = alive
            out["airborne"][e, :, k + 1] = (s2["flags"] & AIR) != 0
            out["p"][e, :, k] = pb
            out["std"][e, :, k] = (torch.exp(0.5 * h["logvar"]) * rs).numpy()
            s = s2
    if key_pickup:
        out["key_pickup"] = key_pickup_policy(1)
    return out


def _stacked(model):
    """Every member's layer weights stacked (members, in, out), for one batched matmul per layer. Rebuilt
    when the weights change (torch counts in-place updates in _version)."""
    ver = tuple(p._version for p in model.parameters())
    c = getattr(model, "_stack_cache", None)
    if c is None or c[0] != ver:
        lins = [[m for m in mem if isinstance(m, torch.nn.Linear)] for mem in model.members]
        W = [torch.stack([l[i].weight.detach() for l in lins]).transpose(1, 2).contiguous() for i in range(len(lins[0]))]
        B = [torch.stack([l[i].bias.detach() for l in lins])[:, None, :].contiguous() for i in range(len(lins[0]))]
        c = model._stack_cache = (ver, W, B)
    return c[1], c[2]


@torch.no_grad()
def rollout_batched(model, geom, s0: np.ndarray, actions: np.ndarray, dts, worlds: list, deadline: float | None = None,
                    keep_rows: bool = False, key_pickup=False) -> dict:
    """rollout() with all members in one batched call per step (2-3x faster, same numbers). deadline
    (time.perf_counter() value): stop after the step that passes it; out["steps"] says how many steps were
    rolled out (the arrays keep their full size; the steps after it are left as they were at the stop).
    keep_rows: also out["rows"] (E, n, K+1) agent rows (flags, state, jump key) for the planner.
    A step-k model (model.step > 1): actions / dts / worlds stay per frame (K frames); the arrays are per
    model step, ceil(K / k) of them (column j = frame j x k; out["step"] = k), the world at each step's first
    frame."""
    import time
    require_geom(model, geom)
    pp = model_prior(model)
    E = len(model.members)
    f = getattr(model, "step", 1)
    if f > 1:                                  # a step-k model: K frames -> ceil(K / k) steps (module docstring)
        from wm.features import actions_into
        dts = np.asarray(dts, np.float64)
        sa = step_actions(actions, f)
        K = sa.shape[1]
        dfr = np.concatenate([dts, np.repeat(dts[-1:], K * f - len(dts))]) if dts.ndim == 1 else None
        dts = dfr.reshape(K, f).sum(1) if dts.ndim == 1 else             np.concatenate([dts, np.repeat(dts[:, -1:], K * f - dts.shape[1], 1)], 1).reshape(len(dts), K, f).sum(2)
        worlds = [worlds[min(j * f, len(worlds) - 1)] for j in range(K)]
        actions = sa
    else:
        actions_into = None
    n, K = actions.shape[:2]
    dts = np.asarray(dts, np.float64)
    W, B = _stacked(model)
    s = np.concatenate([s0] * E)                          # member-major: rows e*n .. e*n+n-1 are member e
    a_all = np.tile(actions, (E, 1) + (1,) * (actions.ndim - 2))
    out = {k: np.zeros((E, n, K + 1), np.float32) for k in ("x", "y", "vx", "vy", "alive", "airborne")}
    out["p"] = np.zeros((E, n, K, len(BIN_NAMES)), np.float32)
    out["std"] = np.zeros((E, n, K, 4), np.float32)
    if key_pickup:
        out["keys"] = np.zeros((E, n, K + 1), np.uint32)
    rows = np.zeros((E, n, K + 1), s0.dtype) if keep_rows else None
    alive = np.ones(E * n, np.float32)
    rs = torch.as_tensor(getattr(model, "res_scale", RES_SCALE))

    def put(k, s, alive):
        for name in ("x", "y", "vx", "vy"):
            out[name][:, :, k] = s[name].reshape(E, n)
        out["alive"][:, :, k] = alive.reshape(E, n)
        out["airborne"][:, :, k] = ((s["flags"] & AIR) != 0).reshape(E, n)
        if rows is not None:
            rows[:, :, k] = s.reshape(E, n)
        if key_pickup:
            out["keys"][:, :, k] = s["keys"].reshape(E, n)
    put(0, s, alive)
    done = K
    for k in range(K):
        dt = dts[k] if dts.ndim == 1 else np.tile(dts[:, k], E)
        w = worlds[k]
        if actions_into is not None:
            X = torch.as_tensor(featurize_k(s, a_all[:, k], dt, geom, w.boxes, w.pf, w.pf_prev, w.dt_prev_s, w.fans))
        else:
            X = torch.as_tensor(featurize(s, a_all[:, k], dt, geom, w.boxes, w.pf, w.pf_prev, w.dt_prev_s, w.fans))
        z = ((X - model.mu) / model.sd).view(E, n, -1)
        for i in range(len(W) - 1):
            z = torch.nn.functional.silu(torch.baddbmm(B[i], z, W[i]))
        h = model.split(torch.baddbmm(B[-1], z, W[-1]).reshape(E * n, -1))
        move = (h["mean"] * rs).numpy() + prior(s, dt, f, w.fans, pp)
        pb = torch.sigmoid(h["bin"]).numpy()
        s2 = advance(s, a_all[:, k], move, pb, h["state"].argmax(-1).numpy())
        if key_pickup:
            geom.lock_context.pickup_after_step(s, s2, f)
        s = s2
        alive = alive * (1.0 - pb[:, EVENTS["death"]])
        put(k + 1, s, alive)
        out["p"][:, :, k] = pb.reshape(E, n, -1)
        out["std"][:, :, k] = (torch.exp(0.5 * h["logvar"]) * rs).numpy().reshape(E, n, 4)
        if deadline is not None and time.perf_counter() > deadline and k + 1 < K:
            done = k + 1
            break
    out["steps"] = done
    out["step"] = f
    if rows is not None:
        out["rows"] = rows
    if key_pickup:
        out["key_pickup"] = key_pickup_policy(f)
    return out
