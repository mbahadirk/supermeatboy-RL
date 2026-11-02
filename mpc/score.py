"""How good a predicted future is, in the PPO reward's units (a tile of progress = 1, a death = 15).

Per ensemble member, per candidate:
  progress  the drop of the remaining way to the goal, step by step (the route's closeness field, the field
            PPO is paid by; outside it the walking distance to the goal, counted worse than any point in it),
            weighted by the chance of being alive, discounted, and trusted less once the members disagree
            on the position by more than about a tile (their spread is about half the real error)
  key       reaching the route's next key (the field then switches to the next leg, as for PPO)
  landing   ending on the ground (or on a wall) rather than in the air
  value     the PPO critic at the predicted end state (the return after the horizon), if given
  death     the model's death chance, combined with a geometric check against the future hazard boxes
            (each member's own predicted position kept away by the body size + the hazard's uncertainty
            radius; how much the members disagree shows in the worst member), step by step. It is charged by when it happens: a death at the next step costs the full death
            weight, one at the end of the horizon (1 - LATE) of it. So when every candidate is predicted to die
            (a model that does not know this level's hazards, or a real dead end), dying later still beats
            dying sooner, and the planner keeps choosing instead of stalling.

Total = mean over members - kappa x std over members (plans the members disagree on lose) - death weight x
the worst member's timed death cost (an uncertain future counts as risky, not as safe).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from realpath import box_distance
from wm.features import BIN_NAMES

TILE = 20.0
BODY = 6.0               # game units: realgame.Level.grid_with_hazards marks cells this close to a hazard
WAYPOINT_R = 30.0        # realgame.WAYPOINT_R
ON_WALL = BIN_NAMES.index("on wall")
LATE = 0.5               # a death at the end of the horizon costs (1 - LATE) x the death weight


@dataclass
class Weights:
    progress: float = 1.0      # per tile
    death: float = 15.0        # realgame.R_DEATH
    key: float = 10.0          # realgame.R_KEY
    land: float = 1.0
    wall: float = 0.5
    value: float = 0.5
    kappa: float = 1.0
    gamma: float = 0.98        # per frame
    trust_tiles: float = 1.0   # 2 x spread beyond this many tiles: credit fades


def score(out: dict, rem_fn, w: Weights, check=None, rem_next_fn=None, waypoint=None, value_end=None) -> dict:
    """out: wm.rollout.rollout_batched result; rem_fn(x, y) -> remaining way (game units, same shape);
    check: [(boxes (M, 5), radius (M,))] for steps 1..H; value_end: (E, n) critic values at the end."""
    K = int(out.get("steps", out["x"].shape[2] - 1))
    x, y = out["x"][:, :, :K + 1].astype(np.float64), out["y"][:, :, :K + 1].astype(np.float64)
    alive = out["alive"][:, :, :K + 1]
    E, n, _ = x.shape
    spread = np.sqrt(x.var(0) + y.var(0))                                   # (n, K+1)
    trust = np.exp(-np.maximum(0.0, 2.0 * spread - w.trust_tiles * TILE) / TILE)
    rem = rem_fn(x, y)
    key = np.zeros((E, n))
    if waypoint is not None and rem_next_fn is not None:
        d = np.hypot(x - waypoint[0], y - waypoint[1])
        hit = d < WAYPOINT_R
        kr = np.where(hit.any(2), hit.argmax(2), K + 1)                     # first step at the key (E, n)
        rem2 = rem_next_fn(x, y)
        steps = np.arange(K + 1)[None, None, :]
        at = np.take_along_axis(rem, np.minimum(kr, K)[..., None], 2) - np.take_along_axis(rem2, np.minimum(kr, K)[..., None], 2)
        rem = np.where(steps >= kr[..., None], rem2 + at, rem)              # continuous where the leg switches
        key = np.where(kr <= K, w.key * np.take_along_axis(alive, np.minimum(kr, K)[..., None], 2)[..., 0], 0.0)
    disc = w.gamma ** np.arange(1, K + 1)
    gain = (rem[:, :, :-1] - rem[:, :, 1:]) / TILE                          # tiles per step (E, n, K)
    # geometric hazard check: the chance of not having touched a hazard by each step
    geom_free = np.ones((E, n, K + 1))
    hit_free = np.ones((E, n))
    if check is not None:
        for k in range(1, K + 1):
            boxes, radius = check[k - 1]
            if not len(boxes):
                continue
            for b, r in zip(boxes, radius):                                 # each member's own position: their
                d = box_distance(x[:, :, k], y[:, :, k], b)                 # disagreement is in the worst member
                p = 1.0 / (1.0 + np.exp(-(BODY + r - d) / 2.0))
                hit_free *= 1.0 - np.where(d > BODY + r + 20.0, 0.0, p)
            geom_free[:, :, k] = hit_free
    for k in range(1, K + 1):                                               # (steps with no boxes keep the last)
        if check is None or not len(check[k - 1][0]):
            geom_free[:, :, k] = geom_free[:, :, k - 1]
    surv = alive * geom_free                                                # alive by each step (E, n, K+1)
    progress = w.progress * (gain * surv[:, :, 1:] * trust[None, :, 1:] * disc).sum(2)
    d_model = 1.0 - alive[:, :, K]
    d_geom = 1.0 - geom_free[:, :, K]
    death = 1.0 - surv[:, :, K]
    dies = surv[:, :, :-1] - surv[:, :, 1:]                                 # chance of dying at each step
    timing = 1.0 - LATE * np.arange(1, K + 1) / max(1, K)
    death_cost = (dies * timing).sum(2)                                     # (E, n)
    t_death = np.where(death > 1e-6, (dies * np.arange(1, K + 1)).sum(2) / np.maximum(death, 1e-6), np.nan)
    air_end = out["airborne"][:, :, K]
    p_wall = out["p"][:, :, K - 1, ON_WALL] if K > 0 else np.zeros((E, n))
    land = (1.0 - air_end) * w.land + air_end * p_wall * w.wall
    value = np.zeros((E, n)) if value_end is None else w.value * (w.gamma ** K) * surv[:, :, K] * value_end
    J = progress + key + land + value
    total = J.mean(0) - w.kappa * J.std(0) - w.death * death_cost.max(0)
    # first predicted landing of the members' mean path (for logs / the viewer)
    air = out["airborne"][:, :, :K + 1].mean(0) > 0.5
    land_k = np.where((air[:, :-1] & ~air[:, 1:]).any(1), (air[:, :-1] & ~air[:, 1:]).argmax(1) + 1, -1)
    return {"total": total, "progress": progress.mean(0), "key": key.mean(0), "land": land.mean(0),
            "value": value.mean(0), "unc": w.kappa * J.std(0), "death": death.max(0), "death_model": d_model.max(0),
            "death_cost": w.death * death_cost.max(0), "death_step": _earliest(t_death),
            "death_geom": d_geom.max(0), "spread_end": spread[:, K], "landing_step": land_k,
            "mean_x": x.mean(0), "mean_y": y.mean(0), "spread": spread, "steps": K}


def _earliest(t_death: np.ndarray) -> np.ndarray:
    """The earliest expected death step over the members per candidate; -1 where no member predicts one."""
    t = np.where(np.isnan(t_death), np.inf, t_death).min(0)
    return np.where(np.isfinite(t), t, -1.0)
