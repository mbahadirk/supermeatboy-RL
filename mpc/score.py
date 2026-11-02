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
            (kept away from by the body size + the hazard's uncertainty radius + twice the member spread)

Total = mean over members - kappa x std over members (plans the members disagree on lose) - death weight x
the worst member's death chance (an uncertain future counts as risky, not as safe).
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
    progress = w.progress * (gain * alive[:, :, 1:] * trust[None, :, 1:] * disc).sum(2)
    # geometric hazard check
    hit_free = np.ones((E, n))
    if check is not None:
        for k in range(1, K + 1):
            boxes, radius = check[k - 1]
            if not len(boxes):
                continue
            margin = BODY + 2.0 * spread[:, k]                              # (n,)
            for b, r in zip(boxes, radius):
                d = box_distance(x[:, :, k], y[:, :, k], b)
                p = 1.0 / (1.0 + np.exp(-(margin[None, :] + r - d) / 2.0))
                hit_free *= 1.0 - np.where(d > margin[None, :] + r + 20.0, 0.0, p)
    d_model = 1.0 - alive[:, :, K]
    d_geom = 1.0 - hit_free
    death = 1.0 - (1.0 - d_model) * (1.0 - d_geom)
    air_end = out["airborne"][:, :, K]
    p_wall = out["p"][:, :, K - 1, ON_WALL] if K > 0 else np.zeros((E, n))
    land = (1.0 - air_end) * w.land + air_end * p_wall * w.wall
    value = np.zeros((E, n)) if value_end is None else w.value * (w.gamma ** K) * alive[:, :, K] * value_end
    J = progress + key + land + value
    total = J.mean(0) - w.kappa * J.std(0) - w.death * death.max(0)
    # first predicted landing of the members' mean path (for logs / the viewer)
    air = out["airborne"][:, :, :K + 1].mean(0) > 0.5
    land_k = np.where((air[:, :-1] & ~air[:, 1:]).any(1), (air[:, :-1] & ~air[:, 1:]).argmax(1) + 1, -1)
    return {"total": total, "progress": progress.mean(0), "key": key.mean(0), "land": land.mean(0),
            "value": value.mean(0), "unc": w.kappa * J.std(0), "death": death.max(0), "death_model": d_model.max(0),
            "death_geom": d_geom.max(0), "spread_end": spread[:, K], "landing_step": land_k,
            "mean_x": x.mean(0), "mean_y": y.mean(0), "spread": spread, "steps": K}
