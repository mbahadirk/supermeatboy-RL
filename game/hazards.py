"""Observed native kill-shape geometry, shared by PPO maps and reward routes.
Ported from main game/level.py; contains no predictions or action selection.
"""
import math
import numpy as np
from realpath import BODY, box_distance

SHAPE_BOX, SHAPE_CIRCLE, SHAPE_GAMEBOX, SHAPE_SEGMENT = 0, 1, 2, 3


def game_box_kills(px, py, a: float, b: float, ang: float, body: float = BODY) -> np.ndarray:
    """The game's box-box test for character centres at (px, py) relative to a kill box's centre (world frame):
    with D1 from the centre to the kill box (its own frame) and D2 from the kill box's centre to the character's
    axis-aligned box (half size body), it kills when px^2 + py^2 >= D1^2 + D2^2."""
    px, py = np.asarray(px, np.float64), np.asarray(py, np.float64)
    c, s = math.cos(ang), math.sin(ang)
    u, v = px * c + py * s, -px * s + py * c
    d1 = np.maximum(np.abs(u) - a, 0.0) ** 2 + np.maximum(np.abs(v) - b, 0.0) ** 2
    d2 = np.maximum(np.abs(px) - body, 0.0) ** 2 + np.maximum(np.abs(py) - body, 0.0) ** 2
    return px * px + py * py >= d1 + d2


def segment_chebyshev(dx, dy, hl, ang) -> np.ndarray:
    """Chebyshev distance (max(|ex|, |ey|)) from points at offsets (dx, dy) from a segment's middle to the segment of
    half length hl at angle ang (broadcast arrays). f(t) = max(|a + t u|, |b + t v|) on t in [-1, 1] is convex and
    piecewise linear: its minimum is at an end or where a piece changes (a term zero, or the two equal)."""
    u, v = hl * np.cos(ang), hl * np.sin(ang)
    a, b = -dx, -dy
    best = np.full(np.broadcast(a, b, u).shape, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for t in (np.full_like(best, -1.0), np.full_like(best, 1.0), -a / u, -b / v, (b - a) / (u - v),
                  -(a + b) / (u + v)):
            t = np.clip(np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
            best = np.minimum(best, np.maximum(np.abs(a + t * u), np.abs(b + t * v)))
    return best


def hazard_shapes(hz: list, keep=None) -> tuple[list, list, list]:
    """hazard_boxes with the game's own geometry: (boxes, kept, shape types). A measured kill shape keeps its exact
    geometry and type (kind 1 a SHAPE_CIRCLE (x, y, r, r, 0), kind 2 a SHAPE_GAMEBOX (x, y, a, b, angle)); a row
    without measured shapes keeps its size box as SHAPE_BOX (as hazard_boxes)."""
    out, kept, shape = [], [], []
    for k, h in enumerate(hz):
        if len(h) > 7 and h[7] is not None:
            for s in h[7]:
                kind, x, y, a, b, ang = s[:6]
                if int(kind) == 1:
                    out.append((float(x), float(y), float(a), float(a), 0.0))
                    shape.append(SHAPE_CIRCLE)
                elif int(kind) == 3:              # a laser beam [3, x0, y0, x1, y1, 0]
                    x0, y0, x1, y1 = float(x), float(y), float(a), float(b)
                    out.append((0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * math.hypot(x1 - x0, y1 - y0), 0.0,
                                math.atan2(y1 - y0, x1 - x0)))
                    shape.append(SHAPE_SEGMENT)
                else:
                    out.append((float(x), float(y), abs(float(a)), abs(float(b)), float(ang)))
                    shape.append(SHAPE_GAMEBOX)
                kept.append(k)
            continue
        if keep is not None and k < len(keep) and not keep[k]:
            continue
        x, y, w, hgt, sx, sy, ang = h[:7]
        sx, sy = abs(sx or 1.0), abs(sy or 1.0)
        if w <= 0 and hgt <= 0:
            w = hgt = 24.0
        out.append((x, y, 0.5 * w * sx, 0.5 * hgt * sy, ang))
        kept.append(k)
        shape.append(SHAPE_BOX)
    return out, kept, shape


def hazard_reaches(px,py,b,kind=SHAPE_BOX,body=BODY):
    """Whether an axis-aligned character box touches this native shape at each centre."""
    dx,dy=np.asarray(px)-b[0],np.asarray(py)-b[1]
    if kind==SHAPE_CIRCLE:
        return np.hypot(np.maximum(np.abs(dx)-body,0.),np.maximum(np.abs(dy)-body,0.))<=b[2]
    if kind==SHAPE_GAMEBOX:
        return game_box_kills(dx,dy,b[2],b[3],b[4],body)
    if kind==SHAPE_SEGMENT:
        return segment_chebyshev(dx,dy,b[2],b[4])<=body
    return box_distance(px,py,b)<body


def hazard_extent(b,kind=SHAPE_BOX,body=BODY):
    """Conservative search radius only; never used as the collision shape."""
    if kind==SHAPE_GAMEBOX:return 4.*(abs(b[2])+abs(b[3])+body)+10.
    return math.hypot(b[2],b[3])+math.sqrt(2.)*body
