"""Same-exit comparison for a local crossing; incomplete waits are not completed crossings."""
from __future__ import annotations

import numpy as np

RADIUS = 16.0
WINDOW = 8
TEMPLATES = 4


def at_exit(x, y, supported, masks, target):
    """One exit predicate shared by real observations and private imagined members."""
    near = np.hypot(x-target['x'], y-target['y']) <= RADIUS
    return near & supported & ((np.asarray(masks,np.uint32) & target['keys']) == target['keys'])


def arrivals(x, y, supported, masks, target, step):
    """First directly predicted supported arrival, with each member's own required keys."""
    near = at_exit(x,y,supported,masks,target)
    near[..., 0] = False
    return np.where(near.any(-1), near.argmax(-1)*step, -1)


def complete_cost(arrival, contact, soft, scale, extra=0):
    """Every member reaches the exit and survives until the slowest arrival. No daring exception."""
    end = arrival.max(0)
    complete = (arrival > 0).all(0) & ~(((contact > 0) & (contact <= end)).any(0))
    return np.where(complete, end + scale*soft + extra, np.inf), end, complete


def shifted(segments, actions, age, length, padded):
    """Consume elapsed actions, including a selected wait, instead of restarting them."""
    age = int(age)
    if age < 0 or age >= len(actions):
        return None
    segs = [(nm,max(0,a-age),min(length,b-age)) for nm,a,b in segments if b>age and a<age+length]
    return segs, padded(np.asarray(actions,dtype=np.int64)[age:],length)
