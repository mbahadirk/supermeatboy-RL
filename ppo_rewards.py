"""Small event-based PPO shaping; standing/waiting never triggers this cost."""
import math

REPEAT_HOP_COST = 0.03
REPEAT_HOP_FREE = 3
REPEAT_HOP_RADIUS = 20.0
REPEAT_HOP_CAP = 0.6  # per life/wave, even if progress later resets the local counter


class RepeatHops:
    def __init__(self):
        self.origin=None
        self.count=0
        self.paid=0.0

    def progress(self):
        self.origin=None
        self.count=0

    def takeoff(self,launch,stalled):
        if self.origin is None or math.hypot(launch[0]-self.origin[0],launch[1]-self.origin[1])>REPEAT_HOP_RADIUS:
            self.origin=tuple(launch)
            self.count=0
        self.count+=1
        if not stalled or self.count<=REPEAT_HOP_FREE:return 0.0
        cost=min(REPEAT_HOP_COST,max(0.0,REPEAT_HOP_CAP-self.paid))
        self.paid+=cost
        return cost


def physical_jump(action,was_grounded,state):
    # Distinguishes an actual jump from a held button, walking off a ledge, or simply waiting.
    return bool(action[2] and was_grounded and (state[4]&2) and state[3]>0 and state[6]==0)
