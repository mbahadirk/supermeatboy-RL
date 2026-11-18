"""Recorded agent finish rewards, before wm.features' /5 scaling.

The planner still minimizes time/risk; this is a supervised world-model target,
not a policy-gradient update. Only a confirmed fixed-step finish earns an A+ bonus.
"""
from levels_smb import a_plus_time

REWARD_PROTOCOL = "finish-aplus-v1"
FINISH_REWARD = 5.0
A_PLUS_BONUS = 10.0
A_PLUS_SAMPLE_WEIGHT = 2.0


def finish_outcome(level, frames, dt_ms, locked):
    seconds = round(frames * dt_ms / 1000.0, 3)
    par = a_plus_time(level) if level else None
    grade = seconds <= par if locked is True and par is not None and frames > 0 and seconds > 0 else None
    return {"seconds": seconds, "a_plus_seconds": par, "a_plus": grade,
            "reward": FINISH_REWARD + (A_PLUS_BONUS if grade is True else 0.0)}


def outcome_rewards(level, frame, states, dt_ms, locked):
    """Post-action row rewards; frame zero has no preceding action.

    Nonterminal/death rewards remain undefined as in older planner captures.
    The collector selects consecutive lives; stale terminal rows cannot pay twice.
    """
    reward = finish_outcome(level, frame, dt_ms, locked)["reward"] if frame > 0 else float("nan")
    return [reward if len(s) > 6 and s[6] == 2 else float("nan") for s in states]
