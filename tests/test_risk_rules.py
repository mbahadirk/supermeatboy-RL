"""The risk rules of 2026-11-13: the soft-risk margin stops growing at MARGIN_HORIZON, a death costs at least
DEATH_MIN_FRAMES (not a dead end's 500), and remembered deaths that repeat at one place at other wave frames are
on at every frame."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from plan import safety as S                                               # noqa: E402
from plan import planner as P                                              # noqa: E402


def test_margin_is_held_from_the_margin_horizon_on():
    m = S.margin(np.array([1, 8, S.MARGIN_HORIZON, 32, 48]))
    assert m[0] < m[1] < m[2] and m[2] == m[3] == m[4]
    assert m[2] == dict(S.ERR90)[16]


def test_a_death_costs_less_than_a_dead_end_by_default():
    assert P.DEATH_MIN_FRAMES < P.DEAD_END_FRAMES


def test_one_remembered_death_is_a_fact_about_its_own_wave_frame():
    dm = S.DeathMemory([("L", 100, 50.0, 60.0)])
    assert len(dm.boxes("L", 100)) == 1 and len(dm.boxes("L", 100 + S.WINDOW)) == 1
    assert len(dm.boxes("L", 100 + S.WINDOW + 1)) == 0 and len(dm.boxes("other", 100)) == 0


def test_deaths_repeating_at_one_place_at_other_frames_are_always_on():
    # the 2-1 pattern: the same point, 48 frames later each wave; and one unrelated death elsewhere
    dm = S.DeathMemory([("L", 329, -111.8, -193.3), ("L", 381, -111.8, -193.3), ("L", 429, -111.8, -193.3),
                        ("L", 128, -397.0, -194.0)])
    for frame in (0, 329, 700, 5000):
        b = dm.boxes("L", frame)
        here = b[np.hypot(b[:, 0] + 111.8, b[:, 1] + 193.3) < 1]
        assert len(here) == 1 and here[0, 2] == S.DEATH_HALF          # one box for the place, at every frame
    assert len(dm.boxes("L", 128)) == 2 and len(dm.boxes("L", 300)) == 1   # the single death keeps its window
    # two deaths at one place within the window of each other are still one moment, not a place
    dm2 = S.DeathMemory([("L", 100, 0.0, 0.0), ("L", 100 + S.WINDOW, 0.0, 0.0)])
    assert len(dm2.boxes("L", 1000)) == 0 and len(dm2.boxes("L", 100)) == 2
    # added later: the cache is rebuilt
    dm2.add("L", 400, 3.0, 0.0)
    assert len(dm2.boxes("L", 1000)) == 1
