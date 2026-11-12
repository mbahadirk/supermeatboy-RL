"""The A+ target: the game's par time per level (levels_smb.a_plus_time, from chapters.json)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import levels_smb as L                                                     # noqa: E402

needs_chapters = pytest.mark.skipif(not L.chapters(), reason="no imported chapters.json")


@needs_chapters
def test_par_times_come_from_the_chapter_files():
    assert L.a_plus_time("1-1") == 3.0 and L.a_plus_time("1-14") == 8.0 and L.a_plus_time("1-15") == 10.0
    assert L.a_plus("1-14", 5.05) is True and L.a_plus("1-15", 10.77) is False and L.a_plus("1-15", 10.0) is True


def test_unknown_level_or_time_is_unknown():
    assert L.a_plus_time(None) is None and L.a_plus_time("no-such-level") is None
    assert L.a_plus("no-such-level", 1.0) is None and L.a_plus("1-1", None) is None
