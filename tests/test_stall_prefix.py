"""Stall credit belongs to the executable prefix, not a repeatedly postponed tail."""
from types import SimpleNamespace

import numpy as np
import pytest

from game.level import LockContext
from plan.planner import Planner


def planner(window=8, keys=()):
    pl = Planner.__new__(Planner)
    pl.f, pl.stall_window, pl._dead_t = 4, window, 10000.
    pl.geom = SimpleNamespace(lock_context=LockContext(np.zeros((10, 30), np.uint32),
                                                      dict(ox=0., oy=0., keys=keys)))
    pl.novelty = lambda x, y: np.zeros(x.shape[0])
    pl.time_to_go = lambda x, y, masks, **kw: (300.-x, None, None, None, None)
    return pl


def paths():
    # Columns every four frames. Both paths reach x=80 at frame 24, but
    # the first keeps waiting through frame 16; the second starts moving now.
    x = np.array([[[0, 0, 0, 0, 0, 40, 80], [0, 5, 12, 24, 40, 60, 80]]], float)
    return dict(x=x, y=np.zeros_like(x), vy=np.zeros_like(x), jumps=None)


def test_far_progress_does_not_exempt_restarted_waits():
    pl = planner()
    for stalled in (20., 28., 36.):              # logged 4-17 frames 242/250/258
        costs, time, credit = pl.prefix_stall(paths(), [16, 24], 0, [], stalled, 300.)
        assert credit.tolist() == [8, 8]
        assert time.tolist() == [300., 288.]
        assert costs.tolist() == [stalled, 0.]


@pytest.mark.parametrize('window,first,credited', [(8, 4, 4), (6, 24, 4), (2, 24, 0), (16, 24, 16)])
def test_credit_never_passes_the_decision_move_or_model_boundary(window, first, credited):
    costs, _, frames = planner(window).prefix_stall(paths(), [first, first], 0, [], 20., 300.)
    assert frames.tolist() == [credited, credited]
    assert costs[0] == 20                        # the first path only promises later movement


def test_future_novel_tiles_do_not_waive_a_stationary_prefix():
    pl = planner()
    pl.novelty = lambda x, y: np.mean(x[:, 1:] > 10, axis=1)
    costs, _, _ = pl.prefix_stall(paths(), [16, 24], 0, [], 100., 300.)
    assert costs[0] == 100 and costs[1] == 0


def test_real_movement_can_still_receive_prefix_exploration_credit():
    pl = planner()
    pl.time_to_go = lambda x, y, masks, **kw: (np.full_like(x, 300.), None, None, None, None)
    pl.novelty = lambda x, y: np.mean(x[:, 1:] > 10, axis=1)
    costs, _, _ = pl.prefix_stall(paths(), [16, 24], 0, [], 100., 300.)
    assert costs.tolist() == [100., 50.]


def test_prefix_uses_private_keys_and_actual_platform_frame():
    pl = planner(keys=[(30., 0.)])               # only the moving path gets within native pickup radius
    seen = []
    def times(x, y, masks, **kw):
        seen.append((masks.copy(), kw['at']))
        return np.where(masks != 0, 200., 300.), None, None, None, None
    pl.time_to_go = times
    costs, _, frames = pl.prefix_stall(paths(), [16, 24], 0, [], 100., 300.)
    assert frames.tolist() == [8, 8] and costs.tolist() == [100., 0.]
    assert seen[0][0].tolist() == [[0, 1]] and seen[0][1] == 8


def test_death_in_prefix_cannot_earn_credit():
    pl = planner()
    costs, _, _ = pl.prefix_stall(paths(), [16, 24], 0, [], 100., 300., np.array([[-1, 4]]))
    assert costs.tolist() == [100., 100.]


def test_disabled_or_free_stall_has_no_lookup_cost():
    pl = planner()
    pl.time_to_go = lambda *a, **kw: pytest.fail('inactive stall must not look up a field')
    assert not pl.prefix_stall(paths(), [16, 24], 0, [], 0., 300.)[0].any()


def test_native_frame_real_routes_are_not_indexed_as_step_four_paths():
    pl = planner()
    x = np.arange(25, dtype=float)[None, None] * 2
    out = dict(x=x, y=np.zeros_like(x), vy=np.zeros_like(x))
    costs, time, frames = pl.prefix_stall(out, [24], 0, [], 100., 300., step=1)
    assert frames.tolist() == [8] and time.tolist() == [284.] and costs.tolist() == [0.]


def test_short_progress_receives_partial_credit_without_a_full_reset():
    pl = planner()
    x = np.array([[[0., 0., 0.], [0., 1., 3.], [0., -1., -3.]]])
    out = dict(x=x, y=np.zeros_like(x), vy=np.zeros_like(x))
    costs, _, _ = pl.prefix_stall(out, [8, 8, 8], 0, [], 100., 300.)
    assert costs.tolist() == [100., 70., 100.]
    costs2, _, _ = pl.prefix_stall(out, [8, 8, 8], 0, [], 200., 300.)
    assert costs2[0]-costs2[1] > costs[0]-costs[1]  # charge growth must eventually distinguish moving/waiting


def test_old_progress_cannot_discount_a_new_stationary_prefix():
    pl=planner()
    x=np.array([[[0.,0.,0.],[0.,1.,3.]]])
    out=dict(x=x,y=np.zeros_like(x),vy=np.zeros_like(x))
    # The historical best is ten frames above the current state's 300. That old gain cannot exempt waiting.
    costs,_,_=pl.prefix_stall(out,[8,8],0,[],100.,310.)
    assert costs.tolist()==[100.,70.]
