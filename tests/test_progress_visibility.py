"""Progress cannot be borrowed through the solid half of a supporting tile."""
import numpy as np
import pytest
from realpath import Planner, RewardPath, tile_shape, tile_classes


def geometry(word):
    words=np.zeros((7,7),np.uint32)
    words[3,3]=word
    pl=Planner(words,0.,0.,[10.,10.],[[130.,130.]],[])
    k=tile_classes(words)
    blocked=np.logical_or.reduce(list(k.values()))
    return pl,blocked,tile_shape(words)


@pytest.mark.parametrize('shape',[1,2,3])
@pytest.mark.parametrize('rotation',range(4))
@pytest.mark.parametrize('flip',range(4))
def test_half_and_slope_orientations_keep_only_visible_side(shape,rotation,flip):
    pl,blocked,sh=geometry(1|(shape<<18)|(rotation<<12)|(flip<<14))
    ox,oy=int(sh.ox[3,3]),int(sh.oy[3,3])
    x,y=70+6*ox,70+6*oy
    assert pl.progress_visible(x,y,3+ox,3+oy,blocked)
    assert not pl.progress_visible(x,y,3-ox,3-oy,blocked)


def test_half_floor_allows_travel_above_it_and_boundary_tangency():
    # Reference half is upper solid; flip y makes it a supporting lower half.
    pl,blocked,_=geometry(1|(1<<18)|(2<<14))
    assert pl.progress_visible(70.,76.,4,3,blocked)
    assert pl.progress_visible(70.,70.,4,3,blocked)
    assert not pl.progress_visible(70.,69.,4,3,blocked)


def test_reward_value_rejects_credit_below_floor_but_preserves_above():
    pl,blocked,_=geometry(1|(1<<18)|(2<<14))
    path=RewardPath(pl,[(0,0),(6,6)],'clear')
    path.length=1000.
    path._blocked=blocked
    path._field=np.full((1,7,7),-np.inf,np.float32)
    path._field[0,2,3]=900.  # below floor; must not be sampled from above
    path._field[0,4,3]=100.
    path.waypoints=[]
    assert path.value(70.,76.)==pytest.approx(72.)
    assert path.value(70.,54.)>800.  # legitimately below the same floor


def test_open_space_value_and_full_wall_visibility():
    pl,blocked,_=geometry(1)
    assert not pl.progress_visible(50.,70.,4,3,blocked)
    assert pl.progress_visible(50.,90.,4,4,blocked)
    words=np.zeros((7,7),np.uint32)
    pl=Planner(words,0.,0.,[10.,10.],[[130.,130.]],[])
    path=RewardPath(pl,[(0,0),(6,6)],'clear')
    path.length=1000.
    path._blocked=np.zeros_like(words,dtype=bool)
    path._field=np.full((1,7,7),100.,np.float32)
    path.waypoints=[]
    assert path.value(70.,70.)==100.


@pytest.mark.parametrize('tile_id',[99,100,96,97,98])
def test_dynamic_special_tiles_keep_existing_interpolation(tile_id):
    pl,blocked,_=geometry(tile_id|0x100)
    # A static reward route cannot decide whether this agent opened/broke this tile.
    # Both first and cached queries must leave the established handling unchanged.
    assert pl.progress_visible(70.,70.,4,3,blocked)
    assert pl.progress_visible(70.,70.,4,3,blocked)
