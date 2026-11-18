"""Grounded launch credit is separate from geometric wall support and actual goal contact."""
import numpy as np

from plan.planner import Planner
from plan.geometry import Terrain
from plan.planner import carry_bars
from wm.buffer import ROW
from types import SimpleNamespace


def test_without_a_platform_launch_existing_terminal_times_are_unchanged():
    pl=Planner.__new__(Planner)
    t=np.array([20.,np.inf,300.])
    assert pl.landing_time(t,False) is t
    pl._launch=None
    assert pl.landing_time(t,False) is t


def test_a_wall_touch_or_flight_cannot_buy_progress_from_a_grounded_platform_launch():
    pl=Planner.__new__(Planner);pl._launch=(266.,416.)
    times=np.array([[180.,200.,430.],[380.,np.inf,30.]])
    grounded=np.array([[False,True,False],[False,False,True]])
    expected=np.array([[416.,200.,430.],[416.,np.inf,30.]])
    np.testing.assert_array_equal(pl.landing_time(times,grounded),expected)


def test_real_support_geometry_is_not_redefined_by_the_terminal_rule():
    pl=Planner.__new__(Planner);pl._launch=(266.,416.)
    words=np.zeros((10,10),np.uint32);words[:,5]=1
    terrain=Terrain(words,0.,0.)
    g,side=terrain.support(np.array([94.]),np.array([90.]))
    assert not g[0] and side[0]==1
    # It is still a wall a controller can jump from; it simply earns no unproved shortcut bonus.
    assert pl.landing_time(np.array([180.]),g)[0]==416.


def test_observed_departure_survives_hops_and_updates_only_at_a_landing():
    pl=Planner.__new__(Planner);pl.launches={};pl.lv=SimpleNamespace(plats=[(0,0,30,10,False)])
    pl.observe_launch(0,14,266,0,416,True)
    pl.observe_launch(0,23,287,0,419,False)
    assert pl.launches[0]==dict(frame=14,y=266.,t=416.,keys=0)
    pl.observe_launch(0,130,226,0,391,True)
    pl.observe_launch(0,14,266,0,416,True)  # late observations cannot rewind the transfer
    assert pl.launches[0]['frame']==130
    pl.observe_launch(0,140,240,1,380,False)  # objective/private-key transition
    assert not pl.launches
    pl.lv.plats=[]
    pl.observe_launch(0,150,266,1,350,True)
    assert not pl.launches


def rows(x=100.,y=16.,vx=0.,vy=0.):
    r=np.zeros(1,ROW);r['x']=x;r['y']=y;r['vx']=vx;r['vy']=vy
    return r


def test_horizontal_carry_replaces_learned_dx_and_keeps_velocity_relative():
    before=[(100.,0.,30.,10.,False)];after=[(108.,0.,30.,10.,False)]
    a=rows(vx=0);b=rows(x=108.,vx=0)  # even an already correct model must not receive carry twice
    assert carry_bars(a,b,before,after,1000/60,4)[0]
    assert b['x'][0]==108 and b['vx'][0]==0
    # Accelerate by 8 units/s every frame: relative displacement (8+16+24+32)/60.
    b=rows(x=100.,vx=32.)
    carry_bars(a,b,before,after,1000/60,4)
    np.testing.assert_allclose(b['x'],108+80/60,atol=1e-5)
    # Reverse-moving support is handled identically.
    b=rows();carry_bars(a,b,before,[(92.,0.,30.,10.,False)],1000/60,4)
    assert b['x'][0]==92


def test_carry_does_not_snap_takeoff_landing_nonrider_or_unsupported_departure():
    bar=[(100.,0.,30.,10.,False)];next_bar=[(108.,0.,30.,10.,False)]
    for a,b in [(rows(),rows(y=34.,vy=280)),(rows(y=30.,vy=-200),rows()),
                (rows(x=150),rows(x=150))]:
        old=b.copy()
        assert not carry_bars(a,b,bar,next_bar,1000/60,4).any()
        np.testing.assert_array_equal(b,old)
    a=rows();b=rows();old=b.copy()
    assert not carry_bars(a,b,bar,[(108.,2.,30.,10.,False)],1000/60,4).any()
    assert not carry_bars(a,b,[(100.,0.,10.,30.,True)],[(108.,0.,10.,30.,True)],1000/60,4).any()
    assert not carry_bars(a,b,bar,[],1000/60,4).any()
    np.testing.assert_array_equal(b,old)


def test_walking_off_edge_receives_only_the_supported_frames_of_carry():
    a=rows(x=133.,vx=180.);b=rows(x=145.,vx=180.,y=15.,vy=-10.)
    bar=[(100.,0.,30.,10.,False)];after=[(108.,0.,30.,10.,False)]
    carry_bars(a,b,bar,after,1000/60,4)
    # First frame is supported, then the feet leave: +12 own motion, +2 platform carry.
    np.testing.assert_allclose(b['x'],147.,atol=1e-5)
    assert b['y'][0]==15. and b['vy'][0]==-10.  # no snap back onto the bar


def test_descending_support_carry_preserves_native_contact_lag_without_double_counting():
    a=rows(y=17.3333);bar=[(100.,0.,30.,10.,False)]
    after=[(100.,-5.3333,30.,10.,False)]
    for learned_y in [17.3333,15.,12.]:
        b=rows(y=learned_y)
        assert carry_bars(a,b,bar,after,1000/60,4)[0]
        np.testing.assert_allclose(b['y'],12.,atol=1e-4)
        assert b['vy'][0]==0.
    b=rows(y=36.,vy=280.)  # actual takeoff from a descending support
    old=b.copy();assert not carry_bars(a,b,bar,after,1000/60,4).any()
    np.testing.assert_array_equal(b,old)
