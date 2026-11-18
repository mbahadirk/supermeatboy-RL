import numpy as np

from plan import carrier_cross as C
from plan.carrier import Carrier
from plan import safety as S
from game.level import SHAPE_CIRCLE
from mpc.candidates import ACTIONS


def test_transport_obstruction_is_seen_before_the_station_prefix_expires():
    fb=S.FrameBoxes(np.array([[100.,16.,20.,20.,0.]]),np.array([S.KIND_HAZARD]),['h'],np.array([SHAPE_CIRCLE]))
    bars=[(float(x),0.,30.,10.,False) for x in range(0,121,4)]
    x=np.arange(0,121,4);y=np.full(len(x),16.)
    target=C.obstruction(bars,[None]+[fb]*(len(x)-1),x,y,4)
    assert target['id']=='h' and target['hit_in']>16
    assert target['direction']==1 and target['edge']>126.
    # Projectiles remain evasions, not static far-side crossing objectives.
    shot=S.FrameBoxes(fb.boxes,np.array([S.KIND_SHOT]),['s'],fb.shape)
    assert C.obstruction(bars,[None]+[shot]*(len(x)-1),x,y,4) is None


def test_return_on_the_near_side_does_not_count_as_crossing():
    target=dict(direction=1,edge=130.,keys=1)
    x=np.full((3,2,9),140.);x[:,0]=110.;g=np.ones_like(x,bool);m=np.ones_like(x,np.uint32)
    collision=np.full((3,2),-1)
    end=C.arrival(x,g,m,target,collision,4)
    assert end.tolist()==[-1,4]
    m[2,1]=0
    assert C.arrival(x,g,m,target,collision,4).tolist()==[-1,-1]


def test_every_member_must_be_settled_and_alive_at_the_common_far_side():
    x=np.full((3,1,9),140.);g=np.zeros_like(x,bool);g[0,:,2:]=True;g[1,:,3:]=True;g[2,:,5:]=True
    target=dict(direction=1,edge=130.,keys=0);m=np.zeros_like(x,np.uint32);hit=np.full((3,1),-1)
    assert C.arrival(x,g,m,target,hit,4)[0]==24
    hit[0,0]=20
    assert C.arrival(x,g,m,target,hit,4)[0]==-1


def test_crossing_intent_survives_an_early_near_side_platform_return():
    c=Carrier();bar=[(100.,0.,30.,10.,False)]
    c.observe(0,0,100.,16.,0.,0,bar);c.observe(0,1,100.,16.,0.,0,bar)
    target=dict(direction=1,edge=130.,keys=0)
    p=dict(frame=1,index=0,count=1,keys=0,depart=2,return_by=40,actions=[0]*96,
           segments=[['cross',0,96]],crossing=target)
    c.adopt(0,1,p)
    c.observe(0,2,101.,30.,250.,0,bar)
    c.observe(0,10,110.,16.,0.,0,bar);c.observe(0,11,110.,16.,0.,0,bar)
    assert c.returns[0]['crossing']==target and c.context(0,11,0,bar)['crossing']==target
    far_bar=[(140.,0.,30.,10.,False)]
    c.observe(0,20,140.,16.,0.,0,far_bar);c.observe(0,21,140.,16.,0.,0,far_bar)
    assert not c.returns and 'crossing' not in c.context(0,21,0,far_bar)
    c.adopt(0,1,p)
    assert not c.returns


def test_templates_cover_long_holds_forward_motion_countersteering_and_waits():
    options=C.templates(96,1)
    assert len(options)==144
    for _,a in options:
        assert a.shape==(96,) and a.min()>=0 and a.max()<len(ACTIONS)
    _,a=next(o for o in options if o[0][0][0]=='carrier cross R wait 8 hold 15 turn 24 brake 16')
    assert all(not any(ACTIONS[i][:2]) for i in a[:8])
    assert ACTIONS[a[8]][1] and ACTIONS[a[8]][2]
    assert ACTIONS[a[32]][0] and not ACTIONS[a[32]][2]
    left=C.templates(96,-1)[0][1]
    assert ACTIONS[left[0]][0]


def test_supported_ascent_is_only_an_intermediate_waypoint():
    y=np.array([[[-250.,-230.,-210.,-205.]], [[-250.,-240.,-225.,-210.]]])
    supported=np.ones_like(y,bool);mask=np.zeros_like(y,np.uint32);hit=np.full((2,1),-1)
    target=dict(direction=1,edge=130.,keys=0)
    assert C.ascent(y,supported,mask,target,hit,4,np.array([-230.]))[0]==8
    hit[0,0]=6
    assert C.ascent(y,supported,mask,target,hit,4,np.array([-230.]))[0]==-1


def test_near_side_wall_waypoint_keeps_the_far_side_objective():
    c=Carrier();bar=[(100.,0.,30.,10.,False)]
    c.observe(0,0,100.,16.,0.,0,bar)
    target=dict(direction=1,edge=150.,keys=0)
    p=dict(frame=0,index=0,count=1,keys=0,depart=0,return_by=40,actions=[0]*96,
           segments=[['climb',0,96]],crossing=target,waypoint_y=36.)
    c.adopt(0,0,p)
    c.observe(0,8,120.,40.,100.,0,bar,static_supported=True)
    assert not c.returns and c.context(0,8,0,bar)['crossing']==target
    c.observe(0,20,160.,40.,0.,0,bar,static_supported=True)
    c.observe(0,21,160.,40.,0.,0,bar,static_supported=True)
    assert not c.observed
