"""Same-exit crossing decisions, private keys, contacts, consumed waits and async adoption ownership."""
from types import SimpleNamespace

import numpy as np
import pytest

from plan import passage as P
from plan.commit import Committer
from plan.moves import padded
from plan.planner import Planner


def test_wait_must_pay_for_the_same_complete_static_crossing():
    cost,_,ok=P.complete_cost(np.array([[16,24,-1],[20,28,-1]]),np.full((2,3),-1),
                             np.array([.4,.4,0.]),200.)
    assert ok.tolist()==[True,True,False]
    assert cost[0]<cost[1] and np.isinf(cost[2])


def test_wait_can_win_when_the_same_crossing_really_becomes_less_risky():
    cost,_,_=P.complete_cost(np.array([[16,24],[16,24]]),np.full((2,2),-1),np.array([.8,.1]),200.)
    assert np.argmin(cost)==1


def test_a_dead_member_cannot_be_hidden_by_other_members_reaching_the_exit():
    cost,_,ok=P.complete_cost(np.array([[16],[20]]),np.array([[-1],[12]]),np.array([0.]),200.)
    assert not ok[0] and np.isinf(cost[0])


def test_target_requires_supported_arrival_with_each_members_own_keys():
    x=np.array([[[0.,20.,40.]],[[0.,20.,40.]]]);y=np.zeros_like(x)
    masks=np.array([[[0,1,1]],[[0,0,0]]],np.uint32)
    sup=np.ones_like(x,bool);target=dict(x=40.,y=0.,keys=1)
    assert P.arrivals(x,y,sup,masks,target,4).tolist()==[[8],[-1]]
    sup[:]=False
    assert (P.arrivals(x,y,sup,masks,target,4)<0).all()
    assert not P.at_exit(40.,0.,False,1,target)  # flying through the region is not a real supported arrival
    assert not P.at_exit(40.,0.,True,0,target)
    assert P.at_exit(40.,0.,True,1,target)


def test_adopted_wait_is_consumed_instead_of_being_prepended_again():
    segs=[('wait 8f',0,8),('run R 16f',8,24)]
    actions=np.r_[np.zeros(8,np.int64),np.full(16,6,np.int64)]
    remaining,acts=P.shifted(segs,actions,8,16,padded)
    assert remaining==[('run R 16f',0,16)] and (acts==6).all()
    assert P.shifted(segs,actions,24,16,padded) is None


@pytest.mark.parametrize('on',[True,False])
def test_crossing_feedback_caps_a_long_configured_interval(on):
    p=SimpleNamespace(frame=100,actions=[0]*48,first_move_frames=24,trial_frames=0,decision_frames=8)
    c=Committer(replan=32,on=on)
    c.decided(0,p,100)
    assert c.due(0)==108
    assert c.triggers(0,107,0.,0.)==[]
    assert 'interval' in c.triggers(0,108,0.,0.)


def test_old_or_speculative_plan_adoption_cannot_rewind_observed_stall_history():
    pl=Planner.__new__(Planner)
    from plan.carrier import Carrier
    pl.carrier=Carrier()
    pl.passages={};pl.prev={};pl.long={};pl.moves=None
    pl.route=SimpleNamespace(using={})
    pl.stalls={0:dict(best=210.,since=90)}
    p=SimpleNamespace(agent=0,frame=50,trial_frames=0,from_prev=True)
    p._effects=dict(route_using=None,trial=None,memory=None,long=None,
                    stall=dict(best=280.,since=20),passage=None)
    pl.adopt(p)
    assert pl.stalls[0]==dict(best=210.,since=90)


def test_real_crossing_pipeline_compares_wait_and_consumes_adopted_prefix(monkeypatch):
    from game.level import LockContext
    from mpc.candidates import act
    from plan import planner as M
    from plan import safety as S
    from plan.planner import state_row
    pl=Planner.__new__(Planner)
    pl.H=48;pl.f=4;pl.model=None;pl.stall_window=8;pl._dead_t=10000.
    pl.complete_max=0;pl.modes=False
    assert pl.future_frames==56  # explicit wait must retain future coverage with airborne completion disabled
    pl.veto_members=1;pl.risk_members='worst';pl.early_margin=True;pl.bounds=None;pl.moves=None
    pl.lv=SimpleNamespace(ox=0.,oy=0.)
    pl.geom=SimpleNamespace(lock_context=LockContext(np.zeros((10,30),np.uint32),dict(ox=0.,oy=0.,keys=[])))
    pl.memory=SimpleNamespace(penalty=lambda *a:0.)
    pl.floor_for=lambda *a:None;pl._corner=lambda *a:None
    pl.support=lambda x,y,*a,**kw:(np.ones_like(x,bool),None,None)
    def imagine(_model,_members,_geom,rows,actions,*a,**kw):
        dx=np.where(actions==act(1,run=True),2.,0.)
        x=rows['x'][:,None]+np.c_[np.zeros(len(rows)),np.cumsum(dx,axis=1)]
        x=np.repeat(x[None,:,::4],2,axis=0)
        return dict(x=x,y=np.zeros_like(x),vy=np.zeros_like(x),jumps=None)
    monkeypatch.setattr(M,'imagine',imagine)
    # A hazard after the exit must not invalidate reaching the exit itself.
    fb=S.FrameBoxes(np.array([[60.,0.,1.,10.,0.]]),np.array([S.KIND_HAZARD],np.int8),['later'])
    frames=[None]+[fb]*64;worlds=[None]*64
    actions=np.full(32,act(1,run=True),np.int64)
    target=dict(x=40.,y=0.,keys=0,t=260.)
    row=state_row(0.,0.,0.,0.,0,0)
    result,note=pl._passage(row,100,0,worlds,frames,16.667,{},target,
                           [([('run R 32f',0,32)],actions)],None,0.,300.,200.)
    assert note['complete']==2 and note['direct_cost']<note['wait_cost']
    assert result['wait']==0
    cached=dict(frame=100,target=target,segments=[('wait 8f',0,8),('run R 32f',8,40)],
                actions=np.r_[np.full(8,act(0)),actions].tolist())
    again,note=pl._passage(row,108,0,worlds,frames,16.667,{},None,[],cached,0.,300.,200.)
    assert note['candidates']==1 and again['cached']
    assert again['segments'][0][0]=='run R 32f' and again['wait']==0
