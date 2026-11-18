import numpy as np

from plan.carrier import Carrier, on_bar, proof, choose
from plan.carrier import station


BAR=[(100.,0.,30.,10.,False)]


def observed():
    c=Carrier();c.observe(0,0,100.,16.,0.,0,BAR)
    c.observe(0,1,100.,16.,0.,0,BAR)
    return c


def promise(frame=1):
    return dict(frame=frame,index=0,count=1,keys=0,depart=frame+4,return_by=frame+20,
                actions=[3]*8+[0]*40,segments=[['jump up hold 3',0,8],['pad',8,48]])


def test_support_identity_and_pending_return_survive_an_airborne_replan_and_pocket():
    c=observed();p=promise();c.adopt(0,1,p)
    c.observe(0,5,101.,35.,280.,0,BAR)
    assert c.context(0,5,0,BAR)['index']==0
    assert c.returns[0]['return_by']==21
    c.adopt(0,9,p)
    c.adopt(0,5,None)  # an older plan cannot cancel the return revalidated at f9
    assert c.returns[0]['return_by']==21
    c.observe(0,6,104.,30.,0.,0,BAR,static_ground=True)
    assert c.context(0,6,0,BAR) is not None  # a pocket cannot erase a promised return
    c.adopt(0,10,None)  # unsafe controls can be discarded without discarding the support objective
    c.observe(0,11,104.,30.,0.,0,BAR,static_ground=True)
    assert c.context(0,11,0,BAR)['returning'] and not c.returns
    c.observe(0,21,110.,16.,0.,0,BAR)
    c.observe(0,22,110.,16.,0.,0,BAR)
    assert 0 not in c.returns
    c.adopt(0,1,p)  # stale async adoption cannot resurrect the completed jump
    assert 0 not in c.returns


def test_death_keys_count_and_level_reset_end_old_support_tasks():
    for change in ['dead','keys','count']:
        c=observed();c.adopt(0,1,promise())
        c.observe(0,5,150.,50.,100.,1 if change=='keys' else 0,
                  BAR+BAR if change=='count' else BAR,dead=change=='dead')
        assert not c.returns and not c.observed
    c=observed();c.observe(0,5,150.,30.,0.,0,BAR,static_ground=True)
    assert not c.observed  # actual walking transfer without a pending jump


def test_each_member_must_return_and_survive_until_the_shared_deadline():
    contact=np.ones((3,3,9),bool)
    contact[:,1,1:4]=False  # valid evasive hop and settled return
    contact[2,2,4:]=False  # one member misses the carrier
    collisions=np.full((3,3),-1)
    eligible,continuous,arrival=proof(contact,collisions,4)
    assert eligible.tolist()==[True,True,False]
    assert continuous.tolist()==[True,False,False]
    assert arrival[1]==20
    collisions[2,1]=18  # every member must survive until the shared return at +20f
    assert not proof(contact,collisions,4)[0][1]


def test_cached_return_uses_remaining_deadline_instead_of_restarting_wait():
    contact=np.zeros((2,1,13),bool);contact[:,:,2:5]=True
    # Two settled samples by +16f; unrelated unsupported padded tail cannot postpone that return.
    assert proof(contact,np.full((2,1),-1),4,np.array([4]))[0][0]
    assert not proof(contact,np.full((2,1),12),4,np.array([4]))[0][0]


def test_neutral_ride_wins_redundant_hop_but_a_safer_evasion_is_allowed():
    ok=np.array([True,True]);continuous=np.array([True,False]);pos=np.array([1.,0.]);cost=np.array([100.,80.])
    assert choose(ok,continuous,np.array([0.,0.]),pos,cost)==0
    assert choose(ok,continuous,np.array([.2,0.]),pos,cost)==1
    assert choose(ok,continuous,np.array([0.,0.]),pos,cost,cached=1)==1
    assert choose(np.zeros(2,bool),continuous,np.zeros(2),pos,cost) is None


def test_carrier_contact_is_not_a_wall_touch_or_the_original_platform_position():
    assert on_bar(100.,16.,BAR[0])
    assert not on_bar(100.,30.,BAR[0])
    assert not on_bar(100.,16.,(170.,0.,30.,10.,False))
    assert not on_bar(100.,16.,(100.,0.,10.,30.,True))


def test_no_jump_input_cannot_invent_takeoff_from_a_confirmed_support():
    from plan.planner import carry_bars
    from wm.buffer import ROW
    a=np.zeros(1,ROW);a['x']=100.;a['y']=16.
    b=a.copy();b['y']=30.;b['vy']=200.
    assert carry_bars(a,b,BAR,[(108.,0.,30.,10.,False)],1000/60,4,resting=np.array([True]))[0]
    assert abs(float(b['y'][0])-16.)<.02 and b['vy'][0]==0.
    b=a.copy();b['y']=30.;b['vy']=200.
    assert not carry_bars(a,b,BAR,[(108.,0.,30.,10.,False)],1000/60,4,resting=np.array([False]))[0]


def test_station_requires_safe_attached_feedback_prefix_and_never_overrides_a_veto():
    contacts=np.ones((3,5,13),bool);contacts[2,1,3]=False
    collision=np.full((3,5),-1);collision[0,0]=24;collision[0,2]=12
    jump=np.array([False,False,False,True,False]);veto=np.array([False]*4+[True])
    ok,k=station(contacts,collision,jump,veto,4)
    assert k==4 and ok.tolist()==[True,False,False,False,False]


def test_actual_goal_contact_wins_while_a_moving_carrier_task_is_active():
    import torch
    from game.level import Level
    from plan import safety as S
    from plan.planner import Planner,state_row
    from plan.progress import ProgressField
    from wm import features as F
    from wm.rollout import World
    from wm.model import WorldModel
    from wm.protocol import KEY_TILES_PROTOCOL
    words=np.zeros((15,30),np.uint32)
    info=dict(name='carrier-goal',W=30,H=15,ox=0.,oy=0.,spawn=[100.,116.],goals=[[140.,116.]],
              objs=[],lethalTypes=[],pf=[[100.,100.,0]])
    level=Level(info,words.tobytes())
    field=ProgressField(words,0.,0.,level.spawn,level.goals,[])
    model=WorldModel(n_in=F.layout(KEY_TILES_PROTOCOL).n_in(1),members=1,layers=0,step=1,res_scale=np.ones(4))
    model.info={'feature_protocol':KEY_TILES_PROTOCOL}
    with torch.no_grad():
        model.members[0][0].weight.zero_();model.members[0][0].bias.zero_()
        model.members[0][0].bias[8:8+len(F.BIN_NAMES)]=-100
    pl=Planner(model,level,field)
    pl.observe(0,0,100.,116.,vx=0.,vy=0.)
    worlds=[World(pf=[[100.+f,100.,0]]) for f in range(pl.future_frames+1)]
    fb=S.FrameBoxes(np.zeros((0,5)),np.zeros(0,np.int8),[])
    plan=pl.plan(state_row(100.,116.,0.,0.,0,5),0,worlds,[None]+[fb]*pl.future_frames)
    assert pl.carrier.context(0,0,0,level.plats) is not None
    assert plan.counts['carrier']['active']
    assert plan.goal_touch is not None and plan.mode!='carrier'


def test_unproved_return_allows_only_safe_short_recovery_from_above():
    from types import SimpleNamespace
    from plan.planner import Planner
    from wm.buffer import ROW
    from wm.rollout import World
    pl=Planner.__new__(Planner);pl.f=4;pl.H=48;pl.complete_max=0;pl.modes=False
    ter=SimpleNamespace(point_solid=lambda x,y:np.zeros(np.shape(x),bool),
                        support=lambda x,y:(np.zeros(np.shape(x),bool),np.zeros(np.shape(x),int)))
    pl.support_terrain=lambda keys:ter
    pl.time_to_go=lambda x,y,*a,**kw:(np.full(np.shape(x),100.),None,None,None,None)
    pl._tt=lambda t:np.asarray(t)
    x=np.broadcast_to(np.linspace(80.,95.,13),(2,2,13)).copy();y=np.full_like(x,30.)
    out=dict(x=x,y=y,vy=np.zeros_like(y),last=np.zeros((2,2),ROW))
    acts=np.zeros((2,48),int);final=[([('wait 8f',0,8),('pad',8,48)],a) for a in acts]
    collision=np.array([[4,-1],[-1,-1]])
    worlds=[World(pf=[[100.,0.,0]]) for _ in range(49)]
    task=dict(index=0,count=1,keys=0,attached=False)
    def call():return pl._carrier_choice(np.zeros(1,ROW),0,task,None,None,out,acts,final,collision,
        np.zeros(2,bool),np.full(2,-1),np.zeros(2),np.ones(2),worlds,0,np.zeros((2,2),int),100.,
        [None]*49,16.667,{})
    b,proposal,note=call()
    assert b==1 and proposal is None and note['recovery'] and not note['full_return_proved']
    assert note['proof_frames']==8
    out['y'][:]=10.  # already below the platform: do not invent a reachable landing
    assert call()[0] is None
    out['y'][:]=30.;worlds[8].pf=[]
    assert call()[0] is None  # an intermediate missing carrier cannot be indexed as the old support
