"""Observed moving-support tasks and time-aligned return proofs, independent of level names."""
from __future__ import annotations

import numpy as np

from realpath import BODY
from plan.carrier_cross import far_side


def on_bar(x, y, bar):
    bx,by,hw,hh,upright=bar
    return ((np.abs(np.asarray(x)-bx)<hw+BODY-.5) &
            (np.abs(np.asarray(y)-BODY-by-hh)<1.5) & ~np.bool_(upright))


class Carrier:
    def __init__(self):
        self.observed={}
        self.returns={}
        self.adopted={}

    def observe(self,agent,frame,x,y,vy,keys,bars,static_ground=False,dead=False,static_supported=False):
        agent=int(agent);old=self.observed.get(agent)
        if old is not None and frame<old['frame']:return
        if dead or not bars or (old is not None and (old['keys']!=keys or old['count']!=len(bars))):
            self.observed.pop(agent,None);self.returns.pop(agent,None);old=None
        if dead or not bars:return
        contact=next((i for i,b in enumerate(bars) if on_bar(x,y,b) and vy is not None and abs(vy)<1.),None)
        pending=self.returns.get(agent)
        crossing=(old or {}).get('crossing')
        if (pending is not None and pending.get('waypoint_y') is not None and
                (static_ground or static_supported) and y>=pending['waypoint_y']):
            self.returns.pop(agent,None);pending=None  # keep the far-side goal; plan its next ascent/exit
        if contact is None:
            if (static_ground or static_supported) and crossing is not None and far_side(x,crossing):
                settled=(old.get('static_settled',0)+1) if frame==old['frame']+1 else 1
                if settled>=2:
                    self.observed.pop(agent,None);self.returns.pop(agent,None);return
                old['static_settled']=settled
            if static_ground and pending is None and not (old is not None and old.get('returning')):
                self.observed.pop(agent,None)
            elif old is not None:
                old.update(frame=int(frame),x=float(x),y=float(y),attached=False,settled=0,
                           returning=old.get('returning',False) or not static_ground)
                if not (static_ground or static_supported):old['static_settled']=0
            return
        same=old is not None and old['index']==contact
        settled=old['settled']+1 if same and frame==old['frame']+1 else 1
        arrived=settled>=2 and (crossing is None or bool(far_side(x,crossing)))
        self.observed[agent]=dict(index=contact,count=len(bars),keys=int(keys),frame=int(frame),
                                  x=float(x),y=float(y),attached=True,settled=settled,
                                  returning=bool(same and old.get('returning',False) and not arrived))
        if crossing is not None and not arrived:self.observed[agent]['crossing']=crossing
        if pending is not None and (pending['index']!=contact or
                                   (frame>=pending['depart'] and arrived)):
            self.returns.pop(agent,None)

    def context(self,agent,frame,keys,bars):
        task=self.observed.get(int(agent))
        if task is None or task['keys']!=keys or task['count']!=len(bars) or task['frame']>frame:return None
        return dict(task)

    def adopt(self,agent,frame,proposal,release=False):
        """Only adopted controls create a return promise; never revive an already observed return."""
        agent=int(agent);old=self.returns.get(agent)
        if frame<self.adopted.get(agent,-1):return
        self.adopted[agent]=int(frame)
        if release and agent in self.observed:
            self.observed[agent]['returning']=False
            self.observed[agent].pop('crossing',None)
        if old is not None and old['frame']>frame:return
        if proposal is None:
            self.returns.pop(agent,None);return
        observed=self.observed.get(agent)
        if (observed is None or observed['index']!=proposal['index'] or observed['keys']!=proposal['keys']
                or observed['count']!=proposal['count']):return
        crossed=proposal.get('crossing') is None or bool(far_side(observed['x'],proposal['crossing']))
        if observed['frame']>=proposal['depart'] and observed['attached'] and observed['settled']>=2 and crossed:return
        observed['returning']=True
        if proposal.get('crossing') is not None:observed['crossing']=dict(proposal['crossing'])
        self.returns[agent]={**proposal,'actions':list(proposal['actions']),
                             'segments':[list(s) for s in proposal['segments']]}


def proof(contact, collision, step, limit=None):
    """All members return to THIS bar, with two settled model samples and no contact before that.

    contact: E,n,K+1. Collision frames are full, per-member first actual contact (negative: none).
    A pending return uses its remaining deadline rather than restarting a fresh full-horizon jump.
    """
    _,n,K1=contact.shape
    stop=np.full(n,K1-1,int) if limit is None else np.minimum(np.asarray(limit,int),K1-1)
    columns=np.arange(K1)[None,:]
    missing=~contact.all(0)
    first=np.where(missing & (columns<=stop[:,None]),columns,K1).min(1)
    continuous=~(missing & (columns<=stop[:,None])).any(1)
    pair=np.zeros_like(missing)
    pair[:,1:]=~missing[:,1:] & ~missing[:,:-1]
    pair &= (columns>=first[:,None]+2) & (columns<=stop[:,None])
    arrival=np.where(continuous,0,np.where(pair.any(1),pair.argmax(1),K1))
    end=np.where(continuous,stop,arrival)*step
    safe=~((collision>0)&(collision<=end[None,:])).any(0)
    eligible=(arrival<=stop)&safe
    return eligible,continuous,arrival*step


def choose(eligible,continuous,soft,position,estimate,cached=None):
    idx=np.flatnonzero(eligible)
    if not len(idx):return None
    if cached is not None and eligible[cached]:return int(cached)
    # Avoid unnecessary takeoff while allowing a safer evasive return when holding is threatened.
    near=idx[soft[idx]<=np.min(soft[idx])+.02]
    grounded=near[continuous[near]]
    pool=grounded if len(grounded) else near
    return int(min(pool,key=lambda i:(float(position[i]),float(estimate[i]))))


def station(contact,collision,jumping,veto,step,frames=16):
    """Two feedback intervals must remain attached; a later fixed tail will be replanned."""
    prefix=min(contact.shape[-1]-1,max(1,frames//step))
    ok,continuous,_=proof(contact,collision,step,np.full(contact.shape[1],prefix))
    return ok & continuous & ~np.asarray(jumping,bool) & ~np.asarray(veto,bool),prefix
