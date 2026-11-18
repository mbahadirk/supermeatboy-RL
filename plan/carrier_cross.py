"""Anticipatory crossing of an obstacle on a moving support's transport corridor."""
from __future__ import annotations

import numpy as np

from mpc.candidates import act
from realpath import BODY
from plan import safety as S
from game.level import SHAPE_CIRCLE


def far_side(x,target):
    return target['direction']*(np.asarray(x)-target['edge'])>=0.


def arrival(x,supported,masks,target,collision,step):
    """Two common far-side support samples, private keys, all members alive to common arrival."""
    good=far_side(x,target)&supported&((masks & target['keys'])==target['keys'])
    common=good.all(0)
    settled=common[:,1:]&common[:,:-1]
    end=np.where(settled.any(1),(settled.argmax(1)+1)*step,-1)
    ok=(end>0)&~((collision>0)&(collision<=end[None,:])).any(0)
    return np.where(ok,end,-1)


def ascent(y,supported,masks,target,collision,step,height):
    """A supported ascent waypoint, not a completed obstacle crossing. All members must gain height."""
    high=y>=np.asarray(height)[None,:,None]
    good=high&supported&((masks & target['keys'])==target['keys'])
    good[...,0]=False
    first=np.where(good.any(-1),good.argmax(-1)*step,-1)
    end=first.max(0)
    at=np.maximum(0,end//step)
    still_high=high[:,np.arange(y.shape[1]),at].all(0)
    ok=(first>0).all(0)&still_high&~((collision>0)&(collision<=end[None,:])).any(0)
    return np.where(ok,end,-1)


def obstruction(bars,frames,x,y,step):
    """First static lethal shape intersecting the forecast transport corridor; no projectile trigger."""
    if len(bars)<2 or abs(bars[-1][0]-bars[0][0])<1.:
        return None
    direction=1 if bars[-1][0]>bars[0][0] else -1
    first=next((fb for fb in frames if fb is not None),None);last=frames[-1]
    if first is None or last is None:return None
    stable=[]
    for i,hid in enumerate(first.ids):
        if first.kind[i]!=S.KIND_HAZARD or hid not in last.ids:continue
        j=last.ids.index(hid)
        if np.max(np.abs(first.boxes[i]-last.boxes[j]))<=1.:stable.append(hid)
    filtered=[]
    for fb in frames:
        if fb is None:filtered.append(None);continue
        idx=[i for i,hid in enumerate(fb.ids) if hid in stable and fb.kind[i]==S.KIND_HAZARD]
        take=fb.take(idx)
        filtered.append(S.FrameBoxes(take.boxes,take.kind,take.ids,take.shape,take.grow))
    sw=S.sweep(np.asarray(x)[None,None],np.asarray(y)[None,None],filtered,None,None,step)
    hit=int(sw.contact_frame[0,0]);bid=int(sw.contact_box[0,0])
    if hit<1 or bid<0:return None
    hid=sw.ids[bid];fb=filtered[min(len(filtered)-1,int(np.ceil(hit/step)))]
    if fb is None or hid not in fb.ids:return None
    box=np.asarray(fb.boxes[fb.ids.index(hid)],float)
    bx,by,hw,hh,angle=box
    extent=abs(np.cos(angle))*hw+abs(np.sin(angle))*hh
    # Circle boxes already carry the radius in both extents; rotation is immaterial.
    if int(fb.shapes()[fb.ids.index(hid)])==SHAPE_CIRCLE:extent=hw
    edge=bx+direction*(extent+BODY+S.margin(8))
    if direction*(edge-float(x[0]))<=0:return None
    return dict(id=hid,direction=direction,edge=float(edge),keys=0,hit_in=hit,
                box=box.tolist())


def templates(length,direction):
    """Bounded forward jump / countersteer grid; vertical hold and horizontal turn are independent."""
    result=[]
    for delay in (0,8,16,32):
        for hold in (10,15,22):
            for turn in (16,24,32):
                for brake in (8,16):
                    for run in (False,True):
                        a=np.full(length,act(0),np.int64)
                        for k in range(delay,min(length,delay+turn+brake)):
                            d=direction if k<delay+turn else -direction
                            a[k]=act(d,jump=k<delay+hold,run=run)
                        # A long vertical hold can extend beyond horizontal braking.
                        for k in range(delay+turn+brake,min(length,delay+hold)):
                            a[k]=act(0,jump=True)
                        side='R' if direction>0 else 'L'
                        name=f'carrier cross {side} wait {delay} hold {hold} turn {turn} brake {brake}'+(' run' if run else '')
                        result.append(([(name,0,min(length,delay+turn+brake)),
                                        ('pad',min(length,delay+turn+brake),length)],a))
    return result


def climb_templates(length,direction):
    """Use the approach wall to gain height before a forward exit; real wall repress stays in advance()."""
    result=[]
    for delay in (0,8):
        for climb in (24,40,56):
            for hold in (6,12):
                for forward in (12,24):
                    for brake in (8,16):
                        a=np.full(length,act(0),np.int64)
                        a[delay:min(length,delay+climb)]=act(direction,True)
                        begin=delay+climb+4  # release jump before trying to leave the wall/top rim
                        for k in range(begin,min(length,begin+forward+brake)):
                            d=direction if k<begin+forward else -direction
                            a[k]=act(d,jump=k<begin+hold)
                        end=min(length,begin+forward+brake)
                        name=f'carrier climb {direction} wait {delay} climb {climb} exit {hold} forward {forward} brake {brake}'
                        result.append(([(name,0,end),('pad',end,length)],a))
    return result
