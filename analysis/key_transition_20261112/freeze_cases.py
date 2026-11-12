"""Freeze 24 validation-only recorded windows before evaluating the pickup implementation."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from wm.dataset import Dataset,canonical,digest,norm
from wm.buffer import transitions
BASE=Path(__file__).resolve().parent
ds=Dataset.load(ROOT/'runs/experiments/tile_shapes_epochs_v1/dataset.json')
source=norm(ROOT/'runs/collect_2-10_ppo50_01/wm_data')
candidates={'pickup_then_lock':[],'pickup_other':[],'near_key_no_pickup':[]}
rule_counts={'eligible_transitions':0,'observed_pickups':0,'native_endpoint_rule_mismatches':0}
t0=time.monotonic()
for meta,rows,world in ds.iter_shards(source,split='validation',max_shards=160,max_rows=1500000,max_seconds=90):
    entry=ds.entries[meta['_dataset_index']]
    if entry['level']!='2-10':continue
    rows=ds.rows_for(meta,rows,'validation')
    _,lm=ds.level(source,'2-10');key=np.asarray(lm['keys'][0])
    tr=transitions(rows);s,b=tr['s'],tr['s2']
    touched=np.hypot(b['x']-key[0],b['y']-key[1])<22
    predicted=s['keys']|touched.astype(np.uint32)
    rule_counts['eligible_transitions']+=len(s)
    rule_counts['observed_pickups']+=int(((s['keys']==0)&(b['keys']==1)).sum())
    rule_counts['native_endpoint_rule_mismatches']+=int((predicted!=b['keys']).sum())
    for agent in np.unique(rows['agent']):
        r=np.sort(rows[rows['agent']==agent],order='frame')
        if len(r)<65:continue
        pickups=np.flatnonzero((r['keys'][1:]==1)&(r['keys'][:-1]==0))+1
        owner=entry['agents'][str(int(agent))]
        base=dict(entry=meta['_dataset_index'],agent=int(agent),group=owner['group'],
                  shard=entry['shard'],shard_sha256=entry['shard_sha256'],world_sha256=entry['world_sha256'])
        if len(pickups):
            at=int(pickups[0]);start=max(0,at-16)
            if at-start<8 or start+48>=len(r):continue
            tail=r[at:start+49]
            lock=(tail['keys']==1)&(tail['x']>=-20)&(tail['x']<=0)&(tail['y']>=100)&(tail['y']<=120)
            category='pickup_then_lock'if lock.any()else'pickup_other'
            candidates[category].append(dict(**base,frame=int(r[start]['frame']),horizon=48,
                                              observed_pickup_frame=int(r[at]['frame']),
                                              observed_lock_frame=int(tail['frame'][np.flatnonzero(lock)[0]])if lock.any()else None,
                                              initial_keys=int(r[start]['keys']),category=category))
        elif not r['keys'].any():
            d=np.hypot(r['x']-key[0],r['y']-key[1]);at=int(np.argmin(d))
            start=max(0,at-16)
            if d[at]>60 or start+48>=len(r):continue
            candidates['near_key_no_pickup'].append(dict(**base,frame=int(r[start]['frame']),horizon=48,
                                                        observed_pickup_frame=None,observed_lock_frame=None,
                                                        initial_keys=int(r[start]['keys']),category='near_key_no_pickup'))
chosen=[]
for name,items in candidates.items():
    # Stable seeded ranking; one case per complete group and round-robin file coverage.
    by={}
    for c in sorted(items,key=lambda c:digest(canonical([0,c['group'],c['frame']]))):by.setdefault(c['shard'],[]).append(c)
    picked=[];used=set()
    for depth in range(max(map(len,by.values()),default=0)):
        for shard in sorted(by):
            if depth<len(by[shard])and len(picked)<8:
                c=by[shard][depth]
                if c['group']not in used:picked.append(c);used.add(c['group'])
    chosen+=picked
body=dict(protocol='recorded-key-window-v1',manifest_sha256=ds.hash,source=source,key=key.tolist(),
          selection='validation agents only; seeded full-group ordering, file round-robin; up to8 per category',
          cases=chosen,available={k:len(v)for k,v in candidates.items()},native_rule_check=rule_counts,
          model_sha256='d979f1dd1b42bd36488139496a3a0f4a7c269de595388d318a686ca2c41b2673',
          limits=['Recorded future hazards/platforms oracle; no game-completion evidence.',
                  'Lock region uses stored x/y anchor, not a controlled collision experiment.',
                  'Step4 interpolation and delayed opening are approximations; step1 endpoint rule can be checked exactly.'])
out=dict(body,sha256=digest(canonical(body)))
with(BASE/'cases.json').open('x',encoding='utf-8')as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps(dict(available=out['available'],selected={k:sum(c['category']==k for c in chosen)for k in candidates},
                      native=rule_counts,sha256=out['sha256'],seconds=time.monotonic()-t0)))
