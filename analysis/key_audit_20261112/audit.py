"""Read-only audit of newly appended 2-10 waves; center/segment passages are observation proxies."""
import collections
import gzip
import io
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from wm.buffer import transitions
from wm.dataset import Dataset, digest, geometry_id, trace_id
from wm.features import wind

OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'runs/collect_2-10_ppo50_01/wm_data'

def save(name, value):
    with (OUT/name).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

def segment_hits(a, b, lo=(-20.,100.), hi=(0.,120.)):
    p = np.column_stack([a['x'], a['y']]).astype(float)
    q = np.column_stack([b['x'], b['y']]).astype(float)
    delta = q-p
    lower, upper = np.zeros(len(p)), np.ones(len(p))
    ok = np.ones(len(p),bool)
    for axis in range(2):
        moving = delta[:,axis] != 0
        ok &= moving | ((p[:,axis]>=lo[axis]) & (p[:,axis]<=hi[axis]))
        v = np.where(moving, delta[:,axis], 1.)
        t0, t1 = (lo[axis]-p[:,axis])/v, (hi[axis]-p[:,axis])/v
        lower = np.maximum(lower,np.where(moving,np.minimum(t0,t1),-np.inf))
        upper = np.minimum(upper,np.where(moving,np.maximum(t0,t1),np.inf))
    return ok & (lower <= upper)

catalog=Dataset.catalog([str(SOURCE)],2000,seed=0)
save('inventory.json',catalog)
entries=[e for e in catalog['entries'] if e['level']=='2-10']
assert 0<len(entries)<=150
assert sum(e['raw_rows'] for e in entries)<=3_000_000
raw_level=(SOURCE/'levels/2-10.npz').read_bytes()
with np.load(io.BytesIO(raw_level),allow_pickle=False) as z:
    words, lm=z['words'],json.loads(str(z['meta']))
assert len(lm['keys'])==1
key=np.asarray(lm['keys'][0],float)
geom=geometry_id(words,lm,'2-10')
counts=collections.Counter()
groups=collections.Counter()
examples=[]
files=[]
start=time.monotonic()
for k,e in enumerate(sorted(entries,key=lambda e:e['id'])):
    assert time.monotonic()-start<120
    p=Path(e['id']); content=p.read_bytes()
    with np.load(io.BytesIO(content),allow_pickle=False) as z:
        rows,meta=z['rows'],json.loads(str(z['meta']))
    tr=transitions(rows);s,b=tr['s'],tr['s2']
    counts.update(shards=1,rows=len(rows),transitions=len(s),partial_shards=int(bool(meta.get('partial'))),
                  key_owned_transitions=int((s['keys']!=0).sum()),
                  key_acquisitions=int(((s['keys']==0)&(b['keys']==1)).sum()),
                  finish_labels=int((b['status']==2).sum()),death_labels=int((b['status']==1).sum()))
    hit=segment_hits(s,b)
    counts['key_owned_lock_segment_hits']+=int((hit&(s['keys']==1)&(b['keys']==1)).sum())
    counts['unowned_lock_segment_hits']+=int((hit&(s['keys']==0)&(b['keys']==0)).sum())
    wb=p.with_name(p.stem+'_world.json.gz').read_bytes()
    world=json.loads(gzip.decompress(wb))
    fans=[];known=False;frames={};contexts={};ids={}
    for w in sorted(world,key=lambda w:w['f']):
        if isinstance(w.get('fx'),dict) and 'fans'in w['fx']:
            fans=w['fx']['fans'];known=isinstance(fans,list)
        if known:
            ident=json.dumps(fans,sort_keys=True)
            if ident not in ids:ids[ident]=len(ids);contexts[ids[ident]]=fans
            frames[int(w['f'])]=ids[ident]
    codes=np.array([frames.get(int(f),-1) for f in s['frame']])
    counts['wind_context_known_transitions']+=int((codes>=0).sum())
    for code,fs in contexts.items():
        at=codes==code
        fx,fy=wind(s['x'][at],s['y'][at],fs)
        counts['wind_nonzero_transitions']+=int((np.hypot(fx,fy)>0).sum())
    for agent in np.unique(rows['agent']):
        r=np.sort(rows[rows['agent']==agent],order='frame')
        group=trace_id(r,geom)
        complete=not meta.get('partial',False)
        if complete:groups[group]+=1
        counts['agent_episodes']+=1
        counts['complete_episodes']+=int(complete)
        counts[{0:'last_alive',1:'last_dead',2:'last_finish'}[int(r[-1]['status'])]]+=1
        a=s['agent']==agent
        acquired=np.flatnonzero(a&(s['keys']==0)&(b['keys']==1))
        passages=np.flatnonzero(a&hit&(s['keys']==1)&(b['keys']==1))
        if len(acquired):
            counts['agents_acquiring_key']+=1
            i=acquired[0]
            distance=float(np.hypot(b[i]['x']-key[0],b[i]['y']-key[1]))
            counts['acquisitions_within_30_of_key']+=int(distance<=30)
            later=passages[s[passages]['frame']>=b[i]['frame']]
            counts['agents_key_then_lock_region']+=int(bool(len(later)))
            if len(examples)<12:
                ex=dict(shard=p.name,agent=int(agent),key_frame=int(b[i]['frame']),key_distance=distance,
                        finished=int(r[-1]['status'])==2,lock_frame=int(s[later[0]]['frame'])if len(later)else None)
                if len(later):
                    j=later[0]
                    ex.update(lock_from=[float(s[j]['x']),float(s[j]['y'])],lock_to=[float(b[j]['x']),float(b[j]['y'])])
                examples.append(ex)
    files.append(dict(**e,sha256=digest(content),world_sha256=digest(wb)))
    if (k+1)%20==0:print(json.dumps(dict(shards=k+1,seconds=time.monotonic()-start)),flush=True)
counts['complete_unique_full_traces']=len(groups)
counts['complete_duplicate_excess']=sum(groups.values())-len(groups)
result=dict(counts=dict(counts),key=key.tolist(),level_sha256=digest(raw_level),files=files,examples=examples,
            seconds=time.monotonic()-start,
            lock_region=[-20,0,100,120],limitations=[
                'Lock passage uses recorded center segments; not a controlled collision experiment.',
                'Zero unowned passages does not alone prove keyless agents were stopped at the lock.',
                'Exact full trace uniqueness does not guarantee independent prefixes or worlds.'])
save('key_coverage.json',result)
print(json.dumps(dict(counts=result['counts'],seconds=result['seconds'],examples=examples[:2])),flush=True)
