"""Separate later lock windows; never label a 260-frame recorded replay as model imagination."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from wm.dataset import Dataset,canonical,digest,norm
BASE=Path(__file__).resolve().parent
ds=Dataset.load(ROOT/'runs/experiments/tile_shapes_epochs_v1/dataset.json')
source=norm(ROOT/'runs/collect_2-10_ppo50_01/wm_data')
items=[]
for meta,rows,world in ds.iter_shards(source,split='validation',max_shards=160,max_rows=1500000,max_seconds=90):
    e=ds.entries[meta['_dataset_index']]
    if e['level']!='2-10':continue
    rows=ds.rows_for(meta,rows,'validation')
    for agent in np.unique(rows['agent']):
        r=np.sort(rows[rows['agent']==agent],order='frame')
        pickup=np.flatnonzero((r['keys'][1:]==1)&(r['keys'][:-1]==0))+1
        if not len(pickup):continue
        pi=int(pickup[0]);pf=int(r['frame'][pi])
        hit=np.flatnonzero((r['keys']==1)&(r['x']>=-20)&(r['x']<=0)&(r['y']>=100)&(r['y']<=120)&(r['frame']>=pf))
        if not len(hit):continue
        gi=int(hit[0]);gf=int(r['frame'][gi]);start=max(0,gi-16)
        if r['keys'][start]!=1:continue
        items.append(dict(entry=meta['_dataset_index'],shard=e['shard'],agent=int(agent),
                          group=e['agents'][str(int(agent))]['group'],frame=int(r['frame'][start]),horizon=48,
                          observed_pickup_frame=pf,observed_lock_frame=gf,frames_pickup_to_lock=gf-pf,
                          carried_mask_start=max(int(r['frame'][0]),pf-16),initial_keys=1))
items.sort(key=lambda c:digest(canonical([0,c['group']])))
body=dict(protocol='recorded-later-gate-windows-v1',manifest_sha256=ds.hash,source=source,
          available=len(items),cases=items[:8],minimum_recorded_delay=min(c['frames_pickup_to_lock']for c in items),
          selection='Validation-only agents, recorded key then lock-region; seeded full-group order, first8',
          limits=['All available lock examples here may share a source wave; not independent trials.',
                  'Full pickup-to-lock observed carry is teacher-forced geometric rule replay, not 260+ frame world-model prediction.',
                  'Later48-frame model windows start with keys1; they test guard/feature consistency, not imagined pickup.'])
out=dict(body,sha256=digest(canonical(body)))
with(BASE/'gate_cases.json').open('x',encoding='utf-8')as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps(dict(available=len(items),selected=len(items[:8]),minimum_delay=body['minimum_recorded_delay'],sha256=out['sha256'])))
