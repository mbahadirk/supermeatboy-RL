"""Offline diagnosis only. No runtime edits, policy updates or action selection."""
import json
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from realpath import Planner, tile_shape, SHAPE_HALF, SHAPE_SLOPE, FIELD_DECAY, FIELD_UP

root=Path(__file__).resolve().parent
v=json.loads((root/'passive_sample.json').read_text())
info=v['info']
words=np.array(info['words'],dtype=np.uint32).reshape(info['H'],info['W'])
pl=Planner(words,info['ox'],info['oy'],[-90.,-290.],[[32.007041931152344,-132.71995544433594]],[])
shape=tile_shape(words)

def inside_solid(x,y):
    c,r=pl.cell(x,y)
    if not (0<=c<pl.W and 0<=r<pl.H):return True
    kind=shape.kind[r,c]
    if not kind:return False
    u=(x-info['ox'])/20-c-.5
    w=(y-info['oy'])/20-r-.5
    dot=shape.ox[r,c]*u+shape.oy[r,c]*w
    if kind in (SHAPE_HALF,SHAPE_SLOPE):return dot<=0
    return True

def segment_clear(x,y,cx,cy):
    # Diagnostic sampling of existing tile shapes; not a production collision algorithm.
    n=max(2,math.ceil(math.hypot(cx-x,cy-y)/.5))
    return not any(inside_solid(x+(cx-x)*t/n,y+(cy-y)*t/n) for t in range(1,n+1))

def candidates(path,x,y):
    path.value(x,y)  # initializes the existing field
    f=path._field[0]
    c0,r0=pl.cell(x,y)
    free=lambda c,r:0<=c<pl.W and 0<=r<pl.H and not path._blocked[r,c]
    out=[]
    for c in (c0-1,c0,c0+1):
        for r in (r0-1,r0,r0+1):
            if not free(c,r):continue
            if c!=c0 and r!=r0 and not (free(c,r0) or free(c0,r)):continue
            fv=f[r,c]
            if not np.isfinite(fv):continue
            cx,cy=pl.center((c,r))
            way=math.hypot(x-cx,y-cy)+(FIELD_UP-1)*max(0.,cy-y)
            out.append(dict(center=[cx,cy],value=float(fv)-FIELD_DECAY*way,clear=segment_clear(x,y,cx,cy)))
    return sorted(out,key=lambda q:q['value'],reverse=True)

paths=pl.find(3)
out={'route_scope':'Reconstructed routes WITHOUT hazards/sweeps; not a dump of the running trainer routes.',
     'paths':[]}
for path in paths:
    q=dict(points=path.points.tolist(),mode=path.mode,probes=[])
    for x in [60,80,100,110,120,140,160]:
        cs=candidates(path,x,36.)
        visible=[c for c in cs if c['clear']]
        q['probes'].append(dict(x=x,y=36.,current_value=path.value(x,36.),
                               winning_candidate=cs[0] if cs else None,
                               shape_visible_value=max(0.,visible[0]['value']) if visible else None))
    out['paths'].append(q)
states=[s for row in v['rows'] for s in row['states']]
coords=np.array([s['s'][:2] for s in states])
out['sample']=dict(rows=len(v['rows']),characters=len({s['id'] for s in states}),
                   wall_seconds=v['rows'][-1]['wall_seconds'],xy_min=coords.min(axis=0).tolist(),xy_max=coords.max(axis=0).tolist(),
                   vy_abs_p95=float(np.percentile([abs(s['s'][3]) for s in states],95)))
waves=[json.loads(line) for line in (root.parents[1]/'runs/aplus_v2/waves.jsonl').read_text().splitlines()]
waves=[r for r in waves if r['level']=='1-14']
out['waves']=dict(count=len(waves),finished=sum(r['finished'] for r in waves),dead=sum(r['dead'] for r in waves),
                  truncated=sum(r['truncated'] for r in waves),agents=sum(r['agents'] for r in waves),
                  last_ten_dead=sum(r['dead'] for r in waves[-10:]),last_ten_truncated=sum(r['truncated'] for r in waves[-10:]))
(root/'diagnosis.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('sample',out['sample']);print('waves',out['waves'])
for n,path in enumerate(out['paths']):
    print('route',n)
    for q in path['probes']:
        print(q['x'],round(q['current_value'],2),q['winning_candidate'],q['shape_visible_value'])
