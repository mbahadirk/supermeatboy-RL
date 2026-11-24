"""Read-only native projectile/timing and end-corridor audit."""
import collections
import json
import math
import sys
import zlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from game.hazards import hazard_shapes,hazard_reaches,SHAPE_CIRCLE
HERE.mkdir(exist_ok=True)
run=ROOT/'runs/aplus_v2_batch2048'
waves=[json.loads(s) for s in (run/'waves.jsonl').read_text().splitlines()]
waves=[r for r in waves if r['level']=='1-20'];complete={r['wave'] for r in waves}
stats={};births=[];info=None;prev_ids=set();prev_wave=None;previous={};ended=set();speeds=[]
with (run/'native_frames.jsonl.gz').open('rb') as f:
    limit=f.seek(0,2);f.seek(0);dec=zlib.decompressobj(31);tail=b''
    while f.tell()<limit:
        raw=tail+dec.decompress(f.read(min(1024*1024,limit-f.tell())))
        lines=raw.split(b'\n');tail=lines.pop()
        for line in lines:
            if not line:continue
            row=json.loads(line);m=row['payload']
            if row.get('resolved_level')!='1-20':continue
            if m['ev']=='info':info=row
            if m['ev']!='frame' or m['wave'] not in complete:continue
            wave,frame=m['wave'],m['frame']
            st=stats.setdefault(wave,dict(reached_end=set(),deaths=[],contact_shot=0,contact_static=0,
                end_deaths=0,end_contact_shot=0,end_contact_static=0,live_end_frames=0,positions=collections.Counter()))
            if wave!=prev_wave:prev_ids=set();prev_wave=wave
            sh=m.get('sh',[]);sx=m.get('shx',[])
            ids={s[0] for s in sh}
            for k,s in enumerate(sh):
                ext=sx[k] if k<len(sx) else None
                if ext:speeds.append([ext[0],math.hypot(ext[1],ext[2])])
                if frame>0 and s[0] not in prev_ids:
                    births.append(dict(wave=wave,frame=frame,id=s[0],owner=s[0]//10,x=s[1],y=s[2],r=s[3],ext=ext))
            prev_ids=ids
            static,_,kinds=hazard_shapes(m['hz'],None)
            for k,s in enumerate(m['s']):
                x,y=s[:2];key=(wave,k)
                if s[6]==0:
                    if x>150:
                        st['reached_end'].add(k);st['live_end_frames']+=1
                        st['positions'][(round(x/20)*20,round(y/20)*20)]+=1
                elif s[6]==1 and previous.get(key,0)==0:
                    shot=[p for p in sh if bool(hazard_reaches(x,y,(p[1],p[2],p[3],p[3],0.),SHAPE_CIRCLE))]
                    fixed=[b for b,kind in zip(static,kinds) if bool(hazard_reaches(x,y,b,kind))]
                    end=x>150
                    st['deaths'].append(dict(agent=k,frame=frame,x=x,y=y,end=end,
                        contact_shot_ids=[p[0] for p in shot],contact_static=bool(fixed)))
                    st['contact_shot']+=bool(shot);st['contact_static']+=bool(fixed)
                    if end:
                        st['end_deaths']+=1;st['end_contact_shot']+=bool(shot);st['end_contact_static']+=bool(fixed)
                previous[key]=s[6]

assert info
(HERE/'native_info.json').write_text(json.dumps(info),encoding='utf-8')
out=dict(waves=len(waves),agents=sum(r['agents'] for r in waves),finished=sum(r['finished'] for r in waves),
         dead=sum(r['dead'] for r in waves),telemetry_bytes=limit,end_region='x>150, diagnostic near-goal region; not a certified route milestone',
         observed_deaths=sum(len(s['deaths']) for s in stats.values()),
         end_reached=sum(len(s['reached_end']) for s in stats.values()),
         end_deaths=sum(s['end_deaths'] for s in stats.values()),
         end_contact_shot=sum(s['end_contact_shot'] for s in stats.values()),
         end_contact_static=sum(s['end_contact_static'] for s in stats.values()),
         births=births,launchers={},native_waves={})
ages=[];owners=collections.Counter()
for wave,st in stats.items():
    for d in st['deaths']:
        if not d['end']:continue
        for sid in d['contact_shot_ids']:
            owners[sid//10]+=1
            seen=[b['frame'] for b in births if b['wave']==wave and b['id']==sid and b['frame']<=d['frame']]
            if seen:ages.append(d['frame']-max(seen))
out['end_projectile_owners']=dict(owners)
out['end_shot_contact_age']=dict(n=len(ages),min_p50_max=np.percentile(ages,[0,50,100]).tolist() if ages else None,
                                within5frames=sum(a<=5 for a in ages))
for owner in sorted({b['owner'] for b in births}):
    q=[b for b in births if b['owner']==owner]
    intervals=[]
    for wave in complete:
        f=sorted({b['frame'] for b in q if b['wave']==wave})
        intervals+=np.diff(f).tolist()
    out['launchers'][str(owner)]=dict(births=len(q),median_birth_xy=np.median([[b['x'],b['y']] for b in q],axis=0).tolist(),
        kinds=sorted({b['ext'][0] for b in q if b['ext']}),
        interval_frames_p10_p50_p90=np.percentile(intervals,[10,50,90]).tolist() if intervals else None,
        first_birth_frames=[min(b['frame'] for b in q if b['wave']==wave) for wave in sorted({b['wave'] for b in q})][:12])
for kind in sorted({s[0] for s in speeds}):
    q=[s[1] for s in speeds if s[0]==kind]
    out.setdefault('projectile_speeds',{})[str(kind)]=dict(p10_p50_p90=np.percentile(q,[10,50,90]).tolist())
for wave,st in stats.items():
    st['reached_end']=sorted(st['reached_end'])
    st['end_top_positions']=[[x,y,n] for (x,y),n in st.pop('positions').most_common(5)]
    out['native_waves'][str(wave)]=st
(HERE/'results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print({k:v for k,v in out.items() if k not in ['births','native_waves']})
for k in list(out['native_waves'])[-5:]:
    st=out['native_waves'][k];print('wave',k,{j:v for j,v in st.items() if j!='deaths'})
