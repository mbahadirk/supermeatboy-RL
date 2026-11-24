"""Summarize a completed bounded realtest without changing its artifacts."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
run=root.parents[1]/'runs/aplus_v2_1_14_visibility_fix'
stats={}
with gzip.open(run/'native_frames.jsonl.gz','rt',encoding='utf-8') as stream:
    for line in stream:
        row=json.loads(line);f=row['payload']
        if f['ev']!='frame':continue
        st=stats.setdefault(f['wave'],dict(ledge=set(),escaped=set(),samples=0,jump_held=0))
        acts=row.get('policy_response',{}).get('a',[])
        for k,s in enumerate(f['s']):
            x,y=s[:2]
            if s[6]!=0:continue
            if k in st['ledge'] and x<60 and y<20:st['escaped'].add(k)
            if x>60 and 25<y<50:
                st['ledge'].add(k)
                if k<len(acts):
                    st['samples']+=1
                    st['jump_held']+=acts[k][2]
waves=[json.loads(s) for s in (run/'waves.jsonl').read_text().splitlines()]
config=json.loads((run/'run_config.json').read_text())
checkpoint=Path(config['checkpoint']['path'])
out=dict(waves=len(waves),agents=sum(r['agents'] for r in waves),
         finished=sum(r['finished'] for r in waves),dead=sum(r['dead'] for r in waves),
         truncated=sum(r['truncated'] for r in waves),
         input_checkpoint_unchanged=hashlib.sha256(checkpoint.read_bytes()).hexdigest()==config['checkpoint']['sha256'],
         wall_seconds=sum(r['wall_seconds'] for r in waves),per_wave=[])
for wave,s in stats.items():
    if wave not in {r['wave'] for r in waves}:continue  # ignore the stop handshake's partial next wave
    out['per_wave'].append(dict(wave=wave,ledge=len(s['ledge']),escaped=len(s['escaped']),
                               ledge_action_samples=s['samples'],jump_action_fraction=s['jump_held']/max(1,s['samples'])))
(root/'retest_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
