"""PPO run provenance and optional raw native telemetry, without any model/planner dependency."""
import base64
import gzip
import json
from pathlib import Path
import numpy as np


class SessionLog:
    def __init__(self,run_dir,config,record_states=False):
        self.root=Path(run_dir)
        self.record_states=record_states
        self.stream=None
        self.results=[]
        (self.root/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')

    def event(self,payload,data=None,level=None,response=None):
        if not self.record_states or payload.get('ev') not in ('info','frame','wave_end','level'):return
        if self.stream is None:
            self.stream=gzip.open(self.root/'native_frames.jsonl.gz','at',encoding='utf-8',compresslevel=1)
        row=dict(payload=payload,resolved_level=level)
        if response is not None:row['policy_response']=response
        if data is not None:row['tiles_base64']=base64.b64encode(data).decode('ascii')
        self.stream.write(json.dumps(row,separators=(',',':'))+'\n')
        if payload.get('ev')=='wave_end':self.stream.flush()

    def wave(self,level,message,steps_ms,wall_seconds,eval_mode):
        st=message.get('status') or []
        times=[float(t) for t in (message.get('finish_seconds') or []) if isinstance(t,(int,float)) and t>0]
        row=dict(level=level,wave=message['wave'],frames=message['frame'],agents=len(st),
                 finished=st.count(2),dead=st.count(1),truncated=st.count(0),
                 status=st,finish_frames=message.get('finish'),native_finish_seconds=message.get('finish_seconds'),
                 fastest_native_seconds=min(times) if times else None,wall_seconds=wall_seconds)
        if steps_ms:
            row['dt_ms']=dict(min=float(min(steps_ms)),max=float(max(steps_ms)),mean=float(np.mean(steps_ms)))
        name='evaluation.jsonl' if eval_mode else 'waves.jsonl'
        with (self.root/name).open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        if eval_mode:
            self.results.append(row)
            total=sum(r['agents'] for r in self.results);won=sum(r['finished'] for r in self.results)
            summary=dict(waves=len(self.results),agents=total,finished=won,success_fraction=won/max(1,total),
                         levels=sorted(set(r['level'] for r in self.results)))
            (self.root/'evaluation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        return row

    def close(self):
        if self.stream is not None:self.stream.close();self.stream=None
