"""Real-checkpoint regression of the unchanged opt-out frozen experiment contract."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from wm.dataset import Dataset,file_hash
from wm.experiment import evaluate,read_roster
BASE=Path(__file__).resolve().parent
EXPERIMENT=ROOT/'runs/experiments/tile_shapes_epochs_v1'
OUT=BASE/'frozen_rollout_compatibility.json'
if OUT.exists():raise FileExistsError(OUT)
torch.set_num_threads(2)
old_path=EXPERIMENT/'epoch_12/evaluation.json'
before_hash=file_hash(old_path);before=json.loads(old_path.read_text())
ds=Dataset.load(EXPERIMENT/'dataset.json')
with np.load(EXPERIMENT/'features.npz',allow_pickle=False)as z:d={k:z[k]for k in z.files}
roster=read_roster(EXPERIMENT/'evaluation_roster.json')
t=time.monotonic()
after=evaluate(ds,d,roster,EXPERIMENT/'epoch_12/model/wm_model.pt',
               expected_checkpoint_sha256='d979f1dd1b42bd36488139496a3a0f4a7c269de595388d318a686ca2c41b2673',
               batch_size=2048,max_seconds=180)
result=dict(before_report_sha256=before_hash,original_report_unchanged=before_hash==file_hash(old_path),
            manifest_sha256=ds.hash,roster_sha256=roster['sha256'],checkpoint_sha256=after['checkpoint']['sha256'],
            one_step_exactly_equal=after['one_step']==before['one_step'],
            open_loop_exactly_equal=after['open_loop']==before['open_loop'],seconds=time.monotonic()-t,
            note='Default key_pickup=False reproduced original one-step and key-change-censored v2 rollout metrics; no source-hash equality with historical code is implied.')
with OUT.open('x',encoding='utf-8')as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
assert result['original_report_unchanged']and result['one_step_exactly_equal']and result['open_loop_exactly_equal']
