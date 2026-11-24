"""Bounded realtest training from the preserved aplus_v2 checkpoint."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from realgame import Trainer,build_parser

args=build_parser().parse_args([
    '--agents','40','--shape-inputs','--realtest','--wave-seconds','30','--end-after','0',
    '--load','runs/aplus_v2/model.pt','--name','aplus_v2_1_14_visibility_fix',
    '--record-states','--no-monitor','--seed','0',
])
args.training_wave_limit=5
t=Trainer(args)
on_info=t.on_info
def info(m,data):
    on_info(m,data)
    if t.level.name!='1-14':
        print('[retest] expected 1-14, found '+t.level.name+'; stopping',flush=True)
        t.stopping=True
t.on_info=info
on_wave=t.on_wave_end
def wave(m):
    on_wave(m)
    if t.waves>=args.training_wave_limit:t.stopping=True
t.on_wave_end=wave
t.run()
