"""Rebuild the measured recovery summary from isolated run artifacts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = [
    'eval_old_fixed_1_9_20261124_b',
    'eval_current_fixed_1_9_20261124_a',
    'eval_current_tiles_fixed_1_9_20261124_a',
    'ppo_recovery_1_9_20261124_a',
    'eval_video_single_1_9_20261124_a',
]
out = {}
for name in NAMES:
    folder = ROOT / 'runs' / name
    config = json.loads((folder / 'run_config.json').read_text())
    metric = 'evaluation.jsonl' if config['evaluation'] else 'waves.jsonl'
    rows = [json.loads(line) for line in (folder / metric).read_text().splitlines()]
    times = [r['fastest_native_seconds'] for r in rows if r['fastest_native_seconds'] is not None]
    checkpoint = Path(config['checkpoint']['path'])
    out[name] = dict(
        evaluation=config['evaluation'], obs_size=config['obs_size'],
        waves=len(rows), agents=sum(r['agents'] for r in rows),
        finished=sum(r['finished'] for r in rows),
        finished_per_wave=[r['finished'] for r in rows],
        fastest_native_seconds=min(times) if times else None,
        dt_ms_min=min(r['dt_ms']['min'] for r in rows),
        dt_ms_max=max(r['dt_ms']['max'] for r in rows),
        wall_seconds=sum(r['wall_seconds'] for r in rows),
        input_checkpoint_unchanged=hashlib.sha256(checkpoint.read_bytes()).hexdigest() == config['checkpoint']['sha256'],
        written_checkpoints=len(list(folder.rglob('*.pt'))),
    )
old = json.loads((ROOT / 'runs' / NAMES[0] / 'run_config.json').read_text())
new = json.loads((ROOT / 'runs' / NAMES[1] / 'run_config.json').read_text())
out['baseline_pair_source_hashes_equal'] = old['sources'] == new['sources']
(Path(__file__).parent / 'results.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
