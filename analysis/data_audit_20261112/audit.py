"""Read-only, bounded audit of the new PPO collection; no training or source mutation."""
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
from wm.buffer import eval_levels, transitions
from wm.dataset import Dataset, digest, geometry_id, trace_id
from wm.features import wind

OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'runs/collect_2-10_ppo50_01/wm_data'
LIMIT_SECONDS = 120
LIMIT_ROWS = 8_000_000

def save(name, value):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

catalog = Dataset.catalog([str(SOURCE)], max_candidates=2000, seed=0)
save('inventory.json', catalog)
started = time.monotonic()
levels, groups, geometries, sources, examples = {}, {}, {}, [], []
total_rows, stop = 0, 'complete'
for ei, entry in enumerate(sorted(catalog['entries'], key=lambda e: e['id'])):
    if time.monotonic() - started > LIMIT_SECONDS:
        stop = 'soft_time_budget_between_shards'
        break
    if total_rows + entry['raw_rows'] > LIMIT_ROWS:
        stop = 'raw_row_budget_before_decode'
        break
    p = Path(entry['id'])
    content = p.read_bytes()
    with np.load(io.BytesIO(content), allow_pickle=False) as z:
        rows, meta = z['rows'], json.loads(str(z['meta']))
    level = meta['level']
    if level not in geometries:
        lp = SOURCE / 'levels' / (level + '.npz')
        data = lp.read_bytes()
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            words, lm = z['words'], json.loads(str(z['meta']))
        geometries[level] = (geometry_id(words, lm, level), lm, digest(data))
    geom, lm, _ = geometries[level]
    c = levels.setdefault(level, collections.Counter())
    c.update(shards=1, raw_rows=len(rows), partial_shards=int(bool(meta.get('partial'))))
    total_rows += len(rows)
    # Common owner's eligibility and sorting semantics.
    tr = transitions(rows)
    s, nxt = tr['s'], tr['s2']
    c.update(transitions=len(s), death_labels=int((nxt['status'] == 1).sum()),
             finish_labels=int((nxt['status'] == 2).sum()),
             key_owned_start_transitions=int((s['keys'] != 0).sum()),
             key_mask_change_transitions=int((s['keys'] != nxt['keys']).sum()))
    changed = np.flatnonzero(s['keys'] != nxt['keys'])
    for i in changed:
        if len(examples) < 30:
            examples.append(dict(shard=p.name, level=level, agent=int(s[i]['agent']),
                                 frame=int(s[i]['frame']), before=int(s[i]['keys']), after=int(nxt[i]['keys']),
                                 position=[float(nxt[i]['x']), float(nxt[i]['y'])], keys=lm.get('keys')))
    # Bind and read the exact world bytes. Carry fan context within this shard only.
    wp = p.with_name(p.stem + '_world.json.gz')
    wb = wp.read_bytes()
    world = json.loads(gzip.decompress(wb))
    contexts, mapping, frame_codes, fans, known = {}, {}, {}, [], False
    for w in sorted(world, key=lambda w: w['f']):
        if isinstance(w.get('fx'), dict) and 'fans' in w['fx']:
            fans = w['fx']['fans']
            known = isinstance(fans, list)
        if known:
            ident = json.dumps(fans, sort_keys=True)
            if ident not in mapping:
                mapping[ident] = len(mapping)
                contexts[mapping[ident]] = fans
            frame_codes[int(w['f'])] = mapping[ident]
        dt = w.get('dt')
        c['world_frames'] += 1
        c['invalid_or_unlocked_dt'] += int(not isinstance(dt, (int, float)) or
                                         not np.isfinite(dt) or abs(dt - 16.667) > .01)
    codes = np.array([frame_codes.get(int(f), -1) for f in s['frame']], np.int32)
    c['wind_context_known_transitions'] += int((codes >= 0).sum())
    for code, fs in contexts.items():
        if not fs:
            continue
        at = codes == code
        fx, fy = wind(s['x'][at], s['y'][at], fs)
        c['wind_nonzero_transitions'] += int((np.hypot(fx, fy) > 0).sum())
    ordered = rows[np.lexsort((rows['frame'], rows['agent']))]
    _, begin = np.unique(ordered['agent'], return_index=True)
    file_groups = []
    for part in np.split(ordered, begin[1:]):
        group = trace_id(part, geom)
        complete = not meta.get('partial', False)
        groups.setdefault((level, complete), collections.Counter())[group] += 1
        c['agent_episodes'] += 1
        c['complete_agent_episodes' if complete else 'partial_agent_episodes'] += 1
        status = int(part[-1]['status'])
        c[{0: 'last_alive', 1: 'last_dead', 2: 'last_finish'}[status]] += 1
        statuses = meta.get('status')
        agent = int(part[0]['agent'])
        if isinstance(statuses, list) and agent < len(statuses):
            c['metadata_terminal_without_terminal_row'] += int(statuses[agent] in (1, 2) and status == 0)
            c['metadata_terminal_mismatch'] += int(statuses[agent] in (1, 2) and status in (1, 2) and statuses[agent] != status)
        file_groups.append(group)
    sources.append(dict(shard=p.name, level=level, shard_sha256=digest(content), world_sha256=digest(wb),
                        geometry_sha256=geometries[level][2], traces=len(file_groups),
                        unique_traces=len(set(file_groups))))
    if (ei + 1) % 25 == 0:
        print(json.dumps(dict(shards=ei + 1, rows=total_rows, seconds=round(time.monotonic()-started, 2))), flush=True)

for level, c in levels.items():
    full = groups.get((level, True), {})
    c['complete_unique_full_trace_groups'] = len(full)
    c['complete_duplicate_episode_excess'] = sum(full.values()) - len(full)
    c['partial_unique_trace_groups'] = len(groups.get((level, False), {}))
totals = sum(levels.values(), collections.Counter())
result = dict(source=str(SOURCE), source_files=sources, inventory_shards=catalog['files_listed'],
              processed_shards=len(sources), stop_reason=stop,
              complete_listed_rows=stop == 'complete' and len(sources) == catalog['files_listed'],
              bounds=dict(seconds=LIMIT_SECONDS, rows=LIMIT_ROWS), seconds=time.monotonic()-started,
              held_out_overlap=sorted(set(levels) & eval_levels()), totals=dict(totals),
              levels={k: dict(v) for k,v in levels.items()}, key_change_examples=examples,
              limitations=['Full exact-trace repetition only; shared prefixes and near duplicates are not independent.',
                           'Wind is common features.wind exposure, not ground-truth force measurement.',
                           'No 2-10 lock passage claim without a recorded 2-10 wave.',
                           'This audit does not train, predict, or prove dataset sufficiency.'])
save('coverage.json', result)
print(json.dumps({k: result[k] for k in ('processed_shards','stop_reason','seconds','held_out_overlap','totals')}), flush=True)
