# PPO 1-9 recovery — 2026-11-24

The old policy still passes 1-9 through ONLYPPO. The current `ppo_aplus_v1` checkpoint
does not pass in the measured trials even after timing and map fixes. A separate model,
initialized from the successful old policy and trained in the current 524-input structure,
now passes. This is a warm start, not recovery of the stuck checkpoint's weights.

## Measured native-game trials

All multi-agent trials: level 1-9, 40 agents, 30 simulated seconds, end-after=0,
stochastic actions, policy/numpy seed=0. Native game RNG was **not seeded**. Both
initial frozen evaluations have identical source hashes; checkpoint observation widths
differ as required by their original interfaces. See each run's `run_config.json`.

| Run under `runs/` | Learning | Waves | Successes | Fastest native seconds |
| --- | --- | --- | --- | --- |
| eval_old_fixed_1_9_20261124_b | No; archived 282 inputs | 5 | 12/200 | 5.650019 |
| eval_current_fixed_1_9_20261124_a | No; current 524 inputs | 5 | 0/200 | — |
| eval_current_tiles_fixed_1_9_20261124_a | No; current, corrected live tile decoding | 2 | 0/80 | — |
| ppo_recovery_1_9_20261124_a | Yes; old policy widened to 524 | 5 | 46/200 | 5.216680 |
| eval_video_single_1_9_20261124_a | No; recovered best, **one greedy agent** | 3 | 3/3 | 19.249878 |

Recovery wave successes: 2, 1, 13, 16, 14. Five PPO updates were performed. These
are online training outcomes under changing weights, **not a frozen final-policy 23%
success estimate**. The separate final video test used the best checkpoint (saved
before the fifth update), one agent and greedy actions. All three times were
19.249878 native seconds: repeatable in these trials, **not A+** (1-9 par is 8s).
The 5.216680s training finish is below par but was not written to the native map save:
these runs deliberately did not enable realtest. No video file was recorded.

Every observed native step in these trials was 16.6666675359ms. Frame/60 console
estimates differ slightly from native elapsed times; this report uses native times.
Single-agent evaluation took 58.37 wall seconds across three ~19.25s completions.
Forty-agent evaluation still has extra wall overhead; fixed physics does not establish
that every workload achieves real-time rendering.

`summarize.py` rebuilds `results.json` from original run artifacts. Input checkpoint
SHA256s remain unchanged; all frozen evaluation directories contain zero `.pt` files.
Native raw info/tiles/frames are in gzip telemetry; corrected-map/recovery runs also
record policy responses. Logs are `analysis/<run-name>.log`. The initial old trial
`eval_old_fixed_1_9_20261124_a` stopped safely before actions because midlevel 1-9
was identified as unknown; it contributes no trials above.

## Changes and what they establish

- Fixed native physics by default (`--no-lock-step` opts out), matching HZ60 used by
  rewards/actions. Patch VA0x450ef6 owns only expected original bytes 7c53; unexpected
  bytes abort. Both shutdown paths restore eb53 to 7c53. Enabled after initialization.
  Post-test read-only live check confirmed original `[124,83]` bytes restored.
- `--eval --load ...` disables sample collection, updates and checkpoint writes.
  Added bounds/level guard, seed, optional greedy actions, source/checkpoint hashes,
  native elapsed/dt statistics and optional `--record-states` telemetry. Default eval
  limit 20 waves; 0 means unlimited. Seed scope explicitly excludes native game RNG.
- Midlevel identity now includes actual solid breakable ID100 cells (49 on 1-9).
  Previously 0.979515 match narrowly missed 0.98. Threshold was not relaxed.
- Live block updates decode occupancy from current tile IDs/flags. Neighbor-bit changes
  such as c00164 -> 2c00164 keep blocks solid; actual removal to 100 opens them.
  Comparing whole words to spawn tiles incorrectly opened still-present blocks.
- Widening old checkpoints initializes added columns to **zero**, preserving initial
  actor/value outputs for arbitrary added features. The new features can learn normally.
  Existing same-width checkpoints are not modified by this loader change.
- No additional wait penalty, world model or action planner was introduced. Existing
  PPO reward-route computation remains. PPO still selects all actions.

The frozen current policy's failures establish a learned-behavior problem in this
sample; geometry correction alone does not repair its weights. Timing inconsistency
is a real environment issue, but no variable-vs-fixed controlled learning experiment
proves it originally caused the regression. Different pretraining/curriculum remain
relevant. Warm-starting a demonstrated skill is the practical recovery demonstrated here;
generalization to all levels and reliability of single-agent A+ are unmeasured.

## Reuse without changing originals

Run in `.worktrees/onlyppo`, with game open on 1-9. Clean single-agent visual playback
(tested for three bounded waves, unlimited below until Ctrl+C):

```powershell
$env:SMBRL_IMPORTED='C:\Users\pahstoner\PycharmProjects\REA-MeatBoy\smbrl\imported\smb_static'
python realgame.py --eval --eval-level 1-9 --eval-waves 0 --agents 1 --greedy --seed 0 --shape-inputs --wave-seconds 30 --end-after 1 --no-monitor --load runs/ppo_recovery_1_9_20261124_a/levels/1-9/best.pt
```

This plays; use a screen recorder for video. It does not learn or overwrite checkpoints.
For more PPO training, load the latest model into a **new** run:

```powershell
python realgame.py --agents 40 --shape-inputs --wave-seconds 30 --end-after 0 --realtest --load runs/ppo_recovery_1_9_20261124_a/model.pt --name ppo_recovery_continue
```

The latter enables existing native A+ save / five extra waves / next-level behavior;
that progression was not re-tested during these 1-9 comparisons. No auto-training is
left running. Archived best and original ppo_aplus_v1 models were not overwritten.

Validation: 40 pytest tests passed (one installed pygame/pkg_resources warning).
Coverage includes native patch ownership/cleanup, frozen evaluation, provenance/raw
telemetry, seeded policy sampling, tile identity/occupancy and widening invariance.
