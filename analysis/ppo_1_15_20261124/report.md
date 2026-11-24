# 1-15 checkpoint and replay audit — 2026-11-24

This investigation is offline/read-only. No game attachment, live eval run, training
interruption or runtime edit was made. A byte snapshot of the active model was read
and saved as `current_snapshot.pt`; the real training model was not overwritten.
The user continued training during analysis, so results describe the recorded snapshot,
not the latest continually changing weights. Original input checkpoint hashes and per-layer
finite/norm/range statistics are in `results.json`.

## Native experience

Run: `runs/aplus_v2_1_14_continue`, realtest/40 agents/30s/shape inputs/fixed timestep.
At the wave-summary snapshot:1-14 had22 waves,143 finishes/880 trials;1-15 had30
completed waves,zero finishes/1200 trials,1146 deaths and54 timeouts. These are
online results from changing policies, not a fixed-policy success-rate estimate.
Unlike the earlier1-14 plateau, most1-15 trials die rather than standing alive
through the whole timeout.

Native states and requested actions were read at a fixed49,871,688-byte boundary
of the growing gzip telemetry. The read includes some partial next-wave samples;
wave-summary counts and frame-based metrics therefore have separate boundaries.

| Native wave | Jump requested, fraction of live action samples | Ground-to-upward-air takeoffs counted | Main occupied region |
| --- | --- | --- | --- |
| 1 | 71.52% | 238 | lower corridor |
| 20 | 86.81% | 341 | x120..160,y~-240 |
| 26 | 89.31% | 358 | x120..180,y~-240 |
| 30 | 69.54% | 1146 | x120..140,y~-80..-60 |

Takeoffs are counted from consecutive live native frames, grounded to airborne with
positive vy; wall jumps that remain airborne are not included. This is **not the same
as the earlier1-14 case where holding jump produced virtually no new floor jumps**.
In1-15 repeated real jumps occur below the slab/platform, without consistently completing
the lateral movement around its edge and climbing into the next corridor. Geometry
requires going around the slab; repeated upward movement beneath it does not suffice.
The later recorded cluster is higher, so “absolutely no position progress/learning” is
not supported, although there is still no finish in this snapshot. Exploration/stochastic
variation and learning are not isolated as causes of that change.

`regions.png` shows native solid polygons, green spawn, pink goal, blue wave26 common
positions, yellow wave30 common positions. Hazards/projectiles are omitted from this sketch.

## Weight health and behavior

Seven checkpoints compared: archived282-input successful policy, recovered524 policy,
aplus_v2's1-13 best before the long1-14 stay, aplus_v2 after the long stay, visibility-test
model, the continuation run's successful1-14 best, and a snapshot of current1-15 weights.

All stored parameter tensors are finite: no NaN/Inf. Largest absolute parameter across
actor/critic is2.395 in the oldest model and3.042 in the current snapshot. Actor first-layer
norm24.23 ->31.29; output bias absolute maximum remains~0.11. No numeric explosion or
all-zero/constant-output network was found. Some tanh saturation is present even in the
successful old network; on the same first-wave probes second-layer abs(tanh)>.99 fraction
is~54.7% old and62.9% current. That alone does not establish damaging saturation.

Relative actor parameter L2 change:33.9% before -> after the long1-14 period,18.0% from
there to the later successful1-14 best,9.0% from that best to the1-15 snapshot. Different
weights demonstrate updating, not useful learning or collapse by themselves.

Behavior was compared on **identical offline probe observations**, not just raw weights:
119 first-wave probes and198 wave26 probes,121 of the latter in the lower stuck region.
Complete native info/tiles and each selected wave's native states/actions were replayed
through the current observation producer, with updates disabled. The initial reward routes
were recomputed; original later-wave route definitions/logits and prior-wave hazard sweeps
were not logged, so these are controlled reconstructed input probes, **not exact original
online probabilities or success-rate predictions**. The current model uses the preserved
snapshot for all comparisons. Old282 policies receive only their existing input prefix.

| Checkpoint | Mean jump probability on same121 lower-region probes | Mean action entropy, nats |
| --- | --- | --- |
| Before long1-14 stay (1-13 best) | 98.97% | 1.036 |
| After long1-14 stay | 96.10% | 0.795 |
| Successful1-14 best after correction | 92.89% | 1.031 |
| Current1-15 snapshot | 97.55% | 0.805 |

The jump bias already exists before the long1-14 stay. It did not originate solely from
those~70 waves. The current policy still varies left/right: on the lower probes mean
left35.86%,right61.65%; action entropy is not near zero. Several jump-bearing actions
can remain diverse while the **jump/no-jump decision is highly concentrated**. This is
a local behavior/credit-assignment and exploration problem; global numerical weight
collapse is not supported. Returning to the old model alone does not remove the bias.

## Prior skill drift and causal limits

On335 archived successful1-9 states using a fixed historical reconstructed observation
set, argmax agreement with the recorded actions is82.1% for recovery,72.5% before long
1-14 training,62.4% after it,58.5% for the successful1-14 best,64.2% for the current1-15
snapshot. These are **action agreement, not level completion rates**. Geometry/field code
changed since these historical inputs were reconstructed, and teacher-forced states do
not measure rollout recovery. The drift supports checking partial forgetting/interference;
it does not prove catastrophic forgetting or that70 waves caused the1-15 failure. A live
old-skill retest would be needed to claim skill loss. No such run was launched here.

The existing PPO update uses only newly gathered trajectories and then clears its batch.
There is no periodic rehearsal of earlier levels. Prolonged unsuccessful training can
reinforce locally safe/rewarded behavior and change older skills. It is an important
mechanism to investigate, but a controlled alternative training history was not run.

## Smallest next experiment, no change made here

Do not infer that tensor corruption requires starting over. Diagnose the lower passage's
actual lateral progress, ceiling contacts and jump timing. Any shaping should target repeated
ceiling-hitting jumps with no new reachability/progress, preserve useful wall jumps and dynamic
obstacle waiting, and be validated on the same recorded conditions before broad penalties.
Check the route-lookahead's direction at the problematic edge against recorded actions;
this audit does not establish that reward guidance is perfect after the earlier fix.

For retention, compare an earlier successful checkpoint and the latest checkpoint on
identical old-level live conditions, keeping originals. Periodic **on-policy** practice of
previous levels can then test forgetting prevention. Do not naively feed old action traces
to PPO as if newly collected on-policy experience. No reset, auto jump re-press, reward
change or replay-learning mechanism was added in this investigation.

Reproduction: `audit.py`, `late_probe.py`, `archived_probe.py` (set SMBRL_IMPORTED to main
imported/smb_static). `audit.py` intentionally takes a fresh active-model snapshot on each
run; preserve existing current_snapshot/results to reproduce this particular moment.
All derived metrics and source checkpoint SHA256s are in results.json; offline input
arrays and frame/agent/position metadata are in the two probe NPZ files. Probe outputs
are diagnostics, not a request to use eval mode for the user's continuing realtest run.
