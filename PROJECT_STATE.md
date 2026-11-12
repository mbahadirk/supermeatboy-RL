# Super Meat Boy RL: the state of the project (2026-11-03; planner step 4 added 2026-11-08; 7.9 added 2026-11-10; 7.10 wave search 2026-11-11)

## Shots not yet fired, recorded shots, and the A+ target (2026-11-12; offline, uncommitted)

**Diagnosis on 1-15** (`runs/capture_2-14_route_check01`, the user's run with the `wm_step4` model):
- 90 waves gave 5 finishes and 85 deaths, every one from a shot.
- The killing shot was a median 23 frames old at the death; 92% were younger than 48 frames. The planner never saw them, because only flying shots were predicted.
- 1-15's launchers fire from fixed points with exact periods of 119 or 238 frames.

**`mpc/future.py`:**
- **Recorded shots:** where a trusted recording is used, its shots are taken as recorded. This includes shots fired later, with radius `REC_RADIUS`. A separate `shots_trusted` flag stops this once the shots differ from the recording by more than `MATCH_TOL`, as aimed shots would.
- **Launcher model, from this wave only:**
  - A shot seen from its first frame is a birth at its spawn point (`LAUNCH_KEY` 2 units).
  - Two births in one wave give the period; the period is kept for the level, the phase is re-seen each wave.
  - The last whole flight is the launcher's template.
  - Future births at last + m x P follow the template by age, with margin `SHOT_BASE` + one frame of motion; an unfinished template is extended with `shot_future`.
  - Nothing is predicted before a launcher's period has been seen, or once it has been silent for longer than P + 1.

**Measured** (`runs/experiments/launcher_future_v1/evaluate.py` and `evaluation.json`). Recorded 1-15 waves were replayed as first lives. The question: does a decision h frames before each of the 85 shot deaths see the killing shot at the death place?

| h (frames) | launchers off | launchers on |
|---|---|---|
| 16 | 57/85 | 69/85 |
| 24 | 41/85 | 58/85 |
| 32 | 25/85 | 48/85 |
| 48 | 7/85 | 27/85 |

- Predicted not-yet-fired shots matched a real shot 22,024 of 22,024 times.
- Replay mode, with the recording from earlier waves: 85/85 at 16 and 32 frames, 83/85 at 48.
- The remaining first-life misses are deaths before a launcher's second birth: about frame 490 for the 238-frame launchers. Exact prediction from frame 0 would need the launchers' timers read from the game, which is not done.
- Tests: two new ones in `tests/test_shot_future.py`.

**A+ target.**
- `levels_smb.a_plus_time(level)` is the game's `par` from the level's `.chapter` file (`chapters.json`); `a_plus(level, seconds)` is seconds <= par.
- Agent finish events record `a_plus_seconds` and `a_plus`, and the console prints the time against A+.
- `plan/benchmark.py` reports:
  - the A+ time;
  - "A+ ok" within K: the first wave with an A+ finish;
  - "A+ fin": logged A+ finishes;
  - overall levels with an A+ best.
- `plan/first_eval.py` records and reports `a_plus_seconds` and `a_plus` for a first-life success.
- On the user's run, 1-14's first 20 waves were all 9.93 s against an A+ of 8.0. Later route trials found 5.05 s, and 71 of the 98 finishes were A+.
- Planned bests in `best_times.json` are at or under par on 1-5, 1-8, 1-13, 1-14, 1-16, 1-17 and 1-19, and over it on 1-15 (10.77 vs 10.0). These are multi-wave bests, not first lives.
- **Unverified:**
  - whether A+ means at most par exactly;
  - whether the game's timer starts on our frame 0. An old note says 1-1 got A+ in 203 frames, 3.38 s against a par of 3.0, which conflicts with a plain "at most par" rule.
  - The planner's objective was not changed; it already minimises time.
- Full suite: 371 passed, plus 3 new A+ tests, excluding the pre-existing `test_cling_rule_fast_falls_are_not_support` failure.

## Route trials: a much better plan than the best route is tried once (2026-11-12; offline, uncommitted)

**Rule** (`plan/planner.py` `route_trial_choice`, `RouteTrials`). The trial is offered where the run's best route would be played, after the incumbent rule kept the route.
- **Candidate:** the final-stage candidate with the lowest estimate among those that have no veto, beat the route's estimate by more than `--route-trial-gain` frames (default 20; 0 = off), and whose key was never tried.
- **Key:** the FailureMemory state key of the decision plus the candidate's moves.
- **How it plays:** the candidate is played with mode `route-trial`. `Plan.trial_frames` (its non-pad moves) are played exactly: `Committer.triggers` stays silent, so there is no surprise, danger or tile replan. A reflex override cancels the trial before it starts.
- **Limits:** one trial per wave, by one agent. A key is never tried again in this level for the run, whatever the outcome. Like BestRoute, trials are not saved and `set_level` starts them fresh.

**Recording:**
- During the window, each real position is compared with the plan's predicted path. The first frame off by more than the committer's surprise tube, max(1.5 x tube(k), 3), is the deviation: frame, k, predicted, real, error and tube.
- Events in events.jsonl:
  - `route trial start`: moves, estimate, route estimate, gain, route frame and finish;
  - `route trial window`: held or not, the deviation, max error;
  - `route trial`: the outcome — `died` (with the death cause from plan.safety's explanation), `faster` (BestRoute.finished replaced the route with this wave), `slower`, or `no finish` (closed at the wave end).

**How often it would fire, measured on logs:** decisions that used the route and had a logged unvetoed alternative more than 20 frames better.
- bench_20261110-032340: 1-14 2,313; 1-13 222; 1-5 54; 1-8 48.
- bench_20261111-014526: 1-8 220; 1-5 78.

The field's T is optimistic, so many trials are expected to end "not held". The never-again memory bounds them.

**Checks:**
- Tests: `tests/test_route_trials.py` (choice, held / faster, deviation, death, committer silence) and a planner test on the flat level (the standing-still route is replaced once, the same wave falls back to the route, the next wave does not repeat the candidate).
- The two earlier best-route planner tests set gain 0. The key-pickup stub got `trials`.
- Full suite: 369 passed, excluding the pre-existing `test_cling_rule_fast_falls_are_not_support` failure.
- Not run live.

## Fan-wind physics prior and shot/moving-piece futures (2026-11-12; offline, uncommitted, not deployed)

**Fan wind in the world-model prior.**
- **Measured law** (2-10 recordings, read-only; 421,924 airborne in-wind transitions):
  - Per frame, airborne: `vy' = vy + (W - 550) dt`, with W taken at the frame's start position.
  - `W = st * c` along the fan axis, where `c = 1 - (Euclidean distance from the fan point) / range`, x5 when `c >= 0.5`.
  - The wind acts where `along >= 0`, `|lateral| <= hw` (no +6 band) and `c >= 0`.
  - |residual| median/p90/p99 is 0.010/0.017/0.022, and 99.85% are under 0.5. The old `wind` formula gets 27.5%.
  - 634 of the 635 outliers are ceiling hits.
  - Grounded and wall wind, and the x axis, are unmeasured; every recorded fan is vertical.
- **Code:**
  - New versioned prior `gravity-fan-wind-v1` (`wm/protocol.py`, `wm/features.py` `fan_force` / `prior` / `model_prior`), applied only to airborne rows. Checkpoints without a marker are `gravity-v1`, byte-identical to before.
  - The prior is selected from the checkpoint. Training opts in with `wm.train --prior-protocol`, and `--init` keeps the parent's prior.
  - Rollouts and planner `imagine` apply the checkpoint's prior with each step's fans.
  - `Level.wind_at` (PPO observation) and the `wind` input columns still use the old approximation; changing them needs a new feature protocol.
- **Experiment `runs/experiments/wind_prior_v1`:**
  - Same as `wind_control_v1` (old 321-column features, 3 epochs) except the prior. A gravity-v1 re-fit reproduced the old weights bit for bit.
  - On 2-10 validation, wind_control_v1 → wind_prior_v1:
    - fan one-step position median/p90 1.326/2.662 → 0.955/1.942;
    - fan velocity 33.0/61.9 → 21.9/49.9;
    - no-fan rows unchanged;
    - overall metrics unchanged.
  - Fan rides, |y| error at +48: three rides 116 → 33. Two deep fan-bottom rides 98 → 102: the prior alone is correct there, but the learned residual (only 1,320 of 388,982 samples had wind) cancels part of the lift.
  - Not yet combined with the larger 113-shard tile-shapes manifest or the 12-epoch schedule above.

**Hazard futures on a first life (`mpc/future.py`).**
- Before, recorded "shots" were extrapolated in a straight line with an extra radius of 2 + 0.5 x speed x t. Measured on recordings:
  - 4-4 and 4-6 shots are lava balls with constant d2y = -0.150 units/frame²;
  - bullets (1-15, 1-16, 2-3, 3-1) move straight;
  - linear error at +48 is 180 units on lava balls, while the extra radius was 119-192.
- **Now, shots:**
  - Each shot is fitted by least squares as a quadratic over its last 8 consecutive frames in this wave, with a margin from the fit residual and bounded fallbacks for 1-2 frame histories.
  - Reused ids are split into separate flights.
  - A shot is dropped once its centre is surely inside a solid tile, using the planner's live `pl.terrain`, passed by `agent.py` `observe(..., terrain=)`.
- **Now, moving kill boxes without a trusted recording:** a box that has repeated its own motion in this wave is looked up one period back with radius 1. Otherwise the old linear rule applies. World-model input boxes are unchanged.
- **Measured on recorded waves replayed as first lives** (`runs/experiments/shot_future_v1/evaluation.json`):

  | Class | Misses at +8/+16/+32/+48, before → after | Mean radius at +48, before → after |
  |---|---|---|
  | Lava ball | 9.6/34/77/95% → 0% | 86 → 20 |
  | Bullet | 1.9-4.6% → 0% | 139 → 39 |
  | Moving piece | 19/27/45/60% → 10/15/23/33% | – |

  - The remaining moving-piece misses are frames before one period has been seen. None of the alternatives tried reduced both misses and area.
  - Bullet phantoms at +48 fell from 2,283 to 101 thanks to the wall stop.
  - 4-6 lava balls vanish in lava, a hazard rather than a tile, so they are not stopped.
  - Cost on 4-6: 1.7 → 3.1 ms per frame.
- **Unverified:**
  - live behaviour;
  - a terminal fall-speed cap;
  - turning, bouncing or aimed shots;
  - planner decisions re-run with the new boxes.

**Tests.** New `tests/test_wind_prior.py` (11) and `tests/test_shot_future.py` (7). The full suite gave 362 passed and 1 failed: `test_planner.py::test_cling_rule_fast_falls_are_not_support`. Its stub lacks `Planner.support_terrain`, which the imagined-key-transition patch above added; the failure is not from these changes.

## Private imagined key transitions (2026-11-12; implemented and checked offline)

The normal planner now advances each ensemble member/candidate's private key mask, feeding the subsequent
tile inputs, collision guard and support/time lookup. The implementation is in `game/level.py`,
`wm/rollout.py`, `plan/planner.py` and `plan/commit.py`; no model weights, feature widths/protocols, loss,
progress-field edges or reward formula changed. `LockContext` owns the deterministic rule from
`realgame.js`: strict distance <22 at the post-update endpoint, all keys independently of route order.
Progress legs derive from owned masks. Shared tile words/cache/base terrain are not opened or mutated.

The incoming model step uses its OLD mask for guard collision; acquisition is computed afterward and can
open locks only for the NEXT model step. Step1 uses that endpoint ordering. At step4 the four intermediate
integer endpoints are linearly interpolated between corrected model endpoints; a continuous segment crossing
the circle does not suffice. This is an approximate intermediate path, not intermediate native physics.
Opening can lag up to3 frames relative to the rule's OWN interpolated pickup, not necessarily relative to
observed gameplay. The output records this policy explicitly. Completion and Committer support calculations
also use private masks. Raw `wm.rollout` methods retain default `key_pickup=False`; normal Planner enables
the rule explicitly. Planner's False option disables pickup dynamics, not the new native-circle geometric
contact scoring; it is not a promise to reproduce the former AABB contact heuristic.

BestRoute now records native key masks and their knownness. A key-map route with an unknown or mismatching
initial mask cannot bypass model/guard judgment using its recorded real path; the fallback reason is logged.
Known recorded endpoint masks select endpoint support/T. Legacy routes without masks may still be trusted
on maps without keys. This closes an unsafe context-equality exception; first-life evaluation has no route
memory and is unaffected. No broader held/flag/memory matching was added.

**Actual bounded checks.** Evidence is in `analysis/key_transition_20261112/rapor.md`, `cases.json`,
`gate_cases.json`, `evaluation_final.json`, `baseline_stall.json`, `after_stall.json` and
`frozen_rollout_compatibility.json`; the earlier `evaluation.json` remains preserved. Key-window and frozen
compatibility checks use the unchanged12-epoch checkpoint
`d979f1dd1b42bd36488139496a3a0f4a7c269de595388d318a686ca2c41b2673`; the stall audit also references preserved wm_step4.
Only manifest-validation agents were selected for the key checks; no fit, game connection or save/persistent
memory-file write occurred. The final 24-window model check took6.88 seconds.

- On recorded positions, the shared rule reproduced 38,032 true consecutive endpoints and45 native pickups
  with zero mask differences. This is teacher-forced rule correctness, not motion prediction.
- Eight positive pickup and eight near-key negative 48-frame windows were frozen before evaluation.
  Enabled raw and guarded rollouts each predicted8/8 positive pickups by ensemble majority; one of eight
  negatives was a false pickup, with all five members agreeing. Disabled masks necessarily stayed fixed
  (0/8 detected). Coarse first-owned-boundary timing error on detected positives had median0/p90+4 frames
  against observed timing; this includes model-motion/interpolation error and is not a calibrated rate.
- Whole-window motion error did not improve: guarded +48 position median/p90 remained14.66/107.56 game units
  with pickup disabled or enabled. Private state consistency is fixed; long-horizon physical error remains.
- The earliest recorded pickup-to-lock entry in the available validation episodes was260 frames later.
  None of the positive 48-frame windows proves the full acquisition-to-gate sequence. Eight later owned-key
  gate windows were checked separately: carrying masks along the OBSERVED path gave zero differences;
  on the same observed gate-entry segments, mask0 clipped8/8 and mask1 clipped0/8 in the guard. This is
  private-geometry/teacher-forced evidence, not a 260-frame model rollout or live keyless collision proof;
  examples can share a source wave. Recorded x/y remains a body-anchor proxy.

**Stall decision audit.** From the same recorded wave_00001/agent0/frame200 state in the earlier 2-10 planner
run, with fresh memories/no replay and recorded future worlds, the old `wm_step4` chooses left-jump then
wait: T317.38 ->318.38, estimate366.38 at48 frames. The12-epoch model chooses two right run-jumps:
T317.38 ->295.74, estimate355.74 at60 frames (48+12 completion), reaching about x239 in its chosen path versus
about x31 for the old choice. Each model's actions/scores are IDENTICAL before/after this key patch. Thus
the newer model already produces a useful rightward candidate at this one state; no progress-field/horizon
patch was made. This is not an escape/completion demonstration in the actual game. Historical initial-field
reconstruction differences documented in the field audit still apply; data/encoding also differ between models.

The old frozen 419-input v2 evaluation was rerun with raw `key_pickup=False`: one-step panels on272,085 samples
and open-loop panels on128 starts matched EXACTLY (39.56 seconds); the original report was unchanged.
Eighteen focused key tests cover private/timing/geometry/replay boundaries, alongside earlier feature,
experiment and commit regressions; overlapping test runs are not added as independent tests. Source records,
Root's final agent/first-eval integration also passed43 tests in48.12 seconds (only the existing pygame
pkg_resources deprecation warning), without attaching the game. These runs are not summed across overlaps.
Source records,
manifests, checkpoints and earlier evidence were preserved. No retraining, deployment or live gameplay was
performed in this phase. Earlier notes below describing unchanged imagined keys are pre-patch history.

## New 2-10 evidence and shape-aware epoch results (2026-11-12; offline, not deployed)

**New recorded evidence.** The user's 2-10 collection added 70 files / 1,449,419 raw rows to
`collect_2-10_ppo50_01`: 3,450 completed agent traces, all distinct exact full traces; 2,034 recorded key
0-to-1 changes near the key; 517 same-agent key-then-lock-region segments; 512 final finish rows and
473,777 computed nonzero-wind transitions. See `analysis/key_audit_20261112/rapor.md` and
`key_coverage.json`. These reduce the earlier fan/key data gap. The region test joins recorded x/y positions,
not the full collision body: 15 keyless segments also intersect the lock rectangle, so keyless blocking
was NOT established. No records were deleted/repaired from this proxy. The updated complete collection has
443 files / 8,631,891 rows; the experiment below did not consume that entire archive.

**Fixed experiment.** `runs/experiments/tile_shapes_epochs_v1` extends the previous 73-file snapshot with
40 explicit completed PPO files (two per new level, eight for 2-10), giving 113 shards / 1,402,115 raw rows.
Manifest SHA-256: `1ef6e5ec30453b11d4f295c3daf5dcda57c8884031115d2482c0060b88290dbe`.
All selected step-4 samples were featurized with explicit `tile-shapes-v1`: 1,396,847 x 419 inputs,
1,124,762 train / 272,085 validation. Validation includes 270,624 full four-frame targets and 1,461
terminal-short targets; 16,826 full targets start with a key owned. No training/evaluation split membership
or feature layout was inferred from file order. All three arms used fresh weights, seed0, five members,
3 x 256 hidden units, batch4096 and two CPU threads; no init, demos or self-correction. OneCycleLR duration
scales with each requested epoch count: this is an epoch/schedule-budget comparison, not identical LR paths.

The pre-training v2 roster SHA-256 is
`5a4560be433da6e7aa7e19a318dcee8cc8aad5c61ecc255163ec550b2c1bfd39`: 128 fixed starts across 21 levels.
Six starts are on 2-10 (four initial-wind / two zero-input-wind; three keys=0 / three keys=1). Resource and
encoding limits and the recorded-key-change censor policy are sealed into the roster. The 0.5 event
threshold and comparison rule were fixed before fitting. Source hashes remained unchanged through all
training and evaluation. Raw evidence is `training_plan.json`, `evaluation_roster.json`,
`training_results.json`, each `epoch_03/06/12/{training_result.json,evaluation.json}`, `comparison.json` and
the readable `sonuc_raporu.md`.

| Measured quantity | 3 epochs | 6 epochs | 12 epochs |
| --- | ---: | ---: | ---: |
| Full four-frame position median / p90 (n270,624) | 0.679 / 4.156 | 0.454 / 2.935 | 0.307 / 2.098 |
| Initial-wind position median / p90 (n11,854) | 0.628 / 1.782 | 0.509 / 1.528 | 0.381 / 1.311 |
| Key-owned position median / p90 (n16,826) | 0.851 / 4.148 | 0.580 / 2.831 | 0.417 / 2.202 |
| +48-frame position median / p90 (n101) | 25.659 / 87.358 | 21.983 / 73.446 | 15.310 / 61.773 |
| Fit wall seconds | 275.6 | 444.5 | 810.8 |
| Peak process working set, GiB | 3.22 | 3.21 | 3.18 |

The complete training sequence took 1,531.0 seconds (25.5 minutes), with no timeout. Six and twelve epochs
both pass the predefined comparison against three: overall/wind/zero-wind four-frame position p90 and
+16/+48-frame position p90 do not worsen, with at least one strict reduction. Twelve is best among these
arms on the reported main motion metrics. This does not imply that every metric improves monotonically:
the six-epoch +8 and +32 p90 values were worse than three. Wind strata classify the initial model-input
columns6/7, not the presence/absence of fan effects over the entire future interval.

At threshold0.5, full-target finish recall improves from99/128 (77.3%) to116/128 (90.6%), but precision is
only0.117 to0.256. Full-target death precision is still0.103 at twelve epochs (319/359 positives recalled).
The alive-endpoint death false-alarm rate falls2.05% ->1.53% ->1.03%. Full and terminal-short labels are
reported separately; overlapping four-frame chains are not independent terminal events. These event heads
are not calibrated safety probabilities. The normal `agent.py` planner still scores geometric contacts,
time-to-go and memory/risk terms, not the learned finish/death/reward heads.

**Remaining limits.** Open-loop errors use recorded actions and future hazard/platform observations with
the start map fixed. At +48,26 starts are terminal-censored and one lacks a row; no selected prefix actually
encountered a key change. Therefore the v2 key-change censor is implemented but this roster does not test
prediction through a pickup. Known-key starting patches are represented correctly, while imagined states
still retain their initial keys: contact checking can advance the progress-field leg without updating
future tile/guard geometry for newly imagined keys. More epochs did not implement that missing transition.

Features contain no explicit goal/key object positions or identity channels (only physical state, local
tile geometry, hazards/platforms/wind; current keys affect the tile patch). This is a representation limit
relevant to finish/pickup interpretation, not proof that observed correlations cannot predict finishes.
The 2-10 time-field plateau documented below also remains unresolved; a better motion model alone has
not demonstrated an escape from it. New data and block encoding changed together relative to the old
321-input experiment, so their individual effects were not isolated. This one-seed, same-level trace-split
test is not unseen-level/first-life or agent-completion evidence. No checkpoint was deployed or loaded into
the game, and no planner score/edge/imagined-key behavior was changed in this experiment.

## Handoff audits (2026-11-12; offline evidence, no production change)

Root's bounded inventory/coverage of the user's new PPO collection is saved under
`analysis/data_audit_20261112/coverage.json`: 373 shards, 7,182,472 raw rows, 7,164,622 eligible transitions,
16 levels (1-1 through 1-14 plus 3-1/3-2), no held-out levels. There are 17,850 agent episodes: 17,050 complete,
all distinct full exact traces; 800 partial episodes with 707 distinct traces across 17 partial shards.
Last observed outcomes are 6,952 finishes, 8,884 deaths and 2,014 alive tails (including partials). Recorded
wind context is known throughout but all computed exposure is zero; key masks/changes are also all zero.
The user confirmed that 2-10 has not yet been collected in this new run. No invalid dt or missing/mismatched
terminal metadata was found. The collection adds ordinary movement/finish diversity, not fan/key evidence;
50 simultaneously recorded agents are not automatically independent local transitions. Root measured this
audit at 66.6 seconds. Existing tile-shape tests were independently checked (17 passed).

The read-only field diagnosis is in `analysis/field_audit_20261112/{audit.py,graph_audit.json,point_checks.json,report.md}`.
Two single-thread field builds (0.743 seconds excluding startup/imports) show that the stall heuristic's
plateau exists with normal clear mode and jump limits (5,12), not an unlimited-jump fallback. The T graph
from pedestal (30,-150) ascends diagonally through (130,-50) to fan-support at (150,-30), then reaches the
central wall and key. Every diagonal ascent costs max(5,5.41)=5.41 frames; entering a supported fan cell
resets the jump budget. Thus horizontal travel fits into the same vertical-time lower bound: stall x30
and fan x190 at y=-154 both have T317.38, despite zero versus about +1800.526 computed upward wind force.
Removing only lift in a diagnostic counterfactual restores a horizontal fresh-start T gradient; that
counterfactual is not a proposed fix, since it removes real fan capability.

Correction to the earlier description: y=-154 across x30..230 is not one physical floor. Only the stall is
on its pedestal; other positions can be airborne. T_support/T_rise are flat there, but T_fall at x70 is
322.38 and x110 is330.53. Runtime support/velocity selects the appropriate lookup. The graph is optimistic
about jump state, fan support and parallel travel; this proves a heuristic plateau, not that a physical
route is impossible or that 48 frames alone is the cause. Candidate reach/pruning/completion/risk at the
actual stalled decision still needs a separately bounded analysis before choosing a fix.

The saved frame-0 reconstruction has 26 kill-shape boxes versus 25 in the historical initial field event,
and 244,991 versus 245,009 edges, although route cost87.4, spawn T412 and stall T317.38 match. The missing
pre-frame-info hazard snapshot is not reconstructed or invented. No production score/edge/model change,
new fit, deployment, game attachment or model rollout was performed in these handoff audits; existing
recordings, manifests and checkpoints were preserved.

## Planner decision audit, 2-10 stall, tile shapes (2026-11-12; read-only audit + uncommitted code)

**Decision audit (measured, read-only).** The audit script, kept outside the repo, reads `plans.jsonl`. It was run on the user's planner run `bench_20261111-014526`: 5,053 decisions on 1-5, 1-8, 1-11, 4-6, 4-4 and 3-1.
- Triggers: interval 4,028; move 629; surprise 614; support 168; danger 13.
- Decisions:
  - the previous plan was kept 2,013 times (40%);
  - best-route (previous wave's fastest route) 1,078 times. This route does not exist in `--eval-first`;
  - gain switches 966, with a median gain of +31 frames. Only 28% were under 10 frames, so most switches are not score noise;
  - risk switches 675; princess 287; veto 17.
- Real horizontal reversals were rare: 55 of 2,879 adopted plans pointed against the previous plan's path, mostly in the air.
- **More important:** 494 adopted plans end with a larger time-to-go than they started (`t_end > t_start + 4`). On 1-11, all candidates end at the same worse point, each marked "no progress".
- On 1-11, 4-4, 4-6 and 3-1, about half of the kept plans' estimates did not fall over 8 frames. The kept plan made no progress.
- Inference: the dominant failure is a progress-field plateau or trap that the 48-frame horizon cannot cross. It is not indecision.

**2-10 stall (measured; cause partly inferred).** `collect_2-10_fan_keys_01`:
- Waves 2-6 are identical. The agent falls from the spawn (-30,130) to (30,-154) and waits about 1,000 frames until the timeout.
- 130 of 158 decisions have an empty plan.
- Field: 2 legs (the key at (69.8,130), then the goal). T at the spawn is 412 frames.
- At the stall, T is 317 along the whole floor (x 30..230) and falls only with height, about 11 frames per 40 units, at every x.
- The fans are at x≈-140 and x≈181, blowing up; lift cells fill both columns. Planned paths never passed x=85.
- So no 48-frame candidate shows progress, and the key is never reached. This is why no key-owned data exists.
- **Not yet verified:** whether the field wrongly allows rising anywhere in that open region, or is right and needs more than 48 frames. The next offline step is to trace the field's route cells and edge types from the stall cell. Possible fixes after that: correct the field's edges, or measure progress toward the next route cell as a sub-goal.

**Tile shapes: slopes and half blocks with direction (code, uncommitted; 312 tests pass, 17 new in `tests/test_tile_shapes.py`).**

Before this change:
- The world model saw slopes only as one ordinal scalar, (open corner+1)/4, and half blocks without orientation.
- PPO's 11x11 grid and surface map treated slopes and half blocks as full blocks.
- The monitor drew every solid cell as a square.
- Level 1-3 has 19 slopes in all four orientations and no half blocks.

Changes:
- **Owner:** `realpath.tile_shape(words)` gives each cell a kind (empty/full/half/slope/special) and the open direction (ox, oy):
  - slope: the signs of its open corner;
  - half block: minus `half_normal`;
  - everything else: (0,0).

  `realpath.solid_polygon` gives the solid part of a cell. `slope_open_corner` and `half_normal` (moved from plan/geometry.py) share one flip/rotate decoder. `plan/geometry.Terrain` uses the same owner.
- **Monitor** (`monitor.py _level_bg`): draws slopes as the solid triangle and half blocks as the solid half. An offscreen render of 1-3 matches the game's diagonal corners.
- **World model:** new feature protocol `tile-shapes-v1` (`wm/protocol.py`).
  - The tile patch has 6 channels: full, half, slope, special, open-x, open-y. That is 389 inputs at step 1 and 419 at step 4; the old layout stays 291/321.
  - The protocol is selected from the checkpoint's recorded `feature_protocol`, never from data. Unknown or conflicting markers are errors, and `model_layout` checks the input width.
  - Old checkpoints, including the default `runs/wm_step4`, get byte-identical old features. Regression hashes:
    - collect_2-10_explore_01 step 4: `6befca3a…ba68344` (the earlier audit hash, reproduced);
    - demo_1-3 step 1: `75e09e7b…8465ccb`;
    - demo_1-3 step 4: `4c6f85d6…bb5e8c4a0`.
  - Fresh training uses the new protocol only on explicit opt-in: `wm.train --feature-protocol tile-shapes-v1`. `--init` keeps the parent's protocol.
  - `wm.experiment` is pinned to the old protocol for its frozen 321-column cache. The live model swap in agent.py refuses a model with a different protocol.
- **PPO:** `realgame.py --shape-inputs`, default OFF. Off means exactly 282 inputs, as before.
  - On, it appends two 11x11 maps (open-x, open-y; outside the map 0) from the cached `Level.open_dirs`, for 524 inputs.
  - An older `--load` checkpoint is widened (new columns ×0.1). `real_20261103-012157/model.pt` was checked offline: it loads identically with the flag off and is widened to 524 with it on.
  - Loading a wider checkpoint without the flag now raises an error; this is one-way. `mpc/offline.py` has the same flag. Reward and actions are unchanged.
- **Unverified:**
  - the live monitor;
  - live PPO with `--shape-inputs`;
  - any model trained on `tile-shapes-v1` (none exists);
  - half-block orientations on real data (synthetic only).

**PPO architecture (discussion, no change).** The user proposed deeper layers or attention against memorization. My position:
- First measure memorization: the same PPO checkpoint on a trained level versus an unseen level, on the first life.
- Level diversity and input design matter more than depth:
  - level-specific absolute inputs (goal direction and distance, route progress) are a memorization channel;
  - the flat MLP over flattened 11x11 maps has no spatial sharing;
  - cell types are close scalars, for example crumble 0.8 versus solid 1.0.
- Candidates, in this order: one-hot cell channels with a small CNN, then attention over variable sets of nearby objects (hazards, fans, platforms, keys), not a full transformer.
- Any change of input shape beyond widening means retraining PPO from scratch.

**State and next steps:**
- The user is collecting PPO data across training levels with `realgame.py --agents 50 --end-after 0 --wave-seconds 30 --paths 3 --load runs/real_20261103-012157 --wm-collect --name collect_2-10_ppo50_01`, without `--shape-inputs`. Held-out levels must not be entered.
- Proposed order:
  1. 2-10 field trace, offline.
  2. Epoch experiment E1: same manifest, seed and roster, fresh 3/6/12 epochs, with metrics fixed in advance. Decide whether to run it on the old or the new tile protocol, and keep every arm on one protocol.
  3. Verify 2-10 key pickup → lock pass in the PPO data:
     - the keys mask goes 0→1 near the key;
     - the same agent passes the lock cell, row 20 / col 19, about x -20..0, y 100..120;
     - agents without the key are stopped there.
  4. Data experiment E2 on a new manifest extension.
- Everything from the 2026-11-11 stage and this one is uncommitted in the working tree.

## First bounded wind-control experiment (2026-11-11; frozen inputs and evaluation contract)

The user authorized a first offline feasibility experiment. Its NEW immutable manifest is
`runs/experiments/wind_control_v1/dataset.json`, SHA-256
`d2ac03c2643d37682584cf6ccab45ddd3b6fca265245f3ac18a0e70b841e5162`.
The explicit selection has 73 shards, 391,506 raw rows, 16 sources and 1,234 distinct full exact-trace groups:
the prior 39-shard representative snapshot, 31 completed 2-10 exploration waves, and three representative
completed 1-8 terminal-capture traces. Partial waves and the repeated planner 2-10 probe are excluded; no
automatic archive discovery, repair, prefix deduplication or new split algorithm was introduced.

Root built a bounded cache (`features.npz`) with step=4, frac=1, max_samples=0 and opt-in provenance. It has
388,982 x 321 feature rows: 312,778 train (310,831 full, 1,947 terminal-short) and 76,204 validation (75,739 full,
465 terminal-short). Initial-input nonzero-wind rows number 1,084 train / 281 validation; full-chain counts
are 1,075 / 281. These differ from the earlier one-frame exposure counts by definition. The float16 X array
occupies about 249.7 MB; compressed cache is about 29 MB. Build/cache took 25.7 seconds. Provenance is a
compact N x 4 int64 array of manifest entry index, agent, start frame and current keys; ordinary feature
build output is unchanged unless `provenance=True` is explicitly requested with a manifest.

`wm.experiment.prepare_roster` freezes validation-only row indices and seeded agent-start identities before
training. The recorded roster `evaluation_roster.json` has SHA-256
`b6fa16faa6fb46acb5e368df2fd85c726979cda6bb3db0dec4d78a731e549b5f`:
all 76,204 validation feature rows and 128 rollout starts (16 eligible levels x 8). The 2-10 starts contain
four initial-wind and four zero-input-wind cases from six exact-trace groups; historical comparison has 965
validation rows from that source, all keys=0. A cache-content hash binds array values, dtype, shape and row
alignment; the roster also binds the manifest and feature protocol. Selection is fixed, and its hash must
not be silently changed after training begins.

The fresh architecture is the existing five-member, 3 x 256 step-4 model, seed 0, two CPU threads,
batch 4,096, no init, no demos, no self-correction. Root's discarded train-only warmup used two warmup plus
eight timed updates at that architecture/batch: mean 0.2048 seconds/update, approximately 15.57 seconds for
76 updates/epoch; reported peak process working set was 990,490,624 bytes (about 945 MiB). The first diagnostic
memory-API call failed and remains recorded separately; `warmup.json` records the corrected bounded probe.
No warmup checkpoint was retained or used to initialize the experiment. Three epochs were chosen from cost
before inspecting validation error, within a 15-minute training wall budget. These setup figures alone
establish no model improvement; completed measurements are recorded below.

**Measured fit and evaluation:** `training_result.json` reports successful completion (exit 0, no timeout)
in 77.736 seconds wall / 126.953 seconds CPU, peak process working set 1,156,579,328 bytes (about 1.08 GiB).
The recorded training-code source hashes remained unchanged. The fresh checkpoint is
`model/wm_model.pt`, SHA-256 `15543315b5125633ea8e23bc24fe58e46a00c7836cfd086d7c47c2a2508608ec`.
It records three epochs, 312,778 training samples, no demos/self-correction, the strict manifest and current
feature protocol. It was NOT published to or loaded by the game. The raw roster-file digest in the training
result is `380e9d1a9aeab4558bbddfe44429032e967b093f4422be25ef9bc596f6e166f7`; the canonical document seal above
is a separate digest of the roster body, not a changed case selection.

`historical_reference.json` and `candidate_evaluation.json` compare only the matched 2-10 validation subset
to the old model. On 281 full four-frame nonzero-initial-wind targets, position median/p90 improved from
2.894/5.798 to 1.326/2.662 game units; velocity median improved from 69.169 to 32.978. On 663 full zero-input-wind
targets, position median/p90 worsened from 0.836/5.887 to 1.716/6.777, and velocity median worsened from 21.481
to 47.010. Across all 944 matched full targets, position median worsened from 1.063 to 1.604 (p90 improved
5.878 to 5.458). This is a mixed result, not a replacement-model approval or an isolated causal proof about
wind training. There is no old-model global comparison across the other levels.

The eight matched fixed 2-10 starts gave +48-frame position median/p90 55.096/146.040 for the reference versus
53.779/66.979 for the candidate, with only six scored survivors and two terminal-censored cases in each.
The small, correlated survivor-only sample cannot establish safer planning, survival prediction or level
completion. The candidate's global full-target position median/p90 was 1.221/6.223 over 75,739 targets;
this is a standalone strict-split measurement, not an improvement claim against the old model.

At threshold 0.5 the candidate recalled zero of 20 validation finish labels (five full / fifteen short),
with zero predicted finishes. Full-target death recall was 124/150, but precision was only 124/3,936 (3.15%);
3,810/75,584 alive-endpoint targets produced a death false alarm (5.04%). Terminal-short death recall was
372/450 and precision 372/378, a different target population. These probabilities/thresholds are not declared
calibrated or safe for planner decisions. Every experimental key mask was zero; no real key-owned example
or live agent benefit was demonstrated. Existing old checkpoints/data remain preserved.

The new evaluator does not train. Its strict-model role retains `Dataset.require_model`; historical-reference
is a separate, explicit role restricted to `collect_2-10_explore_01/wm_data` validation cases with keys=0.
Old weights/metadata are never relabelled or attached to the new strict manifest. The strict report contains
a matched historical subset so both models use exactly the same cases, recorded prior and each model's own
residual scale. There is no historical-model comparison across the mixed-seen old levels. Source chronology
and the previous keys=0 byte-equality proof are evidence, not cryptographic assurance about all prior training.
Outputs use exclusive creation and bind loaded checkpoint bytes, roster and manifest hashes.

Primary one-step position/velocity median/p90 metrics use FULL four-frame targets, with counts overall,
initial-input wind/nonwind and 2-10. Terminal-short event panels remain separate; threshold 0.5, event-positive,
predicted-positive, precision/recall and alive-endpoint death false-alarm denominators are explicit. This is
not a calibration result. Raw open-loop +8/16/32/48 position metrics use the same recorded actions/worlds,
without a guard; terminal/missing-row/action/world evidence is censored and counted, never padded into fake
truth. Geometry applies recorded tile changes up to each start only, not future crumble changes or imagined
key pickups. These survivor-only errors do not measure correct predicted death timing or planner success.
Shared prefixes/local correlations/shared tiles and the absence of real key-owned data remain limitations;
the experiment does not establish unseen-level completion or fan-physics accuracy without its actual results.

## Known-key world-model tile inputs (2026-11-11; deterministic component checks)

The common feature builder now applies each row's already-known `keys` mask to a private copy of its local
tile patch. Different agents, candidates and ensemble trajectories can share LevelGeom without opening each
other's locks or mutating its words/padding/cache. Offline build/self-correction/evaluation and planner/MPC
inference consume the same feature function and domain-owned lock association. The existing Level lock-group
algorithm is extracted into a shared helper; its native go-payload ordering and physics are unchanged.
New static collector metadata records that applied key-index/group association with `private-locks-v1`
and `python-go-payload` provenance. Existing static files are not overwritten. Old single-key geometry can
reconstruct the association through the same domain owner; owned masks in old multi-key data without an
applied association are explicitly unsupported, while keys=0 inputs remain usable without guessing.

Input dimensions and model heads are unchanged (291 for step 1; 321 for step 4). Active feature semantics are
`per-agent-key-tiles-v1`; checkpoint **trained** semantics are a separate field. Existing unlabelled weights
remain `legacy-unspecified`, are still loadable for ordinary inference/explicit legacy compatibility, and
are never silently relabelled. The shared strict dataset/model boundary requires the current feature
protocol as well as the identical manifest. Fresh/continued weights acquire the new trained marker only
after a successful nonzero fit; notably `wm.train --init` without self-correction still performs no fit and
preserves the parent's trained marker. Reports and inference metadata expose active versus trained semantics.
No existing data, model, manifest, split/group identity or PPO reward/control behavior was changed.

Bounded independent checks compared old HEAD versus current keys=0 features on the 2-10 exploration recording:
5,474 step-4 samples (321 inputs) and X/yc/yb/shard bytes matched exactly; X SHA-256 is
`6befca3a27c224f62c99ec9df7fb01ad986b5f9cddd245e3b2179d5d0ba68344`.
In RAM only, the actual saved 2-10 lock at (-10,110) produced a closed mask=0 patch and open mask=1 patch,
with equal offline reconstruction/runtime association and unchanged shared geometry. Synthetic tests cover
mixed masks, grouped planner buffers, tile updates/reset, provenance, unknown mapping rejection and no-fit
lineage. These are deterministic input-correctness checks, not model training, model rollout, or live pickup.

Imagined key pickup, radius/timing rules, scoring and guard transitions are deliberately unchanged: predicted
states retain their current key mask. The probes below contain only keys=0, so this after-key correction
does not by itself explain failure to reach the key or establish a learned benefit. Wind accuracy and a
future imagined pickup transition require separate evidence. No additional blind-random collection is proposed.

## Bounded fan/key probes on 2-10 (2026-11-11; recording evidence)

The user selected 2-10, which is not a held-out level. The initial planner collection audit
`runs/collect_2-10_fan_keys_01/collection_audit_v1.json` contains six shards / 6,791 rows: five completed
20-second timeout waves and one partial cut. All recorded key masks remained zero; two fans were reported,
with nonzero computed wind in 185 of 6,785 eligible known-context transitions. Longer repeated planner waves
therefore did not demonstrate key collection or new model learning.

The subsequent single-agent model-free exploration audit
`runs/collect_2-10_explore_01/collection_audit_v1.json` contains 32 shards / 5,515 rows: 31 completed waves
with 31 distinct complete exact-trace groups and one partial wave. There were 29 deaths, two timeouts and
zero finishes; all 31 completed final targets passed capture checks. Nonzero computed wind occurred in
1,365 / 5,483 eligible known-context transitions (24.9%). Every key mask was still zero. The nearest sampled
position to the key was about 65.7759 units away, at approximately (3.99, 130.86); the direct approach from
the spawn is blocked by a wall in the inspected map. More blind random waves are not currently proposed.
These probes demonstrate diverse movement/wind exposure and recording quality, not successful key/lock
behavior, independent local samples, trained-model improvement or generalization.

The saved 2-10 static geometry has one key and one lock cell. Reconstructing the existing Level mapping from
saved words/origin/dimensions/keys gives `[[819]]` (row 20, column 19), matching the actual Level class and
key object. The probe audit identified that the then-current feature path ignored recorded key masks;
the deterministic known-key input correction above now addresses that component, while imagined pickup
remains unchanged. No actual key-owned frames exist in these probes to validate learned after-key behavior,
and these cases are not declared training-ready. No existing recording was repaired or used for new training
by this probe audit. The completed-only trace split has 24 train episodes / 4,475 transitions (1,084 nonzero
wind, 23 exposed episodes) and seven validation episodes / 965 transitions (281 nonzero wind, six exposed
episodes). Wind appears on both sides; this is exposure evidence, not an accuracy result.

## Native wave-end targets in new recordings (2026-11-11; tests and bounded live check)

The collector now preserves the exact final next-state supplied by the existing game script's `wave_end.s`.
When the message reports frame F, the action of update F-1 has already run. A target row at F is appended
only for a previously alive agent whose last row is F-1 with a recorded action, whose native state/status
and wave identity agree, and whose caller supplies its held-input mirror. All eight state fields come from
the native snapshot. Early-dead/finished agents are not extended to F; an already observed F row is not
duplicated or overwritten. Ambiguous duplicate agent/frame records and missing/nonconsecutive/invalid evidence
remain unextended, with explicit reasons in metadata. There is no inference or repair of missing positions.

The new row's action is -1: it is a next-state target, not a new control decision. Its held count is the
runner/trainer mirror AFTER the last applied input, not a claim to read an internal game timer. Its reward
belongs to that last input. Planner-family controllers retain NaN rewards; the PPO trainer forwards the SAME
final step-cost/death/finish reward that it already puts in its trajectory, after computing it. No reward
formula, PPO math, policy decision or game-script behavior changed. A final survivor (timeout or quota) can
provide an alive target: its last transition remains nonterminated and its episode remains truncated.

New shards identify `capture_protocol: wave-end-target-v1` and list appended/skipped agents, held/reward
provenance, and the end frame in `final_capture`. Exact partial hz/sh from the end message is stored in
`final_observation` with target-only/incomplete-context provenance. No frame-F input-world record is fabricated:
dt/platform/fan/tile-change context was not supplied there, and the existing F-1 input world remains unchanged.
Missing final reward stays NaN with an unknown/undefined source; missing held evidence prevents append. The
existing NaN-to-reward-target handling is outside this change. Partial S/stop/level-switch cuts and legacy
messages without final states preserve their original rows with explicit missing-evidence reasons.

The earlier bounded coverage results below describe OLD immutable recordings and are not retroactively
repaired. Existing data, manifests, checkpoints and trace-split-v1 group definitions are unchanged; a newly
captured true target naturally produces new content/group hashes. Tests exercise exact final fields, held /
reward alignment, mixed early/late terminal lives, alive tails, short chains, duplicate/end callbacks,
legacy/partial evidence and PPO forwarding without running training. The existing runner fake-message tests
also check new targets and absence of synthetic targets for legacy messages.

**Measured live collector check:** `runs/terminal_capture_1-8_check01/capture_audit_v2.json` audits 28 shards /
5,421 rows from the user's normal collection run on the already-seen level 1-8. All 27 completed waves
(26 finishes, one death) have the verified consecutive frame-F target, matching native status, action -1 and
post-action held mirror; planner target rewards remain NaN. Metadata finishes/deaths now match the observed
next-row labels: 26/26 and 1/1. The ordinary input-world timeline ends at F-1. Partial wave 28 stopped with
rows 0..128 and no appended state, with its cut reason recorded.

This is recording/protocol evidence, not generalization or new training success: the first life died at
frame 169, no learn-model event occurred, and the completed waves contain only three exact-trace groups
(death wave 1, first finish wave 2, and the same trace repeated in waves 3..27). First-life evaluation still
prohibits collection. Clean-detach acknowledgement, overlay appearance and live PPO reward forwarding were
not verified by this planner collection check; PPO alignment remains covered by offline caller tests.

## Bounded data coverage before another training run (2026-11-11)

`wm.coverage inventory` and `wm.coverage report` measure data evidence without fitting models, building large
feature arrays, or rolling models out. Inventory requires an explicit source list/request and candidate /
per-source caps. `Dataset.catalog` samples one filename index per evenly spaced bin with a deterministic
source/seed-specific RNG, then visits sources round-robin; only NPZ metadata and row-array headers are read.
Uninspected files' levels, states and events remain unknown. `Dataset.create` now also accepts explicitly
selected shard paths with a new-row budget and a soft time limit, preserving the original lexical default and
the existing trace group definition. Explicit selections must belong to declared sources and fit the caps;
invalid traces are rejected with the source path and agent index, never repaired or silently deduplicated.

Coverage reporting requires an immutable manifest, max-shards and max-rows. The dataset owner checks whole
shard row headers before decoding a shard that would exceed the row budget; no partial agent trace is counted
as a complete new episode. Soft time caps are checked between shards, not hard real-time interruption of an
individual compressed file. Output records processed/unprocessed IDs, decoded rows, elapsed time and stop
reason. Completeness of this measurement and completeness of the manifest's listed sources are separate.

Counters are grouped by source, level and split. Raw episode occurrences mean (shard, agent) recordings;
pool agent IDs are not independent experience. Exact-trace-deduplicated core counts and repeat factors are
reported separately; a global unique group may appear in multiple source buckets. Flag-ground/air/wall use
the model's existing AIR/WALL bit grouping (WALL first), not physical contact truth. Landing and blocked are
the existing label heuristics. One-frame transition and K-step chain denominators stay separate; terminal-short
chains are distinguished from full chains and from alive-at-record-end/time-limit tails that may yield no
K-step training sample. Platform presence/nearest-centre-within-200-units are proximity proxies, not evidence
of standing or landing on a platform. Wind is the nonzero `wm.features.wind` input proxy after a fan report is
known and carried forward within a shard; unknown context has its own denominator. Proxy/outcome dedup is not
claimed because world context and metadata outcomes are not included in the exact state/action trace hash.

**Measured bounded review (root audit, not a whole-archive claim):**

- Sources were the 14 folders listed in the actual checkpoint metadata from
  `runs/first_eval_1-8_check01/run.json`; request/budgets/seed 0 are in
  `runs/coverage_checks/source_request_v1.json`.
- `inventory_v1.json`: 203 of 1,903 listed NPZ files inspected, at most 16 per source, 18 sampled levels and
  32 source-level strata; zero state rows decoded, about 0.37 s. The root recorded an explicit deterministic
  source-level round-robin selection in `selection_v1.json`: 40 shards / 408,950 header rows.
- Builder validation rejected `runs/real_20261102-165351/wm_data/wave_00077.npz` (1-5): 50 duplicate
  agent/frame pairs at frame 401 had conflicting state/action/held/reward values. The bounded validity audit
  inspected only the selected 40 files / 408,950 rows. `validity_audit_v1.json` records the failure; the file
  was preserved. `selection_v2.json` explicitly excludes that complete file, with no repair or replacement.
- `representative_snapshot_v2.json` / `coverage_v2.json`: 39 shards, 385,471 raw rows, 14 sources and
  17 levels; 1,200 episode occurrences / 1,200 exact groups; 384,271 eligible one-frame transitions
  (309,148 train / 75,123 validation). Coverage took about 3.05 s. All 23,025 observed world frames had
  known, locked 16.667 ms steps. The snapshot remains an incomplete subset of the 1,903 listed files.
- In this subset: flag-wall 6,799 / 384,271 (1.77%); platform-centre proximity 8,049 / 384,271 (2.09%);
  computed nonzero wind 0 / 384,271 known contexts. These are input/label proxies, not gameplay capabilities
  or proof that the whole archive lacks wind data. Validation has no examples for 1-1, 1-8 or 2-1; 1-2 has
  zero training transitions and 1,199 validation transitions. This selection does not establish adequate
  per-level or per-event coverage and is NOT automatically training-ready.

**Data-quality limit measured before training:** the recorder can omit an agent's final next-state row when
its last update ends a wave. Outcome evidence can then exist only in shard metadata. Coverage therefore
reports raw `meta.status[agent]` outcomes, last-row status and terminal labels separately, without inventing
the missing state/transition. In the 39-file subset metadata reports 30 finishes / 782 deaths, while eligible
transitions contain 9 finish / 763 death labels: 40 terminal outcomes have no terminal row (21 finishes,
19 deaths). An earlier separate 21-file planner snapshot had 20 metadata terminal outcomes and zero terminal
last rows. This is a bounded recording-quality finding, not a proven explanation of all training failures;
older multi-agent PPO recordings can have different terminal coverage. Existing collector/training behavior
was not changed in this measurement step.

At the time of this measurement, terminal recording was still an open next step; the subsequent new-capture
fix is documented above and does not repair these old artifacts. Remaining decisions are to verify that fix
in a seen-level collection run, decide the treatment of conflicting old traces, and fill the measured
validation/event gaps before approving controlled training.
No game connection, actual model training, model rollout or full-archive state fingerprint scan was run here.
The reviewed artifacts under `runs/coverage_checks/` preserve the inventory, selection, exclusion and report
provenance. CLI syntax can be inspected without a data scan with `python -m wm.coverage --help`.

## Dataset snapshots and exact-trace separation (2026-11-11)

New world-model training and offline evaluation CLI calls must explicitly choose `--manifest FILE` or
`--legacy-split`. Legacy means historical global-shard-modulo behavior, labelled as such in new reports,
checkpoint metadata and learner state; it does not establish independence. `agent.py --learn` similarly
requires `--dataset-manifest FILE` or `--legacy-split` before attaching to the game and forwards that choice.
Ordinary planner/PPO/inference behavior without learning is unchanged.

`wm.dataset.Dataset` owns the versioned `trace-split-v1` protocol. A manifest freezes normalized source/shard
identities and content hashes for each selected shard, world timeline and static level file. Consumers verify
these at their loading boundary and decode the same checked bytes. Manifest creation requires an explicit
maximum number of NEW shards to fingerprint; it never scans the entire archive implicitly. Coverage records
selected/candidate counts and whether the LISTED sources were complete. Bounded/incomplete snapshots are
explicit subsets, not a whole-archive audit. Excluded demo replay folders (1-16 and 4-6, per the user's existing
preference) are rejected; ordinary non-demo gameplay on these levels remains allowed.

Split membership belongs to each agent's complete exact state/action trace, not its shard number. Trace
identity includes static semantic geometry and ordered frame/state/action fields; agent ID, wave ID and
reward are excluded. Signed zero is canonicalized and non-finite physical states are rejected. Equal traces
therefore stay together across waves/shards, even within mixed-agent crowd shards. This is NOT a guarantee
against shared prefixes, near duplicates, correlated local transitions, shared-world/crowd effects, or unseen
level leakage. Held-out LEVEL exclusions remain separate from trace validation.

`wm.features`, `wm.train`, `wm.selfcorrect`, `wm.compare_steps`, `wm.evaluate`, and `wm.learner` use the same
owner. Training masks (including normalizer, event weighting and step-k residual scaling) come from trace
membership. Validation agents cannot enter self-correction or optimizer samples. Missing train/validation
samples are an explicit error; strict mode never substitutes training rows for validation. Checkpoints record
the manifest reference/hash/protocol, actual training/cycle counts and exact parent-checkpoint hash.

Existing checkpoints are retained as legacy evidence. Attaching a new manifest never relabels their weights
as newly separated training. Strict `--init`/evaluation requires the identical checkpoint manifest. A manifest
extension is written to a NEW file, preserves old membership, and records its parent hash; it requires a fresh
training experiment in this first version. New files do not enter an old snapshot. The strict learner runs
one initial cycle (or one explicitly requested `--once` cycle) and then waits; extra files are pending, not
training triggers. Its selected validation entry IDs (at most 40) are fixed in state. Existing unrelated output
checkpoints are never overwritten by a new strict session.

Strict step-4 training and `wm.compare_steps` position evaluation are supported. The older `wm.evaluate`
event-report indexing and self-correction target generator only support step 1 in strict mode; other steps
give explicit errors instead of mislabelled results. Guard/planning/reward code has not been changed.

No new actual training or real-game evaluation was run for this step. Small synthetic tests replace fitting
and model rollouts with probes. A separately bounded review snapshot can demonstrate duplicate grouping and
missing-validation rejection; it is not automatically a training-ready dataset. Full-archive fingerprinting,
coverage measurements and a new training experiment require a separate bounded measurement brief. CLI syntax
can be inspected without starting training with `python -m wm.dataset --help` and `python -m wm.train --help`.

## Current direction and first-life evaluation (2026-11-11)

The user clarified the target: a world-model/planning agent that finishes a previously unseen level quickly
on its first life, accepting a measured speed/risk trade-off. PPO is retained as an earlier method; replaying
a successful route or searching by repeatedly playing the target level does not demonstrate this target.
The older status summaries below describe earlier stages, not the acceptance criteria for this direction.

The first implementation step adds `agent.py --eval-first LEVEL`, an opt-in evaluation protocol. It uses one
planned agent, zero explore agents, a frozen checkpoint, and no persistent death/move/best-time file reads or
writes. Training, data collection, demos, and real-game wave search are rejected. Observations from the SAME
life remain available (e.g. hazard/platform velocity); no previous wave is used to predict the future. Normal
planner and PPO behavior is unchanged when the flag is absent. Existing run evidence is never appended to or
overwritten: the evaluation requires a new empty run folder.

From `smbrl/`, with the game already on 1-18 (the user chooses the level):

```powershell
python agent.py --eval-first 1-18 --model runs/wm_step4/wm_model.pt --name first_eval_1-18_v1
python -m plan.first_eval report runs/first_eval_1-18_v1
```

Use a fresh run name for a later experiment. The process stops after success, death, or the wave timeout.
`run.json` records requested/effective configuration, the exact loaded checkpoint bytes' SHA256, checkpoint
metadata, and source revision/dirty state. `evaluation.json` records validity separately from the outcome:
success/death/timeout, or invalid/incomplete evidence. Frame 0, wave 1, the spawn, the confirmed fixed step,
and contiguous frames are required. A manual stop/level change is incomplete; malformed data or a planning /
script error is invalid. A timeout is not a death. Both game time and wall time are recorded, with planning
costs: lockstep slow motion must not be confused with real-time speed. Only the first life is acted on; the
existing go/next-frame-stop protocol releases the game without applying a second life's inputs.

A declared held-out level is not proof that a checkpoint never trained on it. `training_overlap` is `seen`
when the checkpoint lists the target in its data, otherwise `unknown`; the declared split is reported apart.
Older multi-wave logs are not relabelled as genuine unseen-level evaluations. Current checkpoints lack a
complete audited training lineage, so this step establishes an evaluation protocol, not a generalization result.

Next steps, requiring separate reviewed briefs: audit data coverage and checkpoint lineage; evaluate multi-step
physics and event prediction by situation; then simplify/compare planning rules and improve global route search
before collecting targeted training data. No reward tuning, new planning algorithm, or training was added here.
The protocol is checked with synthetic message-boundary tests, not by training a clone. **Measured live protocol
check:** the user's `runs/first_eval_1-8_check01` completed a valid first-life success: 250 frames, 4.16675 game
seconds, 8.99641 attempt wall seconds, and 46 decisions. States 0..249 were contiguous, no best route was used,
no future frame came from a previous-wave recording, and no second-wave plan was made. The checkpoint had
already seen 1-8, so this is a protocol check, not unseen-level performance or evidence of improved playing.
**Still unverified:** a clean-detach acknowledgement (not logged), overlay appearance, and general success
rates. The recorded stop behavior does not establish an acknowledged detach from the game.

This document describes the project as it is: what it is for, what each part does and why it was built that
way, what has been measured, what does not work, and where the code and ideas got tangled. Read it first
before changing anything. Any agent working on this project gets this file as its briefing.

Companion files (more detail, older):

- `PROJECT_HISTORY.md`: every problem from the start and how it was solved (reverse engineering, rewards,
  routes).
- `MBRL_PROPOSAL.md`: world model plan and results, Phases 1 to 4.
- `MPC_PROPOSAL.md`: planner plan and results, Phase 5.

Labels used here:

- **measured**: numbers from runs or tests;
- **inferred**: reasoning from code or data, not checked in the game;
- **unverified**: built but never checked in the running game.

---

## 1. The goal and the rules

**Goal (the user's words, condensed).**

- Agents that really play Super Meat Boy: real Meat Boy characters inside the user's own copy of the game,
  not a Python clone.
- They should learn the game's movement well enough to finish levels, including levels they never trained
  on.
- The second direction (since 2026-11-02): add a learned model of the game's dynamics (world model) and a
  planner (MPC) that tries possible futures before acting. Optimize for reliable prediction and
  generalization, not just a higher score on the training levels.

**Fixed constraints** (from the user, still in force):

- Real game only. No procedural levels, no Python clone for training.
- Structured game state only: no images, CNN or visual encoder. No LLM reasoning or external reasoning APIs.
- Keep PPO and the game integration working.
- New parts stay modular and switchable; with them off, the system behaves as before.
- Measure everything: validate predictions against real gameplay and prefer measurable claims.
- Planner-picked actions never train PPO as if they were PPO's own samples.

**Working rules:**

- `smbrl/` is its own git repository. Commit after every finished step on `main`.
- `ONLYPPO` is a frozen branch (the PPO-only version from before the world-model work). Never commit to it.
- Never hard-kill the trainer while it is attached to the game: it crashed the game once. Stop with
  Ctrl+C or by creating `runs/<name>/STOP`.
- Do not damage the user's save records. Agents never pick up bandages, and best times are not written to
  the game.
- Only read-only probes on a running game or a running training.
- **The user chooses the level** (they move through the game themselves). The trainer trains on whatever
  level the game shows.
- **New since 2026-11-03:** ideas are discussed and planned with the planning agent. Code is written by a
  coder agent that does only what the plan says (section 11).

---

## 2. The short version: where it stands

**Works (measured):**

- Up to 200 real Meat Boys play a level in waves; PPO learns from all of them.
  - Finished levels: 1-1 to 1-10, 1-12, 1-13, 1-16, 2-1, 2-2, 2-9, 3-3, 3-4, 4-4, 4-6.
- The game's time step is locked (16.7 ms every step), so physics no longer depends on Python's speed.
- **Hazards on the map** are the kill shapes the game itself tests each frame (section 5.2), so moving shapes
  inside a still piece (1-14's saws) are seen.
- **Leaving the level** kills an agent (since 2a08582, verified only by data analysis).
- **The latest run on 4-6:** route progress rose to 96–99% within 15 minutes and 4-6 was finished for the
  first time (12.97 s).
- **The world model** predicts World 1 movement well: 8.9 units median position error 32 frames ahead,
  deaths AUC 0.955.

**Does not work, or is unclear:**

- **No way to measure progress honestly.**
  - There is no fixed benchmark: no fixed level set, no fixed time budget, no held-out score.
  - Each run is on whatever level the user opened, and runs continue from different earlier models.
  - "Is it getting better?" cannot be answered with numbers. This is the main reason the project feels out
    of control.
- **Best-time records are partly wrong:**
  - impossible times exist: 1-3 in 0.03 s, 1-1 in 0.4 s, 1-8 in 0.72 s, 2-9 in 0.93 s;
  - times before the lock (2026-11-02) count frames / 60 while frames took about 36 ms, so they are not
    seconds.
- **The world model only knows World 1.** On World 4 it predicts certain death everywhere, so the planner
  cannot work there.
- **The planner (MPC)** finished 1-1 in the game, but it is slow (40 to 110 ms per decision) and has not
  been tested further.
- **Several changes were never checked in the running game** (section 9).

---

## 3. The repository: what is live, what is old

There are three generations of the system in the folder, plus the world-model layer. Only generation 3 and
the layer on top of it are in use.

| Generation | Files | Status |
|---|---|---|
| 1. Python clone of the game + Stable-Baselines3 | `engine.py`, `env.py`, `levels.py`, `levels_smb.py`, `train.py`, `play.py`, `viewer.py`, `watch.py`, `reward_rail.py`, `smoke_test*.py`, `tools/solve_levels.py`, `tools/capture/validate_physics.py` | **old, unused**. The user ruled out the clone. `gemini_changes.md` (Turkish) documents a Gemini agent's edits to this generation (`env.py`, `reward_rail.py`, `viewer.py`), not to the live system. |
| 2. In-game arena (SB3 policy driving copies in the game) | `ingame.py`, `ingame_arena.js` (`train.py --game`) | **old, unused** |
| 3. Real-game PPO trainer | `realgame.py`, `realgame.js`, `realpath.py`, `monitor.py` | **live** |
| 3b. Unused dashboard | `realgame_hud.py` | imported by nothing |
| 4. World model + planner | `wm/`, `mpc/`, `eval_levels.json`, `tests/test_mpc.py` | live, switched off by default (`--control ppo`) |
| Tools | `tools/smbl_extract.py` (level files: palette names), `tools/capture/*` (early RE capture scripts), `tools/extract_archive.py` | `smbl_extract.py` is used by `realgame.piece_names`; the rest are old |

**Repository tangle:**

- `smbrl/` is its own git repository (remote `origin`, branches `main`, `ONLYPPO`, `training-rewards`).
- The parent repository `REA-MeatBoy` (the REA reverse-engineering tool) **also tracks 24 files of
  `smbrl/`** from two old commits (`0ee8495b1 v1`, `22a3b6b62`). Its `git status` shows them as modified
  and the newer files as untracked. Two repositories own the same files.
- Branch `training-rewards` (82f26cc) is an old stopping point, older than `ONLYPPO`. Nothing uses it.
- `ONLYPPO` (446223e) is the PPO system before the world-model work. It **lacks fixes that also matter for
  PPO** (made later on `main`):
  - the time-step lock;
  - the measured hazard reach;
  - death on leaving the level;
  - agents seeing the outside as deadly.

  So "use ONLYPPO for PPO" means PPO without these fixes.
- `runs/` holds 567 MB of runs (ignored by git), many of them aborted tests.

---

## 4. How the live system works (generation 3)

### 4.1 Game side: `realgame.js` (Frida script inside `SuperMeatBoy.exe`)

**Agents.**
- The game keeps a pool of 40 spare Meat Boys per level (built for the end-of-level replay); with more
  than 40 agents, extra ones are built the same way at the level load.
- Each agent is switched to mode 2 (+0x9e0): full physics and collision, but no replay recording and no
  global death or level-reset code.
- The real character is hidden and frozen, and serves as the camera.

**Lockstep.** Every frame, the script runs these steps in order:
1. It sends every agent's state (x, y, vx, vy, flags, movement state, status, keys) and the level's live
   state to Python. The live state is the hazards (position, size, scale, angle), flying shots, platforms,
   tile changes, wind fans and the time step.
2. It waits for Python's inputs.
3. It runs each agent's update with those inputs.

So Python's speed sets the game's speed.

**Time-step lock** (default; `--no-lock-step` turns it off).
- The game normally stretches its step when a frame comes late (36 ms median, up to 70 ms measured).
- A 2-byte patch at 0x450ef6 makes it always take its fixed 16.67 ms step.
- **Why:** physics stays the same however slow Python is, hazards repeat exactly per wave, and the world
  model can know the future step. The cost is slow motion when Python is slow.

**Death and finish.**
- **Death:**
  - The game asks a function (0x51e7d0) whether a lethal contact kills; for an agent the script records
    the death and answers "no", so the game's own death code never runs.
  - Also counted as dead: being put back on the spawn by the game, and leaving the screen (`outOfLevel`).
  - Off screen (read from the binary, 0x520fd3): the player (mode 0 only) dies when its centre leaves the
    camera's view x 1.15. The camera is clamped to the level's camera bounds (level object +0x1d38..+0x1d44),
    so for a followed player each axis dies at bounds -/+ 0.15 x the view half size (bounds narrower than
    the view: centre +- 1.15 x half size). Mode 2 skips this check, so the script computes that kill box per
    level and wave and kills agents outside it; levels without camera bounds keep the old map-edge rule
    (10 units past left, right, top; 60 below). The agents' map and the routes treat outside cells as death.
- **Finish:** a level-end call during an agent's update counts as its finish, if the agent is within 60
  units of a goal piece.

**Waves.**
- All agents spawn together after a level reset (saws, platforms and crumble blocks back to their start).
- One life each; dead agents stay where they died.
- The wave ends when all agents are dead or finished, when `--end-after` agents have finished, or after
  `--wave-seconds`.
- Between waves the game waits for Python (PPO update).

**Keys.**
- Keys are per agent: touching a key gives it to that agent only.
- The lock tiles it opens are opened only while that agent updates.
- Bandage pickup and warps are turned off for agents (save safety).

**Drawing.** The routes from Python are drawn into the game picture through Direct3D 9.

### 4.2 Python side: `realgame.py` (one file, 1624 lines)

**Level** (`class Level`):
- the live tile map;
- goals, spawn, keys, bandages, portals, wind fans, orbs, platforms;
- which lethal pieces are decoration by palette name (world-model inputs and old recordings only);
- the walking distance to the goal (BFS).

**Hazards** (`hazard_boxes`):
- every lethal piece becomes the boxes of its kill shapes, as the game's narrow phase tests them now
  (section 5.2); rows without shapes fall back to the piece's size box.

**Map for the agents** (`grid_with_hazards`): cell codes for empty, solid, hazard, goal, wind, orb,
accelerator and platform.

**Sealed pockets** (`set_sealed`): open cells nobody can reach without crossing a wall or hazard (e.g.
inside lava pits) are filled in as hazard or solid.

**Routes** (`realpath.py`, section 5): up to 3 routes from spawn to goal, plus a bandage route. They are
rebuilt when moving hazards are seen in new places (at most 8 times per level).

**Observation:** 282 inputs (section 6). **Actions:** 10 (combinations of left, right, jump and run).

**PPO.**
- Network: separate policy and value networks, each 2 layers of 256 tanh units.
- Settings: Adam with lr 3e-4, gamma 0.995, lambda 0.95, clip 0.2, 4 epochs, minibatch 2048, entropy
  0.01.
- Update after `--batch` (16384) samples, between waves.

**Phases per level:**
- **FINISH:** until some agent finishes.
- **SPEED:** after that, finish time matters and half the agents use the fastest finisher's route.

**Saving:**
- `runs/<name>/model.pt` and `<level>_model.pt` (S key);
- `levels/<level>/best.pt` with `best.json` (fastest finish, with its action list).

**Monitor** (`monitor.py`, a pygame window in the same process):
- the map, routes, agents and hazards;
- the leader's action probabilities;
- learning curves;
- sliders for the agent count and the route shares.

**Options:**
- `--agents`, `--wave-seconds`, `--end-after`, `--batch`, `--paths`, `--load`, `--name`, `--test`;
- `--no-lock-step`;
- `--wm-collect`, `--probe-char`;
- `--no-monitor`;
- `--control ppo|mpc|mpc+ppo` and `--mpc-*`.

---

## 5. Map and routes: what and why

### 5.1 Routes (`realpath.py`)

**Problem they solve.** In a platformer the goal is often behind walls. Straight-line or BFS distance
rewards walking into dead ends, so agents got stuck in local optima.

**What a route is.** A cell path from spawn to goal found with A* and a jump model. Each search state
records how much rise and sideways travel are left in the current jump. Searches through keys are
sequenced.

**Search modes**, tried in order until one finds a route:

| Mode | Hazards |
|---|---|
| clear | closed |
| timed | moving-hazard sweeps cost extra |
| risky | hazards cost extra |
| squeeze | narrow gaps through half blocks |

**Alternatives.** Each found route pushes the next search away from it (`SPREAD_RADIUS`). A new route is
kept only if it goes a different way: fewer than `SAME_WAY` 4 tiles of overlap, with no wall between.

**Closeness field: how progress is measured.**
- Every open cell gets "route progress near here minus the walking distance to the route". Walls block,
  climbing counts double, and it reaches at most 10 tiles.
- There is one field per leg of the route (a leg ends at a key).
- **Why:** an agent should be paid for getting closer to the goal along the route, not for standing on the
  line. And a spot's value is fixed, so waiting earns nothing (earlier versions were exploited by agents
  standing in corners).

**Platforms.** Fixed platforms become walls; moving ones are "doors" that only the timed modes may cross.

### 5.2 Hazard kill shapes (`realgame.js` killShapes; replaced `hazard_reach.py`)

**Problem.**
- The map first treated a piece as deadly within 6 units of its size box, then grew each kind's box by a
  reach measured from deaths (`hazard_reach.py`, c27ff6d to this change).
- Neither is what the game tests. On 1-14 the "movin" saws never change their position: the saw moves inside
  the piece's animation. 26 of 26 benchmark deaths there were unexplained.

**Fix (measured live on 1-14).**
- The pieces check (FUN_004cd350) calls FUN_0045a0e0, which plays the piece type's animation at the piece's
  time (+0x30 + +0x38) and tests each child with a collision kind: a circle (radius max(w, h) x scale) or a
  turned box (FUN_0045b160).
- realgame.js runs that same call on a private copy of the scratch shape, against a far-away copy of a real
  character box, and records every shape tested. Each hazards row gets the shapes as an eighth field.
- `game.level.hazard_boxes` turns the shapes into boxes for the BODY contact rule. A piece with no shape now
  kills nothing now (the "particle" pieces on 1-14).
- The name rule (`DECOR`, `Level.hz_keep`) is left only for the world model's inputs and for rows recorded
  before this change. Those old rows fall back to the size box, without the old measured reach.

**Measured.**
- 1-14 saws: circles of r 20.35 (scale 0.6) and 16.96 (scale 0.5), travelling 93.6 / 78 units over the
  2.667 s loop.
- Benchmark `bench_20261108-194129`, 1-14: the deaths explained went from 0 / 26 to 26 / 26. The shapes were
  rebuilt from a live phase table, because the wave reset sets +0x30 to 0. 1 of 9735 survivor positions
  falls inside a box.

### 5.3 Moving hazards

- Agents see the two nearest moving hazards (offset and velocity).
- Routes avoid every position a moving hazard has been seen at ("sweeps"), or cross them at a cost in the
  timed mode.
- With the time-step lock, moving hazards are at exactly the same place at the same wave frame in every
  wave (measured on 1-13 and 1-14, 0.0 units off).
- **Not yet used for PPO** (only the planner uses it): future hazard positions are known from earlier waves.

---

## 6. Reward and observation (PPO)

### 6.1 Reward (per agent, per life)

| Term | Amount | Why it is there |
|---|---|---|
| Progress (closeness field) | +1 per tile, **paid both ways** | Going back costs what going forward paid, so turning back is never free. Agents used to drift backward. |
| Key | +10 once per key | Keys open the way |
| Exploring | +0.25 per new 2x2-tile square this life, less where many agents went | Find other ways in narrow levels |
| Running toward the route on the ground | up to +0.006 per frame | Run instead of hopping |
| Jump from the ground | -0.08 | Constant short hops |
| Stuck (no new best progress and no new square for 1.5 s) | -0.02 rising to -0.04 per frame | Agents stood still in corridors |
| Jump spam while stuck | -0.15 per jump after the 2nd | Jumping in place |
| Stuck cap | the two rows above total at most -4 per stuck stretch | Dying (-15) must never be cheaper than being stuck |
| More than 6 tiles from every route | -0.003 per tile per frame (up to 6 tiles) | A light pull back toward the routes |
| Death | -15 | |
| Finish, FINISH phase | +50 + up to 20 for speed; -0.002 per frame | |
| Finish, SPEED phase | +50 + 60 x best/own time (at most 1.5x) + 25 for a record; -0.01 per frame | Get faster once the level is solved |

- **Fitness** (best agent shown and saved) uses the best progress, not the return.
- **Tried and removed:** killing agents that stood still for 1 s. It made some levels impossible.
- **Inferred:** with 12 terms, many hand-tuned against something seen in the monitor, it is hard to tell
  which term drives a given behavior ("the agents do nonsense").

### 6.2 Observation (282 inputs)

| Inputs | Count |
|---|---|
| Position, velocity, on ground, on wall, facing, goal direction and walking distance | 10 |
| Local map 11x11 (cell codes; outside the level = hazard since 2a08582) | 121 |
| Own route: points 2 and 6 tiles ahead, progress, distance from it | 6 |
| Two nearest moving hazards: offset, velocity | 8 |
| Wall beside left / right, sliding | 3 |
| Wind here, in a wind column, nearest orb | 6 |
| Surface map 11x11 (floor / wall / ledge / ceiling per solid cell) | 121 |
| Stuck time, jumps while stuck | 2 |
| Nearest moving platform: offset, velocity | 4 |
| Jump key held time | 1 |

- Inputs were added over time.
- An older model loads by giving the new input columns small starting weights (`load_policy`, "widened").
- **So the current PPO model** is one network, carried across many levels with `--load`, whose inputs
  changed under it several times.

---

## 7. World model and planner (generation 4)

### 7.1 Data (`wm/buffer.py`, `--wm-collect`)

- Per wave, a shard `wave_N.npz`: one row per agent per frame (state, jump key, action, reward). Next to it,
  `wave_N_world.json.gz` with the per-frame time step, hazards, platforms, shots, tile changes and wind,
  and `levels/<name>.npz` with the static level.
- Shards record their source (`ppo` / `mpc`) and mode.
- Planner runs write to `wm_data_mpc/`.
- **Collected:**
  - `real_20261102-035612`: 183 waves, levels 1-5 to 1-11, before the lock;
  - `real_20261102-165351`: 647 waves, 17.3 million steps, levels 1-1 to 1-17, locked;
  - `real_20261103-*`: World 4 (4-6), a few hundred waves.

### 7.2 World model (`wm/features.py`, `wm/model.py`, `wm/train.py`)

**Inputs** (291), all relative to the agent: velocity, time step, jump held, position inside its tile,
wind, the 32 flag bits, movement state, action, the 7x7 tiles around it (4 channels), the 3 nearest
hazards, the nearest platform.

**Outputs:**
- the correction to plain physics (same velocity plus gravity), as mean and variance;
- the next flag bits;
- death, finish, landing, blocked;
- the next movement state and the step's reward.

**Training:**
- An ensemble of 5 networks, each 3 x 256 SiLU (size is configurable). Loss: beta-NLL.
- Self-correction: it also trains on its own 8-step rollouts (35% share).
- Every 5th wave is held out for testing.
- The current model `runs/wm_locked/wm_model.pt` was trained on 1.42 million transitions from World 1.

**Measured** (held-out waves, World 1, 16.7 ms steps):

| Frames ahead | Position error median / 90% (units; a tile is 20) |
|---|---|
| 1 | 0.11 / 0.43 |
| 8 | 1.47 / 4.98 |
| 16 | 3.70 / 13.9 |
| 32 | 8.86 / 38.1 |

- Deaths within 32 frames: AUC 0.955.
- Landing frame: right at the median.

**Not working:** on World 4 the death output says 13 to 94% per step on safe ground, because it never saw
those hazards. Its movement predictions there are fine. It needs data from more worlds.

**Held-out evaluation levels** (`eval_levels.json`): 1-18, 1-19, 1-20, 2-5, 2-6, 2-7. The trainer refuses
PPO updates and data collection there, and `wm.train` skips them.

### 7.3 Planner (`mpc/`, `realgame.py --control mpc|mpc+ppo`)

**Each frame:**
1. About 30 action sequences, 16 frames long, from four sources: the previous plan and mutations of it,
   PPO's top actions, and a library of moves (runs, jumps of several lengths and delays, double and wall
   jumps, turns).
2. All 5 models roll each sequence forward.
3. Each sequence is scored on:
   - progress on the route field;
   - death, by when it is predicted, using the worst model, plus a geometric check against future hazard
     boxes;
   - landing and keys;
   - the PPO critic's value at the end;
   - the models' disagreement.
4. The first action of the best sequence is played.

- **Future hazards** come from the level's recording of an earlier wave, or are extrapolated with a
  growing safety radius.
- **Kept clean:** planner-picked trajectories never go to PPO, and every decision is logged to
  `mpc_plans.jsonl`.
- **Measured:**
  - in the game it finished 1-1 twice;
  - a decision costs 40 to 110 ms; the game runs slower, the physics is unaffected;
  - it fails on World 4 (the death-output problem above).
- **Not built yet:** the live planning viewer (Phase 6), validation of predictions against what really
  happened (step 3), the horizon/candidate benchmark, architecture experiments, the held-out evaluation.

### 7.4 Planning agent (`agent.py`)

The explainable beam planner (`plan/planner.py`) plays the real game; no PPO anywhere. Same game side as
`realgame.py` (`realgame.js` unchanged), the time step always locked (it stops if the lock fails).

**How to run** (start the game, enter a level, then from the project root):

```bash
python smbrl/agent.py --agents 1 --wm-collect
```

```bash
python smbrl/agent.py --agents 1 --explore 40 --wm-collect
```

- `--agents N`: planned agents (0..N-1). `--explore K`: K more agents with sticky random actions (the previous
  one kept with p = 0.75) mixed with random chains of 1-3 library moves, only for world-model data.
- `--replan 2` (frames between a planned agent's decisions; in between it plays its plan), `--budget-ms 120`,
  `--model runs/wm_locked/wm_model.pt`, `--wave-seconds 20`, `--end-after 1`, `--name` (default `plan_<time>`),
  `--no-overlay`, `--telemetry 127.0.0.1:47800` (or `off`), `--death-memory smbrl/death_memory.json`,
  `--move-memory smbrl/move_memory.json` (or `none`), `--hysteresis 3` (frames, chosen from closed loops, since 2026-11-08), `--threads 2`,
  `--seed 0`.
- Stop: Ctrl+C or a file `runs/<name>/STOP`.

**Per level:** the Level with its kill box, hazard boxes with measured reach, sealed pockets, the progress
field V (outside cells closed), FutureWorld and the planner (one per run, `Planner.set_level` per level). V is
rebuilt between waves when moving hazards are seen in new places, the kill box changes or a platform reaches
new cells (at most 4 times per level). V is built on the level's start tiles, as the routes are; tile changes
(crumble blocks, locks) reach the planner's model inputs and support geometry every frame (`Planner.tiles`).

**Hazard future** (`mpc.future.FutureWorld`): a level's first wave is recorded (meanwhile the moving hazards are
extrapolated with a growing radius); later waves look the future up in the recording while their hazards match it
(time step locked), and a matching wave extends it. Until 2026-11-09 a level switch mid-wave (no wave end) left the
old level's unfinished wave as the new level's "recording": frame 0 mismatched and no wave after the first level
ever used the recording (bench_20261109-023841 / -001206 / 20261108-221922: 0 waves on 1-8, 1-11, 1-13, 1-14;
only 1-5, the first level, did), so every saw was extrapolated (half size 20.4 predicted as about 41 at +32 frames).
`new_level` now drops it: the same sequences use the recording on every wave but each level's first. Logged:
plans.jsonl `future_rec` (frames of the decision's hazard future from the recording, 0 = all extrapolated) and the
wave event's `future` {frames, recording (frames before the wave, 0 = it ran extrapolated), trusted, mismatch
[frame, units], extended}. **Unverified** in the game.

**Deaths.** Every death of a planned agent is checked with `plan.safety.explain_death` (the move into the death
position against that frame's hazards and shots, or leaving the level). An unexplained one is added to the
death memory (level, wave frame, position) and saved; the planner treats it as a hazard at that wave frame
(+-30 frames) from then on, also in later runs. Not on held-out levels.
- A death on a wave's last update (with one planned agent and no explore agents, every death) is never shown in
  a frame message: its position is the plan's prediction for that frame (`estimated` in the log), checked
  against the last frame's hazards and the ones predicted for the death frame.

**Logs** in `runs/<name>/` (JSON lines, flushed every 200 lines and at every wave end):
- `plans.jsonl`: one `Plan.to_dict()` per decision (moves, predicted path, landings, wall jumps, nearest
  hazards, veto, score, the top rejected alternatives with their reason and predicted path, predictions at +1,
  +4, +8, +16, timing, counts per stage) plus `level`, `wave` and `real` (the reported state that frame);
- `events.jsonl`: surprises (`position`, `unpredicted death`), `death` (explained or not, estimated, added to
  memory), `point of no return` (since step 4: the decision charged for the death, or none found), `failed
  move`, `finish`, `level`, `kill box`, `hazards`, `field`, `plan error`, `lock`, `wave` (the per-wave summary);
- `states.jsonl`: per frame, frame, wave, level and every planned agent's `[i, x, y, vx, vy, status]` (replay);
- `--wm-collect`: `wm_data/` shards with source `planner`, `explore` or `planner+explore`, and `agent_sources`
  (which indices were planned / explored) in each shard's meta. The reward column is NaN (no reward here).

**Per wave** one line: level, planned agents finished / died, deaths explained by geometry vs unexplained,
surprises, decisions, ms per decision (median / 95%), decisions with stage 1 / stage 2 cut short, failed moves.
(Stage 1 "cut short" counted every decision of a step-4 model until 2026-11-09: it compared model steps with
frames. It now counts decisions whose 73 moves were not all rolled out; in bench_20261109-161551 that was 1 of
10,840, stage 2 0.3-2.7% per level.)

**Same state, same decision within a level** (2026-11-09): the surprise trigger's error table (plan/commit.py
ErrorTube: the run's own 90% followed errors) used to be refreshed every wave and every 600 frames. In
bench_20261109-161551 a drift of hundredths of a unit flipped a surprise trigger that sat at the threshold and
broke 12 identical 2.9 s waves of 1-8 (wave 15) and 11 identical 5.65 s waves of 1-13 (wave 13). The table is now
frozen at each level start (the first level of a run: plan.explain.ERR90_AT). Fixed constants were not chosen:
the table's +4 entry ranged 4.4 to 12 units (median per run) over the last four runs. Other inputs that can still
differ between identical waves, measured there: memories after a failure, death or unexplained death (by design);
the hazard recording's length (decisions within 93 frames of its end are partly extrapolated until a longer wave
extends it: 460 of 2,885 on 1-8); a field rebuild after wave 1 (1-13, 1-14); stage 2 cut by the wall-clock
deadline (0.3-2.7% of decisions; removed 2026-11-10, section 7.9).

**Best route** (2026-11-10, plan/planner.py `BestRoute`): per level and run, the fastest finished wave's real
states (x, y, vx, vy) and played actions are kept (agent.py's finish -> `Planner.finished`). A decision state within
6 units (body half size) and 1 unit / frame per velocity component of a state of that wave adds the wave's recorded
actions from there as a final-stage candidate ("best route"), re-scored with every member and the full safety
check. It is chosen like an incumbent: unless vetoed, its soft risk exceeds the chosen plan's by more than
RISK_SWITCH (0.2), or another plan touches the goal sooner. A death while playing it blocks injection wherever the
route would reach that frame within the horizon, until a faster finish replaces it. Logged per decision as
`best_route` {frame, dist, frames, finish, used, why, judged, real}; the mode is `best-route` (gold path). Not saved
across runs. Since 2026-11-10 judged on its recorded real path when the match is close (section 7.9).
Offline replay of bench_20261109-161551 (live states, so open loop: it shows the choice at each real state, not
where the route would lead): 1-11 (no finish) 0 matches. 1-13 waves 13-26: the route (wave 2, 338 frames) is taken
from frame 0 through frame 114, past the live divergence at frame 94, and again at 131; then rejected for soft risk
0.45. 1-8 waves 15-26: the route (wave 3, 174 frames) matches every early decision but is rejected for soft risk
0.35-1.00 vs 0.00 (also at the divergence, frame 23), so it is taken only in the last ~35 frames where the slow
route already rejoins it. (The open point here, the world model's soft risk on a proven route, is section 7.9's
first change.) **Unverified** in the game.

**Forgiveness** (2026-11-10): a finished wave clears every failure-memory charge of the (cell, first move) pairs the
agent decided in it; an event `forgiven` lists them. A route that worked is no longer kept off by earlier loop
charges. Deaths are never forgiven (the point-of-no-return memory's death forgiveness of 65a8a0d was removed the
same day, section 7.9). **Unverified** in the game.

**Drawn in the game** (first planned agent, at most every `--replan` frames): the chosen plan's predicted path
(green), its airborne completion (since step 4: fainter green, dashed, landing yellow), predicted landings (yellow
dots), wall jumps (cyan dots), the top 3 rejected alternatives faintly (red when vetoed, grey otherwise).

**Telemetry** for a separate viewer: JSON over UDP (`viz/telemetry.py`, the schema is in its docstring):
`level` (map, kill box, goals; at the start and every 2 s), `frame` (agents, hazard boxes now and predicted at
+8 / +16 frames, remembered deaths), `plan` (every decision), `event`. Never blocks; drops are counted.

**Viewer** (`viz/viewer.py`, pygame, its own process; it never sends anything to the agent). Live, next to a
running `agent.py` (listens on 127.0.0.1:47800, `--listen host:port` for another address):

```bash
python smbrl/viz/viewer.py
```

Replay of a finished run's logs (`plans.jsonl`, `events.jsonl`, `states.jsonl`); the folder name is the run's
name that `agent.py` prints (`plan_<date>-<time>` by default), for example:

```bash
python smbrl/viz/viewer.py --replay smbrl/runs/plan_20261103-120000
```

The newest run: `python smbrl/viz/viewer.py --replay latest` (the newest `runs/plan_*` folder). Per decision it
shows the chosen path, predicted landings and wall jumps, the clearance to the nearest hazards, the rejected
alternatives with their reasons, and each prediction (+1 / +4 / +8 / +16) against the real position (the full
list is in the module docstring). Keys: Tab agent, H help, A alternatives, F future hazards, Z zoom on the agent,
S screenshot; replay: Space, Left / Right, + / -, N next event, D next surprise or unexplained death. A replay
has hazards only when the run has `wm_data` (`--wm-collect`); its +8 / +16 boxes are the recorded ones, not the
planner's prediction (that is not logged).

**Checked without the game only** (`tests/test_agent.py`: a recorded 1-13 wave through a fake script). Not yet
checked in the game: the overlay's look, decision cost with 2+ planned agents, the wave-end death estimate.

### 7.5 Step 4 (2026-11-08): scores in frames, airborne completion, point of no return (offline only)

Details, numbers and the offline checks: `PLAN_PLANNER_ROUND2.md` section 7. In short:
- **Time field T** (`plan/progress.py`): frames to the goal on realpath's graph (same states, modes, closed cells,
  key legs as V), step frames from measured speeds (**measured**: 5.00 frames per tile on the ground and in the
  air, 5.41 per tile up, 7.74 down; `plan/measure_time.py`). T_support, and airborne T_rise / T_fall by the
  body's vertical velocity. V stays for display.
- **Score** (`plan/planner.py`): estimated frames to the goal = frames used + T(end) + penalty frames; every
  expected death (soft risk, a completion where all continuations die, the point-of-no-return rate) costs
  DEATH_FRAMES = max(T_support(spawn) - T(start), 500); refuted plans: their shortfall in frames (failure
  memory below); lower is better; no uncertainty factor.
  `--hysteresis` defaults to 160 frames, the measured 90% re-score noise of a kept plan (closed loops: switch rate
  0.62 -> 0.09, but 2 deaths in the 1-14 pit kept longer; PLAN_PLANNER_ROUND2.md 7.6). Wall contact is support
  at any speed (**measured**: wall jumps work at every recorded falling speed).
- **Airborne completion** (final stage): a plan not landing by a majority of members is continued with keep /
  neutral / reverse for up to 90 frames and scored at its best landing; all die: one death (DEATH_FRAMES).
- **MoveMemory** (`--move-memory smbrl/move_memory.json`): each death charges the move taken at the latest
  decision of the wave that had a safe alternative (a repeated identical death one decision earlier per repeat, since
  2026-11-10: 7.9); `point of no return` events.
- **Failure memory, the loop breaker** (`plan.planner.FailureMemory`, since 2026-11-09; replaces the first-move
  V check, which never judged a plan whose payoff was in its later moves): every decision records its start
  state's key (cell, supported, sign of vx and vy beyond 1 unit / frame), its first move, T there and the T its
  plan predicts at its end (whole horizon). When the agent leaves the cell and comes back to the same key in the
  same wave, that plan is judged by the best real T from its decision to the return (or its end): under half the
  predicted drop is a failure charged to (cell, first move) as the shortfall in frames (what its estimate
  overclaimed), with a `failed move` event naming the loop; at least half (also at its end without a return)
  forgives the latest charge. Deaths are MoveMemory's. Per level for the run (not saved; waves keep it).
  Offline on bench_20261109-023841's real states: 1-5 wave 7's 28-frame loop is charged from frame 205 and the
  first move at the loop cell changes from 232 on (24 frames into the loop); 1-14's timeouts are not caught (its
  plans reach half their predicted drop, the progress is lost later). **Unverified** in the game.
- **Known limits (measured)**: the world model stops falling about 6 units above the map's bottom edge (its tile
  input pads outside the map with solid; open padding with the model unchanged fails, see 9.13: retrain), so
  falls out of the 1-14 pit are not predicted; after the speed work (7.6) a decision takes 142-175 ms median at
  the 120 ms budget, the completion 24-63 ms of it (over 30 ms on 1-14 and 1-11; options in 7.6, not chosen). **Unverified**: everything in the game.


### 7.6 Round 3 (2026-11-08): 4-frame world model, planning modes, Ctrl+C (offline only)

- **Ctrl+C** (`shutdown.py`): the first Ctrl+C (or STOP) stops cleanly, every Frida call bounded, then the process
  leaves with `os._exit` (Frida's exit-time teardown kept it alive; no Python thread did). A second Ctrl+C or no
  clean stop in 10 s unloads the script (hooks gone, a waiting game thread released; the time-step lock bytes
  stay patched: the game keeps its fixed step) and exits. Viewer, offline planner, training: Ctrl+C exits at once.
- **Step-4 world model** (`runs/wm_step4`, `python -m wm.train <14 locked wm_data> --step 4 --sample-frac 0.07
  --max-samples 1500000`): one step = 4 frames (the 4 actions in, events of the 4 frames out; step and target
  scale in the model file). Open-loop position error on 249 waves neither model trained on (`wm.compare_steps`),
  median / 90%, units: +4 0.79/3.56 -> 0.38/2.60, +16 5.99/30.56 -> 3.19/15.71, +48 39.79/166.57 -> 17.95/61.93
  (wm_locked -> wm_step4; wm_step4 also saw more locked data). **Default** in agent.py and plan/offline.py.
- **Planner on steps**: horizons in frames (48 / 45 = 12 steps), moves judged at the next step boundary, sweep /
  contact / guard on step positions, the chosen path interpolated to frames; commit tube at 4, 8, 16 frames.
  Offline at 120 ms: 220 -> 86 ms per decision (1-14), 210 -> 101 (2-1); 33 -> 13.5 and 31 -> 15.5 ms per frame.
- **Planning modes** (`--modes on`, default; plan/planner.py docstring): long-safe (member 0, 96 frames, blue),
  normal (member 1, green), reflex (member 2, red), explore agents on members 3 and 4 (purple, every 16 frames).
  Closed loops (9 starts on 1-13, 1-8, 1-14, 300 frames, no deadlines; the world is wm_locked member 0): 343ad35
  2 finished, 0 died, switch rate 0.20, 317 ms per decision; now 3 finished, 0 died, 0.27, 106 ms; chosen modes
  normal 95.1%, long-safe 4.6%, reflex 0.3%. Long-safe costs about 73 ms when it runs (no deadline: those
  decisions take about 163 ms); an explore agent about 30-36 ms per decision, 1.9-2.3 ms per game frame each.
- **Unverified**: everything in the game (no run with the game since).


### 7.7 Demos (2026-11-08, offline only)

- **Demo-trained step-4 model** `runs/wm_step4_demo2`: as `runs/wm_step4` (the same 14 locked wm_data folders,
  `--step 4 --sample-frac 0.07 --max-samples 1500000`, 5 members) plus `--demos runs/demos/recorded.json
  --demo-share 0.2`, without the 1-16 and 4-6 demos (9 demos; no runs/demo_* shards existed). Open-loop error,
  median / 90%, units (`wm.compare_steps`, runs/wm_step4_demo2/compare_report.md):

  | model | +4 | +16 | +32 | +48 |
  |---|---|---|---|---|
  | wm_step4, 379 held-out waves | 0.47 / 3.08 | 3.95 / 17.24 | 10.92 / 42.20 | 19.59 / 71.20 |
  | wm_step4_demo2 | 0.44 / 2.95 | 3.70 / 16.66 | 10.37 / 40.31 | 18.93 / 67.86 |
  | wm_step4, demo rows of 3 held-out waves | 0.58 / 4.09 | 5.11 / 31.59 | 24.28 / 69.71 | 43.88 / 108.23 |
  | wm_step4_demo2 | 0.65 / 4.84 | 4.52 / 25.83 | 22.24 / 78.38 | 45.89 / 98.22 |

  Closed loops (plan/offline.py closed_loop, 300 frames, no deadlines, modes on, hysteresis 16; starts 1-13 0 /
  100 / 200 and 1-8 0 / 30 / 60 from real_20261102-165351, 1-14 0 / 100 / 200 from plan_20261107-200300: the
  folders with the longest second waves): wm_step4 4 finished (1-8 x3, 1-14 from 100), 0 died; wm_step4_demo2 3
  finished (1-8 from 30 / 60, 1-14 from 0), 0 died. Fewer finishes: **runs/wm_step4 stays the default**.
- **Replay fix**: the first replays (runs/demo_*_20261108-*) wrote no shard: copies left standing after the
  demo's actions kept the wave alive to --wave-seconds (20 s), and S / Ctrl+C dropped the wave. Demo waves now end
  1 s after the actions run out ("demo done"), and any wave cut by S, a level change or a stop is written as a
  shard marked "partial" (agent.py and realgame.py). The replays need re-running.
- **User exclusions**: 1-16 (very slow, would teach unwanted movement) and 4-6 (finished through a bug) are
  USER_EXCLUDED (plan.demos.USER_EXCLUDED): not replayed, not in recorded.json, their replay folders and demo
  rows skipped by training. Nothing deleted.
- **Train mode** (`agent.py --learn`): wm.learner fine-tunes in the background (2 threads, below-normal priority)
  on every locked wm_data folder plus the demos and the agent's own new bests (runs/demos/self.json), every 20 new
  shards or new self-demo, and publishes runs/wm_live/wm_model.pt only when the median error at +16 and +48 on 40
  fixed held-out waves is not worse; the agent loads it between waves. One cycle on the current data (16 folders,
  600000 transitions, 1 epoch): 603 s, of which 544 s building the features, 43 s training, 16 s evaluating; it
  improved wm_step4 on the 40 waves (+16 4.10 -> 4.03, +48 19.57 -> 19.29 median; a test folder, not runs/wm_live).
  **Unverified**: everything in the game, including the learner beside a live run.

### 7.8 Standing still fixed: one window per stage (2026-11-08, offline only)

- **Problem** (bench_20261108-185846): 1-11 0/6, 1-13 0/2, 1-14 0/3 finishes; decisions keeping a do-nothing plan
  1-11 717/885, 1-13 376/425, 1-14 198/324; the user's 1-1 run plan_20261108-172037 stood at the lower right wall
  (154, -154) for about 1200 frames each wave. Cause: since 4e16bf0 the kept plan was judged on H - shift frames,
  the others on H; estimate = frames + T(end) with an optimistic T (p90 speed), so the shorter window won.
- **Fix** (`b08aea4`): final stage, every candidate ends at H (unplanned tail = pad). Stage 1 ranks every move at
  its common end (24 frames). `--hysteresis` 3. (b08aea4 judged veto and soft risk on the planned frames only;
  superseded below.)
- **Safety window now** (`1559563`, then the tail rule after it): safety covers every frame the score credits.
  Final stage, a candidate without a goal touch, frames 1..H: veto on 1..VETO_FRAMES (16); soft risk by the ERR90
  margin on its later planned frames (pads excluded; the kept previous plan: its remaining planned frames);
  on the never-planned pad tail up to H only real contact counts (soft risk 1, no margin). Stage 1, frames 1..24:
  veto and margin on the move's own frames, contact only on its held last action. A goal touch at t: frames
  1..t, margin rule. Stage 2, long-safe, reflex and explore judge the frames they score (unchanged).
  counts["final"]["safe_alternatives"] (MoveMemory's point of no return) uses the window's risk plus the
  completion's deaths.
  - Why: b08aea4's planned-frames window credited a pad tail through a saw with the progress of passing it and
    no risk (bench_20261108-221922: 1-8 4/99, 95 identical deaths on h2; 150 of 160 deaths had an adopted plan
    with no veto, soft < 0.5 and hazard contact in its tail). 1559563 judged 1..H with the margin everywhere;
    live (bench_20261109-001206) 1-8 34/47, overall 61% (was 18%), but 1-13 0/16 (all timeouts): passages near
    saws were refused, the margin (38 units at 32 frames, 60 at 48) on the pad tail cost most of a death (1-13:
    2628 of 3393 logged near-miss maxima in the tail). The contact-only tail keeps the 1-8 holes caught.
  - Offline (replays with the logged plan as the incumbent; closed loops on 21 starts): see the commit messages
    of 1559563 and the tail rule. **Unverified** in the game: the tail rule.
- **Still share** (`5209c61`): plan.analyze_run.is_still (chosen plan only wait / pad), per level in
  plan.benchmark. **Benchmark** (`3f19070`): `python smbrl/plan/benchmark.py cmd | report <run>` from the folder
  above smbrl/; the run command keeps the overlay and passes --wm-collect. Since 2026-11-10 it also gives the run
  its own empty death and point-of-no-return memories (`--death-memory` / `--move-memory` in runs/<name>/), and
  the report shows median_s next to best_s (also in the deltas).
- Closed loops (13 starts: 9 as before + 1-11 x3 + 1-1 demo; 300 frames, no deadlines): 9cabc9b (h16) 4 finished,
  4 died, switch 0.29, still 0.27; now (h3) 4 finished, 3 died, switch 0.58, still 0.06. 1-1 from the right wall
  (plan_20261108-172037 wave 2): 9cabc9b stands and dies; now leaves (h3: moves, no finish in 300 frames; h16
  finishes). The closed loops are noisy (one start flips between finish and death across settings).
- **Unverified**: everything in the game; 1-11 dies in every closed loop (old and new).

---

### 7.9 Real-path route, deterministic search, repeated deaths, held-out block (2026-11-10, offline only)

Live evidence (bench_20261110-003407, ab505ab): 1-14 33/34 finished (median / best 7.5 / 6.3 s), 1-13 20/32 (13.1 /
8.4 s), 1-8 15/67: waves 1 and 2 finished, then 50 waves died identically at frame 234 on the static saw h4. Offline
checks replay the run's logged states through Planner.observe / plan / played / finished with the FutureWorld flow as
agent.py (scratch harness rep.py; open loop: the states are the live ones, so it shows each decision, not where it
would lead).

- **Best route on its real path** (`e0d7833`, plan/planner.py `Planner._route_real`). Before, the route candidate was
  judged by the world model's imagination: 1-8's route (wave 1) was refused at frame 0 of every later wave with
  "soft risk 1.00 vs 0.00". With the time-step lock the game repeats the route from the same state, so when the
  state is within REAL_RADIUS 1 unit and REAL_SPEED 0.5 u/f of the route's state and the hazard future of the judged
  frames is the trusted recording (`plan(recorded=...)`, agent.py passes future_inputs' count), its veto, soft risk,
  goal touch and estimate come from the recorded real path against the decision's deadly boxes (margin dist + dv x
  frames: 0 for an exact match, contact only; never ERR90). Window: to the finish when it lies within the future
  frames (estimate: finish frame - route frame + 1), else every recorded frame they cover (frames + T at the real
  end). A route state at another wave frame falls back on levels with moving platforms or changed tiles. Else the old
  judging; `best_route.judged` says which ("real" / "model: why"). A chosen real-judged route predicts its real path.
  Offline: 1-8 waves 2-20 the route is taken at frame 0 of every wave (19/19, live 0/19 after wave 2); 1-13 (second
  session) waves 2-18 1,093 route decisions (1,069 real-judged; live 887). Closed checks (the best wave's states
  replayed as the next wave, a decision every 8 frames): 1-8 17/17 and 1-13 70/70 decisions take the route from
  frame 0 to the finish. Real-judged uses whose recorded path touches the current wave's real hazards: 0 (108,000
  frames checked); 9 model-judged uses (4-6 units off, the old rule) touch h2 64-96 frames ahead.
- **Deterministic search** (`e0a8be9`). The frame-99 split of 1-8 waves 2 / 3 (same state, 'run-jump R hold 22' vs
  'run wall-jump R hold 15') was the hazard recording's length, not the deadlines: replayed without deadlines both
  waves' 17 decisions up to frame 99 are the live ones, and wave 3's state with wave 2's recording (35 of the 96
  future frames recorded at frame 99, the rest extrapolated) chooses wave 2's move. From wave 3 on that input no
  longer changes. The deadlines were nondeterministic as well (offline at 120 ms, the same decisions repeated: 196
  of 408 identical); every stage now runs complete, capped by its candidate and step counts (budget_ms only
  reports), and Terrain.on_padding probes only bodies near the map's bottom edge (same result). Same decisions,
  same machine (shared with a world-model training, so slower than live): median ms/decision 220 -> 214, cuts per
  decision stage 1 26 -> 0 and stage 2 686 -> 0 of 816, repeats identical 408/408.
- **Repeated identical deaths** (`ac166af`, plan/planner.py `MoveMemory.died`). The 50 1-8 deaths were all charged to
  the decision at frame 164, whose 17 candidates all touched the goal in the model (the charge, 6,500 frames by wave
  20, only broke ties); no decision after 164 had a safe alternative, 156 and 143 did. Now a death within
  REPEAT_FRAMES 2 wave frames and BODY units of an earlier one of the run is a repeat, and its r-th repeat charges
  the (r + 1)-th latest distinct (cell, first move) pair with a safe alternative: each repeat one decision earlier
  (events: `repeat`, `walked_back`). A finish forgives no death (65a8a0d's death forgiveness removed; the failure
  memory's stays). Offline, the whole 1-8 sequence (open loop): before, every repeat charged 164 and nothing changed;
  now the charge moves 164 -> 156 -> 143 and the first move at 143 changes after the 2nd repeat (the second series,
  12 units away, also at its 2nd repeat). Finishing waves 1, 2, 13, 14, 15 decide as before; open loop the live death
  keeps coming, so the charge walks back to frame 17 by wave 34 and the later finishing waves 48-64 differ from frame
  16 on (450 of 559 finishing-wave decisions unchanged). In the game a changed choice changes the states: not
  verified. A DeathMemory box at the death place would change nothing there (the alternatives' imagined paths never
  come near it).
- **Benchmark held-out block** (plan/benchmark.py, benchmark.json `held_out_levels`): 1-18 and 1-20 (World 1 like the
  trained levels, which the world model knows; 1-18 five spinners, 696 units spawn to goal; 1-20 twenty saws and 21
  crumble blocks, 541 units; not 1-19, 2,259 units, nor the World 2 levels the model never saw). They are played in
  the same run after the five levels and reported in their own "held-out" block (never in the overall). Held-out
  protection, checked: agent.py collects no world-model data, records no self-demo and charges no death or
  point-of-no-return memory there (tests/test_agent.py), and plays them with the same command; realgame.py makes no
  PPO update and collects no data there (eval_levels.json's "refuses without --eval" was wrong: there is no --eval,
  it skips). In-run state (failure memory, best route, hazard recording) still adapts across the waves of a run
  there; nothing is saved.
- **Unverified** in the game: all of the above. Not measured: ms/decision on an idle machine (live was 133-186 median
  per level at ab505ab).

### 7.10 Wave search in the real game (2026-11-11, offline only; PLAN_GAME_SEARCH.md step 3)

`agent.py --search` (`--search-agents`, default 50; `plan/search.py`): the game is the simulator, no world model, no
planner. Step 2 (**measured**, 196 waves of 41-50 agents): crumble blocks are shared (the first toucher breaks a
block for everyone 29-30 frames later) and one agent's replay reproduces exactly (780 / 780 trajectories).

- Nodes: actions from the wave start, end frame, real end state; one per cell key (cell, supported, sign of vx / vy,
  keys), the soonest. Selection by T (plan/progress.py) + 1 frame per branch already played + 60 x the death share.
  Agent 0 replays the best node (the fastest trusted finish, else the lowest T) and is compared with the record
  (0.5 units, 30 u/s, keys; a finish: the same frame); a mismatch drops the node ("interference" when another agent
  touched a special tile on its recorded path first, else "mismatch"). The others replay their node exactly, then a
  1-3 move library chain or (25%, when a longer sequence goes through the node) a perturbed copy of that sequence's
  rest; after the first trusted finish half of them play perturbed copies of the fastest finish.
- Insertion after the wave at move ends (copies: every 8 frames); not within 10 frames of the agent's death, not
  untrusted. Guard: a state is trusted only if every special tile (bit 8) that changed by its frame and that its
  agent touched (centre within 7 units of the cell) changed at that agent's own first touch + 29-30 frames (crumble);
  any other change on its path (another agent's break, a block reached after it broke, a non-crumble special tile)
  makes it untrusted. Tile words compared without bits 22-25 (open-neighbour flags). Levels without special tiles
  skip the guard; the verifier runs on every level.
- The wave ends 10 frames after the longest sequence ran out ("search done") or when all are done. Logs:
  search.jsonl per wave, finish / death events and times.jsonl rows for every agent (source "search"), states of
  every agent, the wave event's replay_route and all_agent_frames, shards with source "search", the level event's
  model ("none (search)"; planner runs now record their model path there too). The benchmark report reads a search
  run (finishes of source "search", game frames over every agent, a wave with a trusted finish recorded counts as a
  replay wave).
- Held-out levels: searched like any (no shards, no memory, the archive lives in memory for the run). The planner
  is unchanged with the search off: tests/data/search_off_golden.json (replies, decisions, shard labels of three
  configurations) was recorded at af19e82 before the change and is compared exactly.
- **Unverified** in the game: everything above (tests drive agent.py with a fake 1-D game). Open (inferred from step 2's tile words, not checked): Level.apply_tiles
  and plan.geometry.solid_now take any change of a special tile's word as "gone", so an intact crumble block whose
  open-neighbour bits changed counts as broken for the planner's terrain (and the search's support probe); the
  search's own break detection masks those bits. Crowd interference can drop a good node on crumble levels (the
  verifier's "interference" label counts it); step 4 shows how often.

## 8. Results so far (measured)

### 8.1 Best finish per level

These come from `runs/*/levels/*/best.json`.

| Level | Best (s) | Run | Wave | Note |
|---|---|---|---|---|
| 1-1 | 0.40 | real_20261102-023821 | 289 | **impossible**, false finish or a time unit problem |
| 1-2 | 1.82 | real_20261102-023821 | 400 | before the lock: not seconds |
| 1-3 | 0.03 | real_20261101-142145 | 33 | **impossible** (2 frames) |
| 1-4 | 3.70 | real_20261102-023821 | 433 | before the lock |
| 1-5 | 5.28 | real_20261102-165351 | 130 | locked |
| 1-6 / 1-7 / 1-8 | 1.60 / 1.57 / 0.72 | real_20261102-035612 | | before the lock; 1-8 suspicious |
| 1-9 | 4.75 | real_20261101-151650 | 124 | before the lock |
| 1-10 | 2.55 | real_20261102-035612 | 164 | before the lock |
| 1-12 / 1-13 / 1-16 | 1.78 / 4.35 / 15.68 | real_20261102-165351 | 390 / 487 / 605 | locked |
| 2-1 / 2-2 / 2-9 | 2.32 / 2.65 / 0.93 | | | before the lock; 2-9 suspicious |
| 3-3 / 3-4 / 4-4 | 3.68 / 4.68 / 3.35 | | | before the lock |
| 4-6 | 12.97 | real_20261103-012157 | 49 | locked, after the map fixes |

- **Time units:** the time is (finish frame + 1) / 60. Only since the lock is a frame 1/60 s. Before it, a
  frame was about 36 ms, so older "seconds" are about half the real time.
- **Impossible times:** a finish only needs a level-end call while the agent is within 60 units of a goal
  piece. The very short ones point to a stale level-end call near the spawn, or a goal near the spawn.
  **Unverified.**

### 8.2 The latest 4-6 run (`real_20261103-012157`, 50 agents, after the map and out-of-level fixes)

| Waves | Route progress (max / mean) | Note |
|---|---|---|
| 1 to 20 | 20% → 40% / 24% | |
| 30 to 40 | 50% → 90% / 35 to 50% | |
| 49 | 100% | first finish (12.97 s) |
| 50 to 59 | 96 to 98% / 75% | no further finishes; agents die in the last part |

- PPO entropy fell from 0.63 to 0.37 and KL is very small (about 0.002). The policy is getting
  deterministic.

---

## 9. Problems and open questions (ranked)

1. **No benchmark, so no way to tell better from worse.**
   - Runs differ in level, start model, agent count and code version, and have no time budget.
   - Changes are judged by looking at the monitor.
   - Most of the rest of this list could only be found because of that.
2. **Finish records are unreliable:** impossible times exist, and times before the lock are in different
   units. Fitness, best models and the SPEED phase all build on them.
3. **One PPO model carried from level to level, with inputs added under it.** No measurement of whether
   it keeps what it learned (old levels) or generalizes (new ones).
4. **The reward has 12 hand-tuned terms.**
   - Each was added against one observed behavior.
   - Their interactions were never measured (e.g. explore vs stuck vs progress both ways).
5. **`realgame.py` is one 1,624-line file** holding the level model, map, hazards, routes glue, rewards,
   observation, PPO, saving and Frida I/O. Every change touches it.
6. **Old generations, two repositories, stray branches.**
   - Generations 1 and 2, `realgame_hud.py` and `gemini_changes.md` sit next to the live code.
   - The parent repository tracks old copies of `smbrl/` files.
   - `ONLYPPO` lacks fixes PPO needs.
7. **Changed but never checked in the running game:**
   - the out-of-level margin;
   - the jump re-press on walls;
   - platform boxes against the drawn platforms;
   - the hazard reach for kinds measured only on World 1;
   - what accelerator tiles do;
   - whether the lava fill ever closes a real area.
8. **The world model's knowledge is limited to World 1**, so the planner is blocked on other worlds.
9. **The planner is slow** (40 to 110 ms per decision), mostly a fixed cost per frame ahead.
10. **Agents stall in the last part of a level** (4-6 at 96%). Inferred causes, not measured:
    - the policy has become nearly deterministic (low entropy);
    - SPEED-phase rewards change once one agent finishes.
11. **The S key saves `unknown_model.pt`** when the game reports the level name late.
12. **Some slope tiles are modelled wrong (measured 2026-11-07, not fixed).** Real recorded agents are inside
    the solid triangle `plan/geometry.py` gives some 45-degree slope cells:
    - 1-13: about 5,900 alive rows overlap by more than 1 unit, nearly all in cells (27,22) word 0x424c301d,
      (14,17) and (12,15) word 0x414ce00c, (12,20) 0x414c001d; agent centres at 30-40% into the cell;
    - 1-7 (27,38) 0x428c2003 and 1-6 (28,8) 0x428c200d: about 180 rows each.
    - Measured with `python -m plan.geometry --overlap` (slopes on) against `--no-slopes`: 9,817 vs 2,487
      rows at 1 unit over 2.09 M rows of 5 runs.
    - Suspected cause: the open-corner decode (`realpath.slope_open_corner`, rotation / flip bits) is wrong
      for these words (second byte 0x4c / 0x8c), or these cells are not solid triangles in the game (e.g.
      corner pieces the player passes). Until it is resolved the planner's collision guard leaves slope cells
      open; support, safety and the progress field still use the slope shapes.
13. **The world model cannot fall out of the map (measured 2026-11-08, not fixed).** `wm/features.LevelGeom`
    pads outside the map with solid, so imagined bodies stop about 6 units above the map's bottom edge; a fall out
    of a level whose kill box is below the map (the 1-14 pit) is never predicted. Opening the padding below the
    map with the model unchanged was measured and **fails**: on 1-14 rows within 3 tiles of the bottom, 1-step
    error 0.87 -> 1.05 units (median; 90% 2.03 -> 2.29) and 16-step 11.6 -> 12.3 (28.6 -> 31.7); the pit's no-jump
    bodies now die (106 of 120 continuations, before 9), but none of the 30 recorded wall-jump-backs off the pit's
    left wall is predicted any more (before 12 of 30): the model never saw open cells there.
    **Follow-up: retrain the world model with open padding below the map**, then repeat that measurement and
    remove the strict xfail `test_completion_over_the_1_14_pit_sees_every_continuation_die`.

---

## 10. How the project lost direction (retrospective)

This is a summary of the conversation history, for the next plan.

- **The work turned to the world model and planner** on 2026-11-02 (the `ONLYPPO` split). Phases 2 to 5
  took most of the effort.
  - The training-quality problems of PPO stayed open: no benchmark, unreliable records, reward complexity.
- **Fixes that matter for PPO were made on `main`** in the middle of that work: the lock, the hazard reach,
  out-of-level death, the outside counting as deadly.
  - So `main` mixes "PPO fixes" with "world-model work", and `ONLYPPO` lacks the fixes.
- **Most changes were reactions to a screenshot or a monitor observation.**
  - They were not tested against a fixed before/after measurement.
  - Some later had to be reverted (the still-kill) or corrected (the hazard margin counted twice).
- **Instructions with placeholders were run as written**, e.g. `<run>` in a command, which emptied the
  hazard table. Commands given to the user must be complete and safe to run as is.
- **The planning agent wrote code itself.** Ideas, design and code changes came in one stream, so it was
  hard to see what was decided and why.

---

## 11. Working agreement from now on

**Roles:**
- **The user** decides goals and priorities, runs the game and the training, and chooses levels.
- **The planning agent** (this conversation) discusses ideas, analyzes, writes plans and briefs the coder
  agent. It checks the coder's result against the plan and reports to the user. It does not write
  production code itself.
- **The coder agent** does only what its brief says, in the files the brief names.
  - It does not change design, rewards or defaults unless the brief asks.
  - It runs the tests the brief lists.
  - It commits each step on `main` with the message format below and reports what it changed and what it
    could not do.

**A brief to the coder agent always contains:**
- the goal in one sentence;
- the files to change;
- the exact behavior wanted;
- what must stay unchanged;
- how to test it;
- what done looks like.

**Commands for the user are complete and safe to run as is**: no placeholders.

**Commits:** Conventional Commits (`feat(...)`, `fix(...)`, `docs:`), one step per commit, on `main` of
`smbrl/`. Never on `ONLYPPO`.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| agent | one real Meat Boy character driven by the trainer |
| wave | one attempt of all agents together, one life each |
| route / path | a planned way from spawn to goal (`realpath.RewardPath`) |
| leg | the part of a route up to the next key |
| closeness field | per cell, route progress minus walking distance to the route; progress is paid by its change |
| FINISH / SPEED phase | before / after the first finish on a level |
| sweep | every position a moving hazard has been seen at |
| sealed pocket | open cells nobody can get into, filled in as hazard or solid |
| kill shapes | the circles / turned boxes a piece's animation tests for a kill now (`realgame.js` killShapes) |
| lock / time-step lock | the patch that makes the game always take its fixed 16.67 ms step |
| world model | the learned predictor of the next state (`wm/`) |
| MPC / planner | tries action sequences in the world model and plays the first action of the best (`mpc/`) |
| held-out levels | levels nothing trains on, for the final test (`eval_levels.json`) |

## 13. Commands that work as written

```bash
python smbrl/realgame.py --agents 100 --end-after 5 --load smbrl/runs/real_20261103-012157 --wm-collect
```

```bash
python -m pytest smbrl/tests -q
```

```bash
python smbrl/agent.py --agents 1 --wm-collect
```

```bash
python smbrl/agent.py --agents 1 --explore 40 --wm-collect
```

The wave search (section 7.10; 50 agents: enter or restart the level after attaching so all 50 exist):

```bash
python smbrl/agent.py --search --wm-collect
```

The benchmark (prints the full agent.py command, the five levels and the two held-out ones, and the report command):

```bash
python smbrl/plan/benchmark.py cmd
```
