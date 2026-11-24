# ONLYPPO project state — 2026-11-20

## Commit / Colab handoff — 2026-11-24

Read COLAB_HANDOFF.md first for the next agent: exact user scope, local data/checkpoints,
missing conversion/export/resume/comparison tools and validation limits. Colab work has not
started. Main pre-existing planner/performance edits remain separate from this session commit.

## Requested Colab WM retraining scope — 2026-11-24

User requests a Colab world model trained from all collected valid data EXCEPT chapter4,
then comparison with earlier WMs. Explicit answer includes the previously held-out nonchapter4
levels; old eval_levels.json exclusion list is not the requested new training scope. Runtime
exclusions have NOT been changed yet. Keep independent trace/attempt validation and freeze sources.
Preparation blockers verified: ONLYPPO native_frames.jsonl.gz has no WM-shard converter yet
(raw sh/shx parity does not make it a directly loadable wm_data shard; reward/held/terminal
missing evidence must not be fabricated). All-source scan also finds thousands of test/tmp
fixtures: inventory is analysis/colab_all_data_20261124/inventory.json, with85 collected-source
candidates/12818 shards before quality/duplicate checks, plus native runs. Not an audited final
training set. Do not blindly recursively glob every wave npz into training.
Current compare_steps strict path requires all checkpoints share a manifest; old strict models
have different manifests. Existing --legacy-split cannot bypass that. A common external
comparison adapter with train-overlap labels is needed, without relabeling checkpoints.
Baselines inspected: human_v3_gpu step4,tile-shapes-v1,width419,5x256/3 layers; human_v4_gpu
step4,tile-accel-v1,width664; human_v4p_gpu step4,tile-plat-v1,width664. All gravity-v1; all
have some chapter4 training listed (2,2,6 levels respectively), so chapter4 generalization
comparisons are not equally unseen for old/new models. New baseline-aligned training proposal:
step4,tile-plat-v1,gravity-v1,5 members,hidden256,layers3,seed0; start fresh (current --init
skips base epochs). Train --load-features on Colab after local source freeze/conversion/split.
Do not silently sample 'all data': default max_samples500000 caps; all transitions require
explicit uncapped preparation and a RAM/streaming plan. build caps after feature construction,
so lowering max_samples alone does not bound preparation RAM. Current trainer saves final
weights after fit, not resumable optimizer checkpoints per epoch. No training/export started.

## Shared projectile observations ported to ONLYPPO — 2026-11-24

User authorized main's projectile observation mechanism in ONLYPPO, with raw WM-data compatibility.
Report: main analysis/ppo_projectile_port_20261124/report.md; verify_replay.py/replay_parity.json.
Byte-identical game/projectiles.py in both worktrees owns parse_shx, existing Launcher period
contract, spawn grouping and shot_track slot detection. Main game.level re-exports parser;
mpc.future delegates observation-only helpers; future trajectories/WM/planner stay in main.
ONLYPPO imports none of wm/mpc/plan/agent. Raw sh/shx alignment/units and WM collector ROW/frame
schema unchanged. Native shx speed now reaches PPO even on first visible frame; unknown metadata
keeps fallback estimation. No unverified native fire timer or hardcoded1-20 period was added.
CLI projectile inputs default on (--projectile-inputs; --no-projectile-inputs for old widths).
Append2x13 observed source/period/phase fields after prior inputs:524->550 with shape maps,
282->308 without. period_known/phase_known/velocity_known explicit; period persists within level,
phase resets each wave and expires after a missed expected shot. Verified periods rank before
unconfirmed short tracks, then nearest source. Before first fresh birth/known period no exact
phase claim. This is main's observed shot-clock knowledge, not full FutureWorld forecasts.
Zero-column migration preserves initial actor/value outputs;308->550 relocates clock columns
past inserted shape maps; dropping trained features refused. Saved metadata/schema/source hashes
updated. Original checkpoints untouched; further training needed to use new inputs.
Validation:ONLYPPO106 tests passed;main41 focused passed;601 actual1-20 recorded frames matched
main FutureWorld's source periods/last births/ids/radii;four verified119-frame sources. Native
runtime untouched/no live gameplay test launched; completion improvement remains unmeasured.
Continue from runs/aplus_v2_batch2048/model.pt into new --name aplus_v2_projectiles using
--shape-inputs --projectile-inputs --realtest --batch2048 --record-states, after Ctrl+C saves
current training. Details/limits/command and raw-schema-vs-network-vector distinction in report.

## PPO 1-20 shooter timing audit (proposal only) — 2026-11-24

ONLYPPO analysis/ppo_1_20_shooters_20261124/report.md/results.json and audit.py preserve
native evidence. Read-only run aplus_v2_batch2048 snapshot53waves/2120trials/0finishes,
2015deaths.127 agent-wave attempts enteredx>150. Of99 observed deaths in right/end region,
73 overlap launcher1 projectile and20 static hazard (contact matches, not native cause flags).
Projectile contact ages0..24frames,median5;42/73 within5frames, some on birth frame.
Four kind0 launchers: observed flying births every119frames,first sampled frame85;
projectile speed360units/s. Do not hardcode this period as a universal shooter rule.
PPO sees flying shot map and nearest2 moving hazards within200units, but no shooter timer,
pre-fire/reload phase or wave clock. Policy is a feedforward MLP. Native shx velocities are
sent but Python currently estimates velocity by position differences, zero for first-seen IDs.
This supports the missing pre-shot-information hypothesis, not proof of sole cause or that
phase inputs alone guarantee completion. Existing static launcher rules mainly validate
kind5; actual1-20 kind0 state/timer interpretation needs verification against native births.
Minimum proposed change: expose native shooter position/direction/phase as appended PPO
features with existing zero-column widening, and use aligned native shx velocity for new
shots. Keep ONLYPPO free of WM/action planners; no generic waiting penalty. No runtime,
weights, live eval or game state was changed by this audit; no new features implemented yet.

## ONLYPPO shorter PPO feedback batch — 2026-11-24

User authorized the 1-20 audit's small batch experiment. ONLYPPO realgame.py now defaults
--batch to2048 (was16384); explicit --batch still overrides it. Updates remain between waves
once the threshold is met; the full collected batch is consumed, not truncated to2048.
No reward, architecture, learning-rate, minibatch-size or realtest progression changes.
Main runtime unchanged. Existing trainer processes retain their parsed setting until restarted.
No checkpoint was overwritten or live run started by this edit. Continue from the latest
runs/aplus_v2_1_14_continue/model.pt in a new run, e.g. --name aplus_v2_batch2048.
Validation: CLI default2048 and explicit override16384 checked; Python compilation/diff checks.
Live improvement remains unverified; compare first-saw clearance/survival, not wave count alone.

## PPO 1-20 early-death audit (no runtime change) — 2026-11-24

ONLYPPO analysis/ppo_1_20_20261124/report.md/results.json and audit.py preserve evidence.
Original realtest kept running; no game attachment, policy update or model overwrite by audit.
Full snapshot207 waves:0/8280 finishes,8279 deaths/1 timeout. Most sampled individual deaths
occur frame5 near(-320,-220), under overhead native saw center(-319.88,-173.50),radius47.48;
spawn(-330,-250). Initial reconstructed route points left, but policy often jumps/right.
Hazard is present in reconstructed local PPO grid. Exact original live routes not recorded.
Only100589 action samples accumulated in207 waves (~1.4 full40-agent30s waves). Production
batch16384/end-of-wave reset simulation implies6 PPO.update feedback rounds (each includes
multiple minibatch gradient steps), not207. Initial batch carry is unknown; min/max both6.
Sampled reconstructed returns median~-14.75, dominated by death-15; no positive death exploit
shown. Current tensors finite,maxabs3.345. Identical reconstructed spawn-neighborhood probes:
1-19 best jump79.54%,current61.24%,earlier1-13 best28.64%. Some behavior changed; no global
numeric collapse. Alternative safe starts remain rare. Reconstructed rewards/logits are not
original training values; full caveats in report. First small experiment: existing --batch2048
in separate realtest, keeping originals/rewards intact; feedback-frequency benefit is a
counterfactual scheduling result, not verified completion improvement. No new blanket jump,
waiting penalty or architecture change was made. More wave count alone is a poor measure here.

## PPO 1-15 weight/replay audit (no runtime change) — 2026-11-24

ONLYPPO analysis/ppo_1_15_20261124/report.md and results.json contain evidence, checkpoint
SHA256s and reproducible scripts. User run aplus_v2_1_14_continue remained active; only a
model snapshot and recorded frames were read. At summary snapshot:1-14 had143/880 finishes
in22 waves;1-15 had0/1200 in30 waves,1146 deaths/54 timeouts. Wave26 real jumping repeats
under the lower slab;358 grounded-to-upward-air takeoffs,89.31% jump requests. Unlike1-14,
this is not simply held jump with no physical jumps. Later wave30/partial31 clusters are
higher (-80..-60 vs -240), so absolute no-progress is not supported; still no finish then.
Seven checkpoint tensors all finite; no NaN/Inf/numeric explosion or global constant policy.
Same121 reconstructed lower-region input probes: jump mean before long1-14 stay98.97%,
after96.10%,successful1-14 best92.89%,current1-15 snapshot97.55%. Bias predates70-wave stay;
current left/right still differ and entropy~0.805nats. No controlled causal evidence that70
waves alone caused collapse. Fixed historical1-9 action agreement drifts72.5->62.4% across
long1-14 stay (current64.2%), suggesting possible partial forgetting, NOT live success rates.
Exact native later-wave routes/logits not recorded: offline probe routes were reconstructed
without prior-wave hazard sweep memory. Full limitations in report. No new penalties,
auto jump changes, weight reset, game attachment or eval gameplay were added. Suggested
next test targets ceiling/lateral progress, with careful waiting/wall-jump preservation;
retention needs old-level live checks and on-policy rehearsal, not naive old PPO replay.

## PPO 1-14 progress visibility implemented — 2026-11-24

Supersedes the proposal-only diagnosis below. ONLYPPO realpath.py now rejects local field
candidates across solid tile geometry using cached analytic polygon clipping from existing
shape owners. No PPO inputs/reward coefficients/jump semantics changed. Dynamic special tiles
retain prior handling because static route words cannot determine per-agent opening/breaking.
96 tests passed, including rotated/flipped half/slope geometry, clear/full-wall connections,
above/below-floor separation and dynamic special compatibility. Main runtime was not changed.
Evidence: ONLYPPO analysis/ppo_1_14_20261124/report.md, retest.log, retest_results.json.
Learning-on realtest from preserved aplus_v2:5 completed waves,0/200 finishes,9 deaths,191 timeouts.
Test saved to runs/aplus_v2_1_14_visibility_fix/model.pt and stopped; original checkpoint SHA
unchanged, native timing bytes restored7c53. Correction is implemented;1-14 is NOT solved.
Ledge jump action fraction99.91..99.97%, actual hops0.0/s first3waves and0.1/s last2: held jump
rarely becomes a new press on the floor (existing native re-press applies only on airborne walls).
Wave5 one agent reached lower-left escape region after ledge; no finish. Max progress70->77%
is not controlled evidence because hazard sweep recognition rebuilt routes after wave3.
The final dynamic-special guard was tested after the live run;1-14 has zero special tiles,
so its tested geometry is unchanged. Detailed raw frames/actions now exist for this run.

## PPO 1-14 plateau diagnosis (proposal only) — 2026-11-24

ONLYPPO evidence: analysis/ppo_1_14_20261124/report.md, passive_sample.json, diagnosis.json;
capture.py and analyze.py reproduce collection/analysis. aplus_v2 snapshot:65 completed1-14
waves,0/2600 finishes,653 deaths,1947 timeouts; last10 waves12 deaths/388 timeouts.
Historical raw frames were not enabled. A separate passive25s observer found all40 characters
on the same ledge (x75..121,y35.86..36.01), above native goal(32,-132.72).
RewardPath.value() samples nearby free cell centers without checking solid geometry between
the character and candidate. On half-floor tiles, progress leaks from below the supporting floor.
Hazard-free reconstructed routes reproduce a local maximum atx110 matching the live cluster;
this is strong evidence, not an exact dump of live trainer routes or a live causal validation.
Minimum proposed fix: shape-aware local visibility for field candidates at RewardPath.value,
reusing existing tile-shape ownership/caching. Improves progress and PPO route lookahead together.
Do not increase waiting penalties, add level-specific actions or change model inputs. Continue
from aplus_v2 after correction, with raw telemetry in a separate realtest run. No eval required.
No runtime, weight or active trainer configuration change was made during this diagnosis.

## Training Monitor Full HD rendering — 2026-11-24

Both main and ONLYPPO monitor.py now draw a native1920x1080 canvas (previously1500x900),
with map width1100 and fonts18/16/23. Resized windows use smoothscale instead of nearest-neighbor
scale; aspect-preserving viewport and mouse-coordinate conversion remain shared. Native-size
presentation directly blits without filtering. No agent/training behavior changed.
Validation: headless1920x1080,1280x720,900x700 draws and slider mouse mapping passed;
Full HD preview inspected at .worktrees/onlyppo/analysis/monitor_fullhd_20261124.png.
ONLYPPO suite40 passed (one existing pygame/pkg_resources warning). Restart the trainer to load
updated monitor code; save the active run with Ctrl+C before resuming its model.pt.

## PPO 1-9 recovery implemented and live-tested — 2026-11-24

Supersedes the earlier historical-comparison recommendations below. Evidence and commands:
`analysis/ppo_recovery_20261124/report.md`, reproducible `summarize.py` / `results.json`.
Default native physics is now fixed 60Hz with owned-byte validation and shutdown restoration;
`--no-lock-step` is an explicit legacy comparison. Live cleanup confirmed original 7c53 restored.
Added frozen `--eval --load`, `--eval-waves` (20 default, 0 unlimited), `--eval-level`, policy
`--seed`, optional `--greedy`, `--record-states`, source/checkpoint hashes and native dt/times.
Evaluation never collects PPO samples, updates weights or writes checkpoint files.

Fixed two producer bugs: midlevel identity includes solid breakable ID100 cells; live tile
updates decode current IDs/flags instead of treating changed neighbor bits as disappeared blocks.
Old checkpoint widening now zeroes appended columns, preserving original actor/value outputs.
No extra waiting penalty, world model or action planner was added; all actions remain PPO.

Matched-source frozen comparison: old282 policy12/200 finishes (best native5.650019s), current524
ppo_aplus_v1 policy0/200. Corrected live tiles alone: current0/80. All observed dt16.6666675ms;
native game RNG is not seeded. Fixed timing alone did not repair learned weights. Original cause
is not fully isolated: historical pretraining/curriculum and timing differ.

Warm-started the archived successful policy in current524-input structure, then trained five PPO
updates in a separate run: 2,1,13,16,14 finishes =46/200 online outcomes; fastest native5.216680s.
This is not a frozen final-policy success-rate measurement. Latest trained model:
`runs/ppo_recovery_1_9_20261124_a/model.pt`; best successful checkpoint:
`runs/ppo_recovery_1_9_20261124_a/levels/1-9/best.pt` (both require --shape-inputs).
Separate frozen one-agent greedy test of the latter passed3/3 at19.249878 native seconds each,
not A+. No video or native map record was written in these tests. Original checkpoints unchanged;
all eval dirs contain zero .pt files. No test training remains running. Validation:40 tests passed.

From ONLYPPO, with 1-9 open (set SMBRL_IMPORTED to main/imported/smb_static if needed):
```powershell
python realgame.py --eval --eval-level 1-9 --eval-waves 0 --agents 1 --greedy --seed 0 --shape-inputs --wave-seconds 30 --end-after 1 --no-monitor --load runs/ppo_recovery_1_9_20261124_a/levels/1-9/best.pt
```
This plays without learning; record video with an external screen recorder. For further training,
use --load runs/ppo_recovery_1_9_20261124_a/model.pt with a new --name and --shape-inputs.

## PPO 1-9 historical/current comparison — 2026-11-24

Report in main worktree: analysis/ppo_1_9_compare_20261124/report.md; reproducible teacher-forced inputs
and policy_audit.py. No game run, model update or runtime edit. Available current checkpoint is
ONLYPPO ppo_aplus_v1/model.pt datedNov20; no full current1-9 recording/config exists. Historical d245429
is a committed proxy, not a proven exact training commit. Isolated archived source/data helpers stay
under main analysis/, not in ONLYPPO (no WM/planner added).

On335 known-success states with current geometry, archived282-input policy matches recorded actions83.0%
vs current524 policy27.5% (NOT live success rates). At first right wall: old left77.0%, current19.5%;
current jump99.96%. Old policy still responds well through current perception; main reward route points
are identical and reconstructed success return149.58061 vs149.57734. Zeroing extra242 shape features barely
changes current agreement. Base PPO class AST and existing reward constants match historical code exactly.
The stronger old collection has fixed16.667ms recorded dt; ONLYPPO preserves variable steps while usingHZ60
for rewards/input durations/velocity. Physics inconsistency is a strong candidate, not an isolated causal
proof. Old collector resumed a pretrained model,50 agents/all-agent waves/30s; current began fresh with40
and A+ +5 advancement. Actual old result:17/350 complete trials (4.86%),already3 successes in first wave.
New curriculum/initialization differs; quota/A+ advance cannot directly explain an unsolved1-9 where neither
fires. Current1200-frame budget also exceeds the335-action success. Increasing wait penalty is not justified.

Recommended next experiment: no-learning old/new checkpoint comparison under the same fixed native dt,
geometry and trial budget, then current full telemetry. No runtime change authorized/implemented in this
comparison. All17 archived traces are contiguous, start at native spawn, max frame move8.281 units and
finish within13.817 of native goal. Nov1 137-input/pre-lock result is less comparable than the282-input PPO.

## Archived PPO 1-9 successes found — 2026-11-24

Audit: analysis/ppo_1_9_archive_20261124/report.md (main worktree), inventory.json and successful_traces.json.
Older real_20261101-151650 has1-9 best.json4.75 frame/60 seconds, agent22,285 actions, best.pt; this
pre-timing-lock value is not verified wall-clock/A+. collect_2-10_ppo50_01 has1-9 best.json5.5833 seconds,
agent6,335 actions and best.pt. All335 actions exactly match wm_data/wave_00173.npz agent6 (local wave7;
saved best's wave166 uses a different counter). That collection has8 1-9 shards with17 distinct successful
PPO traces; every last status is2 and max endpoint distance to native goal is13.817 units. Full positions,
velocities, actions/rewards and sibling world JSON.gz are present. Metadata explicitly source:ppo,mode:train;
--wm-collect recorded experience, not WM/planner action selection. No model/run/game was changed.
Current ONLYPPO has no saved1-9 best. These archives support later comparison; no new penalty/fix was added.

## Replay continuation and gentle repeated-hop cost — 2026-11-20

Supersedes the earlier open realtest continuation issue below. Replaced fabricated XInput A pulses with
native result-screen Next Level callback VA0x4aa050, once after HUD13/60 frames onLeave. Uses mscdecl and
exceptions:propagate: Windows OutputDebugString's continuable debug exception was otherwise stolen by
Frida as a system error, aborting the fade/load. A live hot-corrected callback returned and loaded 1-2
after the separate test run earned A+ at wave3 and completed five further waves4..8; trainer received the
new level and saved/detached. No cold second final-source run was performed. Evidence and limits:
analysis/replay_hops_20261120/report.md, native_next_probe.json; analysis/realtest_progress_20261120/native_next_b.log.

1-9 live sample (no saved successful replay available): eleven of40 agents made2-4 takeoffs within4sec
while x-range stayed below40 units. Added physical repeated-hop cost only: first3 free, then0.03 each
from nearby launch positions when stalled, total cap0.6 per life/wave. Progress/different launch positions
reset local count; standing/waiting never triggers this cost. Existing standing/stall costs were not raised.
1-9 learned improvement needs further training/observation. Separate test runs preserved ppo_aplus_v1.
Validation: 33 tests passed (one installed pygame/pkg_resources warning); Node syntax, Python compilation
and diff checks passed. Tests cover the real trainer reward path and native callback/lifecycle contract.

## Native kill-shape perception — 2026-11-20

Prior ONLYPPO changes were committed/pushed as 3c0101b before this fix. The PPO observation path had
converted native circles, game boxes and laser segments to covering rectangles; monitor and reward routes
then consumed those rectangles. Main's observed-geometry producers and native contact formulas now live
in game/hazards.py (no model, future simulation or action planner). The current PPO pipeline retains shape
tags through initial info, frames, flying shots, hazard IDs/motion/sweeps, grid rasterization, sealed-pocket
classification and the existing reward-route generator. Monitor draws native circles and laser segments.
Circle tests retain the player's square body, so physically unsafe near-circle positions remain blocked;
empty native shape lists still mean inactive/harmless, and unmeasured older rows keep their box fallback.

The 282/524 observation layouts and Policy/PPO parameters, actions, optimizer and reward constants remain
compatible with saved models. Training now receives more accurate hazard cells; historical box-only golden
fixtures are retained as legacy compatibility checks, not a claim of current main/PPO parity. Main's PPO
still has the old box map; its world-model/planner already had the exact geometry used as this port's source.

Validation: 28 tests passed (one installed pygame/pkg_resources deprecation warning). New tests cover a
safe diagonal corner beside a native saw, unsafe body contact, sealing/reward-route consistency, actual
trainer observations and moving-saw/projectile type preservation, needle/beam contact, stable IDs/inactive
phases, and offscreen monitor circle pixels. Compilation/diff checks pass. This fix has not been tested
in a live 1-6 session. No running game, model, save or replay was modified for this perception change.
The earlier realtest result-screen continuation issue remains open; this task does not claim it fixed.

2026-11-20 — IN PROGRESS: realtest native transition and five extra waves.
The first A+ now starts a non-resetting five-completed-wave countdown, retaining the best native A+.
23 tests passed, including historical winner validation, duplicate waves, native finish ownership and
continue retries. Live run live_aplus_fix_a completed A+ wave1 + waves2..6 then loaded 1-2, but emitted
repeated terminal wave6 events during transition. Current code prevents that re-entry and retries continue.
Final live run live_aplus_fix_b reached A+ + five extra waves and native HUD13/main state22, but repeated
continue pulses have NOT loaded the next level. The final native fix is therefore NOT VERIFIED/COMPLETE.
Computer Use was stopped by the user's physical Escape key. Test B was asked to save/detach via STOP.
Original ppo_aplus_v1 checkpoint and native save were backed up under analysis/realtest_progress_20261120;
separate live test runs preserve the original checkpoint. Resume investigation at the native result-screen
input path; do not claim the current final version reliably advances. Logs: live_a.log and live_b.log.

2026-11-20: Fixed PPO realtest mid-level identity handoff. Latest local run real_20261120-165825
saved under levels/unknown; no requested native realtest result was present. Five finished agents is the
default wave quota, not proof of A+. identify_level previously ignored SMBRL_IMPORTED, and even a Python
resolved name was never sent back to native curName, so strict finish identity checks could reject it.
Identification now uses levels_smb.IMPORT_DIR; the initial realtest go reply conveys the observed-map
identity, adopted only while native name is unknown and wave is zero. Known level names stay protected.
22 tests passed, including configured-asset identification and native identity/finish handoff; Node syntax,
Python compilation and diff checks passed. Live save/advance remains unverified. No existing save,
checkpoint, run or running process was changed. Fresh command must include SMBRL_IMPORTED and --realtest;
resume with explicit --load runs/<name>/model.pt (a repeated --name alone does not restore weights).

2026-11-20: Training Monitor fonts increased from 14/12/18 to 16/14/20 (normal/small/heading).
Policy rows, chart labels and panel spacing increased accordingly; route summaries wrap within the panel.

2026-11-20: Training Monitor network diagram and its unused per-wave policy snapshots were removed.
The window is resizable using pygame RESIZABLE and a scaled canvas: map, charts and mouse controls scale together
with aspect ratio preserved. PPO actions/rewards/updates are unchanged.
Validation: main tile/monitor checks 18 passed; ONLYPPO suite 21 passed. Headless small/large-window
drawing and scaled slider clicks passed; Python compilation/diff checks passed. Desktop visual review
was not performed.

This branch runs the original PPO agents in the real game using `realgame.py`. Its current perception was
ported selectively from main. **There is no `wm/`, `mpc/`, `plan/`, `agent.py`, learned world model, action
planner, rollout engine, online learner or new A+ reward logic in this port.**

The in-game timer label shows `BEST <time> | WAVE <number>` (also before the first finish).
Wave follows the native wave counter and restarts for each level. Label buffers refresh only when
the wave changes. Live visual placement has not been verified.

## A+ automatic progression — 2026-11-20

`python realgame.py --agents 40 --shape-inputs --realtest` enables native A+ saving and progression.
Keep your existing `--load` and observation flags when resuming a policy. It cannot be combined with
scripted `--test`. The first native A+ starts a countdown of five additional completed waves; later A+
results can improve the saved time without restarting the countdown. Waves retain normal `--end-after`
behaviour. PPO processes each completed wave before the result is saved and native continuation runs.

The grade uses the native HUD clock captured at the successful goal contact, compared with the extracted
`chapters.json` par. Frame/60 remains the existing PPO reward/console metric, not the realtest grading
clock. Since 2026-11-24 the default native physics step is fixed60Hz. The native clock restarts each realtest wave. Unknown par/time
disables automatic saving/advancement instead of estimating a grade. The first automatic save requires a
backup of the running game's `UserData/savegame.dat` under ignored `backups/`; backup failure skips the
transition. The policy is checkpointed and requested results logged under the run's `realtest.jsonl`.
That log records requests, not independently verified native map records. The existing native finish and
result-screen continuation save the A+ time and load the next level. Level/wave/agent/time validation and
a duplicate guard prevent stale/repeated finish directives. Manual S retains its previous behaviour.

This separate worktree has no imported assets. Point it at the existing personal extraction before running:

```powershell
$env:SMBRL_IMPORTED='C:\Users\pahstoner\PycharmProjects\REA-MeatBoy\smbrl\imported\smb_static'
python realgame.py --agents 40 --shape-inputs --realtest
```

Validation: **21 tests passed**, including existing observation parity and PPO training, native clock
capture, validated between-wave handoff, actual forced-finish time writing in fake memory, duplicate/stale
requests, missing metadata/time/backup and flag compatibility. Node syntax and Python compilation pass.
AST comparison with branch HEAD confirms unchanged Policy/PPO, reward constants/actions, finish_reward,
on_wave_end and agent_fitness. No model/planner dependencies were added. **Native map saving/automatic
progression has not been live-tested**; tests use synthetic messages and fake memory, without accessing
or modifying a running game, its save, or existing models/replays.

## What was brought up to date

- Native `realgame.js` observations: actual animated kill shapes, crawlers/fliers, laser beam segments only
  while active, flying projectiles (including launcher kind 5 and measured missile shrapnel), and the camera's
  death boundary. Hazard lists retain stable per-shape identities. The script refreshes the boundary each wave.
- `game/level.py`: live tile changes, connected key/lock groups, fan/orb fields, moving platform positions and
  velocities, hazard-region conversion, death-boundary masks, and optional half-block/slope direction maps.
  This module contains observed-game geometry only. Main's imagined key/teleport paths, model feature metadata,
  ensemble path helpers and future laser prediction code were deliberately not copied.
- `realgame.py`: the current PPO observation constructor and hazard tracking. Outside-map observations mark
  death instead of a usable wall. Region growth counts as hazard motion. Projectiles reach the local hazard
  map and the nearest-moving-hazard inputs through the same path as main's PPO.
- `realpath.py`: the **pre-existing PPO reward-route generator** retains its role; shared tile decoding,
  camera boundary exclusion and portal traversal fixes were ported. Its class was already named `Planner` in
  ONLYPPO before this change. It supplies reward/direction routes, never selects agent actions. No new action
  planner was introduced; `PPO.act` still supplies every non-test action and PPO supplies every policy update.
- The monitor draws oriented half blocks and slopes using that same tile decoder.

## Running and model compatibility

From this branch's `smbrl` directory:

```powershell
python realgame.py --agents 40 --shape-inputs
python realgame.py --agents 40 --shape-inputs --load "C:\path\to\old_run\model.pt"
```

Without `--shape-inputs`, the existing 282-input interface remains. With it, two 11x11 direction maps are
appended: 524 inputs, matching main's PPO option. An older checkpoint retains its old columns and initializes
the added ones to zero, preserving initial actor/value outputs; further PPO training can use the new inputs. Widened checkpoints must be loaded
with the same flag; silently dropping their inputs is refused. Saved metadata records the observation size.

PPO optimizer/update code, action space and S behaviour are preserved. Since 2026-11-24, native timing
is fixed 60Hz by default; --no-lock-step restores legacy variable timing. The small repeated-hop cost
above is additive; existing reward constants retain their values.
Imported assets and previous checkpoints are ignored runtime data and are not copied into a fresh worktree.
Use your existing assets/run files when running it; the native game itself is never bundled.

## Scope of “same view”

Parity is with **main's PPO observations**, not with its world model/planner's extra inputs or predicted
futures. Raw laser phase/parameter and projectile velocity fields may be sent by the adapter, as on main;
PPO's existing input interface uses current hazard maps and position-derived nearest-hazard velocities.
Neither branch's PPO receives future laser trajectories or imagined states. Shrapnel is omitted until its
live memory layout has been measured, exactly as in the source adapter; no guessed layout was introduced.

## Validation and provenance — initial 2026-11-17 perception port

- `python -m pytest tests -q --tb=short`: **8 passed**. Tests compare observations byte-for-byte with main
  for six scenarios, both 282/524 layouts, including private keys, changing tiles/platforms, lasers, hazards
  and outside-map positions. Golden hashes and source hashes are in `tests/data/perception_parity.json`.
- Real trainer message handlers feed observations into PPO, complete trajectories, perform a PPO update,
  and save/reload checkpoints in both layouts. Tests block imports of `wm`, `mpc`, `plan` and `agent`.
- Node executes the native adapter using fake memory and checks enemy/laser on/off states, shots, the
  death-boundary helper and the emitted frame message. `node --check realgame.js` and Python compilation pass.
- `perception_port.json` records the source file hashes and excluded components. Main had concurrent
  uncommitted perception work; the port includes the observed-game portions present when it was made.
- AST comparison against the pre-port branch confirmed identical `Policy`, `PPO`, `finish_reward`,
  `on_wave_end`, `agent_fitness`, `ACTIONS` and all `R_*` reward constants.
- **Not live-tested against the running game.** Mock memory does not prove executable offsets or shrapnel
  calibration on a live session. No live game save, checkpoint or training data was modified during tests.

Main's unrelated uncommitted changes are preserved in the original worktree. This branch's worktree is
`smbrl/.worktrees/onlyppo`; its documentation is authoritative for PPO-only behaviour.
