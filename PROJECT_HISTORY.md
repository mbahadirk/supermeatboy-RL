# Super Meat Boy RL: project history

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

What was built, which problems came up and how each was solved, from the first look at the game files to
PPO agents playing as real Meat Boys inside the game. Newest work is at the end. Addresses are for the
Epic Games Store build of `SuperMeatBoy.exe` (32-bit, image base 0x400000).

## Where the project stands

- `realgame.js` is a Frida script that runs inside the game. Up to 200 real Meat Boy characters play a level
  at the same time ("agents"), in waves. Each frame it sends their state and the level's live state to
  Python and waits for their inputs (lockstep).
- `realgame.py` is the trainer: PPO (PyTorch, CPU) with a 282-input observation and 10 actions, rewards
  from reward routes, waves, saving, the monitor.
- `realpath.py` plans the reward routes on the live tile map (A* with a jump model) and measures progress
  along them (closeness fields).
- `monitor.py` is a pygame window: map, routes, agents, hazards, the leader's action probabilities, curves.

```bash
python smbrl/realgame.py --agents 100 --end-after 5 --load smbrl/runs/<run>
```

Stop with Ctrl+C or by creating `runs/<name>/STOP`; never kill the process (that crashed the game once).

---

## 1. Reverse engineering the game (before any RL)

| Problem | What was found / done |
|---|---|
| How does Meat Boy move? | Physics is data-driven: `CharProps/*.props` (11 floats). Real values: gravity 550, ground accel 13/frame, air accel 8, jump 300, wall jump 350 up / 300 out, wall slide gravity x0.6. Calibrated on ~51k live frames; a clone matched the game's jump arc exactly. |
| Where are the levels? | `gamedata.dat` is an archive (TOC decoded); 591 `.smbl` levels extracted. `tools/smbl_extract.py` decodes tiles (RLE layers, tile word bits: id, special bit 8, rotation, flip, shape 18-19), objects, movement paths, chapter order. |
| Live state | Frida hooks on the character update (`FUN_00520d30`): position +0x84/+0x88, velocity +0x9c/+0xa0, state +0x7b0, flags +0x7b4 (bit 1 airborne, 0x200 on a wall). The tile map object gives the collision layer; the collision world `[0x7035c4]` holds the level pieces (+0x124). |
| Which pieces kill? | The world's type table (row flags: 0x06 goal, bit 0 lethal, 0x0c bandage, 0x44 key). Hazard sizes come from the game itself (`FUN_004ca860`), times the piece scale, turned by its angle (+0x2c). |

## 2. A Python clone, then into the real game

- First a clean-room clone of the physics with SB3 PPO (`engine.py`, `env.py`, `train.py`). It solved
  levels like 1-1, 2-4 and 3-1, and a trained model even won 1-1 in the real game through a virtual XInput
  pad.
- **Decision:** the clone was dropped. Agents must be real Meat Boys in the real game, not a copy of it.
- **How:** the game keeps 40 spare Meat Boys per level for its replay. `realgame.js` takes them over
  (mode +0x9e0 = 2, our own), feeds each one inputs through the character's input handlers, and steps
  them inside the hidden main character's update.

| Problem | Solution |
|---|---|
| Agents teleported onto the goal after a PPO update | The game's time step includes the time it waited for Python. Long work now happens only at level info / wave end, and agents are held still for one frame after any long wait (> 80 ms). |
| Agents drawn where they died | The render position (+0x824) is copied from the position on wave reset. |
| Reward routes not visible | Routes are drawn into the game with Direct3D 9 at Present, using the camera matrix captured mid-render (0x4ac6a9). |
| Killing the trainer crashed the game | Safe stop through the STOP file and a Ctrl+C handler that finishes cleanly. |
| A second agent (Gemini) had edited the project | Its stagnation penalty outweighed the win reward and its draw hook broke drawing; both fixed. |

## 3. Reward routes and what agents can see

| Problem (reported in training) | Solution |
|---|---|
| Routes went through walls (3-2) | Half blocks are walls; slopes are open only through their open corner (shape decode checked: 1329 vs 4 tiles). |
| Routes went through saws and spikes | Hazard modes, first that works: hazards closed, moving sweeps crossable with timing, everything crossable at a high cost, squeeze through half tiles. A cell is closed when the body (12x12) would touch a hazard box there, so tight gaps the body fits through stay open. |
| Routes "flew" where the player must wall-jump | Jump model: after leaving a floor or wall a route may rise at most 5 tiles and move 12 sideways, then only fall. A cell beside a wall counts as a surface, so climbing by wall jumps is allowed. Higher jumps, then no limits, only if nothing fits. |
| Keys and bandages | Routes visit the keys one by one (nearest reachable first), then the goal; a separate route goes through the bandage. Bandage pickup stays disabled for agents so the save file's records are not touched. |
| One agent's key opened locks for everyone | Per-agent keys: the game script sets the lock tiles to each agent's own key state right before that agent's update and restores them afterwards. |
| Broken walls still drawn intact | Every frame the game script sends the tile words that changed; the map and monitor are live. |
| Agents blind to moving and thrown saws | Hazards are re-read every frame. The two nearest moving hazards (offset and velocity) are inputs. Shots in flight are read from the launcher array (+0x2c30, slots of 0xf8 bytes). |
| Hazard angle read as scale | Piece scale is at +0x24/+0x28 and angle at +0x2c. |
| Single route only | Up to 3 routes plus a bandage route; agents follow the route they are furthest along; a route nobody gets far on keeps fewer agents. |
| Planning took up to 52 s | Precomputed moves, plain lists, an expansion cap and a time budget: worst level about 8 s, all 220 chapter levels get a route. |
| Fans, wind and orbs (2-4) | Fan and orb arrays read from the world; the wind force is computed like the game (`FUN_004c40e0`). Inputs: wind here, in a wind column, nearest orb. Wind columns count as ground for routes. |
| Accelerator surfaces (3-6) | Tile id 98 is marked in the map (what it does was not traced). |
| Portals | Paired in piece order (`FUN_004c8c10`); routes go through them. Agent teleports seen 3 times also become links. |
| Walls and floors looked the same to the network | A second 11x11 map marks what each solid cell offers: floor +1, wall -1, ledge corner +0.5, ceiling -0.5, inside +0.25. Plus wall-left / wall-right / wall-slide inputs. |
| False portal links | Some deaths make the game put the agent back on the spawn; those jumps are counted as deaths, never as portals. |

## 4. 2026-11-02: more agents, a better reward, platforms

### 100 agents

- **Problem:** `--agents 100` always gave 40.
- **Cause 1:** extra characters were only built at a level load, and training always started mid-level.
  Fixed by building them when attaching too.
- **Cause 2:** building them still failed with "system error". The game's clone constructor writes a log
  line with `OutputDebugStringA`, which raises an exception that Windows normally handles inside the
  call. Frida's default for native calls catches every exception, including that one.
- **Fix:** the character factory is called with `exceptions: 'propagate'`.

### Reward by closeness instead of "stay on the line"

- **Problem:** agents hugged the route line. Progress only counted from visible route points, there was a
  penalty beyond 2 tiles, and another route only counted within 2.5 tiles of it.
- **Fix:** each route has a closeness field. A spot is worth "progress of a nearby route point minus the
  walking distance to it". Walls block it and climbing a tile counts double.
- **Exploit found right after:** standing in a corner near a far stretch of the route earned up to 11 tiles
  of progress. The distance cost was only half a tile per tile, and a "3 tiles past your best" window crept
  forward every frame.
- **Fix:** a tile of distance now costs a full tile, every solid tile is a wall, and the window was
  replaced by route legs. The route counts only up to the next key the agent has not touched. A spot's
  value is fixed, so waiting earns nothing (measured: 0 on every level).

### Getting stuck, jumping in place, exploring

- **Stuck:** no new best progress and no new 2x2-tile square for 1.5 s costs a little every frame,
  rising, together with jump spam capped at 4 per stuck stretch. The cap is below a death, so dying is
  never the cheaper way out.
- **Exploring:** the first visit in a life to a 2x2-tile square pays a bonus, more where few agents have
  been.
- **Inputs:** two new inputs say how long the agent has been stuck and how often it jumped there.
- **Tried and dropped:** killing an agent that stood still for 1 s made levels that need waiting
  impossible.
- **Death penalty:** raised from 5 to 15.

### Agents turning back for no reason

- **Cause:** progress only paid new best ground, so going back was free. And stepping 2 tiles away reset
  the stuck timer, which made walking back the cheapest way to wait for a saw.
- **Fix:** progress pays both ways every frame, so going back costs exactly what going forward paid. The
  stuck timer resets only on new progress or a new square. Fitness still uses the best progress.

### Route splitting

- **Problem:** a route counted as "different" if it was more than 1 cell from earlier ones. So 66% of all
  extra routes (189 of 286 on the 220 chapter levels) were copies 2 cells beside the first route.
- **Fix:** a route cell runs the same way as an earlier route when it is within 4 tiles of it with no wall
  between. The search is also pushed that far.
- **Result:** 136 extra routes, none of them copies, every level still has a route.

### Moving platforms

- **Problem:** agents could not see them.
- **Found in the game:** platforms are in their own world array (count +0x2dc4, data +0x2dd4, 0x14
  bytes: piece, three tile shapes, upright flag). Collision (around 0x4cde59) tests a 60x20 bar (20x60
  upright).
- **Agents now see:** platform cells in the map, floor in the surface map, and the nearest platform's
  offset and speed.
- **Routes:** a platform seen in place for 4 s is a wall. Where moving platforms go is crossed only with
  timing, so the planner looks for another way first. Spots above platforms count as ground.

### Lava pits

- **Problem:** pits are made of a few stacked lava pieces whose boxes leave gaps, so the map showed open
  space inside the pit.
- **Fix:** open cells that cannot be reached from the spawn, goal, keys or portals without crossing a
  wall or a fixed hazard are filled in, as hazard or solid. Pockets over 150 cells stay open.

### Wall climbing

- **Problem:** a wall jump needs a new jump press, and agents could not see whether their jump key was
  held.
- **Fix:** jump kept held for 6+ frames in the air against a wall is released for one frame and pressed
  again. A new "jump held" input shows the key state.

---

## Rewards now (per agent, per life)

| Term | Amount |
|---|---|
| Progress (closeness to routes) | +1 per tile, both ways |
| Key | +10 once per key |
| Exploring | +0.25 per new 2x2-tile square, less where many agents went |
| Running toward the route on the ground | up to +0.006 per frame |
| Jump from the ground | -0.08 |
| Stuck (no progress, no new square for 1.5 s) | -0.02 to -0.04 per frame |
| Jump spam while stuck | -0.15 per jump after the 2nd |
| Stuck cap | at most -4 per stuck stretch, for the two rows above |
| More than 6 tiles from every route | -0.003 per tile per frame |
| Death | -15 |
| Finish (before anyone finished) | +50 + up to 20 for speed, -0.002 per frame |
| Finish (speed phase) | +50 + 60 x best/own time, +25 for a record, -0.01 per frame |

## Observation (282 inputs)

| Inputs | Count |
|---|---|
| Position, velocity, ground, wall, facing, goal direction / distance | 10 |
| Local map 11x11 | 121 |
| Own route | 6 |
| Moving hazards | 8 |
| Walls | 3 |
| Wind / orbs | 6 |
| Surface map 11x11 | 121 |
| Stuck | 2 |
| Platform | 4 |
| Jump held | 1 |

Older models load: new input columns start small.

## Not verified in the running game yet

- The jump re-press on walls.
- Platform boxes lining up with the drawn platforms.
- What accelerator tiles (id 98) do.
- Whether the lava fill ever closes a real area (pockets over 150 cells stay open).
- Moving saws on tracks are assumed to update their piece position like other movers.

## 2026-11-17 — current PPO perception, still PPO only

The user requested that ONLYPPO see everything main's PPO can see, with no world model or action planner.
Ported live kill-region/enemy/laser/projectile observations and camera death boundaries from `realgame.js`,
plus their PPO geometry consumers into `game/level.py`. The trainer uses main's observation construction,
stable hazard identities and optional half-block/slope input maps. The monitor shares the corrected tile
decoder. The old reward-route generator remains, with its geometry/boundary/portal fixes; it does not choose
actions. PPO rewards, updates, action space, manual S saving and the branch's time-step behaviour are unchanged.

`--shape-inputs` enables the same 524-input layout as main; without it existing 282-input policies still load.
Older checkpoints can widen while keeping learned columns; wider checkpoints require the flag when resumed.
No `wm/`, `mpc/`, `plan/`, `agent.py`, online learner, future predictions or A+ training/realtest features were
added. See PROJECT_STATE.md for usage, feature boundaries and source provenance.

Validation: 8 focused tests passed, including main/PPO observation parity, real PPO updates from synthetic
game messages, checkpoint compatibility, blocked forbidden imports and native adapter execution in fake
memory. Syntax/compilation checks passed. Native live-game behaviour is not yet verified.

## 2026-11-20 — PPO-only --realtest

Added automatic native A+ saving/next-level progression to realgame.py. Grade uses captured native HUD
time and extracted chapter par, never frame/60. A+ bypasses the finish quota; PPO terminal-wave processing
precedes a checked finish directive in the go reply. Other finishes retry. Back up the native save once
per run before automatic recording; missing metadata/time/backup skips advancement. Existing PPO rewards,
updates, observations, action selection and manual S remain unchanged; no world model/action planner added.
See PROJECT_STATE.md for command, asset path and caveats. Validation: 21 tests, Node syntax/Python
compilation and AST parity passed; live native save/progression remains unverified.

## 2026-11-20 — wave beside BEST

ONLYPPO native HUD now displays `BEST <time> | WAVE <number>`, including before any finish.
Uses the existing native wave counter; buffers refresh once per wave rather than each draw.
Node syntax and existing fake-memory adapter checks pass; live visual placement unverified.
