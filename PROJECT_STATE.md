# Super Meat Boy RL: the state of the project (2026-11-03; planner step 4 added 2026-11-08; 7.9 added 2026-11-10; 7.10 wave search 2026-11-11)

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

Details refer to the ONLYPPO worktree; main runtime was not changed.

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

## ONLYPPO 1-9 recovery — 2026-11-24

Implementation and live evidence are in `.worktrees/onlyppo`; main runtime was not changed.
See `.worktrees/onlyppo/analysis/ppo_recovery_20261124/report.md` and its shared project notes.
Fixed native60Hz timing with safe restoration, frozen evaluation/provenance/telemetry, native tile
identity/live occupancy decoding, and zero initialization of widened checkpoint inputs.
Matched-source frozen old/current comparison:12/200 vs0/200; corrected tiles alone current0/80.
Warm-starting old skills into current524-input PPO and training five updates produced46/200 online
finishes (not a frozen final success rate), best native5.216680s. Recovered model is
`.worktrees/onlyppo/runs/ppo_recovery_1_9_20261124_a/model.pt`; successful best is
`levels/1-9/best.pt` under that run. Both need --shape-inputs. Separate frozen single greedy agent
passed3/3 in19.249878s each (not A+). Original checkpoints preserved; no native map record/video
written and no training left running. Native timing restored to original7c53 after tests.
40 tests passed. No WM/action planner added to ONLYPPO. Original regression cause remains partly
unisolated; timing/map fixes alone do not repair the existing learned stuck policy.

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

2026-11-20: ONLYPPO replay fix now uses native Next Level callback (mscdecl, exceptions:propagate);
Windows debug-print exceptions had interrupted direct calls. Corrected callback loaded1-2 live after
A+ plus5 extra waves; test agent saved/detached. 1-9 live loop analysis led to event-only repeated-hop
cost0.03 after3 free local launches, cap0.6/life; standing/waiting incurs no added cost. 33 tests passed.
See .worktrees/onlyppo/analysis/replay_hops_20261120/report.md and branch PROJECT_STATE.md for evidence
and limits. Original PPO run preserved. Main runtime code unchanged.

2026-11-20: ONLYPPO previous features pushed at 3c0101b. A subsequent local perception fix in
`.worktrees/onlyppo` preserves native circle/game-box/laser tags through PPO grids, sealing, existing
reward routes and monitor drawing (main observed-geometry formulas, no WM/action planner). 28 tests
passed; live 1-6 unverified. Branch PROJECT_STATE.md is authoritative. Main source remains untouched.

2026-11-20: ONLYPPO realtest mid-level identity fix is in `.worktrees/onlyppo`: configured imported
assets now drive level identification; initial go reply synchronizes resolved name to native. 22 tests
passed; live progression unverified. See branch PROJECT_STATE.md for evidence and fresh/resume usage.

2026-11-20: Training Monitor fonts increased from 14/12/18 to 16/14/20 (normal/small/heading).
Policy rows, chart labels and panel spacing increased accordingly; route summaries wrap within the panel.

2026-11-20: Training Monitor network diagram and its unused per-wave policy snapshots were removed.
The window is resizable using pygame RESIZABLE and a scaled canvas: map, charts and mouse controls scale together
with aspect ratio preserved. PPO actions/rewards/updates are unchanged.
Validation: main tile/monitor checks 18 passed; ONLYPPO suite 21 passed. Headless small/large-window
drawing and scaled slider clicks passed; Python compilation/diff checks passed. Desktop visual review
was not performed.

2026-11-20: Main native HUD now shows `BEST <time> | WAVE <number>`, including before the first finish,
matching ONLYPPO. Uses the existing native wave counter (reset per level); label buffers refresh only
on wave changes. Node syntax check passed; live visual placement remains unverified.

## ONLYPPO --realtest (2026-11-20)

Implemented in the separate `.worktrees/onlyppo` worktree (ONLYPPO branch), using `realgame.py --realtest`.
Every native A+ triggers save/next-level after PPO wave processing; other results retry. Native HUD time
and chapter par determine grade, with native-save backup and checked go/finish handoff. No world model,
action planner or reward change. Branch PROJECT_STATE.md/PROJECT_HISTORY.md contain usage and limits.
Validation: 21 tests plus syntax/compilation and PPO/reward AST parity; live native progression unverified.
Main's existing source/run/model changes are preserved; no commit/push performed for this addition.

## Ready proof-boundary handoff and future support context (2026-11-19; implemented, native speed unverified)

User authorized fixes after online_check21 audit. Runner.route_triggers now routes ready replacements FIRST,
then re-reads the CURRENT committed plan and enforces its proof deadline before any action. Ready/state-valid
boundary handoff succeeds; unready/invalid/expired results still stop/replan. Mandatory event priority,
late-state/speed checks, prefix reconciliation, private task intent and safety limits remain. Single-worker
react and pool/legacy events paths share this final guard. Do not restore a pre-handoff expiry check.

Plan.support_platforms is an owned per-native-frame copy of the decision future's platform rows, anchored
at plan.frame. Both normal/explore producers include it when platforms exist; static/legacy default None.
Committer recomputes predicted support using those future positions and the existing private incoming-step
key masks. It evaluates the translated path after adoption while leaving world platform coordinates fixed;
late adoption does not restart/shift the forecast anchor. Live actual support checks remain live. This is
context transport, not stale support-label caching. No checkpoint/feature/CLI protocol or threshold change.

Before/after state-matched fixture at frame112/old-proof-end112: cancel2/no commit -> commit1/no cancellation.
Two recorded native A+ paths, unchanged controls and actual recorded future platform context: native followed
frames1324 each; support triggers59->20 (wave27),25->12 (147).55 old triggers removed;3 new mismatches remain
visible, total84->32. These are replay/fixture checks, NOT a new native speed/success/A+ measurement.
Existing4-2 boarding/overshoot remains a separate behavioral task; not claimed solved by this update.

No game/agent/run/save/model/memory/ONLYPPO writes. Restart needed. Before sources/evidence/limits and prepared
online_check22 command: analysis/handoff_support_fix_20261119/report.md, comparison.json, command.ps1, manifest.json.
Final full regression: **702 passed, one existing pygame/pkg_resources warning,363.36s**. Focused checks
and replay probes overlap this coverage; do not add their counts. Python compilation, scoped diff checks
and clean ONLYPPO status passed. Performance/native A+ outcomes after this patch remain unmeasured.

## online_check21 replay audit (2026-11-19; analysis only, two confirmed scheduling/support bugs)

Bound844 completed native lives/events:4-18 first149,4-1 first6,4-2 first600,4-3 first13,4-4 first76.
Logs continued growing; later lives/cuts excluded. Native4-18 waves27/147 BOTH A+22.084s, wall84.15/87.10s,
speed0.262/0.254x vs preserved old successful0.147x. Success retained, speed still unacceptable. Whole-level
game/wall ratios:4-18 .293,4-1 .611,4-2 .497,4-3 .567,4-4 .475. Different levels are not matched benchmarks.
4-3 finishes8/13, one A+10.95s.4-2 dies600/600 (548 h47#3, median59f);4-4 dies76/76.4-1 no finish/6.

Confirmed regression in the prior performance edit: route_triggers expires the old proof BEFORE polling/
taking a ready replacement, while ahead_start targets that exact boundary. Deterministic ready-at112 /
old-proof-end112 reproduction cancels both jobs with zero commits. Native A+wave27 logs29 expiry incidents/
58 cancelled jobs,29 primary targets at the expiry. Readiness is not logged, so don't claim all were ready.
Next fix must allow a validated ready handoff first, then stop/replan if none; keep the safety deadline.

Second confirmed source/recording issue: Committer.decided computes predicted support with live lv.plats
for the entire future path. Planner explanation uses future bars correctly. On the two successful replays,
84 reconstructed support mismatches become32 using actual future bar positions;55 old mismatches resolve
(3 new mismatches appear). Example plan120/frame125 position error~0.05, yet old stationary-bar expectation
says air and cancels for landed early; time-aligned bar says ground, matching native pose. Preserve support
events; carry producer-owned future support through async adoption instead of widening tubes/silencing checks.

Parallelism is active:4-18 speculative starts6557/uses1277/cancels5191; support drops3740, proof drops1976;
small station only87/6231 decisions.4-2 samplewave596 outruns the boarding bar, continues falling and last
reflex at52 already predicts death+8; native death59. Contact/landing-width details still need measurement,
do not blindly widen on_bar. Prioritize handoff/support fixes, then boarding/speed matching and4-4 timing.
No production/game/agent/run/save/model/memory/ONLYPPO changes. Evidence:
analysis/replay_check21_20261119/report.md, summary.json, support_audit.json, handoff_probe.json, manifest.json.

## online_check20 / first native A+ and performance implementation (2026-11-19)

Baseline source/tests/docs committed as f5ef6eb and pushed to origin/main at the user's request BEFORE
performance edits. Local analyses, checkpoints, run data, personal timing/death/move JSON and ONLYPPO were
excluded. The run confirms wave24 native finish f1340 /1341 frames /22.35s, A+ against24.5s, reward15.
Wall time151.84s/game speed0.147x,205 decisions, decision median529.93ms. Sync waits42.269s plus ahead
waits42.919s account for85.188s. IMPORTANT: check_ms_per_frame53.233 includes route_triggers/ahead_take
blocking; it is nested in planning87.86ms/frame, not a separate pure geometry cost to add to it.

Performance changes preserve safety thresholds, ensemble size, horizon, async wait modes and online fitting:
- game.level owns bounded64-entry profile tables keyed by all rounded dimensions/angles in box order.
  Unique ray tables plus box indices replace repeated future-frame ray-table expansion; reach/paired/unpaired
  clearance consume the same representation. Translation/growth are applied separately and never cached away.
- Terrain batches the six foot/body/wall probes and evaluates partial tile shapes only where needed.
  Planner.support batches future platforms by private mask, retaining the inside-foot rule, per-frame geometry,
  platform-count changes and private key terrain. No independent downstream ground repair.
- _BoxRun.period keeps the original per-frame search/state schedule and period candidates; only endpoint-
  matching offsets need spread accumulation, and only their required prefix. Do not skip periodic searches
  merely because current future frames are recorded; later period discovery depends on that history.
- Transport-obstruction trigger drops shapes whose full kill reach cannot intersect the transport rectangle;
  candidate crossing safety still uses all hazards. Shared CARRIER_MOVES owns both station proposal lists.
- Confirmed attached riders with no crossing/pending return, obstacle trigger, passage or danger event may
  try a7-move station search (<=2 firsts,4 branches,4 finalists). Disabled on key/portal maps, with a proven
  route, or T0<=2H/at goal. It is adopted ONLY when the existing all-member16f attached/collision-free station
  proof succeeds, with <=8f feedback. Otherwise it reruns the full search BEFORE adoption, so speculative
  small-search memories/controls cannot replace the full result. Shared library/search views are local;
  no mutation of Planner.lib, checkpoint or process-global search settings. Full obstacle search remains.
- The recorded --planners2/auto run actually used the single-job events scheduler after platform detection
  (zero speculative starts). Runner.pool_ahead now keeps both workers active in auto's events phase too.
  Timed jobs need not block at their target; danger/start/surprise/support still cancel/wait as before.
  Existing late-state/speed/action-prefix/stale-prev checks and speculative reconciliation remain. This is
  NOT forcing react on a moving map. CLI still accepts the same --async-wait auto --planners2 command.
- Plan.execution_limit owns the execution proof deadline. A late worker cannot extend a station/recovery/
  ascent tail; the runner stops for a new decision at that deadline, rejects already expired incoming plans
  before commit, and clamps primary lead to the deadline. Secondary speculation may assume later controls
  but cannot execute them without reconciliation and a newly valid adopted proof. No checkpoint format change.

The station optimization can change unexecuted tails/steering choices; it does not prove native completion
or real-time speed preservation. Four initial exact-optimization cases retained identical entire decisions;
after station search, those four retained identical first8 actions (one used the small search).64 recorded
danger-check answers retained identical contact/id/risk. Final controlled timings/tests are recorded in
analysis/perf_check20_20261119/report.md. Do not claim1x from offline benchmarks. Restart required; no new CLI
flag. No live game/agent was started/stopped, no runs/saves/models/memories/ONLYPPO were modified.

Isolated6-state/two-repeat comparison against f5ef6eb: station medians613.46->247.53ms and551.76->198.48ms;
four full-search decisions remain identical, with mixed timings (some slower), aggregate decision median
614.58->606.13ms. All12 first8 actions identical. Replay-plus-decisions wall43.589->31.522s, CPU51.297->36.094s,
peak working set478,756,864->460,279,808 bytes.64 profiled native-path checks: median9.2114->4.32015ms, identical
answers. These do NOT measure the later pool scheduler benefit or native FPS/A+ preservation. Final full
regression: **692 passed, one existing pygame/pkg_resources warning,262.38s**. Earlier pre-scheduler685
passed/501.73s and final focused47/32.82s are overlapping validation, not additional distinct tests. Compilation,
git diff --check and clean ONLYPPO checks passed. Performance changes remain local after the requested baseline
push. Reviewable next command: analysis/perf_check20_20261119/command.ps1 (latest online_check20 checkpoint,
fresh online_check21, same auto/2-worker/online settings,30s wave cap vs unchanged24.5s A+ par). Not executed.

## First reported 4-18 finish and performance follow-up (2026-11-19)

User reports the first successful 4-18 completion after anticipatory crossing, but observed speed around
0.1x is unacceptable. Latest run is online_check20. Preserve this implementation in git before performance
work. Earlier model-only failures below remain historical evidence; they are not the latest native outcome.
No new native finish/A+ rate has yet been measured by this follow-up. Next: bound completed run evidence,
profile decision timing, reduce repeated work without dropping crossing or weakening safety.

## Anticipatory carrier crossing (2026-11-19; implemented, full obstacle/section still unproved)

User-approved follow-up to online_check19: search a geometry-derived far-side obstacle goal BEFORE safe
16f station ranking can suppress launch preparation. Static transport-corridor lethal shapes trigger it;
projectiles remain in safety checks but do not create stationary crossing goals. Existing future horizon,
144 jump/forward/countersteer proposals OR48 wall-climb/exit proposals, one-member screen then <=16 plus
incumbent all-member checks. Solid terrain above the obstacle selects climbing; no level/obstacle hardcode,
new argument/checkpoint format or collision-threshold relaxation. Goal/proven route priority remains.

Full crossing needs common settled far-side support with every member alive to arrival. A checked20-unit
supported ascent is separately labelled climb-approach/full_crossing_proved:false, never a completed crossing.
Only checked crossing/ascent can override station. Cached controls consume elapsed time; feedback <=8f AND
proved endpoint. Observed far-side intent survives near-side reattachment/control discard; actual ascent
support clears controls but retains the goal. Two consecutive actual far-side support observations end it.
Async adopted effects retain the existing synchronization/reset semantics. Implementation: plan/carrier_cross.py,
plan/carrier.py and plan/planner.py; regressions: tests/test_carrier_cross.py.

Important correction to prior geometry interpretation: solid pillar x60..140/y-240..-120 blocks the apparent
gap above the lower circle. Simple high/forward hop is insufficient. Eight matched actual-past decisions
choose early checked ascent in three cases, but prove NO complete crossing. Coarse model feedback launches
atf943 and reaches y~-207 atf959, then next supported ascent fails all-member validation (some fail support,
another hits upper h24 before slower arrival); ordinary safety takes over. It STILL does not clear the
obstacle/finish4-18. Never mark this section solved or claim A+/native survival. Development median decision
769.0->1065.6ms in the eight cases; uncontrolled timings, no live FPS/speed improvement claim. This additional
bounded search adds work. Full regression679 passed/one existing pygame warning in518.71s; final-source
focused carrier/crossing suite17 passed in10.37s. The broader focused suite113 passed in41.77s.
Compilation, git diff --check and clean ONLYPPO status checks passed. The full suite preceded only the
final diagnostic reason wording; final focused checks imported that wording too.

No native game/agent/run/save/published-model/memory/ONLYPPO writes. Local checkpoint copy/evidence only.
Restart required, same command/checkpoint format. Details and remaining failure:
analysis/carrier_crossing_20261119/report.md, comparison.json, sequence.json, ascent_audit.json, manifest.json.

## online_check19 / obstacle crossing blocked by carrier priority (2026-11-19; analysis only)

Bound first43 completed lives/shards0..42:43 deaths/no finish, maxx70.813 vs goal564.73. New carrier active
in2122/3189 decisions. Three deaths hit the same stationary lower circle h23 near(98.94,-233.76), radius27.73,
while body rides at y~-254.3. Lives2/22/43 die f990/994/987 nearx68..71. Carrier passes through its kill region;
a rising/forward crossing and time-aligned far-side support is required. Upper circle h24 constrains height.
56 station decisions across six lives log negative future h23 clearance, often+40..48f, yet approve16f
attached prefixes. Later8f recovery/reflex begins after launch preparation time is exhausted. Causes are
short station prefix/rank_soft0 priority, extended search only when eligibility is gone, narrow short-upward
carrier templates and position-ranked survivors, no explicit far-side crossing task. User observation is
supported: new control over-prioritizes attachment before an obstacle. Next: early transport-corridor hazard
trigger, geometry-derived longer/directional/countersteer proposals, common far-side exit on future carrier
or validated next support, persistent crossing until actual arrival. Do NOT just drop carrier tracking or
force a jump. No production/live/run/save/model/memory/ONLYPPO changes. Source/phase/death evidence:
`analysis/carrier_check19_20261119/report.md`.

## online_check18 / observed carrier task (2026-11-18; implemented, live finish unverified)

New plan/carrier.py retains carrier identity and returning intent across replans/pocket contacts;
discarding unsafe controls does not erase the return objective. Actual settled arrival/death/keys/count/
wave/level/field resets govern lifetime; adopted controls use an unchanged absolute return deadline and
consume their remaining actions. Old async adoptions cannot revive completed returns/cancel newer ones.
Station keeping uses all-member attached/collision-free 16f prefix, <=8f feedback, relative-centre steering.
Evasions need all-member same-moving-support arrival; bounded extension uses existing H+completion future.
Unproved short recovery is explicitly logged, safe for8f/feedback4f, approaching from above; no safe proof
means ordinary safety fallback. Real goal/proven route/checked walking transfer keep priority. Non-jumping
confirmed rider rest noise cannot invent a takeoff; walk-off/jump semantics remain. New mode/color carrier
(blue) and counts.carrier diagnostics. No checkpoint/feature/CLI format changes; ONLYPPO untouched.

Matched recorded decisions improve unnecessary late jumps. Model-feedback probes STILL contain difficult
projectile/fallback cases and do not prove native survival or completion. Do not mark 4-18 solved. Full
regression672 passed/one existing warning (323.52s); after late intent/count handling, final focused suite
148 passed/one existing warning (60.91s).
No live agent/game/run/save/model/memory writes. Report/limits: analysis/carrier_20261118/report.md.
Proposed next command: learned runs/online_check18/online/wm_model.pt, new run online_check19, same replay/
async settings, --wave-seconds30 (A+ par24.5 unchanged). Restart required; command.ps1 is reviewable.

## online_check18 / long-run outcome (2026-11-18; analysis only)

Bound first 268 completed lives, shards 0..267: 268 death events/no finish (267 wave-end done, one time at
1200f also has h14 death). Stable contacts: 153 board, 76 descend, 10 reach bottom segment. Max actual x65.26
vs goal564.73. Contact +4f median errors horizontal0.144/descent0.463/bottom0.413 (3136/1607/96 overlapping
followed windows). Old false-ground diagnostic 0/2301 ground completions. Prior physics/surface errors are
not the dominant remaining issue: many jumps release support, enter pockets near x30..60, and future
replans replace the predicted return while the carrier departs. Lower segment has 34 departure events,
27 with positive takeoff; life155 is a detailed example. Deaths: upper shelf151, lower hazards66, shots48,
unknown3. Need per-agent moving-support identity/relative-position task, maintained through bends and
evasive-jump return/verified transfer; ground endpoint alone does not prove useful route continuity.
Online learner trains world dynamics, not that control policy. Separately wave cap is20s, imported4-18
A+ par24.5s; natural carrier route may exceed the cap. Diagnostic suggestion --wave-seconds30 does not
change A+ target. Logged median game_speed0.261, median decision255.325ms: still not real-time.
No production/live/run/save/model/memory/ONLYPPO change. `analysis/platform_check18_20261118/report.md`.

## online_check17 / standable surfaces and descending carry (2026-11-18; implemented)

Terrain.support now requires a supporting foot to have solid below and free space at body-centre height,
for both tiles and platform boxes. Tolerated vertical-wall penetration no longer counts as ground; wall
side/jump ability remains unchanged. This shared owner fixes completion/modes/passage/real support without
a separate downstream scoring patch. All 286 audited wall-like endpoints lose false ground; all 6578
observed stable bar contacts retain ground. Of those old endpoints, 285 remain wall, one deeply embedded
point becomes unsupported. Real floor/wall corners and half-block/slope tests remain valid.

New measured issue at the right-to-down bend: descending top riders move with the bar while native vy=0.
carry_bars now replaces learned vertical carry once for continuing descending non-upright riders, keeping
the observed contact offset; takeoff/new landing/rising/upright behavior remains. Same frozen model and
42 recorded descending +4f windows: median endpoint error 5.4633 -> 0.6965. Eight matched planner decisions
exercise the bend. Model-feedback sequence from actual f248 approach through f408: sampled ground 4/21 ->
17/21; corrected sequence follows descent. This is NOT live survival/exit/goal/A+ proof or section completion.
Restart required; same command/checkpoint format. No live/run/save/model/memory/ONLYPPO changes.
Focused checks 75 passed, final descending/surface/planner checks 57 passed. Two old golden completion
labels updated ground->wall; all other old recorded fields unchanged. Final full suite: **662 passed,
1 existing warning in 382.97s**. Exact full checks/limits/evidence:
`analysis/platform_surface_20261118/report.md`.

## online_check17 / 4-18 actual-play audit (2026-11-18; analysis only)

Folder has two starts. Bound second session's 89 completed lives to shards 19..107 (runner wave from shard
metadata, NOT global row wave+1): 89 deaths/no finish. New baseline logs present in all 3681 decisions;
departure guard active in 3592, absent only at initial spawn. 52 lives board, max continuous stable contact
178f vs prior recording's 25f. Native followed-platform +4f error: median 0.2392/p90 1.3853 units (1177
overlapping matched-control windows). Horizontal carry is improved; completion remains unsuccessful.
Deaths: upper shelf h9/10/11 41; lower h13/14 25; shots 17; other/unknown 6. Most boarding lives run/jump
ahead of the bar, reach x~-46 wall, then fall or get shot. Bar turns DOWN near f301 at x=-267.8; only four
lives have stable contact at/after f300. Vertical riding is still model/guard driven, not explicit carry.
Found a remaining landing loophole: tolerated wall penetration puts a foot's below AND centre-height probe
inside the wall, but support calls it ground. 286 chosen ground-completion records across 59 lives match
this condition. Example life4 f272 predicts ground (-45.33,142.12), risk0, although it is the vertical face.
Next correction should validate actual standable top surfaces at the shared support/landing boundary,
then verify the descending platform/transfer; don't just increase risk or extend guard lifetime again.
Snapshots, native death examples, plot and limits: `analysis/platform_418_check17_20261118/report.md`.
No production/live/run/save/model/memory/ONLYPPO change in this audit; later growing-run lives excluded.

## online_check16 / 4-18 carry and transfer correction (2026-11-18; implemented)

User-approved follow-up replaces learned horizontal displacement for confirmed continuing riders with
integrated relative native vx + platform displacement, BEFORE tile collision. It does not add carry
twice or alter native vx. Carry stops when walking off within a coarse step; takeoffs/new landings,
non-riders, upright/vertical-moving bars retain learned motion. Small supported vertical drift is reset.
Per-agent departure history is owned by real observe(), retained during airborne replanning/hops, refreshed
at supported landings, cleared on death/wave/level/field/private-key objective changes. Speculative plans
and async adoption cannot replace it. Logs include platform_launch.baseline_y/baseline_t; no new flag/model.

Same frozen final online_check16 model, 42 overlapping actual-control platform +4f windows: median endpoint
error 6.1501 -> 0.6578, p90 7.0036 -> 1.2992 units. Six matched recorded decisions retain previously lost
airborne context. Model-feedback riding probe from an already-boarded native state has 10/12 grounded
samples vs 8/12 before; this is NOT native play or successful boarding/exit/finish proof. New live run
required; restart with the same command. Linear within-step velocity integration is an approximation.
Focused platform/passage tests 37 passed; async/commit tests 47 passed. Final full suite: **658 passed,
1 existing warning in 243.30s**. Exact limits/hashes: `analysis/platform_418_fix_20261118/`.
No game/agent/run/save/model/memory/ONLYPPO writes.

## online_check16 / 4-18 follow-up (2026-11-18; live failure confirmed, analysis only)

Previous platform-launch heuristic did NOT solve real play: latest recording has 92 completed deaths,
zero finishes; launch context is present in 612/1599 decisions. It is decision-local and switches off
after an upward hop, allowing immediate forward launches on the next airborne replan.
Ten lives contact the bar; longest stable contact is only 25 frames. Native carry is now measured:
68 stable transitions have median dx - next_vx*dt = 1.8161 units/frame, matching bar displacement.
Eight matched/followed +4f platform predictions underpredict horizontal movement by median 6.9249 units.
Native vx can be zero while the platform carries the player. collide_bars has no explicit carry; the
learned predictions here do not cover it adequately. Do not call the previous isolated-state tests a
platform fix. Next scoped work: support-relative carry without double counting learned motion, maintain
landing/boarding context through airborne replans, then validate wait/board/ride/transfer sequences.
No new production change or live action this follow-up. Evidence: `analysis/platform_418_recheck_20261118/`.

## online_check15 / 4-18 moving-platform launch (2026-11-18; implemented, offline verified)

Latest completed 4-18 visit, shards 247..288: 42 deaths/no finish. Agent leaves the spawn ledge before
the moving bar arrives. Terminal scoring credits distant predicted wall touches as supported progress,
even without a demonstrated landing. Grounded decisions in moving-bar worlds now cap airborne/wall
terminal progress at starting T; predicted ground/platform landing restores ordinary credit. All terminal
scoring consumers share this rule. Neutral wait and opposite walking 8/16f remain in the final candidates
so early pruning cannot discard all grounded alternatives. Collision geometry, wall-jump ability, risk
coefficients and actual-goal/verified-route priority are unchanged. Logs: counts.platform_launch.

Seven matched recorded decisions (six duplicated grounded poses plus one airborne control), with both
the saved online model and the command's starting model: early forward launches become waiting/braking
choices; airborne control unchanged. Synthetic stationary-ledge probes with elapsed stall time wait while
the bar is distant and choose movement at later bar phases; these are not a real boarding demonstration.
This heuristic can defer a legitimate wall route whose landing exceeds the horizon. Real platform riding,
finish/A+ and FPS remain unverified. No training or game/agent/run/save/memory/ONLYPPO writes.
Full suite: **653 passed, 1 existing warning in 275.77s**; focused checks: **70 passed in 29.95s**.
Report/reproduction/hashes: `analysis/platform_418_20261118/`. Restart the agent to load the change.

## online_check15 / same-exit crossing controller (2026-11-18; implemented)

Six completed 4-17 lives in shards 58..63: 4 timeouts, 2 deaths, no finish. Historical improvement was still
discounting each new wait (540 stalled frames charged as only 26.7), while incomplete waits avoided pricing
the crossing they would eventually need. Approved correction is implemented: prefix credit now uses its own
current state; real observe() owns stall history and an older async adoption cannot rewind it.

New local `passage` mode compares direct and explicitly delayed candidates to ONE supported exit, derived
from the ordinary progress/move tree. Every member must reach it with required private keys and without
predicted contact until the slowest arrival; incomplete waits cannot win that comparison. Existing risk
coefficients, margins and collision geometry are unchanged. No proved exit means ordinary-planner fallback,
not a forced hazardous move. Actual goal/proven-route/trial priority is retained.
Adopted controls, including any selected wait, are consumed and revalidated from each new state. Real/model
arrival uses one shared supported/keyed exit predicate. Crossing feedback is capped at 8f (current model
step 4); objective/level/wave resets clear the cache. It is cyan in game/viewer and logged in Plan.passage
and counts.passage. No new argument/checkpoint format; restart the agent.

Matched offline sampling of 87 upper-bottleneck states across all six lives, same saved online_check15
checkpoint: waits 37 -> 11, 61 selected complete crossings, no selected veto. Separately checked the exact
human_v4p_gpu starting model from the user's command: 5 matched cases, 2 baseline waits -> no initial waits,
all 5 corrected decisions use proved crossings. These are recorded-state decisions, not new real finishes,
death-rate/A+ measurements or a real-time claim. Initial crossing search adds cost; cached revalidation is
smaller, and existing normal search still runs. Full measurements/limits: `analysis/passage_20261118/report.md`.

Full regression: **650 passed, 1 warning in 199.64s** outside the network-restricted sandbox. Final focused
crossing/prefix/worker/key checks after the shared arrival predicate: **63 passed in 8.24s**. Compilation and
whitespace checks passed; warning is the existing pygame/pkg_resources deprecation. No game/agent was
started/stopped/attached, no run/save/model/memory file was written, no commit/push, and ONLYPPO is untouched.

## online_check14 / remaining stalls and paths after 4-9 (2026-11-18)

Seven bounded completed 4-17 lives (shards 0..6) all time out at 1200f. The prior prefix fix is loaded,
but binary 10-T-frame credit charges short movement and waiting identically; the 20-unit cell lookup
also hides small movement. Fixed with proportional executable-prefix stall credit and a subcell time
lookup shared by observed stall history/prefix scoring. Main estimates, collision/risk rules and the
3f hysteresis remain unchanged. With the command's human_v4p_gpu starting checkpoint, recorded f604/895
now choose run R 8f instead of wait 8f; f331 is an unchanged control. No live/A+ claim.

In-game paths working before 4-9 but disappearing later were traced with a bounded read-only render probe:
the old 0x4ac6a9 scene hook was skipped while Present/shader constants ran. Camera capture now observes
the common D3D SetVertexShaderConstantF boundary; first world camera wins until Present, with lazy hook
installation/replacement. Overlay milestones and draw errors are persisted in events.jsonl.

Full suite: 639 passed and one obsolete binary-credit expectation failed; after updating that expectation
and final render-boundary change, all 30 targeted regressions pass. Native overlay/laser fixtures pass;
compilation/whitespace checks pass. Report, exact validation and limitations:
`analysis/saw_recheck_20261118/report.md`. Running processes need a restart. No agent/run/save/memory was
modified; the render probe only observed and detached its own instrumentation. ONLYPPO remains untouched.

## online_check13 / 4-17 saw hesitation (2026-11-17; prefix-credit fix implemented)

Bounded to completed shards through 115: two visits, 59 lives = 33 finishes / 15 deaths / 11 timeouts,
zero A+. Best real finish is 752 frames / 12.534 s against par 12 s (32 frames to save).
The upper bottlenecks around (-446,186) and (-385,186) show repeated waiting. Their h18/h19 saw kill
circles are static in the best life. Generic 13.9-unit soft margins and large death-scaled near-pass costs
outweigh movement until accumulated stall penalties win. Best-life frames 242/250/258 repeatedly choose
`wait 16f, walk R 8f` eight frames apart at the same position; predicted endpoint progress exempts each
from stall cost although movement is continually postponed. This scoring loophole is now fixed.

Local followed moving-prefix p90 position errors at +4/+8/+16 are 1.55/2.89/3.41 units (610/544/212 samples),
versus the generic +16 margin 13.9. These selected/survived samples are not a global or untried-candidate
safety bound. Recommended order: fix executed-progress credit/keep a movement-start deadline, then calibrate
local near-pass margins/correction horizon, then A+ budget among safe candidates and checked route joins.
The existing collision/risk/veto/daring rules are unchanged; the separate margin calibration, A+ budgeting
and route-join proposals remain deferred. The implemented fix is `Planner.prefix_stall`: T improvement
and novelty credit stop at the earlier of configured `--replan` and first move end, rounded down to a
direct model-step boundary. Stage 2 reuses its parent's identical prefix; final scoring checks all members.
Private key masks, platform timing and teleports are retained. Verified real-route candidates use real
frame prefixes. No lookups are added when no stall is chargeable. `stall_prefix_frames` / `stall_prefix_t`
are logged with scores, and alternative reasons name the credited prefix instead of the distant endpoint.
The live runner passes the same horizon to local and spawned planners; no new CLI flag is needed.

Offline comparison on the recorded frame-242/250/258 states (same saved online checkpoint, controlled
visits/stall history): baseline chooses `wait 16f, walk R 8f` in all three; fixed scoring chooses
`walk R 8f, wait 8f` in all three, without an early-contact veto. This is isolated decision evidence,
not a new real-game finish or A+ result. Eleven focused prefix cases cover postponement, boundary rounding,
future novelty, immediate exploration, private keys, predicted deaths, inactive stalls and real-frame paths.
Reproducible measurements and plan examples:
`analysis/saw_stall_20261117/audit.py`, `report.json`, `report.md`.
Fix comparison: `verify_fix.py` and `fix_comparison.json` beside them. Restart the agent to load this change.
Final full suite: **635 passed, 1 warning in 360.07s (0:06:00)**; the warning is the existing
pygame/pkg_resources deprecation. Focused prefix/key tests: 30 passed. Compilation and diff whitespace
checks passed. No live game was run and no run/save/memory data or ONLYPPO code was changed.

## online_check12 / 4-12 key objective loss (2026-11-17; fixed offline)

Read-only latest-run inspection, bounded to the first 22 completed 4-12 shards while logging continued:
one correctly detected key, no pickups; waves 1/2 die, 3..22 stall at the locked wall until timeout.
After wave 2 the field changes from 2 legs / spawn T 532 to **1 leg / T 20**, with saved `key_order=[]`.
Cause: homing-beam sweep history still enters `Runner.build_field` even though candidate safety now models
those beams separately. In clear mode it blocks the key route. `plan.progress.route_legs` then drops an
unreachable key and returns a goal-only route; `_build` accepts it before trying timed mode. The same
fallback exists in `realpath.Planner._route`.

Offline replay of the hazard tracking reproduces the loss. With the same wave-2 inputs, timed/risky/squeeze
all retain the key; excluding the two reactive beam histories also restores two legs and spawn T ~439.
Implemented after the diagnosis:
- `realpath.route_legs` is now the shared owner of route-leg search used by both `Planner._route` and
  `ProgressField`. Failure to reach a required key rejects that mode and tries timed/looser routing.
  Incomplete goal-only routes are not accepted. If all modes fail, the progress field keeps every key
  stage, reports `no complete route` and leaves unreachable costs infinite. All keys grouped in a route
  cell must be present in the mask before advancing that leg (viewer and planner agree).
- `EyeTracker.progress_ids` identifies reactive homing beams for progress-map exclusion. The agent and
  offline `planner_inputs` skip those identities before accumulating static/sweep obstacles; ordinary
  hazards, plain lasers and the eye bodies remain. Candidate-dependent beam safety is still active.
- `analysis/key_target_20261117/verify_fix.py` replayed completed waves 1/2 without running a game or writing
  into runs/: **clear, 2 legs, key_order=[[0]], spawn T=439.3, route includes key cell (32,7), zero accumulated
  homing sweep boxes**. Exact result and route: `fix_result.json` in that folder.
- Five regressions in `tests/test_required_keys.py`: timed key routing, unreachable-key invariants,
  grouped key masks, progress exclusion with an actual laser-contact veto, and recorded 4-12 rebuilding.
  They passed. Full final suite: **624 passed, 1 warning in 354.67s (0:05:54)**; warning is the existing
  pygame/pkg_resources deprecation. Python compilation and diff whitespace checks passed. A new live
  run is needed to verify executing the complete route.

Historical evidence and exact log references: `analysis/key_target_20261117/report.md`. Restart the agent
to use the new source; an already-running Python/planner worker keeps its loaded code. No process was
stopped or attached, no save/memory/run data was changed, and ONLYPPO remains untouched.

## Reactive hazards: continuation of Claude's unfinished port (2026-11-17)

**Main only; no commit/push and no game/Frida execution.** Existing A+/realtest work and ONLYPPO were preserved.
Detailed contract, per-file summary, measured evidence and limitations:
`analysis/reactive_hazards_20261117/completion_report.md` (audit script and JSON results beside it).

- Native observation additions: launcher kind 5 (radius 1 x piece scale), aligned `shx` kind/velocity,
  shrapnel kind 16/radius 10, per-frame `lz` (state/timer/angle/animation clock), level `lzp` parameters.
  Existing `sh` is `[id,x,y,radius]`; no incompatible wire-format change. `game.level` owns parsing and
  `wm.buffer` records the extra fields. **Shrapnel capture requires live calibration of unknown offsets;
  fragments are omitted before it succeeds. This part remains unverified against a running game.**
- `plan/lasers.py` rolls each candidate's homing eyes with the decoded float32 aim, live-tile sampled rays,
  state schedule and swept-cross contact rule. `FutureWorld` removes only modelled homing beams from stale
  futures; nonhoming beams remain recorded. Angles must be known before a recorded beam is removed.
  Safety sweeps, all planning stages/continuations, reflex, commit checks and real-route judging share the
  same beam context and laser ids. Single leading planned agent is the accurate targeting configuration;
  independently planned nonleaders remain an approximation.
- Completion fixes: recorded wait animation length/clock and multiplier now drive wait-to-on; missing turn
  divisor differs from explicit zero; portal sweeps split entry/exit instead of crossing the teleport gap;
  beam-table occlusion discontinuities get live casts; casts accumulate float32 steps. Legacy size-only
  homing beam rows are also removed once replaced by a usable model.
- `online_check11`, 4-16 replay: **142 waves / 14,999 steps / 141 of 141 deaths predicted on the exact step /
  zero false survived-step contacts**. Replay is reanchored to observed current angles, not a claim of
  long-horizon accuracy. Vectorised/scalar aim mismatches: zero. Beam-length p99 error <0.01 units, with one
  retained 99.68-unit geometry outlier (cause not established; no contact-verdict effect).
- Recorded planner comparison: 4-16 frame 70 changes from **18/18 vetoed** to **0/19 vetoed**; selected path
  has positive beam clearance, but `safe_alternatives` remains zero. Frame 90 still has all candidates vetoed.
  No claim that the level is solved. Six sampled decisions on 4-6/3-2/4-10 match the HEAD decision hashes.
- Isolated sweep timing, 95 paths x 48 frames: no-eye added median **0.0049 ms**; two eyes add **12.351 ms**.
  End-to-end 4-16 decision medians add ~9.5 / 76.9 / 10.8 ms at frames 30/70/90; changed search work is included.
  No realtime/FPS claim. Full resource/timing and reproducibility details are in the report.
- Unit/native mock coverage: parsing/recording, angle/cast/contact, walls, state timing, candidate dependence,
  portals, legacy rows, sweep/planner/commit integration. `node --check realgame.js` passes; live executable
  offsets, draw-driven wait timing and shrapnel calibration still need live/static verification.
- Final full suite: **619 passed, 1 warning in 297.47s (0:04:57)** with `python -m pytest -q -p no:cacheprovider
  tests --tb=short`; warning is the existing pygame/pkg_resources deprecation. Focused laser/shot/route/async
  tests: **86 passed in 15.86s**. Native fake-memory test (including shrapnel calibration), JS syntax, Python
  compilation and diff whitespace checks pass. Source/report provenance: `completion_manifest.json`.

## ONLYPPO perception parity, separate branch/worktree (2026-11-17)

User request: update ONLYPPO to see what main's **PPO** sees, without adding world models or action planners.
Implemented on branch `ONLYPPO` in `.worktrees/onlyppo`, preserving main's existing work. The native adapter
and PPO geometry now include animated kill regions, enemies, laser beams, projectiles, camera death bounds,
live tiles/keys/platforms and optional half-block/slope inputs. `--shape-inputs` gives main's 524-input layout;
the default 282-input layout and old checkpoint loading remain. No `wm/`, `mpc/`, `plan/`, `agent.py`, online
learner, future rollout, A+ reward changes or realtest were brought over. The branch's existing reward-route
generator (already named `realpath.Planner`) remains a reward/direction helper, not an action controller.

Validation: 8 focused tests pass in that worktree, including main-PPO observation hashes, actual PPO gradient
updates, checkpoint migration, import isolation and native-message producers executed with fake memory.
AST comparison against the pre-port branch confirms `Policy`, `PPO`, `finish_reward`, `on_wave_end`,
`agent_fitness`, all `R_*` constants and `ACTIONS` are unchanged. Native live-game behaviour remains untested.
Details/provenance: `.worktrees/onlyppo/PROJECT_STATE.md`, `PROJECT_HISTORY.md`, and `perception_port.json`.
Local branch commit: `edb6781` (`feat: align PPO-only game perception with current PPO`). Nothing was pushed.

## A+ online rewards and native `--realtest` progression (2026-11-17)

- User-selected semantics: **only A+ finishes are saved to the game and advance the level**. Ordinary
  finishes retry. Add `--realtest` to an existing `agent.py` command; combine with `--online` for training.
  The flag alone does not start a learner, change the checkpoint/replay pair, or turn on recording.
- `wm/rewards.py` owns `finish-aplus-v1`: normal finish **5**, A+ finish **15** (5 + 10 bonus).
  A+ uses the chapter par and the existing rounded `frames * dt / 1000` time, with a confirmed time-step
  lock and a positive finish time. Unknown par/unconfirmed lock: no A+ bonus or automatic native finish.
- `agent.py` records the reward on the **post-action terminal row**, including exact `wave_end.s` targets.
  First/death/nonterminal rewards keep their previous undefined semantics; there is no fabricated terminal
  state. Multi-step feature targets sum the terminal reward once, then scale /5 (ordinary 1, A+ 3).
  `times.jsonl` includes grade, par, reward and protocol; finish events include the reward.
- `wm/online.py` identifies A+ lives from terminal reward rows only in versioned new shards, and samples
  their transitions with **2x weight** within the own-data portion (pending and absorbed waves).
  Existing feature demo masks keep agent/frame selection aligned through filtering. Replay remains 90%;
  validation and publication gates are unchanged. No legacy/PPO rewards are retrospectively relabelled.
  The newest wave remains validation until a later cycle. Training is supervised world-model fitting,
  **not policy-gradient RL**; planning still minimizes estimated time/risk. A higher A+ success rate needs
  a live comparison, and is not established by these tests.
- `--realtest`: select a current planned A+ finish (explore agents and historical bests cannot trigger it),
  finish the wave, flush its training shard, back up `UserData/savegame.dat` once per run, and send the existing
  native `finish` request with **that run's time**. `realgame.js` already performs the game's own completion,
  record update (only if faster) and result-screen continuation. No direct save-format editing was added.
  The request is latched until the next level info. Backup failure logs `realtest save skipped` and keeps
  retrying; it does not request a 999.99 native finish. Manual S retains its existing behaviour.
- Incompatible modes are refused before attachment: `--eval-first`, human/search/demo, unlocked time step,
  or zero planned agents. `--online` and `--learn` retain their own existing requirements.
- Tests: `tests/test_aplus.py` covers reward boundaries, both terminal capture paths, step-1/step-4 targets,
  online sampling and actual gradient updates, flush-before-native-request, exact current time, one backup,
  duplicate suppression, non-A+ / explore / unknown / unlocked / disabled / stopping / backup failure,
  and incompatible CLI combinations. Native map saving/next-level UI has **not been live-tested** in this
  change; the existing S mechanism is reused. Chapter transitions follow the game's own continuation UI.
- Executed validation: `python -m pytest tests/test_aplus.py tests/test_buffer.py tests/test_online.py
  tests/test_learner.py tests/test_first_eval.py tests/test_agent.py -q --tb=short`: **104 passed** (one existing
  pygame deprecation warning). After adding zero-frame and weighted-gradient coverage, the final
  `tests/test_aplus.py` run passed **26 tests** (overlaps the earlier suite). Python compilation and scoped
  `git diff --check` passed. Concurrent reactive-hazard edits elsewhere in the workspace were preserved.

## Moving platforms (4-4), human_v4 measured, tile-plat-v1 (2026-11-15)

**4-4** (online_check01: 15 waves, 0 finishes):
- The bar pf[1] parks flush with the ledge (x 150, top -100) at frames 0.. and again at 361-422. A ceiling above blocks the hop, so walking on is the only way.
- The model predicts falling through the bar: human_v3 has 0 rows of the agent on a bar and 29 within 60 units, and removing its bar inputs barely changes its errors.
- So the agent waits until the bar leaves, then dies in saw h4. That death itself is a wall-jump error: contact is predicted about 4 units early, and a wall jump is expected on a held key.

**Changes:**
- e2bc0f2: the viewer (viz/viewer.py, telemetry pf) draws platforms, live and in replay.
- dd1a0f7 + 2c62016 `tile-plat-v1`:
  - tile-accel-v1 inputs, plus each step's bars painted into the tile patch as plain full tiles (cells at least 0.5 covered; a real tile wins), in featurize / featurize_k / build / features_into. No carry.
  - tile-accel-v1 stays loadable, because the user trained runs/experiments/human_v4_gpu with it.
  - human_v4/features.npz was rebuilt as tile-plat-v1: 1,859,143 transitions, 4-1..4-4 added.
  - Only 3,068 rows have a painted cell, and 38 (all 4-2) have one under the agent. So painting is a prior, not proof: the 4-4 live run is the test.
- 5e01b83: Planner.support, landings, the map-bottom rule and the guard (bar tops; no sides, no carry) use each imagined step's bars (`bars_at(worlds, frame)`), not the bars at decision time. The 4-4 f440 phantom landing is gone. Levels without bars are unchanged.
  - Open: plan/commit.py `Committer.decided` still uses the live bars.

**human_v4 (tile-accel-v1) against human_v3** (`human_v4/accel_check.py`):

| Set | Measure | human_v3 | human_v4 |
|---|---|---|---|
| 3-6 validation | accel-wall vy error, one step (median) | 79 | 1.7 |
| 3-6 validation | open loop +48 | 20.3 | 2.3 |
| 3-5 (never trained) | accel-wall vy error | 38.6 | 13.2 |
| 3-5 | rest of the level, +48 | 9.9 | 16.5 |
| 3-5 | plain-wall vy error, p90 | 43 | 230 |

On 3-5 the transfer is partial and other frames got worse; the plain walls near accelerators may be what drives the p90. The live test is online learning on 3-5.

**Surprise curiosity (deferred step 4):** stopped before any code. In online_check01 the surprising tiles are exactly where the agent is stuck (3-5 (27,14): 1,199 surprises in 9,337 frames), so counting them as new would keep it there.
**Route field and accelerators (step 2b):** no change needed. realpath already climbs accelerator walls, and 3-5's "high jumps" route is the launch off the column top.

## online_check01 reviewed; accelerators, online acceptance, practice, best-route start (2026-11-15)

**The run** (`runs/online_check01`, human_v3, `--online`; events.jsonl):

| Level | Waves | Finishes | Notes |
|---|---|---|---|
| 2-10 | 49 | 33, all A+ | 8.35 s on the first life, 6.4 s best |
| 3-1 | 32 | 31, all A+ | 6.3 s on wave 1, then 6.75 / 7.17 s |
| 3-2 | 25 | 12 | 13 deaths; best 7.75 s, A+ 3 times |
| 3-3 | 23 | 9 | A+ 4 times |
| 3-4 | 51 | 35 | 8.88 s -> 7.07 s, beat the previous best |
| 3-5 | 44 | 0 | 10 falls, then 34 time-outs |
| 3-6 | 130 | 0 | every wave a death; never left the first shaft (max y about -375, goal at 707) |
| 2-6 (held out) | 36 | 0 | 35 time-outs; 18 of 37 waves identical to the previous one |

Every gain came from level memory (route trials, move memory, best route). The world model was not involved: the online learner published v1 once and refused the other 54 cycles.

**Accelerators** (tile low byte 98; 3-3..3-6) were measured from the run:
- Floor accelerators are conveyors. They move the character 3.33 units/frame without changing vx; bit 0x2000 = left.
- Wall accelerator 0x2001062 (face on the tile's left side) gives vy +14.5/frame; a plain wall gives -5.5.
- 0x1001062 (right face) also gives vy +14.5/frame (40 contacts in later 3-6 waves). 0x3001062 (3-5 only) has never been touched.
- `realpath.tile_shape` counted all of them as plain full blocks, so the world model could not see them.
- The blocking ones are the walls (3-5, 3-6). Floors in 3-3 / 3-4 are finished.
- The online fresh loss on 3-3 / 3-4 was 1.2-3.0, against 0.01-0.18 on 3-1 / 3-2.

**Changes:**
- **ae7076c `wm/online.py`:**
  - The forgetting bar is the base (--init) model's replay loss, 0.0597 (bar 0.0626). v1 had dropped it to 0.0203, which then refused everything.
  - Training is incremental: pending waves, with absorbed ones rehearsed at 30% of own rows, at most 1500 steps.
  - Offline replay of 40 online_check01 shards through the real main: 32 of 39 cycles published, replay loss stayed 0.04-0.06, cycles took 2-16 s, and the 3-6 fresh loss fell 0.90 -> 0.11.
  - With the accel-blind human_v3 this teaches "walls push up" to every wall, so use `--online` on accelerator levels with a tile-accel-v1 model.
- **a6cc655 `tile-accel-v1`:**
  - realpath.accel_bits is the one owner of accelerator detection (accel, flip 0x2000, wall 0x1000, face_a 0x1000000, face_b 0x2000000); game/level.py uses it.
  - The new input protocol has 11 tile channels, feat_dim 634, step-4 n_in 664. tile-shapes-v1 inputs are byte-identical to before.
  - human_v4 (`runs/experiments/human_v4`, README there):
    - human_v3 plus human_v3_check01 and online_check01 wm_data (432 shards). 3-5 is left out entirely, as the test of transfer to an unseen level.
    - features.npz: 139 MB, 1,839,438 transitions. Colab training is pending (the user).
- **4546320 practice:** with `--online`, a held-out level's move memory and death memory are on after its first full wave in the run. Held-out entries are never written to the json files (MoveMemory/DeathMemory.transient), so a later run's first life stays clean.
- **2937600 best-route ahead start:**
  - 3-1's 6.3 s route was lost at wave 2 f152. The ahead decision started from the model's guess (0.92 units off), and the margin made a 2.7-unit pass of wave 1 look like soft risk 0.92.
  - On the best route with its recorded actions, `Planner.predict_start` now returns the route's recorded state.
  - Open: a death while playing the route's actions under a library candidate's mode is not reported as "best route died" (the died_at rule for late replays is undecided).

**Not done:**
- The route field (realpath) still ignores accelerators. 3-5 fell back to "high jumps". This waits for a live human_v4 test.
- Surprise-driven exploration when stalled is deferred until the same test.

## Learning from its own play: online learning, curiosity, daring (2026-11-14)

**The user's report** (`runs/human_v3_check01`, human_v3): 3-2 stuck at the same place, 2-10 could not do a long jump away from the goal, 2-6 (wind, held out) not understood. The agent must learn from its tries, as PPO did, a general skill rather than the level.

**Why it learned nothing** (measured on check01):
- The world model never changes during a run. `wm.learner` (`--learn`) trains on a fixed manifest or old folders and not on held-out levels.
- Held-out levels are not recorded at all (2-5, 2-6: no shards), so there was nothing to learn wind from.
- It never tried the move. In 3-2 no wave entered the passage's left side (0 frames left of x 0 at y 230-290, 5 waves); every wave repeated the same decisions (the time step is locked).
- In 3-2 every plan that moves on was vetoed (10 of 17 final candidates; a touch of the floor salt within 16 frames). A veto is ranked below every other plan, so the stall charge (824 frames at the end) could not change the choice.
- 1-2 (finished before) stood 1,155 frames at (-12, -254). The field is flat there (T = 80 from x -160 to 40), every 48-frame plan tied and the stall charge was the same for all. The user's way leaves that stretch to the left and climbs.

**Online learning** (`wm/online.py`, `agent.py --online`):
- A child process (below-normal priority, 2 threads) watches the run's recorded waves.
- After each wave it builds the new wave with the model's own encoding, fine-tunes a copy of the published model on the run's earlier waves (lr 3e-5, 4 passes, each batch 90% replay of human_v3's training data: `runs/experiments/human_v3/replay.npz`, 150k + 20k rows, `make_replay.py`).
- It tests the copy on the newest wave (never trained on). It publishes `runs/<name>/online/wm_model.pt` only when the newest wave's one-step loss is lower and the replay validation loss is at most 5% higher.
- Between waves the agent loads a newer published model (main process and the planner process: `set_model`).
- Held-out levels are recorded to `runs/<name>/wm_heldout` with `--online` (never a `wm_data` folder: offline training never sees them). A first-life evaluation records and learns nothing.
- The base model is never written.
- Measured (`runs/experiments/online_v1`):
  - A cycle takes 6-8 s (3-2, 1-4 waves of data).
  - lr 1e-4, half replay: fresh loss 0.007 -> -0.035, replay +6%, the agent's own 1-8 waves +48 p90 8.5 -> 11.3 (forgetting).
  - lr 3e-5, 90% replay: fresh -> -0.027, replay +0-4%, other levels' open-loop errors even or a little lower (1-8 +48 3.6/8.5 -> 3.3/8.3; 3-1 5.3/68.0 -> 5.0/67.0; 1-1 35.0/76.8 -> 36.3/77.5).
  - An anchor to the start weights kept the replay flat but learned half as much (`sweep.py`; off).
  - In the loop test with waves arriving one by one (`online_v1/loop`), the 2% slack refused every candidate; now 5%.

**Daring** (`Planner.plan`, `ranked`):
- While stalled (the stall rule), a vetoed plan is ranked like the others at one death's cost (soft risk raised to 1). Once the stall charge outgrows a death, the agent tries it.
- REFLEX does not override it then.
- 3-2 offline (`online_v1/curious.py`, check01 wave 1 stalled since 371): at f700 it takes `run-jump L hold 10 > wall-jump L hold 8` (risk 1.00, T 217 -> 202) instead of waiting.

**Curiosity** (`Planner.visit`, `Planner.novelty`):
- Tile visit counts per level for the run, from the real positions (every wave; kept across level changes).
- While stalled, a plan pays the stall charge only for the share of its path in tiles already visited.
- 1-2 offline (stalled since 64): f200 `wall-jump L hold 15 > wait 8f` (walks back along the floor) -> `run-jump L hold 22 > run R 16f` (charge 76 -> 25); f400 -> `run-jump L hold max > wall-jump R hold 8` (276 -> 92).

**Not run in the game.** Tests: `tests/test_online.py`.

**Unknowns:**
- Whether 2-6's wind is learned from a few waves.
- Whether daring kills too often where waiting is right (a moving hazard with a long period).
- 2-10 was not in check01; not looked at.

## 3-2's passage at (41, 226): the model, and human_v3 (2026-11-14)

**PPO passes it** (`runs/collect_more_ppo50_02`, 3-2, 41 waves of 50 agents):
- 1,061 of 1,954 lives get above y 290 (past the gap at x -80..-40); 389 reach the goal's height.
- The way: run left on the platform (y 226), a running jump off its left edge (x 35-41), over the floor salt to the left wall at (-74, 258), a wall jump right up through the gap (x -49, y 285).
- The real paths are not close to the salt (`crumble_v1/ppo_cross.py`, 22 crossings from f141 of wave 29): smallest clearance median 13.4 units (min 9.8); the margin rule on the real path gives risk 0.02 (median).

**The planner sees it as deadly because the model is wrong there:**
- From PPO's own running state (f141, f146) every crossing has risk 0.84-1.00 (`crumble_v1/ppo_state.py`).
- human_v2 rolled with PPO's own actions: the flight to the wall is right within 5-10 units, the wall jump is not. 4 of 5 members predict a fall from (-71, 258) where the game goes up to (-46, 303).
- Median error by 4-frame step, second worst member: 0, 0.4, 1.0, 3.3, 7.5, 10.1, 11.6, 8.3, 8.2, 14.8, 29.0, 43.6.
- human_v2 trained on 8 waves of 3-2 (`collect_2-10_ppo50_01`); `collect_more_ppo50_02` was not in its data.

**human_v3** (`runs/experiments/human_v3/prepare.py`; manifest 21f5f3a5...):
- human_v2's 866 shards, plus 48 of `collect_more_ppo50_02` (3-2, 2-12, 3-1) and 145 of the planner's own runs (`human_v2_check01..04`, the excluded levels left out).
- Held out: 8 of the 41 3-2 waves (every 5th: `eval_waves.json`; wave 29, the one measured above, is one).
- Demos: human_v2's (the user's A+ finishes).
- Features for the GPU: `runs/experiments/human_v3/features.npz`, 129 MB, 1,758,968 transitions (347,217 validation); 3-2 is 257,969 of them, 3-1 147,741, 2-12 40,034; same options as human_v2 (`--step 4 --sample-frac 0.4`).
- Trained by the user on Colab: `runs/experiments/human_v3_gpu/wm_model.pt` (48 epochs, batch 4096, cuda). 3-2 is now a trained level (as 3-1 did with the user's play).

**human_v3 checked** (not yet run in the game):
- One-step report (its own 347,217 held-out transitions): position error median 0.08 / p90 0.89 (human_v2 on its own set: 0.10 / 1.01); death precision 0.68 (0.62).
- The held-out 3-2 wave 29, PPO's own actions from f141 (`MODEL=... crumble_v1/ppo_cross.py 141`, 22 crossings):
  - The worst member's error by 4-frame step, median: human_v2 0, 0.5, 1.4, 5.8, 12.8, 16.2, 27.9, 31.7, 34.4, 35.9, 41.0, 61.8; human_v3 0, 0.1, 0.3, 0.5, 1.2, 1.3, 1.9, 2.0, 2.8, 3.2, 4.3, 4.2.
  - 3 of 5 members draw the wall jump up through the gap as the game does.
- The user's held-out play (`compare.py --human`, `human_v3/compare/human.json`, 18 levels), weighted by samples:
  - +16: median 0.71 -> 0.64, p90 6.10 -> 5.09.
  - +48: median 4.00 -> 3.68, p90 34.9 -> 33.7.
  - Per level at +48 p90: 7 better, 7 worse (3-1 16.8 -> 28.6, 5 waves; 1-13 29.1 -> 35.2), 4 even. At +32 median: 13 better, 1 worse.
- Not checked: the held-out levels (1-18..1-20, 2-5..2-7).
- The planner at 3-2's passage (`MODEL=... crumble_v1/ppo_state.py 29 0 ...`, `stall_rule.py`), human_v2 -> human_v3:
  - From PPO's state at f146 (41, 226, vx -299): risk 1.00, estimate 378 -> `run-jump L hold 22 > run-jump R hold max`, risk 0.00, T 222 -> 159, estimate 227.
  - From the agent's own arrival (check04 wave 3, f176, (82, 238), vx -220): `wait 8f > wait 8f` (269) -> `run-jump L hold 10 after 8f` twice, T 232 -> 191, estimate 239.
  - Standing at (41, 244) with no speed it still waits (it needs a run-up), until the stall rule's charge grows.

## The stall rule (2026-11-14)

**Live run** (`runs/human_v2_check04`, the full jump, the reflex rule, the crumble clock):
- 3-1: finished on the first life in 414 frames; 12 of 18 waves finished (2 died to the ceiling shot, 2 in h5#1, 2 timed out).
- 2-5 (held out): 2 of 2 waves finished, 511 frames.
- 3-2: 0 of 10 waves finished. The agent climbs to (41, 226) by frame ~177 and stays within 60 units of it for about 1,020 frames.

**Why it stood:** a waiting plan scores frames + T at its end (268-274), as if the way on were free after the wait. The plans that go on cost soft risk near the floor salt h29 / h30 (303-328), so waiting won every decision.

**Rule** (`Planner.stalled`, `stall_penalty`, `STALL_FREE` = 60, `STALL_STEP` = 10, `agent.py --stall-free`, -1 = off):
- Per agent: the best T reached and the frame it was reached; it moves on when a decision's T is 10 frames lower.
- A plan that does not end 10 frames below the best T pays the frames since then, beyond the 60 free ones. A plan that does pays nothing. No part credit (a hop in place lowers T by ~7 for a moment).
- Charged in stage 1, stage 2 and the final estimate; a vetoed plan is still never chosen over one that is not.
- Logged: `score.stalled`, `score.stall_frames`, and the alternatives' reason "stalled: no progress for N frames".
- Offline (`crumble_v1/stall_rule.py`, 3-2 wave 3, as if stalled since f180): f300 (60 charged) still waits; f400 (160) takes `jump L hold 6 > jump up hold 15`, risk 0.78, T 225 -> 213; f640 (400) takes a risk 1.00 plan, T 221 -> 191.
- Not run in the game. Levels that need a wait longer than about 2-3 s for a moving part will now be rushed.

## Narrow passages: the reflex rule, not the walls (2026-11-14)

**The user's report:** agents are thrown off by hazards behind walls and play badly in narrow sections (3-2).

**Live run** (`runs/human_v2_check03`, 3-2, the full jump; 0 of 9 waves finished):
- The goal is up (y 566); the spawn is at y -190 in a shaft 120 units wide with salt in both walls.
- Wave 1 climbed to y 232 and died there (h28).
- The later waves went down the shaft and stayed at its bottom until the time ran out.

**Hazards behind walls: measured, not the cause.**
- Our kill shapes are right in 3-2 (`runs/experiments/crumble_v1/alive_in_kill.py`, PPO, `runs/collect_more_ppo50_02`, 6 waves):
  - 119,010 alive rows, 82 inside a kill shape (0.07%).
  - 262 deaths, 98% within 3 units of a shape.
- The salt's boxes lie inside the wall tiles, but the game's box test reaches out of the wall. A body touching the shaft's right wall dies on 150 of 186 sampled heights, the left wall on 88 (`check03`, y -724..16).
- A full wall tile is 20 units thick and the largest margin is 13.9, so a hazard behind a full wall never enters the soft risk.
- A reachability test (`walled`: can the body get to where it dies without entering a solid tile) was written and removed again. It changed nothing in these cases and cost 50% more planning time in the shaft (310 -> 460 ms).
- 17 of 3-2's 130 kill shapes cannot be touched from any free position; none is near the agent's path.

**The cause: the reflex override.**
- REFLEX replaced the chosen plan whenever it had any early risk: a hazard within margin(k) in frames 1..16, no touch.
- It then took the short move with the least early risk, whatever its progress.
- The normal estimate did not price those frames at all, so the two rules were never weighed against each other.
- Overrides in the live runs:
  - `human_v2_check02` (3-1): 2,946 of 6,787 decisions for early risk, 15 for a veto.
  - `human_v2_check03` (3-2): 199 of 1,060 for early risk, 3 for a veto.
- Examples (3-2 wave 1): f16 "early risk 0.09": `jump up hold 6`, estimate 531, over plans at 364; f72 "early risk 0.03".

**Rule now** (`_verdict(early_margin=True)`, `Planner.early_margin`, `agent.py --reflex veto|any-risk`, default veto):
- A near pass in the planned frames 1..16 is soft risk like a later one (1 - clearance / margin(frame)), priced in every stage's estimate. A touch there is still a veto.
- REFLEX overrides only a vetoed plan. The reflex move stays a candidate.
- The best route's real-path verdict prices the early frames the same way.
- `--reflex any-risk` is the old rule.
- Offline (`crumble_v1/reflex_rule.py`, 3-2 wave 1 f16): old `jump up hold 6` (estimate 428) -> new `jump R hold 22 > run-jump R hold max` (352).
- Goldens re-recorded (they pass with `early_margin=False`).
- Not run in the game. Near passes are now weighed, not refused: more deaths near hazards are possible.

**Crumble blocks: measured** (`crumble_v1/delay.py`, 4 levels, 164 touched tiles):
- The first word change comes a median 29 frames after the first touch (p90 29-30).
- The tile is gone 29-30 frames after the touch at the earliest (median 29-30).
- Untouched neighbours change too (a chain); their timing is not measured.

## Crumble blocks in the imagined world (2026-11-14)

**Crumble clock** (`Planner.crumble`, `Planner.touch`, `CRUMBLE_DELAY` = 30):
- Every real state (`Planner.observe`): an intact crumble tile within 1 unit of the body box is expected gone 30 frames later.
- The game's own tile change for a cell (`Planner.tiles`) ends the expectation. So does a new wave.
- A tile still there 10 frames past its time is forgotten (the touch was not one).
- Untouched neighbours stay solid until the game changes them.

**Imagined steps** (`imagine(stages=...)`, `Planner.crumble_stages`, `Planner.imagine_options`):
- A row's model steps that start at or after a tile's time see it open: in the collision guard's tiles (`Terrain.opened`) and in the model's tile inputs (`LevelGeom.opened`, a view with those cells' channels 0, as an opened lock's).
- Rows with their own start offsets (stage 2, the completion) get their own stage.
- The plan made ahead (`predict_start`) counts from its own frame.
- Before the first touch nothing is expected: the field (crumble cells cost 1 frame) still leads the agent to the wall.
- Not changed: the field's crumble cost, the support rule at a plan's end, the corner risk's tiles.

**Checked on the user's 3-1 wave that passes the wall** (`crumble_v1/wall_pass.py`; touch f398, the game's tiles gone f428 / 430 / 436, the user through at f451):
- The clock's times from the user's real states: 429, 430, 436.
- One decision from the user's state, clock off -> on:
  - f404 (354, -8): T 48 -> 48, not past the wall -> T 48 -> 15, past it at +41.
  - f412 (346, -18): T 48 -> 45 -> T 48 -> 5, past it at +30 (y -30).
  - f420 (326, 21): T 58 -> 46 -> T 58 -> 3, past it at +27 (y -30).
- `plan.offline.closed_loop` gives the planner the recorded agent's touches before START.
- Tests: `tests/test_crumble.py`.
- Not run in the game. Not covered: arriving at the wall low and fast (the 4 deaths in h6 came within 5 frames of the touch).

## 3-1: the full jump, and the crumble wall (2026-11-14)

**Live run** (`runs/human_v2_check02`, after the corner and shot fixes; 0 of 24 waves finished):
- From wave 2 on, no wave died in the pit.
- Every wave stood at the pit's edge (x 145-175) for about 130 frames, waiting and jumping in place, before crossing.
- Most waves then stood at (160, -34) above the saw floor until the time ran out (560 frames).
- 4 waves died in the saw h6 at the crumble wall's foot (354, -36): the agent slid down the wall at about 4 units/frame.

**Why it waited** (`guard_side_v1/edge_wait.py`, `human_jump.py`):
- The library's longest jump held the key 22 frames (and released it in the padding).
- The user holds it 30-66 frames at the pit (median about 47; 26 waves).
- The world model, rolled with the user's own actions from x 120, flies the pit with every member (x 440: real -199, members -201..-210).
- Held 22 frames it is 35 units lower at x 400 and falls into the salt. 30 is close; 36 and more are the same (the jump stops rising).
- So every crossing the planner could try died, and waiting (369 frames) won over jumping (440-480).
- The live alternatives at the edge: "hits h10" 662 times.

**Full jump** (`plan/moves.py`, 73 -> 82 moves):
- New moves: `jump up hold max`, `jump L/R hold max`, `run-jump L/R hold max` (also after 4f / 8f). The key is held to the move's end (24 frames).
- `padded` keeps a held key through the padding.
- The completion's "keep" continuation keeps it too.
- `run-jump L/R hold max` is in CORE_SECOND, so "run, then the full jump" is searched as a second move.
- Offline (MARGIN_HORIZON unchanged at 16):
  - From the user's state at x 101 / 120: `run-jump R hold max`, risk 0, estimate 296 / 307 (waiting is 369).
  - From the agent's own state at x 73: `run-jump R hold max after 4f`, estimate 338 (it chose `walk R 8f`, 364).
  - At (160, -34): `run-jump R hold max`, risk 0, T 93 -> 47 (it waited).
- Stage 1 evaluates 12% more moves.
- `Planner.last_stage2` (offline checks).
- A cap of 8 on the margin's horizon was tried. With the full jump it is not needed for these decisions, so it was not changed.

**Offline harness fix** (`plan.offline.closed_loop`):
- The current wave's frames before START are now fed to the future world, as the live agent saw them.
- Before, a flying shot had a one-frame track and looked 641 units wide after the recording. Every completion "hit shot0".

**The crumble wall** (x 360, y -40..40, tile id 100; the only way to the goal at (529, -33)). From the user's wave 2:
- The body touches it at f398.
- The touched tiles change at f428 (30 frames later): 0x3000164 -> 0x3400164 / 0x3800164 (cracking) -> 0x100 (gone), 2-8 frames apart, spreading to the neighbours.
- The user slides about 12 frames, wall-jumps away, comes back and passes at f451.
- 2 of the user's waves also died sliding into h6.

**Crumble wall: not done (needs a decision):**
- `realpath.CRUMBLE_COST` = 1 frame per crumble cell (the field thinks the wall is nearly free).
- The planner's world keeps the wall solid until it breaks.
- The completion stops at the first wall contact (support), so a slide into a saw right below is not seen.
- `solid_now` counts a cracking tile (any changed word) as open; whether the game does is unknown.

## 3-1 with human_v2: the corner of the salt pit (2026-11-13)

**Live run** (`runs/human_v2_check01`, the user's two runs: `--risk-members second`, then `mean`; 0 of 11 waves finished):
- 6 of the 11 deaths are at (434, -234..-237) in salt h15, on the far side of the pit.
- 3 are in h1#1 at about (404, -225), after a jump into the pit that every candidate then vetoed.
- 1 is in h3, and 1 is unexplained at (354, -39).
- `runs/experiments/guard_side_v1/deaths.py` lists the last plans before each death.

**The corner.**
- The landing cell (x 440..460, y -240..-220, word 0x43842011) is a half block with its lower half solid (top y = -230). It is not a slope.
- In the game the step (429.8, -224.7) -> (436.3, -228.7) ends at (434.0, -228.7). The feet were 0.7 below the top, so the body hit the block's side and slid down into the salt.
- Our guard used a box shrunk by its 1-unit tolerance. To that box the move along x looked free, so it put the body on top of the block at (436.3, -224.7).

**Guard side rule** (`plan.planner.collide`, `SIDE_SHRINK` = 0.05):
- Applies when the x move is free only for the shrunk box (not for the near-exact box) and the y move alone is free.
- Then y is kept and x goes back to contact, as the game does. The corner step now gives (434.01, -228.7).
- Unit test: `test_collide_lands_on_the_half_floor_and_keeps_the_free_axis`.

**Not enough alone** (`guard_side_v1/corner_decisions.py`: the same decisions under both rules):
- The model's path is about 4 units higher than the real one: (431, -221) predicted where the game was at (430, -225).
- In imagination the feet clear the corner by about 3 units, so the guard never acts.
- The planner keeps choosing a short hop (`run-jump R hold 3`).
- The user crosses the face x = 440 at y -197..-208 (feet about 20 units above the top) in 22 recorded crossings of `runs/human_01`.

**Ledge-corner risk** (`plan.planner.corner_risk`, `agent.py --corner-risk on|off`, default on):
- A step whose x move the guard lets through, but which would hit a block's side if the body were up to margin(k) lower (`plan.safety.margin`), passes a ledge with its feet c units above the top.
- When the slide down that face from the top reaches a deadly box or leaves the level, the step costs 1 - c / margin(k) of soft risk.
- That risk combines with the hazard soft risk by the same members rule (`_verdict(corner=...)`, never a veto).
- Checked in stage 1, stage 2, the final stage, the modes' `_judge` and the airborne completion (`complete`: margin at H + k).
- The window is every frame the score credits. The pad tail counts too, because its landing's T is credited (3-1 f109: the corner was in the pad).
- A missed ledge with a safe floor below costs nothing.
- The user's own paths (`guard_side_v1/corner_on_human.py`, 17,198 windows of 48 frames, `runs/human_01`): 2.2% charged.
  - 0% on 1-1..1-6, 1-9, 1-10, 2-3, 2-4.
  - 7.2% on 1-7, 5.8% on 1-8, 5.5% on 2-1, 4.3% on 2-2, 2.7% on 3-1.
  - Mean risk when charged 0.2-0.57.
  - Where the user really passes corners close, the agent will now be more careful; `--corner-risk off` gives the old planner.
- Cost: 3.4-4.9 ms per call (`corner_risk` on the 3-1 final and stage-2 sets), 3-4 calls per decision.
- Offline (`corner_decisions.py`): the 3-1 plans that end at the corner now carry risk 1.0 (f109: 0.217 -> 1.0, f141: 0 -> 1.0). The chosen jump does not change: from that state every alternative already had risk 1.0.

**Why the agent jumps from too far back: the ceiling shot** (`mpc/future.py`):
- A launcher fires a shot every 119 frames at (111, -111), above the solid ceiling (y -160).
- A shot seen 1-2 frames is bounded by SHOT_V_MAX k in every direction, through walls: 161 units at +16, 481 at +48.
- Every birth triggered "danger shot0" (all candidates "hits shot0") in every wave of both runs. The reflex mode waited, then jumped early (`run-jump R hold 10` from x 137-167; the user jumps from about x 150-160 and lands about 20 units higher).
- The live decision at f93 was planned ahead from f85's world (the async planner), so it still saw the newborn shot's circle.
- Now (`FutureWorld.flying_shot`): a shot born less than 3 frames ago from a launcher whose flight is known follows that flight by age. The margin is one frame of motion (7 units instead of 161).
- A flight cut by the wave's end is kept as the launcher's partial template.
- Only the level's very first shot (nothing known yet) keeps the old bound.
- Replayed (`guard_side_v1/shot_birth.py`, wave 2 birth at 442): no shot trigger, normal mode.

**Bug fixed on the way:**
- `@lru_cache` sat on `class Launcher` (meant for `_fit_weights`), so every `Launcher()` was one shared object: every spawn point of every level shared one period and template.
- It is now on `_fit_weights`; `test_every_launcher_is_its_own`.

**Not verified in the game.** The offline closed loop uses runs/wm_locked as the game.

## 3-1 salt pit, a less fearful risk rule, training on a GPU (2026-11-14)

**3-1** (shapes_check01: 12 of 13 waves time out at (136, -254)):
- The way on is a ~55-frame running jump over a salt pit, under a low ceiling (y = -160).
- The 10 recorded PPO crossings never enter our kill regions (0 of 7627 frames).
- The model predicts that jump well: from PPO's own actions, the +48 error is 1-5 units and the members' clearance 8-25.
- The field falls along it (315 -> 237).
- Approaching at full speed, every jump candidate had at least one member grazing the salt. With the soft risk of the worst member that is a whole death (+145 frames), so the planner slowed down, and from the pit's edge a standing jump cannot make it.
- An averaged-member rule tried offline died in the pit of the stand-in world (runs/wm_locked is not the game), so the offline evidence is not conclusive either way.

**Risk rule now** (`plan.planner.RISK_MEMBERS`, `Planner.risk_members`, `agent.py --risk-members`): `second` (new default) = the second worst member's soft risk, so one member alone grazing a hazard costs nothing extra and two agreeing members still cost a whole death; `worst` = the old rule; `mean`. The golden planner cases were re-recorded.

**human_v2** (`runs/experiments/human_v2/prepare.py`):
- human_v1 plus the user's 27 waves of 3-1 (22 trained, 5 held out), 197 A+ demos.
- The CPU training was stopped by the user, to be done on Colab.

**Training on a GPU** (`wm/train.py`):
- `--save-features NPZ` builds the features from the manifest (every data option applies), writes them with the validation split and the manifest's reference, and stops.
- `--load-features NPZ` trains from that file alone; no recordings or manifest paths are needed.
- `--device cuda`.
- Checked end to end on CPU with a small file: 24,248 transitions, 0.4 MB. The model loads in the agent.

**human_v2 trained on Colab** (`runs/experiments/human_v2/gpu/wm_model.pt`; features.npz 105 MB, 1,396,348 transitions; 48 epochs, batch 4096, cuda; not yet run in the game):
- Held-out human play (`HUMAN_EVAL=runs/experiments/human_v2/eval_waves.json compare.py --human`, `compare_human.json`, human_v1 -> human_v2):
  - 3-1 (5 waves, never trained on by either) is clearly better: +32 0.6 / 14.8 -> 0.5 / 8.7, +48 1.2 / 33.4 -> 0.8 / 16.8 (median / 90%).
  - The other 17 levels are about even, a few units up or down at +48.
- 1-14..1-17 recordings (`compare_new_runs.json`): even (1-15 +48 median 11.1 -> 9.9, 1-14 24.1 -> 24.8).
- The 3-1 pit offline from the real stall (`salt_3-1_from.py 80`, runs/wm_locked as the game, `--risk-members second`): neither model crosses. Both slow down at the edge and die later in the stand-in world. Only a live run can tell.

## 2-11 locks the game dissolves, and 2-16 laser beams (2026-11-14; code-read, not yet checked live)

**2-11:**
- After a key is picked up the game dissolves its lock tiles one after another: in the recording, 11 frames after the pickup, then one every 7 frames.
- realgame.js put them back (closed) after every frame, and each key holder saw every lock open at once.
- With one agent in the game (`agent.py`, `n_total == 1`), `Level.use_native_locks()` now gives every key an empty lock group. realgame.js then neither opens nor re-closes anything, so the game's own dissolving is what the agent meets, and the live tile changes (`tc`) show each tile go. The world model and the guard see the live tiles.
- Per-agent locks stay for several agents (PPO).
- The 2-11 stand-still itself (waiting at (43, -134), every move "no progress") is not explained by this and is not checked.

**Lasers** (Hospital palette lazeron / lazeroff / lazerwait):
- The level build (FUN_004c8c10, 0x4c96ec) puts each laser into the world array at +0x2ce8: u16 count +0x2d00, pointer +0x2d10, 0x24 bytes each, +0 the piece.
- FUN_004cd350 tests every entry whose +8 is 1 with FUN_004cbd40. The beam is the segment +0x14/+0x18 -> +0x1c/+0x20, and it kills when it crosses the character's box (half 6).
- `realgame.js` appends one hazard row per laser (lethal type 2000, id `laser<entry>`) with kill shape `[3, x0, y0, x1, y1, 0]` while +8 == 1.
- `game.level` SHAPE_SEGMENT: clearance = Chebyshev distance to the segment - BODY (exact; checked against brute force).
- All four 2-16 deaths in enemies_2-15_check01 lie on the beam line of the laser at (-208, 37).
- Not known yet: that +8 is the "on" state for plain lasers (the level build writes the homing flag there first), and the end point's meaning. realgame.js logs the raw entries on the first frames and whenever +4..+0x10 change.

## Kill shapes exactly as the game tests them (2026-11-14; offline-checked on recordings, not yet run live)

**Finding: the game's box test is not an overlap test.** Needle and salt pieces (kill shape kind 2) killed agents 8-17 units outside the box + BODY we used, so the agent thought the gaps were safe.
- The character's hazard box is char+0x71c: an OBB of half size char+0x874 (props 0x21d = 6), built at 0x5218bf. So BODY = 6 was right.
- Two type-0 bounds go through FUN_00476720 at 0x476744. With d the distance between the centres, D1 the distance from the character's centre to the kill box and D2 from the kill box's centre to the character's box (FUN_004765f0, point-to-box), the character dies when d^2 >= D1^2 + D2^2.
- That reaches a + 6 + sqrt(12 a) along the box's axes: 23.6 / 19.9 units for a 7.88 x 5.66 needle, against 13.9 / 11.7 for box + BODY. It reaches furthest on the diagonals (30.7).
- On all recorded Hospital play (runs/experiments/needle_shape_v1/exact.py): of 3284 death frames next to a box shape, this test explains 95.4%, box + BODY 1.9%. Of 247,193 alive frames, 0.4% contradict it.
- Circles (kind 1: saws, enemies, shots) are tested exactly (closest point of the character's box within r). Our bounding square was too big at its corners.

**Change** (`game/level.py`, `plan/safety.py`, `mpc/future.py`, `plan/planner.py`, `plan/commit.py`, `viz/`):
- Every deadly shape keeps the game's geometry and a type: SHAPE_BOX (remembered deaths, rows without measured shapes), SHAPE_CIRCLE, SHAPE_GAMEBOX.
- `game.level.kill_clearance` is the one rule. Circles exact. Game boxes: a star-shaped region traced on 144 rays (`game_box_profile`, cached per (a, b, angle)), interpolated (5-9 sign mismatches in 20,000 random points). Beyond 30 units past the furthest radius, r - that radius is used.
- `FrameBoxes` carries `shape` and `grow` (a future's uncertainty radius, no longer added to the half sizes); `take`, `clearance`, `reach`.
- `hazard_boxes` (the progress field, grids, PPO) keeps the contact convention, with game boxes as `box_reach`, the box covering the region.
- Telemetry rows are `[x, y, hw, hh, a, kind, shape, grow]`. The viewer draws circles as circles, boxes as their turned boxes, and the game-box death region as a faint outline (`kill_outline`).

**Effect on recorded deaths** (`explain_again.py`; runs logged them all "unexplained"): now explained 3-1 6/7, 3-2 5/7, 2-14 7/7, 2-12 3/3, 2-9 6/8, 4-1 2/2.
- Left unexplained: 2-16 4, 2-17 3, 2-18 1 (lasers / other, not read yet), and 2-13 / 2-15 in human_v1_check01 (enemies; that run predates the enemy reader).

**Cost.** Offline decision median, plain boxes -> exact shapes: 3-2 168 -> 203 ms, 2-14 207 -> 254 ms, 1-14 148 -> 142 ms.

## Enemies the agent could not see: SMBAnimal crawlers and fliers (2026-11-14; code-read, not yet checked live)

**Finding** (human_v1_check01 on 2-9, 2-12..2-15: every death "none", unexplained):
- The pieces of type 44 and 45 (lethal-listed, never a kill shape) are the Hospital palette's "particle" and "smoke": decoration, not the killers.
- The moving killers are not pieces at all. They are `SMBAnimal` objects (SMBAnimal.cpp, vtable 0x682488, ctor FUN_005129e0), which the earlier note "SMBAnimal = background critters" missed.
- Kind +0x1cc indexes 0x6f1a00: 0-3 groundcritter, 4-7 aircritter (background), 8-11 groundenemy, 12-15 airenemy, 16 bosscritter, 17 charger, 18 boo, 19 hellbossboy.
- Manager: `[[0x7035c4] + 0x2e48]` (u16 count +0x18, capacity +0x1a, pointer array +0x28; FUN_00513430 adds one).
- Kill test: the character update FUN_00520d30 at 0x5218ef, right after the pieces check, calls FUN_00513200(char, char+0x71c box). Every animal of kind 8-15 or 17-19 whose state +0x1d0 is not 5 is tested with FUN_00476720 against its bounds at +0x13c: a circle (type 1), radius +0x140 (10 from the ctor), centre +0x144/+0x148 (world: FUN_00476720 uses it directly).
- The next check after it, FUN_0052f770, is the boss (DAT_007047c4); not read.

**Reader** (`realgame.js` buildEnemies / enemyRows):
- The enemies are appended to the hazard rows after the lethal pieces, as `[x, y, 2r, 2r, 1, 1, 0, [[1, x, y, r, r, 0]] or []]`, with lethal type 1000 + kind and id `enemy<address>`.
- Everything that reads hazards (kill boxes, futures, world-model inputs, death explanation, viewer) gets them without other changes.
- The first 3 frames of a level log each enemy's kind, state, bounds flags, radius, centre, +0xd8 position and +0xe8 spawn, to check this layout in the game.

**Not known yet:**
- whether the centre is updated every frame;
- whether the agents' wave reset also resets the enemies;
- whether fliers home in on the hidden real character (the camera).

## human_v1: the first model trained with the user's own play (2026-11-14; offline, not run in the game)

**Data** (`runs/experiments/human_v1/prepare.py`):
- `runs/human_01` (agent.py --human): 805 waves on 17 levels (1-1..1-13, 2-1..2-4), every frame locked, 271 finishes, 233 under A+ with no duplicate route.
- Manifest: wide_v1 (200 PPO-era shards) plus 644 human shards. Every 5th human shard (161) is held out (`eval_waves.json`).
- Demos: the 188 human A+ finishes of the training shards (`aplus_train.json`), oversampled to 19.8% of the training samples (`--demos ... --demo-share 0.2`).
- `--sample-frac 0.4` (RAM; demos kept whole): 1,390,570 transitions. Otherwise the epoch_48 settings (tile-shapes-v1, step 4, 3x256, seed 0, 48 epochs, 2912 s).
- Model: `runs/experiments/human_v1/epoch_48/model/wm_model.pt`. Its held-out loss was 0.022 at epoch 36 and 0.024 at 48.

**Held-out human play** (`compare.py --human`, `compare_human.json`): the new model is better on every level, mostly by a factor of 2-5 at +32/+48. Median at +48, epoch_48 -> human_v1:

| 1-1 | 1-9 | 1-11 | 1-12 | 1-13 | 2-1 | 2-2 | 2-3 | 2-4 |
|---|---|---|---|---|---|---|---|---|
| 12.4 -> 2.3 | 6.3 -> 0.9 | 10.3 -> 1.8 | 7.1 -> 4.3 | 19.8 -> 5.4 | 14.6 -> 3.3 | 17.8 -> 7.1 | 6.6 -> 2.0 | 50.6 -> 10.3 |

- The 90% values fall too, with two exceptions: 1-6 +48 (28.9 -> 39.3, 2 waves) and 2-3 +48 (31.3 -> 34.3).
- The held-out waves are other attempts on the same levels as the training waves: this measures your routes on known levels, not unseen ones.

**The 1-14..1-17 recordings** (`compare_new_runs.json`, no human data on these levels): about even.
- Better up to +16 frames on all four levels, and on 1-16 and 1-17 at +48.
- Worse at +48 on 1-14 (median 21.8 -> 24.1) and 1-15 (10.6 -> 11.1), and some 90% values at +32 (1-14 44.5 -> 47.6, 1-15 46.3 -> 50.4).
- This arm also sampled each wave at 0.4 instead of 1.0, so the human data's effect here is not isolated.

It loads and plans in agent.py (simulate.py, 1-14, 0.73x).

## Bolder risk rules, your own play as data, A+ examples (2026-11-13)

**Risk rules** (`747d8d4`; the golden planner cases re-recorded after it):
- 2-1 (nowait_epoch24_check01): the agent stood at (-155, -194) for whole waves. Every move that made progress got "risk 0.54 near h18 at frame 32", costing about 268 frames, while waiting cost only its frames.
- The soft-risk margin (`plan.safety.margin`) now stops growing at `MARGIN_HORIZON` = 16 frames (13.9 units instead of 38 at +32 and 60 at +48). Plans are corrected long before their far frames are played.
- A death costs at least `DEATH_MIN_FRAMES` = 120 frames instead of 500 (`--death-min`).
- `DeathMemory`: unexplained deaths within 16 units of each other at wave frames more than 30 apart are on at every frame. In 2-1 the agent died at one place once per wave, 48 frames later each time. There our kill box from the game's shape is about 14 units smaller than where it dies; the box read itself is not fixed.
- In the offline replay of the 2-1 stand-still, the first decision's risk cost went from 301 to 0 and still plans fell from 36 to 8 of the decisions. The 3 golden cases that changed all wait less (1-13 f200: "wait 16f" became "jump up").
- Not run live yet.

**Your own play** (`agent.py --human`, `plan/human.py`): agent 0 takes the keyboard (arrows or A/D, jump Space/Up/W, run Shift/Ctrl; never S) or the first XInput pad, read each frame while the game window is in front. It is recorded with `--wm-collect` exactly like an agent (source "human"), and finishes print the A+ verdict.

**A+ examples** (`python -m plan.aplus RUN... | --all`, writes runs/demos/aplus.json for `wm.train --demos`): finishes under the level's A+ time, with the time-step lock, rows 0..f without a gap, no teleport, not held-out or user-excluded, and no exact duplicate route. Over all existing recordings (`runs/demos/aplus_existing.json`): 7077 kept, mostly PPO on 1-1..1-13 and 2-10; the planner on 1-14 (25), 1-15 (2) and 1-17 (17).
- Some PPO times are far under A+ (1-1 median 1.5 s against 3 s, 1-3 1.5 s against 9 s). The par lookup (`levels_smb.a_plus_time`) may be picking a level from the wrong chapter; this is not checked.

## Model comparison on new recordings, the 1-17 stall offline, --veto-members (2026-11-13)

**Two models on recordings neither trained on** (`runs/experiments/model_compare_v1/compare.py`, results in `step4_vs_epoch12.json`): open-loop position error, median / 90%, game units (a tile is 20), with the agents' real actions and recorded hazards. `wm_step4` is the model in use; `epoch_12` is `runs/experiments/tile_shapes_epochs_v1/epoch_12/model/wm_model.pt` (tile-shapes-v1, 12 epochs, never run in the game).

| Level (waves) | Model | +4 | +8 | +16 | +32 | +48 |
|---|---|---|---|---|---|---|
| 1-14 (26) | wm_step4 | 0.8 / 3.4 | 2.3 / 9.1 | 7.4 / 25.5 | 20.9 / 60.2 | 41.3 / 80.2 |
| | epoch_12 | 0.7 / 3.0 | 2.0 / 7.7 | 5.0 / 22.6 | 16.4 / 59.0 | 34.8 / 82.8 |
| 1-15 (31) | wm_step4 | 0.6 / 3.3 | 2.3 / 8.5 | 6.3 / 21.8 | 18.0 / 66.7 | 32.1 / 101.1 |
| | epoch_12 | 0.5 / 3.2 | 1.9 / 7.6 | 4.7 / 19.7 | 12.4 / 63.4 | 20.4 / 99.9 |
| 1-16 (8) | wm_step4 | 0.7 / 4.5 | 2.4 / 12.1 | 6.9 / 37.4 | 19.9 / 73.8 | 47.8 / 129.8 |
| | epoch_12 | 0.5 / 2.6 | 1.5 / 8.0 | 4.7 / 22.4 | 19.1 / 60.6 | 40.3 / 92.0 |
| 1-17 (25) | wm_step4 | 1.3 / 5.0 | 3.5 / 12.3 | 8.9 / 31.1 | 20.9 / 73.4 | 30.5 / 127.7 |
| | epoch_12 | 0.6 / 3.7 | 1.8 / 10.8 | 6.1 / 23.9 | 15.5 / 60.3 | 29.0 / 83.9 |

- epoch_12 is better at nearly every horizon on all four levels (2-14 has one wave: no conclusion). Both are still 1-2 tiles off at +48 frames (median), 3-6 tiles at the 90th percentile: long plans are guesses in both.
- `wm.compare_steps --legacy-split` refuses the strict checkpoint, so compare.py calls `roll_wave` directly on waves recorded after both trainings.
- **24 epochs** (`runs/experiments/model_compare_v1/epoch_24/model/wm_model.pt`; same manifest, tile-shapes-v1, seed 0 and settings as the 12-epoch arm, 4 threads, 1718 s; never run in the game). Same waves, `three_models.json`:

| Level | Model | +4 | +8 | +16 | +32 | +48 |
|---|---|---|---|---|---|---|
| 1-14 | epoch_24 | 0.3 / 1.9 | 1.2 / 4.7 | 3.9 / 19.1 | 12.9 / 51.1 | 26.3 / 73.2 |
| 1-15 | epoch_24 | 0.3 / 2.5 | 1.2 / 6.0 | 2.9 / 14.1 | 8.9 / 48.6 | 16.1 / 95.6 |
| 1-16 | epoch_24 | 0.3 / 1.8 | 1.1 / 5.9 | 3.4 / 24.7 | 15.2 / 63.4 | 27.2 / 94.0 |
| 1-17 | epoch_24 | 0.4 / 3.2 | 1.1 / 10.2 | 2.9 / 21.6 | 8.8 / 48.6 | 17.9 / 68.2 |

  - Better than epoch_12 at every horizon's median on all four levels; three 90% values on 1-16 (+16, +32, +48; n 64) are slightly worse than epoch_12's. Against wm_step4 the median error is roughly halved up to +32 (1-17: +16 8.9 -> 2.9, +48 30.5 -> 17.9).
  - Held-out loss was still falling slowly at the end (0.024 -> 0.023 over the last three epochs): more epochs alone will give little more.
  - Decision time is about the same for the three models (offline closed loop on 1-14, median 145 / 151 / 154 ms).
  - It loads and plans in agent.py (simulate.py with `MODEL=`); the wind prior is not in it (gravity-v1).
- **48 epochs** (`runs/experiments/model_compare_v1/epoch_48/model/wm_model.pt`; same manifest and settings, 3 threads, 3158 s; never run in the game). Same waves, `epoch24_vs_epoch48.json`, median / 90%:

| Level | +4 | +8 | +16 | +32 | +48 |
|---|---|---|---|---|---|
| 1-14 | 0.2 / 1.7 | 0.8 / 4.0 | 2.7 / 15.9 | 11.9 / 44.5 | 21.8 / 73.4 |
| 1-15 | 0.2 / 1.4 | 0.7 / 4.7 | 1.7 / 11.8 | 5.6 / 46.3 | 10.6 / 93.0 |
| 1-16 | 0.3 / 1.4 | 0.8 / 6.6 | 2.8 / 22.1 | 8.9 / 55.3 | 22.3 / 93.2 |
| 1-17 | 0.3 / 1.9 | 0.9 / 7.4 | 2.0 / 18.1 | 6.7 / 43.0 | 14.7 / 62.1 |

  - Better than epoch_24 at every median on all four levels, and at every 90% value but two (1-14 +48: 73.2 -> 73.4; 1-16 +8: 5.9 -> 6.6, n 64).
  - Its own held-out loss was lowest at epochs 24-36 (0.013) and rose to 0.016 by epoch 48, while the position errors on later recordings kept falling. The loss includes the event heads (death / finish) and the members' uncertainty, so the rise may be those getting overconfident; not checked.

**The 1-17 stall does not reproduce offline** (`model_compare_v1/stall_1-17.py`, `stall_1-17.json`): from the stalled agent's recorded state (frame 208 of a 1200-frame wave, recorded hazards), with runs/wm_locked standing in for the game, the planner goes from T 187 to 71 (wm_step4) or 81 (epoch_12) in 500 frames and stays alive. So the stall is not a property of the planner's rules alone; in the game the plans there come out lower and shorter than predicted (the largest surprises: predicted y 17-22 units above the real one, x 18 beyond). They are not concentrated at slope or half tiles (13 of 69 surprises within 30 units of one). The cause in the game is not pinned down.

**--veto-members N** (`plan.planner._verdict` quorum, `Planner.veto_members`, default 1 = unchanged): the members that must touch within the veto window for a veto; with more than 1, a touch by fewer members is soft risk (their share). In the offline stall replay 1 and 3 gave the same run (same end, 2 vs 1 vetoed chosen plans): it is not what holds the agent there. Kept as an option for tight passes; no evidence yet that it helps.

## Planning without making the game wait: --async-wait danger (2026-11-13; simulated, not run live)

**Why.** In shots_1-15_check01 a decision took a median 86-119 ms and came every 6 frames: 16-21 ms of planning per frame against the game's own 12.6 ms, so 0.41-0.50x. --async-plan alone did not help on 1-17 (0.41x): any trigger other than a timed one dropped the plan made ahead and waited for a new one (2298 dropped, 489 used).

**Rule** (`agent.py`, default with `--async-plan`; `--async-wait all` is the old behaviour):
- The game waits for a decision only on a **danger** trigger and when the agent has no plan (`WAIT_KINDS`).
- Any other trigger (surprise, support, missed, interval, move end) starts a background plan for `--async-lead` frames later, from the state the committed plan's next actions are imagined to reach. The committed plan plays on meanwhile. Triggers that fire while it is being made are absorbed (`ahead_absorbed` in the wave event).
- At its frame the plan is adopted when the real start is within the surprise tube at that lead (the error the surprise rule itself allows a prediction of that age; 3 units before) and `ASYNC_DV`. Otherwise another background plan starts; after `ASYNC_MISSES` = 2 misses in a row the next decision waits, from the real state.
- Cost: the reaction to a surprise comes `--async-lead` frames (8 = 133 ms) later.

**Simulation** (`runs/experiments/async_plan_v1/simulate.py`, env `MODES=sync,all,danger`, `LEAD`, `DATA`, `LEVEL_NAME`; game frame simulated as 12 ms; the stand-in physics is runs/wm_locked, so outcomes are not comparable, only speed):

| Level | sync | wait all | wait danger | wait danger, lead 12 |
|---|---|---|---|---|
| 1-14 (400 f) | 0.53x | 0.74x | 0.76x | 0.81x |
| 1-15 | 0.42x | 0.42x | 0.67x | 0.76x |
| 1-17 | 0.31x | 0.32x | 0.66x | 0.75x |

Python per frame fell to a median 3.4 ms (1-14); what is left is waiting for a plan that is not ready at its frame (a plan takes about 100 ms, 8 frames of game are about 125 ms) and the decisions that still wait. In the 1-17 simulation 39 of 49 background plans missed their start (the stand-in model's wall moves differ from the planner's); in the real 1-17 run the followed error at +8 was a median 3.2 units against a tube of about 7.5.

## The progress line in the viewer, and the 1-17 stall (2026-11-13)

**Progress line (viz/viewer.py, P key).** The agent's progress is the time field T (plan/progress.py): frames to the goal from every cell, by Dijkstra on realpath's move graph. It is a field, not a line. The line it implies is drawn:
- `plan.progress.best_times(field)` is the per-cell minimum of T_support / T_rise / T_fall. `descent_path` walks from the agent's cell to the neighbour with the lowest value until the goal, leg by leg through the keys. Each step strictly lowers the value, because every state has a move to a state lower by the move's frames.
- agent.py sends it in the level message's `"progress"` (float16, dropped after `"words"` when too big) and saves `runs/<name>/fields/<level>.npz` for replays.
- P cycles through three views: the line (cyan, ticks every 60 frames to go) plus the field's route from the spawn (orange); then also a heat map; then off. The panel shows the frames to the goal from the agent.
- For older runs, `runs/experiments/progress_view_v1/build_fields.py RUN LEVEL...` builds approximate offline fields. They have no kill-box outside cells or sealed pockets.

**What shots_1-15_check01 actually shows** (sync, 416 waves):
- **1-15:** 52 finishes in 123 waves, the first at wave 14. 47 were under A+ (10.0 s); the best was 8.32 s. All 70 deaths were shots.
- **1-16:** 25 finishes in 29 waves, all under A+ (9.0 s); the best was 5.10 s.
- **1-17:** 264 waves and 0 finishes. 259 waves timed out after 1200 frames, and from wave 6 on every wave is the same frame for frame: the field reaches T = 149 and stops at (72, -434).
- The async run `async_1-15_check01` played only 1-17: 19 waves, 0 finishes, speed 0.41x against 0.50x synchronous. 489 ahead plans were used and 2298 dropped, mostly to surprise and support triggers near walls.

**1-17 causes (found, not fixed).**
- **h5 is a saw circling a block:** centre about (30, -350), radius about 60, period 100 frames.
- **First life:** `_BoxRun` needs two periods (200 frames) to verify a period. Until then the saw is extrapolated linearly, which is wrong for an orbit (86° of turn in 24 frames). Its predicted clearance jumped between decisions (103 -> 16 -> 101 -> 41 units). All 5 deaths came in reflex mode with every candidate vetoed 4-12 frames ahead: the agent had been cornered.
- **The stall** (trusted recording, so the saw is known): the chosen plans' T at the end (median 186) is above T at the start (176). 432 rejected alternatives were "no progress", and the progress-making ones "hit h5 at frame 8-16".
  - The hard veto is contact by ANY of the 5 members within 16 frames, so tight passes near the saw are vetoed.
  - Wall moves in the right shaft surprise often ("support: on a wall", +1 error 1-3 units), and the agent dithers: 82 switches against 116 kept.
  - Waves are deterministic and a timeout charges nothing, so the same 1200 frames repeat forever.

## Planning in parallel: agent.py --async-plan (2026-11-12; simulated, not run live)

**Why.** A decision costs about 100 ms, the game waits for it, and one comes about every 7 frames. The user's 1-14 / 1-15 run played at 0.56x real time.
- Fewer candidates barely helps. On 30 recorded 1-15 decisions, the median was 101.5 ms at 8/16/16 (stage-1 keep / branch / final) and 79 ms at 4/8/6. The cost is in about 40 model steps per decision, each spent mostly in per-step Python and numpy work.
- One pure win was committed (adacac5): the key-pickup path returns at once on a level without keys. Decisions stay identical and the median fell from 114.7 to 101.5 ms.
- A GPU would not help: the model is small, and the time goes to Python.

**Design** (`plan/worker.py`, `agent.py`):
- **The planner process.** With `--async-plan` the planner, with its memories, runs in a child process (`PlannerProcess`, spawn).
  - A background thread was tried first and dropped. The GIL made its plans twice as slow (186 vs 92 ms), and the waits made the game slower than synchronous play.
  - Every planner call goes through one ordered pipe: observe, played, the frame's level changes, finished, new_wave and plans. Observations sent while a plan is being made therefore wait for it.
  - The game loop keeps its own Planner for geometry only (support, bounds, terrain).
- **Planning ahead.**
  - When the committed plan's next timed decision (`Committer.due`: interval, move end, short) is a whole number of model steps away and at most `--async-lead` frames (default 8), the child starts it.
  - It plans from the state the committed actions are imagined to reach (`Planner.predict_start`: the ensemble member nearest the mean) with this frame's future shifted, and with `plan(defer=True)`: previous plan, route use, route trial, surprise / move / failure memories wait for `Plan.commit_fn`.
- **At the target frame.** The plan is used only if:
  - no trigger other than a timed one fired;
  - the real start is within `MIN_TUBE` (3 units) and `ASYNC_DV` (0.5 units/frame) of the predicted one.
  
  Its path and predictions are then moved onto the real start. Otherwise it is discarded and the decision is made synchronously as before.
- **Calling a plan off.** An earlier trigger, a death, a finish, a wave end or a level change calls it off: `plan.planner.CANCEL` makes `imagine()` stop at its next model step.
- **Limits.** Not with `--eval-first`, explore agents, `--search`, `--demo`, or more than one planned agent. `--learn` model swaps are not forwarded to the child.

**Simulated** (`runs/experiments/async_plan_v1/simulate.py`): the Runner is driven with a recorded wave's hazards; `runs/wm_locked` stands in for the game's physics; the game's frame is slept as 12 ms.

| Level, setting | Speed | Runner time per frame, p90 | Decisions planned ahead / total |
|---|---|---|---|
| 1-14, synchronous | 0.51x | 81 ms | – |
| 1-14, async, replan 8 | 0.70x | 11 ms | 36/42 |
| 1-14, async, replan 12 + lead 12 | 0.78-0.80x | 6-7 ms | 21-24/29 |
| 1-15, async | about 0.46-0.48x | – | 7-15 |

- On 1-14 the agent ends at the same place with the same trigger mix in both modes.
- On 1-15 the stand-in model disagrees with the planner's on jumps, so many 12-frame predictions are off by more than 3 units and those decisions stay synchronous. The real 1-15 followed errors are a median of 1.8 units at +8 and 3.4 at +16.
- **Also (2eaec20):** the per-frame danger check builds only the deadly boxes (`future_inputs(deadly_only=True)`), with boxes identical to before. Simulated 1-14 async went from 0.70x to 0.72x, with a runner p90 of 9 ms. This applies to synchronous play too.
- Tests: `tests/test_async_plan.py` covers due, deferred effects, ordering in the child, commit and cancel. Full suite: 376 passed, excluding the known `test_cling_rule_fast_falls_are_not_support` failure.
- Not run against the game.

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
