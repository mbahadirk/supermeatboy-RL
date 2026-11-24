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

2026-11-20, ONLYPPO --realtest:

Implemented in the separate `.worktrees/onlyppo` worktree (ONLYPPO branch), using `realgame.py --realtest`.
Every native A+ triggers save/next-level after PPO wave processing; other results retry. Native HUD time
and chapter par determine grade, with native-save backup and checked go/finish handoff. No world model,
action planner or reward change. Branch PROJECT_STATE.md/PROJECT_HISTORY.md contain usage and limits.
Validation: 21 tests plus syntax/compilation and PPO/reward AST parity; live native progression unverified.
Main's existing source/run/model changes are preserved; no commit/push performed for this addition.

2026-11-19, implemented authorized fixes after online_check21 audit: ready handoff is judged before the
CURRENT plan's proof expiry; unusable replacements still stop at the limit and mandatory events retain
priority. Both plan producers carry owned, time-aligned support_platforms context through async transport;
Committer uses it on late/translated paths with existing private-key semantics. No safety threshold or
checkpoint change. Matched valid fixture: cancel2/commit0 -> cancel0/commit1. Two unchanged recorded A+
paths: support triggers84->32,55 old mismatches resolve,3 new remain exposed. No native FPS/finish claim,
no live/run/save/model/memory/ONLYPPO changes.4-2 boarding remains separate. Evidence/checks:
analysis/handoff_support_fix_20261119/report.md; final verification recorded in PROJECT_STATE.md.
Final full suite702 passed/one existing warning (363.36s); compilation/diff/ONLYPPO checks passed.
Prepared online_check22 command uses latest online_check21 checkpoint; it has not been executed.

2026-11-19, bounded read-only online_check21 replay audit:844 complete lives across4-18/4-1/4-2/4-3/4-4.
4-18 retained two A+22.084s finishes at0.262/0.254x (old successful0.147x), still not real-time. Identified
own prior proof-expiry ordering regression: old proof expires before a ready boundary job can be taken;
deterministic ready fixture cancels both. Also Committer caches future path support against stationary
adoption-time bars;55/84 mismatches over two native successes resolve with time-aligned actual future bars.
Next fix: valid ready handoff before expiry, producer-owned future support trace; retain hard stop and genuine
support events.4-2 dies600/600,548 same h47#3 after forward boarding overshoot;4-3 succeeds8/13/one A+;
4-4 dies76/76. Scope/evidence: analysis/replay_check21_20261119/report.md. No production changes in this audit.

2026-11-19, committed/pushed working A+/carrier implementation as f5ef6eb before speed work. Native
online_check20 wave24 confirms first4-18 A+:22.35s vs24.5s, but151.84s wall/0.147x,205 decisions,
85.188s combined sync/ahead waits. check_s includes blocking route_triggers, so its53.233ms/frame is not
pure extra geometry cost. Performance implementation shares compact/cached kill profiles, batches terrain/
future-support queries, preserves periodic-search history while pruning impossible offsets, filters distant
transport trigger shapes, and tries a small station-only search with the unchanged all-member16f proof.
Rejected small searches return to the full planner before adoption; obstacle/key/portal/proven-route/near-goal
cases retain full search. Same horizon/ensemble/safety/async policy. Native speed/finish under these performance
edits remains unverified. Details/check results: analysis/perf_check20_20261119/report.md and PROJECT_STATE.md.

The auto/events phase also kept --planners2 in a single-job schedule (zero speculative starts in the native
success). It now uses the parallel pool while retaining event waits; Plan.execution_limit prevents execution
or late adoption beyond a carrier proof and caps the primary forecast deadline. This changes timed-job
scheduling, not the mandatory danger/surprise/support waits. Native scheduling speed still needs measurement.
Final full regression692 passed/one existing warning in262.38s; compilation/diff/ONLYPPO checks passed.
Current performance edits are local after the baseline push. Native next-run command is prepared, not run,
at analysis/perf_check20_20261119/command.ps1; successful-run data and models remain unchanged.

2026-11-19, user reports first actual 4-18 completion with the anticipatory crossing implementation,
but game runs roughly0.1x. Asked to commit/push the current implementation before optimizing. Latest run
online_check20; quantitative completion/speed audit pending. Prior offline failure remains historical.

2026-11-19, implemented user-approved anticipatory carrier obstacle search, before16f station priority.
Geometry-derived far-side goal, bounded directional/held-jump/countersteer OR wall-climb/exit proposals;
all-member collision/support checks, unchanged safety thresholds/horizon, consumed cached controls,
feedback capped by proof endpoint. Real goal/proven route priority remains. Crossing intent survives
near-side return/discarded controls; actual ascent waypoint preserves far-side goal, actual settled far-side
support completes it. Solid pillar blocks the previously supposed direct-hop gap; ascent is explicitly
intermediate, not a completed crossing. Three/eight matched states choose early ascent, but coarse sequential
model feedback STILL fails the later wall exit. No native obstacle/4-18 completion/A+ proof, no speed claim
(additional search costs time). No live/run/save/published-model/memory/ONLYPPO writes. Evidence/limits:
analysis/carrier_crossing_20261119/report.md. Full regression679 passed/one existing warning (518.71s);
final-source carrier/crossing17 passed (10.37s), compilation/diff/ONLYPPO checks passed.

2026-11-19, read-only online_check19 obstacle audit: first43 completed lives,43 deaths/no finish. h23 lower
circle blocks continued attached travel; lives2/22/43 hit it nearx68..71 after logged future collisions were
repeatedly deferred by16f station and8f recovery. 56 relevant station records/six lives. Carrier priority
and delayed extended-search trigger suppress needed anticipatory crossing; short up3/6 proposals/finalist
position ranking do not establish a far-side exit. Recommended early transport-to-crossing task with
directional/longer jump+countersteer and timed far-side support, preserved until actual arrival. No production/
live/run/save/model/memory/ONLYPPO change. Evidence: analysis/carrier_check19_20261119/report.md.

2026-11-18, implemented the approved moving-carrier/return task, all-member short station keeping and
same-support evasive return validation; pending controls are consumed with their original deadline.
Observed return intent survives unsafe-control discard/pocket landings; native settled arrival/reset
ends it. Guarded short recovery is labelled unproved and fed back at4f; no safe candidate falls back to
normal safety. Goal/proven route/checked transfer priority preserved. New blue carrier mode/logs; confirmed
non-jumping rider rest drift no longer invents takeoff. Full regression672 passed/one warning (323.52s),
final focused148 passed/one warning (60.91s) after small intent/count fixes. Model probes still fail some projectile/fallback
cases: no native section/A+ success claimed. No live/run/save/model/memory/ONLYPPO writes. Next command
uses learned online_check18 model, fresh online_check19 and --wave-seconds30. Details: analysis/carrier_20261118/.

2026-11-18, online_check18 analysis only, bounded completed lives1..268/shards0..267: zero finishes,
153 boarding,76 descending,10 bottom-leg riders. Actual continued-contact predictions are now low-error
(+4f median0.144/0.463/0.413 by segment); false-ground diagnostic0/2301. Remaining failure is control
continuity: repeated jumping/pocket detours replace future returns as the carrier moves away. Max x65.26,
goal564.73; deaths upper151/lower66/shots48/unknown3. Need maintained moving-support/verified transfer task,
not more endpoint scoring patches. Also observed20s cap is below imported A+ par24.5s; recommend diagnostic
--wave-seconds30 while retaining grade target. Logged median game_speed0.261. No production/live/run/save/
model/memory/ONLYPPO changes. Detailed life155 and bounded evidence: `analysis/platform_check18_20261118/`.

2026-11-18, implemented the approved standable-surface correction at Terrain.support: below-foot solid
must have clearance at body-centre height. All 286 audited false-ground endpoints rejected, 6578 native
platform contacts retained; wall-side behavior unchanged. Added measured descending-bar carry, preserving
native contact offset and replacing learned carry once. On 42 descending +4f windows, median error
5.4633 -> 0.6965 units. Model-feedback f248..408 follows the platform bend/descent (ground samples 4/21 ->
17/21), not a native survival/section/A+ proof. Only two old golden labels changed ground->wall; other
old golden fields unchanged. Full final suite 662 passed/one existing warning in 382.97s.
No live/run/save/checkpoint/memory/ONLYPPO writes. Restart required.
Validation and limitations: `analysis/platform_surface_20261118/report.md`.

2026-11-18, read-only online_check17 audit: second session shards 19..107, 89 deaths/no finish; older first
session and later growing recording excluded. 52 boarding lives/max continuous 178f; native matched-control
platform +4f endpoint error median 0.2392 (1177 windows). New transfer guard stays active. Actual failures
shift toward running off ahead of the platform, hitting the x~-46 wall and falling/shots: shelf 41, lower
hazards 25, projectiles 17, other/unknown 6. Bar turns downward near f301. Remaining scoring loophole:
small tolerated wall penetration is mislabeled ground by foot probes, bypassing landing credit restriction
(286 suspicious chosen completions across 59 lives). No production/live/run/save/model/memory/ONLYPPO changes.
Evidence and precise session mapping: `analysis/platform_418_check17_20261118/report.md`.

2026-11-18, implemented the approved online_check16 platform correction: confirmed horizontal riders use
support displacement + integrated relative native vx instead of learned dx, avoiding double carry. Tile
collision sees the carried motion; walking off stops carry, takeoff/new landing/vertical bars stay learned.
Real observed departure history survives airborne replans and cannot be overwritten by plan adoption.
The same frozen final model on 42 recorded-control +4f contact windows improves median endpoint error
6.1501 -> 0.6578 units (p90 7.0036 -> 1.2992). Six decision probes and an already-boarded model-feedback
sequence check continuity; no native boarding/transfer/finish/A+ claim. No live game/agent/run/save/model/
memory or ONLYPPO changes. Full suite 658 passed/one existing warning (243.30s); focused platform checks
37 passed, async/commit 47 passed. Validation/reproduction: `analysis/platform_418_fix_20261118/report.md`.

2026-11-18, online_check16 disproves effectiveness of the previous platform-launch heuristic: 92 deaths,
zero finishes. Loaded context drops on the next airborne replan after a small upward hop. Measured native
platform carry in 68 stable transitions: median 1.8161 units/frame beyond native vx*dt. Eight actual-followed
+4f platform predictions miss median 6.9249 horizontal units. Ten lives touch the bar, longest stable ride
25 frames. Analysis only; no production/game/agent/run/save/model/memory/ONLYPPO changes. Proposed next fix
is measured support-relative carry plus persistent transfer context, verified through full boarding/riding
sequences, not more isolated grounded choices. Reproduction and limitations:
`analysis/platform_418_recheck_20261118/report.md`.

2026-11-18, small 4-18 platform correction: latest visit's 42 completed lives all die after leaving the
spawn before the bar arrives. Predicted distant wall touches were earning terminal progress without a
ground/platform landing. Grounded launches in moving-bar worlds now withhold that credit until a predicted
ground/platform endpoint, retaining neutral and braking finalists through pruning. Geometry/risk settings,
actual goals and verified routes remain unchanged; airborne starts are unchanged. Both online and starting
checkpoints improve the initial choices in matched offline cases; no real boarding/finish/A+ claim.
Synthetic waiting probes include elapsed stall costs. Full suite 653 passed/one existing warning (275.77s),
focused checks 70 passed (29.95s). No live game/agent or run/save/model/memory/ONLYPPO writes. Details and
the heuristic's wall-route tradeoff: `analysis/platform_418_20261118/report.md`.

2026-11-18, implemented the user-approved same-exit local crossing controller after online_check15 still
stalled on 4-17. Prefix credit now measures improvement from the current state; observe() owns real stall
history and async adoption cannot overwrite it. Local direct/wait alternatives must reach the same supported,
keyed exit with all members surviving until arrival; incomplete waits are not cheap completed crossings.
Adopted waits/controls are consumed, revalidated and fed back at most every 8f; normal safety/risk settings
and goal/proven-route priority remain. Mode/log/viewer integration included, no level-specific route or flag.
Across 87 recorded states from six lives: waits 37 -> 11, 61 complete crossings, no selected veto. The exact
human_v4p_gpu starting model also improved the two baseline waits among five checked cases. No new live
finish/A+/death-rate claim. Full suite 650 passed/one existing warning (199.64s); final focused checks 63
passed (8.24s). No game/agent/run/save/memory/ONLYPPO writes; report and hashes in
`analysis/passage_20261118/`. Initial crossing search has a measured computation cost; see report limits.

2026-11-18, online_check14 follow-up: first seven completed 4-17 lives still time out; loaded prefix credit
was too binary and cell-quantised to distinguish short movement from waiting. Stall history/prefix scoring
now use the same subcell field and proportional credit (actual 10f improvement still resets history).
The command's starting checkpoint now picks immediate movement at recorded f604/895; this is decision
evidence, not a live finish/A+. Risk/collision/hysteresis parameters stay unchanged.
User clarified paths disappear after 4-9. Read-only native probe observed shader/Present calls while the
old scene-layer capture hook was skipped; capture moved to common SetVertexShaderConstantF with lazy
device hooks and logged milestones/errors. Shared native test harness covers measured-camera drawing
without a scene callback. Full suite 639 passed/one old binary-credit expectation failed, then that
expectation was updated and 30 final targeted regressions passed. Details and exact limits:
`analysis/saw_recheck_20261118/report.md`. No live agent/model/run/save/memory or ONLYPPO writes.

2026-11-17, implemented 4-17 stall-credit correction: future end-of-plan progress no longer exempts an
executed wait prefix. `Planner.prefix_stall` limits progress/novelty credit to the next decision/first move
boundary, retains native key/portal/platform contexts and uses real prefixes for verified routes. Local and
spawned planners receive the runner's replan interval; collision/risk constants and existing daring behaviour
are unchanged. Three recorded postponement states now choose immediate walking instead of resetting a 16f
wait in an isolated same-checkpoint comparison. No live-game/A+ claim. Report and comparison artifacts:
`analysis/saw_stall_20261117/`; regression coverage: `tests/test_stall_prefix.py`.
Final full suite: **635 passed, 1 warning in 360.07s (0:06:00)** (existing pygame/pkg_resources warning).
Focused prefix/key tests: 30 passed; compilation and diff whitespace checks passed.

2026-11-17, 4-17 replay analysis (online_check13, no production edits): 33 finishes in 59 bounded completed
lives, best 12.534 s, par 12 s. Found both conservative near-pass pricing at static saws and repeated
`wait 16f` plans replanned after 8f while endpoint progress waives stall cost. Recommended first correction:
credit actual committed progress / retain a movement-start deadline, then calibrate local uncertainty
and safe-candidate A+ budgeting. Evidence and limitations: `analysis/saw_stall_20261117/report.md`.

2026-11-17, required-key fix after online_check12 diagnosis: shared `realpath.route_legs` now rejects
incomplete key routes so timed/looser modes can be tried; progress fallback retains required key stages
even if no route is found. All masks for grouped keys are required. Reactive homing beam histories are
excluded from progress obstacles in the live agent and offline field builder, while per-candidate beam
collision safety remains active. Wave-1/2 replay now has two legs, key_order=[[0]], spawn T=439.3 and a
route through the key. Five new regressions pass; details/results are in `analysis/key_target_20261117/`.
Final full suite: **624 passed, 1 warning in 354.67s (0:05:54)** (existing pygame/pkg_resources warning).
No game was run; actual key collection/finish still requires a new live run with the agent restarted.

2026-11-17, latest-run diagnosis: online_check12 / 4-12 detects the key correctly but loses its route stage
after wave 2. Reactive beam sweep histories block the conservative key route, and the existing unreachable-key
fallback accepts a direct goal leg before trying timed routing. The agent then stalls at a lock with keys=0.
Offline tracking reproduces the change; timed routing on the same inputs still includes the key. This was
the initial diagnosis; the subsequent fix is recorded above. See `analysis/key_target_20261117/report.md`.

Reactive-hazard continuation (2026-11-17, main, uncommitted): completed/reviewed Claude's candidate-dependent
homing-eye integration and observation additions. Fixed wait animation scheduling, missing-angle fallback,
legacy beam suppression, portal beam sweeps and ray-table occlusion edges. Offline 4-16 replay predicts all
141 deaths across 14,999 steps without false survived-step contacts; sampled no-eye decisions match HEAD.
Shrapnel memory offsets remain calibrated rather than statically decoded, so first observations can be
missing. No game was run and no runs/save/memory files were written. Per-file changes, measurements and
remaining limitations: `analysis/reactive_hazards_20261117/completion_report.md`; see also PROJECT_STATE.md.
Final verification: `619 passed, 1 warning in 297.47s (0:04:57)` in the full suite; focused tests 86 passed.
The warning is the existing pygame/pkg_resources deprecation. Native fake-memory checks include shrapnel
calibration; no live offset verification is implied.

Cross-branch update (2026-11-17): ONLYPPO perception parity was implemented separately in
`.worktrees/onlyppo`, commit `edb6781`. Current PPO observation/native geometry improvements were ported;
world models, MPC, action planners and online learners were excluded. Eight focused tests passed; live-game
verification remains pending. See that branch's PROJECT_STATE.md and PROJECT_HISTORY.md for the full record.

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

### How far hazards really kill (World 4 lava and fire)

- **Problem:** the map counted a piece as deadly within 6 units of its box (Meat Boy's half size), for
  every kind of piece.
  - On 4-6, lava kills about 19 units above its strip and 40+ units past its ends, and fire falls 17 units
    above and below.
  - So the floor between lava strips, the gaps between stacked fire pieces and the top of a lava pit looked
    open.
  - Routes went through them, and only 35% of the deaths on 4-6 were in cells the map called deadly.
- **Fix:** `hazard_reach.py` measures, per kind of piece, how far beyond its box agents die versus how
  close survivors get (along the piece's own axes, from recorded deaths: about 18,000 in 3 runs).
  - The result goes to `hazard_reach.json`. `realgame.hazard_boxes` grows each box by that much, so the
    agents' map, sealed pockets, routes, monitor and the planner's hazard check all use it.
  - Kinds without enough evidence keep 6 units, and a reach is never below 6.
  - On 4-6: deaths in deadly cells went from 35% to 50% and the right lava pit is sealed; survivors in
    deadly cells stay at 0.07%.
  - On 1-9: deaths in deadly cells went from 69% to 91%.
  - The deaths still left in open cells happen where survivors also pass all the time (shots, moving
    pieces): timing, not a deadly cell.
  - Re-measure after collecting on new worlds: `python smbrl/hazard_reach.py <run>/wm_data ...`.

### Leaving the level

- **Problem:** the real player dies off screen, but our agents skip the game's own check (their mode turns
  off its global death code). The script only killed agents that fell 60 units below the map.
  - On 4-6, 932 lives flew off the open left and right edges and were still "alive" thousands of units
    away.
  - On 1-12, lives went 294 units past the edge, came back and finished.
  - The agents' map also showed everything outside the edge as solid wall.
- **Fix:**
  - `realgame.js` kills an agent more than 10 units past the left, right or top edge of the map (60 below,
    as before).
  - The agents' map shows the outside as hazard, with no surfaces.
  - The game's exact rule was not read from the binary: Ghidra timed out with the training running.
- **Update:** the real rule was read later: dead outside the camera's view x 1.15, with the camera clamped
  to the level's camera bounds. Agents now die at that kill box (sent to Python as `kb`; the map, routes and
  safety check use it); the map-edge rule stays only for levels without camera bounds.
- **Also:** `hazard_reach.py` with a placeholder folder had overwritten the measured table with an empty
  one. It now refuses folders that do not exist and data without deaths, keeps kinds the new data does not
  cover, and `--all` measures every recorded run.

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

## 2026-11-17: A+ rewards and real-player save/progression mode

The user requested A+-focused online learning and `agent.py --realtest`, then explicitly selected
**save only A+ finishes and advance; retry non-A+ finishes**.

Previously planner captures left reward targets undefined (converted to zero for fitting), and native
recording/advancing required S. `wm/rewards.py` now defines versioned finish rewards: 5 ordinary, 15 A+,
using the chapter par and confirmed locked timing. Both frame-message and exact final-wave terminal rows
carry the reward. `wm.online` uses the existing /5 targets and gives successful A+ lives 2x sampling weight
within its own-data batch share; replay proportion, fresh validation and acceptance gates are preserved.
This is world-model fitting, not a new RL policy or a claim of measured A+ performance improvement.

`--realtest` selects this run's planned A+ result, completes recording, backs up the game's save, and invokes
the existing native S finish/record/continue mechanism using the actual current time rather than a historical
best. The game keeps an existing faster record. It suppresses duplicate requests, refuses incompatible modes,
and skips native saving/advancing if backup fails. It does not implicitly enable online learning.

Regression coverage is in `tests/test_aplus.py`, plus the existing agent, buffer, online, learner and first-life
evaluation suites. Native saving and result-screen/chapter progression still require an in-game smoke test;
no live game save was changed while implementing/testing this feature. Full contract: PROJECT_STATE.md's
"A+ online rewards and native --realtest progression" entry.

Validation: the focused six-file suite passed 104 tests; the final A+ suite, after two extra cases, passed
26 tests (these overlap, not 130 distinct tests). Python compilation and scoped diff whitespace checks passed.
