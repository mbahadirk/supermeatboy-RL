# Colab world-model handoff — 2026-11-24

The user will continue Colab preparation with another agent. This handoff does not start
training/export or change data exclusions. Current authorized scope is **all valid collected
experience outside chapter 4**, including the formerly held-out non-chapter-4 levels.
Do not silently keep the old exclusion list or silently downsample the requested data.

## Worktrees and latest implementation

- Main: repository root. Shared game/projectiles.py owns the existing raw sh/shx parser,
  observed Launcher period contract and current-flight detector. game.level re-exports
  parse_shx; mpc.future uses the shared observation helpers. No native timer guess was added.
- ONLYPPO: .worktrees/onlyppo, branch ONLYPPO (uppercase). It has no WM/MPC/action planner
  packages. PPO receives 26 appended source/phase fields (524 -> 550 with --shape-inputs),
  enabled by default (--projectile-inputs / --no-projectile-inputs). Native first-frame shot
  velocity is consumed. Old weights widen with zero added columns; originals stay intact.
- Both copies of game/projectiles.py must remain identical. Raw sh/shx units/alignment and
  the existing world-model Collector ROW/frame formats were not changed. A PPO550 vector
  is NOT a world-model664 vector; use raw states/world messages to construct WM features.
- Verification before handoff: ONLYPPO106 tests passed; main41 relevant tests passed;
  actual1-20 replay601 frames matched main's observed launcher facts, four verified119-frame
  periods. New inputs have NOT been live-tested for completion improvement.

## Local evidence / checkpoints (not uploaded to git)

- ONLYPPO runs/aplus_v2_batch2048 is the last inspected continuing realtest run (batch2048).
  Recommended new run name after enabling clock inputs: aplus_v2_projectiles. Its existence
  and the currently active run must be checked, not assumed from a suggested command.
- Other preserved PPO checkpoints: runs/aplus_v2, ppo_recovery_1_9_20261124_a,
  aplus_v2_1_14_visibility_fix, aplus_v2_1_14_continue, under ONLYPPO runs/.
- Main runs/experiments/{human_v3_gpu,human_v4_gpu,human_v4p_gpu}/wm_model.pt are baselines.
  All step4,5 members,hidden256,layers3,gravity-v1. Feature encodings: tile-shapes-v1/419,
  tile-accel-v1/664,tile-plat-v1/664 respectively. Their metadata lists chapter4 training
  on2,2,6 levels: chapter4 is not equally unseen for all historical models.
- Runs, original game assets, weights, raw replay arrays and native saves remain local.
  A fresh git clone does not contain them. Source-adjacent analysis reports explain which
  artifacts are local-only. No model snapshot .pt is staged in this handoff.

## Required preparation before Colab

1. Ask the user to save/stop any active trainer before freezing its files. Preserve originals.
2. Audit actual collection folders. Local analysis/colab_all_data_20261124/inventory.json is
   a broad scan, NOT a validated dataset. It includes many test/tmp fixtures before filtering;
   85 collected-folder candidates/12818 shards is still pre-quality/pre-dedup inventory.
3. Implement and verify native_frames.jsonl.gz -> WM training-shard conversion. This does
   not exist yet. Preserve state/action chronology, native dt, held mirroring and terminal
   evidence. Do not invent missing rewards or terminal states. Handle repeated run names,
   wave counters restarting on level changes and concatenated gzip members correctly.
4. Apply this experiment's chapter4 exclusion, keeping prior eval_levels.json semantics
   for old models/evidence. Current wm.train/build still hard-exclude the old held-out levels;
   the requested experiment needs an explicit scoped alternative, not metadata relabeling.
5. Freeze train/validation trace/attempt identities; avoid adjacent-frame leakage and label
   overlaps with historical training. Include an actual coverage report and excluded reasons.
6. Plan RAM before uncapped feature export. Default max_samples500000 is not all data;
   build applies this cap after feature construction, so it does not bound preparation RAM.
   Implement streaming/chunking if every requested transition does not fit. No feature bundle
   or Colab code archive has been prepared yet.
7. Add durable intermediate training/optimizer checkpoints before a long Colab session:
   current fit saves only final weights. --init currently skips base --epochs; it is not a
   conventional resume command. Strict init also requires matching dataset manifests.
8. Proposed first fresh-model configuration: step4,tile-plat-v1,gravity-v1,5 members,
   hidden256,layers3,seed0; match human_v4p as the nearest baseline. Once prepared, use
   --load-features on Colab, CUDA,batch4096,48epochs, a new Drive output folder. No training
   was launched and no result/size/runtime guarantee has been established.
9. Prepare an external fixed comparison evaluator. Current wm.compare_steps --manifest
   rejects historical models with different manifests; --legacy-split also rejects strict
   checkpoints. Do not spoof manifest lineage. Each model must use its own feature/prior
   protocol on identical fixed evaluation states/actions, with training overlap explicitly
   labeled. Compare4/8/16/32/48-frame position errors, events as supported, and later the same
   gameplay budgets. Do not equate loss across different validation sets with improvement.

Isolated staged-snapshot verification: ONLYPPO106 passed; main39 passed and2 tests
skipped because their local recorded assets are intentionally not in git. Working-tree
main checks with local assets previously passed41. No commit relies on unrelated dirty
main planner/performance hunks.

## Scope of commits being handed off

ONLYPPO implementation/tests/notes and main's shared observations/monitor/notes are committed.
Main also contains pre-existing unrelated dirty planner/carrier/performance work, intentionally
left out of this commit request's session scope. Inspect git status before touching it; do not
reset, discard or assume that work is included in the remote commit. Staged main files may
contain only selected observation hunks while other hunks remain in the working tree.
