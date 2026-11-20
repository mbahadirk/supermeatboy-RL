# ONLYPPO project state — 2026-11-20

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
scripted `--test`. Every A+ ends the wave immediately, regardless of `--end-after`; PPO processes the
completed wave before the native finish/continue request. Other results retry the same level.

The grade uses the native HUD clock captured at the successful goal contact, compared with the extracted
`chapters.json` par. Frame/60 remains the existing PPO reward/console metric, not the realtest grading
clock: this branch uses variable steps. The native clock restarts each realtest wave. Unknown par/time
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
the added ones small; it needs further PPO training to use the new inputs. Widened checkpoints must be loaded
with the same flag; silently dropping their inputs is refused. Saved metadata records the observation size.

PPO reward values, PPO optimizer/update code, action space, S behaviour and ONLYPPO's variable-step timing are
preserved. A read-only timer hook supports observed shrapnel calibration; no time-step code patch was added.
Imported assets and previous checkpoints are ignored runtime data and are not copied into a fresh worktree.
Use your existing assets/run files when running it; the native game itself is never bundled.

## Scope of “same view”

Parity is with **main's PPO observations**, not with its world model/planner's extra inputs or predicted
futures. Raw laser phase/parameter and projectile velocity fields may be sent by the adapter, as on main;
PPO's existing input interface uses current hazard maps and position-derived nearest-hazard velocities.
Neither branch's PPO receives future laser trajectories or imagined states. Shrapnel is omitted until its
live memory layout has been measured, exactly as in the source adapter; no guessed layout was introduced.

## Validation and provenance

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
