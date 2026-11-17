# ONLYPPO project state — 2026-11-17

This branch runs the original PPO agents in the real game using `realgame.py`. Its current perception was
ported selectively from main. **There is no `wm/`, `mpc/`, `plan/`, `agent.py`, learned world model, action
planner, rollout engine, online learner, `--realtest` or new A+ reward logic in this port.**

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
