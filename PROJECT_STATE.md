# Super Meat Boy RL: the state of the project (2026-11-03; planner step 4 added 2026-11-08)

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
- **Hazards on the map** use a measured kill reach per piece type, so lava and fire no longer leave open gaps.
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
| 4. World model + planner | `wm/`, `mpc/`, `hazard_reach.py`, `hazard_reach.json`, `eval_levels.json`, `tests/test_mpc.py` | live, switched off by default (`--control ppo`) |
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
- which lethal pieces are decoration (by palette name);
- the walking distance to the goal (BFS).

**Hazards** (`hazard_boxes`):
- every lethal piece becomes a rotated box;
- each box is grown by the measured kill reach of its kind (`hazard_reach.json`, section 5.2).

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

### 5.2 Hazard kill reach (`hazard_reach.py`, since c27ff6d)

**Problem.**
- The map treated a piece as deadly within 6 units of its box (Meat Boy's half size), for every kind.
- Measured on 4-6, lava kills about 19 units above its strip and 40+ units past its ends. Fire falls kill
  17 units above and below.
- So the map showed open cells inside lava and fire, and routes went through them.

**Fix.**
- Per kind of piece (its palette name), it measures how far beyond its box agents die (deaths are put on
  the nearest real hazard), against how close survivors get.
- A reach is accepted at the first of p90, p75 or p50 of the deaths that fewer than 1% of survivors fall
  inside. It is never below 6, and kinds with too few deaths keep 6.
- Boxes are grown by that reach in one place (`hazard_boxes`), so the agents' map, sealed pockets, routes,
  monitor and the planner's hazard check all agree.
- `python smbrl/hazard_reach.py --all` re-measures from every recorded run. It refuses missing folders,
  never writes an empty table, and keeps kinds the new data does not cover.

**Measured** (from about 18,000 recorded deaths):

| Kind | Kill reach x / y (units beyond the box) |
|---|---|
| lava | 42.7 / 19.2 |
| fire falls ("forest - river copy") | 6.0 / 16.6 |
| saw | 8.6 / 8.4 |
| ammo1 | 19.1 / 27.2 |
| spinner2 | 6.0 / 60 (capped) |

**Effect** (share of deaths that fall in cells the map marks deadly):

| Level | Before | After |
|---|---|---|
| 4-6 | 51% | 67% |
| 1-9 | 69% | 91% |

Survivors in deadly cells stay below 0.2%.

The deaths still left in open cells happen where survivors also pass often (flying shots, moving pieces):
that is timing, not a deadly cell.

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
  `--move-memory smbrl/move_memory.json` (or `none`), `--hysteresis 3` (frames, since step 4), `--threads 2`,
  `--seed 0`.
- Stop: Ctrl+C or a file `runs/<name>/STOP`.

**Per level:** the Level with its kill box, hazard boxes with measured reach, sealed pockets, the progress
field V (outside cells closed), FutureWorld and the planner (one per run, `Planner.set_level` per level). V is
rebuilt between waves when moving hazards are seen in new places, the kill box changes or a platform reaches
new cells (at most 4 times per level). V is built on the level's start tiles, as the routes are; tile changes
(crumble blocks, locks) reach the planner's model inputs and support geometry every frame (`Planner.tiles`).

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
  body's vertical velocity. V stays for display and the failure memory.
- **Score** (`plan/planner.py`): estimated frames to the goal = frames used + T(end) + penalty frames (risk 50 per
  unit, failure 5, the point-of-no-return charge); lower is better; no uncertainty factor. Wall contact is support
  at any speed (**measured**: wall jumps work at every recorded falling speed).
- **Airborne completion** (final stage): a plan not landing by a majority of members is continued with keep /
  neutral / reverse for up to 90 frames and scored at its best landing; all die: risk.
- **MoveMemory** (`--move-memory smbrl/move_memory.json`): each death charges the move taken at the latest
  decision of the wave that had a safe alternative; `point of no return` events.
- **Known limits (measured)**: the world model stops falling about 6 units above the map's bottom edge (its tile
  input pads outside the map with solid), so falls out of the 1-14 pit are not predicted; completion costs
  19-102 ms median per decision at the 120 ms budget, over its 20 ms target. **Unverified**: everything in the game.

---

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
| kill reach | how far beyond its box a piece kills, per kind (`hazard_reach.json`) |
| lock / time-step lock | the patch that makes the game always take its fixed 16.67 ms step |
| world model | the learned predictor of the next state (`wm/`) |
| MPC / planner | tries action sequences in the world model and plays the first action of the best (`mpc/`) |
| held-out levels | levels nothing trains on, for the final test (`eval_levels.json`) |

## 13. Commands that work as written

```bash
python smbrl/realgame.py --agents 100 --end-after 5 --load smbrl/runs/real_20261103-012157 --wm-collect
```

```bash
python smbrl/hazard_reach.py --all
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
