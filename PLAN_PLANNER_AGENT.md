# Plan: an agent that knows what it does and why (replacing 100-agent PPO training)

Status: **draft for review** (2026-11-03). Background: `PROJECT_STATE.md`. Reference for the planning style:
PocketWorld (github.com/ZhouYinLong-lab/Pocket-World-Model):

- a world model that predicts structured kinematics;
- collision-aware categorical CEM (256+ candidates, horizon 16 to 48), checked against the real geometry;
- closed-loop replanning;
- imagined vs real rollouts shown side by side;
- no RL value function.

## What the user wants

1. **Decisions it can explain.** The agent decides by computing consequences, before acting:
   - where a jump will land;
   - where a wall jump will happen;
   - how to pass a saw or hazard without touching it.
2. **Visible reasoning.** All of this is drawn, so its behavior can be analyzed.
3. **It works on levels it never saw.**
   - It never jumps into something it knows will kill it.
   - No wall spam, no pointless loops.
4. **No more 100-agent PPO as the main system.**
5. **The route planner (path finder) is only a progress counter**, not a rail to follow or the reward
   shape. The objective needs a proper replacement.
6. **No glitches.** The out-of-world / invisible-wall areas must not be usable.

## Design (draft)

**Main loop:** a planning agent, one per Meat Boy, 1 by default.

- **Every decision:**
  1. Read the structured state.
  2. Generate candidate action sequences.
  3. Imagine their futures with the world model.
  4. Check each future against the real level geometry.
  5. Score the futures.
  6. Play the first action.
  7. Replan.
- **No policy network is trained.** What learns is the world model, from data on many levels.

**Pieces, and who owns what:**

1. **Shared game state (`smbrl/game/`).**
   - Moved out of `realgame.py` unchanged: Level, tiles, hazard boxes with the measured reach, sealed
     pockets, platforms, wind, keys.
   - Both the old trainer and the new agent import it.
2. **Movement: the world model (`wm/`, exists).** It predicts position, velocity, flags and state.
   - It generalizes for movement (World 4: fine).
   - **Its death output is not used for decisions**: on World 4 it was wrong everywhere.
3. **Safety: geometry, not learning (`smbrl/plan/safety.py`).** For every imagined frame it checks the
   body box (the 6-unit half size) against:
   - the hazard boxes at that frame's time, grown by the measured kill reach;
   - the level boundary (outside = death).

   Future hazard positions come from earlier waves of this level (exact with the time-step lock), or are
   extrapolated with a growing margin. Outputs per candidate:
   - first contact frame and which hazard;
   - the smallest clearance to each nearby hazard;
   - out-of-level.

   This generalizes, because the game gives every hazard's position and size.
4. **Progress counter (`smbrl/plan/progress.py`).**
   - A distance-to-goal field over the level: walls block, climbing a cell costs more, and keys split it
     into legs (goal through the keys in order).
   - Computed once per level, plus when tiles change. Looked up for thousands of points at once.
   - This is what "path finder as a progress counter" means: the route search's geometry becomes a
     number per cell. No rail and no tether.
5. **Objective (score of an imagined future):**
   - **Progress:** the drop of the remaining distance by the end of the plan, credited only while alive.
   - **Death:** from the geometric check, charged by when it would happen (earlier = worse). The
     pessimistic ensemble member decides.
   - **Clearance:** a small penalty for passing within a few units of a hazard (prefer the safe line).
   - **Landing:** where the plan ends — on the ground, on a wall (able to jump), or in the air over
     nothing safe.
   - **Repetition:** a penalty for plans that end where the agent already spent many frames without
     progress (wall spam, loops). A visit count per cell, this life.
   - **Uncertainty:** where the ensemble members disagree, the credit counts less.
   - **No hand-tuned shaping terms from the old reward** (hop cost, run bonus, explore bonus, stuck
     timer, tether).
6. **Planner (`smbrl/plan/planner.py`):**
   - **Search:** categorical CEM over macro-actions (an action held for 2, 4 or 8 frames), warm-started
     from the shifted previous plan. Seeded with a library of moves: runs, jumps of several lengths and
     delays, wall jumps, waits.
   - **Size:** 128 to 256 candidates, horizon 32 to 48 frames, 2 to 3 refits.
   - **Shortlist:** cheap passes use 1 ensemble member; the shortlist of about 16 is re-scored with all 5
     members and the full safety check (PocketWorld does the same).
7. **Explanation (`Plan` object), per decision:**
   - the chosen sequence and its predicted path (mean and spread);
   - the predicted landing points;
   - the wall contacts where a jump press follows (wall-jump points);
   - per nearby hazard, the smallest clearance and at which frame;
   - the death frame, if any;
   - the score parts;
   - the top rejected alternatives, each with its reason ("hits saw #3 at frame 12", "lands in lava",
     "no progress", "repeats a stuck spot").
8. **Runner (`smbrl/agent.py`):**
   - Uses `realgame.js` unchanged: same lockstep and waves.
   - Runs planning agents, logs every Plan and every real state, and collects world-model data
     (source = planner).
   - Has an **explore mode**: many agents with sticky random actions (as PocketWorld), only to collect
     world-model data on new levels. That is the only use of many agents.
9. **Viewer (separate process, UDP, never blocks the agent):**
   - the level, hazards now and their future ghosts;
   - the chosen plan with landing markers, wall-jump markers and clearance rings;
   - the rejected plans colored by reason;
   - imagined vs real: where the model said the agent would be 1, 4, 8 and 16 frames ago vs where it is;
   - the score breakdown.

   The same viewer replays a run's log afterwards.
10. **Evaluation:** a fixed protocol, with seen and held-out levels and the same budget.
    - finish rate and deaths;
    - **known deaths** (it died though its plan predicted safe: a model or geometry error) vs **chosen
      risks**;
    - wall-spam count;
    - time to finish;
    - prediction error.

## The invisible wall / out-of-world bug

**Observed by the user** (4-6, monitor): routes pass over the top of the tower where the real game does
not allow it, and agents hit an invisible wall there without dying.

**Measured from the 4-6 recordings** (`real_20261103-012157`):

- Agents do fly over the top block and survive; no stop at the map top.
- About 3,000 sudden upward stops with no tile there sit on the right side (cols 42 to 46, rows 12 to 23)
  and at a few points around the upper blocks.
- So there is collision geometry not on our map: solid pieces, or bounds we do not read.

**Unknown:** is it the game's screen or camera bounds (the real player dies off screen; our agents skip
that code), or solid non-lethal pieces?

**Plan:**

- Measure it in the game:
  - log the camera's visible world rectangle (`realgame.js` already captures the camera matrix);
  - log every collision of an agent with something that is not a tile, with the piece type it touched.
- Then:
  - **outside the visible rectangle** = out of level (death, and outside the planner's world);
  - **solid pieces** go on the map as solid.

## Order of work (each step: one coder brief, tests, one commit, checked before the next)

1. Move the shared game-state code into `smbrl/game/`, with no change in behavior (tests prove identical
   outputs).
2. `plan/progress.py` and `plan/safety.py`, with tests on recorded levels.
3. `plan/planner.py`: CEM with macro-actions and shortlist re-scoring, plus the `Plan` explanation;
   offline tests on recorded states.
4. `agent.py`: the runner, plan logs, world-model data, explore mode, and the invisible-wall
   measurement (game side: camera rectangle and non-tile collisions).
5. The viewer: live over UDP and as a replay of logs.
6. The evaluation protocol and script.
7. Collect on more worlds (user), retrain the world model, evaluate on held-out levels.

## Open questions for review

- Is geometric safety plus a learned movement model the right split, or should death stay learned too?
- Can a distance field with a 32 to 48 frame horizon avoid dead ends? Levels that need going away from the
  goal for a long time may need intermediate subgoals.
- Is the CEM budget realistic on this CPU? Measured: about 0.5 ms fixed cost per step, 16 candidates x 16
  steps take 18 ms; shortlist re-scoring needed.
- Should PPO stay at all (e.g. as a candidate source)? Its value was learned per level and does not
  generalize.

---

## Decisions after review (2026-11-03): these replace the draft where they differ

1. **Safety is geometric and checks the whole move.**
   - Each step's movement segment is tested against every hazard box, grown by the body size and the
     measured kill reach. Sampled frames alone miss thin hazards: a fall covers about 50 units per frame.
   - **Hard veto** for contact within the first 16 frames.
   - **Beyond 16 frames,** a soft cost, with the margin grown from the measured position error at that
     horizon (90%: 14 units at 16 frames, 38 at 32). It is counted once, not on top of the members' spread.
   - **The learned death output** is logged as a second opinion only, never a veto.
   - **Unexplained deaths:** every real death the geometry did not predict is logged with position, wave
     frame and nearby pieces. Waves repeat exactly, so that point becomes a hazard from then on.
   - **Before any live use,** `safety.py` is measured on the about 18,000 recorded deaths: the share it
     flags, and how often it flags survivors.
2. **The progress counter is a jump-aware cost-to-go field (V).**
   - One backward search per leg over `realpath.Planner`'s jump model (cell, rise used, sideways used,
     falling).
   - Cells it cannot reach get infinity, an explicit dead end.
   - V is looked up at supported points (ground or wall); an airborne plan end is scored at its predicted
     landing.
   - **Checked first on recorded finishes:** the share of 48-frame windows where V does not drop must stay
     under 5%.
   - **Failed moves raise their cost:** when planned progress fails repeatedly on a graph edge, that edge
     costs more and V is rebuilt ("wall jump at C failed 3 times"). This replaces the repetition penalty.
     There is no separate landing term.
3. **The planner is a beam search over named move primitives**, not CEM.
   - **Stage 1:** 64 first moves of 8 to 24 frames (direction, run, jump hold 3 to 22, delay, wall jump).
     Keep the best 8 by safety and V.
   - **Stage 2:** 16 second moves each, from the shared end states, up to 48 frames.
   - **Re-score:** the top 16 with all 5 members and the full safety check.
   - **Defaults:** horizon 48, replan every 2 frames, the previous plan kept as a candidate, about 100 to
     120 ms per decision.
   - Every plan has a readable name ("run, jump hold 10, wall jump back").
4. **PPO is out of the decision loop.** It stays only as a baseline in the evaluation.
5. **Explanations and surprises.**
   - Each Plan stores its predictions at +1, +4, +8 and +16 frames.
   - When those frames come, a "surprise" is logged if the position is off by more than a threshold, or if
     the agent died unexplained.
   - The chosen plan, landing and wall-jump markers are also drawn in the game picture, through the
     existing Direct3D overlay, besides the separate viewer.
6. **Data.** For world-model data, explore mode mixes:
   - sticky random actions;
   - random chains of library moves;
   - the planner's own runs with small random changes.

   Evaluation leaves one world out (train on Worlds 1 to 3, test on 4). Open question for the user: may
   agents be spawned at sampled standable cells (a position write)?
7. **Invisible wall: not from the camera rectangle.** The camera follows the leading agent, so its
   rectangle is not the level bounds. In order:
   - cluster the unexplained collisions offline and overlay every non-lethal piece (pieces outside the
     lethal list may be solid; check the other type flags);
   - find the player's off-screen death check and the camera clamp in the binary;
   - hook the collision function and log what agents touch.

**Order of work:**

1. Invisible-wall analysis.
2. `safety.py`, measured on recorded deaths.
3. V field, measured on recorded finishes.
4. Beam planner with the Plan object and surprise logging (offline).
5. Runner with the in-game overlay, viewer and log replay, and explore mode.
6. Evaluation protocol with the PPO baseline.
7. Retrain the world model on more worlds and evaluate on held-out levels.
