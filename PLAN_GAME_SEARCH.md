# Search in the real game: why, and the plan

Status: **draft for review** (2026-11-11). Background: `PROJECT_STATE.md`, `PLAN_PLANNER_ROUND2.md`.

Labels: **measured** (numbers from runs), **inferred** (from code or data, not checked in the game),
**unverified** (not checked at all).

## 1. What the last benchmark shows

Run `bench_20261110-032340` (88f1023, model `runs/wm_step4`), reported 2026-11-11. All **measured** from its logs.

| Level | Finishes / waves | Decisions | Decisions that replayed the best route | Median = best time |
|---|---|---|---|---|
| 1-5 | 10 / 10 | 528 | 465 (88%) | yes (6.70 s) |
| 1-8 | 17 / 17 | 464 | 421 (91%) | yes (3.43 s) |
| 1-14 | 40 / 40 | 4,100 | 3,960 (97%) | yes (13.18 s) |
| 1-13 | 5 / 11 | 1,306 | 354 (27%) | yes (11.22 s) |
| 1-11 | 0 / 10 | 1,958 | 0 | no finish |
| 1-18, 1-20 (held out) | 0 / 0 | 0 | never played | |

- The 100% levels are one finish replayed frame by frame. With the time-step lock a wave repeats exactly, so
  40 finishes on 1-14 are one sample, not 40.
- The held-out levels have never been played in any benchmark. Generalization, the project's goal, has no number.
- 1-11 has been 0% in all nine benchmarks.
- Waves per level are not equal (10 to 40 against 20 planned), so completion rates across runs do not compare.
- `bench_20261110-041541` (the run meant for `runs/wm_step4_planner`) has 3 waves, about one minute. The new model
  has not been evaluated in the game. Its report also names `runs/wm_step4` as the model, because the report prints
  the spec's model, not the one the run used.

## 2. Why the planner work keeps going in circles

1. **The simulator is wrong and the planner is patched to live with it.** The world model's median error is 3 to 7
   units at 16 frames (a tile is 20). It mispredicts the long held jump on 1-8 and cannot fall out of the map.
   DeathMemory, MoveMemory, FailureMemory, BestRoute, forgiveness, hysteresis and the three modes each repair one
   symptom of that on one level. `plan/planner.py` is 2,169 lines.
2. **Each repair moves another level.** 1-8 across the benchmarks: 9/12, 4/99, 34/47, 42/42, 54/56, 15/67, 17/17.
3. **The benchmark cannot tell a real gain from a replay** (section 1), so there is no signal to steer by.
4. **Source errors stay open while downstream rules are added:** kind-2 kill boxes that look recorded without the
   piece scale (1-11 thorns, **unverified**), crumble blocks counted as permanent support in the time field, slope
   decode, planner data without death and finish rows.
5. **Everything since step 4 is offline only.** The live loop depends on a one-hour manual run.

## 3. The decision proposed

**Use the game itself as the simulator.** With the lock, the same actions from the wave start give the same
states (**measured**: 0.0 units on 1-13 and 1-14; the best-route replay relies on it). The game runs up to 200 real
characters per wave. So a wave can test up to 200 action sequences with exact physics.

**Wave search:** every agent replays a known action prefix exactly, then branches with different moves. The states
the branches reach become new prefixes. The world model is not needed for correctness; later it can rank which
branches to try.

This keeps every fixed constraint: real game only, structured state, no clone, no images, PPO untouched,
switchable (a new mode; the planner behaves as before when it is off).

**What it does not give:** a first-try finish on an unseen level. It solves a level in some number of waves. First
try comes later, from a model and a policy trained on the search's data (steps 6 and 7). The two are measured
apart (section 4).

## 4. Metrics (fixed before any code)

Per level, from the run's logs:

| Metric | Meaning |
|---|---|
| first-try | did wave 1 finish (yes / no) |
| waves to first finish | waves played until the first finish; none within K = 20 is a failure |
| game frames to first finish | all agents' frames summed, so 1 agent x 20 waves and 100 agents x 1 wave compare |
| fresh completion | finishes / waves over waves that did not replay a recorded route |
| replay completion | the same over replay waves, in its own column |
| best time after K waves | seconds |
| distinct outcomes | distinct finish frames and distinct death places: the real sample size |

Trained levels and held-out levels (1-18, 1-20) get the same table, in separate blocks. Only the first K waves of
a level count.

## 5. Design of the wave search

- **Node:** an action list from the wave start, the frame it ends at, and the real state there (x, y, vx, vy,
  supported, keys). Stored per cell key (tile cell, supported, sign of vx and vy, keys held), as FailureMemory keys
  its states. A cell keeps the node that reaches it in the fewest frames.
- **Selection:** per wave, pick frontier nodes by the time field T (frames to the goal, `plan/progress.py`), less
  for nodes already expanded often or whose branches mostly died.
- **Expansion:** each picked node gets a share of the agents. Each replays the node's actions, then plays a chain
  of library moves (`plan/moves.py`), or a perturbed copy of a longer known sequence (`plan.demos.perturb`).
- **Insertion:** after the wave, every agent's states at move ends become nodes when they reach a new cell or
  reach a cell sooner. A state within a few frames of a death is not inserted.
- **Verification (the guard for shared world state):** a node is trusted only after an exact replay reproduced
  its state. Agent 0 always replays the current best node and its state is compared with the recorded one. A
  mismatch marks the node invalid and is logged. This is what catches agents changing the world for each other
  (crumble blocks, see 6.2).
- **After the first finish:** the same loop on the finishing sequence, keeping the fastest (perturbed copies).
- **Wave length:** a wave ends when every agent has played its prefix plus the branch length or is dead, as demo
  waves already end early (`DemoPlay.end_frame`).
- **Held-out levels:** the search may play them (it trains nothing). No world-model data, no memory, as today.
- **Logs:** `search.jsonl` (per wave: nodes picked, branches, new cells, best T, verification result) and the
  existing `states.jsonl`, `events.jsonl`, `times.jsonl`. `--wm-collect` shards mark the source `search`.

Reused as is: `realgame.js` (unchanged in the first version), the Level / hazards / time field, the lock check,
the telemetry and viewer, `plan/moves.py`, `plan.demos.perturb`.

## 6. Order of work

Each step is one coder brief with tests and one commit; the next starts after its check.

1. **Honest report on the existing logs** (no game needed). `plan/benchmark.py report` gains section 4's columns:
   first K waves only, fresh and replay apart, first-try, waves to first finish, distinct outcomes, and the model
   the run really used. Re-report the nine recorded runs: that is today's baseline.
   Done when: the table for `bench_20261110-032340` shows the replay share of section 1.
2. **Measure whether agents share crumble blocks** (read-only, recorded data first). Tile changes are one global
   layer (`realgame.js` tileChanges); keys are per agent, crumble blocks are not (**inferred**). From recorded
   waves with 40 explore agents on crumble levels: does a block break when the first agent touches it, for everyone?
   Done when: a yes / no with counts. If yes, 5's verification guard is required on those levels and per-agent
   crumble state becomes a later game-side question for the user.
3. **Wave search, first version** (`plan/search.py`, a mode of `agent.py`): section 5 without the world model.
   Tests offline against a fake script, as `tests/test_agent.py`.
   Done when: the tests pass and the command for the user is printed complete.
4. **Live comparison** (the user runs it): the search and the planner on the five levels and the two held-out
   ones, K = 20, section 4's table for both. 1-11 and the held-out levels decide.
5. **Source fixes the data needs:** death and finish rows in `agent.py`'s shards; crumble blocks in the time field;
   the kind-2 box scale (a read-only probe in the game).
6. **Retrain the world model on search data** (it covers deaths, finishes and far more states), evaluate on the
   held-out waves and levels. Then use it to rank branches and measure waves to first finish again.
7. **First try:** a policy trained from the search's finishes (PPO stays; search actions never count as PPO's own
   samples), measured by first-try on held-out levels. Then World 2 and on.

The planner gets no new rules until step 4's numbers exist.

## 7. Risks

| Risk | Likelihood | Handling |
|---|---|---|
| Agents change the world for each other (crumble blocks) | likely on 1-11, 1-20 (**inferred**) | step 2 measures; the verification guard rejects nodes that do not reproduce |
| Long prefixes make waves long (a prefix of 300 frames is replayed by every agent) | certain | measured in step 4 as game frames; branch from several depths per wave |
| Too many cells on large levels | unknown | the T field orders the frontier; the archive size is logged |
| 100+ agents slow the game | **measured** before: 200 PPO agents ran | per-frame cost logged; no planning per agent |
| The search finds routes through a bug (as 4-6 did) | possible | the user's exclusion list stays; finishes are checked by the goal distance as today |

## 8. Open questions for the user

1. Is section 4 the right definition of success: "finish an unseen level within K waves" now, "first try" as the
   later target?
2. How many agents may the search use on screen: 100, or the full 200?
3. Is freezing the planner (no new rules until step 4) acceptable?
