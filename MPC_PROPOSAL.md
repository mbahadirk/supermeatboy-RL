# Phase 5 + 6: MPC with the world model, and live planning view: proposal

Nothing in the existing code has been changed for this yet. The proposal says where MPC and the viewer plug
in, what they compute, and how they are measured. It also gives the numbers measured today that the design
rests on. With MPC and the viewer off, every run behaves exactly as now.

## What was measured first

### The time-step lock works (run `real_20261102-165351`, 3 h, levels 1-1 to 1-17)

The run had 647 waves and 17.3 million agent steps. The game's time step was **16.7 ms on every step**
(median, 99% and max all 16.7). Before the lock it was 36 ms median and up to 70 ms. So:

- Physics no longer depends on how fast Python answers. A slow Python frame only slows the wall clock;
  the game runs in slow motion and nothing else changes.
- **For MPC this changes the budget question.** The planner can take 30 or 100 ms per frame without
  making the physics wrong. The game just runs at half or a sixth of real speed. The 10 to 15 ms budget
  now matters for throughput (waves per hour), not for correctness. MPC is only allowed with the lock on.
- The planner knows every future step is 16.7 ms. The game also reports its real step every frame, and
  the planner checks that it stays 16.7.

### Moving hazards repeat exactly across waves

On 1-13 and 1-14, the only levels in the data with moving hazards, every hazard was at the same position
at the same wave frame in every wave: **0.0 units off** (`python -m wm.probes repeat`). On a level the agent
has played at least one wave of, future hazard positions can be **looked up** instead of predicted.
**Not checked yet:** moving platforms (no level in the data has them) and thrown shots (1-x has none).

### What a rollout costs (CPU only: this PyTorch build has no CUDA)

Five ensemble members, from a real 1-13 state, in ms per whole plan:

| candidates | steps ahead | now (`wm.rollout.rollout`, members one by one) | all members in one batched call |
|---|---|---|---|
| 16 | 8 (0.13 s) | 29 | **9** |
| 16 | 16 (0.27 s) | 59 | **18** |
| 32 | 16 | 65 | **25** |
| 32 | 32 (0.53 s) | 139 | **50** |
| 64 | 32 | 161 | 81 |
| 128 | 32 | 232 | 138 |
| 64 | 60 (1 s) | 301 | 143 |

Times are with 2 threads; 1 and 4 threads give about the same. Findings:

- **The cost grows with steps ahead, much less with candidates.** Each model step has about 0.5 ms of
  fixed cost: building the inputs (about a third) plus many small tensor operations. 16 candidates and 64
  candidates differ by only 2x.
- Batching the members is 2 to 3x faster than the existing rollout and gives the same numbers. It will be
  added next to the existing rollout, which stays as it is.
- At 60 frames per second, one second ahead is 60 model steps. The world model was trained on 33 ms steps
  (before the lock), so 30 two-frame steps are worth testing: half the cost, if the error allows (step 6).
- The PPO policy and value network take 0.5 ms for 64 rows, which is negligible.

## Where it plugs in (the existing system stays as it is)

| piece | today | change |
|---|---|---|
| `realgame.js` | lockstep, one reply per frame, actions per agent | **none** |
| `realgame.py` `Trainer.on_frame` | builds the PPO input, `ppo.act`, records the step for PPO | after `ppo.act`: if `--control` is not `ppo`, the planner may replace the action of the agents it controls |
| PPO trajectories | every acted step goes to `ppo.add` | steps whose action came from MPC never go to `ppo.add`; PPO does not update in MPC runs (only PPO-controlled agents train) |
| `wm/buffer.py` Collector | `wm_data/` shards | each shard records `source` (`ppo` / `mpc` / `mpc+ppo`) and `mode` (`train` / `eval`); MPC runs write to `wm_data_mpc/` |
| `wm/model.py` | 5 x 3x256 | `hidden` and `layers` become options (defaults 256 and 3, so saved models load unchanged) |
| `wm/rollout.py` | rollout per member | + `rollout_batched` (same results, 2 to 3x faster); old functions stay |
| `monitor.py` | in-process window | unchanged |

New code lives in its own folders:

```
smbrl/mpc/        controller.py   plan(): candidates -> rollouts -> scores -> first action, timings
                  candidates.py   PPO samples, PPO rollouts in the model, primitives, previous plan, search
                  score.py        progress, death, hazard contact, landing, uncertainty, PPO value
                  future.py       future hazard / platform positions: recorded timeline or extrapolated
                  checks.py       prediction vs reality: the ring of past predictions and their errors
                  report.py       evaluation tables from eval logs
smbrl/viz/        telemetry.py    non-blocking UDP sender (drops, never waits)
                  viewer.py       separate pygame process: level, candidates, prediction vs reality, stats
```

New `realgame.py` options; without them a run is today's run:

- `--control ppo|mpc|mpc+ppo`. Default `ppo` (unchanged).
  - `mpc`: the planner picks every action.
  - `mpc+ppo`: the planner picks, but PPO's action is used when the plan is not finished within
    `--mpc-budget-ms` or every candidate is too unsure. Each fallback is counted and shown, never hidden.
- `--mpc-candidates N`, `--mpc-horizon H`, `--mpc-every K` (replan every K frames, default 1),
  `--mpc-model <wm_model.pt>`.
- `--viz` sends telemetry to `127.0.0.1:47800`. Start the viewer with `python -m viz.viewer` (or `--viz`
  starts it). With no viewer running, the packets are simply dropped.
- `--eval N` runs N attempts per level with no PPO updates and no model saves, and writes
  `eval.jsonl` (see Evaluation).

Single agent: `--agents 1` already works. The world-model data stays diverse because the 100-agent PPO
collection runs keep going between MPC sessions (only one trainer can attach to the game at a time).

## The planner (one decision)

1. **State.** The agent's row (x, y, vx, vy, flags, movement state, jump key held), the level tiles with
   live changes, hazards, platforms and wind. These are the same inputs the world model trains on.
2. **Future of the level.** From `mpc/future.py`, for each of the next H frames:
   - **Recorded:** from earlier waves of this level, at the same wave frame. It is used only while the
     current positions match the recording within 1 unit; otherwise it switches to extrapolated.
   - **Extrapolated:** constant velocity, with an uncertainty radius that grows with time. Used on a new
     level's first wave, and whenever the recording does not match.
   - The rollouts see these future hazards and platforms as inputs. The score keeps the uncertainty radius
     as a safety margin, so an uncertain hazard future is not treated as known.
3. **Candidates** (`mpc/candidates.py`): the share of each kind is configurable.
   - **PPO:**
     - its most likely action and samples from its action probabilities at the current state;
     - **policy rollouts in the model**: at each predicted step the PPO input is rebuilt for the predicted
       state and the policy picks the next action. This is PPO's real contribution, but it costs more; it
       will be measured.
   - **Previous best plan**, shifted one frame (warm start), plus mutations of it: one action changed, the
     jump released earlier or later.
   - **Primitives:**
     - hold right / left, with and without run;
     - jump held 2 to 20 frames, then released;
     - run and jump;
     - wait;
     - wall jump (press while sliding, away from the wall, re-press);
     - combinations with a delay of 0 to 8 frames.
   - **Search:** a cross-entropy method over the 10 discrete actions. It keeps a per-step action
     distribution, samples sequences, keeps the best 10% and refits, for 1 to 2 rounds. It starts from
     the PPO and previous-plan distributions. This finds sequences nobody wrote down.
   - Sequences are built from blocks of 2 to 4 frames of the same action, which shrinks the search. The
     model still steps frame by frame.
4. **Rollouts.** All candidates through all 5 members (`rollout_batched`), H steps.
5. **Score per member** (`mpc/score.py`):
   - **Progress:** the route closeness field (`realpath.RewardPath.value`, the field PPO is paid by). Where
     the field does not reach, the walking distance to the goal (`Level.goal_dist`). Both go around walls,
     so the score looks beyond the horizon: a move away that opens a better way scores well when the
     field says so.
   - **Death:** the predicted death probability, accumulated over the steps.
   - **Hazard contact:** the distance from the predicted position to each future hazard box, with the
     member spread and the hazard's uncertainty radius as margin. This is a geometric check next to the
     model's death output.
   - **Landing:** where it ends on solid ground (stable), or on a wall (able to jump again), versus ending
     in the air above lava.
   - **Keys:** reaching a key moves the field to the next leg, the same as PPO.
   - **PPO value** at the predicted end state, discounted, when the agent is still alive.
6. **Ensemble and uncertainty.**
   - Score = mean over members minus κ x their standard deviation (κ ≈ 1), so plans the models disagree
     on lose.
   - Credit is trusted less after the step where the members' positions spread more than one tile (Phase
     4: spread is about half the real error).
   - Death uses the worst member, so an uncertain future counts as risky, not as safe.
7. **Act.** Execute the first action (or the first K with `--mpc-every K`), observe, replan.

The plan records how many candidates were evaluated, the time of each part (inputs, model, score), the
fallback reason if any, and the chosen sequence. It never reports a plan that did not run.

## Live view (Phase 6, built step by step with the planner)

- **Telemetry** (`viz/telemetry.py`): one UDP packet per decision, compressed JSON, a few KB.
  - The socket never blocks: when the send buffer is full the packet is dropped and counted.
  - The level's tiles are sent in chunks every 2 s, so a viewer started late still draws the level.
  - The trainer never waits on the viewer. Measured cost target: under 0.3 ms per frame.
- **Viewer** (`viz/viewer.py`): its own process (pygame, like the monitor). It draws only the latest packet.
  - **Main view:**
    - level tiles, the player, velocity arrow, movement state;
    - hazards now and where they will be (faded ghosts at +8 and +16 frames), platforms likewise;
    - route and keys.
  - **Candidate trajectories:**
    - green: the chosen one;
    - red: death probability above 0.3;
    - grey: the others;
    - dashed or faded: member spread above one tile.
    - Predicted landing points are marked. The top five list their score parts (progress, death, hazard,
      landing, uncertainty, value), which shows why the green one won.
  - **Prediction vs reality:**
    - the dot where the model said the agent would be now, predicted 1, 4, 8 and 16 frames ago, next to
      where it is;
    - the errors in units;
    - whether the predicted landing and death happened.
  - **Statistics:**
    - candidates, horizon;
    - MPC ms and world-model ms (also the 95th percentile);
    - fallbacks;
    - death probability and spread of the chosen plan;
    - the chosen action sequence;
    - predicted vs actual progress;
    - frames per second (how slow the game is running).

**Prediction vs reality: the exact way.** After a replan, the frames the agent really played differ from
the old plan, so "where the plan said it would be" mixes model error with a change of mind. The trainer
therefore also re-runs the model from the state 1, 4, 8 and 16 frames ago **with the actions really taken**.
That is 4 rows and 16 steps, every 4th frame by default, and its time is shown. This is the model's own
error. Both numbers are logged to `mpc_checks.csv` for the step 3 validation.

## World model: what has to come first, then the experiments

**Retrain on the locked data first.**
- The current model learned on 33 ms steps (16 to 70 ms). Every step is now 16.7 ms, a mostly unseen input.
- The new run has 17 million transitions on 15 levels. The model will be trained on a sample of them plus
  the old data, and the held-out evaluation levels are excluded.
- The Phase 4 tests will be repeated at 16.7 ms.

**Architecture comparison (step 7):**
- **Models compared:**
  - 3x256 (today), 3x512, 5x256, and 5x512 (estimated about an hour on this CPU);
  - all on the same samples, the same held-out waves, the same epochs and seed;
  - each also with self-correction.
- **Reported for each:**
  - error 1, 4, 8, 16 and 32 steps ahead;
  - deaths (AUC, recall);
  - landings (step and spot);
  - calibration: the share of true values within 2 predicted standard deviations, and the spread vs
    error correlation;
  - the batched latency at the MPC sizes above.
- The best model is the most reliable one within the time budget, not the biggest.

**Where the far-ahead error comes from:** each candidate cause gets a measurement before any fix.

| suspected cause | test |
|---|---|
| time step | done: locked; re-measure on the new data |
| capacity | the 4 architectures; if 3x512 does not beat 3x256 on 8 to 32 steps, capacity is not it |
| compounding errors | teacher-forced (each step from the real state) vs free-running error per step; self-correction share |
| hidden state | add the previous step's state and action as extra inputs. If one-step error on the cases that go wrong (jump starts, wall jumps, landings) drops clearly, the current state misses something, and only then is a GRU worth trying. Without such a drop, no recurrent part. |
| data diversity | train on levels A, test on levels B (never seen) vs on held-out waves of A |
| long-horizon training | self-correction with 8 vs 16 steps |
| collisions / landings | error split by situation: free fall, landing frame, wall contact, against a hazard |

## Evaluation (step 8)

- **Held-out levels**, never used for PPO training or world-model training: **1-18, 1-19, 1-20, 2-5,
  2-6, 2-7.** No run has played them so far.
  - They are written to `eval_levels.json`.
  - `wm.train` leaves them out.
  - `realgame.py` refuses PPO updates and world-model collection on them unless `--eval` is set.
- **Seen levels** for comparison: 1-4, 1-9, 1-13 (moving hazards), 1-15.
- **For each level and controller** (PPO, MPC, MPC+PPO), N attempts with the same model files. Recorded:
  - finished, died, best progress, time to finish;
  - attempts until the first finish;
  - frames and replans;
  - planning ms (median and 95%), fallbacks;
  - prediction error at 1, 4, 8 and 16 frames.
- `python -m mpc.report runs/<eval runs>` makes one table per level, seen vs held-out.
- **Reproducibility:**
  - the game is deterministic per wave with the lock (hazards repeat);
  - PPO sampling and the planner's random search use a fixed seed per attempt;
  - the model files and options are saved with the results.

## Order of work: each step is committed on its own and reported before the next

1. *(done)* Integration points, lock check, hazard repeat, rollout cost. This document.
2. **Minimal MPC, single agent:**
   - retrain the world model on the locked data;
   - `rollout_batched`;
   - `--control mpc`;
   - candidates: primitives, PPO samples and the previous plan;
   - score: progress, death, uncertainty and value;
   - plans logged to file.
3. **Validate:** predicted vs real trajectories from the step 2 runs (`mpc_checks.csv`, offline report).
4. **Viewer:** telemetry, level, candidates in colour.
5. **Viewer:** prediction vs reality and the statistics panel.
6. **Benchmark:**
   - horizon 8, 16, 32, 60 and 30 two-frame steps;
   - candidates 16 to 128;
   - CEM and PPO rollouts;
   - each measured by cost, error and level outcome.
7. **World-model architecture and hidden-state experiments.**
8. **Held-out evaluation:** PPO vs MPC vs MPC+PPO.

## Open questions for you

- **Held-out set:** is 1-18, 1-19, 1-20, 2-5, 2-6, 2-7 fine? Once chosen, nothing trains on them.
- **Moving platforms:** which level has them (the one in your earlier screenshot)? A few waves there would
  show whether platforms repeat like hazards do.

## Step 2 results: retrained world model and minimal single-agent MPC

### World model on the locked game (`runs/wm_locked`)

**Data and training:**
- Data: run `real_20261102-165351` (locked, levels 1-1 to 1-17) and the earlier run (variable steps,
  1-5 to 1-11).
- 7% of each attempt's steps plus every death and finish: 1.42 million transitions, every 5th attempt
  held out.
- 8 epochs, then self-correction on 8-step rollouts (35% share).
- No held-out evaluation level appears in the data (`eval_levels.json`).

**One step ahead** (held-out attempts): position error 0.10 median / 0.64 at 90%; deaths AUC 0.996,
recall 0.96.

**Multi-step** (`wm.evaluate`, held-out attempts of the locked run, 67,494 rollouts with the agents' real
actions):

| steps ahead (16.7 ms each) | new model, position median / 90% | old model (`wm_phase4mix`) on the same data |
|---|---|---|
| 1 | 0.11 / 0.43 | 0.50 / 2.26 |
| 4 | 0.59 / 2.02 | 2.71 / 11.6 |
| 8 | 1.47 / 4.98 | 7.25 / 30.6 |
| 16 | 3.70 / 13.9 | 19.7 / 80.3 |
| 32 | 8.86 / 38.1 | 53.3 / 177 |

- **Deaths within 32 steps:** AUC 0.955, recall 0.95 (old model: 0.81, 0.69).
- **First landing:** the step is right at the median (90%: off by 3), the spot is off by 2.3 units (90%:
  13.5).
- **Ensemble spread vs error:** rank correlation about 0.6. The spread is about 0.6x the real error, so the
  planner's rule of "2 x spread" is on the safe side.
- **Too cautious one step ahead:** 100% of true values fall within 2 predicted standard deviations.

### The planner (`mpc/`, `realgame.py --control mpc|mpc+ppo`)

**Built as proposed:**
- Future hazards come from the level's recording, or are extrapolated with a growing safety radius.
- Candidates: previous plan and mutations, PPO's top actions, and a library of moves.
- Score: progress on the route field, death (model plus geometric check), landing, key, PPO value, and the
  members' disagreement.
- Optional time budget (a plan cut short is scored on the steps it has and labelled so).
- `mpc+ppo` fallback.
- One JSON line per decision in `mpc_plans.jsonl`.

**Keeping PPO and the evaluation levels clean:**
- Trajectories where the planner picked any action never reach PPO.
- Planner data goes to `wm_data_mpc/`, labelled `source: mpc`.
- Held-out levels give no PPO updates and no world-model data.

**Checked without the game** (`python -m pytest smbrl/tests`, 9 tests):
- batched rollout == the existing rollout;
- vectorized route values == `RewardPath.value`;
- the hazard filter and all-at-once hazard distances change nothing;
- the PPO-only trainer path still sends every agent's steps to PPO;
- with `--control mpc`, the planned agent's steps are kept out of PPO and its data is labelled;
- a held-out level produces no data and no update.

**Decision cost, measured on recorded 1-13 states** (`python -m mpc.offline`), with the PPO critic value:

| candidates x frames ahead | median | 95% |
|---|---|---|
| 32 x 16 | 57 ms | 67 ms |
| 16 x 16 | 50 ms | 55 ms |
| 32 x 8 | 31 ms | 36 ms |

- **Where the time goes** (32 x 16):
  - rollout about 45 ms (network about 24 ms, model inputs 15 ms, physics and state updates 6 ms);
  - PPO value 8 to 10 ms;
  - scoring 3 to 6 ms.
- **The cost is mostly per step ahead**, so the horizon matters more than the candidate count.
- **Threads:** 2 to 4 are best; 6 is slower.
- **With the lock**, 57 ms per decision means the game runs at about a quarter of real speed and the
  physics are unchanged.
- **Not checked yet:**
  - the planner in the running game;
  - its decisions against what really happens (step 3).

### First live run (`runs/mpc_dev1`, one agent, `--control mpc`) and the fixes after it

- **1-1 (World 1, trained):** both attempts finished (2.72 s, 2.62 s) at 42 ms per decision.
- **4-9, 4-6 (World 4, never in the world model's data):**
  - 4-9: died at 88% of the route.
  - 4-6: one attempt died at the start; the next survived 20 s without getting anywhere.
- **Cause 1, the world model:**
  - Its predicted paths there are fine (1 to 3 units off).
  - Its death output says 13 to 94% per step while the agent stands on solid ground 40+ units from any
    hazard. So every candidate looked fatal (26 of 30 replayed decisions on 4-6) and the planner could not
    choose.
  - It never saw World 4's hazards. **Fix: data from more worlds** (World 2 to 5 levels, held-out levels
    excluded), then retrain.
- **Cause 2, a planner bug:**
  - The geometric hazard check widened its margin by twice the members' spread, although it already checks
    every member's own position and keeps the worst. With the members 19 units apart, a hazard 14 to 55
    units away counted as a sure hit.
  - **Fixed:** margin = body size + the hazard's own uncertainty radius.
- **Death is now charged by when it is predicted:** the next step costs the full weight, the end of the
  horizon half of it. So when everything looks fatal, the latest death wins and the planner still moves.
- **When every candidate is ≥ 90% fatal:**
  - every such decision is counted in the per-attempt summary;
  - in `mpc+ppo` mode PPO decides instead (fallback "all fatal").
