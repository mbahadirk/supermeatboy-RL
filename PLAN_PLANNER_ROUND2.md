# Planner round 2: what the first live run showed, why, and the plan

Status: **draft for review** (2026-11-03). Background: `PROJECT_STATE.md`, `PLAN_PLANNER_AGENT.md`.

**Run analysed:** `runs/plan_20261103-152013`.
- Levels 1-6 to 1-12; 1 planned agent plus 40 explore agents; 11,468 decisions.
- All numbers come from read-only analysis of its `plans.jsonl`, `states.jsonl` and `events.jsonl`.
- The analysis script is in the session scratchpad. Step 1 below makes it a repo tool.

## 1. What the user saw, what the logs show, and why

### 1.1 Slow game, high CPU

**Logs:**
- **Cost per decision:** 130 ms (stage 1: 27, stage 2: 43, final: 59).
- **Frequency:** one decision every 2 frames, so about 65 ms per game frame against 16.7 ms in real time. The game runs at about 1/4 speed, and the game waits for every decision (lockstep).
- **Stage 2 is cut short:** it gets only 40 to 60% of its imagined steps before its deadline (median 0.6). Two-move plans are ranked on half-imagined paths.

**Why:** the planner re-decides from scratch every 2 frames. Most of the budget re-decides what was already decided.

### 1.2 Changes its mind half-way, turns back, gives up a climb

**Logs:**
- **Plans keep changing.** The next 16 actions differ from the previous plan's in 49% of decisions (51% in the air). A plan is followed for 4 frames (median; 90%: 16).
- **Near ties.** In 32% of decisions the chosen plan beats the best rejected one by less than 0.1 (a tenth of a tile). A tiny change in the start state flips the choice.

**Why the scores are so close** (score = V(start) - V(end of the 48-frame window)):
- (a) **The unit is wrong.** V is in realpath route-cost units, not time. Waiting costs nothing, and very different plans can end on cells with the same V.
- (b) **In the air, V is optimistic.** `V_air` assumes the best jump state possible in that cell, whatever the actual speed. A body falling fast into a shaft gets the value of one that could still rise.
- (c) **Most plans end in the air.** 63% do, so most scores rest on that optimistic number.
- (d) **Bold plans get taxed.** The uncertainty factor cuts the progress of 45% of chosen plans, 28% by more than half. Long jumps and wall chains spread more across the ensemble, so they look worse than they are.

**The cost of dropping a plan:** jump height depends on how long the key is held. A plan replaced mid-jump releases the key early, so the agent does not reach where any plan predicted.

### 1.3 "It cannot compute where it will land"

Prediction error (units; 1 tile = 20):

| horizon | plan followed: median / 90% | plan replaced before then: median / 90% |
|---|---|---|
| +8 | 2.3 / 5.5 | 5.0 / 19.2 |
| +16 | 6.0 / 16.5 | 15.3 / 47.4 |
| +48 | (too few followed) | 72.1 / 179.4 |

**Most of the visible misses come from the plan being replaced, not from the model.**

The model's own error still matters:
- 6 units at 16 frames is a third of a tile, enough on narrow ledges.
- It is larger for jumps pressed against a wall: 9 to 11 units at +8.

**The surprise counter is misleading.** It compares positions with plans that were already abandoned, so its "position surprises" (for example 376 on 1-10) are mostly not surprises.

### 1.4 Plans wall jumps on slopes, falls back, gets stuck

**Logs:**
- Not seen in this run's levels: only 1 planned wall jump beside a slope in 1-6 to 1-12.
- The `wall-jump` move against full walls launched upward 44 times out of 44.

**Where it can come from, by design:**
- (a) **realpath** counts a slope cell as support, "a surface to stand on or push against" (`realpath.py:252`). So V believes a climb exists there.
- (b) **`plan/geometry.py`'s wall probe** is 1 unit beside the body at mid height. Against a slope's diagonal face, that point can fall inside the slope triangle and read as a wall.
- (c) **The world model** saw few slope contacts (World 1 data).

**Unknown, to be measured from recorded data (not assumed):** does the game allow wall-slide or wall-jump on a 45-degree slope face, and on which faces?

**Why it stays stuck:**
- `FailureMemory` only penalizes the same first move from the same cell. It fired 9 times in the whole run.
- V never changes, so the planner keeps believing in the slope climb.

### 1.5 Jumps over the princess and waits behind her

**Logs:**
- **1-6 wave 31:** it stood 30 units from the princess for 880 frames, choosing `wait 8f` every time (score 0.2, alternatives 0.2), until the wave timed out.
- **1-6 wave 1:** it never got within 73 units of her in 1200 frames.

**Why: touching the princess does not exist for the planner.**
- A plan is scored only by V at the end of its 48 frames.
- A plan that walks through her ends past her, with a higher V than standing next to her.
- Nothing rewards finishing, or finishing sooner.

### 1.6 1-11: looks for short steps where a long jump is needed, and dies

**Logs:**
- **Same death every wave.** 19 waves, 19 deaths; 18 on the same hazard (h12) at about (75, -435). The geometry explained all of them.
- **No progress before the fall.** About 60 frames before death, every candidate has "no progress" (V rises from 24.5 to 33 to 65).
- **The fall.** The agent hops off at about (145, -158) and falls about 280 units for 60 frames into h12.
- **Death seen too late.** When the fall starts, the death is past the 48-frame window. Soft risk reaches 1.0 48 frames before death and the veto comes 12 frames before. By then every candidate is doomed.
- **Most decisions see nothing better.** 69% of 1-11 decisions, and 89% on 1-12, had no candidate that made progress. V says a route exists, but no 48-frame sequence of the 73 moves reaches a lower V. The progress counter and what the agent can actually do disagree. When nothing makes progress, the choice falls between near-equal bad options, and that looks random.
- **Nothing learns from these deaths.** `DeathMemory` only stores unexplained deaths, so the same fall repeated 18 times.

## 2. Root causes (each one explains several symptoms)

| | Root cause | Symptoms |
|---|---|---|
| R1 | **The objective is not what we want** (reach the goal, quickly, alive). It is scored at a fixed window end, in route-cost units instead of time, without goal contact, with an optimistic V in the air and an uncertainty tax on progress. | 1.2, 1.5, 1.6, part of 1.3 |
| R2 | **The progress counter disagrees with the agent's abilities.** V comes from realpath's grid jump rules, a second physics that knows nothing of the agent's moves, run-ups for long jumps, or slope behaviour. | 1.4, 1.6, part of 1.2 |
| R3 | **No commitment.** It re-decides from scratch every 2 frames on near-tie scores, and the game waits for every decision. | 1.1, 1.2, 1.3 |
| R4 | **Experience is not used.** Only first-move failures and unexplained deaths are remembered, V never changes, and the world model never retrains while playing. | 1.4, 1.6 |
| R5 | **Measurement gaps.** The surprise counter includes abandoned plans, there are no finish times, and no parameter can be changed live. | analysis, user requests |

## 3. Proposed changes

### C1. Score in time, goal contact counts, airborne ends are completed (R1)

- **Score = estimated frames to reach the goal:** frames used by the plan + T(end), where T is a time-to-go field in frames. Lower is better. Death stays a hard veto (first 16 frames) or a risk.
- **Goal contact ends the plan.** Touching the goal (or the next key) anywhere along the imagined path ends the plan there, scored by the frames until the touch. This uses the same swept box-versus-movement test as hazards.
- **Airborne ends are completed.** The imagined path continues (cheaply: 1 member, a fixed continuation such as "keep the direction, no jump") until it lands or dies, up to about 90 more frames. The score is taken at the landing. This removes `V_air`'s optimism and sees the 1-11 fall before jumping.
- **No uncertainty tax on progress.** Uncertainty goes to safety and landing instead: a plan "lands" only if all members land on safe support.
- **Waiting costs frames, and finishing sooner is better,** with no extra terms.

Until C2 exists, T = V × a frames-per-V-unit factor calibrated on recorded finishes.

### C2. Time-to-go field built from the agent's own moves (R2)

This is still "the path finder only as a progress counter": a counter, not a rail. The difference is that it is built from moves the agent can really make.

- **Nodes:** supported spots, meaning standing cells, and wall-contact cells per side.
- **Edges:** each library move, plus long run-ups (run 8 to 40 frames, then jump), imagined from rest at that spot with the world model until it lands. An edge is kept if it lands safely on a node. Its cost is frames.
- **T(node):** frames to the goal, from a backward Dijkstra. In the air, C1's completion gives the landing node.
- **Cost:** built once per level and cached. The size and time must be measured; the guess is a few hundred nodes × about 90 moves, a few seconds batched.
- **Learning on edges:** when a planned edge fails in real play (did not land where expected, or died), its cost goes up or it is removed, and T is rebuilt (cheap). This replaces `FailureMemory` and covers slopes: a wall jump off a slope that fails stops being believed.
- **realpath's V stays** as the fallback where the move graph has no route, and as a comparison.
- **Risk:** the graph trusts the world model over 24 to 60 frames (median error 6 to 8 units at 16 to 32 frames; nodes are 20-unit cells). Measure the share of edges confirmed when actually played.

### C3. Commit, and plan while the game plays (R3)

- **Execute the chosen plan.** Replan when any of these happens:
  - the current move ends;
  - the real state leaves the predicted tube (error > err90 at that horizon);
  - new danger appears;
  - N frames have passed (default 8, changeable live).
- **Hysteresis.** Switch only if the new plan is better by a margin (for example 3 frames of time-to-go), or the current one became unsafe.
- **Plan ahead in parallel.** While frames f to f+k play the committed actions, plan from the predicted state at f+k. The lockstep only waits when the plan is not ready.
- **Expected:** 4 to 8 times fewer decisions, a game near real speed on this CPU, and room for larger search budgets. To be measured.

### C4. Learning while playing (R4)

**Deaths.** For every death, explained or not, find the last decision where a safe alternative existed (the point of no return). Charge the move or edge taken there, through C2's edge costs, so it learns "don't hop off here".

**World model.** A learner process:
- fine-tunes the current model on new `wm_data`, keeping the held-out levels out;
- evaluates on those held-out levels;
- publishes a new model only if it is better.

The agent loads it between waves. When it trains is a CPU question (see open questions).

**So:** there is no separate training ground. It learns while playing:
- the memories and edge costs immediately;
- the world model every N waves;
- explore agents only add data.

### C5. Tooling (R5, and the user's requests)

- **Surprise counter:** only plans actually followed up to that horizon.
- **Finish times:**
  - Frames divided by 60 gives seconds, exact with the time-step lock.
  - Logged per finish in `runs/<name>/times.jsonl`, and per level in `smbrl/best_times.json` (best, run, wave, date).
  - Shown in the console, viewer and training view. The old unreliable `best.json` is not touched.
- **Viewer controls:** a control panel like the training monitor's sliders, sent to the agent over UDP. Changes take effect at the next decision and are logged as events, so replays show them. Controls:
  - horizon, replan interval and hysteresis;
  - time budget and beam sizes;
  - ensemble members used;
  - veto frames;
  - completion length.
- **Training view (second tab of the viewer),** like the training monitor, plotted per wave:
  - finish rate, finish time and best time;
  - deaths (explained, unexplained, learned);
  - decisions per second and game speed;
  - prediction error per horizon on followed plans;
  - world model version and its held-out error.

## 4. Order of work

Each step is one coder brief, with tests, one commit, and a check before the next.

1. **Measurement first** (small):
   - the surprise counter fix;
   - finish and best times;
   - the analysis script as a repo tool (`plan/analyze_run.py`);
   - the slope measurement on recorded data (do slopes allow wall jumps?).
2. **C3:** commit, hysteresis and planning in parallel. Measure the switch rate, follow length and game speed before and after.
3. **C1:** time score, goal contact, airborne completion, with the calibrated V-to-frames factor. Measure on 1-6 (princess) and 1-11.
4. **C2:** the move-graph time field. Measure it against realpath's V on 1-11, 1-12 and recorded finishes.
5. **C4:** the point of no return, edge costs, then the background world-model learner.
6. **C5 viewer controls and training view.** This can run in parallel with steps 2 to 5 if the agent side (`agent.py` control listener) is done in step 2's brief.

## 5. Open questions for the user

- Is C2 (a progress counter built from the agent's own moves instead of realpath's grid) acceptable?
- When may the world-model learner use the CPU: between waves only, or in the background at low priority?
- What should the default replan rule be: every 8 frames plus on surprise, or only on surprise or end of move?

---

## 6. Decisions after review (2026-11-03): these replace sections 3 to 5 where they differ

A second agent reviewed this plan against the code. Its checked findings:

- **Goal not scored: confirmed.** There is no goal logic in `plan/planner.py`; the score is V at the window end (lines 434-436, 490, 531-533).
  - Also missed: **keys.** The leg is fixed at plan start (lines 408-409), so an imagined key pickup is not credited either.
  - The world model already predicts a `finish` event (`wm/features.py:44`) that the planner ignores.
- **V_air: worse than described.** It takes the minimum over all jump states, ignoring velocity (`progress.py:282-289`). On top of that, wall contact counts as support (`planner.py:388-389`), so a body falling fast past a wall gets a fresh-jump value.
- **V is not time.** Its step costs include hazard-margin, crumble, switch and lock extras (`realpath.py:233-247`). One calibrated V-to-frames factor would still not be time.
- **Full C2 (move graph from rest) is too heavy as a first step.**
  - Build time: roughly 20-60 s per level at the measured 10-15 µs per row-step, and again after each world-model update.
  - It misses momentum (run speed carried into jumps; wall slides are never at rest).
  - It cannot represent moving hazards' timing.
  - Its false "unreachable" edges never correct themselves, because those edges are never tried.

**Revised changes:**

- **C0 (new, first): goal and key contact in the plan.** A swept test of the goal box and the next key's box against every imagined path. Touching the goal ends the plan as a success; touching the key switches to the next leg. The world model's `finish` head is logged as a cross-check.
- **C2 becomes A, then B. Full C2 only if B finds many disagreements.**
  - **A: a time field on realpath's existing graph.**
    - A second weight per edge: frames per step (ground vs air, measured from recorded play), without the hazard, crumble and switch extras. Closed cells stay closed.
    - A velocity-aware V_air: separate values for rising and falling states (the index already exists), and wall contact counts as support only when slow enough to cling (threshold measured).
    - Learned per-cell and per-edge penalties from failures and points of no return, then Dijkstra again (0.1-0.3 s).
  - **B: check the route's jumps with the agent's own moves.** For the jump segments of realpath's best routes, try the library moves from the takeoff cells with the world model. Segments no move can realize (for example a "wall climb" on a slope) are closed or penalized, and the field is rebuilt. This is tens of nodes, not hundreds.
- **C1, airborne completion, revised.**
  - A small fan of continuations (keep direction, neutral, reverse), one ensemble member.
  - Each stops at the first ground **or wall** contact; the score is the best safe landing.
  - Deaths during completion count as risk, not veto.
  - Completion runs only in the final stage (about 20 ms); stages 1 and 2 use A's velocity-aware V_air, so the stages agree.
  - A plan "lands" when a majority of members land safely, not all.
- **C3, synchronous first.** Commit to the plan, with hysteresis on the already re-scored previous plan.
  - Replan every 8 frames, plus on a surprise (outside the error tube), the end of a move, or new danger.
  - About 130 / 8 ≈ 16 ms per frame: near real speed with no threads.
  - Asynchronous planning only if that is not enough. Pitfalls: future hazard frames are not available yet; `plan()` changes shared memories; tiles change on another thread; the start state would be the model's own output.
- **C4.** The point-of-no-return charge goes onto A's cell and edge penalties. The world-model learner comes last, because it changes everything built on the model.
- **C5.** First measure the per-frame cost without planning (explore agents, data collection, telemetry): it becomes the speed floor after C3. Parameters become CLI flags first; the viewer's live sliders follow.

**Revised order:**
1. **Measurement:**
   - surprise counter fix;
   - finish events and best times;
   - `plan/analyze_run.py`;
   - slope wall-jump measurement;
   - per-frame overhead without planning.
2. **C0:** goal and key contact.
3. **C3:** synchronous commit, event and interval replanning, hysteresis.
4. **C1 with A,** plus the point-of-no-return charging.
5. **B,** then full C2 only if needed.
6. **Viewer:** live parameter sliders and the training tab.
7. **Asynchronous planning** only if still short of real speed; the world-model learner after that.

---

## 7. Step 4 as built (2026-11-08): C1 with A, plus the point-of-no-return charging

Everything below was built and checked **offline only** (recorded states, the world model, no game). Nothing was
tried in the game: not the overlay, not the timing under the live loop, not whether agents now finish more.

### 7.1 Measured constants (`python -m plan.measure_time runs/plan_2026110*/wm_data`)

548 shards, 2.07 M locked alive steps, levels 1-1, 1-2, 1-6 to 1-14; velocities from position differences.

| constant | value | from |
|---|---|---|
| fx_ground | 5.00 frames / tile | TILE / p90 of abs(dx) on the ground (4.0 u/f); median-based 11.65 |
| fx_air | 5.00 | the same in the air (p90 4.0 u/f); median-based 11.01 |
| fy_up | 5.41 | TILE / mean rise speed (3.69 u/f); median-based 4.90 |
| fy_down | 7.74 | TILE / mean speed of falls over 2 tiles (2.58 u/f); median-based 8.18 |
| CLING_SPEED | inf | wall jumps after a press on a wall: 93-100% in every downward-speed bucket up to 10 u/f (13,385 presses; 8+ u/f only 16) |

The rows' vx / vy fields are units / s (median field / difference: vx 57, vy 60).
M3, T_support at the spawn vs the best finish (reported, not fitted): 1-14 246 vs 321 frames, 1-8 147 vs 129,
1-13 192 vs 351.

### 7.2 What was built (commits)

- `d9f1b22` measure_time (M1, M2, M3).
- `e66640e` the time field T on realpath's graph (`plan/progress.py`): frames per step, a second Dijkstra,
  T_support / T_rise / T_fall per leg, one lookup code path (`_lookup`) for V and T. Build per level: 1-14
  0.16 -> 0.19 s, 1-13 0.14 -> 0.16 s, 1-8 0.16 -> 0.19 s, 1-11 0.32 -> 0.36 s, 1-6 0.10 -> 0.11 s.
- `38086bd` (breaking) the score in frames, the cling rule (`Planner.support`), velocity-aware airborne T, the
  airborne completion fan in the final stage (`Planner.complete`), `MoveMemory` and FailureMemory without deaths,
  analyze_run's frame fields; `--hysteresis` in frames (default 3). One commit: these are one function.
- `6603b00` agent.py: `--move-memory`, death causes to the planner, point-of-no-return events, decisions get
  inputs to H + 90 frames, the overlay's completion.
- `d6d000f` viewer: completion path, landing, "estimate N frames = H + completion k + T to go".
- `adbe209` re-recorded goldens (planner_golden, runner_old_schedule) for the new score.

Small adjustments, explained:
- the rising / falling tables of A2 were built in the field commit (they come out of the same pass);
- after a plan ending without a horizontal input, the fan is neutral / left / right (keep and reverse would both
  be neutral);
- a cell where the graph reaches only rising or only falling states gives both the reachable one;
- a portal crossing costs 1 frame in T;
- an end with T = inf is charged T(start) + DEAD_END_FRAMES (as the old -DEAD_END below the start);
- vetoed candidates break ties by 1e-6 x estimate (1e-3 x frames could reorder veto frames).

### 7.3 Offline checks (old = 0214707; no deadlines unless said; no previous plan; hysteresis 0)

**a. 1-14 pit** (`runs/plan_20261107-200300`): 100 out-of-level deaths with x in -20..130, 277 recorded supported
decisions in the 30 frames before their takeoff.
- old: the chosen plan ends airborne over the pit (x > -40) in 201 / 277 (73%); old itself predicts its plan dies
  (veto or soft risk >= 0.5) in 8 / 277 (3%).
- new: a different first move in 259 / 277 (94%); the chosen plan ends over the pit in 0 / 277; lands outside the
  pit or a different first move 277 / 277.
- the new planner's verdict on the old chosen plans: predicts they die in 8 / 277 (3%). It avoids them because
  they score a median 61 frames worse (10%: 2, 90%: 85), not because it sees the death.
- **Why it does not see the death (a measured contradiction of C1b's premise):** the world model's tile input
  pads outside the map with solid (`wm/features.LevelGeom`), so imagined bodies stop about 6 units above the map's
  bottom edge (y ~ -354; real ones fall on to -362 and die below the kill box at -368.6). Over the pit, keep and
  neutral hover there for all 90 completion frames without leaving the kill box; reverse reaches the pit's left
  wall (x ~ 6), a wall recorded agents did wall-jump off (30 of 2,691 bodies that reached the map bottom came
  back). So neither the safety check nor the completion can see this fall. The long airborne completions make the
  old jumps expensive in frames, and that is what changes the choice. A strict xfail test pins it
  (`test_completion_over_the_1_14_pit_sees_every_continuation_die`).
- Caveat: successful recorded lives also jump right across the pit (x 0..80, y -300..-314, on to x > 130). The new
  choices go left (`run wall-jump L`, `run-jump L ... after 4f`) or make shorter right jumps (`wall-jump R hold
  10`). Whether that is a way through is not checked in the game.

**b. 1-14 hole**: 454afae's 60 moments were not kept as a script. Reconstructed: 60 recorded states on the
half-block floor (runs/plan_20261107-174219, x 60..120, y ~ 36; seed 0); "through the hole" = the plan's path or
its completion below y 15. Old 21 / 60, new 44 / 60 (11 within the 48-frame path, the rest by the completion). The
commit's 54 / 60 is not reproduced by this definition, so ">= 53 / 60" can be confirmed neither way.

**c. 1-11 h12** (`runs/plan_20261103-152013`, 17 deaths on h12): all 269 recorded decisions in the 30 frames before the 17
takeoffs (the agent hops there: only 6 of them were supported, so the supported-only rule of (a) was widened; takeoff
= the last support above y -250, the hop off at about (150, -158)). "Over the fall" = the plan's end more than 2
tiles below the takeoff.
- old: ends over the fall in 95 / 269 (35%); old predicts its own plan dies in 19 / 269 (7%).
- new: ends over the fall in 102 / 269 (38%): **no improvement**; a different first move than old in 216 / 269.
- the new planner's verdict on the old chosen plans: predicts they die in 103 / 269 (38%; in 97 every completion
  continuation dies in h12), so the completion does see this fall (unlike the 1-14 pit); the new choices still
  end over the fall about as often, scored a median 21 frames better than old's (90%: 79).

**d. 1-6 princess wait** (wave 31, 12 decisions 17..40 units from her): old and new both touch the goal in 12 / 12
(step 2's goal contact already fixed it); chosen `run R 8f > run-jump L hold 15` in 11 of 12.

**e. Closed loop** (`plan/offline.py` closed_loop, 300 frames, replan 8, triggers on, no deadlines; old
hysteresis 0.5 V, new 3 frames; 1-13 from 0 / 100 / 200, 1-8 from 0 / 30 / 60, 1-14 from 0 / 100 / 200): old 4
finished, 0 died (1-13 306; 1-8 120 / 115; 1-14 292); new 5 finished, 0 died (1-13 249; 1-8 128; 1-14 241 / 380 /
475). Switch rate mean old 0.47, new 0.58 (higher).

**f. Timing** (30 recorded decisions per level, 2 threads; median ms):

| | budget | total 1-14 / 1-13 / 1-11 | stage 1 | stage 2 | final | completion 1-14 / 1-13 / 1-11 | needing completion |
|---|---|---|---|---|---|---|---|
| old | 120 | 142 / 151 / 153 | 39-41 | 19-20 | 85-97 | - | - |
| new | 120 | 239 / 193 / 220 | 40-41 | 18-19 | 132-188 | 102 / 42 / 72 | 93% |
| old | none | 218 / 234 / 240 | 38-45 | 89-106 | 86-94 | - | - |
| new | none | 254 / 268 / 319 | 43-44 | 90-108 | 110-169 | 20 / 36 / 67 | 70-93% |

**The completion is over its 20 ms budget** (19-102 ms median per decision), and a decision is about 50-90 ms over
the 120 ms budget, because the final stage (completion included) is not cut. Imagine costs about 1 ms per step at
any batch size, so the cost follows the frames rolled, not the rows. Options measured (no deadlines, completion ms
median / 90%, 30 states per level; the first move was the same as with 90 / 3 in 30 / 30 for every option):

| COMPLETE_MAX / continuations | 1-14 | 1-13 | 1-11 |
|---|---|---|---|
| 90 / 3 (built) | 19 / 108 | 36 / 62 | 71 / 96 |
| 45 / 3 | 19 / 60 | 40 / 51 | 66 / 79 |
| 30 / 3 | 19 / 45 | 33 / 41 | 48 / 54 |
| 90 / 2 | 17 / 103 | 34 / 59 | 60 / 99 |

Fewer frames mainly cut the tail; two continuations save about 10%. None reaches 20 ms median on 1-11. Not changed
here (the brief: report, do not cut).

### 7.4 Concerns found while building

- (Resolved in 7.6 item 2.) **A predicted contact after the veto frames costs only RISK_FRAMES = 50 frames**, about what 48 frames of
  waiting cost. On a flat test level behind a hazard strip (no way on), the chosen plan now runs into the strip
  after frame 16 (soft risk 1); before, the uncertainty tax kept it back. The test now checks only the veto window
  and says why.
- (Resolved in 7.6 item 4.) "No progress" (T(start) - estimate <= 0) is borderline for the best possible plan: running at full speed gains
  exactly the frames it takes (fx is the p90 speed), so the share will be high even for good plans.
- The world model's solid padding outside the map (7.3 a) hides falls out of any level whose kill box is below the
  map. Fixing it means retraining with open padding, or a planner rule for the map bottom (which the wall-jump
  recoveries above make non-trivial).

### 7.5 Not verified

Nothing was run in the game. Not checked: the overlay and viewer drawing (only their payloads and text), the live
cost per frame with completion, MoveMemory over real waves (only unit tests and the offline closed loop), whether
the new 1-14 choices find the way across the pit.

### 7.6 Follow-up round (2026-11-08; offline only; old = a403092 unless said)

**1. Pit hover: open padding below the map, model unchanged, measured and rejected.** `wm/features.LevelGeom`
pads outside the map with solid on all four sides. Experiment: the padding below the map opened (top and sides
unchanged), on recorded real rows within 3 tiles of the map's bottom (400 sampled on 1-14,
runs/plan_20261107-200300; 1-11, runs/plan_20261103-152013, has no such rows):

| | solid (as trained) | open below |
|---|---|---|
| 1-step error, median / 90% (units) | 0.87 / 2.03 | 1.05 / 2.29 |
| 16-step error, median / 90% | 11.6 / 28.6 | 12.3 / 31.7 |
| no-jump completion rows 40 frames before 40 pit deaths: died / landed / still airborne (of 120) | 9 / 40 / 71 | 106 / 14 / 0 |
| the 30 recorded wall-jump-backs off the pit's left wall, replayed: predicted to come back | 12 | 0 (all die) |

Open-below is worse on error and loses every recorded recovery (most of them cling to the wall at x ~ 6 between
y -354 and -367, above the kill box at -368.6; the model never saw open cells there). Not adopted; no map-bottom
rule added; the strict xfail stays. Follow-up in PROJECT_STATE (9.13): retrain the world model with open padding
below the map, then repeat this measurement.

**2. One death cost** (`8671609`). DEATH_FRAMES = max(T_support(spawn) - T(start), DEAD_END_FRAMES) for every
expected death: soft risk x DEATH_FRAMES, a completion where every continuation dies 1 x DEATH_FRAMES (with the
window's soft risk at most one death), MoveMemory rate x DEATH_FRAMES. RISK_FRAMES is gone; the hazard-strip test
asserts again that the chosen plan never passes the strip (it does not). Planner golden re-recorded (the strip
case's first move and the scores of cases with soft risk changed); the old-schedule record did not change.
- 1-14 pit (a), new code: chosen plans ending over the pit 0 / 277 (old 0214707: 201 / 277); the planner's own
  death prediction for the old plans is still 8 / 277 (3%): the model cannot fall out of the map (item 1). No
  chosen plan has a predicted death (completion all-die 0, soft risk >= 0.5 0).
- 1-11 h12 (c), 269 decisions: chosen plans whose completion predicts death 15 / 269 (6%); soft risk >= 0.5 18 /
  269. Of the 102 chosen plans that end over the fall, the completion lands above h12 (y > -400) for 55, every
  continuation dies for 15, 32 land by a majority of members (no completion), 0 land at or below h12's level.
  The tiles alone do not show where "across" is (the hop-off at about (150, -158) is not on a tile; likely a
  moving platform), so "lands across" here means "lands above h12's level".

**3. Hysteresis from measured noise** (`5e5d489`). How much a kept plan's re-scored estimate moves between
consecutive decisions whose only trigger was the interval, beyond the frames played: 206 pairs (147 from new-code
closed loops, 59 recorded 1-14 decision pairs re-planned; run 174219 logged no interval-only triggers). |noise|
median 16, 75% 60, **90% 160 frames**; signed median +16 (a kept plan usually looks worse when re-scored; the tail
is soft risk and completion deaths changing, each worth up to DEATH_FRAMES). `--hysteresis` defaults to 160.
Closed loops (9 starts, 300 frames, no deadlines), same code:

| hysteresis | switch rate | finished | died |
|---|---|---|---|
| 3 | 0.62 | 2 | 0 |
| 60 (the 75%) | 0.12 | 2 | 2 (1-14, out of level) |
| 160 (the 90%, the default) | 0.09 | 1 | 2 (1-14, out of level) |

Large margins keep committed plans the model cannot see die (the 1-14 pit). Reported, not tuned.

**4. "No progress"** (`ff5fa40`): T(end) >= T(start), the time-to-go did not shrink (analyze_run and the
planner's alternative reasons).

**5. Speed** (`8518616`, identical results: goldens compared exactly). Per imagined step: stage 1 1.20 -> 0.89 ms,
stage 2 1.54 -> 1.22, final 1.29 -> 1.05, completion 1.08 -> 0.82 (patch table, featurization into the step
buffer, vectorised flag update, preallocated forward buffers, inference_mode). The forward pass (about 0.4-0.65 ms
a step, 4 batched layers of width 256: per-op overhead, not size) is now the largest part.

Per decision at --budget 120 (30 recorded decisions per level, 2 threads; median / 90% ms):

| | total 1-14 / 1-13 / 1-11 | stage 1 | stage 2 | final | completion 1-14 / 1-13 / 1-11 |
|---|---|---|---|---|---|
| 0214707 | 142 / 151 / 153 | 39-41 | 19-20 | 85-97 | - |
| a403092 | 239 / 193 / 220 | 40-41 | 18-19 | 132-188 | 102 / 42 / 72 |
| now | 175 / 142 / 168 | 26-28 | 32-34 | 79-115 | 63 / 24 / 47 |

90%: now 195 / 164 / 183 total. Without deadlines now: total 156 / 177 / 213, completion 16 / 28 / 46 (a403092:
254 / 268 / 319; 20 / 36 / 67). Item 1 changed nothing (not adopted), so falls do not end earlier.
The completion is still over 30 ms median on 1-14 (63) and 1-11 (47) at the 120 budget. Options re-measured
(no deadlines, completion ms median / 90%, 30 states per level; first move unchanged in 30 / 30 for every option):

| COMPLETE_MAX / continuations | 1-14 | 1-13 | 1-11 |
|---|---|---|---|
| 90 / 3 (built) | 15 / 71 | 27 / 42 | 48 / 63 |
| 45 / 3 | 15 / 41 | 29 / 39 | 42 / 50 |
| 30 / 3 | 16 / 28 | 24 / 26 | 31 / 36 |
| 90 / 2 | 15 / 69 | 26 / 37 | 39 / 67 |

Not chosen here: left to the planner's decision.
