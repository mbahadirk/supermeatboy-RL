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
