# World model + MPC for the Super Meat Boy agent: Phase 1 findings and proposal

Status: proposal only. No existing code has been changed for this yet. Phases 2-7 start after approval.

## Phase 1 findings: the system as it is

### How the game and Python talk

- `realgame.js` runs inside the game (Frida). Each game frame, inside the hidden main character's update, it
  sends one `frame` message and **blocks until Python answers** with the agents' inputs (`recv('act').wait()`).
  Game and trainer run in lockstep at the game's 60 Hz.
- **Time budget:** the game's time step includes the time it waited for Python. If one frame takes more
  than **80 ms** (the gap between frames), the script holds all agents still for that frame so nothing
  jumps. Planning must stay well inside that, or it halves the training speed. The useful budget is about
  10-15 ms per frame for everything Python does.
- Long work (route planning, PPO updates) happens only at level info and wave ends. The game waits for a
  `go` message there.

### Real-time data available every frame (measured, not assumed)

| Data | Source | Per frame? |
|---|---|---|
| Agent x, y | char +0x84/+0x88 | yes |
| Agent vx, vy | char +0x9c/+0xa0 | yes |
| Flags: airborne (bit 1), facing (bit 0), on a wall (0x200), dead (0x100) | char +0x7b4 | yes |
| Movement state: 0 idle, 7 jump, 8 fall, 9 wall slide, 11/12 run, 14 death, 6 in wind, 16 intro | char +0x7b0 | yes |
| Status: alive / dead / finished; keys bitmask | game script | yes |
| Hazard boxes [x, y, half w, half h, angle] with ids (velocity derived) | world pieces + game size function | yes |
| Shots in flight (thrown saws, missiles) | launcher array +0x2c30 | yes |
| Moving platforms [x, y, upright] (60x20 bars) | world +0x2dc4/+0x2dd4 | yes |
| Fans and orbs (wind force computed like the game) | world arrays | every 8th frame |
| Tile words that changed (crumble, locks) | tile map diff | when changed |
| Level: full tile map, spawn, goals, keys, bandages, portals | level info | once per level |

Not available (yet): the character's internal timers. The physics measured earlier has hidden state:
- a jump can only start after the key was released for 2+ frames;
- a jump is cut when released before 5 frames;
- right after a wall jump, input is limited for a moment.

Python only knows "jump held" (frames, mirrored from the inputs it sent). Some of this hidden state may
make prediction noisy until it is found in the character object (Phase 2 includes a probe for it).

### Actions, episodes, PPO

- **Actions:** 10 discrete combinations of left / right / jump / run, chosen every frame for every agent.
  The game script holds keys like a pad, and re-presses jump on walls.
- **Episodes:** a wave of N agents (up to 200) starts together after a level reset, 20 s limit.
  - Each agent has one life; death or finish ends its episode.
  - Waves reset the level the same way each time, so moving saws and platforms repeat their motion from
    the wave start (this was seen when waves were introduced; it will be measured in Phase 2).
- **PPO:**
  - Two MLPs (policy and value, 2x256 tanh) on a 282-input observation, CPU only.
  - Batch 16384, 4 epochs, minibatch 2048, clip 0.2, gamma 0.995, lambda 0.95.
  - Updates happen between waves.
- **Observation:** agent-centred, so not tied to one level. Velocity, flags, an 11x11 tile map, an 11x11
  surface map, route direction, the 2 nearest moving hazards, wind, platform, stuck time, jump held.
- **Reward:** progress by closeness to the routes (both ways), death -15, finish +50 and more, keys,
  exploring, stuck penalties. The progress part is an exact function of position (`RewardPath.value`).

## Proposed architecture

```
 game (realgame.js) --frame--> realgame.py (trainer, unchanged PPO) --act--> game
                                    |  \
                       raw transitions   \ planner calls (only for MPC agents)
                                    v      \
                          wm/buffer.py      wm/mpc.py ---- uses ---> wm/model.py (ensemble)
                                    |                |                     ^
                                    v                v                     | weights file
                          wm/train.py (own process)  wm/telemetry.py --UDP--> wm/dashboard.py (own window)
```

New code lives in a separate `wm/` package. The trainer gets small, switchable hooks only:
`--wm-collect`, `--mpc-agents N`, `--telemetry`. With every flag off, behaviour is exactly as today.

### 1. Data and replay buffer (Phase 2)

- **Per agent per frame:**
  - raw state: x, y, vx, vy, flags, movement state, jump held, keys;
  - the action, reward, the next state;
  - terminated (death / finish), truncated (wave time), wave frame number, level name.
  - That's about 20 numbers. Features are rebuilt from the level data at training time instead of
    storing 282 floats per transition: 100 agents x 60 fps is 6000 transitions/s, about 21 million an
    hour.
- **Per level, saved once:**
  - the tile map and its changes;
  - hazard, shot and platform positions by wave frame (the timeline);
  - the routes.
- **Storage:** a ring buffer in memory written to compressed shards on disk (`runs/<run>/wm_data/`), so a
  separate training process can read them.
- **Probes done once:**
  - Are hazards and platforms at wave frame t the same in every wave? If yes, the planner can look up
    their future positions exactly instead of predicting them.
  - Which character fields change from frame to frame (a diff over the 0xc00-byte character)? This is to
    find the jump and wall-jump timers that are hidden now.

### 2. World model (Phase 3)

The model learns the physics the agent controls; the level is looked up, not memorised:

- **Input (all agent-centred, normalised):**
  - velocity (/400, /600), flags, movement state (one-hot), jump held, and the action (one-hot);
  - local solid map and surface map around the predicted position (re-read from the tile map at every
    step of a rollout);
  - nearest hazards, platforms and wind relative to the agent;
  - no absolute position, so nothing level-specific.
- **Network:** an ensemble of 5 MLPs (3x256, SiLU) with shared input normalisation. A GRU is optional; it
  is added only if the prediction errors in Phase 3 show hidden state that the timers above don't cover.
- **Heads:**
  - next state: change of x, y; new vx, vy (Gaussian, so it also gives a confidence);
  - airborne / on a wall / facing (binary), movement state (classes);
  - reward (the parts not computable from position: keys, exploring);
  - death, finish (terminated);
  - landing: becomes grounded next frame;
  - collision: velocity cut by a wall or floor.
  - Losses: Gaussian NLL for continuous outputs, cross-entropy for discrete ones.
- **Uncertainty:** disagreement between the 5 ensemble members. The planner treats high disagreement as a
  risk.
- **Check before moving on:** one-step errors on held-out frames, per movement state (ground, air, wall,
  wind). They are compared to two baselines: "same velocity as now" and the measured physics formulas.

### 3. Multi-step rollouts (Phase 4)

- **Rollout loop:**
  1. Model step.
  2. Read the tile patch at the new position.
  3. Hazards and platforms at that future wave frame, from the timeline, or constant velocity if they
     don't repeat.
  4. Next step.
- **Training:** unrolled over k = 1, 4, 8, 16 frames with teacher forcing turned down step by step, so
  long rollouts don't drift.
- **Measured:** position error, velocity error and death accuracy against the horizon (1-32 frames),
  reported per level and in total.

### 4. MPC as a layer next to PPO (Phase 5)

- **Who plans:** only the agents in an MPC group use it (`--mpc-agents N`); the rest are PPO as today. All
  three variants run in the same waves on the same level, so they can be compared directly:
  - **PPO only:** no change.
  - **PPO + world model ("shield"):** PPO samples an action. The model predicts the next 0.5 s with that
    action held. If death is likely and another of PPO's 3 most likely actions is much safer, that one is
    used.
  - **PPO + world model + MPC:**
    1. Candidates: 16 sequences of 8 steps x 4 frames (0.5 s). Some are sampled from the PPO policy,
       some are fixed patterns (hold right, jump, right+jump, wait then jump, back then jump, ...).
    2. The ensemble rolls them all out in one batch.
    3. Score = progress (exact, from the closeness field) + expected key and exploring reward
       - P(death) x 15 - uncertainty penalty + PPO value of the end state.
    4. Run only the first action, replan next frame (receding horizon).
- **Budget:** 16 x 8 steps x 5 models for one agent is about 640 small MLP evaluations in one batch, a
  few ms on CPU. The time per frame is measured, and the number of MPC agents is capped so a frame stays
  under the budget.
- **PPO learning stays correct:** MPC agents run actions PPO did not choose, so their data would break
  PPO's update. Their transitions go to the world-model buffer and the evaluation only, not into PPO
  updates.

### 5. Dashboard (Phase 6)

- **Separate:** its own process (`wm/dashboard.py`, pygame like the monitor). The trainer sends small
  messages over local UDP. If the dashboard is closed or slow, the trainer doesn't wait and nothing breaks.
- **For one chosen agent (click to pick):**
  - **State:** map, hazards, platforms, wind; position and velocity arrows.
  - **Futures:** every candidate trajectory. The chosen one is solid, risky ones (high P(death)) red,
    uncertain ones (high ensemble disagreement) dashed or faded, the rest grey. Dots mark each future
    step, with hazards and platforms at the same future frames.
  - **Risk:** predicted death spots, landing spots with their success probability, collision marks.
  - **Predicted vs actual:** the predicted point for "now" from 1, 8 and 16 frames ago next to the real
    position, plus position / velocity error and predicted vs actual landing, death and collision events.
  - **Rolling error curves:** by horizon, over time.
  - **Planning table:** each candidate's actions, predicted reward, risk and uncertainty, the selected
    one, the horizon.

### 6. Generalisation (Phase 7)

- **Held-out levels:** for example every 5th level of each chapter, or a whole chapter. The world model
  never trains on their data, and PPO doesn't update there (`--eval`).
- **Measured on seen and unseen levels, for PPO only, PPO + shield and PPO + MPC:**
  - finish rate, deaths per agent, average progress;
  - prediction error by horizon.
- Results go into a report file per run.

## Risks and open questions

1. **Hidden timers:** jump release frames and wall-jump lockouts are not in the data yet. If the probe
   doesn't find them, the GRU is the fallback.
2. **Speed:** the 80 ms lockstep limit is hard. MPC must be batched and capped; it can't run for 100
   agents every frame on CPU.
3. **Repeating hazards:** the hazard timeline is only usable if waves really repeat. If not, the model
   predicts hazard motion from velocity (worse far ahead).
4. **Data balance:** most frames are easy (running on flat ground). Rare events (deaths, wall jumps,
   platforms) will be oversampled in training.
5. **Locks and crumble blocks:** they change the map per agent and over time. Rollouts use the agent's
   own current map (keys included).

## Order of work

| Phase | Deliverable | Done when |
|---|---|---|
| 2 | `wm/buffer.py`, collection flag, the two probes | data from a real training run on disk; probe results documented |
| 3 | `wm/model.py`, `wm/train.py` | one-step errors beat both baselines on held-out frames |
| 4 | multi-step training + error-vs-horizon report | error curves for 1-32 frames |
| 5 | `wm/mpc.py`, `--mpc-agents` | measured frame time within budget; three groups compared in the same waves |
| 6 | `wm/dashboard.py`, telemetry | live view of futures, risks and predicted vs actual |
| 7 | eval mode + report | numbers on unseen levels for all three variants |

Each phase is committed separately and checked against the real game before the next starts.

## Phase 2 results (run real_20261102-035612, 100 agents, levels 1-6 and 1-7)

- **Data:** 22 waves gave 550k agent rows and 548k transitions in 9.5 MB (about 0.4 MB per wave). There
  were 339 deaths and 70 finishes; 1.8k transitions were cut by the wave time limit.
- **Time step:**
  - With 100 agents the game runs at about 28 updates a second: frames take 36 ms at the median, 47 ms
    at 90% and up to 70 ms. 87% of frames are longer than 20 ms.
  - The game moves agents by velocity x the real step. In free flight the error is 0.07 units that way,
    against 3.4 units with a fixed 1/60 s.
  - **Consequence:** the world model must take the time step as an input, and MPC must roll out with the
    step it expects. The recent median is a good guess.
- **Hidden timers:** none found.
  - Of the 659 four-byte fields of the character, 207 change. None counts real time or frames up or down
    and resets at a jump press or release.
  - The jump press/release logic seems to live in the flag bits (+0x7b4, e.g. 0x08050081 vs 0x00050082)
    and the movement state, which are recorded in full.
  - Phase 3 will confirm it: if jumps are predicted well from these, nothing is hidden. A GRU stays the
    fallback.
- **Movement states seen:** 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 21. The model treats them as classes.
- **Do hazards repeat per wave?** Not answered yet: 1-6 and 1-7 have no moving hazards. A run on a level
  with moving saws or platforms is needed.
- **Fixed during the analysis:** the game's wave count starts again on each level, which mixed waves of
  different levels. Each shard now has its own number, and older data is fixed when it is read.

## Phase 3 results (one-step world model)

`python -m wm.train runs/real_20261102-035612/wm_data --epochs 14` trains an ensemble of 5 MLPs on 385k
transitions in about 5 minutes on 4 CPU threads. The report is from 115k transitions of waves it never
trained on (every 5th wave), levels 1-6 and 1-7.

| | world model | velocity + gravity | same velocity |
|---|---|---|---|
| position error, median / 90% (units) | **0.20 / 1.38** | 0.65 / 5.91 | 0.92 / 5.99 |
| velocity error, median / 90% | **3.9 / 42.7** | 15.6 / 213.8 | 21.6 / 219.0 |
| on the ground, position median / 90% | 0.21 / 1.35 | 0.93 / 3.27 | |
| in the air | 0.19 / 1.34 | 0.60 / 5.88 | |
| on a wall | 0.28 / 1.49 | 0.93 / 6.75 | |

- **Events and flags:**
  - airborne 99.1% right, on a wall 99.1%, facing 99.2%;
  - landing: recall 0.99, precision 0.69;
  - blocked by a wall: recall 0.92, precision 0.36;
  - death: recall 0.95, precision 0.30, AUC 1.00;
  - finish: recall 0.88, AUC 1.00;
  - next movement state 94.0% right.
- **Uncertainty is honest:** 97-98% of true values are within 2 predicted standard deviations.
- **What made the difference:**
  - The model predicts only the correction to plain physics (velocity x the real time step, gravity in
    the air), not the whole move. That halved the median error.
  - beta-NLL keeps the uncertainty heads from calling hard steps (jump starts, wall jumps) uncertain
    instead of learning them. That cut the 90% errors 3-4 times.
- **Limits:**
  - Only two levels, no moving hazards or platforms in the data yet.
  - Rare events are found reliably but with false alarms at the 50% cut-off (they were weighted up in
    training).
  - Unseen levels are not tested yet (Phase 7).
