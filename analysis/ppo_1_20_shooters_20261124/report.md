# 1-20 end corridor: projectile timing visibility — 2026-11-24

Read-only/offline investigation of `runs/aplus_v2_batch2048` (batch2048,40 agents,
30s waves,realtest,raw telemetry). No live eval, game attachment, runtime/model change
or training interruption was made. Run was growing; this snapshot contains53 completed
waves,2120 trials,zero finishes,2015 deaths. Not all trials are terminal deaths: the
remaining105 are wave-timeout outcomes. Raw input boundary33,024,601 compressed bytes.

## Native evidence

The start-saw failure is no longer the only observed behavior.127 agent-wave attempts
entered x>150; this is a diagnostic right/end-region threshold, not a certified route
milestone. Live positions in the latest included waves concentrate around x400..440,
y20..-40, by the upper-right shooter before the descending end corridor.

Of1973 deaths whose transitions are visible in frame messages,99 occurred in x>150
(also all in x>250,y between-100 and50 in this snapshot).73 have an overlapping live
projectile according to existing native circle/body geometry;20 overlap a native static
kill shape. Categories may overlap and some deaths match neither. These are contact
matches, **not native engine death-cause flags**. Final deaths reported only in wave_end
have no extra frame-position sample and are not counted in this contact classification.

All73 projectile matches are from launcher entity index1's projectiles, in the upper
right corridor. At those deaths, matching projectile age since first appearance has
min0,median5,max24 native frames.42/73 matches are within5 frames (<=0.0833 simulated
seconds); some happen on the birth frame itself. Merely reacting to already-flying
shots is therefore insufficient in many observed approaches to the muzzle.

Births detected as a flying shot slot newly present compared with the previous frame:

| Native launcher index | Median first observed projectile position | Kind | Observed inter-birth interval |
| --- | --- | --- | --- |
|0 |(-430.11,71.08) |0 |119 frames |
|1 |(430.95,-49.65) |0 |119 frames |
|2 |(430.95,-170.68) |0 |119 frames |
|3 |(430.95,-250.58) |0 |119 frames |

Measured inter-birth p10/median/p90 are all119 frames (~1.9833s), and the recorded
first-birth samples are frame85 after reset. Each entity contributes466 detected births.
Projectile speed is360 units/s. These are observed *flying appearance* times, not raw
fire-timer readings or a universal period promised for other levels. Frame0 is excluded
from birth inference because reset timing may leave stale flying-state data there.

`results.json` retains birth events, contact points/IDs, per-wave reached-region counts,
projectile ages and native speed summaries; `audit.py` reproduces a fresh bounded snapshot.

## What PPO currently receives

Source checked in ONLYPPO `realgame.js`, `realgame.py`, `game/level.py`:

- JS sends flying shot position/radius in `sh`, and kind/native velocity in `shx`.
- Python puts live shots in the local hazard map. It supplies offset/velocity of only
  the two nearest moving hazards inside200 units. Velocity is estimated from consecutive
  positions, not read from `shx`; a newly seen ID therefore starts with estimated speed0.
- The PPO observation contains no shooter cooldown, start delay, shooting/reloading
  state, firing direction/source identity or time since its previous shot. Frame number
  is in telemetry but is not fed as a wave clock to the actor.
- The policy is a feedforward Tanh MLP with no recurrent state/frame history. Jump-held
  and stuck features are present but are not a reliable clock for individual launchers.
- Flying shots are observed reactively; the existing reward-route builder explicitly
  excludes projectile trajectories, so the path is not a promise of timed safe passage.

Thus the user's hypothesis identifies a **real observation gap**: before a shot exists,
the network cannot reliably distinguish safe approach time from imminent firing using
an explicit launcher phase. Even if it learns to dodge visible bullets, it may wait or
drop in front of a muzzle whose next shot appears inside its body. The contact timing
supports this explanation; no controlled phase-feature A/B trial was run, and route
choice/action timing/static-shape contacts can also contribute. Do not claim a timer
feature alone guarantees level completion.

Existing main-worktree static analysis `analysis/reactive_hazards_20261117/rules.md`
section2 records launcher entity fields +4 fire timer,+8 initial delay,+0x10 state,
+0x14 kind,+0x18 target flag. That detailed analysis emphasizes kind5/level4-16, while
these recorded1-20 emitters are kind0. Exact kind0 interpretation and shoot animation
delay must be validated against these measured births before exposing an ETA. A fire
timer alone is not automatically the exact time-to-projectile in every state.

## Minimum proposed next change — not implemented

Expose native shooter position/direction, kind and normalized phase/state facts to PPO;
first verify the kind0 fields against the observed85/119-frame sequence. Use a stable
small number of nearby/relevant shooter slots with explicit presence, instead of encoding
this level number or a hardcoded1.983s assumption. This is observation data for ONLYPPO,
not a world model or action planner. Append any new inputs with existing zero-column
checkpoint widening so current actor/value outputs are initially preserved; additional
PPO training is still required for those inputs to matter.

Separately consume the already measured `shx` velocity for first-frame projectiles with
identity/alignment validation, rather than waiting a frame to estimate it. This improves
reactive information but cannot by itself warn about a projectile that does not yet exist.
Compare end-corridor survival/projectile contacts with the same source checkpoint and
training budget; preserve waiting for real openings rather than increasing generic wait
penalties. No new penalty, safety override or geometry change was applied here.
