# Bounded 2-10 field trace (2026-11-12)

Read-only inputs: `runs/collect_2-10_fan_keys_01/wm_data`, first recorded shard's static geometry and frame-0
world. No model inference, rollout, fitting, game connection, production/test/document modification.
`audit.py` builds two fields with single-thread numerical libraries, total 0.743 s excluding startup/imports.
`graph_audit.json` stores the graph trace/probes/counterfactual; `point_checks.json` records independent
geometry support and computed wind at the same sample coordinates.

## Verified result

The horizontal plateau is produced by the field's edge cost and wind-support abstraction, not by a
no-jump-limit fallback. Actual selected mode is clear, jumps=(5,12), two legs, closed/open locks. Reconstructed
route cost 87.4 and spawn T=412 match the run log; stall T is 317.38. This does NOT by itself prove why every
48-frame candidate failed to approach the fan, nor prove that the graph route is physically impossible.

The exact key-leg T graph path from stall cell (21,7), center (30,-150), is:

`(30,-150) -> (50,-130) -> (70,-110) -> (90,-90) -> (110,-70) -> (130,-50) -> (150,-30) [fan support]`

`-> (130,-10) -> (110,10) [wall support] -> ... -> (110,130) -> (90,130) -> key(70,130)`.

Every diagonal ascending edge costs 5.41 frames. The first five unsupported upward steps consume the
five-tile rise budget. The sixth is permitted because its destination has wind-support; that resets the
jump state. The subsequent wall also resets it. There is no arbitrary unsupported vertical climb: from
(30,-74) T_rise=305.74 but T_fall=348.34, demonstrating phase-dependent finite jump restrictions.

From (190,-150), which is inside the right fan, a different route reaches the same wall and key with
the SAME 85.74-frame key-leg cost. The later goal leg contributes 231.64; total T is 317.38 in both cases.
Reason: step time is max(horizontal5, upward5.41), not their sum. Horizontal movement can therefore be
performed inside the same vertical travel cost. Lift cells permit fresh jump states; the wind's numerical
strength, entry velocity and time needed to accelerate into its column do not enter this field cost.

Concrete counterexample: at (30,-154), physical wind force is zero; at (190,-154), computed wind force is
about +1800.526 in y. Their field times are equal. Moving from stall toward the helpful fan is not rewarded
by this T before climbing. The two states are not physically interchangeable even though this heuristic
assigns equal remaining time. This proves optimistic insensitivity, not which physically fastest route wins.

Controlled counterfactual (NOT a fix): remove only the lift mask, leave words/hazards/outside unchanged.
Mode remains clear and jumps=(5,12); fresh-support T at fixed y=-154 becomes 466.28,456.28,446.28,436.28,
426.28,416.28 for x=30,70,110,150,190,230. The horizontal gradient reappears. Spawn T changes to580.
This isolates the plateau's dependency on lift-enabled shortcuts; removing fans' support would discard real
fan capability and is not a recommended production change.

## Important correction to the older summary

The whole line y=-154 is NOT one physical floor. The stall is on a small pedestal; farther right this
height is airborne above a lower floor. At x=70 and110 the field support mask is false, while x=150/190 is
lift-supported in the graph. T_support and T_rise are flat317.38 across the six sampled x values, but
T_fall at x=70 is322.38 and at110 is330.53. Runtime selects supported/rise/fall from actual terrain support
and vertical velocity. Therefore 'all positions have identical runtime T' was too broad. The best-state
air lookup is optimistic and contributes additional state aliasing, but it does not entirely erase phase.

## Source references

- `agent.py:692-704`: runtime ProgressField construction uses wind/platform lift.
- `game/level.py:247-266`: wind cells are a binary geometric column; strength is not tested here.
- `realpath.py:293-325`: physical adjacency and wind mask form support; support includes the complete lift mask.
- `realpath.py:537-578`: route search resets jump counters on support; bounded rise/side/falling otherwise.
- `plan/progress.py:76-79`: fallback search order; actual trace selected normal(5,12), not fallback.
- `plan/progress.py:255-265`: supported cells emit fresh state; airborne cells emit jump-budget states.
- `plan/progress.py:308-341`: air lookup chooses the best reachable state of the rising/falling phase;
  if only one phase is finite, it supplies the other phase's value too.
- `plan/progress.py:364-374`: runtime lookup chooses support/rise/fall using the supplied support and vy.
- `plan/progress.py:460-499`: destination support bypasses jump limits/reset; step time=max(dx*fx,dy*fy).
- `plan/planner.py:1116-1124`: actual support and velocity are supplied to field.times.
- `plan/planner.py:1679,1703-1728`: 48-frame endpoint T and optional landing completion enter the candidate score.

## What remains unproven

This is not evidence that a 48-frame horizon alone is too short. The heuristic plateau already exists
independently of any world-model rollout, with no unbounded-jump fallback. However, a flat lower bound
can be tolerable if candidates predict a useful ascent. To explain the realized stall completely we still
need one bounded reconstruction of that decision's candidates: do rightward actions reach the fan; what
height/velocity/landing completion is predicted; is a useful candidate absent or pruned; and how do the
risk/memory terms rank it? Increasing the horizon or replacing T with route distance has not been tested.

Reconstruction limit: saved frame-0 supplies26 kill-shape boxes, whereas the initial run field event logged25
static boxes. Current-code reconstruction has244991 edges versus logged245009 (18 fewer), although route
cost87.4, spawnT412, two legs and stallT317.38 match. The pre-frame info hazard snapshot was not recorded
separately. Thus this is exact graph tracing for recorded frame-0 geometry, not a claim of byte-identical
reconstruction of every edge of the historical initial field. No missing snapshot was fabricated.
