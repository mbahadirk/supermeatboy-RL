# 1-14: plateau on a half-block floor — 2026-11-24

## Implemented correction and realtest follow-up

The proposal below was authorized and implemented in ONLYPPO `realpath.py`.
`Planner.progress_visible()` clips the local segment analytically against cached
convex edges derived from the existing tile_shape/solid_polygon owner. RewardPath.value
rejects occluded candidates before accepting their progress. No ray marching, model width,
reward coefficient, jump input or realtest progression change was introduced. Main runtime
was not changed. Reconstructed ledge values now match the diagnostic visible-only values.
10,000 local field queries on that map took0.218s in a warm synthetic check; this is not an
end-to-end FPS claim.

A bounded **learning-on realtest**, not eval, resumed the original aplus_v2 model into
`runs/aplus_v2_1_14_visibility_fix`. Config/source hashes, complete native frames/actions,
wave results and updated model are retained there; harness is `retest.py`, output `retest.log`.
The game initially had to be resumed by the user. Five completed waves,200 agents:
**0 finishes,9 deaths,191 timeouts**. Source checkpoint SHA256 unchanged. The test stopped,
saved and detached; a read-only live check confirmed original timing bytes7c53 restored.
A zero-action stop handshake in partial wave6 is excluded from completed-wave statistics.

First3 waves: all40 agents reached the ledge and none exited toward the lower-left region.
Wave5: one agent later reached x<60,y<20 after being on the ledge; this is an escape-region
metric, not proof of a complete feasible route or finish. Console maximum progress rose
70% ->77%, but routes were rebuilt after wave3 as moving hazards were recognized, so these
percentages are **not a controlled fixed-route improvement measurement**.

The new raw telemetry exposes another strong limitation: on the ledge, jump was requested
for99.91..99.97% of action samples in each wave. Actual hop rate was0.0 in first3 waves and
0.1 in last2. Native applyInput sends held jump state2; a fresh press requires release.
Existing automatic re-press is restricted to airborne wall contact, not standing on a
floor. The trained model mostly holds jump while oscillating on the floor and rarely
releases it to start another jump. Thus the interpolation bug is corrected, but **this
five-wave test did not solve1-14**. No automatic floor re-press or additional penalty was
silently added. Further work should distinguish learned release behavior from route guidance,
rather than claiming the geometry correction alone guarantees recovery.

After the bounded run, a compatibility guard preserves the prior interpolation handling
for dynamic special tiles (keys/crumbles/switches): static route words cannot decide whether
an individual agent has opened/broken them.1-14 has zero such cells, so this guard does not
alter the tested1-14 geometry. It was verified with regression tests, not another live run.
Final validation:96 tests passed (one existing pygame/pkg_resources warning), Python
compilation and diff whitespace checks passed. Tests cover all rotations/flips of half
blocks/slopes, visibility along the open half, boundaries, full walls, open space,
above/below-floor progress separation and dynamic special-tile compatibility.

To continue this corrected training, with1-14 open, from ONLYPPO:
```powershell
python realgame.py --agents 40 --shape-inputs --realtest --wave-seconds 30 --end-after 0 --record-states --load runs/aplus_v2_1_14_visibility_fix/model.pt --name aplus_v2_1_14_continue
```
This is a continuation command, **not a claim that this model already passes**.
See `retest_results.json`, rebuilt by `summarize_retest.py`.

## Observations

`runs/aplus_v2/run_config.json`: 40 agents, 30s waves, realtest, 524 inputs,
fixed native step, learning on. `record_states=false`: there is **no historical
frame replay** for these waves. Wave counts below are a snapshot taken during a
continuing run, not final totals. At diagnosis: 65 completed 1-14 waves, 2600
trials, zero finishes, 653 deaths, 1947 timeouts. Last ten waves: 12 deaths and
388 timeouts. Extending the timeout is not supported as the first fix.

`capture.py` attached a separate, bounded passive Frida observer to the running
game. It read tile words, object coordinates and mode-2 character states; it
made no game-memory writes, native function calls, input changes, training
interruptions or policy updates. Hooks were unloaded and the session detached.
`passive_sample.json` contains 99 snapshots over 24.79 wall seconds (not a full
wave). All 40 characters stayed within x75.09..121.13, y35.86..36.01. The 95th
percentile of absolute vertical velocity is zero. The native goal is at
(32.007, -132.720): they have climbed to the ledge above it and are moving
horizontally there rather than entering the lower chamber.

`positions.png` is a coarse tile occupancy sketch: blue sampled characters,
green spawn, orange goal, red hazard *piece anchor positions*, not live saw
centers or collision extents. Filled cells include the entire bounding cell
of half tiles, so this image is not a shape-accurate collision rendering.

## Reproduced structural defect

`realpath.RewardPath.value()` interpolates progress from nearby unblocked cell
centers. It checks whether a candidate cell is free but does not check whether
the segment to that center crosses the solid part of the character's current
half tile. A standing character at y36 can therefore inherit high progress from
y10, underneath the supporting floor at y20..30. The observation's route
lookahead also uses this progress value, so the defect affects both shaping
and the target features supplied to PPO.

`analyze.py` reconstructs routes using the captured real tile words and goal,
**without hazards/sweeps**. This is not an exact dump of the running trainer's
routes, and the running route choice/hazard field is unknown. It isolates a
geometric defect independent of hazard prediction. `diagnosis.json` preserves
the inputs, route points, candidate centers and diagnostic filtered values.

First reconstructed route, same ledge y36:

| x | Current progress | Winning center | Crosses solid floor? | Progress with solid-crossing candidates rejected |
| --- | --- | --- | --- | --- |
| 60 | 662.02 | (90,10) | Yes | 515.36 |
| 80 | 673.86 | (90,10) | Yes | 495.36 |
| 100 | 682.14 | (110,10) | Yes | 475.36 |
| 110 | 684.00 | (110,10) | Yes | 468.16 |
| 120 | 455.36 | (110,50) | No | 455.36 |

Both reconstructed routes exhibit this leak and a sharp local maximum around
x110, directly inside the observed cluster. Moving left toward the route's
actual opening loses current progress; moving right eventually hits a very
large drop in credited progress. In the diagnostic shape-visible variant,
progress increases consistently toward the left opening instead. This is
strong evidence for misleading progress/target features, **not a live causal
proof that this alone explains every failure**. No full current wave's actions,
rewards or route definitions were recorded. A corrected live run remains needed.

## Minimum proposed change — not applied

At the owner, `RewardPath.value()`, reject local interpolation candidates whose
connection crosses actual solid tile geometry. Use the existing `tile_shape`
orientation/`solid_polygon` owner; do not treat a whole half-tile cell as a wall
for this local visibility test. Cache static shape geometry rather than decoding
it for every agent/frame. The analysis uses sampled line checks as a diagnostic
only; this should not be copied blindly as a costly production loop.

Keep the PPO architecture, observation width, reward magnitudes, stall cap,
waiting behavior, hazard geometry and realtest progression unchanged. This one
correction improves both route progress and the lookahead fed to the policy.
Continue from the latest aplus_v2 checkpoint in a separate run with raw telemetry,
then compare escape from the ledge and finishes, not just time spent alive.
Do not promise the existing learned plateau disappears immediately: PPO may
need to relearn it. No special case for level 1-14, broad death penalty reduction,
larger waiting penalty, world model or action planner is recommended.

Suggested regression coverage if implemented: above/below half-floor credit
separation, flipped/rotated half tiles and slopes, travel within the open part,
and preservation of an unobstructed route's progress. Then a bounded live realtest
run from 1-14 with unchanged checkpoint source; no eval mode required.

Only analysis artifacts and shared notes were written in this investigation.
Runtime files, weights and active trainer configuration were not changed by it.
