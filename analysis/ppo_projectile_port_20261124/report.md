# Shared projectile observations / ONLYPPO inputs — 2026-11-24

User requested main/world-model branch's projectile information in ONLYPPO without
contradicting existing world-model data. The existing main mechanism is **observed shot
births and repeated intervals**, not a raw native shooter cooldown. No native timer
offsets were guessed or added, and no new gameplay experiment was started.

## Single observation contract

`game/projectiles.py` is a pure observation module, present byte-identically in main and
ONLYPPO. It owns aligned `parse_shx`, the existing `Launcher.born` period contract,
spawn-point grouping and current-flight/slot-reuse detection. Main `game.level` re-exports
the parser; `mpc.future` delegates its former Launcher and shot_track code to this owner.
Its future trajectory, template use, world-model and action-planning code remains in main.
ONLYPPO imports no wm/mpc/plan/agent module and copies none of those packages.

Raw messages remain `sh=[id,x,y,radius]`, `shx=[kind,vx,vy]`, with game units, native
seconds and the original alignment. The WM collector's raw ROW dtype, frame schema,
feature producer and existing shards were not changed. PPO vectors and WM training
features are different consumers of the same raw observations; a550-input PPO vector
must not be passed directly to a WM as if it were an old WM feature vector.

ONLYPPO now consumes native `shx` velocity immediately, including on a shot's first
visible frame. Missing/misaligned/unreadable metadata falls back to explicit unknown
rows and the existing displacement estimate. The parser rejects nonfinite metadata;
it never shifts metadata to a different shot after a length mismatch.

## PPO observation extension

Enabled by default in CLI, explicit `--projectile-inputs`; legacy widths use
`--no-projectile-inputs`. Existing base282 and optional242 shape columns remain in order.
26 columns are appended: two observed launcher sources,13 fields each:

`[present, dx/200, dy/200, vx/600, vy/600, radius/50, period/120,
age/120, eta/120, period_known, phase_known, velocity_known, kind/5]`

Kind is-1 when unknown. Times are native wave frames, not wall seconds. Sources with an
established observed period are selected before unconfirmed short tracks, then by distance
with a stable key tie-break. No unconfirmed period is exposed as known. Missile shrapnel
kind16 is excluded from launcher-source feature slots.

The same existing main contract applies: one observed birth supplies a source but no
period; another sufficiently separated birth establishes it. Period survives wave reset,
last-birth phase does not. A new wave needs a fresh observed birth before phase is known.
When a source misses its expected firing by more than the existing1-frame tolerance,
phase becomes unknown. A new level clears all source evidence. No remembered first-shot
time from a different wave is presented as a verified current-wave countdown.

This is a projection of main's observed period knowledge, **not its full future-world
recording/template forecasting**. No future trajectories or action choices are computed
in ONLYPPO. Native first-shot timing and reactive/homing changes that have not yet been
observed remain uncertain. Before the first verified phase, adding these features cannot
make that uncertainty disappear.

## Checkpoint compatibility

With shape inputs:524 ->550; without them:282 ->308. Older137-input checkpoints still
retain their existing prefix when widened. New first-layer columns start at zero for
both actor and critic, preserving their initial outputs for arbitrary new-feature values.
Learning can then assign weights to them. A308-input clock checkpoint widened to550
correctly relocates its clock columns after the new shape maps; silently dropping clock
or shape features is refused. Save metadata records observation width, projectile flag
and `observed-shot-clock-v1`; run_config also hashes the new source owner.

Original user checkpoints were not rewritten. New inputs do **not** instantly improve
an existing policy: the initial added weights are zero, and PPO training is required.

## Verification

- ONLYPPO full suite:106 passed (one installed pygame/pkg_resources warning).
- Main focused tests:41 passed, covering shot futures, cache parity, lasers, observed
  clock equivalence and unchanged WM collector sh/shx records.
- Real1-20 replay:601 frames of recorded sh/shx, checked every frame against the existing
  FutureWorld launcher period/last-birth/id/radius. Four established sources have119-frame
  periods. The historical slot detector also sees four unconfirmed short tracks; they
  do not displace the verified sources in the PPO feature slots.
- Tests cover missing/misaligned native metadata, first-frame velocity reaching actual
  PPO inputs, zero-column actor/value invariance, shape/clock migration, wave/level reset,
  expired phase, mid-flight attachment, shrapnel filtering and raw-row immutability.
- Python compilation and git diff whitespace checks passed.

`verify_replay.py` rebuilds `replay_parity.json`. This is offline input-contract verification,
not a live completion-rate or real-time FPS benchmark. Native game code was not modified
by this port and no live training/eval was started for it.

## Continue in ONLYPPO

Save/stop the currently running trainer with Ctrl+C, leave game on1-20, then from ONLYPPO:

```powershell
python realgame.py --agents 40 --shape-inputs --projectile-inputs --realtest --wave-seconds 30 --end-after 0 --batch 2048 --record-states --load runs/aplus_v2_batch2048/model.pt --name aplus_v2_projectiles
```

New weights/logs go to runs/aplus_v2_projectiles. Current records remain intact; inspect
end-corridor survival and contact timing after the model learns to use the appended inputs.
