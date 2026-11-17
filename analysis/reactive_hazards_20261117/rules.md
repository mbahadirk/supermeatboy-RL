# Homing laser eyes and launcher2 missiles: per-frame rules (2026-11-17)

Source: SuperMeatBoy.exe (image base 0x400000), static analysis only (Ghidra pseudocode of FUN_004d4550 plus
capstone disassembly). Nothing was run against the game. Levels: 4-16 (two homing eyes, one launcher2).
Checks: `python analysis/reactive_hazards_20261117/verify.py` (run from smbrl).

Tags: **[code]** read from the disassembly. **[log]** checked against recorded waves (numbers given).
**[hyp]** inferred, not checked. All coordinates are game units with y up; angles are radians.

## 0. Frame order and the target

- **[code]** One game frame, 0x4aa9b1: the world update FUN_004d4550 runs first. That covers lasers, launchers,
  missiles and shrapnel. The character updates run after it (vtable call at 0x4aa9c6). Kill tests
  (FUN_004cd350) run inside each character's update, so they use this frame's beams and shots.
- **[code]** Both homing eyes and missiles aim at FUN_004aac70(), which is the game's main character
  `[game+0x28]` (or a replay character), at its `+0x84/+0x88`. With realgame.js, that character is hidden.
  At the end of its update it is parked on the leading agent's position *after* the agent's step.
- **[log]** Net effect: the world update of frame f aims at the leader's position as recorded in that frame's
  row (`rows` frame f, read before step f). This is lag 0; lags 1 and 2 fit far worse (section 1.6).
- dt = FUN_00450d10() is the frame time in seconds. It is 1/60 under lock_step (`dt` 16.667 ms in the logs).

## 1. Homing laser eye (`hominglazer*`, laser kind 1)

### Data

- **[code]** The level build at 0x4c96c7 puts each piece whose sprite is hominglazeron/off/wait (world
  `+0x2fd4/+0x2fd8/+0x2fdc`) into the laser array. That array is the ResizeableArray at world+0x2ce8: u16
  count `+0x2d00`, u16 max `+0x2d02`, data `+0x2d10`, 0x24 bytes per entry.
- **[code]** Entry layout:

  | Offset | Field |
  |---|---|
  | `+0` | piece |
  | `+4` | timer (counts up) |
  | `+8` | state |
  | `+0xc` | kind (1 = homing, 0 = plain) |
  | `+0x14/+0x18` | beam start |
  | `+0x1c/+0x20` | beam end |

  At build the sprite is set to "off", the kind to 1, and the timer and state to 0.
- **[code + log]** Piece parameters are the six dwords after the type in the .smbl record, mapped to piece
  `+0x34..+0x48`. Both 4-16 eyes have the same values except `+0x38`:

  | Offset | 4-16 value | Meaning |
  |---|---|---|
  | `+0x34` | 1.0 | animation speed |
  | `+0x38` | 10.3 / 53.45 | not used by the laser code |
  | `+0x3c` | 0.0 | off time (s) |
  | `+0x40` | 999.0 | on time (s) |
  | `+0x44` | 0 | — |
  | `+0x48` | 18 | turn divisor (int) |

  The record parser in tools/smbl_extract.py starts records 12 bytes late. Each "tail" there is really
  the next record's first 12 bytes. The turn divisor 18 is confirmed by the logs: k = 1/18 fits and the
  default 0.1 does not (section 1.6).
- **[code + log]** The piece angle `+0x2c` is in radians and keeps whole turns. The 4-16 eye is placed at
  116.50115 = 3.4038 + 18·2π. The hz eye row's angle column is this raw value; the "constant offset"
  noted earlier was whole turns.
- **[log]** The respawn (realgame's main V_RESPAWN) restores the placed angle: 284/284 eye-waves show it on
  frame 0. The reset loop at 0x4d286c (inside the level-reset function, which also resets launchers at
  0x4d23d0) sets state 0, timer 0 and the sprite to off.
- Other levels with gameplay eyes or launcher2 (layer byte 0x30):
  - 4-20: eye params (1, 6.3, 0, 999, 0, 18) and a launcher2 (1, 0, 0, 0, 0, 0)
  - 4-15: one launcher2 (1, 0, 0, 0, 0, 0), no eyes
  - Frame 0's beam row is stale (computed before the reset). Simulate from frame 1.

### 1.1 Aim (every frame, in every state), 0x4d8573..0x4d894d [code]

```
P  = piece pos (+0x14,+0x18);  th = piece +0x2c
D  = target - P;  if |D| > 0: D *= 1/|D|                (float32)
F  = (cos th, sin th)                                    (M(th)·(1,0), FUN_00494710 / FUN_00494c50)
a  = float32(acos(D·F))                                  (dot in x87, then to double; NaN -> no turn)
X1 = c(sin(th + a));  X2 = c(sin(th - a));  T = c(sin(atan2(D.y, D.x)))
     where c(x) = ceil(float32(x)*1000)/1000             (0x5d1df0 = ceil: its x87 fallback uses
                                                          control word 0x1b3f, round up)
k  = 1/piece[+0x48] (int) if nonzero, else 0.1
if X1 == T:   th += k*a
elif X2 == T: th -= k*a
else:         th unchanged
```

- The turn is not scaled by dt: each frame removes 1/18 of the remaining angle.
- Quirk: the side test compares *sines* rounded up to 1/1000.
  - The mirror case (sin(th+a) = sin(π−T)) can pick the wrong side.
  - Rounding can also make neither branch match, so the eye skips that frame.

### 1.2 Beam geometry (after the aim, same frame), 0x4d8b5a / 0x4d8bb3..0x4d8e38 [code]

```
L    = sqrt((W*20)^2 + (H*20)^2)            W, H = level tile size; 4-16: 1166.19
v    = (L, 0), negated when (piece+0x10 >> 10) & 3 is 1 or 3      (flip bits; 0 for the 4-16 eyes)
end  = P + M(th)·v = P ± L·(cos th, sin th);  start = P (the eye centre, no offset)
beam end = FUN_004663f0(start, end)          the level grid at [[0x7035f4]+0x24]
```

FUN_004663f0 is a sampled ray cast, not an exact one:

- n = ceil(len/10) samples, spacing s = len/n (≈ 9.97).
- Samples are taken at start + i·s·dir for i = 0..n−1, starting at the eye itself.
- Cell index: (floor((x−ox)/20), floor((y−oy)/20)), clamped to the grid.
- The first sample whose cell word has low byte ≠ 0 *is* the beam end, so the beam overshoots into the wall
  by up to one step.
- With no hit, the end is P + L·dir.

Crumble blocks (word 0x…164) block the beam until they break.

The homing kind recomputes the beam in every state. The plain kind recomputes it only in states 1 and 2.

### 1.3 State cycle [code]

The timer `+4` increases by dt every frame (0x4d8529).

| From | To | Condition | On transition |
|---|---|---|---|
| 0 (off, not lethal) | 2 (wait, not lethal) | timer > p+0x3c (strict) | timer = 0; piece+0x30 = 0; sprite = hominglazerwait |
| 2 (wait) | 1 (on, lethal) | piece+0x30 ≥ world[+0x2fd0] · p+0x34 | timer = 0; piece+0x30 = 0; sprite = hominglazeron |
| 1 (on) | 0 (off) | timer > p+0x40 (strict) | timer = 0; sprite = hominglazeroff |

- **0 → 2** is at 0x4d8952. With p+0x3c = 0 this happens on the first update after the reset.
- **2 → 1** is *not* in the world update. It is at 0x4d032f, in the function containing 0x4d0218 (there
  `edi` = world+8). The rest of that function draws the beam, so it is probably the per-piece draw callback
  **[hyp]**.
  - world[+0x2fd0] is the wait animation's length, read from the sprite bank at world init (0x4be4b7 /
    0x4be51d). It comes from the game's data files, not the exe.
  - Both lazerwait and hominglazerwait write the same `+0x2fd0`, and the homing one wins when present.
  - Where piece+0x30 advances (animation time) was not located.
- **1 → 0** writes are at about 0x4d92ac..0x4d931b; 0x4d9324 is the loop label. In 4-16 the on time is
  999 s, so in practice the beam never turns off.
- **[log]** The first lethal frame is 67 in all 282 eye-waves that lived that long (2 eye-waves ended by frame
  37). No on → off was seen in 11,314 on-frames. So in 4-16, off lasts one update and the wait about 66
  frames (~1.1 s, which suggests a 33-frame animation at 30 fps **[hyp]**). The beam then stays lethal
  for the rest of the life.

### 1.4 Lethality and kill width [code + log]

- Only state 1 kills. FUN_004cd350 at 0x4cd42b tests entries with `+8 == 1` through FUN_004cbd40(char, char
  shape, entry). The beam has zero width; the character's extent is what matters. With S = start,
  E = end, d = E − S, the line side function is side(p) = cross(d, p − S):
  1. Return no hit if |c − S| > |d|, where c is the character centre `+0x84`. This is centre distance, not
     box.
  2. Return no hit if (c − S)·d < 0 (behind the eye).
  3. **Swept x.** Take p0 = `+0x8c` (position before the move) and p1 = c. Offset both x by +h if vx > 0
     (−h if vx < 0, 0 if vx = 0). It is a hit if side() has strictly opposite signs at the two points.
  4. **Swept y.** The same with the y offset by the sign of vy.
  5. **Static cross.** Test (c.x ± h, c.y), then (c.x, c.y ± h), for strictly opposite sides.
  - Steps 3–5 test whether the beam line crosses that segment (the leading edge, then a cross through the
    centre). For a still character the reach is δ < h·max(|sin β|, |cos β|), where δ is the perpendicular
    distance to the beam and β is the beam angle. That is between h/√2 and h, not a full box.
- h comes from the character shape: box type 0 reads shape+0x8c/+0x90; circle type 1 reads the radius.
- **[log]** With h = 6 (realpath.BODY), the test explains **141/141 deaths** in 4-16 (online_check11) with
  **0 hits on surviving steps** (14,999 steps checked). h = 4 explains only 41, and h = 8 gives 114 false
  hits. So h = 6 is pinned. Every 4-16 death in this run was a beam kill.

### 1.5 Plain lazer (kind 0, lazeron/off/wait +0x2fc4/+0x2fc8/+0x2fcc) [code]

- Same entry layout and state cycle as the homing eye, with sprites lazeroff/lazerwait/lazeron and the same
  wait length world+0x2fd0. There is no aiming.
- The beam is P + R(piece angle)(±L, 0), ray-cast as above. It is recomputed only in states 1 and 2; in
  state 0 the stored segment is stale but harmless.
- Lethal only in state 1, with the same kill test.

### 1.6 Log verification, homing eye (runs/online_check11, 4-16, 142 waves, 2 eyes, 30,000 eye-frames)

Beam direction is measured from the recorded beam midpoint minus the eye centre. Rounding to 0.01 gives a
precision of about 3e-4 rad.

**Free-run from the placed angle** (|error| in rad):

| Rule | Median | p95 | Within 0.001 |
|---|---|---|---|
| lag 0, k = 1/18 | 0.00005 | 0.0016 | 94.0% |
| lag 0, k = 0.1 | 0.032 | 0.38 | 6% |
| lag 1, k = 1/18 | 0.0054 | 0.052 | 17% |
| lag 2, k = 1/18 | 0.011 | 0.10 | 10% |

**One step from the measured previous direction** (lag 0, 1/18): median 0.00005, p95 0.00021, 99.9% within
0.001. 39/29,716 steps are off by more than 0.003 rad (up to one full turn step, 0.055).

- These are frames where the game skipped a turn our emulation makes, or made one we skip. They are
  deterministic: the same frames repeat across waves with the same inputs.
- **[hyp]** Likely mechanism:
  - `piece+0x2c` keeps whole turns (≈118 and ≈41 rad), where the float32 spacing is 7.6e-6 and 3.8e-6.
  - `th ± a` is rounded on that grid before the sine, so X1 and T can straddle a 1/1000 ceil boundary on
    about 0.1% of frames.
  - The one-step test seeds `th` from the wrapped measured direction, so it cannot reproduce this.
- After a skip, the error decays by a factor of 17/18 per frame.

**Beam length** (sampled ray cast at the measured direction vs the recorded length):

- 29,070 frames: median error 0.0035, p95 0.0047. Signed mean −0.03.
  - It includes 3 frames past the broken crumble where even the crumble-free prediction is one sample
    (≈10 units) short, which explains the max of 329.
- 930 more frames match only once the crumble block at cell (24,10–11) has broken. The wave files do not
  carry tile changes.

**Timing:** see section 1.3.

## 2. Launcher2 (kind 5) and missile2

### Entry layout [code]

- Launcher array: data at world+0x2c40, u16 count `+0x2c30`, 0x9d4 bytes per entry.
- Launcher fields:

  | Offset | Field |
  |---|---|
  | `+0` | piece |
  | `+4` | fire timer (counts down) |
  | `+8` | start delay (counts down) |
  | `+0x10` | state (0 idle, 1 shooting, 2 reloading) |
  | `+0x14` | kind |
  | `+0x18` | sees-target flag |

- 10 shot slots, 0xf8 bytes each. Slot i's physics body B = entry + 0x24 + i·0xf8. Field offsets relative
  to the entry, at slot i (add i·0xf8):

  | Offset | Field |
  |---|---|
  | `+0xa8` | position |
  | `+0xb0` | previous position |
  | `+0xc0` | velocity |
  | `+0x98/+0x9c` | drag terms (B+0x74/0x78) |
  | `+0xe0` | sprite angle |
  | `+0xf4/+0xf8` | acceleration (B+0xd0) |
  | `+0xfc` | state: 0 flying (the only lethal state), 1 exploding, 2 free |
  | `+0x100` | timer |

- Kind map, from the sprite ids each kind uses:
  - 0 shooter1 (+0x2f04..)
  - 1 (+0x2f24..)
  - 2 (+0x2f44..)
  - 3 launcher (+0x2f64 launcher, +0x2f68 shoot, +0x2f6c reload, +0x2f70 missile, +0x2f74 missileexplosion)
  - 4 (+0x2f9c..)
  - **5 launcher2** (+0x2f80 launcher2, +0x2f84 launcher2shoot, +0x2f88 launcher2reload, +0x2f8c missile2,
    +0x2f90 missile2explosion; shoot/reload animation lengths at +0x2f94/+0x2f98)
- Build (0x4c99c1) and reset (0x4d23d0..0x4d2797):
  - kind = 5, state 0, sprite launcher2
  - start delay `+8` = p+0x44 / 1000 s
  - fire timer `+4` = p+0x3c, or 1.0 s when p+0x3c == 0
  - all slots state 2
  - 4-16 launcher2 params: p+0x34..+0x48 = 1, 0, 0, 0, 0, 0, so no delay and a 1.0 s timer.

### 2.1 Launcher per frame (FUN_004d4550, 0x4d5d74..0x4d84ba) [code]

The order within each frame is:

1. `+8 -= dt`. Nothing happens until it is ≤ 0. In 4-16 this is the first frame.
2. **Slot loop**, i = 0..9:
   - Flying slots run the missile update (2.2).
   - Every *non-flying* slot (kinds 1/3/5): if `+0x18 == 1`, set the piece angle `+0x2c` =
     atan2(target − launcher) − π/2, with `target` = FUN_004aac70 `+0x84`. This is instant tracking, as long
     as one slot is free and the last line-of-sight check passed.
3. `+4 -= dt`.
4. **State machine:**

   | State | Condition | Action |
   |---|---|---|
   | 0 | `+4 ≤ 0` | Line-of-sight check (below); then piece+0x30 = 0 and `+4` = world[+0x2f78] |
   | 1 | piece+0x30 ≥ world[+0x2f94] | state 2; piece+0x30 = 0; sprite launcher2reload; **fire** FUN_004c3400(5) |
   | 2 | piece+0x30 ≥ world[+0x2f98] | sprite launcher2; `+4` = p+0x3c or 1.0; state 0 |

   - **State 0 line of sight** (0x4d75e2..0x4d7775): `old` = `+0x18`, then `+0x18` = FUN_004663f0(launcher
     centre → target `+0x84`). This uses the same sampled ray as the beam and returns 1 when no sample is
     solid. Kinds 0/2/4 skip the ray and force both values to 1.
     - **It fires only if `+0x18 == 1` and `old == 1`**: state 1, sprite launcher2shoot. Otherwise it stays
       in state 0.
     - A launcher that did not see the target needs two clear checks in a row, world[+0x2f78] apart.
       Tracking (the angle) starts after the first one.
     - `+0x18` is 0 at level build and the respawn reset does not clear it. After a respawn it keeps its
       value from the previous life **[code: no write found in the reset]**.
   - **No range condition**: only the tile line of sight.
   - **State 0 timer quirk:** `+4` is reset to world[+0x2f78] even for kind 5. That is the *kind-3
     launchershoot* animation length.
   - **Fire period** with steady line of sight = shoot animation + reload animation + 1.0 s (4-16). State 2 → 0
     sets `+4` = 1.0, and the check that follows fires at once because both flags are 1. The world[+0x2f78]
     re-check only adds time after a failed check. All animation lengths come from the data files (`.am`
     palettes); they were not decoded **[open]**.

### 2.2 Spawn, FUN_004c3400(5) [code]

- Takes the first slot with state 2. If none is free, nothing is fired.
- dir = (cos(θ + π/2), sin(θ + π/2)), where θ = launcher piece angle. That is the direction to the target at
  this frame, from the slot loop above.
- pos = launcher centre + 20·dir; vel = 200·dir.
  - The other kinds use the same code with different speeds: 360 for kinds 0/1/3, 400 for kind 2,
    p+0x48 + 400 for kind 4.
- Drag terms 0, timer 0, state 0 (flying). The spawn happens after the slot loop, so the missile does not move
  on its spawn frame.
- Kind 3 loops five times (θ + j·π/2, j = 0..4; the fifth repeats the first direction) — a quirk.
- **[log]** The shared spawn and integration path is confirmed on straight shots (3-1/3-4, online_check01):
  - All 50,982 per-frame displacements are exactly 6.000 (360 u/s · 1/60).
  - 1,758/1,995 births are at exactly 20.0 from the nearest hazard centre; the other 237 are at 5.3, probably
    a different nearest hazard.

### 2.3 Missile update (flying slot, every frame, in slot order), 0x4d5ef0..0x4d7230 [code]

1. Previous position := position.
2. If the tile cell of the position is outside the grid (FUN_00465720), the state becomes 2: the missile is
   removed silently.
3. For each buttholekiller piece (world array +0x2c4c, type +0x2ef8): if it overlaps a circle of r = 10 at
   the missile, the state becomes 2 (removed silently). There are none in 4-16.
4. **Steering:**
   ```
   d = target(+0x84) − pos;  d /= |d|;  v̂ = vel/|vel|;  drag terms := 0
   acc = 750 · (2 − v̂·d) · d        (|v̂·d − 1| + 1; 750 = 0x443b8000)
   ```
   The acceleration is 750 when heading at the target and 2250 when heading away. There is no speed cap and
   no drag (both drag terms are zeroed every frame).
5. **Sprite angle** = atan2(v̂) − π/2. Rendering only.
6. **Integrate**, FUN_004751a0(B, dt), semi-implicit Euler:
   ```
   vel += acc·dt;  pos += vel·dt
   ```
   Drag `vel −= vel·(B+0x78·dt)` applies only when nonzero, and it is zero here. The acceleration is cleared
   afterwards.
7. **Tile collision**, FUN_00466680(B), with world+0xdb4 |= 0x81 during the call. If it reports a hit
   (result +0x10 or +0x14 nonzero):
   - state 1, timer 0, random sprite angle, screen shake FUN_004a2fb0(5.0, 0.2)
   - kind 5 only: **explosion** FUN_004d30a0(pos, result) (2.4)
   - The missile body's collision shape and size were not decoded **[open]**. The slot body is built with
     shape pointer +0x44 = 0, possibly a point or swept test **[hyp]**.

State 1 → 2 happens in the shot draw function (0x4d3a8c) when the explosion animation finishes. **State 1
does not kill.**

No lifetime check on a flying missile was found (slot `+0x100` is reset at spawn and landing; no reader
was located). A missile flies until a tile, the grid edge or a buttholekiller.

### 2.4 Kills

**Missile.** FUN_004cd350 at 0x4cd8e8: every slot in state 0 is a circle against the character shape
(FUN_00476720).

- The radius comes from a switch on kind: 0/3 world+0x2f20, 1 +0x2f40, 2 +0x2f60, 4 +0x2fb8. Kind 5 has no
  case, so the radius is the default **1.0** × launcher piece scale `+0x24`.
- Launcher2 missiles therefore kill only on near-contact: a circle of r = 1 against the character box, about
  |dx|, |dy| < 7 for h = 6 **[hyp]** on exact overlap semantics.
- For kind 5 the missile is not consumed by the hit. Only kind 3 switches to state 1 on hitting the
  character.

**Shrapnel.** FUN_004d30a0, kind 5 only:

- 6 fragments are added to the world array at +0x2c80 (u16 count +0x2c98, data +0x2ca8, 0xe0 bytes each).
- They start at the missile position and move at 300 u/s (0x67e0b8) at angles base + j·π/5, j = 0..5. That
  is a 180° fan in 36° steps.
- The base angle comes from the collision flags: result+0x10 = 1 → π, 2 → 0; then result+0x14 = 4 → −π/2,
  8 → +π/2 (the x flag overrides). Read as "fan away from the wall hit" — a ceiling, floor, left wall or
  right wall — **[hyp]**; the flag meaning was not checked.
- **Movement**, FUN_004d9760 (called right after the launcher loop, so a fragment moves 5 units on its spawn
  frame): straight line, no gravity, no drag (the element is zero-filled). It is removed on its first tile
  collision. There is no lifetime.
- **Lethal**, FUN_004c7e50 from FUN_004cd350 at 0x4cd49e: a circle of **r = 10** at the fragment centre
  against the character shape. The static shape at 0x6f08b4 is made a circle (type nibble 1) at startup by
  0x442e90; its radius 10.0 is in .data. It is gated by world+0xdb4 bit 2, the general "hazards
  kill" bit, which realgame.js does not clear.

**Can a missile be lured into a wall?** Yes, per the code. It steers only by acceleration, with no speed cap,
and dies on any tile it touches. Its turn radius grows with speed (v²/a, with a = 750..2250). A fast missile
passing close to a wall while the target is around a corner will hit it. The wall impact then throws the
shrapnel fan, which is the lethal part near walls.

### 2.5 Logs

- **launcher2 missiles have never been recorded.** realgame.js `shots()` keeps only kinds with an entry in
  `SHOT_R` (0, 1, 2, 3, 4). Kind 5 is skipped, so there are no `sh` entries for missiles in any run.
  Observed: 0 shot entries in 4-15 (1 wave) and 4-16 (142 waves) of online_check11.
- The shrapnel array (+0x2c80) is not recorded either.
- In 4-16 (online_check11) the launcher would not have fired anyway. Its line-of-sight ray from
  (−11.2, 173.6) to the agent was blocked on all 15,283 recorded agent frames: the agent never got above the
  lower area.
- The missile law (2.3–2.4) is therefore **code-only**. The spawn offset/speed path and the integrator are
  shared with straight shots, which are verified (2.2).

## 3. Open questions

1. Animation lengths, which live in the `.am` palettes (imported/smbl_raw/palettes/*hellobstacles.am; the
   record format is not decoded). They give:
   - hominglazerwait: 66 frames measured
   - launcher2shoot, launcher2reload (fire period)
   - missile2explosion (state-1 time)
   - launchershoot (world+0x2f78, used as the kind-5 state-0 re-check timer)

   Where piece+0x30 (animation time) advances, and whether by dt or dt·speed, was not located.
2. Missile body collision shape and size in FUN_00466680, and the exact meaning of the result flags
   (+0x10/+0x14 → shrapnel fan direction).
3. The exact circle-vs-box overlap rule of FUN_00476720 for the r = 1 missile and r = 10 shrapnel (assumed
   ordinary overlap with the h = 6 box).
4. Bit-exact reproduction of the eye's 1/1000 sine comparison (0.13% of frames differ by one turn step).
5. To verify the missile law on logs: add kind 5 (radius 1.0 × scale) to `SHOT_R` and record the +0x2c80
   shrapnel. That is a realgame.js change, outside this task.
