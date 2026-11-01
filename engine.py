"""Clean-room 2D platformer engine with Super Meat Boy-like movement.

ORIGINAL implementation. Contains no code or assets from the original game binary;
only well-known platformer mechanics (run/jump/wall-slide) reimplemented from scratch.

Deterministic, fixed 60 Hz timestep. Screen coordinates: +X right, +Y DOWN.
(The RE'd replay format uses +Y up; flip the sign at the observation boundary if
you want to match it. Internally +Y down keeps tile collision simple.)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math

TILE = 24  # pixels per tile cell


@dataclass
class Physics:
    """Tunable movement constants (to be refined against static RE of the binary)."""
    dt: float = 1.0 / 60.0
    gravity: float = 2600.0          # px/s^2 (down)
    max_fall: float = 1700.0
    run_accel: float = 5200.0        # ground horizontal accel
    air_accel: float = 3400.0
    walk_max: float = 300.0          # max horizontal speed without "run"
    run_max: float = 560.0           # max horizontal speed with "run" held
    ground_friction: float = 6000.0  # decel when no horizontal input (ground)
    air_friction: float = 500.0
    jump_speed: float = 820.0        # initial upward speed on jump
    jump_cut: float = 0.45           # vy *= this when jump released while rising
    coyote_time: float = 0.08        # s: can still jump shortly after leaving ground
    jump_buffer: float = 0.10        # s: jump press is remembered before landing
    wall_slide_max: float = 260.0    # clamped fall speed while wall-sliding
    wall_jump_x: float = 430.0       # horizontal push off a wall
    wall_jump_y: float = 820.0       # upward speed on wall jump
    player_w: float = 18.0
    player_h: float = 22.0
    wind_accel: float = 0.0          # px/s^2 upward while overlapping a wind cell (0 = off)
    wind_max_up: float = 0.0         # px/s cap on upward speed from wind
    # optional mechanics (None / 1.0 keep the original behaviour)
    walk_accel: Optional[float] = None               # accel (ground and air) when "run" is not held
    turn_mult: float = 1.0                           # accel multiplier when input opposes velocity
    release_mult: Optional[float] = None             # no-input decel = this x current accel (replaces frictions)
    wall_slide_gravity_scale: Optional[float] = None  # scale gravity on a wall (replaces wall_slide_max clamp)
    gravity_after_jump: bool = False                 # apply this frame's gravity after a jump impulse
    jump_min_frames: Optional[int] = None            # variable jump: from this many frames after takeoff,
                                                     # not holding jump while rising sets vy = 0 (replaces jump_cut)
    ground_release_stop: bool = False                # no horizontal input on the ground: vx = 0 at once
    air_release_keep: bool = False                   # no horizontal input in the air: keep vx (clamped to the
                                                     # current walk/run max) instead of air friction
    wall_jump_min_frames: Optional[int] = None       # like jump_min_frames, for wall jumps (None: same value)
    jump_release_frames: int = 1                     # a jump press counts only after this many released frames
    ground_turn_lag: bool = False                    # on the ground, the first frame of a new direction still
                                                     # accelerates along the old facing


def smb_measured() -> Physics:
    """Preset from measurements of the user's own copy (RE notes), scaled to this engine's TILE.

    The game uses 20-unit tiles; k = TILE / 20 converts its units/s to px/s here.
    Values come from tools/capture/smb_calibrate.py over ~51k live frames (Meat Boy, 8 levels).
    The character file's 13 / 8 are per-frame accelerations (run / walk); top speed is 30x that.
    Confidence: [measured] read from live velocity changes; [lower bound] only bounded by data;
    [tentative] not isolated by the data yet.
    """
    k = TILE / 20.0
    return Physics(
        gravity=550.0 * k,                # [measured] dvy = -550/60 per frame, 15.7k frames
        max_fall=1000.0 * k,              # [lower bound] no terminal speed seen; fastest fall 880
        run_accel=780.0 * k,              # [measured] 13/frame with run held (ground and air), 1.6k+4.3k frames
        air_accel=780.0 * k,
        walk_accel=480.0 * k,             # [measured] 8/frame without run, 1.6k+2.9k frames
        run_max=390.0 * k,                # [measured] plateau 13x30, 3.5k frames
        walk_max=240.0 * k,               # [measured] plateau 8x30, 1.9k frames
        turn_mult=2.0,                    # [measured] reversing changes vx at 2x accel (1560 / 960)
        ground_release_stop=True,         # [measured] scripted trials: releasing on the ground stops at once (390 -> 0)
        air_release_keep=True,            # [measured] releasing in the air keeps vx, clamped to walk max 240
        ground_turn_lag=True,             # [measured] ground turns: first frame still accelerates the old way
        jump_speed=300.0 * k,             # [measured] vy after takeoff 290.8 = 300 - 550/60, 233 jumps
        wall_jump_min_frames=1,           # [measured] wall-jump taps: vy cut the frame after release
        jump_buffer=1.0 / 60.0,           # [measured] in-game trace: a press 2 frames before touching a wall is lost
        jump_release_frames=2,            # [measured] wall trials: a re-press after 1 released frame is ignored, 2+ works
        jump_min_frames=5,                # [measured] scripted pad trials: holding jump 1-5 frames rises exactly 5
                                          # frames, holding n > 5 rises n; on release vy is set to 0
        gravity_after_jump=True,          # [measured] takeoff frame shows vy = impulse - one frame of gravity
        wall_jump_x=300.0 * k,            # [measured] 300 / 287 (=300-13) / 240 (walk clamp), 150 wall jumps
        wall_jump_y=350.0 * k,            # [measured] 344.5 = 350 - wall-gravity frame, 148 wall jumps
        wall_slide_gravity_scale=0.6,     # [measured] dvy = -330/60 on walls, 3.0k frames (char file 0x218 = 0.4)
        player_w=12.0 * k,                # [tentative] character half-extent 6 units
        player_h=12.0 * k,
        wind_accel=890.0 * k,             # [tentative] 2-4 replays only; too few live frames to measure
        wind_max_up=450.0 * k,            # [tentative] peak live vy ~449 in 2-4
    )


@dataclass
class Action:
    left: bool = False
    right: bool = False
    jump: bool = False
    run: bool = False


class World:
    """One episode on one level. Call reset() then step(action) at 60 Hz."""

    def __init__(self, level, phys: Optional[Physics] = None):
        self.level = level
        self.p = phys or Physics()
        self.reset()

    # ---- lifecycle -------------------------------------------------------
    def reset(self):
        self.x, self.y = self.level.spawn  # top-left of player AABB
        centre = self.level.spawn_center
        if centre:  # exact spawn point (imported levels): centre this physics' AABB on it
            self.x, self.y = centre[0] - self.p.player_w / 2, centre[1] - self.p.player_h / 2
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = self.level.spawn_grounded
        self._skip_gravity = self.level.spawn_grounded  # the game's first live frame applies no gravity
        self.on_wall = 0          # -1 left wall, +1 right wall, 0 none
        self.facing = self.level.spawn_facing
        self.dead = False
        self.won = False
        self.steps = 0
        self.clock = 0           # frames since the level started (moving hazards); kept across restore()
        self._touched = {}       # crumble cell -> clock of first touch
        self._gone = frozenset() # crumble cells that have broken
        self._portal_lock = -1   # portal the player arrived at (no re-teleport until it leaves)
        self._plat_boxes = self._platform_boxes(0)   # (x0, y0, x1, y1) of each moving platform now
        self._on_plat = -1       # platform the player stood on last frame
        self._coyote = 0.0
        self._jump_buf = 0.0
        self._prev_jump = False
        self._jump_frames = -1   # frames since the last takeoff (-1: not in a jump)
        self._jump_min = None    # minimum rising frames of the current jump (ground or wall)
        self._jump_up = 99       # consecutive frames with jump released
        return self.state()

    def snapshot(self) -> dict:
        """Complete dynamic state (everything but the level and physics), for restore()."""
        snap = {k: v for k, v in vars(self).items() if k not in ("level", "p")}
        snap["_touched"] = dict(self._touched)
        return snap

    def restore(self, snap: dict):
        self.__dict__.update(snap)
        self._touched = dict(snap.get("_touched", {}))
        return self.state()

    # ---- collision helpers ----------------------------------------------
    def _solid_overlap(self, x: float, y: float, full_only: bool = False) -> bool:
        """Player AABB at (x, y) overlaps a solid cell; shaped cells (level.shapes) count only where their
        solid part is hit. full_only ignores shaped cells."""
        p = self.p
        x1, y1 = x + p.player_w - 1e-6, y + p.player_h - 1e-6
        c0, c1 = int(math.floor(x / TILE)), int(math.floor(x1 / TILE))
        r0, r1 = int(math.floor(y / TILE)), int(math.floor(y1 / TILE))
        solids, shapes = self.level.solids, self.level.shapes
        for bx0, by0, bx1, by1 in self._plat_boxes:
            if x < bx1 and x1 > bx0 and y < by1 and y1 > by0:
                return True
        gone = self._gone
        for c in range(c0, c1 + 1):
            for r in range(r0, r1 + 1):
                if (c, r) not in solids or (c, r) in gone:
                    continue
                sh = shapes.get((c, r))
                if sh is None:
                    return True
                if not full_only and self._shape_hit(sh, c, r, x, y, x1, y1):
                    return True
        return False

    @staticmethod
    def _shape_hit(sh, c, r, x0, y0, x1, y1) -> bool:
        """Box [x0,x1]x[y0,y1] (y down) against a partial cell; the solid edge is linear across the cell."""
        kind, a, b = sh
        lx0 = max(x0 - c * TILE, 0.0) / TILE
        lx1 = min(x1 - c * TILE, TILE) / TILE
        if kind == "XL":                                         # vertical half block, solid left half
            return lx0 < a - 1e-6
        if kind == "XR":                                         # vertical half block, solid right half
            return lx1 > 1.0 - a + 1e-6
        reach = max(a + (b - a) * lx0, a + (b - a) * lx1)       # tallest solid extent under the box
        if kind == "F":                                          # solid from the cell bottom up to reach
            box_bottom = ((r + 1) * TILE - min(y1, (r + 1) * TILE)) / TILE
            return box_bottom < reach - 1e-6
        box_top = (max(y0, r * TILE) - r * TILE) / TILE          # solid from the cell top down to reach
        return box_top < reach - 1e-6

    def _hazard_overlap(self, x: float, y: float) -> bool:
        p = self.p
        # shrink a little so grazing a corner is not instant death
        m = 3.0
        c0 = int(math.floor((x + m) / TILE))
        c1 = int(math.floor((x + p.player_w - m) / TILE))
        r0 = int(math.floor((y + m) / TILE))
        r1 = int(math.floor((y + p.player_h - m) / TILE))
        hz = self.level.hazards
        for c in range(c0, c1 + 1):
            for r in range(r0, r1 + 1):
                if (c, r) in hz:
                    return True
        return False

    def _crumble_step(self):
        """Crumble blocks touched this frame (standing on, against or under them) start breaking."""
        p, lv = self.p, self.level
        x0, y0, x1, y1 = self.x - 1, self.y - 1, self.x + p.player_w + 1, self.y + p.player_h + 1
        for c in range(int(math.floor(x0 / TILE)), int(math.floor(x1 / TILE)) + 1):
            for r in range(int(math.floor(y0 / TILE)), int(math.floor(y1 / TILE)) + 1):
                if (c, r) in lv.crumble and (c, r) not in self._touched:
                    self._touched[(c, r)] = self.clock
        done = [cr for cr, t in self._touched.items()
                if cr not in self._gone and self.clock - t >= lv.crumble_frames]
        if done:
            self._gone = self._gone | frozenset(done)

    def _portal_step(self):
        p = self.p
        cx, cy = self.x + p.player_w / 2, self.y + p.player_h / 2
        reach = TILE * 0.9
        for i, (a, b) in enumerate(self.level.portals):
            for k, (src, dst) in enumerate(((a, b), (b, a))):
                tag = 2 * i + k
                inside = math.hypot(cx - src[0], cy - src[1]) < reach
                if tag == self._portal_lock:
                    if not inside:
                        self._portal_lock = -1
                    continue
                if inside:
                    self.x, self.y = dst[0] - p.player_w / 2, dst[1] - p.player_h / 2
                    self._portal_lock = 2 * i + (1 - k)   # arrived at the other end
                    return

    @staticmethod
    def _path_at(m: dict, t: float) -> tuple[float, float]:
        """Position on a keyframed path (wraps after the last key) at t seconds."""
        keys, period = m["keys"], m["period"]
        tt = t % period if period > 0 else 0.0
        for (t0, x0, y0), (t1, x1, y1) in zip(keys, keys[1:]):
            if tt <= t1:
                f = (tt - t0) / (t1 - t0) if t1 > t0 else 1.0
                return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        return keys[-1][1], keys[-1][2]

    def mover_positions(self, clock: int | None = None) -> list[tuple[float, float, float]]:
        """(cx, cy, r) of every moving hazard at the given frame (default: now), in px."""
        t = (self.clock if clock is None else clock) * self.p.dt
        return [(*self._path_at(m, t), m["r"]) for m in self.level.movers]

    def _platform_boxes(self, clock: int) -> list[tuple[float, float, float, float]]:
        t = clock * self.p.dt
        out = []
        for m in self.level.platforms:
            x, y = self._path_at(m, t)
            out.append((x - m["hw"], y - m["hh"], x + m["hw"], y + m["hh"]))
        return out

    def _platform_step(self):
        """Move the platforms to this frame; carry a player who stood on one, push one they run into."""
        p = self.p
        old, new = self._plat_boxes, self._platform_boxes(self.clock)
        self._plat_boxes = new
        for i, (a, b) in enumerate(zip(old, new)):
            dx, dy = b[0] - a[0], b[1] - a[1]
            if dx == 0 and dy == 0:
                continue
            x0, y0, x1, y1 = self.x, self.y, self.x + p.player_w, self.y + p.player_h
            carried = i == self._on_plat
            hit = x0 < b[2] and x1 > b[0] and y0 < b[3] and y1 > b[1]
            if not (carried or hit):
                continue
            if carried or (hit and y1 - b[1] <= abs(dy) + 2):   # riding, or the platform rose into the feet
                nx, ny = self.x + dx, b[1] - p.player_h if (carried or dy < 0) else self.y + dy
            else:                                                # side push
                nx, ny = (b[2] if dx > 0 else b[0] - p.player_w), self.y
            self._plat_boxes = [bx for k, bx in enumerate(new) if k != i]   # test static solids only
            if not self._solid_overlap(nx, ny):
                self.x, self.y = nx, ny
            elif not self._solid_overlap(self.x, ny):
                self.y = ny
            self._plat_boxes = new

    def _update_on_platform(self):
        p = self.p
        self._on_plat = -1
        if not self.on_ground:
            return
        x0, x1, foot = self.x, self.x + p.player_w, self.y + p.player_h
        for i, (bx0, by0, bx1, _) in enumerate(self._plat_boxes):
            if x0 < bx1 and x1 > bx0 and abs(foot - by0) < 1.5:
                self._on_plat = i
                return

    def _mover_overlap(self, x: float, y: float) -> bool:
        if not self.level.movers:
            return False
        p, m = self.p, 3.0   # same grazing margin as static hazards
        x0, y0, x1, y1 = x + m, y + m, x + p.player_w - m, y + p.player_h - m
        for cx, cy, r in self.mover_positions():
            dx = cx - min(max(cx, x0), x1)
            dy = cy - min(max(cy, y0), y1)
            if dx * dx + dy * dy < r * r:
                return True
        return False

    def _wind_overlap(self, x: float, y: float) -> bool:
        wind = getattr(self.level, "wind", ())
        if not wind:
            return False
        p = self.p
        c0 = int(math.floor(x / TILE))
        c1 = int(math.floor((x + p.player_w - 1e-6) / TILE))
        r0 = int(math.floor(y / TILE))
        r1 = int(math.floor((y + p.player_h - 1e-6) / TILE))
        return any((c, r) in wind for c in range(c0, c1 + 1) for r in range(r0, r1 + 1))

    def _goal_overlap(self, x: float, y: float) -> bool:
        p = self.p
        cx = (x + p.player_w / 2) / TILE
        cy = (y + p.player_h / 2) / TILE
        gc, gr = self.level.goal_cell
        return abs(cx - (gc + 0.5)) < 0.9 and abs(cy - (gr + 0.5)) < 0.9

    # ---- simulation ------------------------------------------------------
    def step(self, action: Action):
        if self.dead or self.won:
            return self.state()
        p = self.p
        dt = p.dt
        self.steps += 1
        self.clock += 1
        if self.level.platforms:
            self._platform_step()

        move = (1 if action.right else 0) - (1 if action.left else 0)
        old_facing = self.facing
        if move != 0:
            self.facing = move
        if self._jump_frames >= 0:
            self._jump_frames += 1

        # timers
        self._coyote = (self._coyote - dt) if not self.on_ground else p.coyote_time
        self._jump_buf -= dt
        jump_pressed = action.jump and self._jump_up >= p.jump_release_frames
        if jump_pressed:
            self._jump_buf = p.jump_buffer

        # ---- horizontal -------------------------------------------------
        accel = p.run_accel if self.on_ground else p.air_accel
        if p.walk_accel is not None and not action.run:
            accel = p.walk_accel
        vmax = p.run_max if action.run else p.walk_max
        if move != 0:
            push = old_facing if (p.ground_turn_lag and self.on_ground and move != old_facing) else move
            if self.vx * move < -1e-6:  # input opposes motion (tolerance: float residue after a 2x turn);
                                        # judged on the input even when the turn-lag frame pushes the old way
                accel *= p.turn_mult
            self.vx += push * accel * dt
            if self.vx > vmax:
                self.vx = vmax
            elif self.vx < -vmax:
                self.vx = -vmax
        elif p.ground_release_stop and self.on_ground:
            self.vx = 0.0
        elif p.air_release_keep and not self.on_ground:
            self.vx = max(-vmax, min(vmax, self.vx))
        else:
            fr = p.ground_friction if self.on_ground else p.air_friction
            if p.release_mult is not None:
                fr = p.release_mult * accel
            if self.vx > 0:
                self.vx = max(0.0, self.vx - fr * dt)
            elif self.vx < 0:
                self.vx = min(0.0, self.vx + fr * dt)

        # ---- gravity / wind / wall slide -------------------------------
        def apply_gravity():
            if self._skip_gravity:
                self._skip_gravity = False
                return
            scale = 1.0
            if p.wall_slide_gravity_scale is not None and self.on_wall != 0 and not self.on_ground:
                scale = p.wall_slide_gravity_scale  # applies rising or falling while touching a wall
            self.vy += p.gravity * scale * dt
            if p.wind_accel > 0.0 and self._wind_overlap(self.x, self.y):
                self.vy = max(self.vy - p.wind_accel * dt, -p.wind_max_up)
            wall_sliding = (self.on_wall != 0 and not self.on_ground
                            and move == self.on_wall and self.vy > 0)
            if wall_sliding and p.wall_slide_gravity_scale is None and self.vy > p.wall_slide_max:
                self.vy = p.wall_slide_max
            if self.vy > p.max_fall:
                self.vy = p.max_fall

        if not p.gravity_after_jump:
            apply_gravity()

        # ---- jumping ----------------------------------------------------
        can_ground_jump = self.on_ground or self._coyote > 0.0
        if self._jump_buf > 0.0:
            if can_ground_jump:
                self.vy = -p.jump_speed
                self._jump_frames = 0
                self._jump_min = p.jump_min_frames
                self._jump_buf = 0.0
                self._coyote = 0.0
                self.on_ground = False
            elif self.on_wall != 0:  # wall jump
                self.vy = -p.wall_jump_y
                self._jump_frames = 0
                self._jump_min = p.wall_jump_min_frames if p.wall_jump_min_frames is not None else p.jump_min_frames
                self.vx = -self.on_wall * p.wall_jump_x
                self.facing = -self.on_wall
                self._jump_buf = 0.0
        # variable jump height: releasing jump while rising cuts upward speed
        if p.jump_min_frames is not None:
            if (not action.jump) and self.vy < 0 and self._jump_frames >= (self._jump_min or 0):
                self.vy = 0.0
                self._jump_frames = -1
        elif (not action.jump) and self._prev_jump and self.vy < 0:
            self.vy *= p.jump_cut
        self._prev_jump = action.jump
        self._jump_up = 0 if action.jump else self._jump_up + 1
        if p.gravity_after_jump:
            apply_gravity()

        # ---- integrate + resolve X -------------------------------------
        grounded = self.on_ground
        self.on_wall = 0
        dx = self.vx * dt
        nx = self.x + dx
        lift = None
        if (self.level.shapes and self._solid_overlap(nx, self.y)
                and not self._solid_overlap(nx, self.y, full_only=True)):
            # only a ramp, half block or cut corner is in the way: slide over it (45 degrees -> shift <= |dx|),
            # upwards for floor shapes, downwards for ceiling shapes, whether running, rising or falling
            reach = int(abs(dx)) + 3
            lift = next((k for k in range(1, reach) if not self._solid_overlap(nx, self.y - k)), None)
            if lift is None and not grounded:
                lift = next((-k for k in range(1, reach) if not self._solid_overlap(nx, self.y + k)), None)
        if lift is not None:
            self.x, self.y = nx, self.y - lift
        elif self._solid_overlap(nx, self.y):
            step_dir = 1 if self.vx > 0 else -1
            guard = 0
            while (not self._solid_overlap(self.x + step_dir, self.y)) and guard < TILE + 2:
                self.x += step_dir
                guard += 1
            self.on_wall = 1 if self.vx > 0 else -1
            self.vx = 0.0
        else:
            self.x = nx
            if self.level.shapes and grounded and self.vy >= 0 and not self._solid_overlap(self.x, self.y + 1):
                # running down a ramp: stay on it instead of launching off each step
                drop = next((k for k in range(2, int(abs(dx)) + 3) if self._solid_overlap(self.x, self.y + k)), None)
                if drop is not None:
                    self.y += drop - 1

        # ---- integrate + resolve Y -------------------------------------
        was_falling = self.vy > 0
        self.on_ground = False
        ny = self.y + self.vy * dt
        if self._solid_overlap(self.x, ny):
            step_dir = 1 if self.vy > 0 else -1
            guard = 0
            while (not self._solid_overlap(self.x, self.y + step_dir)) and guard < TILE + 2:
                self.y += step_dir
                guard += 1
            if was_falling:
                self.on_ground = True
            self.vy = 0.0
        else:
            self.y = ny

        if self.level.platforms:
            self._update_on_platform()
        if self.level.crumble:
            self._crumble_step()
        if self.level.portals:
            self._portal_step()

        # ---- terminal checks -------------------------------------------
        if (self.y > (self.level.rows + 4) * TILE or
                self._hazard_overlap(self.x, self.y) or self._mover_overlap(self.x, self.y)):
            self.dead = True
        elif self._goal_overlap(self.x, self.y):
            self.won = True
        return self.state()

    # ---- observation -----------------------------------------------------
    def state(self) -> dict:
        p = self.p
        gc, gr = self.level.goal_cell
        goal_x = (gc + 0.5) * TILE
        goal_y = (gr + 0.5) * TILE
        cx = self.x + p.player_w / 2
        cy = self.y + p.player_h / 2
        return {
            "x": self.x, "y": self.y, "vx": self.vx, "vy": self.vy,
            "cx": cx, "cy": cy,
            "on_ground": self.on_ground, "on_wall": self.on_wall,
            "facing": self.facing,
            "goal_dx": goal_x - cx, "goal_dy": goal_y - cy,
            "goal_dist": math.hypot(goal_x - cx, goal_y - cy),
            "dead": self.dead, "won": self.won, "steps": self.steps,
        }
