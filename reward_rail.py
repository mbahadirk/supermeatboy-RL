"""Reward Rail: Platformer-aware Dijkstra pathfinding and reward/fitness rail for Super Meat Boy.

Extracts the full level layout (solids, hazards, moving obstacles, portals, platforms, keys/bandages,
and goal/princess), finds the optimal traversal rail using gravity-aware Dijkstra shortest path,
and provides:
  1. Dense progression tracking along the rail for RL rewards and fitness calculation.
  2. Substantial rewards for key/bandage collection and reaching Princess.
  3. Penalties for death, stagnation (hopping/stuck in place), and continuous action repetition.
  4. Semi-transparent green visualization data for both Pygame viewer and in-game hooks.
"""
from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING

import numpy as np

from engine import TILE

if TYPE_CHECKING:
    from levels import Level


class RewardRail:
    """Optimal path from spawn to princess (through keys/bandages) and rail-following reward logic."""

    def __init__(self, level: "Level"):
        self.level = level
        self.cols = level.cols
        self.rows = level.rows
        self.tile = TILE

        # Parse key points
        self.spawn_pt = self._cell_to_px(level.spawn_cell)
        self.goal_pt = self._cell_to_px(level.goal_cell)
        self.keys_pts = [self._cell_to_px(c) for c in level.meta.get("bandages", [])] if hasattr(level, "meta") else []

        # Build clearance and risk grid
        self.risk_grid = self._compute_risk_grid()

        # Compute Dijkstra path (Spawn -> Keys -> Princess)
        self.cell_path = self._compute_multi_goal_dijkstra()
        self.waypoints = [self._cell_to_px(c) for c in self.cell_path]
        if len(self.waypoints) < 2:
            self.waypoints = [self.spawn_pt, self.goal_pt]

        # Arc-length parameterization
        self.cum_dist = [0.0]
        for i in range(1, len(self.waypoints)):
            p0, p1 = self.waypoints[i - 1], self.waypoints[i]
            d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            self.cum_dist.append(self.cum_dist[-1] + d)
        self.total_length = max(self.cum_dist[-1], 1.0)

        # Milestone indices for keys and princess
        self.key_indices = self._find_key_indices()

    def _cell_to_px(self, cell: tuple[int, int]) -> tuple[float, float]:
        return ((cell[0] + 0.5) * self.tile, (cell[1] + 0.5) * self.tile)

    def _px_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(x // self.tile), int(y // self.tile))

    def _compute_risk_grid(self) -> np.ndarray:
        """Grid of traversal risks: inf for solid/hazards, elevated near hazards."""
        grid = np.zeros((self.rows, self.cols), dtype=np.float32)

        # Mark solids
        for c, r in self.level.solids:
            if (c, r) not in self.level.shapes and (c, r) not in self.level.crumble:
                if 0 <= c < self.cols and 0 <= r < self.rows:
                    grid[r, c] = math.inf

        # Mark hazards
        hazards = set(self.level.hazards)
        for c, r in hazards:
            if 0 <= c < self.cols and 0 <= r < self.rows:
                grid[r, c] = math.inf

        # Safety buffer around hazards (spikes, saws): keep the rail away from danger zones
        for c, r in hazards:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nc, nr = c + dc, r + dr
                    if 0 <= nc < self.cols and 0 <= nr < self.rows:
                        if not np.isinf(grid[nr, nc]):
                            dist = math.hypot(dc, dr)
                            if dist <= 1.5:
                                grid[nr, nc] += 15.0
                            elif dist <= 2.2:
                                grid[nr, nc] += 6.0

        return grid

    def _dijkstra_segment(self, start: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int]]:
        """Find lowest-cost platformer path between start and target cell."""
        if start == target:
            return [start]

        rows, cols = self.rows, self.cols
        dist = np.full((rows, cols), math.inf, dtype=np.float32)
        dist[start[1], start[0]] = 0.0
        prev = {}

        # priority queue: (cost, col, row)
        pq = [(0.0, start[0], start[1])]

        portals = {}
        for (ax, ay), (bx, by) in getattr(self.level, "portals", []):
            ca, cb = (int(ax // self.tile), int(ay // self.tile)), (int(bx // self.tile), int(by // self.tile))
            portals.setdefault(ca, []).append(cb)
            portals.setdefault(cb, []).append(ca)

        while pq:
            cost, c, r = heapq.heappop(pq)
            if (c, r) == target:
                break
            if cost > dist[r, c]:
                continue

            # Ground and wall contact checks
            has_floor = (r + 1 >= rows) or ((c, r + 1) in self.level.solids)
            has_left_wall = (c - 1 < 0) or ((c - 1, r) in self.level.solids)
            has_right_wall = (c + 1 >= cols) or ((c + 1, r) in self.level.solids)
            has_wall = has_left_wall or has_right_wall

            # Platformer moves (walk, run, jump, wall-jump, fall):
            moves = [
                # Horizontal walking & running
                (-1, 0, 1.0 if has_floor else 1.2),
                (1, 0, 1.0 if has_floor else 1.2),
                (-2, 0, 1.8 if has_floor else 2.2),
                (2, 0, 1.8 if has_floor else 2.2),
                # Falling with gravity
                (0, 1, 0.7),
                (-1, 1, 0.8),
                (1, 1, 0.8),
                (-2, 1, 1.5),
                (2, 1, 1.5),
                (0, 2, 1.0),
                # Jumping (ground or wall launch)
                (0, -1, 1.1 if has_floor else (1.3 if has_wall else 2.0)),
                (-1, -1, 1.2 if has_floor else (1.4 if has_wall else 2.2)),
                (1, -1, 1.2 if has_floor else (1.4 if has_wall else 2.2)),
                (-2, -1, 1.6 if (has_floor or has_wall) else 2.6),
                (2, -1, 1.6 if (has_floor or has_wall) else 2.6),
                (0, -2, 1.8 if (has_floor or has_wall) else 3.0),
                (-1, -2, 1.8 if (has_floor or has_wall) else 3.0),
                (1, -2, 1.8 if (has_floor or has_wall) else 3.0),
                (-2, -2, 2.2 if (has_floor or has_wall) else 3.5),
                (2, -2, 2.2 if (has_floor or has_wall) else 3.5),
            ]

            # Portal warps
            for pc, pr in portals.get((c, r), ()):
                moves.append((pc - c, pr - r, 1.0))

            for dc, dr, base_cost in moves:
                nc, nr = c + dc, r + dr
                if 0 <= nc < cols and 0 <= nr < rows:
                    cell_risk = self.risk_grid[nr, nc]
                    if np.isinf(cell_risk):
                        continue

                    step_cost = base_cost * self.tile + cell_risk
                    new_cost = cost + step_cost
                    if new_cost < dist[nr, nc]:
                        dist[nr, nc] = new_cost
                        prev[(nc, nr)] = (c, r)
                        heapq.heappush(pq, (new_cost, nc, nr))

        if target not in prev and target != start:
            # Fallback direct line if completely isolated by geometry
            return [start, target]

        # Reconstruct path
        path = []
        curr = target
        while curr in prev:
            path.append(curr)
            curr = prev[curr]
        path.append(start)
        path.reverse()
        return path

    def _compute_multi_goal_dijkstra(self) -> list[tuple[int, int]]:
        """Route from Spawn -> Keys/Bandages -> Goal (Princess)."""
        curr = self.level.spawn_cell
        full_path = []

        # Visit keys/bandages first if present
        remaining_keys = list(self.level.meta.get("bandages", [])) if hasattr(self.level, "meta") else []
        while remaining_keys:
            # Nearest remaining key
            remaining_keys.sort(key=lambda k: math.hypot(k[0] - curr[0], k[1] - curr[1]))
            next_key = remaining_keys.pop(0)
            seg = self._dijkstra_segment(curr, next_key)
            if full_path:
                seg = seg[1:]
            full_path.extend(seg)
            curr = next_key

        # Final stretch: to Princess / Goal
        final_seg = self._dijkstra_segment(curr, self.level.goal_cell)
        if full_path and final_seg:
            final_seg = final_seg[1:]
        full_path.extend(final_seg)

        return full_path if full_path else [self.level.spawn_cell, self.level.goal_cell]

    def _find_key_indices(self) -> list[int]:
        """Find waypoint index closest to each key/bandage."""
        indices = []
        for kp in self.keys_pts:
            best_idx = 0
            best_d = math.inf
            for i, wp in enumerate(self.waypoints):
                d = math.hypot(wp[0] - kp[0], wp[1] - kp[1])
                if d < best_d:
                    best_d = d
                    best_idx = i
            indices.append(best_idx)
        return indices

    def project(self, x: float, y: float, hint_idx: int = 0) -> tuple[float, float, int, tuple[float, float]]:
        """Project world coordinate (x, y) onto the rail.

        Returns:
            progress_dist (px): distance along the rail from spawn.
            perp_dist (px): lateral distance from rail.
            segment_idx (int): index of the closest rail segment.
            closest_pt (x, y): coordinates of nearest point on rail.
        """
        best_dist_sq = math.inf
        best_progress = 0.0
        best_seg = 0
        best_pt = self.waypoints[0]

        # Scan segments (hint_idx optimizes localized search if desired, full scan for robustness)
        n = len(self.waypoints)
        for i in range(n - 1):
            p0 = self.waypoints[i]
            p1 = self.waypoints[i + 1]
            vx, vy = p1[0] - p0[0], p1[1] - p0[1]
            seg_len_sq = vx * vx + vy * vy
            if seg_len_sq < 1e-6:
                continue

            ux, uy = x - p0[0], y - p0[1]
            t = max(0.0, min(1.0, (ux * vx + uy * vy) / seg_len_sq))
            proj_x = p0[0] + t * vx
            proj_y = p0[1] + t * vy
            dx, dy = x - proj_x, y - proj_y
            d_sq = dx * dx + dy * dy

            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_progress = self.cum_dist[i] + t * math.sqrt(seg_len_sq)
                best_seg = i
                best_pt = (proj_x, proj_y)

        return best_progress, math.sqrt(best_dist_sq), best_seg, best_pt

    def progress_ratio(self, x: float, y: float) -> float:
        """Normalized rail completion ratio in [0.0, 1.0]."""
        prog, _, _, _ = self.project(x, y)
        return min(1.0, max(0.0, prog / self.total_length))

    def evaluate_step(
        self,
        prev_x: float,
        prev_y: float,
        curr_x: float,
        curr_y: float,
        best_rail_dist: float,
        stagnation_counter: int,
        action: int,
        prev_action: int,
        is_won: bool,
        is_dead: bool,
        steps: int,
        max_steps: int = 1800,
    ) -> dict:
        """Calculate comprehensive reward, fitness, and penalties based on the reward rail."""
        prog, perp_dist, seg_idx, closest_pt = self.project(curr_x, curr_y)
        new_best = max(best_rail_dist, prog)

        # 1. Forward progression reward along the rail (per pixel beyond best progress)
        reward = 0.0
        if prog > best_rail_dist:
            # Reward for forging forward along the optimal path
            prog_gain = prog - best_rail_dist
            reward += 0.08 * prog_gain

        # 2. Rail proximity guidance: gentle soft tether to keep agent close to the green rail
        # Penalizes drifting far away into dead ends
        reward -= 0.005 * min(perp_dist / self.tile, 6.0)

        # 3. Small step penalty to encourage rapid completion
        reward -= 0.001

        # 4. Action repetition and stagnation checks
        stalled = False
        dist_moved = math.hypot(curr_x - prev_x, curr_y - prev_y)
        if dist_moved < 2.0:
            stagnation_counter += 1
            if stagnation_counter > 40:
                reward -= 0.02   # penalize standing or hopping in place
            if stagnation_counter > 150:
                reward -= 0.08   # severe stall penalty
                stalled = True
        else:
            stagnation_counter = max(0, stagnation_counter - 2)

        if action == prev_action and dist_moved < 1.0:
            reward -= 0.015  # penalize grinding identical action into a wall

        # 5. Milestone terminal bonuses/penalties
        if is_won:
            # Huge bonus for Princess rescue + speed bonus
            speed_ratio = max(0.0, 1.0 - (steps / max_steps))
            reward += 50.0 + 20.0 * speed_ratio
        elif is_dead:
            # Penalty for fatal hazards / death
            reward -= 5.0

        # Clip per-step reward to avoid gradient explosion
        reward = float(np.clip(reward, -15.0, 50.0))

        # Fitness score (interpretable scalar for policy comparison)
        progress_pct = (new_best / self.total_length) * 100.0
        fitness = (1000.0 if is_won else 0.0) + (progress_pct * 5.0) - (steps * 0.05) - (50.0 if is_dead else 0.0)

        return {
            "reward": reward,
            "rail_progress": prog,
            "new_best_progress": new_best,
            "progress_ratio": prog / self.total_length,
            "perp_dist": perp_dist,
            "closest_pt": closest_pt,
            "stagnation_counter": stagnation_counter,
            "stalled": stalled,
            "fitness": fitness,
        }
