"""Whole-map pygame view shared by `train.py --watch` and `watch.py`.

Displays the level scaled to fit the window with:
  - Dijkstra Reward Rail: semi-transparent green navigation line from spawn to princess.
  - Active agents with custom colors and trails.
  - Real-time RL & Neural Network Telemetry:
      * Action probabilities distribution (policy head pi(a|s)).
      * Value function estimation (critic head V(s)).
      * Rail progress completion gauge (0 - 100%).
      * Leading agent fitness score and stagnation monitor.
  - Interactive hotkeys:
      [TAB] toggle NN telemetry, [D] toggle Reward Rail, [S] real-time speed, [V] drawing on/off.
"""
from __future__ import annotations

import time
import colorsys
from collections import deque

import pygame
import numpy as np

from engine import TILE
from reward_rail import RewardRail

MAX_MAP_W, MAX_MAP_H = 1280, 820
PANEL_W = 390
MIN_WIN_H = 620
TRAIL_LEN = 120
MARK_SECONDS = 1.5

BG = (18, 18, 24)
PANEL_BG = (26, 26, 36)
PANEL_CARD = (34, 34, 48)
COL_SOLID = (60, 64, 82)
COL_HAZARD = (210, 60, 60)
COL_WIND = (40, 70, 110)
COL_GOAL = (240, 120, 180)
COL_SPAWN = (90, 200, 120)
COL_TEXT = (220, 220, 235)
COL_DIM = (145, 145, 165)
COL_BEST = (255, 210, 60)
COL_CRUMBLE = (120, 96, 70)
COL_PLATFORM = (150, 150, 175)
COL_PORTAL = (150, 90, 230)

# Reward Rail & UI accent colors
COL_RAIL_CORE = (45, 235, 95, 150)
COL_RAIL_GLOW = (70, 255, 120, 45)
COL_RAIL_NODE = (50, 255, 110, 210)
COL_KEY = (255, 215, 0)
COL_ACCENT = (70, 200, 255)
COL_BAR_BG = (45, 48, 64)
COL_BAR_FILL = (45, 225, 105)

KEYS = {
    pygame.K_ESCAPE: "quit", pygame.K_SPACE: "pause", pygame.K_s: "slow", pygame.K_v: "draw",
    pygame.K_r: "restart", pygame.K_n: "next", pygame.K_p: "prev",
    pygame.K_TAB: "telemetry", pygame.K_d: "rail",
    pygame.K_PLUS: "faster", pygame.K_EQUALS: "faster", pygame.K_KP_PLUS: "faster",
    pygame.K_MINUS: "slower", pygame.K_KP_MINUS: "slower",
}

ACTION_NAMES = ["NOOP", "LEFT", "RIGHT", "JUMP", "L+JMP", "R+JMP", "R+RUN", "R+R+J", "L+RUN", "L+R+J"]


def agent_color(i: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.65, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


class MapView:
    def __init__(self, title: str):
        pygame.init()
        self.title = title
        self.font = pygame.font.SysFont("consolas", 14)
        self.small = pygame.font.SysFont("consolas", 12)
        self.big = pygame.font.SysFont("consolas", 18, bold=True)
        self.screen = None
        self.trails: dict[int, deque] = {}
        self.marks: deque = deque(maxlen=400)
        self.best_path: list[tuple[float, float]] = []

        self.show_rail = True
        self.show_nn = True
        self.reward_rail: RewardRail | None = None
        self.telemetry: dict | None = None

    # ---- level ------------------------------------------------------------
    def set_level(self, level, name: str):
        self.level, self.name = level, name
        lw, lh = level.cols * TILE, level.rows * TILE
        self.scale = min(MAX_MAP_W / lw, MAX_MAP_H / lh, 2.0)
        self.map_w, self.map_h = int(lw * self.scale), int(lh * self.scale)
        self.win_w = self.map_w + PANEL_W
        self.win_h = max(self.map_h, MIN_WIN_H)
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption(f"{self.title} - {name}")
        self.bg = self._render_level(level)
        self.trails.clear()
        self.marks.clear()
        self.best_path = []
        try:
            self.reward_rail = RewardRail(level)
        except Exception:
            self.reward_rail = None

    def _cell_rect(self, c: int, r: int) -> pygame.Rect:
        ts = TILE * self.scale
        x0, y0 = int(c * ts), int(r * ts)
        return pygame.Rect(x0, y0, max(1, int((c + 1) * ts) - x0), max(1, int((r + 1) * ts) - y0))

    def _render_level(self, level) -> pygame.Surface:
        surf = pygame.Surface((self.map_w, self.map_h))
        surf.fill(BG)
        for c, r in level.wind:
            surf.fill(COL_WIND, self._cell_rect(c, r))
        for c, r in level.solids:
            surf.fill(COL_CRUMBLE if (c, r) in level.crumble else COL_SOLID, self._cell_rect(c, r))
        for a, b in level.portals:
            for x, y in (a, b):
                pygame.draw.circle(surf, COL_PORTAL, self._pt(x, y), max(3, int(TILE * self.scale * 0.6)), 2)
        for c, r in level.hazards:
            rect = self._cell_rect(c, r)
            pygame.draw.circle(surf, COL_HAZARD, rect.center, max(1, rect.w // 2))
        surf.fill(COL_GOAL, self._cell_rect(*level.goal_cell))
        pygame.draw.rect(surf, COL_SPAWN, self._cell_rect(*level.spawn_cell), 2)
        return surf

    def _pt(self, x: float, y: float) -> tuple[int, int]:
        return int(x * self.scale), int(y * self.scale)

    # ---- agent bookkeeping -------------------------------------------------
    def track(self, i: int, cx: float, cy: float):
        """Append an agent's centre (world px) to its trail."""
        t = self.trails.get(i)
        if t is None:
            t = self.trails[i] = deque(maxlen=TRAIL_LEN)
        t.append((cx, cy))

    def end_episode(self, i: int, cx: float, cy: float, outcome: str):
        """outcome: 'won' | 'dead' | 'timeout'. Leaves a fading mark and clears the agent's trail."""
        self.marks.append((cx, cy, outcome, time.time()))
        self.trails.pop(i, None)

    def set_telemetry(self, data: dict):
        """Update live RL / Neural Network telemetry for the leading agent."""
        self.telemetry = data

    # ---- frame --------------------------------------------------------------
    def frame(self, agents, lines, highlight: int | None = None, banner: str | None = None, world=None):
        """agents: iterable of (i, x, y, w, h) in world px (top-left AABB).
        lines: panel text; an item may be a str or a (str, colour) pair.
        world: an engine World whose moving parts (saws, platforms, broken crumble blocks) are drawn."""
        scr = self.screen
        scr.fill(PANEL_BG)
        scr.blit(self.bg, (0, 0))

        # Dynamic world elements
        if world is not None:
            for c, r in world._gone:
                scr.fill(BG, self._cell_rect(c, r))
            for x0, y0, x1, y1 in world._plat_boxes:
                p0, p1 = self._pt(x0, y0), self._pt(x1, y1)
                scr.fill(COL_PLATFORM, pygame.Rect(p0[0], p0[1], max(2, p1[0] - p0[0]), max(2, p1[1] - p0[1])))
            for x, y, r in world.mover_positions():
                pygame.draw.circle(scr, COL_HAZARD, self._pt(x, y), max(2, int(r * self.scale)))

        # 1. Semi-transparent Green Dijkstra Reward Rail
        if self.show_rail and self.reward_rail and len(self.reward_rail.waypoints) > 1:
            rail_surf = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
            pts = [self._pt(*p) for p in self.reward_rail.waypoints]
            # Outer soft glow
            pygame.draw.lines(rail_surf, COL_RAIL_GLOW, False, pts, 7)
            # Core rail line
            pygame.draw.lines(rail_surf, COL_RAIL_CORE, False, pts, 3)
            # Waypoint nodes
            for p in pts:
                pygame.draw.circle(rail_surf, COL_RAIL_NODE, p, max(2, int(3 * self.scale)))
            # Key / Bandage milestone nodes
            for kp in self.reward_rail.keys_pts:
                p = self._pt(*kp)
                pygame.draw.circle(rail_surf, (255, 215, 0, 220), p, max(4, int(5 * self.scale)))
                pygame.draw.circle(rail_surf, (255, 255, 255, 240), p, max(2, int(2 * self.scale)))
            # Princess / Goal highlight ring
            if self.reward_rail.waypoints:
                gp = self._pt(*self.reward_rail.waypoints[-1])
                pygame.draw.circle(rail_surf, (255, 105, 180, 200), gp, max(6, int(8 * self.scale)), 2)
            scr.blit(rail_surf, (0, 0))

        # Best path so far (gold line)
        if len(self.best_path) > 1:
            pygame.draw.lines(scr, COL_BEST, False, [self._pt(*p) for p in self.best_path], 2)

        # Agent trails
        for i, t in self.trails.items():
            if len(t) > 1:
                c = agent_color(i)
                pygame.draw.lines(scr, (c[0] // 2, c[1] // 2, c[2] // 2), False, [self._pt(*p) for p in t], 1)

        # Fading outcome marks (deaths / goals)
        now = time.time()
        while self.marks and now - self.marks[0][3] > MARK_SECONDS:
            self.marks.popleft()
        for cx, cy, outcome, t0 in self.marks:
            x, y = self._pt(cx, cy)
            fade = 1.0 - (now - t0) / MARK_SECONDS
            if outcome == "dead":
                col = (int(255 * fade), int(60 * fade), int(60 * fade))
                pygame.draw.line(scr, col, (x - 4, y - 4), (x + 4, y + 4), 2)
                pygame.draw.line(scr, col, (x - 4, y + 4), (x + 4, y - 4), 2)
            elif outcome == "won":
                pygame.draw.circle(scr, (int(80 * fade), int(255 * fade), int(120 * fade)), (x, y), 7, 2)
            else:
                pygame.draw.circle(scr, (int(150 * fade),) * 3, (x, y), 3)

        # Agents
        for i, x, y, w, h in agents:
            px, py = self._pt(x, y)
            rect = pygame.Rect(px, py, max(3, int(w * self.scale)), max(3, int(h * self.scale)))
            scr.fill(agent_color(i), rect)
            if i == highlight:
                pygame.draw.rect(scr, (255, 255, 255), rect.inflate(4, 4), 2)

        # ----------------- SIDE PANEL: RL & TELEMETRY -----------------
        px0 = self.map_w + 14
        y = 12
        scr.blit(self.big.render(self.name, True, COL_TEXT), (px0, y))
        y += 28

        # Reward Rail progress bar
        if self.reward_rail:
            ratio = self.telemetry.get("rail_ratio", 0.0) if self.telemetry else 0.0
            fitness = self.telemetry.get("fitness", 0.0) if self.telemetry else 0.0

            # Progress bar background
            bar_w = PANEL_W - 28
            bar_h = 16
            pygame.draw.rect(scr, COL_BAR_BG, (px0, y, bar_w, bar_h), border_radius=4)
            fill_w = int(bar_w * max(0.0, min(1.0, ratio)))
            if fill_w > 0:
                pygame.draw.rect(scr, COL_BAR_FILL, (px0, y, fill_w, bar_h), border_radius=4)
            pct_text = f"Rail: {ratio * 100:.1f}% | Fit: {fitness:.1f}"
            scr.blit(self.small.render(pct_text, True, (240, 240, 240)), (px0 + 6, y + 1))
            y += 24

        # Regular training summary lines
        for item in lines:
            text, col = (item, COL_TEXT) if isinstance(item, str) else item
            if text:
                scr.blit(self.font.render(text, True, col), (px0, y))
            y += 19

        # Neural Network Telemetry Dashboard
        if self.show_nn and self.telemetry:
            y += 10
            pygame.draw.line(scr, (50, 54, 70), (px0, y), (px0 + PANEL_W - 28, y), 1)
            y += 8
            scr.blit(self.big.render("Neural Network (PPO)", True, COL_ACCENT), (px0, y))
            y += 24

            # Value & Entropy
            v = self.telemetry.get("value", 0.0)
            ent = self.telemetry.get("entropy", 0.0)
            scr.blit(self.small.render(f"Value V(s): {v:+.3f}   Entropy: {ent:.2f}", True, COL_TEXT), (px0, y))
            y += 18

            # Action Probabilities Distribution
            probs = self.telemetry.get("action_probs")
            if probs is not None and len(probs) > 0:
                chosen = int(np.argmax(probs))
                bar_max_w = PANEL_W - 120
                for a_idx, p in enumerate(probs):
                    if a_idx >= len(ACTION_NAMES):
                        break
                    aname = ACTION_NAMES[a_idx]
                    is_top = (a_idx == chosen)

                    name_col = (50, 240, 140) if is_top else COL_DIM
                    scr.blit(self.small.render(f"{aname:7s}", True, name_col), (px0, y))

                    bx = px0 + 65
                    pygame.draw.rect(scr, COL_BAR_BG, (bx, y + 2, bar_max_w, 10), border_radius=2)
                    pw = int(bar_max_w * float(p))
                    if pw > 0:
                        bcol = (50, 230, 110) if is_top else (100, 120, 160)
                        pygame.draw.rect(scr, bcol, (bx, y + 2, pw, 10), border_radius=2)

                    pct_str = f"{p * 100:4.1f}%"
                    scr.blit(self.small.render(pct_str, True, (220, 220, 220) if is_top else COL_DIM), (bx + bar_max_w + 6, y))
                    y += 15

            # Stagnation & Keys
            stagnation = self.telemetry.get("stagnation", 0)
            keys = self.telemetry.get("keys_collected", 0)
            y += 4
            scr.blit(self.small.render(f"Stagnation: {stagnation}/150   Keys: {keys}", True, (240, 160, 60) if stagnation > 40 else COL_DIM), (px0, y))
            y += 18

        # Hotkeys help footer
        y = max(y + 10, self.win_h - 60)
        scr.blit(self.small.render("[TAB] NN View | [D] Reward Rail | [S] Realtime", True, (130, 130, 150)), (px0, y))
        scr.blit(self.small.render("[V] Drawing   | [Esc] Stop & Save", True, (130, 130, 150)), (px0, y + 16))

        if banner:
            img = self.big.render(banner, True, COL_BEST)
            scr.blit(img, ((self.map_w - img.get_width()) // 2, 8))
        pygame.display.flip()

    def poll(self) -> list[str]:
        """Window events as command names (see KEYS); closing the window is 'quit'."""
        out = []
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                out.append("quit")
            elif e.type == pygame.KEYDOWN and e.key in KEYS:
                cmd = KEYS[e.key]
                if cmd == "telemetry":
                    self.show_nn = not self.show_nn
                elif cmd == "rail":
                    self.show_rail = not self.show_rail
                out.append(cmd)
        return out

    def screenshot(self, path: str):
        pygame.image.save(self.screen, path)

    def close(self):
        pygame.quit()
