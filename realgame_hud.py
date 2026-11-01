"""Live Pygame Telemetry & RL Monitoring Dashboard for realgame.py (Real Super Meat Boy).

Displays in real-time alongside the running game:
  - Live game level layout (tiles, hazards, goals, spawn).
  - Semi-transparent green Dijkstra Reward Rail navigation path.
  - Real agents moving in the game with their trails and leader highlight.
  - Neural Network Decision Dashboard:
      * Softmax action probabilities P(a|s) with live animated bars.
      * Active action choice highlighted in glowing green.
      * Critic Value function V(s) gauge.
      * Policy entropy / exploration indicator.
      * Wave progress bar, fastest finish record, and death/win tallies.
"""
from __future__ import annotations

import time
import colorsys
from collections import deque

import pygame
import numpy as np

TILE = 20.0
PANEL_W = 390
MAX_MAP_W, MAX_MAP_H = 1200, 800
MIN_WIN_H = 620

BG = (18, 18, 24)
PANEL_BG = (26, 26, 36)
COL_SOLID = (60, 64, 82)
COL_HAZARD = (210, 60, 60)
COL_GOAL = (240, 120, 180)
COL_SPAWN = (90, 200, 120)
COL_TEXT = (220, 220, 235)
COL_DIM = (145, 145, 165)
COL_BEST = (255, 210, 60)
COL_ACCENT = (70, 200, 255)
COL_BAR_BG = (45, 48, 64)

# Reward Rail colors
COL_RAIL_CORE = (45, 235, 95, 160)
COL_RAIL_GLOW = (70, 255, 120, 50)
COL_RAIL_NODE = (50, 255, 110, 220)

ACTION_NAMES = ["NOOP", "LEFT", "RIGHT", "JUMP", "L+JMP", "R+JMP", "R+RUN", "R+R+J", "L+RUN", "L+R+J"]


def agent_color(i: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.65, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


class RealgameHUD:
    def __init__(self, title: str = "Super Meat Boy - RL Monitor"):
        pygame.init()
        self.title = title
        self.font = pygame.font.SysFont("consolas", 14)
        self.small = pygame.font.SysFont("consolas", 12)
        self.big = pygame.font.SysFont("consolas", 18, bold=True)
        self.screen = None
        self.level = None
        self.trails: dict[int, deque] = {}
        self.show_rail = True
        self.show_nn = True
        self.telemetry = {}
        self.active = True

    def set_level(self, level):
        """level: realgame.Level instance."""
        self.level = level
        lw = level.W * TILE
        lh = level.H * TILE
        self.scale = min(MAX_MAP_W / lw, MAX_MAP_H / lh, 2.0)
        self.map_w = int(lw * self.scale)
        self.map_h = int(lh * self.scale)
        self.win_w = self.map_w + PANEL_W
        self.win_h = max(self.map_h, MIN_WIN_H)
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption(f"{self.title} - {level.name}")
        self.bg = self._render_level_bg(level)
        self.trails.clear()

    def _pt(self, x: float, y: float) -> tuple[int, int]:
        """Game world coordinates (+Y up, origin at ox, oy) -> Pygame screen coordinates (+Y down)."""
        lv = self.level
        px = int((x - lv.ox) * self.scale)
        py = int((lv.H * TILE - (y - lv.oy)) * self.scale)
        return px, py

    def _render_level_bg(self, level) -> pygame.Surface:
        surf = pygame.Surface((self.map_w, self.map_h))
        surf.fill(BG)
        ts = TILE * self.scale
        # Render solid blocks
        for r in range(level.H):
            for c in range(level.W):
                if level.solid[r, c]:
                    # row 0 in game is bottom row
                    py = int((level.H - 1 - r) * ts)
                    px = int(c * ts)
                    rect = pygame.Rect(px, py, max(1, int((c + 1) * ts) - px), max(1, int((level.H - r) * ts) - py))
                    surf.fill(COL_SOLID, rect)
        # Render Spawn
        sx, sy = self._pt(*level.spawn)
        pygame.draw.circle(surf, COL_SPAWN, (sx, sy), max(4, int(8 * self.scale)), 2)
        # Render Goals
        for gx, gy in level.goals:
            px, py = self._pt(gx, gy)
            pygame.draw.circle(surf, COL_GOAL, (px, py), max(5, int(10 * self.scale)))
            pygame.draw.circle(surf, (255, 255, 255), (px, py), max(2, int(4 * self.scale)))
        return surf

    def update_telemetry(self, data: dict):
        self.telemetry = data

    def frame(self, states: list, hazards: list, leader: int = 0, wave: int = 0, frame_num: int = 0, best_time: float | None = None):
        """Render one frame. states: list of [x, y, vx, vy, flags, st, status]."""
        if self.level is None or self.screen is None:
            return

        scr = self.screen
        scr.fill(PANEL_BG)
        scr.blit(self.bg, (0, 0))

        # Dynamic lethal hazards from game RAM
        for hx, hy, _a, _b, scale in hazards:
            px, py = self._pt(hx, hy)
            rad = max(2, int(TILE * (abs(scale) or 1.0) * 0.9 * self.scale))
            pygame.draw.circle(scr, COL_HAZARD, (px, py), rad)

        # 1. Semi-transparent Green Dijkstra Reward Rail
        if self.show_rail and hasattr(self.level, "rail_pts") and len(self.level.rail_pts) > 1:
            rail_surf = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
            pts = [self._pt(p[0], p[1]) for p in self.level.rail_pts]
            # Outer glow
            pygame.draw.lines(rail_surf, COL_RAIL_GLOW, False, pts, 8)
            # Core green line
            pygame.draw.lines(rail_surf, COL_RAIL_CORE, False, pts, 3)
            # Waypoint nodes
            for p in pts:
                pygame.draw.circle(rail_surf, COL_RAIL_NODE, p, max(2, int(3 * self.scale)))
            # Princess goal circle
            if pts:
                pygame.draw.circle(rail_surf, (255, 105, 180, 210), pts[-1], max(6, int(8 * self.scale)), 2)
            scr.blit(rail_surf, (0, 0))

        # 2. Agent trails and bodies
        for i, s in enumerate(states):
            x, y, _, _, _, _, st = s
            px, py = self._pt(x, y)
            if i not in self.trails:
                self.trails[i] = deque(maxlen=60)
            if st == 0:  # alive
                self.trails[i].append((px, py))
            if len(self.trails[i]) > 1:
                c = agent_color(i)
                pygame.draw.lines(scr, (c[0] // 2, c[1] // 2, c[2] // 2), False, list(self.trails[i]), 1)

            # Draw agent
            pw = max(4, int(14 * self.scale))
            ph = max(4, int(14 * self.scale))
            rect = pygame.Rect(px - pw // 2, py - ph // 2, pw, ph)
            if st == 1:  # dead
                pygame.draw.line(scr, (220, 60, 60), (px - 3, py - 3), (px + 3, py + 3), 2)
                pygame.draw.line(scr, (220, 60, 60), (px - 3, py + 3), (px + 3, py - 3), 2)
            elif st == 2:  # won
                pygame.draw.circle(scr, (80, 255, 120), (px, py), 6, 2)
            else:
                col = agent_color(i)
                scr.fill(col, rect)
                if i == leader:
                    pygame.draw.rect(scr, (255, 255, 255), rect.inflate(4, 4), 2)

        # ----------------- SIDE PANEL: RL TELEMETRY -----------------
        px0 = self.map_w + 14
        y = 12
        scr.blit(self.big.render(f"Super Meat Boy [Real]", True, COL_TEXT), (px0, y))
        y += 24
        scr.blit(self.font.render(f"Level: {self.level.name}", True, COL_ACCENT), (px0, y))
        y += 20
        scr.blit(self.small.render(f"Size: {self.level.W}x{self.level.H}  Lethal: {len(hazards)}", True, COL_DIM), (px0, y))
        y += 22

        # Reward Rail progress bar
        ratio = self.telemetry.get("rail_ratio", 0.0)
        bar_w = PANEL_W - 28
        bar_h = 16
        pygame.draw.rect(scr, COL_BAR_BG, (px0, y, bar_w, bar_h), border_radius=4)
        fill_w = int(bar_w * max(0.0, min(1.0, ratio)))
        if fill_w > 0:
            pygame.draw.rect(scr, (45, 225, 105), (px0, y, fill_w, bar_h), border_radius=4)
        pct_text = f"Reward Rail: {ratio * 100:.1f}%"
        scr.blit(self.small.render(pct_text, True, (240, 240, 240)), (px0 + 6, y + 1))
        y += 24

        # Wave & Timer status
        live_cnt = sum(1 for s in states if s[6] == 0)
        dead_cnt = sum(1 for s in states if s[6] == 1)
        won_cnt = sum(1 for s in states if s[6] == 2)
        scr.blit(self.font.render(f"Wave {wave} | Frame {frame_num}", True, COL_TEXT), (px0, y))
        y += 18
        scr.blit(self.small.render(f"Agents: {live_cnt} live, {dead_cnt} dead, {won_cnt} won", True, (200, 230, 200)), (px0, y))
        y += 18
        b_str = f"{best_time:.2f}s" if best_time is not None else "--.--"
        scr.blit(self.font.render(f"Best Finish Time: {b_str}", True, COL_BEST), (px0, y))
        y += 24

        # Neural Network Dashboard
        if self.show_nn:
            pygame.draw.line(scr, (50, 54, 70), (px0, y), (px0 + PANEL_W - 28, y), 1)
            y += 8
            scr.blit(self.big.render("Neural Network (PPO)", True, COL_ACCENT), (px0, y))
            y += 24

            v = self.telemetry.get("value", 0.0)
            ent = self.telemetry.get("entropy", 0.0)
            scr.blit(self.small.render(f"Value V(s): {v:+.3f}   Entropy: {ent:.2f}", True, COL_TEXT), (px0, y))
            y += 18

            # Action Distribution
            probs = self.telemetry.get("action_probs")
            chosen = self.telemetry.get("chosen_action", 0)
            if probs is not None and len(probs) > 0:
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

            # Stagnation & Fitness
            stag = self.telemetry.get("stagnation", 0)
            fit = self.telemetry.get("fitness", 0.0)
            y += 4
            scr.blit(self.small.render(f"Fitness: {fit:.1f}   Stagnation: {stag}/120", True, (240, 160, 60) if stag > 40 else COL_DIM), (px0, y))
            y += 18

        # Hotkeys footer
        y = max(y + 10, self.win_h - 45)
        scr.blit(self.small.render("[TAB] NN View | [D] Reward Rail | [Esc] Stop", True, (130, 130, 150)), (px0, y))

        pygame.display.flip()

    def poll(self) -> list[str]:
        out = []
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                out.append("quit")
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    out.append("quit")
                elif e.key == pygame.K_TAB:
                    self.show_nn = not self.show_nn
                elif e.key == pygame.K_d:
                    self.show_rail = not self.show_rail
        return out

    def close(self):
        pygame.quit()

