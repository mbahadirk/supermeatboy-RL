"""Live training monitor for realgame.py (pygame window next to the game).

Left: the level as the trainer sees it - tiles, lethal pieces, the reward paths (one colour each), every
agent coloured by the path it follows (x = dead, ring = finished, white box = leader).
Right: the wave and phase, the leader's policy output (action probabilities, value), learning curves per
wave (finishes, path progress, fitness, fastest time) and per PPO update (policy loss, value loss, entropy,
KL, clip fraction, explained variance, gradient norm), and the policy's first-layer weights as a heat map
(rows = hidden units, columns = inputs; red positive, blue negative) with the size of every layer.

Keys (monitor window focused): P paths on/off in the game, W weights on/off, Esc closes the monitor only.
"""
from __future__ import annotations

import os
import math
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_WINDOW_NO_ACTIVATION_WHEN_SHOWN", "1")   # do not take focus from the game
import pygame
import numpy as np

from realpath import PATH_RGB

TILE = 20.0
WIN_W, WIN_H = 1500, 880
MAP_W = 860
PANEL_X = MAP_W + 16
PANEL_W = WIN_W - PANEL_X - 12

BG = (16, 17, 23)
PANEL = (24, 25, 34)
SOLID = (64, 68, 88)
HAZ = (215, 60, 60)
GOAL = (245, 120, 185)
SPAWN = (90, 210, 120)
TEXT = (222, 222, 236)
DIM = (140, 142, 162)
GRID = (44, 46, 60)
GOLD = (255, 210, 70)
ACTION_NAMES = ["idle", "left", "right", "jump", "L+jump", "R+jump", "R+run", "R+run+J", "L+run", "L+run+J"]


class Monitor:
    def __init__(self, trainer):
        self.t = trainer
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Super Meat Boy RL - training monitor")
        self.font = pygame.font.SysFont("consolas", 14)
        self.small = pygame.font.SysFont("consolas", 12)
        self.big = pygame.font.SysFont("consolas", 18, bold=True)
        self.level_id = None
        self.bg = None
        self.show_weights = True
        self.wimg = None
        self.wver = -1
        self.closed = False

    # ---- main loop (main thread) ------------------------------------------------------------------
    def tick(self):
        if self.closed:
            return
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                self.close()
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_w:
                self.show_weights = not self.show_weights
            if e.type == pygame.KEYDOWN and e.key == pygame.K_p:
                self.t.toggle_game_paths()
        try:
            self.draw()
        except Exception as ex:          # a monitor problem must never stop training
            self.screen.fill(BG)
            self.text(f"monitor error: {ex}", 20, 20, HAZ)
        pygame.display.flip()

    def close(self):
        if not self.closed:
            self.closed = True
            pygame.quit()

    # ---- helpers --------------------------------------------------------------------------------------
    def text(self, s, x, y, col=TEXT, font=None):
        self.screen.blit((font or self.font).render(s, True, col), (x, y))

    def _level_bg(self, lv):
        sc = min((MAP_W - 20) / (lv.W * TILE), (WIN_H - 20) / (lv.H * TILE))
        self.sc = sc
        self.ox0 = 10 + (MAP_W - 20 - lv.W * TILE * sc) / 2
        self.oy0 = 10 + (WIN_H - 20 - lv.H * TILE * sc) / 2
        surf = pygame.Surface((MAP_W, WIN_H))
        surf.fill(BG)
        ts = TILE * sc
        for r in range(lv.H):
            for c in range(lv.W):
                if lv.solid[r, c]:
                    x0, y0 = self.ox0 + c * ts, self.oy0 + (lv.H - 1 - r) * ts
                    surf.fill(SOLID, pygame.Rect(int(x0), int(y0), max(1, int(x0 + ts) - int(x0)), max(1, int(y0 + ts) - int(y0))))
        pygame.draw.circle(surf, SPAWN, self.pt(*lv.spawn), max(4, int(8 * sc)), 2)
        for g in lv.goals:
            pygame.draw.circle(surf, GOAL, self.pt(*g), max(5, int(10 * sc)))
        return surf

    def pt(self, x, y):
        lv = self.t.level
        return (int(self.ox0 + (x - lv.ox) * self.sc), int(self.oy0 + (lv.H * TILE - (y - lv.oy)) * self.sc))

    def chart(self, x, y, w, h, title, series, fmt="{:.2f}"):
        """series: [(label, values, colour)] drawn on one shared scale."""
        pygame.draw.rect(self.screen, (30, 32, 44), (x, y, w, h))
        vals = [v for _, s, _ in series for v in s if v is not None and math.isfinite(v)]
        self.text(title, x + 4, y + 2, DIM, self.small)
        if not vals:
            self.text("(no data yet)", x + w // 2 - 40, y + h // 2 - 6, DIM, self.small)
            return
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            hi, lo = hi + 1, lo - 1
        top, bot = y + 30, y + h - 4
        lx = x + 4
        for label, s, col in series:
            s = [v for v in s if v is not None and math.isfinite(v)]
            if not s:
                continue
            n = len(s)
            pts = [(x + 4 + (w - 60) * (k / max(1, n - 1)), bot - (bot - top) * (v - lo) / (hi - lo)) for k, v in enumerate(s)]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, col, False, pts, 2)
            else:
                pygame.draw.circle(self.screen, col, (int(pts[0][0]), int(pts[0][1])), 3)
            lab = self.small.render(f"{label} {fmt.format(s[-1])}", True, col)
            self.screen.blit(lab, (lx, y + 15))
            lx += lab.get_width() + 12
        self.text(fmt.format(hi), x + w - 50, top - 2, GRID, self.small)
        self.text(fmt.format(lo), x + w - 50, bot - 12, GRID, self.small)

    # ---- frame ----------------------------------------------------------------------------------------
    def draw(self):
        t, scr = self.t, self.screen
        scr.fill(PANEL)
        lv = t.level
        if lv is None:
            scr.fill(BG, (0, 0, MAP_W, WIN_H))
            self.text("waiting for a level...", 30, 30, DIM, self.big)
        else:
            if self.level_id is not id(lv):
                self.level_id = id(lv)
                self.bg = self._level_bg(lv)
            scr.blit(self.bg, (0, 0))
            self.draw_map(lv)
        self.draw_panel()

    def draw_map(self, lv):
        t, scr = self.t, self.screen
        for x, y, r in list(t.circles):
            pygame.draw.circle(scr, HAZ, self.pt(x, y), max(2, int(r * self.sc)))
        ov = pygame.Surface((MAP_W, WIN_H), pygame.SRCALPHA)
        for k, p in enumerate(list(t.paths)):
            pts = [self.pt(x, y) for x, y in p.points]
            if len(pts) > 1:
                pygame.draw.lines(ov, PATH_RGB[k % 3] + (170,), False, pts, 4)
        scr.blit(ov, (0, 0))
        states, path_of = list(t.last_states), list(t.path_of)
        for i, s in enumerate(states):
            x, y, st = s[0], s[1], s[6]
            px, py = self.pt(x, y)
            col = PATH_RGB[path_of[i] % 3] if i < len(path_of) else TEXT
            if st == 1:
                pygame.draw.line(scr, (200, 70, 70), (px - 3, py - 3), (px + 3, py + 3), 2)
                pygame.draw.line(scr, (200, 70, 70), (px - 3, py + 3), (px + 3, py - 3), 2)
            elif st == 2:
                pygame.draw.circle(scr, GOLD, (px, py), 6, 2)
            else:
                sz = max(4, int(12 * self.sc))
                rect = pygame.Rect(px - sz // 2, py - sz // 2, sz, sz)
                scr.fill(col, rect)
                if i == t.leader:
                    pygame.draw.rect(scr, (255, 255, 255), rect.inflate(6, 6), 2)

    def draw_panel(self):
        t, scr = self.t, self.screen
        x, y, w = PANEL_X, 10, PANEL_W
        lv = t.level
        self.text(f"{lv.name if lv else '-'}   wave {t.wave_no}  frame {t.frame_no}", x, y, TEXT, self.big)
        y += 24
        phase = "SPEED (beat the best time)" if t.phase == "speed" else "FINISH (reach the goal)"
        self.text(f"phase: {phase}", x, y, GOLD if t.phase == "speed" else (120, 220, 140))
        y += 18
        st = list(t.last_states)
        live = sum(1 for s in st if s[6] == 0)
        dead = sum(1 for s in st if s[6] == 1)
        won = sum(1 for s in st if s[6] == 2)
        best = f"{t.best_time:.2f}s" if t.best_time is not None else "--"
        self.text(f"agents {len(st)}: {live} alive, {dead} dead, {won} finished    best time {best}", x, y)
        y += 18
        px = x
        for k, p in enumerate(list(t.paths)):
            n = sum(1 for j in t.path_of if j == k)
            lab = self.small.render(f"path {k + 1}: {p.length / TILE:.0f} tiles, {n} agents", True, PATH_RGB[k % 3])
            scr.blit(lab, (px, y))
            px += lab.get_width() + 18
        if not t.paths:
            self.text("no reward path for this level", x, y, DIM, self.small)
        y += 22

        # leader's policy output
        probs, val = t.leader_probs, t.leader_value
        self.text("leader's policy  P(action | state)" + (f"    value {val:+.2f}" if val is not None else ""), x, y, (110, 190, 255))
        y += 18
        bw = w - 150
        for a in range(len(ACTION_NAMES)):
            p = float(probs[a]) if probs is not None and a < len(probs) else 0.0
            top = probs is not None and a == int(np.argmax(probs))
            self.text(f"{ACTION_NAMES[a]:8s}", x, y, (90, 240, 150) if top else DIM, self.small)
            pygame.draw.rect(scr, (40, 43, 58), (x + 70, y + 2, bw, 9))
            pygame.draw.rect(scr, (80, 230, 130) if top else (100, 120, 165), (x + 70, y + 2, int(bw * p), 9))
            self.text(f"{100 * p:5.1f}%", x + 76 + bw, y, DIM, self.small)
            y += 13
        y += 6

        H = 80
        hist = list(t.history)
        ups = list(t.updates)
        half = (w - 8) // 2
        self.chart(x, y, half, H, "finished agents / wave", [("fin", [h["finished"] for h in hist], GOLD),
                                                               ("dead", [h["dead"] for h in hist], (210, 90, 90))], "{:.0f}")
        self.chart(x + half + 8, y, half, H, "path progress % / wave", [("max", [h["path_max"] for h in hist], (90, 230, 130)),
                                                                        ("mean", [h["path_mean"] for h in hist], (90, 170, 255))], "{:.0f}")
        y += H + 6
        self.chart(x, y, half, H, "best fitness / wave", [("fit", [h["fit"] for h in hist], (230, 150, 255))], "{:.0f}")
        self.chart(x + half + 8, y, half, H, "fastest finish (s) / wave", [("time", [h["fastest"] for h in hist], GOLD)], "{:.2f}")
        y += H + 6
        self.chart(x, y, half, H, "mean return / wave", [("R", [h["ret"] for h in hist], (120, 220, 220))], "{:.1f}")
        self.chart(x + half + 8, y, half, H, "losses / PPO update", [("policy", [u["pg"] for u in ups], (255, 140, 90)),
                                                                     ("value", [u["vf"] for u in ups], (110, 190, 255))], "{:.3f}")
        y += H + 6
        self.chart(x, y, half, H, "entropy / update", [("H", [u["ent"] for u in ups], (190, 230, 90))], "{:.2f}")
        self.chart(x + half + 8, y, half, H, "KL, clip fraction / update", [("KL", [u["kl"] for u in ups], (255, 200, 90)),
                                                                            ("clip", [u["clip"] for u in ups], (200, 120, 255))], "{:.3f}")
        y += H + 6
        self.chart(x, y, half, H, "explained variance / update", [("EV", [u["ev"] for u in ups], (90, 220, 170))], "{:.2f}")
        self.chart(x + half + 8, y, half, H, "gradient norm / update", [("|g|", [u["gnorm"] for u in ups], (240, 120, 160))], "{:.2f}")
        y += H + 8
        if self.show_weights:
            self.draw_weights(x, y, w, WIN_H - y - 10)

    def draw_weights(self, x, y, w, h):
        t = self.t
        snap = t.weights
        if snap is None:
            return
        if snap["ver"] != self.wver:
            self.wver = snap["ver"]
            W = snap["w0"]                                       # hidden x inputs
            m = float(np.percentile(np.abs(W), 98)) or 1.0      # scale by the bulk, not the few largest
            v = np.clip(W / m, -1, 1)
            rgb = np.zeros(W.shape + (3,), np.uint8)
            rgb[..., 0] = (255 * np.clip(v, 0, 1)).astype(np.uint8)
            rgb[..., 2] = (255 * np.clip(-v, 0, 1)).astype(np.uint8)
            rgb[..., 1] = (60 * (1 - np.abs(v))).astype(np.uint8)
            self.wimg = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
        self.text(f"actor layer 1 weights ({snap['w0'].shape[0]} units x {snap['w0'].shape[1]} inputs; "
                  f"inputs: 10 state, 121 local map, {snap['w0'].shape[1] - 131} path)   [W] hide", x, y, DIM, self.small)
        img_h = h - 34
        if self.wimg is not None and img_h > 10:
            scr = pygame.transform.scale(self.wimg, (w, img_h))
            self.screen.blit(scr, (x, y + 16))
        norms = "  ".join(f"{k} {v:.1f}" for k, v in snap["norms"].items())
        self.text(f"|W| per layer: {norms}", x, y + h - 14, DIM, self.small)
