"""Live training monitor for realgame.py (pygame window next to the game).

Left: the level as the trainer sees it - tiles, lethal pieces, the reward paths (one colour each), every
agent coloured by the path it follows (x = dead, ring = finished, white box = leader). Below it the
controls: the number of agents per wave and how they are split over the routes (both from the next wave;
"auto" = even split, and half on the fastest route once someone finished).
Right: the wave and phase, the leader's policy output (action probabilities, value), learning curves per
wave and per PPO update, and the network as it played the previous wave, run on the last input of that
wave's best agent: input -> hidden 1 -> hidden 2 -> actions, the most active units of each layer, line
width = weight size (red positive, blue negative), node colour = activation.

Keys (monitor window focused): P routes on/off in the game, Esc closes the monitor only.
"""
from __future__ import annotations

import os
import math

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_WINDOW_NO_ACTIVATION_WHEN_SHOWN", "1")   # do not take focus from the game
import pygame
import numpy as np

from realpath import PATH_RGB

TILE = 20.0
WIN_W, WIN_H = 1500, 900
MAP_W = 860
CTRL_H = 150
MAP_H = WIN_H - CTRL_H
PANEL_X = MAP_W + 16
PANEL_W = WIN_W - PANEL_X - 12
MAX_AGENTS = 40

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
SPECIAL = (150, 128, 70)     # crumble / lock / switch blocks still standing
KEY = (255, 230, 40)
PORTAL = (170, 90, 255)
WIND = (40, 70, 95)          # fan wind columns (map background)
ACCEL = (190, 70, 200)       # accelerator tiles
ORB = (120, 200, 255)
N_ROUTES = 4                 # 3 main routes + the bandage route
POS, NEG = (235, 80, 70), (70, 140, 255)
ACTION_NAMES = ["idle", "left", "right", "jump", "L+jump", "R+jump", "R+run", "R+run+J", "L+run", "L+run+J"]
STATE_NAMES = ["x", "y", "speed x", "speed y", "on ground", "on wall", "facing", "goal dx", "goal dy", "goal dist"]
PATH_NAMES = ["route+2 dx", "route+2 dy", "route+6 dx", "route+6 dy", "progress", "off route"]
HAZ_NAMES = [f"mover{k} {q}" for k in (1, 2) for q in ("dx", "dy", "vx", "vy")] + ["wall left", "wall right", "wall slide", "wind x", "wind y", "in wind", "orb dx", "orb dy", "in orb"]
K = 5                                    # local map radius in realgame.observe
EDGES_PER_NODE = 4


def input_name(j: int) -> str:
    if j < len(STATE_NAMES):
        return STATE_NAMES[j]
    j -= len(STATE_NAMES)
    n = (2 * K + 1) ** 2
    if j < n:                            # patch rows top-down
        return f"map {j % (2 * K + 1) - K:+d},{K - j // (2 * K + 1):+d}"
    j -= n
    if j < len(PATH_NAMES):
        return PATH_NAMES[j]
    j -= len(PATH_NAMES)
    if j < len(HAZ_NAMES):
        return HAZ_NAMES[j]
    j -= len(HAZ_NAMES)
    if j < n:                            # surface map, rows top-down
        return f"surf {j % (2 * K + 1) - K:+d},{K - j // (2 * K + 1):+d}"
    j -= n
    if j < 2:
        return ("stuck time", "jumps stuck")[j]
    return f"in {j}"


def split(n: int, share, k: int) -> list[int]:
    """Agents per route for shares (same rule as Trainer.assign_paths)."""
    w = [max(0.0, float(share[j])) if j < len(share) else 0.0 for j in range(k)]
    if sum(w) <= 0:
        w = [1.0] * k
    exact = [n * x / sum(w) for x in w]
    cnt = [int(e) for e in exact]
    for j in sorted(range(k), key=lambda j: exact[j] - cnt[j], reverse=True)[:n - sum(cnt)]:
        cnt[j] += 1
    return cnt


def mix(a, b, f):
    return tuple(int(a[i] + (b[i] - a[i]) * f) for i in range(3))


class Slider:
    def __init__(self, x, y, w, lo, hi, value, col):
        self.x, self.y, self.w, self.lo, self.hi, self.value, self.col = x, y, w, lo, hi, value, col
        self.drag = False

    def knob_x(self):
        return self.x + (self.value - self.lo) / (self.hi - self.lo) * self.w

    def hit(self, pos):
        return self.x - 8 <= pos[0] <= self.x + self.w + 8 and self.y - 9 <= pos[1] <= self.y + 9

    def set_from(self, px):
        f = min(1.0, max(0.0, (px - self.x) / self.w))
        self.value = int(round(self.lo + f * (self.hi - self.lo)))

    def draw(self, scr, active=True):
        col = self.col if active else mix(self.col, PANEL, 0.6)
        pygame.draw.rect(scr, (44, 47, 62), (self.x, self.y - 3, self.w, 6), border_radius=3)
        pygame.draw.rect(scr, col, (self.x, self.y - 3, int(self.knob_x() - self.x), 6), border_radius=3)
        pygame.draw.circle(scr, (240, 240, 245) if active else DIM, (int(self.knob_x()), self.y), 8)


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
        self.closed = False
        y0 = MAP_H + 16
        self.s_agents = Slider(170, y0 + 8, 360, 1, getattr(trainer, "pool", MAX_AGENTS), trainer.want_agents,
                               (230, 230, 240))
        self.s_share = [Slider(170, y0 + 42 + 22 * k, 360, 0, 100, 25, PATH_RGB[k]) for k in range(N_ROUTES)]
        self.auto_rect = pygame.Rect(650, y0 + 36, 130, 26)
        self.net_key = None
        self.net_cache = None

    # ---- main loop (main thread) ------------------------------------------------------------------
    def tick(self):
        if self.closed:
            return
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                self.close()
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_p:
                self.t.toggle_game_paths()
            self.handle_controls(e)
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

    # ---- controls -------------------------------------------------------------------------------------
    def handle_controls(self, e):
        t = self.t
        k = len(t.paths)
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.auto_rect.collidepoint(e.pos):
                t.set_path_share(None if t.path_share is not None else [s.value for s in self.s_share])
                return
            for s in [self.s_agents] + self.s_share[:k]:
                if s.hit(e.pos):
                    s.drag = True
                    s.set_from(e.pos[0])
        elif e.type == pygame.MOUSEMOTION:
            for s in [self.s_agents] + self.s_share:
                if s.drag:
                    s.set_from(e.pos[0])
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            if self.s_agents.drag:
                self.s_agents.drag = False
                t.set_agents(self.s_agents.value)
            if any(s.drag for s in self.s_share):
                for s in self.s_share:
                    s.drag = False
                t.set_path_share([s.value for s in self.s_share[:k]])

    def draw_controls(self):
        t, scr = self.t, self.screen
        y0 = MAP_H + 16
        scr.fill(PANEL, (0, MAP_H, MAP_W, CTRL_H))
        pygame.draw.line(scr, GRID, (0, MAP_H), (MAP_W, MAP_H), 1)
        if not self.s_agents.drag:
            self.s_agents.value = t.want_agents
        now = len(t.last_states)
        self.text("agents per wave", 14, y0, TEXT)
        self.s_agents.draw(scr)
        self.text(f"{self.s_agents.value:2d}" + (f"  (this wave {now})" if now and now != self.s_agents.value else ""),
                  545, y0, GOLD)
        k = len(t.paths)
        auto = t.path_share is None
        n = self.s_agents.value
        if auto:                                  # show what the automatic split gives
            cnt = [sum(1 for j in t.assign_paths(n) if j == r) for r in range(k)]
            for r in range(N_ROUTES):
                if not self.s_share[r].drag:
                    self.s_share[r].value = int(round(100 * cnt[r] / max(1, n))) if r < k else 0
        else:
            cnt = split(n, [s.value for s in self.s_share[:k]], k) if k else []
        paths = list(t.paths)
        for r in range(N_ROUTES):
            y = y0 + 34 + 22 * r
            on = r < k and r < len(paths)
            col = self.route_col(r, paths[r]) if on else GRID
            self.s_share[r].col = col if on else PATH_RGB[r]
            name = ("bandage route" if paths[r].kind == "bandage" else f"route {r + 1}") if on else f"route {r + 1}"
            stall = on and hasattr(t, "stalled") and t.stalled(r)
            self.text(f"{name} share" if on else f"{name} (none)", 14, y, col)
            self.s_share[r].draw(scr, active=on and not auto)
            if on:
                self.text(f"{cnt[r]:2d} agents" + ("  stalled" if stall else ""), 545, y, col)
        pygame.draw.rect(scr, (60, 120, 80) if auto else (50, 53, 70), self.auto_rect, border_radius=5)
        lab = self.font.render("auto split" if auto else "manual split", True, TEXT)
        scr.blit(lab, lab.get_rect(center=self.auto_rect.center))
        self.text("route slider: split by hand", 650, y0 + 70, DIM, self.small)
        self.text("button: back to auto", 650, y0 + 84, DIM, self.small)
        self.text("changes apply from the next wave", 650, y0 + 100, DIM, self.small)
        self.text("P: routes on/off in the game", 650, y0 + 114, DIM, self.small)

    def route_col(self, k, p):
        return PATH_RGB[3] if getattr(p, "kind", "main") == "bandage" else PATH_RGB[k % 3]

    # ---- helpers --------------------------------------------------------------------------------------
    def text(self, s, x, y, col=TEXT, font=None):
        self.screen.blit((font or self.font).render(s, True, col), (x, y))

    def _level_bg(self, lv):
        sc = min((MAP_W - 20) / (lv.W * TILE), (MAP_H - 20) / (lv.H * TILE))
        self.sc = sc
        self.ox0 = 10 + (MAP_W - 20 - lv.W * TILE * sc) / 2
        self.oy0 = 10 + (MAP_H - 20 - lv.H * TILE * sc) / 2
        surf = pygame.Surface((MAP_W, MAP_H))
        surf.fill(BG)
        ts = TILE * sc
        special = getattr(lv, "special", None)
        for r in range(lv.H):
            for c in range(lv.W):
                wind = getattr(lv, "wind", None)
                if wind is not None and wind[r, c] and not lv.solid[r, c]:
                    x0, y0 = self.ox0 + c * ts, self.oy0 + (lv.H - 1 - r) * ts
                    surf.fill(WIND, pygame.Rect(int(x0), int(y0), max(1, int(x0 + ts) - int(x0)), max(1, int(y0 + ts) - int(y0))))
                if lv.solid[r, c]:                # live: broken crumble blocks and opened locks are gone
                    x0, y0 = self.ox0 + c * ts, self.oy0 + (lv.H - 1 - r) * ts
                    col = SPECIAL if special is not None and special[r, c] else                         ACCEL if getattr(lv, "accel", None) is not None and lv.accel[r, c] else SOLID
                    surf.fill(col, pygame.Rect(int(x0), int(y0), max(1, int(x0 + ts) - int(x0)), max(1, int(y0 + ts) - int(y0))))
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
            self.text("(no data yet)", x + 6, y + h // 2, GRID, self.small)
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
            pts = [(x + 4 + (w - 8) * (k / max(1, n - 1)), bot - (bot - top) * (v - lo) / (hi - lo)) for k, v in enumerate(s)]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, col, False, pts, 2)
            else:
                pygame.draw.circle(self.screen, col, (int(pts[0][0]), int(pts[0][1])), 3)
            lab = self.small.render(f"{label} {fmt.format(s[-1])}", True, col)
            self.screen.blit(lab, (lx, y + 15))
            lx += lab.get_width() + 10

    # ---- frame ----------------------------------------------------------------------------------------
    def draw(self):
        t, scr = self.t, self.screen
        scr.fill(PANEL)
        lv = t.level
        if lv is None:
            scr.fill(BG, (0, 0, MAP_W, MAP_H))
            self.text("waiting for a level...", 30, 30, DIM, self.big)
        else:
            ver = (id(lv), getattr(lv, "tile_ver", 0))
            if self.level_id != ver:              # a new level, or tiles changed in the game
                self.level_id = ver
                self.bg = self._level_bg(lv)
            scr.blit(self.bg, (0, 0))
            self.draw_map(lv)
        self.draw_controls()
        self.draw_panel()

    def draw_map(self, lv):
        t, scr = self.t, self.screen
        for x, y, hw, hh, a in list(t.boxes):     # lethal pieces as the routes see them (rotated boxes)
            c, sn = math.cos(a), math.sin(a)
            pts = [self.pt(x + u * c - v * sn, y + u * sn + v * c) for u, v in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
            pygame.draw.polygon(scr, HAZ, pts)
        for kx, ky in getattr(lv, "keys", []):
            px, py = self.pt(kx, ky)
            pygame.draw.polygon(scr, KEY, [(px, py - 7), (px + 5, py), (px, py + 7), (px - 5, py)])
        for bx, by in getattr(lv, "bandages", []):
            pygame.draw.circle(scr, (255, 255, 255), self.pt(bx, by), 6, 2)
        for x, y, r1, r2, s1, s2 in list(getattr(lv, "orbs", [])):
            pygame.draw.circle(scr, ORB, self.pt(x, y), max(3, int(max(r1, r2) * self.sc)), 1)
        for a, b in list(getattr(lv, "portals", [])) + list(getattr(t, "links", [])):
            pygame.draw.line(scr, PORTAL, self.pt(*a), self.pt(*b), 1)
            pygame.draw.circle(scr, PORTAL, self.pt(*a), 6, 2)
            pygame.draw.circle(scr, PORTAL, self.pt(*b), 6, 2)
        ov = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)
        for k, p in enumerate(list(t.paths)):
            for pc in p.pieces():
                pts = [self.pt(x, y) for x, y in pc]
                if len(pts) > 1:
                    pygame.draw.lines(ov, self.route_col(k, p) + (170,), False, pts, 4)
        scr.blit(ov, (0, 0))
        states, path_of, paths = list(t.last_states), list(t.path_of), list(t.paths)
        for i, s in enumerate(states):
            x, y, st = s[0], s[1], s[6]
            px, py = self.pt(x, y)
            if not (0 <= py < MAP_H):
                continue
            col = self.route_col(path_of[i], paths[path_of[i]]) if i < len(path_of) and path_of[i] < len(paths) else TEXT
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
            name = "bandage" if p.kind == "bandage" else f"route {k + 1}"
            lab = self.small.render(f"{name}: {p.length / TILE:.0f} tiles, {n} agents", True, self.route_col(k, p))
            scr.blit(lab, (px, y))
            px += lab.get_width() + 18
        if not t.paths:
            self.text("no reward path for this level", x, y, DIM, self.small)
        y += 20

        probs, val = t.leader_probs, t.leader_value
        self.text("leader's policy  P(action | state)" + (f"    value {val:+.2f}" if val is not None else ""), x, y, (110, 190, 255))
        y += 17
        bw = w - 150
        for a in range(len(ACTION_NAMES)):
            p = float(probs[a]) if probs is not None and a < len(probs) else 0.0
            top = probs is not None and a == int(np.argmax(probs))
            self.text(f"{ACTION_NAMES[a]:8s}", x, y, (90, 240, 150) if top else DIM, self.small)
            pygame.draw.rect(scr, (40, 43, 58), (x + 70, y + 2, bw, 8))
            pygame.draw.rect(scr, (80, 230, 130) if top else (100, 120, 165), (x + 70, y + 2, int(bw * p), 8))
            self.text(f"{100 * p:5.1f}%", x + 76 + bw, y - 1, DIM, self.small)
            y += 12
        y += 6

        hist, ups = list(t.history), list(t.updates)
        H, gap = 60, 6
        cw = (w - 2 * gap) // 3
        charts = [
            ("finished / dead per wave", [("fin", [h["finished"] for h in hist], GOLD), ("dead", [h["dead"] for h in hist], (210, 90, 90))], "{:.0f}"),
            ("route progress %", [("max", [h["path_max"] for h in hist], (90, 230, 130)), ("mean", [h["path_mean"] for h in hist], (90, 170, 255))], "{:.0f}"),
            ("best fitness", [("fit", [h["fit"] for h in hist], (230, 150, 255))], "{:.0f}"),
            ("fastest finish (s)", [("t", [h["fastest"] for h in hist], GOLD)], "{:.2f}"),
            ("hops per agent-second", [("hops", [h.get("hops") for h in hist], (255, 150, 90))], "{:.2f}"),
            ("mean return", [("R", [h["ret"] for h in hist], (120, 220, 220))], "{:.1f}"),
            ("policy loss / update", [("pg", [u["pg"] for u in ups], (255, 140, 90))], "{:.3f}"),
            ("value loss / update", [("vf", [u["vf"] for u in ups], (110, 190, 255))], "{:.2f}"),
            ("entropy / update", [("H", [u["ent"] for u in ups], (190, 230, 90))], "{:.2f}"),
            ("KL, clip fraction", [("KL", [u["kl"] for u in ups], (255, 200, 90)), ("clip", [u["clip"] for u in ups], (200, 120, 255))], "{:.3f}"),
            ("explained variance", [("EV", [u["ev"] for u in ups], (90, 220, 170))], "{:.2f}"),
            ("gradient norm", [("|g|", [u["gnorm"] for u in ups], (240, 120, 160))], "{:.2f}"),
        ]
        for n, (title, series, fmt) in enumerate(charts):
            cx, cy = x + (n % 3) * (cw + gap), y + (n // 3) * (H + gap)
            self.chart(cx, cy, cw, H, title, series, fmt)
        y += 4 * (H + gap) + 4
        self.draw_network(x, y, w, WIN_H - y - 8)

    # ---- network view -----------------------------------------------------------------------------------
    def network(self, net):
        """Forward pass of the snapshot on its agent's input and the units to show (cached per snapshot)."""
        key = (net["wave"], net["agent"])
        if self.net_key == key:
            return self.net_cache
        x = net["obs"]
        (W1, W2, W3), (b1, b2, b3) = net["W"], net["b"]
        h1 = np.tanh(W1 @ x + b1)
        h2 = np.tanh(W2 @ h1 + b2)
        z = W3 @ h2 + b3
        p = np.exp(z - z.max())
        p /= p.sum()
        n_in, n_h = 12, 12
        contrib = np.abs(x) * np.abs(W1).sum(0)                      # how much each input drives layer 1
        ins = [int(j) for j in np.argsort(-contrib)[:n_in] if contrib[j] > 0] or list(range(n_in))
        h1s = [int(j) for j in np.argsort(-np.abs(W1[:, ins] @ x[ins]))[:n_h]]
        h2s = [int(j) for j in np.argsort(-np.abs(W2[:, h1s] @ h1[h1s]))[:n_h]]
        self.net_key = key
        self.net_cache = {"x": x, "h1": h1, "h2": h2, "p": p, "ins": ins, "h1s": h1s, "h2s": h2s,
                          "E": [W1[np.ix_(h1s, ins)], W2[np.ix_(h2s, h1s)], W3[:, h2s]]}
        return self.net_cache

    def draw_network(self, x, y, w, h):
        t, scr = self.t, self.screen
        net = t.best_net
        pygame.draw.rect(scr, (20, 21, 29), (x, y, w, h))
        if net is None:
            self.text("network: shown after the first wave", x + 8, y + 8, DIM, self.small)
            return
        c = self.network(net)
        route = f", route {net['path'] + 1}" if net.get("path") is not None else ""
        self.text(f"network of wave {net['wave']}'s best agent (agent {net['agent']}, fitness {net['fit']:.0f}{route})",
                  x + 6, y + 4, TEXT, self.small)
        self.text(f"most active units, each with its {EDGES_PER_NODE} strongest inputs; width = |weight|, red +, blue -; "
                  "fill = activation", x + 6, y + 18, DIM, self.small)
        top, bot = y + 40, y + h - 10
        cols = [x + 120, x + 120 + (w - 250) / 3, x + 120 + 2 * (w - 250) / 3, x + w - 130]
        layers = [c["ins"], c["h1s"], c["h2s"], list(range(len(c["p"])))]
        vals = [c["x"], c["h1"], c["h2"], c["p"]]
        heads = ["inputs", "hidden 1", "hidden 2", "actions"]

        def ypos(n, k):
            return top + 12 + (bot - top - 24) * (k + 0.5) / n

        pos = [[(cols[L], ypos(len(layers[L]), k)) for k in range(len(layers[L]))] for L in range(4)]
        for L in range(3):                              # edges, weakest first so the strong ones stay on top
            E = c["E"][L]
            m = float(np.abs(E).max()) or 1.0
            keep = [(abs(E[a, b]), a, b) for a in range(E.shape[0])            # each node's 4 strongest inputs
                    for b in np.argsort(-np.abs(E[a]))[:EDGES_PER_NODE]]
            for mag, a, b in sorted(keep):
                f = mag / m
                col = mix((20, 21, 29), POS if E[a, b] > 0 else NEG, 0.25 + 0.75 * f)
                pygame.draw.line(scr, col, pos[L][b], pos[L + 1][a], max(1, int(round(1 + 6 * f * f))))
        best_a = int(np.argmax(c["p"]))
        for L in range(4):
            self.text(heads[L], cols[L] - 30, top - 4, DIM, self.small)
            for k, j in enumerate(layers[L]):
                v = float(vals[L][j])
                px, py = int(pos[L][k][0]), int(pos[L][k][1])
                if L == 3:
                    fill = mix((40, 42, 56), (80, 230, 130), v)
                else:
                    fill = mix((40, 42, 56), POS if v > 0 else NEG, min(1.0, abs(v)))
                pygame.draw.circle(scr, fill, (px, py), 7)
                pygame.draw.circle(scr, (200, 200, 210), (px, py), 7, 1)
                if L == 0:
                    lab = self.small.render(f"{input_name(j)} {v:+.2f}", True, DIM)
                    scr.blit(lab, (px - 12 - lab.get_width(), py - 7))
                elif L == 3:
                    self.text(f"{ACTION_NAMES[j]} {100 * v:.0f}%", px + 12, py - 7,
                              (90, 240, 150) if j == best_a else DIM, self.small)
