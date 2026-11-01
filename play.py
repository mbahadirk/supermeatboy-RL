"""Playable viewer/renderer for the clean-room engine.

Run (interactive window):
    python smbrl/play.py                 # starts on the first level
    python smbrl/play.py --level shaft
Controls:
    Left / Right arrows : move      Up / Space : jump
    Shift               : run        R : restart   N / P : next / prev level
    Esc                 : quit
Headless self-check (no window, for CI):
    python smbrl/play.py --selftest
"""
from __future__ import annotations
import argparse
import os

from engine import World, Action, TILE
import levels

W, H = 960, 540
BG = (18, 18, 24)
COL_SOLID = (60, 64, 82)
COL_SOLID_EDGE = (90, 96, 120)
COL_HAZARD = (210, 60, 60)
COL_GOAL = (240, 120, 180)
COL_PLAYER = (230, 70, 70)
COL_TEXT = (220, 220, 230)


def act_from_keys(pygame) -> Action:
    k = pygame.key.get_pressed()
    return Action(
        left=k[pygame.K_LEFT] or k[pygame.K_a],
        right=k[pygame.K_RIGHT] or k[pygame.K_d],
        jump=k[pygame.K_UP] or k[pygame.K_SPACE] or k[pygame.K_w],
        run=k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT],
    )


def draw(pygame, screen, font, world, level, name, order_idx, order_len):
    p = world.p
    s = world.state()
    # camera centered on player, clamped to level bounds
    lvl_w, lvl_h = level.cols * TILE, level.rows * TILE
    camx = max(0, min(int(s["cx"] - W / 2), max(0, lvl_w - W)))
    camy = max(0, min(int(s["cy"] - H / 2), max(0, lvl_h - H)))
    screen.fill(BG)
    c0, c1 = camx // TILE, (camx + W) // TILE + 1
    r0, r1 = camy // TILE, (camy + H) // TILE + 1
    for c in range(c0, c1 + 1):
        for r in range(r0, r1 + 1):
            x, y = c * TILE - camx, r * TILE - camy
            if (c, r) in level.solids:
                pygame.draw.rect(screen, COL_SOLID, (x, y, TILE, TILE))
                pygame.draw.rect(screen, COL_SOLID_EDGE, (x, y, TILE, TILE), 1)
            elif (c, r) in level.hazards:
                pygame.draw.polygon(screen, COL_HAZARD, [
                    (x, y + TILE), (x + TILE // 2, y), (x + TILE, y + TILE)])
    # goal
    gc, gr = level.goal_cell
    pygame.draw.rect(screen, COL_GOAL,
                     (gc * TILE - camx, gr * TILE - camy, TILE, TILE))
    # player
    pygame.draw.rect(screen, COL_PLAYER,
                     (int(s["x"]) - camx, int(s["y"]) - camy,
                      int(p.player_w), int(p.player_h)))
    # hud
    status = "WON!" if s["won"] else ("DEAD" if s["dead"] else "")
    txt = f"[{order_idx+1}/{order_len}] {name}   x={s['cx']:.0f} y={s['cy']:.0f} " \
          f"vx={s['vx']:.0f} vy={s['vy']:.0f} ground={int(s['on_ground'])} " \
          f"steps={s['steps']} {status}"
    screen.blit(font.render(txt, True, COL_TEXT), (8, 8))
    screen.blit(font.render("arrows=move  space=jump  shift=run  R=restart  N/P=level  Esc=quit",
                            True, (150, 150, 165)), (8, H - 22))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    names = list(levels.LEVELS.keys())
    idx = names.index(args.level) if args.level in names else 0

    if args.selftest:
        screen = pygame.Surface((W, H))
        font = pygame.font.SysFont("consolas", 16)
        for nm in names:
            w = World(levels.get(nm))
            for _ in range(30):
                w.step(Action(right=True, jump=True))
            draw(pygame, screen, font, w, levels.get(nm), nm, names.index(nm), len(names))
        print(f"SELFTEST OK: initialized+rendered {len(names)} levels: {names}")
        pygame.quit()
        return 0

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Clean-room Meat Boy (viewer)")
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    def load(i):
        nm = names[i % len(names)]
        return nm, levels.get(nm), World(levels.get(nm))

    name, level, world = load(idx)
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r:
                    world.reset()
                elif e.key in (pygame.K_n, pygame.K_p):
                    idx = (idx + (1 if e.key == pygame.K_n else -1)) % len(names)
                    name, level, world = load(idx)
        world.step(act_from_keys(pygame))
        if world.state()["dead"]:
            world.reset()
        draw(pygame, screen, font, world, level, name, idx, len(names))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
