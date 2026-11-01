"""Our own, original ASCII level format (not the game's .smbl).

Legend:
  '#'  solid tile
  '^'  hazard (spike/saw) -> death on contact
  'w'  wind (upward push; only used when Physics.wind_accel > 0)
  'S'  spawn
  'G'  goal (Bandage Girl)
  ' '  or '.'  empty

These sample maps are original layouts authored by us for training/eval, so they
carry no third-party copyright. A separate converter can import personal-use
`.smbl` levels later; keep those out of anything distributed.
"""
from __future__ import annotations
from engine import TILE


class Level:
    def __init__(self, name: str, grid: list[str]):
        self.name = name
        self.grid = grid
        self.rows = len(grid)
        self.cols = max(len(r) for r in grid)
        self.solids: set[tuple[int, int]] = set()
        self.hazards: set[tuple[int, int]] = set()
        self.wind: set[tuple[int, int]] = set()
        # partial solids among `solids`: (col, row) -> ("F" floor | "C" ceiling, a, b); the solid part's height
        # (F, from the cell bottom) or depth (C, from the cell top) runs linearly from a at the left edge to b
        # at the right edge, in tile fractions. Cells not listed are full blocks.
        self.shapes: dict[tuple[int, int], tuple[str, float, float]] = {}
        self.spawn_center: tuple[float, float] | None = None  # exact spawn (player centre, px) if known
        self.spawn_grounded = False   # player may jump on the first frame (counts as standing)
        self.spawn_facing = 1
        # moving hazards: dicts {"keys": [(t_seconds, cx_px, cy_px), ...], "period": s, "r": px}; the path
        # wraps after its last key. Positions come from World.mover_positions().
        self.movers: list[dict] = []
        # crumble blocks (subset of solids): break crumble_frames after the player first touches them and stay
        # broken until the episode resets. portals: pairs of centres (px) that teleport the player between them.
        self.crumble: set[tuple[int, int]] = set()
        self.crumble_frames = 18
        self.portals: list[tuple[tuple[float, float], tuple[float, float]]] = []
        # moving platforms: {"keys": [(t, cx, cy)], "period": s, "hw": half width px, "hh": half height px};
        # solid boxes that carry a player standing on them
        self.platforms: list[dict] = []
        spawn_cell = None
        goal_cell = None
        for r, line in enumerate(grid):
            for c, ch in enumerate(line):
                if ch == '#':
                    self.solids.add((c, r))
                elif ch == '^':
                    self.hazards.add((c, r))
                elif ch == 'w':
                    self.wind.add((c, r))
                elif ch == 'S':
                    spawn_cell = (c, r)
                elif ch == 'G':
                    goal_cell = (c, r)
        if spawn_cell is None or goal_cell is None:
            raise ValueError(f"level {name}: needs both S and G")
        self.spawn_cell = spawn_cell
        self.goal_cell = goal_cell
        # player AABB top-left, centered horizontally in the spawn cell, resting
        # near the cell floor (engine will settle it onto the ground on frame 1)
        from engine import Physics
        pw, ph = Physics.player_w, Physics.player_h
        self.spawn = (spawn_cell[0] * TILE + (TILE - pw) / 2,
                      spawn_cell[1] * TILE + (TILE - ph))


# A trivial flat corridor: holding RIGHT must reach the goal (sanity map).
FLAT = Level("flat", [
    "                         ",
    "                         ",
    "                         ",
    " S                     G ",
    "#########################",
    "#########################",
])

# Requires a jump over a small gap and onto a step.
GAP = Level("gap", [
    "                             ",
    "                             ",
    "                        G    ",
    "                     ####### ",
    " S          ^^^              ",
    "########     ########        ",
    "########     ########        ",
])

# A short wall-jump shaft.
SHAFT = Level("shaft", [
    "     G     ",
    "     #     ",
    "#   ##     ",
    "#         #",
    "#         #",
    "#   ##   ##",
    "# S #   ## ",
    "#####   ## ",
    "###########",
])

LEVELS = {lv.name: lv for lv in (FLAT, GAP, SHAFT)}


def get(name: str) -> Level:
    """Built-in original maps by name; "smb:<name>" loads a personal-use import (see levels_smb.py)."""
    if name.startswith("smb:"):
        import levels_smb
        return levels_smb.load(name[4:])
    return LEVELS[name]
