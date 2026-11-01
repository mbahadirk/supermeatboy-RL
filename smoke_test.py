"""Headless smoke test: prove the clean-room engine runs and is deterministic.

Run:  python -I smbrl/smoke_test.py   (from the project root)
"""
from engine import World, Action
import levels


def run(level_name, policy, max_steps=1200):
    w = World(levels.get(level_name))
    traj = []
    for t in range(max_steps):
        a = policy(w, t)
        s = w.step(a)
        traj.append((round(s["cx"], 1), round(s["cy"], 1)))
        if s["dead"] or s["won"]:
            break
    return w.state(), traj


def hold_right(w, t):
    return Action(right=True, run=True, jump=False)


def main():
    ok = True

    # 1) FLAT: holding right must reach the goal
    s, traj = run("flat", hold_right)
    print(f"[flat] won={s['won']} dead={s['dead']} steps={s['steps']} "
          f"end=({s['cx']:.0f},{s['cy']:.0f}) goal_dist={s['goal_dist']:.0f}")
    ok &= s["won"]

    # 2) determinism: identical runs must match exactly
    s2, traj2 = run("flat", hold_right)
    same = (traj == traj2)
    print(f"[determinism] identical_trajectory={same}")
    ok &= same

    # 3) physics sanity: a jump leaves the ground then lands
    def jump_once(w, t):
        return Action(jump=(t < 6))
    w = World(levels.get("flat"))
    max_air = 0.0
    y0 = w.state()["cy"]
    for t in range(120):
        s = w.step(jump_once(w, t))
        max_air = max(max_air, y0 - s["cy"])   # +Y down, so up = y decreasing
        if s["on_ground"] and t > 10:
            break
    print(f"[jump] peak_height_px={max_air:.0f} landed={s['on_ground']}")
    ok &= (max_air > TILE_HINT) and s["on_ground"]

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


TILE_HINT = 24  # expect a jump to clear at least ~1 tile of height

if __name__ == "__main__":
    raise SystemExit(main())
