"""Model predictive control with the world model (wm/): candidates -> rollouts -> scores -> first action.

future.py      where hazards and platforms will be (recorded timeline of this level, or extrapolated)
candidates.py  action sequences to try (previous plan, PPO, primitives)
score.py       progress, death, hazard contact, landing, uncertainty, PPO value
controller.py  MPC.plan(): one decision, with timings; used by realgame.py --control mpc / mpc+ppo
"""
