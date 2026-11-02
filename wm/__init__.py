"""World model for the real-game agents: data collection (buffer), and later the dynamics model, rollouts,
planning (MPC) and the dashboard. Kept apart from the PPO trainer (realgame.py), which only calls in here when
asked to (--wm-collect, --probe-char)."""
