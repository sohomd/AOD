"""
tmaze_env.py — Classic T-Maze memory task (Bakker 2002 style).

The agent sees a CUE only at the very first timestep. It then walks down a
corridor of length L (seeing an uninformative "corridor" observation), and at
the T-junction must turn LEFT or RIGHT according to the cue seen at t=0.

Memory is required ONLY at the junction — the ideal testbed for surprise-driven
selective computation: local dynamics are trivial in the corridor, and the one
decision that matters is at the end.

Observation (flat float32 vector, size 3):
    [cue_signal, at_junction_flag, corridor_position_norm]
      cue_signal        = +1 / -1 at t=0, else 0
      at_junction_flag  = 1 when at the junction, else 0
      position_norm     = corridor progress in [0, 1]

Actions (Discrete 4): 0=up, 1=down, 2=left(forward), 3=right
    In the corridor, action 2 (forward) advances toward the junction.
    At the junction, action 0 (up) = go up-arm, action 1 (down) = go down-arm.

Reward:
    +1.0 for choosing the arm matching the cue at the junction
    -0.1 for choosing the wrong arm
     0   otherwise
    small -0.01 step penalty to encourage moving forward (optional)

This is a minimal, dependency-free implementation with a Discrete-free,
fixed-size Box observation so it plugs directly into a flat-vector encoder.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TMazeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, corridor_length=10, step_penalty=0.0, seed=None):
        super().__init__()
        self.L = int(corridor_length)
        self.step_penalty = float(step_penalty)

        # obs = [cue, at_junction, position_norm]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        # Episode can't exceed corridor + a little slack
        self.max_steps = self.L + 5
        self._rng = np.random.default_rng(seed)
        self._reset_state()

    def _reset_state(self):
        self.pos = 0                 # 0..L ; L == junction
        self.t = 0
        self.cue = 0.0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        # cue is +1 (up) or -1 (down), revealed only at t=0
        self.cue = 1.0 if self._rng.random() < 0.5 else -1.0
        return self._obs(reveal_cue=True), {}

    def _obs(self, reveal_cue=False):
        cue_signal = self.cue if reveal_cue else 0.0
        at_junction = 1.0 if self.pos >= self.L else 0.0
        pos_norm = min(self.pos / max(self.L, 1), 1.0)
        return np.array([cue_signal, at_junction, pos_norm], dtype=np.float32)

    def step(self, action):
        self.t += 1
        reward = -self.step_penalty
        terminated = False
        truncated = False

        at_junction = self.pos >= self.L

        if not at_junction:
            # Only forward (action 2) advances; other actions waste a step.
            if action == 2:
                self.pos += 1
            obs = self._obs(reveal_cue=False)
        else:
            # Decision point: action 0 = up-arm, action 1 = down-arm
            chosen = None
            if action == 0:
                chosen = 1.0
            elif action == 1:
                chosen = -1.0

            if chosen is not None:
                reward += 1.0 if chosen == self.cue else -0.1
                terminated = True
            # if agent dithers (action 2/3) at junction, it just waits
            obs = self._obs(reveal_cue=False)

        if self.t >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, {}
