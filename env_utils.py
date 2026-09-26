"""
env_utils.py — Environment setup with vectorization and optional
observation corruption. Supports three families:

  1. MiniGrid   (image obs, (H,W,C))   e.g. "MemoryS9", "DoorKey"
  2. POPGym     (flat vector obs)       e.g. "popgym-RepeatPreviousEasy-v0"
  3. T-Maze     (flat vector obs)       e.g. "TMaze10"  (local tmaze_env.py)

The encoder in models.py auto-selects CNN vs MLP based on obs rank, so all
three families share the same PPO/GRU/AoD stack.
"""

import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv, AutoresetMode





class ObservationCorruption(gym.ObservationWrapper):
    """With probability p, replace the entire observation with zeros."""

    def __init__(self, env, p=0.0, seed=None):
        super().__init__(env)
        self.p = float(p)
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return super().reset(seed=seed, options=options)

    def observation(self, obs):
        if self.p > 0.0 and self._rng.random() < self.p:
            return np.zeros_like(obs)
        return obs


class FlattenObsWrapper(gym.ObservationWrapper):
    """Extract the 'image' key from MiniGrid dict observations -> (H,W,C)."""

    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Dict):
            raise TypeError(f"Expected Dict obs space, got {type(obs_space)}.")
        if "image" not in obs_space.spaces:
            raise KeyError(
                "MiniGrid obs space has no 'image' key. "
                f"Available: {list(obs_space.spaces.keys())}"
            )
        self.observation_space = obs_space.spaces["image"]

    def observation(self, obs):
        if isinstance(obs, dict):
            return obs["image"]
        return obs


# ── Environment registry ───────────────────────────────────────────────
# MiniGrid short-name -> gym id. (max_steps left to the env's own default.)
MINIGRID_MAP = {
    "MemoryS7": "MiniGrid-MemoryS7-v0",
    "MemoryS9": "MiniGrid-MemoryS9-v0",
    "MemoryS11": "MiniGrid-MemoryS11-v0",
    "MemoryS13": "MiniGrid-MemoryS13-v0",
    "DoorKey": "MiniGrid-DoorKey-8x8-v0",
    "DynamicObstacles": "MiniGrid-Dynamic-Obstacles-8x8-v0",
}

# T-Maze short-name -> corridor length (uses local tmaze_env.py).
TMAZE_MAP = {
    "TMaze10": 10, "TMaze20": 20, "TMaze50": 50,
    "TMaze100": 100, "TMaze250": 250, "TMaze500": 500,
}


def _is_popgym(name):
    return name.startswith("popgym-")


def _make_single_env(env_name, seed, corruption_p=0.0):
    """Return a thunk that builds ONE environment of the requested family."""

    def _init():
        # ---- MiniGrid ----
        if env_name in MINIGRID_MAP:
            import minigrid  # noqa: F401  (registers envs)
            env = gym.make(MINIGRID_MAP[env_name])   # env's own max_steps
            env = FlattenObsWrapper(env)

        # ---- T-Maze (local) ----
        elif env_name in TMAZE_MAP:
            from tmaze_env import TMazeEnv
            env = TMazeEnv(corridor_length=TMAZE_MAP[env_name], seed=seed)

        # ---- POPGym ----
        elif _is_popgym(env_name):
            import popgym  # noqa: F401  (registers envs)
            env = gym.make(env_name)

        else:
            raise ValueError(
                f"Unknown environment: {env_name!r}. Known: "
                f"{sorted(list(MINIGRID_MAP) + list(TMAZE_MAP))} "
                f"or any 'popgym-*' id."
            )

        if corruption_p > 0.0:
            env = ObservationCorruption(env, p=corruption_p, seed=seed)

        env.reset(seed=seed)
        return env

    return _init


def make_env(env_name, n_envs, seed, corruption_p=0.0, use_async=False):
    """Create a vectorized environment for any supported family."""
    fns = [
        _make_single_env(env_name, seed + i, corruption_p)
        for i in range(n_envs)
    ]
    vec_cls = AsyncVectorEnv if use_async else SyncVectorEnv
    return vec_cls(fns, autoreset_mode=AutoresetMode.SAME_STEP)
