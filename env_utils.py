"""
env_utils.py — MiniGrid environment setup with vectorization
and optional observation corruption.
"""

import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv


class ObservationCorruption(gym.ObservationWrapper):
    """
    Timestep-level observation dropout: with probability p, the entire
    observation is replaced with zeros.

    This simulates complete sensor failure at a given timestep rather than
    per-pixel noise. At p=0.3, roughly 30% of timesteps receive a fully
    blank observation while 70% receive the true observation.

    Uses a seeded RNG for reproducibility. The RNG is re-seeded when
    reset(seed=...) is called.
    """

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
    """
    Extracts the 'image' key from MiniGrid dict observations.
    Returns an (H, W, C) uint8 array.
    """

    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError(
                f"Expected Dict observation space, got {type(env.observation_space)}. "
                f"Is this a MiniGrid environment?"
            )
        if "image" not in env.observation_space:
            raise KeyError(
                "MiniGrid observation space does not contain 'image' key. "
                f"Available keys: {list(env.observation_space.keys())}"
            )
        self.observation_space = env.observation_space["image"]

    def observation(self, obs):
        if isinstance(obs, dict):
            return obs["image"]
        return obs  # Already extracted by a prior wrapper


# Mapping from short names used in configs/CLI to Gymnasium environment IDs.
ENV_MAP = {
    "MemoryS7": "MiniGrid-MemoryS7-v0",
    "DoorKey": "MiniGrid-DoorKey-8x8-v0",
    "DynamicObstacles": "MiniGrid-Dynamic-Obstacles-8x8-v0",
}

# Default episode length for all environments.
MAX_EPISODE_STEPS = 256


def _make_single_env(env_name, seed, corruption_p=0.0):
    """
    Factory function that returns a callable creating a single environment.

    The callable is suitable for passing to SyncVectorEnv / AsyncVectorEnv.
    """

    def _init():
        import minigrid  # noqa: F401  # registers MiniGrid environments

        if env_name not in ENV_MAP:
            raise ValueError(
                f"Unknown environment: {env_name!r}. "
                f"Available: {sorted(ENV_MAP.keys())}"
            )

        env_id = ENV_MAP[env_name]

        # Pass max_episode_steps to gym.make to avoid double TimeLimit wrapping.
        # gym.make applies TimeLimit automatically if the registration specifies
        # max_episode_steps; passing it here overrides the registered default.
        env = gym.make(env_id, max_episode_steps=MAX_EPISODE_STEPS)
        env = FlattenObsWrapper(env)

        if corruption_p > 0.0:
            env = ObservationCorruption(env, p=corruption_p, seed=seed)

        # Validate observation space shape for model compatibility
        obs_shape = env.observation_space.shape
        assert len(obs_shape) == 3, (
            f"Expected (H, W, C) observation space after wrapping, "
            f"got shape {obs_shape}"
        )

        env.reset(seed=seed)
        return env

    return _init


def make_env(env_name, n_envs, seed, corruption_p=0.0, use_async=False):
    """
    Create a vectorized MiniGrid environment.

    Args:
        env_name:     Short name (key in ENV_MAP), e.g. "MemoryS7"
        n_envs:       Number of parallel environments
        seed:         Base seed; env i gets seed (seed + i)
        corruption_p: Observation dropout probability (0.0 = no corruption)
        use_async:    If True, use AsyncVectorEnv for parallel stepping

    Returns:
        A Gymnasium VectorEnv with n_envs parallel environments.
    """
    fns = [
        _make_single_env(env_name, seed + i, corruption_p)
        for i in range(n_envs)
    ]
    vec_cls = AsyncVectorEnv if use_async else SyncVectorEnv
    return vec_cls(fns)