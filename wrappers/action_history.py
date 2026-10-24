from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ActionHistory(gym.ObservationWrapper):
    """Append the last ``history`` actions to the observation as constant planes.

    Must sit above the image-processing wrappers so it sees the final
    ``(H, W, C)`` observation, and above the action wrapper so ``action`` is the
    discrete index rather than the raw button vector.

    Action indices are scaled into 0-255 so the observation stays uint8 and the
    stock CnnPolicy keeps working without a custom features extractor.
    """

    def __init__(self, env: gym.Env, history: int = 3) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Discrete):
            raise TypeError(
                f"ActionHistory expects a Discrete action space, got {env.action_space}"
            )

        self._history = history
        self._n_actions = int(env.action_space.n)
        height, width, channels = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(height, width, channels + history),
            dtype=np.uint8,
        )
        self._last_actions = np.zeros(history, dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_actions = np.zeros(self._history, dtype=np.uint8)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._push(action)
        return self.observation(obs), reward, terminated, truncated, info

    def _push(self, action: int) -> None:
        # +1 keeps "no action yet" (0) distinct from action index 0.
        scaled = 255.0 * (int(action) + 1) / (self._n_actions + 1)
        self._last_actions = np.concatenate(
            (
                np.array([scaled], dtype=np.uint8),
                self._last_actions[:-1],
            )
        )

    def observation(self, observation):
        height, width = observation.shape[:2]
        planes = [
            np.full((height, width, 1), int(value), dtype=np.uint8)
            for value in self._last_actions
        ]
        return np.concatenate([observation, *planes], axis=2)
