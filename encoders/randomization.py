from __future__ import annotations

import gymnasium as gym
import numpy as np


class ObservationAugmentation(gym.ObservationWrapper):
    """Photometric and translational jitter applied to the final image observation.

    Colour parameters are drawn once per episode and frozen, so the network sees a
    consistent palette within an episode instead of per-step flicker. The
    translation is drawn per step, which is the regularisation effect we want.

    Observation dtype, shape and value range are preserved.
    """

    def __init__(
        self,
        env: gym.Env,
        contrast: float = 0.15,
        brightness: float = 18.0,
        gamma: float = 0.2,
        shift: int = 2,
        shift_probability: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(env)
        self._contrast = contrast
        self._brightness = brightness
        self._gamma = gamma
        self._shift = shift
        self._shift_probability = shift_probability
        self._rng = np.random.default_rng(seed)
        # contrast, brightness, gamma
        self._colour: tuple[float, float, float] = (1.0, 0.0, 1.0)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._draw_colour()
        return self.observation(obs), info

    def _draw_colour(self) -> None:
        self._colour = (
            1.0 + self._rng.uniform(-self._contrast, self._contrast),
            self._rng.uniform(-self._brightness, self._brightness),
            1.0 + self._rng.uniform(-self._gamma, self._gamma),
        )

    def observation(self, observation):
        image = observation.astype(np.float32)
        contrast, brightness, gamma = self._colour

        if contrast != 1.0:
            image = (image - float(image.mean())) * contrast + float(image.mean())
        if brightness != 0.0:
            image += brightness
        if gamma != 1.0:
            image = 255.0 * np.power(np.clip(image, 0.0, 255.0) / 255.0, gamma)

        image = np.clip(image, 0.0, 255.0).astype(np.uint8)

        if self._shift > 0 and self._rng.random() < self._shift_probability:
            image = self._translate(image, self._rng, self._shift)

        return image

    @staticmethod
    def _translate(
        image: np.ndarray, rng: np.random.Generator, max_shift: int
    ) -> np.ndarray:
        dy = int(rng.integers(-max_shift, max_shift + 1))
        dx = int(rng.integers(-max_shift, max_shift + 1))
        if dy == 0 and dx == 0:
            return image

        height, width = image.shape[:2]
        padded = np.pad(
            image,
            ((max_shift, max_shift), (max_shift, max_shift), (0, 0)),
            mode="edge",
        )
        top = max_shift - dy
        left = max_shift - dx
        return padded[top : top + height, left : left + width]
