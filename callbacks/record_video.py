from uuid import uuid4

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from config import (
    GAME_NAME,
    SEED,
    STATE_NAME,
    VIDEO_DIR,
    VIDEO_EVERY_N_BEST,
    VIDEO_RENDER_FPS,
    MODEL_NAME,
    ENCODER,
)
from encoders import build_encoder
from environment import MarioEnvironment

VIDEO_SEED = SEED + 20_000


def build_video_env(seed: int = VIDEO_SEED):
    return gym.wrappers.RecordVideo(
        MarioEnvironment(
            GAME_NAME,
            STATE_NAME,
            seed=seed,
        ),
        video_folder=VIDEO_DIR,
        episode_trigger=lambda _: True,
        fps=VIDEO_RENDER_FPS,
        name_prefix=f"{MODEL_NAME}-{seed}-{uuid4().hex[:12]}",
    )


class RecordVideoCallback(BaseCallback):
    def __init__(self, record_every: int = VIDEO_EVERY_N_BEST):
        super().__init__()
        self.record_every = max(1, record_every)
        self.new_best_count = 0

    def _on_step(self) -> bool:
        self.new_best_count += 1

        if self.new_best_count % self.record_every != 0:
            return True

        video_env = SubprocVecEnv(
            [lambda: build_video_env()],
            start_method="spawn",
        )
        video_env = build_encoder(ENCODER).wrap_vec_env(video_env)

        try:
            obs = video_env.reset()
            done = [False]
            while not done[0]:
                action, _ = self.model.predict(
                    obs,
                    deterministic=True,
                )
                obs, _, done, _ = video_env.step(action)
        finally:
            video_env.close()

        return True
