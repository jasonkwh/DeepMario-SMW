from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from config import (
    FRAME_STACK,
    GAME_NAME,
    SEED,
    STATE_NAME,
    VIDEO_EVERY_N_BEST,
)
from environment import MarioEnvironment


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
            [
                lambda: MarioEnvironment(
                    GAME_NAME,
                    STATE_NAME,
                    monitor_filename=None,
                    record_video=True,
                    seed=SEED + 20_000,
                )
            ],
            start_method="spawn",
        )
        video_env = VecFrameStack(
            video_env,
            n_stack=FRAME_STACK,
            channels_order="last",
        )
        video_env = VecTransposeImage(video_env)

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
