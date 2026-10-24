from uuid import uuid4

import gymnasium as gym
import stable_retro
from wrappers import (
    ActionHistory,
    ActionWrapper,
    ObservationAugmentation,
    RewardWrapper,
)
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.monitor import Monitor

from config import (
    VIDEO_DIR,
    FRAME_SKIP,
    OBSERVATION_SHAPE,
    VIDEO_RENDER_FPS,
    MAX_EPISODE_STEPS,
    MODEL_NAME,
)


class MarioEnvironment(gym.Wrapper):
    def __init__(
        self,
        game_name,
        state_name,
        monitor_filename=None,
        record_video=False,
        augment=False,
        seed=0,
    ):
        self._initial_seed = seed

        env = stable_retro.make(
            game=game_name,
            state=state_name,
            use_restricted_actions=stable_retro.Actions.ALL,
            render_mode="rgb_array",
        )

        env = ActionWrapper(env)
        env = RewardWrapper(env, state_name)
        env = MaxAndSkipEnv(env, skip=FRAME_SKIP)
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
        env = gym.wrappers.ResizeObservation(env, shape=OBSERVATION_SHAPE)

        if augment:
            env = ObservationAugmentation(env, seed=seed)
        env = gym.wrappers.TimeLimit(
            env,
            max_episode_steps=MAX_EPISODE_STEPS,
        )

        if record_video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=VIDEO_DIR,
                episode_trigger=lambda _: True,
                fps=VIDEO_RENDER_FPS,
                name_prefix=f"{MODEL_NAME}-{seed}-{uuid4().hex[:12]}",
            )

        env = ActionHistory(env)

        super().__init__(
            Monitor(
                env,
                filename=monitor_filename if monitor_filename else None,
            )
        )

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = self._initial_seed

        self._initial_seed = None
        return super().reset(seed=seed, options=options)
