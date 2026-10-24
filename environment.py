import gymnasium as gym
import stable_retro
from wrappers import ActionWrapper, RewardWrapper
from stable_baselines3.common.monitor import Monitor

from encoders import build_encoder
from config import (
    MAX_EPISODE_STEPS,
    ENCODER,
)


class MarioEnvironment(gym.Wrapper):
    def __init__(
        self,
        game_name,
        state_name,
        monitor_filename=None,
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
        env = build_encoder(ENCODER).wrap_env(env, augment=augment, seed=seed)
        env = gym.wrappers.TimeLimit(
            env,
            max_episode_steps=MAX_EPISODE_STEPS,
        )

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
