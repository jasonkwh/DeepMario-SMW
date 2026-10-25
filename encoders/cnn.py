import gymnasium as gym
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

from config import OBSERVATION_SHAPE, FRAME_SKIP, FRAME_STACK
from wrappers import ActionHistory, ObservationAugmentation

from .encoder import Encoder

FEATURES_DIM = 512


class CnnEncoder(Encoder):
    def __init__(self, observation_space=None):
        super().__init__(observation_space)

    def build_policy(self) -> tuple[str, dict]:
        return (
            "CnnPolicy",
            {
                "features_extractor_class": NatureCNN,
                "features_extractor_kwargs": {"features_dim": FEATURES_DIM},
                "normalize_images": True,
            },
        )

    def wrap_env(self, env, *, augment=False, seed=0):
        env = MaxAndSkipEnv(env, skip=FRAME_SKIP)
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
        env = gym.wrappers.ResizeObservation(env, shape=OBSERVATION_SHAPE)
        env = gym.wrappers.ReshapeObservation(
            env, (OBSERVATION_SHAPE[0], OBSERVATION_SHAPE[1], 1)
        )
        if augment:
            env = ObservationAugmentation(env, seed=seed)
        return ActionHistory(env)

    def wrap_vec_env(self, vec_env):
        vec_env = VecFrameStack(vec_env, n_stack=FRAME_STACK, channels_order="last")
        return VecTransposeImage(vec_env)


def build_policy() -> tuple[str, dict]:
    return CnnEncoder().build_policy()
