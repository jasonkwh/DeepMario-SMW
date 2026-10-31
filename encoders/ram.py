from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from .encoder import Encoder
from .ram_observation import RAMObservation
from config import FRAME_SKIP


class RAMEncoder(Encoder):
    def __init__(self, observation_space=None):
        super().__init__(observation_space)

    def build_policy(self) -> tuple[str, dict]:
        return (
            "MlpPolicy",
            {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        )

    def wrap_env(self, env, *, augment=False, seed=0):
        env = MaxAndSkipEnv(env, skip=FRAME_SKIP)
        return RAMObservation(env)

    def wrap_vec_env(self, vec_env):
        return vec_env


def build_policy() -> tuple[str, dict]:
    return RAMEncoder().build_policy()
