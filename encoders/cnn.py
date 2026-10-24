from stable_baselines3.common.torch_layers import NatureCNN

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


def build_policy() -> tuple[str, dict]:
    return CnnEncoder().build_policy()
