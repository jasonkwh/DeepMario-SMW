from stable_baselines3.common.torch_layers import NatureCNN

FEATURES_DIM = 512


class CnnEncoder(NatureCNN):
    def __init__(self, observation_space, features_dim: int = FEATURES_DIM):
        super().__init__(observation_space, features_dim=features_dim)
