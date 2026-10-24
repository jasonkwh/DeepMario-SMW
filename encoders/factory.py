from .cnn import CnnEncoder, FEATURES_DIM
from .types import Encoder

IMPLEMENTED = (Encoder.CNN,)


def build_policy(encoder: Encoder):
    if encoder not in IMPLEMENTED:
        raise RuntimeError(f"ENCODER={encoder.value!r} is not implemented yet")

    return (
        "CnnPolicy",
        {
            "features_extractor_class": CnnEncoder,
            "features_extractor_kwargs": {"features_dim": FEATURES_DIM},
        },
    )
