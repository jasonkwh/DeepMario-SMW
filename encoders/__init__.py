from typing import Any

from .encoder import Encoder, EncoderType

__all__ = [
    "CNNEncoder",
    "Encoder",
    "EncoderType",
    "RAMEncoder",
    "build_encoder",
    "build_policy",
]


def __getattr__(name: str) -> Any:
    if name == "CNNEncoder":
        from .cnn import CNNEncoder

        return CNNEncoder
    if name == "RAMEncoder":
        from .ram import RAMEncoder

        return RAMEncoder
    if name == "build_encoder":
        from .factory import build_encoder

        return build_encoder
    if name == "build_policy":
        from .factory import build_policy

        return build_policy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
