from .encoder import Encoder, EncoderType

__all__ = [
    "CnnEncoder",
    "Encoder",
    "EncoderType",
    "build_policy",
]


def __getattr__(name):
    if name == "CnnEncoder":
        from .cnn import CnnEncoder

        return CnnEncoder
    if name == "build_policy":
        from .factory import build_policy

        return build_policy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
