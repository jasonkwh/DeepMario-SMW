from . import cnn
from .encoder import EncoderType


def build_policy(encoder_type: EncoderType):
    match encoder_type:
        case EncoderType.CNN:
            return cnn.build_policy()
        case _:
            raise RuntimeError(f"ENCODER={encoder_type.value!r} is not implemented yet")
