from . import cnn
from .encoder import Encoder, EncoderType


def build_encoder(encoder_type: EncoderType) -> Encoder:
    match encoder_type:
        case EncoderType.CNN:
            return cnn.CnnEncoder()
        case _:
            raise RuntimeError(f"ENCODER={encoder_type.value!r} is not implemented yet")


def build_policy(encoder_type: EncoderType):
    return build_encoder(encoder_type).build_policy()
