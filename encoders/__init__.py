from .types import Encoder

__all__ = [
    "CnnEncoder",
    "Encoder",
    "build_policy",
]


def __getattr__(name):
    if name in __all__:
        from . import factory

        return getattr(factory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
