from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gymnasium as gym


class EncoderType(str, Enum):
    CNN = "cnn"
    RAM = "ram"
    CNN_YOLO = "cnn-yolo"


class Encoder(ABC):
    def __init__(self, observation_space: gym.Space | None = None):
        self.observation_space = observation_space

    @abstractmethod
    def build_policy(self) -> tuple[str, dict]:
        pass
