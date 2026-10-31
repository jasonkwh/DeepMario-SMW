import gymnasium as gym
import numpy as np

from .encoder import Encoder
from anchor import ANCHORS, WRAM_BASE, WRAM_SIZE
from config import FRAME_SKIP

WORD_SCALE = 4096.0
SPEED_SCALE = 128.0
BYTE_SCALE = 255.0
TIMER_SCALE = 400.0
OBSERVATION_DIM = 16


def _word(ram, offset):
    return int(ram[offset]) | (int(ram[offset + 1]) << 8)


def _signed(ram, offset):
    value = int(ram[offset])
    return value - 256 if value > 127 else value


class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class RAMObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        ram = env.unwrapped.data.memory.blocks[WRAM_BASE]
        assert len(ram) == WRAM_SIZE, f"unexpected WRAM size {len(ram)}"
        self._offset = {
            name: spec["address"] - WRAM_BASE for name, spec in ANCHORS.items()
        }
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=16.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )

    def observation(self, obs):
        ram = self.env.unwrapped.data.memory.blocks[WRAM_BASE]
        offset = self._offset
        timer = (
            int(ram[offset["timer_hundreds"]]) * 100
            + int(ram[offset["timer_tens"]]) * 10
            + int(ram[offset["timer_ones"]])
        )
        return np.array(
            [
                _word(ram, offset["x"]) / WORD_SCALE,
                _word(ram, offset["y"]) / WORD_SCALE,
                _word(ram, offset["layer1_x"]) / WORD_SCALE,
                _signed(ram, offset["x_speed"]) / SPEED_SCALE,
                _signed(ram, offset["y_speed"]) / SPEED_SCALE,
                float(ram[offset["direction"]]),
                ram[offset["blocked"]] / BYTE_SCALE,
                float(ram[offset["in_water"]]),
                float(ram[offset["climbing"]]),
                float(ram[offset["carrying"]]),
                ram[offset["flash_timer"]] / BYTE_SCALE,
                ram[offset["powerup"]] / 3.0,
                ram[offset["item"]] / 4.0,
                timer / TIMER_SCALE,
                ram[offset["level_end_timer"]] / BYTE_SCALE,
                ram[offset["star_timer"]] / BYTE_SCALE,
            ],
            dtype=np.float32,
        )


class RAMEncoder(Encoder):
    def __init__(self, observation_space=None):
        super().__init__(observation_space)

    def build_policy(self) -> tuple[str, dict]:
        return (
            "MlpPolicy",
            {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        )

    def wrap_env(self, env, *, augment=False, seed=0):
        env = SkipEnv(env, skip=FRAME_SKIP)
        return RAMObservation(env)

    def wrap_vec_env(self, vec_env):
        return vec_env


def build_policy() -> tuple[str, dict]:
    return RAMEncoder().build_policy()
