import gymnasium as gym
import numpy as np

from anchor import ANCHORS, WRAM_BASE, WRAM_SIZE

OBSERVATION_LOW = -1.0
OBSERVATION_HIGH = 16.0
TIMER_PARTS = (("hundreds", 100), ("tens", 10), ("ones", 1))


def _raw(ram, anchor, offsets):
    spec = ANCHORS[anchor]
    size = np.dtype(spec["type"]).itemsize
    start = offsets[anchor]
    return float(np.frombuffer(bytes(ram[start : start + size]), dtype=spec["type"])[0])


def _compile():
    """Flatten the obs specs into one entry per observation dimension."""
    features = []
    for name, spec in ANCHORS.items():
        obs = spec["obs"]
        if obs is None:
            continue
        if isinstance(obs, dict):
            features.extend(
                (f"{name}_{part}", name, "bit", mask) for part, mask in obs.items()
            )
        elif isinstance(obs, tuple):
            features.append(("timer", name, "timer", obs[1]))
        elif obs == "flag":
            features.append((name, name, "flag", None))
        else:
            features.append((name, name, "scale", obs))
    return tuple(features)


FEATURES = _compile()
FEATURE_NAMES = tuple(feature[0] for feature in FEATURES)
OBSERVATION_DIM = len(FEATURES)


def build(ram, offsets):
    values = []
    for _, anchor, kind, argument in FEATURES:
        if kind == "timer":
            total = sum(
                _raw(ram, f"timer_{part}", offsets) * weight
                for part, weight in TIMER_PARTS
            )
            values.append(total / argument)
            continue
        raw = _raw(ram, anchor, offsets)
        if kind == "bit":
            values.append(float(bool(int(raw) & argument)))
        elif kind == "flag":
            values.append(float(raw != 0.0))
        else:
            values.append(raw / argument)
    return np.array(values, dtype=np.float32)


class RAMObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        ram = env.unwrapped.data.memory.blocks[WRAM_BASE]
        assert len(ram) == WRAM_SIZE, f"unexpected WRAM size {len(ram)}"
        self._offset = {
            name: spec["address"] - WRAM_BASE for name, spec in ANCHORS.items()
        }
        self.observation_space = gym.spaces.Box(
            low=OBSERVATION_LOW,
            high=OBSERVATION_HIGH,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )

    def observation(self, obs):
        ram = self.env.unwrapped.data.memory.blocks[WRAM_BASE]
        vec = build(ram, self._offset)
        return np.clip(vec, OBSERVATION_LOW, OBSERVATION_HIGH, out=vec)
