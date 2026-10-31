import gymnasium as gym
import numpy as np

from anchor import ANCHORS, WRAM_BASE, WRAM_SIZE

OBSERVATION_LOW = -1.0
OBSERVATION_HIGH = 16.0
TIMER_PARTS = (("hundreds", 100), ("tens", 10), ("ones", 1))


def _raw(ram, name, spec, offsets):
    size = np.dtype(spec["type"]).itemsize
    start = offsets[name]
    return float(np.frombuffer(bytes(ram[start : start + size]), dtype=spec["type"])[0])


def _decode(ram, name, spec, offsets):
    obs = spec["obs"]
    if obs is None:
        return ()
    if isinstance(obs, tuple):
        kind, scale = obs
        return (
            sum(
                _raw(ram, f"timer_{part}", ANCHORS[f"timer_{part}"], offsets) * weight
                for part, weight in TIMER_PARTS
            )
            / scale,
        )
    raw = _raw(ram, name, spec, offsets)
    if obs == "flag":
        return (float(raw != 0.0),)
    if isinstance(obs, dict):
        return tuple(float(bool(int(raw) & mask)) for mask in obs.values())
    return (raw / obs,)


def _feature_names():
    names = []
    for name, spec in ANCHORS.items():
        if spec["obs"] is None:
            continue
        obs = spec["obs"]
        if isinstance(obs, dict):
            names.extend(f"{name}_{part}" for part in obs)
        elif isinstance(obs, tuple):
            names.append(obs[0])
        else:
            names.append(name)
    return tuple(names)


FEATURE_NAMES = _feature_names()
OBSERVATION_DIM = len(FEATURE_NAMES)


def build(ram, offsets):
    values = []
    for name, spec in ANCHORS.items():
        values.extend(_decode(ram, name, spec, offsets))
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
