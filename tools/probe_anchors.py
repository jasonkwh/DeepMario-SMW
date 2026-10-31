import argparse
from collections import Counter

import numpy as np
import stable_retro

from anchor import ANCHORS
from config import GAME_NAME, STATE_NAME
from setup import setup_game

WRAM_BASE = 0x7E0000


def read_raw(ram, spec):
    size = np.dtype(spec["type"]).itemsize
    start = spec["address"] - WRAM_BASE
    return int(np.frombuffer(bytes(ram[start : start + size]), dtype=spec["type"])[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    setup_game()
    env = stable_retro.make(
        game=GAME_NAME,
        state=STATE_NAME,
        use_restricted_actions=stable_retro.Actions.ALL,
        render_mode="rgb_array",
    )
    rng = np.random.default_rng(args.seed)
    env.reset(seed=args.seed)

    seen = {name: set() for name in ANCHORS}
    compared = Counter()
    mismatch = Counter()
    episodes = 1

    for _ in range(args.steps):
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        ram = env.unwrapped.data.memory.blocks[WRAM_BASE]
        for name, spec in ANCHORS.items():
            raw = read_raw(ram, spec)
            seen[name].add(raw)
            if name in info:
                compared[name] += 1
                if int(round(float(info[name]))) != raw:
                    mismatch[name] += 1

        if terminated or truncated:
            env.reset()
            episodes += 1

    print(f"{GAME_NAME} / {STATE_NAME}: {args.steps} steps, {episodes} episodes")
    print(f"{'anchor':16} {'addr':10} {'type':5} {'info':5} {'match':7} distinct")
    for name, spec in ANCHORS.items():
        values = sorted(seen[name])
        info_col = "yes" if compared[name] else "no"
        match_col = (
            f"{compared[name] - mismatch[name]}/{compared[name]}" if compared[name] else "-"
        )
        preview = ",".join(str(v) for v in values[:6])
        if len(values) > 6:
            preview += ",..."
        print(
            f"{name:16} {spec['address']:#08x} {spec['type']:5} "
            f"{info_col:5} {match_col:7} {len(values):3d}  {preview}"
        )

    print()
    failures = []
    for name in ANCHORS:
        if mismatch[name]:
            failures.append(f"{name}: {mismatch[name]} steps disagree with the engine")
        if len(seen[name]) == 1:
            failures.append(f"{name}: never changes, the address is probably wrong")

    print()
    print("note: data.json is written from this same table, so the info column only")
    print("      proves the type parses. A value that never changes is the address check.")
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        raise SystemExit(1)

    print("ok   every anchor parses and moves")


if __name__ == "__main__":
    main()
