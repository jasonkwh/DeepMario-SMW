import argparse
from collections import Counter

import numpy as np
import stable_retro

from anchor import ANCHORS, WRAM_BASE
from config import GAME_NAME, STATE_NAME
from setup import setup_game

# Documented ranges, from the SMW disassembly RAM map.
EXPECTED_VALUES = {
    "player_in_air": {0, 11, 12, 36},
    "pmeter": set(range(0x71)),
    "input_blocked": set(range(14)),
    "direction": {0, 1},
}

# Anchors no published map covers yet: the tool cannot confirm them, only show
# whether they move and whether they look plausible.
UNVERIFIED = (
    "layer1_x",
    "carrying",
    "flash_timer",
    "level_end_timer",
    "star_timer",
)


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
    ground_ok = ground_total = 0
    sign_ok = sign_total = 0
    episodes = 1

    for _ in range(args.steps):
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        ram = env.unwrapped.data.memory.blocks[WRAM_BASE]
        values = {name: read_raw(ram, spec) for name, spec in ANCHORS.items()}
        for name, raw in values.items():
            seen[name].add(raw)
            if name in info:
                compared[name] += 1
                if int(round(float(info[name]))) != raw:
                    mismatch[name] += 1

        if values["player_in_air"] == 0:
            ground_total += 1
            ground_ok += bool(values["blocked"] & 0x04)
        if values["direction"] == 1:
            sign_total += 1
            sign_ok += values["x_speed"] >= 0

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
    print("checks that do not depend on data.json:")
    if ground_total:
        print(
            f"  on ground ({ground_total} steps) -> bottom collision bit set: "
            f"{ground_ok / ground_total:.1%}"
        )
    if sign_total:
        print(
            f"  facing right ({sign_total} steps) -> x_speed >= 0: "
            f"{sign_ok / sign_total:.1%}"
        )
    for name in UNVERIFIED:
        print(f"  {name:16} distinct values: {len(seen[name])}  (no public map)")

    failures = []
    for name in ANCHORS:
        if mismatch[name]:
            failures.append(f"{name}: {mismatch[name]} steps disagree with the engine")
        if name in EXPECTED_VALUES:
            unexpected = seen[name] - EXPECTED_VALUES[name]
            if unexpected:
                failures.append(f"{name}: unexpected values {sorted(unexpected)}")

    print()
    print("note: data.json is written from this same table, so the info column only")
    print("      proves the type parses, not that the address is right. The invariant")
    print("      ratios above are the address check that can fail.")
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        raise SystemExit(1)

    print("ok   every anchor parses and stays inside its documented range")


if __name__ == "__main__":
    main()
