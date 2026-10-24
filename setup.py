from pathlib import Path
import stable_retro
import json
from config import GAME_NAME


def setup_game():
    retro_dir = Path(stable_retro.__file__).parent
    matches = list(retro_dir.rglob(f"{GAME_NAME}/data.json"))

    if not matches:
        raise RuntimeError(
            f"{GAME_NAME} is not integrated into stable_retro yet, so there is no "
            f"data.json to patch under {retro_dir}. Put the game ROM (a .sfc file) "
            "in the repository root and run `python -m stable_retro.import .` first."
        )

    data_path = matches[0]

    print(f"Updating {data_path}")

    data = json.loads(data_path.read_text())

    data["info"].update(
        {
            "x": {
                "address": 8257684,
                "type": "<u2",
            },
            "y": {
                "address": 8257686,
                "type": "<u2",
            },
            # powerup
            # valid values:
            # 00 small, 01 big, 02 cape, 03 fire flower
            "powerup": {
                "address": 8257561,
                "type": "<u1",
            },
            # item box
            # valid values:
            # 00 none, 01 mushroom, 02 fire flower, 03 star, 04 feather
            "item": {
                "address": 8261058,
                "type": "<u1",
            },
            # timer
            "timer_hundreds": {
                "address": 8261425,
                "type": "<u1",
            },
            "timer_tens": {
                "address": 8261426,
                "type": "<u1",
            },
            "timer_ones": {
                "address": 8261427,
                "type": "<u1",
            },
            # completion
            "level_end_timer": {
                "address": 8262803,
                "type": "<u1",
            },
            # star timer
            "star_timer": {
                "address": 8262800,
                "type": "<u1",
            },
        }
    )

    data_path.write_text(json.dumps(data, indent=2) + "\n")

    print(f"Updated {data_path}")


if __name__ == "__main__":
    setup_game()
