from pathlib import Path
import stable_retro
import json
from config import GAME_NAME
from anchor import ANCHORS


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
            name: {"address": spec["address"], "type": spec["type"]}
            for name, spec in ANCHORS.items()
        }
    )

    data_path.write_text(json.dumps(data, indent=2) + "\n")

    print(f"Updated {data_path}")


if __name__ == "__main__":
    setup_game()
