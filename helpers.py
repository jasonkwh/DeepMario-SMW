import os
from pathlib import Path

import torch

from config import (
    CHECKPOINT_DIR,
    MODEL_NAME,
    BEST_MODEL_SAVE_DIR,
    RESUME_FROM,
    TORCH_NUM_THREADS,
)


def get_n_envs() -> int:
    return max(1, len(os.sched_getaffinity(0)))


def configure_torch_threads() -> None:
    torch.set_num_threads(TORCH_NUM_THREADS)
    print(
        f"Torch limited to {TORCH_NUM_THREADS} thread(s); "
        f"env processes get the rest."
    )


def select_training_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training agent on {str(device).upper()}...")
    return device


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def get_checkpoint_path() -> Path | None:
    match RESUME_FROM:
        case "latest":
            return get_latest_checkpoint_path()
        case "best":
            return get_best_model_path()
        case "none":
            return None
        case _:
            raise ValueError(f"Unknown RESUME_FROM value: {RESUME_FROM}")


def get_vecnormalize_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.name == "best_model.zip":
        return checkpoint_path.with_name("best_model_vecnormalize.pkl")

    timestep = checkpoint_path.stem.rsplit("_", 2)[-2]
    return checkpoint_path.with_name(f"{MODEL_NAME}_vecnormalize_{timestep}_steps.pkl")


def remove_best_model_artifacts() -> None:
    best_model_dir = Path(BEST_MODEL_SAVE_DIR)
    for filename in ("best_model.zip", "best_model_vecnormalize.pkl"):
        (best_model_dir / filename).unlink(missing_ok=True)


def get_latest_checkpoint_path() -> Path | None:
    checkpoints = list(Path(CHECKPOINT_DIR).glob(f"{MODEL_NAME}_*_steps.zip"))

    return max(
        checkpoints,
        key=lambda path: int(path.stem.rsplit("_", 2)[-2]),
        default=None,
    )


def get_best_model_path() -> Path | None:
    path = Path(BEST_MODEL_SAVE_DIR) / "best_model.zip"
    return path if path.exists() else None
