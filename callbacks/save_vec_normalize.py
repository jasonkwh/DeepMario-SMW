from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from config import BEST_MODEL_SAVE_DIR


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(
        self,
        save_dir: Path | str = Path(BEST_MODEL_SAVE_DIR),
        name_prefix: str = "best_model",
    ):
        super().__init__()
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = save_dir / f"{name_prefix}_vecnormalize.pkl"

    def _on_step(self) -> bool:
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is None:
            return True

        vec_normalize.save(self.save_path)
        return True
