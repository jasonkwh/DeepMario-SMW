from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from environment import MarioEnvironment
from config import (
    BEST_MODEL_SAVE_DIR,
    FRAME_STACK,
    GAME_NAME,
    STATE_NAME,
    SEED,
)


class TensorboardLogExtensionCallback(BaseCallback):
    def _on_step(self) -> bool:
        for env_index, info in enumerate(self.locals.get("infos", [])):
            for key, value in info.items():
                if key in ("powerup", "item"):
                    self.logger.record(
                        f"env_{env_index}/state/{key}",
                        int(value),
                    )

                elif key in ("x", "y", "timer", "max_progress"):
                    self.logger.record(f"env_{env_index}/state/{key}", float(value))
                    self.logger.record_mean(f"state/{key}", float(value))

                elif key.startswith("reward/") or key.startswith("termination/"):
                    self.logger.record(f"env_{env_index}/{key}", float(value))
                    self.logger.record_mean(key, float(value))

        return True


class LevelMetricsEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eval_max_x: list[float] = []
        self._eval_cleared: list[float] = []
        self._current_max_x = 0.0

    def _log_success_callback(self, locals_, globals_) -> None:
        super()._log_success_callback(locals_, globals_)

        info = locals_.get("info") or {}
        if not isinstance(info, dict):
            return

        self._current_max_x = max(
            self._current_max_x,
            float(info.get("max_progress", 0.0)),
            float(info.get("x", 0.0)),
        )

        if locals_.get("done"):
            cleared = float(info.get("termination/Completion", 0.0)) > 0.0
            self._eval_max_x.append(self._current_max_x)
            self._eval_cleared.append(1.0 if cleared else 0.0)
            self._current_max_x = 0.0

    def _on_step(self) -> bool:
        is_eval_step = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0

        if is_eval_step:
            self._eval_max_x = []
            self._eval_cleared = []
            self._current_max_x = 0.0

        continue_training = super()._on_step()

        if is_eval_step:
            if not self._eval_max_x:
                print(
                    "LevelMetricsEvalCallback: no evaluation episodes were recorded. "
                    "Check that EvalCallback still passes _log_success_callback to evaluate_policy."
                )
                return continue_training

            mean_max_x = float(np.mean(self._eval_max_x))
            clear_rate = float(np.mean(self._eval_cleared))

            self.logger.record("eval/mean_max_x", mean_max_x)
            self.logger.record("eval/clear_rate", clear_rate)

            if self.verbose >= 1:
                print(
                    f"Eval level metrics: mean_max_x={mean_max_x:.1f} "
                    f"clear_rate={clear_rate:.2f} "
                    f"({int(sum(self._eval_cleared))}/{len(self._eval_cleared)})"
                )

        return continue_training


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


class RecordVideoCallback(BaseCallback):
    def _on_step(self) -> bool:
        video_env = SubprocVecEnv(
            [
                lambda: MarioEnvironment(
                    GAME_NAME,
                    STATE_NAME,
                    monitor_filename=None,
                    record_video=True,
                    seed=SEED + 20_000,
                )
            ],
            start_method="spawn",
        )
        video_env = VecFrameStack(
            video_env,
            n_stack=FRAME_STACK,
            channels_order="last",
        )
        video_env = VecTransposeImage(video_env)

        try:
            obs = video_env.reset()
            done = [False]
            while not done[0]:
                action, _ = self.model.predict(
                    obs,
                    deterministic=True,
                )
                obs, _, done, _ = video_env.step(action)
        finally:
            video_env.close()

        return True
