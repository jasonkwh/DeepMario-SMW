from stable_baselines3.common.callbacks import BaseCallback


class TensorboardLogExtensionCallback(BaseCallback):
    def _on_training_start(self) -> None:
        self.logger.record("train/n_envs", self.training_env.num_envs)

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

                elif key.startswith(("reward/", "termination/", "life_lost/")):
                    self.logger.record(f"env_{env_index}/{key}", float(value))
                    self.logger.record_mean(key, float(value))

        return True
