import sys
import subprocess
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
)
from encoders import build_policy
from evaluation import evaluate_level_metrics
from helpers import (
    build_vec_env,
    configure_torch_threads,
    select_training_device,
    get_best_model_path,
    get_checkpoint_path,
    get_vecnormalize_path,
    linear_schedule,
    remove_best_model_artifacts,
)
from callbacks import (
    LevelMetricsEvalCallback,
    TensorboardLogExtensionCallback,
    RecordVideoCallback,
    SaveVecNormalizeCallback,
)

from config import (
    ENCODER,
    MODEL_NAME,
    N_ENVS,
    TB_LOG_NAME,
    BEST_MODEL_SAVE_DIR,
    TENSORBOARD_LOG_DIR,
    CHECKPOINT_DIR,
    MONITOR_FILENAME,
    MONITOR_EVALUATION_FILENAME,
    LEARNING_RATE,
    N_STEPS,
    BATCH_SIZE,
    ENT_COEF,
    TOTAL_TIMESTEPS,
    EVAL_EVERY,
    EVAL_N_ENVS,
    N_EVAL_EPISODES,
    CHECKPOINT_EVERY,
    SEED,
    N_EVAL_EPISODES_FINAL,
    TARGET_KL,
    N_EPOCHS,
    GAMMA,
    GAE_LAMBDA,
    CLIP_RANGE,
    VF_COEF,
    MAX_GRAD_NORM,
    CLIP_REWARD,
    VIDEO_EVERY_N_BEST,
)


def final_evaluation(model, eval_env, step: int):
    metrics = evaluate_level_metrics(
        model,
        eval_env,
        n_eval_episodes=N_EVAL_EPISODES_FINAL,
        deterministic=True,
    )
    print(
        f"Final evaluation: reward {metrics['mean_reward']:.2f} "
        f"+/- {metrics['std_reward']:.2f} over {metrics['n_episodes']} episodes"
    )
    print(
        f"  max_x   mean {metrics['mean_max_x']:.1f} | best {metrics['best_max_x']:.1f}"
    )
    print(
        f"  cleared {metrics['clear_rate'] * 100:.0f}% "
        f"({metrics['cleared_episodes']}/{metrics['n_episodes']}) "
        f"| mean episode length {metrics['mean_length']:.0f} steps"
    )

    logger = getattr(model, "logger", None)
    if logger is not None:
        try:
            logger.record("final/mean_reward", metrics["mean_reward"])
            logger.record("final/mean_max_x", metrics["mean_max_x"])
            logger.record("final/clear_rate", metrics["clear_rate"])
            logger.dump(step)
        except Exception:
            pass

    return metrics


def get_model_to_save(model, device):
    best_model_path = get_best_model_path()

    if best_model_path is not None:
        best_model = PPO.load(best_model_path, device=device)
        print("Using the best model selected during training for final evaluation.")
        return best_model
    else:
        print(
            "No evaluated best model was created. Using the last trained model for final evaluation."
        )
        return model


def main():
    env = None
    eval_env = None

    configure_torch_threads()
    print(f"Tensorboard run name: '{TB_LOG_NAME}'")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "stable_retro.import", "."],
            capture_output=True,
            text=True,
        )

        if result.stderr.strip():
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(result.stdout, file=sys.stderr)
            raise RuntimeError(
                "ROM import failed with exit code "
                f"{result.returncode}. Put the game ROM (a .sfc file, which .gitignore "
                "excludes) in the repository root and run again."
            )

        print(result.stdout)

        env = build_vec_env(
            N_ENVS,
            augment=True,
            monitor_filename=MONITOR_FILENAME,
        )

        checkpoint_path = get_checkpoint_path()
        if checkpoint_path is None:
            remove_best_model_artifacts()
        vecnormalize_path = (
            get_vecnormalize_path(checkpoint_path)
            if checkpoint_path is not None
            else None
        )

        if vecnormalize_path is not None and vecnormalize_path.exists():
            env = VecNormalize.load(vecnormalize_path, env)
            env.training = True
            env.norm_reward = True
            print(f"Loaded VecNormalize statistics from '{vecnormalize_path}'.")
        else:
            if vecnormalize_path is not None:
                print(
                    f"VecNormalize statistics not found at '{vecnormalize_path}'. "
                    "Starting with fresh reward normalization."
                )
            env = VecNormalize(
                env,
                norm_obs=False,
                norm_reward=True,
                clip_reward=CLIP_REWARD,
                gamma=GAMMA,
            )

        eval_env = build_vec_env(
            EVAL_N_ENVS,
            monitor_filename=MONITOR_EVALUATION_FILENAME,
            seed_offset=10_000,
        )
        eval_env = VecNormalize(
            eval_env,
            norm_obs=False,
            norm_reward=False,
            training=False,
            gamma=GAMMA,
        )

        device = select_training_device()

        if checkpoint_path is not None:
            model = PPO.load(
                checkpoint_path,
                env=env,
                device=device,
                tensorboard_log=TENSORBOARD_LOG_DIR,
            )
            reset_num_timesteps = False
        else:
            policy_name, policy_kwargs = build_policy(ENCODER)
            model = PPO(
                policy=policy_name,
                env=env,
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=1,
                learning_rate=linear_schedule(LEARNING_RATE),
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                ent_coef=ENT_COEF,
                target_kl=TARGET_KL,
                n_epochs=N_EPOCHS,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
                clip_range=CLIP_RANGE,
                vf_coef=VF_COEF,
                max_grad_norm=MAX_GRAD_NORM,
                tensorboard_log=TENSORBOARD_LOG_DIR,
                seed=SEED,
            )
            reset_num_timesteps = True

        model.logger.record("train/n_envs", N_ENVS)

        if model.num_timesteps < TOTAL_TIMESTEPS:
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS - model.num_timesteps,
                callback=[
                    TensorboardLogExtensionCallback(),
                    CheckpointCallback(
                        save_freq=max(CHECKPOINT_EVERY // N_ENVS, 1),
                        save_path=CHECKPOINT_DIR,
                        name_prefix=MODEL_NAME,
                        verbose=2,
                        save_vecnormalize=True,
                    ),
                    LevelMetricsEvalCallback(
                        eval_env,
                        best_model_save_path=BEST_MODEL_SAVE_DIR,
                        log_path=BEST_MODEL_SAVE_DIR,
                        eval_freq=max(EVAL_EVERY // N_ENVS, 1),
                        n_eval_episodes=N_EVAL_EPISODES,
                        deterministic=True,
                        verbose=2,
                        render=False,
                        callback_on_new_best=CallbackList(
                            [
                                RecordVideoCallback(VIDEO_EVERY_N_BEST),
                                SaveVecNormalizeCallback(),
                            ]
                        ),
                    ),
                ],
                reset_num_timesteps=reset_num_timesteps,
                tb_log_name=TB_LOG_NAME,
            )
        else:
            print(
                f"Model has already reached {TOTAL_TIMESTEPS} timesteps. Skipping training."
            )

        # Evaluate and save the same policy: the best checkpoint when available.
        final_model = get_model_to_save(model, device)
        final_evaluation(final_model, eval_env, step=model.num_timesteps)

        # save the final model
        final_model.save(MODEL_NAME)
        print(f"Final evaluated model saved as '{MODEL_NAME}.zip'")
    finally:
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()
