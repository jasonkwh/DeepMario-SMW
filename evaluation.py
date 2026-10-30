import numpy as np

from config import EVAL_DETERMINISTIC, N_EVAL_EPISODES_FINAL

CLEARED_KEY = "termination/Completion"


def _scalar(value) -> float:
    array = np.asarray(value, dtype=float).ravel()
    return float(array[0]) if array.size else 0.0


def evaluate_level_metrics(
    model,
    env,
    n_eval_episodes: int,
    deterministic: bool = True,
) -> dict:
    num_envs = env.num_envs

    remaining = [n_eval_episodes // num_envs] * num_envs
    for index in range(n_eval_episodes % num_envs):
        remaining[index] += 1

    rewards = [0.0] * num_envs
    lengths = [0] * num_envs
    max_xs = [0.0] * num_envs
    cleared = [False] * num_envs

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_max_x: list[float] = []
    episode_cleared: list[float] = []

    observations = env.reset()

    while any(budget > 0 for budget in remaining):
        actions, _ = model.predict(observations, deterministic=deterministic)
        observations, step_rewards, dones, infos = env.step(actions)

        for index in range(num_envs):
            if remaining[index] == 0:
                continue

            info = infos[index] if isinstance(infos, (list, tuple)) else infos
            rewards[index] += _scalar(step_rewards[index])
            lengths[index] += 1
            max_xs[index] = max(
                max_xs[index],
                float(info.get("max_progress", 0.0)),
                float(info.get("x", 0.0)),
            )
            if float(info.get(CLEARED_KEY, 0.0)) > 0.0:
                cleared[index] = True

            if dones[index]:
                episode_rewards.append(rewards[index])
                episode_lengths.append(lengths[index])
                episode_max_x.append(max_xs[index])
                episode_cleared.append(1.0 if cleared[index] else 0.0)

                remaining[index] -= 1
                rewards[index] = 0.0
                lengths[index] = 0
                max_xs[index] = 0.0
                cleared[index] = False

    if not episode_rewards:
        return {
            "n_episodes": 0,
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "mean_max_x": 0.0,
            "best_max_x": 0.0,
            "clear_rate": 0.0,
            "cleared_episodes": 0,
            "mean_length": 0.0,
        }

    return {
        "n_episodes": len(episode_rewards),
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_max_x": float(np.mean(episode_max_x)),
        "best_max_x": float(np.max(episode_max_x)),
        "clear_rate": float(np.mean(episode_cleared)),
        "cleared_episodes": int(np.sum(episode_cleared)),
        "mean_length": float(np.mean(episode_lengths)),
    }


def final_evaluation(model, eval_env, logger, step: int):
    metrics = evaluate_level_metrics(
        model,
        eval_env,
        n_eval_episodes=N_EVAL_EPISODES_FINAL,
        deterministic=EVAL_DETERMINISTIC,
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

    if logger is None:
        print("No logger available: final metrics were not written to TensorBoard.")
    else:
        logger.record("final/mean_reward", metrics["mean_reward"])
        logger.record("final/mean_max_x", metrics["mean_max_x"])
        logger.record("final/clear_rate", metrics["clear_rate"])
        logger.dump(step)

    return metrics
