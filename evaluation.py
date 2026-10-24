import numpy as np

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
    """Run evaluation episodes and report level progress, not just reward.

    Rewards are shaped (progress, coins, timer), so mean reward alone cannot
    tell whether a policy actually gets further into the level or finishes it.
    This measures the furthest x reached per episode and the share of episodes
    that end with the level cleared.
    """
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_max_x: list[float] = []
    episode_cleared: list[float] = []

    for _ in range(n_eval_episodes):
        reset_result = env.reset()
        observations = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        done = False
        total_reward = 0.0
        length = 0
        max_x = 0.0
        cleared = False

        while not done:
            actions, _ = model.predict(observations, deterministic=deterministic)
            observations, rewards, dones, infos = env.step(actions)

            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            total_reward += _scalar(rewards)
            length += 1
            max_x = max(
                max_x,
                float(info.get("max_progress", 0.0)),
                float(info.get("x", 0.0)),
            )
            if float(info.get(CLEARED_KEY, 0.0)) > 0.0:
                cleared = True
            done = bool(_scalar(dones))

        episode_rewards.append(total_reward)
        episode_lengths.append(length)
        episode_max_x.append(max_x)
        episode_cleared.append(1.0 if cleared else 0.0)

    return {
        "n_episodes": n_eval_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_max_x": float(np.mean(episode_max_x)),
        "best_max_x": float(np.max(episode_max_x)),
        "clear_rate": float(np.mean(episode_cleared)),
        "cleared_episodes": int(np.sum(episode_cleared)),
        "mean_length": float(np.mean(episode_lengths)),
    }
