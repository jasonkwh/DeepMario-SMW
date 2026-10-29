import gymnasium as gym
from .reward_types import RewardState
from .reward_rules import (
    CompletionRule,
    TimeUpPenaltyRule,
    LifeLostPenaltyRule,
    ScoreEventRule,
    ProgressRule,
    PowerUpRule,
    TimerPenaltyRule,
)
from .progress_estimators import create_progress_estimator


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, state_name: str):
        super().__init__(env)
        self.prev_state: RewardState | None = None
        self.rules = [
            # order matters
            # 1. termination rules
            CompletionRule(),
            TimeUpPenaltyRule(),
            LifeLostPenaltyRule(),
            # 2. reward shaping rules
            TimerPenaltyRule(),
            ProgressRule(create_progress_estimator(state_name)),
            ScoreEventRule(),
            PowerUpRule(),
        ]
        self.metric_defaults = {
            key: 0.0
            for rule in self.rules
            for key in (
                f"reward/rules/{rule.name}",
                f"termination/{rule.name}",
            )
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_state = RewardState.from_info(info) if "x" in info else None

        for rule in self.rules:
            rule.reset(self.prev_state)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = info.copy()

        cur_state = RewardState.from_info(info)
        metrics = self.metric_defaults.copy()
        info["reward/base"] = reward
        reward = 0.0
        info["timer"] = cur_state.timer

        if self.prev_state is None:
            for rule in self.rules:
                rule.reset(cur_state)
        else:
            for rule in self.rules:
                reward_delta, rule_terminated = rule.apply(self.prev_state, cur_state)
                metrics[f"reward/rules/{rule.name}"] = reward_delta
                reward += reward_delta

                if rule_terminated:
                    terminated = True
                    metrics[f"termination/{rule.name}"] = 1.0
                    break

        # get metrics from all rules, in case if terminated
        for rule in self.rules:
            metrics.update(rule.get_metrics())

        self.prev_state = cur_state
        info["reward/total"] = reward
        info.update(metrics)

        return obs, reward, terminated, truncated, info
