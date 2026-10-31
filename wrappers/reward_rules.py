from abc import ABC, abstractmethod
from .reward_types import RewardState
from .progress_estimators import ProgressEstimator, XProgressEstimator


class RewardRule(ABC):
    metric_keys: tuple[str, ...] = ()

    def reset(self, prev_state: RewardState | None = None) -> None:
        """Reset the rule state."""
        pass

    @abstractmethod
    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        """Return (reward_delta, terminated)."""

    @property
    def name(self) -> str:
        return type(self).__name__.removesuffix("Rule")

    def get_metrics(self) -> dict[str, float]:
        return {}


class CompletionRule(RewardRule):
    COMPLETION_REWARD = 100.0

    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        if prev_state.is_level_completed(cur_state):
            return self.COMPLETION_REWARD, True
        return 0.0, False


class TimeUpPenaltyRule(RewardRule):
    TIME_UP_PENALTY = -10.0

    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        if prev_state.is_times_up(cur_state):
            return self.TIME_UP_PENALTY, True
        return 0.0, False


class LifeLostPenaltyRule(RewardRule):
    LIFE_PENALTY = -5.0
    LAST_LIFE_PENALTY = -20.0

    def __init__(self):
        self._metrics: dict[str, float] = {}

    def reset(self, prev_state: RewardState | None = None) -> None:
        self._metrics = {}

    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        self._metrics = {}

        if not prev_state.is_lost_life(cur_state):
            return 0.0, False

        self._metrics = {
            "life_lost/x": float(cur_state.x),
            "life_lost/lives": float(cur_state.lives),
        }

        if prev_state.is_dead(cur_state):
            return self.LAST_LIFE_PENALTY, True
        return self.LIFE_PENALTY, False

    def get_metrics(self) -> dict[str, float]:
        return self._metrics


class ScoreEventRule(RewardRule):
    COIN_REWARD = 10.0
    OTHER_REWARD = 5.0
    OTHER_CAP = 50.0
    metric_keys = ("reward/score_event/coin", "reward/score_event/other")

    def __init__(self):
        self._other_paid = 0.0
        self._metrics: dict[str, float] = {}

    def reset(self, prev_state: RewardState | None = None) -> None:
        self._other_paid = 0.0
        self._metrics = {}

    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        self._metrics = {}

        if not prev_state.has_increased_score(cur_state):
            return 0.0, False

        if prev_state.has_collected_coin(cur_state):
            self._metrics = {"reward/score_event/coin": self.COIN_REWARD}
            return self.COIN_REWARD, False

        reward_delta = max(
            0.0, min(self.OTHER_REWARD, self.OTHER_CAP - self._other_paid)
        )
        if reward_delta > 0.0:
            self._other_paid += reward_delta
        self._metrics = {"reward/score_event/other": reward_delta}

        return reward_delta, False

    def get_metrics(self) -> dict[str, float]:
        return self._metrics


class PowerUpRule(RewardRule):
    POWERUP_REWARDS = {
        1: 0.1,  # Big
        2: 0.2,  # Cape
        3: 0.2,  # Fire
    }
    POWERUP_LOST_PENALTY = -0.1
    STAR_REWARD = 0.3

    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        reward_delta = 0.0

        # traditional power ups
        if prev_state.gains_powerup(cur_state):
            reward_delta += self.POWERUP_REWARDS[cur_state.powerup]
        if prev_state.loses_powerup(cur_state):
            if not prev_state.is_lost_life(cur_state):
                reward_delta += self.POWERUP_LOST_PENALTY

        # star power
        if prev_state.gains_star(cur_state):
            reward_delta += self.STAR_REWARD

        return reward_delta, False


class TimerPenaltyRule(RewardRule):
    TIMER_LOW_THRESHOLD = 100
    TIMER_PENALTY_LOW = -0.0005
    TIMER_PENALTY_NORMAL = -0.00025

    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        if cur_state.timer < self.TIMER_LOW_THRESHOLD:
            return self.TIMER_PENALTY_LOW, False
        return self.TIMER_PENALTY_NORMAL, False


class ProgressRule(RewardRule):
    PROGRESS_REWARD = 0.1

    def __init__(
        self,
        estimator: ProgressEstimator,
    ):
        self.estimator = estimator
        self.max_progress = 0.0
        self._metrics: dict[str, float] = {}

    def reset(self, prev_state: RewardState | None = None) -> None:
        self.max_progress = self.estimator.estimate(prev_state)
        self._metrics = {"max_progress": float(self.max_progress)}

    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        cur_progress = self.estimator.estimate(cur_state)
        progress_delta = max(0.0, cur_progress - self.max_progress)

        reward_delta = 0.0
        if progress_delta > 0.0:
            self.max_progress = cur_progress
            reward_delta = self.PROGRESS_REWARD * progress_delta

        self._metrics = {"max_progress": float(self.max_progress)}

        return reward_delta, False

    def get_metrics(self) -> dict[str, float]:
        return self._metrics
