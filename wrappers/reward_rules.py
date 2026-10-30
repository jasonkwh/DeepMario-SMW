from abc import ABC, abstractmethod
from .reward_types import RewardState
from .progress_estimators import ProgressEstimator, XProgressEstimator


class RewardRule(ABC):
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
    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        if prev_state.is_level_completed(cur_state):
            return 100.0, True
        return 0.0, False


class TimeUpPenaltyRule(RewardRule):
    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        if prev_state.is_times_up(cur_state):
            return -10.0, True
        return 0.0, False


class LifeLostPenaltyRule(RewardRule):
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

        if prev_state.is_lost_life(cur_state):
            self._metrics = {
                "life_lost/x": float(cur_state.x),
                "life_lost/lives": float(cur_state.lives),
            }

            if prev_state.is_dead(cur_state):
                return -20.0, True
            return -5.0, False
        return 0.0, False

    def get_metrics(self) -> dict[str, float]:
        return self._metrics


class ScoreEventRule(RewardRule):
    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        if prev_state.has_increased_score(cur_state):
            if prev_state.has_collected_coin(cur_state):
                return 1.0, False
            return 0.5, False
        return 0.0, False


class PowerUpRule(RewardRule):
    POWERUP_REWARDS = {
        1: 0.1,  # Big
        2: 0.2,  # Cape
        3: 0.2,  # Fire
    }

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
                reward_delta -= 0.1

        # star power
        if prev_state.gains_star(cur_state):
            reward_delta += 0.3

        return reward_delta, False


class TimerPenaltyRule(RewardRule):
    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        if cur_state.timer < 100:
            return -0.0005, False
        return -0.00025, False


class ProgressRule(RewardRule):
    def __init__(
        self,
        estimator: ProgressEstimator,
    ):
        self.estimator = estimator
        self.max_progress = 0.0

    def reset(self, prev_state: RewardState | None = None) -> None:
        self.max_progress = self.estimator.estimate(prev_state)

    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        cur_progress = self.estimator.estimate(cur_state)
        progress_delta = max(0.0, cur_progress - self.max_progress)

        if progress_delta > 0.0:
            self.max_progress = cur_progress
            return 0.1 * progress_delta, False

        return 0.0, False

    def get_metrics(self) -> dict[str, float]:
        return {
            "max_progress": float(self.max_progress),
        }
