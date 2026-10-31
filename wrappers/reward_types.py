from dataclasses import dataclass


@dataclass(frozen=True)
class RewardState:
    lives: int
    coins: int
    score: int
    x: int
    y: int
    level_end_timer: int
    timer: int
    powerup: int
    star_timer: int

    @classmethod
    def from_info(cls, info: dict) -> "RewardState":
        return cls(
            lives=info.get("lives", 0),
            coins=info.get("coins", 0),
            score=info.get("score", 0),
            x=info.get("x", 0),
            y=info.get("y", 0),
            level_end_timer=info.get("level_end_timer", 0),
            timer=cls._get_current_time(info),
            powerup=info.get("powerup", 0),
            star_timer=info.get("star_timer", 0),
        )

    @staticmethod
    def _get_current_time(info: dict) -> int:
        hundreds = info.get("timer_hundreds", 0)
        tens = info.get("timer_tens", 0)
        ones = info.get("timer_ones", 0)
        return hundreds * 100 + tens * 10 + ones

    def is_times_up(self, current: "RewardState") -> bool:
        return self.timer > 0 and current.timer == 0

    def is_level_completed(self, current: "RewardState") -> bool:
        return self.level_end_timer == 0 and current.level_end_timer > 0

    def is_lost_life(self, current: "RewardState") -> bool:
        return current.lives < self.lives

    def is_dead(self, current: "RewardState") -> bool:
        return self.lives == 0 and current.lives == -1

    def has_collected_coin(self, current: "RewardState") -> bool:
        return current.coins > self.coins

    def has_increased_score(self, current: "RewardState") -> bool:
        return current.score > self.score

    def gains_powerup(self, current: "RewardState") -> bool:
        return current.powerup != 0 and self.powerup == 0

    def loses_powerup(self, current: "RewardState") -> bool:
        return current.powerup == 0 and self.powerup != 0

    def gains_star(self, current: "RewardState") -> bool:
        return current.star_timer > 0 and self.star_timer == 0
