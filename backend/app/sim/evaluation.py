"""環境と方策を変えても同じ定義で計測する評価指標。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from app.contracts import StepResult


@dataclass
class EvaluationMetrics:
    """完了エピソードの率と、停止中を含む車両ステップ平均速度。"""

    reasons: Counter = field(default_factory=Counter)
    completed: int = 0
    lane_episodes: int = 0
    signal_episodes: int = 0
    vehicle_steps: int = 0
    speed_sum: float = 0.0

    def observe(self, result: StepResult, learners: np.ndarray) -> None:
        active = result.active & learners
        self.vehicle_steps += int(active.sum())
        if result.drive is not None:
            speed = result.drive.ground_speed if result.drive.ground_speed is not None else result.drive.speed
            self.speed_sum += float(np.abs(speed[active]).sum())
        for episode in result.episodes:
            if not learners[episode.slot]:
                continue
            self.completed += 1
            self.reasons[episode.reason] += 1
            self.lane_episodes += episode.lane_departures > 0
            self.signal_episodes += episode.signal_violations > 0

    def to_dict(self) -> dict:
        def rate(count: int) -> float | None:
            return count / self.completed if self.completed else None

        return {
            "completed_episodes": self.completed,
            "episode_outcomes": dict(self.reasons),
            "arrival_rate": rate(self.reasons["goal"]),
            "collision_rate": rate(self.reasons["collision"]),
            "lane_departure_rate": rate(self.lane_episodes),
            "signal_violation_rate": rate(self.signal_episodes),
            "vehicle_steps": self.vehicle_steps,
            "mean_speed_m_s": self.speed_sum / self.vehicle_steps if self.vehicle_steps else None,
        }
