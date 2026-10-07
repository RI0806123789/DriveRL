"""固定周期を保つ信号プランと時間帯別の青配分。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from app import config


def _weights(values: tuple[float, ...]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError("青の配分比は有限の非負数で指定してください")
    return result


@dataclass(frozen=True)
class SignalPeriod:
    """指定した時刻から使う、群番号順の青の配分比。"""

    start_hour: float
    group_weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        hour = float(self.start_hour)
        if not math.isfinite(hour) or not 0.0 <= hour < 24.0:
            raise ValueError("時間帯の開始は 0 以上 24 未満の時刻で指定してください")
        object.__setattr__(self, "start_hour", hour)
        object.__setattr__(self, "group_weights", _weights(self.group_weights))


@dataclass(frozen=True)
class SignalPlan:
    """周期境界だけで青の配分を切り替える信号設定。"""

    mode: Literal["fixed", "adaptive"] = "fixed"
    green_sec: float = config.SIGNAL_GREEN_SEC
    yellow_sec: float = config.SIGNAL_YELLOW_SEC
    all_red_sec: float = config.SIGNAL_ALL_RED_SEC
    green_min_sec: float = config.SIGNAL_GREEN_MIN_SEC
    start_hour: float = 0.0
    group_weights: tuple[float, ...] = ()
    time_of_day: tuple[SignalPeriod, ...] = ()
    demand_gain: float = 1.0
    demand_cap: float = 20.0

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "adaptive"):
            raise ValueError("信号プランは fixed または adaptive を指定してください")
        for name in ("green_sec", "yellow_sec", "all_red_sec", "green_min_sec", "demand_cap"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("信号の時間と需要の上限は有限の正数で指定してください")
            object.__setattr__(self, name, value)
        if self.green_sec < self.green_min_sec:
            raise ValueError("青の時間は最短青時間以上にしてください")
        hour = float(self.start_hour)
        gain = float(self.demand_gain)
        if not math.isfinite(hour) or not 0.0 <= hour < 24.0:
            raise ValueError("開始時刻は 0 以上 24 未満で指定してください")
        if not math.isfinite(gain) or gain < 0.0:
            raise ValueError("需要の倍率は有限の非負数で指定してください")
        if not math.isfinite((self.green_sec + self.yellow_sec + self.all_red_sec) * 2.0):
            raise ValueError("信号の周期が有限の範囲を超えています")
        if not math.isfinite(self.demand_cap * gain):
            raise ValueError("需要の上限と倍率の積が有限の範囲を超えています")
        object.__setattr__(self, "start_hour", hour)
        object.__setattr__(self, "demand_gain", gain)
        object.__setattr__(self, "group_weights", _weights(self.group_weights))
        periods = tuple(self.time_of_day)
        if any(not isinstance(period, SignalPeriod) for period in periods):
            raise ValueError("時間帯は SignalPeriod で指定してください")
        if any(a.start_hour >= b.start_hour for a, b in zip(periods, periods[1:])):
            raise ValueError("時間帯は開始時刻の昇順で、重複なく指定してください")
        object.__setattr__(self, "time_of_day", periods)
