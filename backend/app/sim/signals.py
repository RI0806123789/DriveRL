"""交通信号機の現示（灯色）制御。"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from app import config
from app.contracts import MapSignal

__all__ = [
    "SignalController",
    "GREEN",
    "YELLOW",
    "RED",
    "signal_speed_limit",
    "constrain_accel",
]

GREEN = 0
YELLOW = 1
RED = 2

DEFAULT_GREEN_SEC = config.SIGNAL_GREEN_SEC
DEFAULT_YELLOW_SEC = config.SIGNAL_YELLOW_SEC
DEFAULT_ALL_RED_SEC = config.SIGNAL_ALL_RED_SEC


class SignalController:
    """信号機の集合に対して、時刻から灯色を決める。"""

    def __init__(
        self,
        signals: Sequence[MapSignal],
        *,
        green_sec: float = DEFAULT_GREEN_SEC,
        yellow_sec: float = DEFAULT_YELLOW_SEC,
        all_red_sec: float = DEFAULT_ALL_RED_SEC,
    ) -> None:
        self.green = float(green_sec)
        self.yellow = float(yellow_sec)
        self.all_red = float(all_red_sec)
        self.half_cycle = self.green + self.yellow + self.all_red
        self.cycle = self.half_cycle * 2.0

        self._count = len(signals)
        self._groups = np.array([s.group for s in signals], dtype=np.int8)

        offsets = np.array(
            [(s.node_id * 7919) % max(1, int(self.cycle)) for s in signals],
            dtype=np.float64,
        )
        self._offsets = offsets

        self._buffer = np.full(self._count, RED, dtype=np.uint8)

    def __len__(self) -> int:
        return self._count

    @property
    def count(self) -> int:
        return self._count

    def phases(self, sim_time: float) -> list[int]:
        """各信号の灯色を返す。並びは `MapData.signals` と同じ。"""
        if self._count == 0:
            return []

        t = (float(sim_time) + self._offsets) % self.cycle
        local = np.where(self._groups == 0, t, (t + self.half_cycle) % self.cycle)

        out = self._buffer
        out.fill(RED)
        out[local < self.green] = GREEN
        np.putmask(
            out,
            (local >= self.green) & (local < self.green + self.yellow),
            YELLOW,
        )
        return out.tolist()

    def describe(self) -> str:
        """ログ用の説明。"""
        return (
            f"信号 {self._count} 基 / サイクル {self.cycle:.0f} 秒"
            f"（青 {self.green:.0f} + 黄 {self.yellow:.0f} + 全赤 {self.all_red:.0f}）"
        )


STOP_MARGIN_M = 1.0

BRAKE_USE_RATIO = 0.8


def signal_speed_limit(
    distance: np.ndarray,
    phase: np.ndarray,
    speed: np.ndarray,
    max_decel_abs: float,
    dt: float = 0.05,
) -> np.ndarray:
    """信号の色ごとに許される速度の上限を返す。"""
    brake = max(1e-3, float(max_decel_abs) * BRAKE_USE_RATIO)
    dt = max(float(dt), 1e-6)

    room = np.maximum(0.0, distance - STOP_MARGIN_M)

    at = brake * dt
    stop_limit = -at + np.sqrt(at * at + 2.0 * brake * room, dtype=np.float64)

    v = speed.astype(np.float64)
    stopping_distance = (v * v) / (2.0 * brake) + v * dt
    can_stop = stopping_distance <= room

    limit = np.full(distance.shape, np.inf, dtype=np.float64)
    limit = np.where(phase == RED, stop_limit, limit)
    limit = np.where((phase == YELLOW) & can_stop, stop_limit, limit)
    limit = np.where(np.isfinite(distance), limit, np.inf)
    return limit


def constrain_accel(
    accel_cmd: np.ndarray,
    speed: np.ndarray,
    speed_limit: np.ndarray,
    dt: float,
    max_accel: float,
    max_decel_abs: float,
) -> np.ndarray:
    """指令加速度を、次のステップで速度上限を超えないように抑える。"""
    limit = np.asarray(speed_limit, dtype=np.float64)
    if not np.isfinite(limit).any():
        return accel_cmd

    dt = max(float(dt), 1e-6)
    with np.errstate(invalid="ignore"):
        a_max = (limit - speed.astype(np.float64)) / dt

    max_accel = max(float(max_accel), 1e-6)
    max_decel_abs = max(float(max_decel_abs), 1e-6)
    cmd_max = np.where(a_max >= 0.0, a_max / max_accel, a_max / max_decel_abs)
    cmd_max = np.where(np.isfinite(cmd_max), cmd_max, 1.0)

    out = np.minimum(accel_cmd.astype(np.float64), cmd_max)
    return np.clip(out, -1.0, 1.0).astype(np.float32)
