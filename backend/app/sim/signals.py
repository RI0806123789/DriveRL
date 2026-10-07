"""交通信号機の現示（灯色）制御。"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from app import config
from app.contracts import MapSignal
from app.sim.signal_plan import SignalPlan

__all__ = [
    "SignalController",
    "GREEN",
    "YELLOW",
    "RED",
    "signal_speed_limit",
    "stop_speed_limit",
    "constrain_accel",
]

GREEN = 0
YELLOW = 1
RED = 2

DEFAULT_GREEN_SEC = config.SIGNAL_GREEN_SEC
DEFAULT_YELLOW_SEC = config.SIGNAL_YELLOW_SEC
DEFAULT_ALL_RED_SEC = config.SIGNAL_ALL_RED_SEC
DEFAULT_GREEN_MIN_SEC = config.SIGNAL_GREEN_MIN_SEC


def _scatter01(keys: np.ndarray) -> np.ndarray:
    """交差点のキーを 0〜1 の実数へ散らす（splitmix64 の finalizer）。"""
    z = keys.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


class SignalController:
    """信号機の集合に対して、時刻から灯色を決める。"""

    def __init__(
        self,
        signals: Sequence[MapSignal],
        *,
        green_sec: float = DEFAULT_GREEN_SEC,
        yellow_sec: float = DEFAULT_YELLOW_SEC,
        all_red_sec: float = DEFAULT_ALL_RED_SEC,
        green_min_sec: float = DEFAULT_GREEN_MIN_SEC,
        plan: SignalPlan | None = None,
    ) -> None:
        self.plan = plan
        if plan is not None:
            green_sec = plan.green_sec
            yellow_sec = plan.yellow_sec
            all_red_sec = plan.all_red_sec
            green_min_sec = plan.green_min_sec
        self.green = float(green_sec)
        self.yellow = float(yellow_sec)
        self.all_red = float(all_red_sec)
        self.green_min = float(green_min_sec)
        self.cycle = (self.green + self.yellow + self.all_red) * 2.0

        self._count = len(signals)
        groups = np.array([s.group for s in signals], dtype=np.float64)

        phase_count: dict[int, int] = {}
        for s in signals:
            key = int(s.phase_key)
            phase_count[key] = max(phase_count.get(key, 1), int(s.group) + 1)
        counts = np.maximum(
            2.0,
            np.array([phase_count[int(s.phase_key)] for s in signals], dtype=np.float64),
        )
        self.max_groups = int(counts.max()) if self._count else 0

        self._green = np.maximum(
            self.green_min, self.cycle / counts - self.yellow - self.all_red
        )
        slots = self._green + self.yellow + self.all_red
        self._cycles = slots * counts
        self._shift = slots * groups
        self._yellow_end = self._green + self.yellow
        self.max_cycle = float(self._cycles.max()) if self._count else self.cycle

        keys = np.array([s.phase_key for s in signals], dtype=np.int64)
        self._offsets = _scatter01(keys) * self._cycles

        self._buffer = np.full(self._count, RED, dtype=np.uint8)
        if plan is not None and self._count:
            self._init_plan(keys, groups.astype(np.int64), phase_count)

    def _init_plan(
        self, keys: np.ndarray, groups: np.ndarray, phase_count: dict[int, int]
    ) -> None:
        unique, self._head_key = np.unique(keys, return_inverse=True)
        self._key_counts = np.array([max(2, phase_count[int(key)]) for key in unique])
        self._key_slot_base = np.cumsum(self._key_counts) - self._key_counts
        self._slot_key = np.repeat(np.arange(len(unique)), self._key_counts)
        self._slot_group = np.arange(self._slot_key.size) - self._key_slot_base[self._slot_key]
        self._head_slot = self._key_slot_base[self._head_key] + groups
        self._key_cycles = np.maximum(
            self.cycle, self._key_counts * (self.green_min + self.yellow + self.all_red)
        )
        self._key_offsets = _scatter01(unique) * self._key_cycles
        self._active_cycle = np.full(len(unique), np.nan)
        self._slot_green = np.zeros(self._slot_key.size)
        self._slot_start = np.zeros(self._slot_key.size)
        self._last_time: float | None = None
        plan = self.plan
        assert plan is not None
        self._period_hours = np.array([period.start_hour for period in plan.time_of_day])
        patterns = (plan.group_weights,) + tuple(period.group_weights for period in plan.time_of_day)
        self._pattern_weights = np.ones((len(patterns), self.max_groups))
        for row, weights in enumerate(patterns):
            length = min(len(weights), self.max_groups)
            self._pattern_weights[row, :length] = weights[:length]
        self._pattern_weights /= np.maximum(1.0, self._pattern_weights.max(axis=1))[:, None]

    def __len__(self) -> int:
        return self._count

    @property
    def count(self) -> int:
        return self._count

    def phases(self, sim_time: float, demand: np.ndarray | None = None) -> list[int]:
        """各信号の灯色を返す。並びは `MapData.signals` と同じ。"""
        if self._count == 0:
            return []

        if self.plan is not None:
            return self._planned_phases(float(sim_time), demand)

        local = (float(sim_time) + self._offsets + self._shift) % self._cycles

        out = self._buffer
        out.fill(RED)
        out[local < self._green] = GREEN
        np.putmask(out, (local >= self._green) & (local < self._yellow_end), YELLOW)
        return out.tolist()

    def _planned_phases(self, sim_time: float, demand: np.ndarray | None) -> list[int]:
        plan = self.plan
        assert plan is not None
        if not np.isfinite(sim_time):
            raise ValueError("信号の時刻は有限の値で指定してください")
        cycle = np.floor((sim_time + self._key_offsets) / self._key_cycles)
        if self._last_time is not None and sim_time < self._last_time:
            self._active_cycle.fill(np.nan)
        changed = cycle != self._active_cycle
        if changed.any():
            boundary = cycle * self._key_cycles - self._key_offsets
            hour = (plan.start_hour + boundary / 3600.0) % 24.0
            if self._period_hours.size:
                pattern = np.searchsorted(self._period_hours, hour, side="right")
                pattern[pattern == 0] = self._period_hours.size
            else:
                pattern = np.zeros(cycle.size, dtype=np.int64)
            weights = self._pattern_weights[pattern[self._slot_key], self._slot_group].copy()
            if plan.mode == "adaptive" and demand is not None:
                counts = np.asarray(demand, dtype=np.float64)
                if counts.shape != (self._count,) or not np.isfinite(counts).all() or (counts < 0.0).any():
                    raise ValueError("信号の需要は灯器順の有限の非負配列で指定してください")
                grouped = np.bincount(
                    self._head_slot,
                    weights=np.minimum(counts, plan.demand_cap) / plan.demand_cap,
                    minlength=weights.size,
                )
                grouped = np.minimum(grouped, 1.0) * plan.demand_cap
                weights *= 1.0 + np.minimum(grouped, plan.demand_cap) * plan.demand_gain
            scale = np.maximum.reduceat(weights, self._key_slot_base)
            weights /= np.maximum(1.0, scale[self._slot_key])
            total = np.bincount(self._slot_key, weights=weights, minlength=cycle.size)
            uniform = total[self._slot_key] == 0.0
            weights[uniform] = 1.0
            total[total == 0.0] = self._key_counts[total == 0.0]
            spare = self._key_cycles - self._key_counts * (self.green_min + self.yellow + self.all_red)
            green = self.green_min + spare[self._slot_key] * weights / total[self._slot_key]
            slots = green + self.yellow + self.all_red
            end = np.cumsum(slots)
            key_ends = np.cumsum(self._key_counts) - 1
            key_origin = np.zeros(cycle.size)
            key_origin[1:] = end[key_ends[:-1]]
            start = end - slots - key_origin[self._slot_key]
            slot_changed = changed[self._slot_key]
            self._slot_green[slot_changed] = green[slot_changed]
            self._slot_start[slot_changed] = start[slot_changed]
            self._active_cycle[changed] = cycle[changed]
        self._last_time = sim_time
        local = (sim_time + self._key_offsets[self._head_key]) % self._key_cycles[self._head_key]
        elapsed = local - self._slot_start[self._head_slot]
        green = self._slot_green[self._head_slot]
        out = self._buffer
        out.fill(RED)
        out[(elapsed >= 0.0) & (elapsed < green)] = GREEN
        np.putmask(out, (elapsed >= green) & (elapsed < green + self.yellow), YELLOW)
        return out.tolist()

    def describe(self) -> str:
        """ログ用の説明。"""
        if self._count == 0:
            return "信号 0 基"
        if self.plan is not None:
            mode = "需要応答" if self.plan.mode == "adaptive" else "固定"
            return (
                f"信号 {self._count} 基 / {mode}プラン / サイクル {self.cycle:.0f} 秒"
                f"（現示は最大 {self.max_groups} 通り・最短青 {self.green_min:.0f}"
                f" + 黄 {self.yellow:.0f} + 全赤 {self.all_red:.0f}）"
            )
        return (
            f"信号 {self._count} 基 / サイクル {self.cycle:.0f} 秒"
            f"（現示は最大 {self.max_groups} 通り・青 {self._green.min():.0f}〜"
            f"{self._green.max():.0f} + 黄 {self.yellow:.0f} + 全赤 {self.all_red:.0f}）"
        )


STOP_MARGIN_M = 1.0

BRAKE_USE_RATIO = 0.8


def stop_speed_limit(
    distance: np.ndarray,
    max_decel_abs: float,
    dt: float = 0.05,
    margin_m: float = STOP_MARGIN_M,
) -> np.ndarray:
    """距離 `distance` の手前 `margin_m` で止まりきれる速度の上限を返す。"""
    brake = max(1e-3, float(max_decel_abs) * BRAKE_USE_RATIO)
    dt = max(float(dt), 1e-6)
    room = np.maximum(0.0, np.asarray(distance, dtype=np.float64) - float(margin_m))
    at = brake * dt
    return -at + np.sqrt(at * at + 2.0 * brake * room, dtype=np.float64)


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

    stop_limit = stop_speed_limit(distance, max_decel_abs, dt, STOP_MARGIN_M)

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
