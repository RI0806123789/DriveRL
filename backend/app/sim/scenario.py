"""再現可能な交通・路面・信号のシナリオ設定と背景交通。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from app import config
from app.sim.dynamics import VehicleDynamics
from app.sim.signal_plan import SignalPeriod, SignalPlan
from app.warn import warn_once

if TYPE_CHECKING:
    from app.sim.env import SimulationEnv


def _range(name: str, value: float, low: float, high: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} は {low}〜{high} の有限値にしてください")


@dataclass(frozen=True)
class DriverBehavior:
    """背景車の目標速度・時間車間・ペダル特性・操作のばらつき。"""

    speed_ratio: float = 0.9
    headway_sec: float = 1.5
    accel_scale: float = 1.0
    brake_scale: float = 1.0
    steering_noise: float = 0.0

    def __post_init__(self) -> None:
        for name, low, high in (
            ("speed_ratio", 0.2, 1.2), ("headway_sec", 0.5, 4.0),
            ("accel_scale", 0.2, 1.5), ("brake_scale", 0.2, 1.5),
            ("steering_noise", 0.0, 0.1),
        ):
            _range(name, getattr(self, name), low, high)


@dataclass(frozen=True)
class TrafficPeriod:
    """開始時刻 [時] 以降の背景車の台数。"""

    start_hour: float
    count: int

    def __post_init__(self) -> None:
        _range("start_hour", self.start_hour, 0.0, 24.0 - 1e-6)
        if type(self.count) is not int or not 0 <= self.count <= config.MAX_VEHICLES:
            raise ValueError("count は車両スロット数以内の非負整数にしてください")


@dataclass(frozen=True)
class RoadEvent:
    """指定道路の車線上へ期間つきの障害物を配置する。"""

    edge_id: int
    kind: str = "construction"
    start_sec: float = 0.0
    duration_sec: float = 60.0
    fraction: float = 0.5

    def __post_init__(self) -> None:
        if type(self.edge_id) is not int or self.edge_id < 0:
            raise ValueError("edge_id は非負整数にしてください")
        if self.kind not in ("parked", "construction", "closure"):
            raise ValueError("kind は parked / construction / closure のいずれかです")
        _range("start_sec", self.start_sec, 0.0, 86400.0 * 365)
        _range("duration_sec", self.duration_sec, 0.05, 86400.0 * 365)
        _range("fraction", self.fraction, 0.1, 0.9)


@dataclass(frozen=True)
class Scenario:
    """通信パラメータと独立した任意の環境設定。None のときは従来の環境。"""

    name: str = "baseline"
    dynamics: VehicleDynamics | None = None
    signals: SignalPlan | None = None
    learner_vehicles: int = 1
    background_vehicles: int = 0
    start_hour: float = 12.0
    periods: tuple[TrafficPeriod, ...] = ()
    drivers: tuple[DriverBehavior, ...] = ()
    driver_spread: float = 0.0
    road_weights: dict[int, float] = field(default_factory=dict)
    events: tuple[RoadEvent, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name は空でない文字列にしてください")
        if type(self.learner_vehicles) is not int or type(self.background_vehicles) is not int:
            raise ValueError("車両数は整数にしてください")
        if self.learner_vehicles < 1 or self.background_vehicles < 0 or self.learner_vehicles + self.background_vehicles > config.MAX_VEHICLES:
            raise ValueError("学習車は 1 台以上、合計は車両スロット数以内にしてください")
        _range("start_hour", self.start_hour, 0.0, 24.0 - 1e-6)
        _range("driver_spread", self.driver_spread, 0.0, 0.5)
        starts = [p.start_hour for p in self.periods]
        if starts != sorted(set(starts)) or any(p.count > self.background_vehicles for p in self.periods):
            raise ValueError("periods は開始時刻の昇順で重複なく、予約した背景車の台数以内にしてください")
        if self.drivers and len(self.drivers) != self.background_vehicles:
            raise ValueError("drivers は背景車の台数と同じ数を指定してください")
        if len(self.events) > config.MAX_OBSTACLES // 12:
            raise ValueError("同時イベントの障害物が上限を超えない数にしてください")
        for key, weight in self.road_weights.items():
            if type(key) is not int or key < 0:
                raise ValueError("road_weights のキーは道路 ID にしてください")
            _range("road weight", weight, 0.0, 1000.0)
        if self.road_weights and sum(self.road_weights.values()) <= 0:
            raise ValueError("road_weights は少なくとも 1 本に正の重みが必要です")

    def traffic_count(self, sim_time: float) -> int:
        if not self.periods:
            return self.background_vehicles
        hour = (self.start_hour + sim_time / 3600.0) % 24.0
        period = self.periods[-1]
        for candidate in self.periods:
            if candidate.start_hour > hour:
                break
            period = candidate
        return period.count


def load_scenario(path: str | Path) -> Scenario:
    """未知のキーも拒否して JSON 設定を読み込む。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("シナリオは JSON オブジェクトにしてください")
    values = dict(payload)
    model = values.pop("model", "kinematic")
    physics = values.pop("physics", {})
    if model not in ("kinematic", "dynamic"):
        raise ValueError("model は kinematic / dynamic のいずれかです")
    if model == "kinematic" and physics:
        raise ValueError("physics を指定するときは model を dynamic にしてください")
    values["dynamics"] = VehicleDynamics(**physics) if model == "dynamic" else None
    if "signals" in values:
        plan = dict(values["signals"])
        plan.setdefault("start_hour", values.get("start_hour", 12.0))
        plan["time_of_day"] = tuple(SignalPeriod(**p) for p in plan.get("time_of_day", []))
        values["signals"] = SignalPlan(**plan)
    values["periods"] = tuple(TrafficPeriod(**p) for p in values.get("periods", []))
    values["drivers"] = tuple(DriverBehavior(**p) for p in values.get("drivers", []))
    values["events"] = tuple(RoadEvent(**p) for p in values.get("events", []))
    values["road_weights"] = {int(k): v for k, v in values.get("road_weights", {}).items()}
    return Scenario(**values)


def standard_scenario() -> Scenario:
    """通常運転で常時使う詳細物理・背景交通・需要応答信号。"""
    return Scenario(
        name="standard",
        dynamics=VehicleDynamics(),
        learner_vehicles=1,
        background_vehicles=3,
        start_hour=12.0,
        driver_spread=0.2,
        periods=(
            TrafficPeriod(0, 1), TrafficPeriod(8, 3), TrafficPeriod(10, 2),
            TrafficPeriod(18, 3), TrafficPeriod(21, 1),
        ),
        signals=SignalPlan(
            mode="adaptive",
            start_hour=12.0,
            time_of_day=(
                SignalPeriod(0, (1, 1)), SignalPeriod(8, (2, 1)),
                SignalPeriod(10, (1, 1)), SignalPeriod(18, (1, 2)),
                SignalPeriod(21, (1, 1)),
            ),
        ),
    )


class ScenarioTraffic:
    """専用乱数で背景交通と一時的な道路イベントを管理する。"""

    def __init__(self, scenario: Scenario, seed: int, env: SimulationEnv) -> None:
        self.scenario = scenario
        self.rng = np.random.default_rng([seed, 101])
        self.mask = np.zeros(config.MAX_VEHICLES, dtype=bool)
        self.mask[scenario.learner_vehicles:scenario.learner_vehicles + scenario.background_vehicles] = True
        self.drivers: dict[int, DriverBehavior] = {}
        for i, slot in enumerate(np.flatnonzero(self.mask)):
            if scenario.drivers:
                driver = scenario.drivers[i]
            else:
                spread = scenario.driver_spread
                driver = DriverBehavior(
                    speed_ratio=float(self.rng.uniform(0.9 - spread * 0.4, 0.9 + spread * 0.4)),
                    headway_sec=float(self.rng.uniform(1.5 - spread, 1.5 + spread)),
                    accel_scale=float(self.rng.uniform(1.0 - spread, 1.0 + spread)),
                    brake_scale=float(self.rng.uniform(1.0 - spread, 1.0 + spread)),
                    steering_noise=spread * 0.02,
                )
            self.drivers[int(slot)] = driver
        edges = {e.id: e for e in env.map_index.data.edges}
        missing = (set(scenario.road_weights) | {e.edge_id for e in scenario.events}) - edges.keys()
        if missing:
            raise ValueError(f"地図にない道路 ID です: {sorted(missing)}")
        self.roads = [edges[k] for k in scenario.road_weights]
        weights = np.asarray(list(scenario.road_weights.values()), dtype=np.float64)
        self.weights = weights / weights.sum() if weights.size else weights
        self.events = [(event, edges[event.edge_id]) for event in scenario.events]
        for event, edge in self.events:
            self.road_point(edge, event.fraction)
            if event.kind == "closure" and float(edge.width) > 120.0:
                raise ValueError("通行規制を表現できる道路幅の上限を超えています")
        self._event_ids: dict[int, list[int]] = {}
        self._started_events: set[int] = set()

    @staticmethod
    def road_point(edge, fraction: float) -> tuple[float, float, float, float]:
        points = np.asarray(edge.polyline, dtype=np.float64)
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc = np.concatenate(([0.0], np.cumsum(lengths)))
        if arc[-1] <= 0.0:
            raise ValueError(f"道路 {edge.id} の長さがゼロです")
        distance = float(fraction) * arc[-1]
        i = min(len(lengths) - 1, int(np.searchsorted(arc, distance, side="right") - 1))
        direction = (points[i + 1] - points[i]) / max(float(lengths[i]), 1e-9)
        center = points[i] + direction * (distance - arc[i])
        lane = min(float(edge.width) * 0.25, 1.75)
        return float(center[0] - direction[1] * lane), float(center[1] + direction[0] * lane), float(direction[0]), float(direction[1])

    def prepare(self, env: SimulationEnv, *, initial: bool = False) -> bool:
        """台数は指定時刻で切り替え、新しく出る背景車の道路を重みで選ぶ。"""
        changed = False
        attempts = 0
        count = self.scenario.traffic_count(env.sim_time)
        for i, slot in enumerate(np.flatnonzero(self.mask)):
            slot = int(slot)
            if slot == env.commandeered_slot:
                continue
            wanted = i < count
            if not wanted:
                env._drop_from_respawn_queue(slot)
                if env.world.fleet.active[slot]:
                    env.world.deactivate(slot)
                    changed = True
                continue
            if not env.world.fleet.active[slot] and slot not in env._respawn_queue:
                if not initial and attempts >= 1:
                    continue
                attempts += 1
                changed |= self.spawn(env, slot)
        changed |= self._update_events(env)
        if changed:
            env.world.project_all()
        return changed

    def spawn(self, env: SimulationEnv, slot: int) -> bool:
        if self.roads:
            edge = self.roads[int(self.rng.choice(len(self.roads), p=self.weights))]
            x, y, tx, ty = self.road_point(edge, float(self.rng.uniform(0.15, 0.85)))
            why = env.world.activate_at(slot, (x, y), clearance_m=config.VEHICLE_LENGTH * 4, heading=math.atan2(ty, tx))
            if why is not None:
                warn_once("scenario.road_spawn", f"指定道路に背景車を出せませんでした: {why}")
                env.world.deactivate(slot)
            placed = why is None
        else:
            placed = env.world.activate(slot)
        return placed

    def _update_events(self, env: SimulationEnv) -> bool:
        changed = False
        now = env.sim_time
        for i, (event, edge) in enumerate(self.events):
            if now >= event.start_sec + event.duration_sec:
                for obstacle in self._event_ids.pop(i, []):
                    changed |= env.world.remove_obstacle(obstacle)
                continue
            if now < event.start_sec or i in self._started_events:
                continue
            self._started_events.add(i)
            x, y, tx, ty = self.road_point(edge, event.fraction)
            if event.kind == "parked":
                positions = [(x + t * tx, y + t * ty, 0.9) for t in (-1.4, 0.0, 1.4)]
            elif event.kind == "construction":
                positions = [(x + t * tx, y + t * ty, 0.5) for t in np.arange(-4.0, 4.1, 2.0)]
            else:
                width = float(edge.width)
                count = min(12, max(2, int(math.ceil(width / 1.8))))
                radius = max(1.0, width / (2.0 * count))
                center_x, center_y = x + ty * min(float(edge.width) * 0.25, 1.75), y - tx * min(float(edge.width) * 0.25, 1.75)
                positions = [(center_x - ty * t, center_y + tx * t, radius) for t in np.linspace(-width / 2 + radius, width / 2 - radius, count)]
            ids = []
            for px, py, radius in positions:
                obstacle_id = env.world.add_obstacle(px, py, radius)
                if obstacle_id is None:
                    warn_once("scenario.event_capacity", "道路イベントの障害物が上限に達しました")
                    break
                ids.append(obstacle_id)
            self._event_ids[i] = ids
            changed |= bool(ids)
        return changed

    def adjust_commands(self, slot: int, accel: float, steer: float) -> tuple[float, float]:
        driver = self.drivers[slot]
        scale = driver.accel_scale if accel >= 0.0 else driver.brake_scale
        noise = float(self.rng.normal(0.0, driver.steering_noise)) if driver.steering_noise else 0.0
        return float(np.clip(accel * scale, -1.0, 1.0)), float(np.clip(steer + noise, -1.0, 1.0))
