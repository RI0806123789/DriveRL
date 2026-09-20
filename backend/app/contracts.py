"""バックエンド内部の共有契約（データ構造とインターフェース）。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from app import config


@dataclass(frozen=True)
class MapPreset:
    """事前定義された読み込み対象エリア（memo 5章「複数の固定エリアを事前プリセット」）。"""

    id: str
    name: str
    description: str
    center_lat: float
    center_lon: float
    radius_m: float
    signals_at_all_intersections: bool = True

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "centerLat": self.center_lat,
            "centerLon": self.center_lon,
            "radiusM": self.radius_m,
        }


@dataclass(frozen=True)
class Bounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def to_wire(self) -> dict[str, float]:
        return {"minX": self.min_x, "minY": self.min_y, "maxX": self.max_x, "maxY": self.max_y}


@dataclass
class MapNode:
    """道路ネットワークの交差点／端点。"""

    id: int
    x: float
    y: float


@dataclass
class MapEdge:
    """道路セグメント。polyline は始点が u、終点が v に対応する。"""

    id: int
    u: int
    v: int
    lanes: int
    width: float
    oneway: bool
    speed_limit: float
    polyline: list[tuple[float, float]]
    length: float


@dataclass
class MapBuilding:
    """建物のフットプリント。outline は閉じない（末尾と先頭は重複させない）。"""

    id: int
    height: float
    outline: list[tuple[float, float]]


@dataclass
class MapSignal:
    """交通信号機（車両用）。OSM の `highway=traffic_signals` ノードから作る。"""

    id: int
    node_id: int
    x: float
    y: float
    heading: float
    group: int
    road_width: float


@dataclass
class MapSign:
    """最高速度標識（規制標識「最高速度」）。"""

    id: int
    node_id: int
    edge_id: int
    x: float
    y: float
    heading: float
    speed_limit: float


@dataclass
class MapData:
    """1 プリセット分の静的マップデータ。JSON にそのまま落とせる範囲だけを持つ。"""

    preset_id: str
    name: str
    center_lat: float
    center_lon: float
    radius_m: float
    bounds: Bounds
    nodes: list[MapNode]
    edges: list[MapEdge]
    buildings: list[MapBuilding]
    signals: list[MapSignal] = field(default_factory=list)
    signs: list[MapSign] = field(default_factory=list)

    def to_wire(self) -> dict[str, Any]:
        """docs/protocol.md 2.2 の map メッセージ本体を返す（type は呼び出し側で付与）。"""
        return {
            "presetId": self.preset_id,
            "name": self.name,
            "bounds": self.bounds.to_wire(),
            "nodes": [{"id": n.id, "x": round(n.x, 3), "y": round(n.y, 3)} for n in self.nodes],
            "edges": [
                {
                    "id": e.id,
                    "u": e.u,
                    "v": e.v,
                    "lanes": e.lanes,
                    "width": round(e.width, 2),
                    "oneway": e.oneway,
                    "speedLimit": round(e.speed_limit, 2),
                    "polyline": [[round(px, 3), round(py, 3)] for px, py in e.polyline],
                }
                for e in self.edges
            ],
            "buildings": [
                {
                    "id": b.id,
                    "height": round(b.height, 2),
                    "outline": [[round(px, 3), round(py, 3)] for px, py in b.outline],
                }
                for b in self.buildings
            ],
            "signals": [
                {
                    "id": s.id,
                    "nodeId": s.node_id,
                    "x": round(s.x, 3),
                    "y": round(s.y, 3),
                    "heading": round(s.heading, 4),
                    "group": s.group,
                    "roadWidth": round(s.road_width, 2),
                }
                for s in self.signals
            ],
            "signs": [
                {
                    "id": sg.id,
                    "nodeId": sg.node_id,
                    "edgeId": sg.edge_id,
                    "x": round(sg.x, 3),
                    "y": round(sg.y, 3),
                    "heading": round(sg.heading, 4),
                    "speedLimit": round(sg.speed_limit, 3),
                }
                for sg in self.signs
            ],
        }


@dataclass
class OccupancyGrid:
    """ENU 平面を等間隔セルに区切った占有マップ。"""

    origin_x: float
    origin_y: float
    cell_size: float
    width: int
    height: int
    building: np.ndarray

    def world_to_cell(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ENU 座標をセル添字 (col, row) に変換する。範囲外の値も返るので呼び出し側でクリップする。"""
        col = np.rint((np.asarray(x) - self.origin_x) / self.cell_size).astype(np.int32)
        row = np.rint((np.asarray(y) - self.origin_y) / self.cell_size).astype(np.int32)
        return col, row

    def sample(self, layer: np.ndarray, x: np.ndarray, y: np.ndarray, outside: bool) -> np.ndarray:
        """指定レイヤを ENU 座標で参照する。グリッド外は outside の値を返す。"""
        col, row = self.world_to_cell(x, y)
        inside = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        out = np.full(np.shape(col), outside, dtype=bool)
        if inside.any():
            out[inside] = layer[row[inside], col[inside]]
        return out

    def sample_building(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """建物内部なら True。グリッド外は False（何もない空間扱い）。"""
        return self.sample(self.building, x, y, outside=False)


class MapIndex(Protocol):
    """経路探索・最近傍探索・レイキャストを提供する実行時インデックス。"""

    data: MapData
    occupancy: OccupancyGrid

    def nearest_node(self, x: float, y: float) -> int:
        """指定座標に最も近い道路ノード ID を返す。"""
        ...

    def nearest_road_point(self, x: float, y: float) -> tuple[float, float, int, float]:
        """指定座標を最寄りの道路中心線上にスナップする。"""
        ...

    def shortest_path(self, src_node: int, dst_node: int) -> list[int] | None:
        """ノード ID 列で最短経路を返す。到達不能なら None。"""
        ...

    def route_polyline(
        self, node_path: Sequence[int], resample_m: float = 2.0
    ) -> list[tuple[float, float]]:
        """ノード列をエッジのポリラインに展開し、等間隔にリサンプルした点列を返す。"""
        ...

    def lane_route_polyline(
        self, node_path: Sequence[int], resample_m: float = 2.0
    ) -> list[tuple[float, float]]:
        """左側通行の車線に沿った走行経路を返す。"""
        ...

    def signals_on_route(
        self, points: Sequence[tuple[float, float]]
    ) -> list[tuple[float, int]]:
        """経路が通過する信号を (経路始点からの弧長 [m], MapData.signals の添字) で返す。"""
        ...

    def speed_limits_on_route(
        self, points: Sequence[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """経路に適用される規制速度を (弧長 [m], 規制速度 [m/s]) の区切りで返す。"""
        ...

    def random_node_pair(
        self, rng: np.random.Generator, min_distance_m: float = 150.0
    ) -> tuple[int, int]:
        """経路が存在し、かつ十分離れた出発ノードと目的ノードの組を返す。"""
        ...

    def raycast(
        self,
        origin_x: np.ndarray,
        origin_y: np.ndarray,
        angles: np.ndarray,
        max_distance: float,
        step: float = 1.0,
    ) -> np.ndarray:
        """占有グリッド上で建物までの距離を測る。"""
        ...

    def collides_with_building(self, corners: np.ndarray) -> bool:
        """車両の外接矩形（shape (4, 2) の頂点列）が建物と重なるかを厳密に判定する。"""
        ...

    def collides_with_buildings(
        self, corners: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """複数台ぶんをまとめて判定する。shape (N,) bool。"""
        ...


@dataclass(frozen=True)
class _ParamSpec:
    """1 パラメータのワイヤ名・型・値域。"""

    wire: str
    kind: type
    minimum: float | None = None
    maximum: float | None = None


_PARAM_SPECS: dict[str, _ParamSpec] = {
    "vehicle_count": _ParamSpec("vehicleCount", int, 0, None),
    "pedestrian_count": _ParamSpec("pedestrianCount", int, 0, None),
    "sim_speed": _ParamSpec("simSpeed", float, 0.25, 8.0),
    "learning_rate": _ParamSpec("learningRate", float, 1e-6, 1e-2),
    "gamma": _ParamSpec("gamma", float, 0.5, 0.9999),
    "clip_range": _ParamSpec("clipRange", float, 0.01, 0.9),
    "entropy_coef": _ParamSpec("entropyCoef", float, 0.0, 0.5),
    "rollout_length": _ParamSpec("rolloutLength", int, 16, 2048),
    "max_speed": _ParamSpec("maxSpeed", float, 1.0, 40.0),
    "reward_goal": _ParamSpec("rewardGoal", float, 0.0, 1000.0),
    "reward_collision": _ParamSpec("rewardCollision", float, -1000.0, 0.0),
    "reward_progress": _ParamSpec("rewardProgress", float, 0.0, 50.0),
    "reward_offroad": _ParamSpec("rewardOffroad", float, -100.0, 0.0),
    "reward_time": _ParamSpec("rewardTime", float, -10.0, 0.0),
    "reward_signal": _ParamSpec("rewardSignal", float, -1000.0, 0.0),
    "reward_overspeed": _ParamSpec("rewardOverspeed", float, -1000.0, 0.0),
    "obey_signals": _ParamSpec("obeySignals", bool),
    "obey_speed_signs": _ParamSpec("obeySpeedSigns", bool),
    "weather_rain": _ParamSpec("weatherRain", float, 0.0, 1.0),
    "weather_fog": _ParamSpec("weatherFog", float, 0.0, 1.0),
    "weather_auto": _ParamSpec("weatherAuto", bool),
}

_TRUE_WORDS = {"true", "1", "yes", "on"}
_FALSE_WORDS = {"false", "0", "no", "off"}


@dataclass
class ParamPatchResult:
    """`SimParams.apply_wire()` の結果。"""

    changed: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    clamped: list[str] = field(default_factory=list)

    @property
    def has_problem(self) -> bool:
        return bool(self.rejected or self.clamped)


def coerce_bool(value: Any) -> bool | None:
    """真偽値へ変換する。解釈できなければ None。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return None if not math.isfinite(float(value)) else bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    return None


@dataclass
class SimParams:
    """docs/protocol.md 2.5 の params に対応。"""

    vehicle_count: int = 4
    pedestrian_count: int = 16
    sim_speed: float = 1.0
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_range: float = 0.2
    entropy_coef: float = 0.001
    rollout_length: int = 256
    max_speed: float = 13.9
    reward_goal: float = 100.0
    reward_collision: float = -100.0
    reward_progress: float = 1.0
    reward_offroad: float = -1.0
    reward_time: float = -0.05
    reward_signal: float = -60.0
    reward_overspeed: float = -5.0
    obey_signals: bool = True
    obey_speed_signs: bool = True
    weather_rain: float = 0.0
    weather_fog: float = 0.0
    weather_auto: bool = False

    def to_wire(self) -> dict[str, Any]:
        return {spec.wire: getattr(self, snake) for snake, spec in _PARAM_SPECS.items()}

    def apply_wire(
        self,
        patch: dict[str, Any],
        *,
        max_vehicles: int = 64,
        max_pedestrians: int = 64,
    ) -> ParamPatchResult:
        """camelCase の部分更新を検証してから適用する。"""
        result = ParamPatchResult()
        reverse = {spec.wire: snake for snake, spec in _PARAM_SPECS.items()}

        for wire_key, value in patch.items():
            snake = reverse.get(wire_key)
            if snake is None:
                continue
            spec = _PARAM_SPECS[snake]
            current = getattr(self, snake)

            if spec.kind is bool:
                coerced: Any = coerce_bool(value)
                if coerced is None:
                    result.rejected.append(wire_key)
                    continue
            else:
                if isinstance(value, bool):
                    value = int(value)
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    result.rejected.append(wire_key)
                    continue
                if not math.isfinite(number):
                    result.rejected.append(wire_key)
                    continue

                low = spec.minimum
                high = spec.maximum
                if snake == "vehicle_count":
                    high = float(max(1, int(max_vehicles)))
                elif snake == "pedestrian_count":
                    high = float(max(0, int(max_pedestrians)))
                clipped = number
                if low is not None:
                    clipped = max(clipped, float(low))
                if high is not None:
                    clipped = min(clipped, float(high))
                if clipped != number:
                    result.clamped.append(wire_key)
                coerced = int(round(clipped)) if spec.kind is int else float(clipped)

            if coerced != current:
                setattr(self, snake, coerced)
                result.changed.append(snake)

        return result


@dataclass
class VehicleSnapshot:
    id: int
    active: bool
    x: float
    y: float
    heading: float
    speed: float
    steer: float
    collided: bool
    reached_goal: bool
    goal: tuple[float, float]
    progress: float = 0.0
    signal_violations: int = 0
    lane_departures: int = 0
    speed_limit: float = 0.0
    speed_violations: int = 0
    #: 制動指令が出ているか（ブレーキランプ）
    braking: bool = False
    #: 方向指示器。-1=左 / 0=消灯 / +1=右
    turn_signal: int = 0
    route: list[tuple[float, float]] | None = None

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "active": self.active,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "heading": round(self.heading, 4),
            "speed": round(self.speed, 3),
            "steer": round(self.steer, 4),
            "collided": self.collided,
            "reachedGoal": self.reached_goal,
            "goal": [round(self.goal[0], 3), round(self.goal[1], 3)],
            "progress": round(self.progress, 4),
            "signalViolations": self.signal_violations,
            "laneDepartures": self.lane_departures,
            "speedLimit": round(self.speed_limit, 3),
            "speedViolations": self.speed_violations,
            "braking": self.braking,
            "turnSignal": int(self.turn_signal),
        }
        if self.route is not None:
            out["route"] = [[round(px, 2), round(py, 2)] for px, py in self.route]
        return out


@dataclass
class ObstacleSnapshot:
    id: int
    x: float
    y: float
    radius: float

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "radius": self.radius,
        }


@dataclass
class PedestrianSnapshot:
    """NPC 歩行者 1 人。`stride` は手足の振りの位相で、描画のためだけに送る。"""

    id: int
    x: float
    y: float
    heading: float
    stride: float = 0.0
    crossing: bool = False

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "heading": round(self.heading, 3),
            "stride": round(self.stride, 2),
            "crossing": self.crossing,
        }


@dataclass
class FrameSnapshot:
    """docs/protocol.md 2.3 の frame に対応。vehicles は常に全スロット分含む。"""

    tick: int
    sim_time: float
    vehicles: list[VehicleSnapshot]
    obstacles: list[ObstacleSnapshot]
    pedestrians: list[PedestrianSnapshot] = field(default_factory=list)
    signals: list[int] = field(default_factory=list)
    detections: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    weather: dict[str, float] | None = None

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tick": self.tick,
            "simTime": round(self.sim_time, 3),
            "vehicles": [v.to_wire() for v in self.vehicles],
            "obstacles": [o.to_wire() for o in self.obstacles],
        }
        if self.pedestrians:
            payload["pedestrians"] = [p.to_wire() for p in self.pedestrians]
        if self.weather is not None:
            payload["weather"] = {k: round(v, 3) for k, v in self.weather.items()}
        if self.signals:
            payload["signals"] = self.signals
        if self.detections:
            payload["detections"] = {
                str(slot): dets for slot, dets in self.detections.items()
            }
        return payload


def validate_hidden_sizes(value: Any) -> tuple[list[int] | None, str]:
    """`set_network` の hiddenSizes を検証する。"""
    if not isinstance(value, (list, tuple)):
        return None, "hiddenSizes は数値の配列で指定してください"
    if not (config.PPO_HIDDEN_MIN_LAYERS <= len(value) <= config.PPO_HIDDEN_MAX_LAYERS):
        return None, (
            f"隠れ層の数は {config.PPO_HIDDEN_MIN_LAYERS}〜"
            f"{config.PPO_HIDDEN_MAX_LAYERS} 層にしてください（指定は {len(value)} 層）"
        )
    out: list[int] = []
    for i, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None, f"{i + 1} 層目が数値ではありません"
        if not math.isfinite(float(raw)):
            return None, f"{i + 1} 層目が有限の数値ではありません"
        width = int(raw)
        if not (config.PPO_HIDDEN_MIN_WIDTH <= width <= config.PPO_HIDDEN_MAX_WIDTH):
            return None, (
                f"{i + 1} 層目の幅は {config.PPO_HIDDEN_MIN_WIDTH}〜"
                f"{config.PPO_HIDDEN_MAX_WIDTH} にしてください（指定は {width}）"
            )
        out.append(width)
    return out, ""


@dataclass
class MetricsSnapshot:
    """docs/protocol.md 2.6 の metrics に対応。"""

    tick: int = 0
    wall_time: float = 0.0
    updates: int = 0
    episodes: int = 0
    mean_episode_reward: float = 0.0
    mean_episode_length: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    collision_rate: float = 0.0
    #: そのうち歩行者に当たった割合（`collision_rate` にも含まれる）
    pedestrian_collision_rate: float = 0.0
    goal_rate: float = 0.0
    steps_per_sec: float = 0.0
    signal_violations: float = 0.0
    speed_violations: float = 0.0
    lane_deviation: float = 0.0

    def to_wire(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "wallTime": round(self.wall_time, 2),
            "updates": self.updates,
            "episodes": self.episodes,
            "meanEpisodeReward": round(self.mean_episode_reward, 3),
            "meanEpisodeLength": round(self.mean_episode_length, 1),
            "policyLoss": round(self.policy_loss, 5),
            "valueLoss": round(self.value_loss, 5),
            "entropy": round(self.entropy, 4),
            "approxKl": round(self.approx_kl, 5),
            "collisionRate": round(self.collision_rate, 3),
            "pedestrianCollisionRate": round(self.pedestrian_collision_rate, 3),
            "goalRate": round(self.goal_rate, 3),
            "stepsPerSec": round(self.steps_per_sec, 2),
            "signalViolations": round(self.signal_violations, 3),
            "speedViolations": round(self.speed_violations, 3),
            "laneDeviation": round(self.lane_deviation, 3),
        }


@dataclass
class EpisodeResult:
    """1 台分のエピソードが終了したときの記録。"""

    slot: int
    total_reward: float
    length: int
    reason: str
    signal_violations: int = 0
    speed_violations: int = 0
    lane_deviation: float = 0.0
    lane_departures: int = 0
    #: 終了の原因が衝突のとき、相手が歩行者だったか
    hit_pedestrian: bool = False


@dataclass
class StepResult:
    """SimulationEnv.step() の戻り値。"""

    obs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    active: np.ndarray

    truncated: np.ndarray | None = None
    episodes: list[EpisodeResult] = field(default_factory=list)


@dataclass
class InterventionEvent:
    """memo 5章「介入も現実の交通現象の一部」。学習を止めずステップ境界で適用する。"""

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


TAXI_PHASE_IDLE = "idle"
TAXI_PHASE_APPROACHING = "approaching"
TAXI_PHASE_WAITING = "waiting"
TAXI_PHASE_RIDING = "riding"
TAXI_PHASE_ARRIVED = "arrived"

TAXI_PHASES: tuple[str, ...] = (
    TAXI_PHASE_IDLE,
    TAXI_PHASE_APPROACHING,
    TAXI_PHASE_WAITING,
    TAXI_PHASE_RIDING,
    TAXI_PHASE_ARRIVED,
)


def _point_wire(point: tuple[float, float] | None) -> list[float] | None:
    return None if point is None else [round(point[0], 3), round(point[1], 3)]


@dataclass
class TaxiStatus:
    """docs/protocol.md 2.10 の taxi に対応。実用モードの配車 1 件ぶんの状態。"""

    phase: str = TAXI_PHASE_IDLE
    vehicle_id: int = -1
    pickup: tuple[float, float] | None = None
    dropoff: tuple[float, float] | None = None
    route: list[tuple[float, float]] = field(default_factory=list)
    route_revision: int = 0
    eta_seconds: float = 0.0
    remaining_distance_m: float = 0.0
    message: str = ""

    def to_wire(self, *, include_route: bool = False) -> dict[str, Any]:
        """`route` は数百点になるので、版が変わったときだけ載せる（frame と同じ約束）。"""
        payload: dict[str, Any] = {
            "phase": self.phase,
            "vehicleId": self.vehicle_id,
            "pickup": _point_wire(self.pickup),
            "dropoff": _point_wire(self.dropoff),
            "routeRevision": self.route_revision,
            "etaSeconds": round(self.eta_seconds, 1),
            "remainingDistanceM": round(self.remaining_distance_m, 1),
            "message": self.message,
        }
        if include_route:
            payload["route"] = [[round(px, 2), round(py, 2)] for px, py in self.route]
        return payload
