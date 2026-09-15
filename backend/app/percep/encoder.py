"""検出結果とナビ情報から PPO の観測ベクトルを組み立てる。"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

from app import config
from app.contracts import SimParams
from app.percep.types import DEFAULT_CAMERA, CameraSpec, DetClass, Detection, PerceptionResult

if TYPE_CHECKING:
    from app.sim.world import World


__all__ = ["OBS_OFFSETS", "encode_observations"]

_OFF_SELF = 0
_OFF_GOAL = _OFF_SELF + config.OBS_SELF_DIM
_OFF_ROUTE = _OFF_GOAL + config.OBS_GOAL_DIM
_OFF_LANE = _OFF_ROUTE + config.OBS_ROUTE_DIM
_OFF_SIGNAL = _OFF_LANE + config.OBS_LANE_DIM
_OFF_SIGN = _OFF_SIGNAL + config.OBS_SIGNAL_DIM
_OFF_VEHICLE = _OFF_SIGN + config.OBS_SIGN_DIM
_OFF_OBSTACLE = _OFF_VEHICLE + config.OBS_VEHICLE_DIM
_OFF_FREESPACE = _OFF_OBSTACLE + config.OBS_OBSTACLE_DIM
assert (
    _OFF_FREESPACE + config.OBS_FREESPACE_DIM == config.OBS_DIM
), "OBS_* の内訳が OBS_DIM と一致しない"

OBS_OFFSETS: dict[str, int] = {
    "self": _OFF_SELF,
    "goal": _OFF_GOAL,
    "route": _OFF_ROUTE,
    "lane": _OFF_LANE,
    "signal": _OFF_SIGNAL,
    "sign": _OFF_SIGN,
    "vehicles": _OFF_VEHICLE,
    "obstacles": _OFF_OBSTACLE,
    "freespace": _OFF_FREESPACE,
}

_ROUTE_OFFSETS = np.arange(1, config.OBS_ROUTE_POINTS + 1, dtype=np.float32) * np.float32(
    config.OBS_ROUTE_SPACING
)
_ROUTE_SCALE = np.float32(config.OBS_ROUTE_SPACING * config.OBS_ROUTE_POINTS)

_FREESPACE_ANGLES = np.linspace(
    -math.pi / 2.0, math.pi / 2.0, config.OBS_FREESPACE_DIM, dtype=np.float32
)
_FREESPACE_HALF = float(math.pi / max(config.OBS_FREESPACE_DIM - 1, 1)) * 0.5

_ASSUMED_WIDTH_M: dict[DetClass, float] = {
    DetClass.TRAFFIC_LIGHT: 1.2,
    DetClass.SPEED_SIGN: float(config.SPEED_SIGN_DIAMETER),
    DetClass.VEHICLE: float(config.VEHICLE_WIDTH),
    DetClass.OBSTACLE: float(config.OBSTACLE_RADIUS) * 2.0,
}


def _finite(value: float | None) -> float | None:
    """有限な値だけを通す。NaN/inf は「無かった」ことにする。"""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _iter_class(result: PerceptionResult, cls: DetClass) -> Iterator[Detection]:
    """そのクラスの検出を信頼度の降順（＝格納順）で返す。"""
    for det in result.detections:
        if det.cls == cls:
            yield det


def _best(result: PerceptionResult, cls: DetClass) -> Detection | None:
    return next(_iter_class(result, cls), None)


def _bearing(det: Detection, spec: CameraSpec) -> float:
    """検出の中心方向を自車座標系の方位 [rad]（左が正）で返す。"""
    cx = (float(det.x0) + float(det.x1)) * 0.5
    if not math.isfinite(cx):
        return 0.0
    return -math.atan2((cx - 0.5) * spec.width, spec.focal_px)


def _distance(det: Detection, spec: CameraSpec) -> float:
    """検出までの推定距離 [m]。`near`〜`far` に収める。"""
    given = _finite(det.distance)
    if given is not None and given > 0.0:
        return min(max(given, float(spec.near)), float(spec.far))

    real = _ASSUMED_WIDTH_M.get(det.cls)
    width_px = (float(det.x1) - float(det.x0)) * spec.width
    if real is None or not math.isfinite(width_px) or width_px <= 1e-3:
        return float(spec.far)
    return min(max(spec.focal_px * real / width_px, float(spec.near)), float(spec.far))


def _confidence(det: Detection) -> float:
    conf = _finite(det.confidence)
    return 0.0 if conf is None else min(max(conf, 0.0), 1.0)


def _local_xy(det: Detection, spec: CameraSpec) -> tuple[float, float, float]:
    """検出の推定位置を自車座標系（前方 +x / 左 +y）の (x, y) と距離で返す。"""
    dist = _distance(det, spec)
    bearing = _bearing(det, spec)
    fx = float(spec.forward) + dist * math.cos(bearing)
    fy = -float(spec.right) + dist * math.sin(bearing)
    return fx, fy, dist


def _pick(
    result: PerceptionResult, cls: DetClass, limit: int, spec: CameraSpec, max_range: float
) -> list[tuple[float, Detection]]:
    """観測に載せる検出を選ぶ。戻り値は (距離, 検出) の**距離昇順**。"""
    picked: list[tuple[float, Detection]] = []
    for det in _iter_class(result, cls):
        dist = _distance(det, spec)
        if dist > max_range:
            continue
        picked.append((dist, det))
        if len(picked) >= limit:
            break
    picked.sort(key=lambda item: item[0])
    return picked


def _freespace_bins(det: Detection, spec: CameraSpec) -> np.ndarray:
    """検出が塞いでいる方位のビンを bool 配列で返す。"""
    left = -math.atan2((float(det.x0) - 0.5) * spec.width, spec.focal_px)
    right = -math.atan2((float(det.x1) - 0.5) * spec.width, spec.focal_px)
    lo, hi = (right, left) if right <= left else (left, right)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return np.zeros(config.OBS_FREESPACE_DIM, dtype=bool)
    mask = (_FREESPACE_ANGLES >= lo - _FREESPACE_HALF) & (
        _FREESPACE_ANGLES <= hi + _FREESPACE_HALF
    )
    if not mask.any():
        mask = np.zeros(config.OBS_FREESPACE_DIM, dtype=bool)
        mask[int(np.argmin(np.abs(_FREESPACE_ANGLES - (lo + hi) * 0.5)))] = True
    return mask


def _encode_camera(
    row: np.ndarray,
    result: PerceptionResult | None,
    spec: CameraSpec,
    speed: float,
    max_speed: float,
    freespace_m: np.ndarray | None = None,
) -> None:
    """観測 1 行のカメラ認識ブロックを埋める（ナビ側の欄には触らない）。"""
    row[_OFF_SIGNAL + 0] = 1.0
    row[_OFF_SIGN + 0] = 1.0
    row[_OFF_SIGN + 1] = min(max(speed / max_speed - 1.0, -1.0), 1.0)
    if freespace_m is None:
        free = np.full(config.OBS_FREESPACE_DIM, float(config.OBS_FREESPACE_MAX_DISTANCE))
    else:
        free = np.clip(
            np.asarray(freespace_m, dtype=np.float64).reshape(config.OBS_FREESPACE_DIM),
            0.0,
            float(config.OBS_FREESPACE_MAX_DISTANCE),
        )

    if result is None or not result.detections:
        row[_OFF_FREESPACE : _OFF_FREESPACE + config.OBS_FREESPACE_DIM] = free / float(
            config.OBS_FREESPACE_MAX_DISTANCE
        )
        return

    lane = _best(result, DetClass.LANE)
    if lane is not None:
        lateral_raw = _finite(lane.lateral)
        lateral = 0.0 if lateral_raw is None else lateral_raw
        head_err = _bearing(lane, spec)
        row[_OFF_LANE + 0] = min(
            max(lateral / float(config.OBS_LATERAL_RANGE), -1.0), 1.0
        )
        row[_OFF_LANE + 1] = math.sin(head_err)
        row[_OFF_LANE + 2] = math.cos(head_err)
        row[_OFF_LANE + 3] = _confidence(lane)

    signal = _best(result, DetClass.TRAFFIC_LIGHT)
    if signal is not None:
        dist = _distance(signal, spec)
        row[_OFF_SIGNAL + 0] = min(dist / float(config.OBS_SIGNAL_RANGE), 1.0)
        phase = signal.phase
        if phase is not None and 0 <= int(phase) <= 2:
            row[_OFF_SIGNAL + 1 + int(phase)] = 1.0
        row[_OFF_SIGNAL + 4] = _confidence(signal)

    sign = _best(result, DetClass.SPEED_SIGN)
    if sign is not None:
        limit = _finite(sign.speed_limit)
        if limit is not None and limit > 0.0:
            row[_OFF_SIGN + 0] = min(limit / max_speed, 1.0)
            row[_OFF_SIGN + 1] = min(max((speed - limit) / max_speed, -1.0), 1.0)
            row[_OFF_SIGN + 2] = _confidence(sign)

    vehicle_range = float(config.OBS_VEHICLE_RANGE)
    for i, (dist, det) in enumerate(
        _pick(result, DetClass.VEHICLE, config.OBS_VEHICLE_COUNT, spec, vehicle_range)
    ):
        base = _OFF_VEHICLE + i * config.OBS_VEHICLE_FIELDS
        fx, fy, _ = _local_xy(det, spec)
        row[base + 0] = min(max(fx / vehicle_range, -1.0), 1.0)
        row[base + 1] = min(max(fy / vehicle_range, -1.0), 1.0)
        row[base + 2] = min(dist / vehicle_range, 1.0)
        row[base + 3] = _confidence(det)

    obstacle_range = float(config.OBS_OBSTACLE_RANGE)
    for i, (_dist, det) in enumerate(
        _pick(result, DetClass.OBSTACLE, config.OBS_OBSTACLE_COUNT, spec, obstacle_range)
    ):
        base = _OFF_OBSTACLE + i * config.OBS_OBSTACLE_FIELDS
        fx, fy, _ = _local_xy(det, spec)
        row[base + 0] = min(max(fx / obstacle_range, -1.0), 1.0)
        row[base + 1] = min(max(fy / obstacle_range, -1.0), 1.0)
        row[base + 2] = _confidence(det)

    for cls in (DetClass.VEHICLE, DetClass.OBSTACLE):
        for det in _iter_class(result, cls):
            dist = _distance(det, spec)
            if dist >= config.OBS_FREESPACE_MAX_DISTANCE:
                continue
            bins = _freespace_bins(det, spec)
            np.minimum(free, dist, out=free, where=bins)
    row[_OFF_FREESPACE : _OFF_FREESPACE + config.OBS_FREESPACE_DIM] = np.clip(
        free, 0.0, config.OBS_FREESPACE_MAX_DISTANCE
    ) / float(config.OBS_FREESPACE_MAX_DISTANCE)


def encode_observations(
    world: "World",
    params: SimParams,
    perceptions: dict[int, PerceptionResult],
    *,
    freespace: dict[int, np.ndarray] | None = None,
    spec: CameraSpec = DEFAULT_CAMERA,
) -> np.ndarray:
    """認識結果とナビ情報から (MAX_VEHICLES, OBS_DIM) float32 を作る。"""
    n = config.MAX_VEHICLES
    obs = np.zeros((n, config.OBS_DIM), dtype=np.float32)

    fleet = world.fleet
    active = fleet.active
    idx = np.flatnonzero(active)
    m = int(idx.size)
    if m == 0:
        return obs
    sel: slice | np.ndarray = slice(None) if m == n else idx

    x = fleet.x[sel]
    y = fleet.y[sel]
    heading = fleet.heading[sel]
    speed = fleet.speed[sel]
    cos_h = np.cos(heading).astype(np.float32)
    sin_h = np.sin(heading).astype(np.float32)
    max_speed = np.float32(max(float(params.max_speed), 1e-3))

    obs[sel, _OFF_SELF + 0] = speed / max_speed
    obs[sel, _OFF_SELF + 1] = fleet.steer[sel] / np.float32(config.MAX_STEER)

    goal_x, goal_y = world.goal_positions()
    gdx = goal_x[sel] - x
    gdy = goal_y[sel] - y
    goal_range = np.float32(config.OBS_GOAL_RANGE)
    obs[sel, _OFF_GOAL + 0] = (gdx * cos_h + gdy * sin_h) / goal_range
    obs[sel, _OFF_GOAL + 1] = (-gdx * sin_h + gdy * cos_h) / goal_range
    obs[sel, _OFF_GOAL + 2] = np.minimum(
        np.hypot(gdx, gdy).astype(np.float32) / goal_range, np.float32(1.0)
    )

    look = world.lookahead_points(_ROUTE_OFFSETS)[sel]
    ldx = look[:, :, 0] - x[:, None]
    ldy = look[:, :, 1] - y[:, None]
    route_local = np.empty_like(look)
    route_local[:, :, 0] = (ldx * cos_h[:, None] + ldy * sin_h[:, None]) / _ROUTE_SCALE
    route_local[:, :, 1] = (-ldx * sin_h[:, None] + ldy * cos_h[:, None]) / _ROUTE_SCALE
    obs[sel, _OFF_ROUTE : _OFF_ROUTE + config.OBS_ROUTE_DIM] = route_local.reshape(
        m, config.OBS_ROUTE_DIM
    )

    speed_all = fleet.speed
    for slot in idx:
        slot = int(slot)
        _encode_camera(
            obs[slot],
            perceptions.get(slot),
            spec,
            float(speed_all[slot]),
            float(max_speed),
            None if freespace is None else freespace.get(slot),
        )

    obs[~active, :] = 0.0
    np.nan_to_num(obs, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    return obs
