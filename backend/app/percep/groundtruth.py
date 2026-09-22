# -*- coding: utf-8 -*-
"""world の真値から「理想の検出結果」を作る。"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from app import config
from app.percep.geometry import CameraPose, camera_pose, project_points
from app.percep.types import (
    CLASS_QUOTA,
    DEFAULT_CAMERA,
    LANE_LOOKAHEAD_M,
    LANE_POLYLINE_POINTS,
    PEDESTRIAN_HALF_WIDTH,
    PEDESTRIAN_HEIGHT,
    SIGN_BOARD_Z,
    SIGN_RADIUS,
    SIGNAL_BEYOND_MARGIN,
    SIGNAL_HEAD_Z,
    SIGNAL_HOUSING_H,
    SIGNAL_HOUSING_W,
    CameraSpec,
    DetClass,
    Detection,
    PerceptionResult,
    facing_viewer,
    pack_by_class_quota,
)
from app.percep.weather import CLEAR, Weather

if TYPE_CHECKING:
    from app.sim.world import World


__all__ = [
    "clear_static_cache",
    "detect_ground_truth",
    "detect_ground_truth_batch",
    "freespace_ground_truth",
]

logger = logging.getLogger("autoware_sim")

_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """同じ失敗を初回だけログに残す（code_review B-15 / E-01 / E-02）。"""
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.exception(message)


MIN_BOX_PX = 1.5

LANE_HALF_WIDTH_M = 1.6
LANE_SAMPLE_M = 2.5

OCCLUSION_STEP_M = 2.0
OCCLUSION_MAX_SAMPLES = 64

FREESPACE_ANGLES = np.linspace(
    -math.pi / 2.0, math.pi / 2.0, config.OBS_FREESPACE_DIM, dtype=np.float32
)

VEHICLE_BLOCK_RADIUS = (config.VEHICLE_LENGTH + config.VEHICLE_WIDTH) * 0.25


def _box_from_points(
    pose: CameraPose,
    spec: CameraSpec,
    points: np.ndarray,
    *,
    require_all_in_front: bool = True,
) -> tuple[float, float, float, float] | None:
    """点群を囲む正規化 BBox を返す。視野外・背後・小さすぎるものは None。"""
    u, v, depth = project_points(pose, spec, points)
    front = depth > spec.near
    if require_all_in_front:
        if not bool(front.all()):
            return None
    else:
        if not bool(front.any()):
            return None
        u, v = u[front], v[front]

    x0 = float(np.min(u))
    x1 = float(np.max(u))
    y0 = float(np.min(v))
    y1 = float(np.max(v))

    if x1 <= 0.0 or y1 <= 0.0 or x0 >= spec.width or y0 >= spec.height:
        return None
    if (x1 - x0) < MIN_BOX_PX and (y1 - y0) < MIN_BOX_PX:
        return None

    x0 = min(max(x0, 0.0), float(spec.width))
    x1 = min(max(x1, 0.0), float(spec.width))
    y0 = min(max(y0, 0.0), float(spec.height))
    y1 = min(max(y1, 0.0), float(spec.height))
    if x1 - x0 <= 0.0 or y1 - y0 <= 0.0:
        return None
    return (x0 / spec.width, y0 / spec.height, x1 / spec.width, y1 / spec.height)


@dataclass(frozen=True)
class _StaticScene:
    """信号・標識の 3D 位置をあらかじめ配列にしたもの。"""

    signal_head: np.ndarray
    signal_across: np.ndarray
    signal_heading: np.ndarray
    signal_stop: np.ndarray
    sign_board: np.ndarray
    sign_across: np.ndarray
    sign_heading: np.ndarray
    sign_limit: np.ndarray


_STATIC_CACHE: tuple[object, _StaticScene] | None = None


def clear_static_cache() -> None:
    """静的シーンのキャッシュを捨てる（テストとマップ入れ替えの検証用）。"""
    global _STATIC_CACHE
    _STATIC_CACHE = None


def _build_static_scene(map_index) -> _StaticScene:
    """`MapData` から信号・標識の 3D 形状を組み立てる。"""
    data = getattr(map_index, "data", None)
    signals = list(getattr(data, "signals", []) or [])
    signs = list(getattr(data, "signs", []) or [])
    nodes = list(getattr(data, "nodes", []) or [])
    node_xy = {int(n.id): (float(n.x), float(n.y)) for n in nodes}

    head = np.zeros((len(signals), 3), dtype=np.float64)
    across_sig = np.zeros((len(signals), 2), dtype=np.float64)
    heading_sig = np.zeros(len(signals), dtype=np.float64)
    stop = np.zeros((len(signals), 2), dtype=np.float64)
    for i, sig in enumerate(signals):
        cx, cy = node_xy.get(int(sig.node_id), (float(sig.x), float(sig.y)))
        cos_h = math.cos(float(sig.heading))
        sin_h = math.sin(float(sig.heading))
        left_x, left_y = -sin_h, cos_h
        beyond = float(sig.road_width) * 0.5 + SIGNAL_BEYOND_MARGIN
        offset = float(sig.road_width) * 0.25
        head[i, 0] = cx + cos_h * beyond + left_x * offset
        head[i, 1] = cy + sin_h * beyond + left_y * offset
        head[i, 2] = SIGNAL_HEAD_Z
        across_sig[i] = (left_x, left_y)
        heading_sig[i] = float(sig.heading)
        stop[i] = (float(sig.x), float(sig.y))

    board = np.zeros((len(signs), 3), dtype=np.float64)
    across_sgn = np.zeros((len(signs), 2), dtype=np.float64)
    heading_sgn = np.zeros(len(signs), dtype=np.float64)
    limit = np.zeros(len(signs), dtype=np.float64)
    for i, sgn in enumerate(signs):
        cos_h = math.cos(float(sgn.heading))
        sin_h = math.sin(float(sgn.heading))
        board[i] = (float(sgn.x), float(sgn.y), SIGN_BOARD_Z)
        across_sgn[i] = (-sin_h, cos_h)
        heading_sgn[i] = float(sgn.heading)
        limit[i] = float(sgn.speed_limit)

    return _StaticScene(
        signal_head=head,
        signal_across=across_sig,
        signal_heading=heading_sig,
        signal_stop=stop,
        sign_board=board,
        sign_across=across_sgn,
        sign_heading=heading_sgn,
        sign_limit=limit,
    )


def _static_scene(map_index) -> _StaticScene:
    global _STATIC_CACHE
    cached = _STATIC_CACHE
    if cached is not None and cached[0] is map_index:
        return cached[1]
    scene = _build_static_scene(map_index)
    _STATIC_CACHE = (map_index, scene)
    return scene


def _blocked_by_fleet(
    world: "World",
    eye: tuple[float, float],
    targets: np.ndarray,
    viewer: int,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """視線が**他車の車体**で遮られているターゲットを True で返す。"""
    t = int(targets.shape[0])
    if t == 0:
        return np.zeros(0, dtype=bool)
    fleet = world.fleet
    occ = np.flatnonzero(fleet.active)
    occ = occ[occ != int(viewer)]
    if occ.size == 0:
        return np.zeros(t, dtype=bool)

    corners = fleet.corners()[occ].astype(np.float64)
    q1 = corners.reshape(-1, 2)
    q2 = np.roll(corners, -1, axis=1).reshape(-1, 2)
    owner = np.repeat(occ.astype(np.int64), 4)

    p1 = np.array(eye, dtype=np.float64)
    d1 = targets.astype(np.float64) - p1
    d2 = q2 - q1

    cross = d1[:, None, 0] * d2[None, :, 1] - d1[:, None, 1] * d2[None, :, 0]
    diff = q1[None, :, :] - p1[None, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = (diff[..., 0] * d2[None, :, 1] - diff[..., 1] * d2[None, :, 0]) / cross
        uu = (diff[..., 0] * d1[:, None, 1] - diff[..., 1] * d1[:, None, 0]) / cross
    eps = 1e-6
    hit = np.isfinite(tt) & np.isfinite(uu)
    hit &= (tt > eps) & (tt < 1.0 - eps) & (uu >= 0.0) & (uu <= 1.0)
    if exclude is not None:
        hit &= owner[None, :] != exclude[:, None]
    return hit.any(axis=1)


def _line_of_sight(map_index, eye: tuple[float, float], targets: np.ndarray) -> np.ndarray:
    """視線が建物を貫いていないターゲットを True で返す。shape (M,)。"""
    m = int(targets.shape[0])
    if m == 0:
        return np.zeros(0, dtype=bool)
    grid = getattr(map_index, "occupancy", None)
    if grid is None:
        return np.ones(m, dtype=bool)

    dx = targets[:, 0] - eye[0]
    dy = targets[:, 1] - eye[1]
    dist = np.hypot(dx, dy)
    steps = int(min(OCCLUSION_MAX_SAMPLES, max(2, math.ceil(float(np.max(dist)) / OCCLUSION_STEP_M))))
    t = np.linspace(0.0, 1.0, steps + 2, dtype=np.float64)[1:-1]
    sample_x = eye[0] + dx[:, None] * t[None, :]
    sample_y = eye[1] + dy[:, None] * t[None, :]
    try:
        blocked = grid.sample_building(sample_x, sample_y)
    except Exception:
        _warn_once(
            "line_of_sight",
            "遮蔽判定に失敗しました。以後この画では遮蔽を考えず、"
            "建物の裏の物体にも正解ラベルが付きます（初回のみ記録）",
        )
        return np.ones(m, dtype=bool)
    return ~np.asarray(blocked, dtype=bool).reshape(m, -1).any(axis=1)


def _detect_signals(
    world: "World",
    slot: int,
    spec: CameraSpec,
    pose: CameraPose,
    scene: _StaticScene,
    max_distance: float,
) -> list[tuple[float, Detection]]:
    """前方の信号機。灯色は `world.signal_phases` の真値をそのまま入れる。"""
    out: list[tuple[float, Detection]] = []
    head = scene.signal_head
    if head.shape[0] == 0:
        return out

    eye = (pose.eye_x, pose.eye_y)
    dx = scene.signal_stop[:, 0] - eye[0]
    dy = scene.signal_stop[:, 1] - eye[1]
    dist = np.hypot(dx, dy)
    heading = float(world.fleet.heading[slot])
    facing = facing_viewer(
        scene.signal_head[:, 0],
        scene.signal_head[:, 1],
        scene.signal_heading,
        eye[0],
        eye[1],
        heading,
    )
    candidates = np.flatnonzero((dist <= max_distance) & facing)
    if candidates.size == 0:
        return out

    visible = _line_of_sight(world.map_index, eye, scene.signal_stop[candidates])
    candidates = candidates[visible]

    phases = world.signal_phases
    half_w = SIGNAL_HOUSING_W * 0.5
    half_h = SIGNAL_HOUSING_H * 0.5
    for i in candidates:
        center = head[i]
        across = scene.signal_across[i]
        corners = np.array(
            [
                [center[0] - across[0] * half_w, center[1] - across[1] * half_w, center[2] - half_h],
                [center[0] + across[0] * half_w, center[1] + across[1] * half_w, center[2] - half_h],
                [center[0] - across[0] * half_w, center[1] - across[1] * half_w, center[2] + half_h],
                [center[0] + across[0] * half_w, center[1] + across[1] * half_w, center[2] + half_h],
            ],
            dtype=np.float64,
        )
        box = _box_from_points(pose, spec, corners)
        if box is None:
            continue
        idx = int(i)
        phase = int(phases[idx]) if 0 <= idx < len(phases) else 2
        out.append(
            (
                float(dist[idx]),
                Detection(
                    cls=DetClass.TRAFFIC_LIGHT,
                    x0=box[0], y0=box[1], x1=box[2], y1=box[3],
                    confidence=1.0,
                    phase=phase,
                    distance=float(dist[idx]),
                ),
            )
        )
    return out


def _detect_speed_signs(
    world: "World",
    slot: int,
    spec: CameraSpec,
    pose: CameraPose,
    scene: _StaticScene,
    max_distance: float,
) -> list[tuple[float, Detection]]:
    """最高速度標識。規制速度は `MapSign.speed_limit` の真値。"""
    out: list[tuple[float, Detection]] = []
    board = scene.sign_board
    if board.shape[0] == 0:
        return out

    eye = (pose.eye_x, pose.eye_y)
    dx = board[:, 0] - eye[0]
    dy = board[:, 1] - eye[1]
    dist = np.hypot(dx, dy)
    heading = float(world.fleet.heading[slot])
    facing = facing_viewer(
        board[:, 0], board[:, 1], scene.sign_heading, eye[0], eye[1], heading
    )
    candidates = np.flatnonzero((dist <= max_distance) & facing)
    if candidates.size == 0:
        return out

    visible = _line_of_sight(world.map_index, eye, board[candidates, :2])
    candidates = candidates[visible]

    for i in candidates:
        center = board[i]
        across = scene.sign_across[i]
        corners = np.array(
            [
                [center[0] - across[0] * SIGN_RADIUS, center[1] - across[1] * SIGN_RADIUS, center[2] - SIGN_RADIUS],
                [center[0] + across[0] * SIGN_RADIUS, center[1] + across[1] * SIGN_RADIUS, center[2] - SIGN_RADIUS],
                [center[0] - across[0] * SIGN_RADIUS, center[1] - across[1] * SIGN_RADIUS, center[2] + SIGN_RADIUS],
                [center[0] + across[0] * SIGN_RADIUS, center[1] + across[1] * SIGN_RADIUS, center[2] + SIGN_RADIUS],
            ],
            dtype=np.float64,
        )
        box = _box_from_points(pose, spec, corners)
        if box is None:
            continue
        idx = int(i)
        out.append(
            (
                float(dist[idx]),
                Detection(
                    cls=DetClass.SPEED_SIGN,
                    x0=box[0], y0=box[1], x1=box[2], y1=box[3],
                    confidence=1.0,
                    speed_limit=float(scene.sign_limit[idx]),
                    distance=float(dist[idx]),
                ),
            )
        )
    return out


def _detect_vehicles(
    world: "World", slot: int, spec: CameraSpec, pose: CameraPose, max_distance: float
) -> list[tuple[float, Detection]]:
    """他車両。車体の 8 頂点（路面と屋根の 4 隅）を投影して箱にする。"""
    out: list[tuple[float, Detection]] = []
    fleet = world.fleet
    others = np.flatnonzero(fleet.active)
    others = others[others != int(slot)]
    if others.size == 0:
        return out

    eye = (pose.eye_x, pose.eye_y)
    dx = fleet.x[others].astype(np.float64) - eye[0]
    dy = fleet.y[others].astype(np.float64) - eye[1]
    dist = np.hypot(dx, dy)
    near = others[dist <= max_distance]
    if near.size == 0:
        return out

    visible = _line_of_sight(
        world.map_index,
        eye,
        np.column_stack((fleet.x[near], fleet.y[near])).astype(np.float64),
    )
    near = near[visible]
    if near.size == 0:
        return out

    centers = np.column_stack((fleet.x[near], fleet.y[near])).astype(np.float64)
    near = near[~_blocked_by_fleet(world, eye, centers, slot, exclude=near.astype(np.int64))]
    if near.size == 0:
        return out

    corners_all = fleet.corners()
    height = float(config.VEHICLE_HEIGHT)
    for other in near:
        flat = corners_all[int(other)].astype(np.float64)
        pts = np.empty((8, 3), dtype=np.float64)
        pts[:4, :2] = flat
        pts[:4, 2] = 0.0
        pts[4:, :2] = flat
        pts[4:, 2] = height
        box = _box_from_points(pose, spec, pts, require_all_in_front=False)
        if box is None:
            continue
        d = float(math.hypot(float(fleet.x[other]) - eye[0], float(fleet.y[other]) - eye[1]))
        out.append(
            (
                d,
                Detection(
                    cls=DetClass.VEHICLE,
                    x0=box[0], y0=box[1], x1=box[2], y1=box[3],
                    confidence=1.0,
                    distance=d,
                ),
            )
        )
    return out


def _detect_obstacles(
    world: "World", slot: int, spec: CameraSpec, pose: CameraPose, max_distance: float
) -> list[tuple[float, Detection]]:
    """ユーザーが置いたパイロン。円柱を視線に垂直な板で近似する。"""
    out: list[tuple[float, Detection]] = []
    xy = world.obstacle_xy
    if xy.shape[0] == 0:
        return out

    eye = (pose.eye_x, pose.eye_y)
    dx = xy[:, 0].astype(np.float64) - eye[0]
    dy = xy[:, 1].astype(np.float64) - eye[1]
    dist = np.hypot(dx, dy)
    height = float(config.OBSTACLE_HEIGHT)

    near = np.flatnonzero(dist <= max_distance)
    if near.size == 0:
        return out
    near = near[_line_of_sight(world.map_index, eye, xy[near].astype(np.float64))]
    if near.size:
        pts = xy[near].astype(np.float64)
        near = near[~_blocked_by_fleet(world, eye, pts, slot)]

    for i in near:
        i = int(i)
        d = float(dist[i])
        if d < 1e-3:
            continue
        radius = float(world.obstacles[i].radius) if i < len(world.obstacles) else config.OBSTACLE_RADIUS
        ax, ay = -dy[i] / d, dx[i] / d
        cx, cy = float(xy[i, 0]), float(xy[i, 1])
        pts = np.array(
            [
                [cx - ax * radius, cy - ay * radius, 0.0],
                [cx + ax * radius, cy + ay * radius, 0.0],
                [cx - ax * radius, cy - ay * radius, height],
                [cx + ax * radius, cy + ay * radius, height],
            ],
            dtype=np.float64,
        )
        box = _box_from_points(pose, spec, pts, require_all_in_front=False)
        if box is None:
            continue
        out.append(
            (
                d,
                Detection(
                    cls=DetClass.OBSTACLE,
                    x0=box[0], y0=box[1], x1=box[2], y1=box[3],
                    confidence=1.0,
                    distance=d,
                ),
            )
        )
    return out


def _detect_pedestrians(
    world: "World", slot: int, spec: CameraSpec, pose: CameraPose, max_distance: float
) -> list[tuple[float, Detection]]:
    """NPC 歩行者。視線に垂直な板（肩幅 × 身長）へ近似して箱にする。"""
    out: list[tuple[float, Detection]] = []
    xy = world.pedestrian_xy
    if xy.shape[0] == 0:
        return out

    eye = (pose.eye_x, pose.eye_y)
    dx = xy[:, 0] - eye[0]
    dy = xy[:, 1] - eye[1]
    dist = np.hypot(dx, dy)

    near = np.flatnonzero(dist <= max_distance)
    if near.size == 0:
        return out
    near = near[_line_of_sight(world.map_index, eye, xy[near])]
    if near.size:
        near = near[~_blocked_by_fleet(world, eye, xy[near], slot)]

    for i in near:
        i = int(i)
        d = float(dist[i])
        if d < 1e-3:
            continue
        ax, ay = -dy[i] / d, dx[i] / d
        cx, cy = float(xy[i, 0]), float(xy[i, 1])
        pts = np.array(
            [
                [cx - ax * PEDESTRIAN_HALF_WIDTH, cy - ay * PEDESTRIAN_HALF_WIDTH, 0.0],
                [cx + ax * PEDESTRIAN_HALF_WIDTH, cy + ay * PEDESTRIAN_HALF_WIDTH, 0.0],
                [cx - ax * PEDESTRIAN_HALF_WIDTH, cy - ay * PEDESTRIAN_HALF_WIDTH, PEDESTRIAN_HEIGHT],
                [cx + ax * PEDESTRIAN_HALF_WIDTH, cy + ay * PEDESTRIAN_HALF_WIDTH, PEDESTRIAN_HEIGHT],
            ],
            dtype=np.float64,
        )
        box = _box_from_points(pose, spec, pts, require_all_in_front=False)
        if box is None:
            continue
        out.append(
            (
                d,
                Detection(
                    cls=DetClass.PEDESTRIAN,
                    x0=box[0], y0=box[1], x1=box[2], y1=box[3],
                    confidence=1.0,
                    distance=d,
                ),
            )
        )
    return out


def _detect_lane(
    world: "World", slot: int, spec: CameraSpec, pose: CameraPose, max_distance: float
) -> tuple[float, Detection] | None:
    """走行車線。経路（＝車線中心線）への射影から作る。"""
    state = world.slots[int(slot)]
    route = state.route
    if route.shape[0] < 2:
        return None
    cum = state.route_cum
    arc = float(world.arc[int(slot)])

    lookahead = min(float(LANE_LOOKAHEAD_M), float(max_distance))
    if lookahead < LANE_SAMPLE_M * 2.0:
        return None
    offsets = np.arange(0.0, lookahead + 1e-6, LANE_SAMPLE_M, dtype=np.float64)
    targets = arc + offsets
    usable = int(np.count_nonzero(targets <= cum[-1]))
    if usable < 2:
        return None
    targets = targets[:usable]
    cx = np.interp(targets, cum, route[:, 0])
    cy = np.interp(targets, cum, route[:, 1])
    if cx.size < 2:
        return None

    tx = np.gradient(cx)
    ty = np.gradient(cy)
    norm = np.maximum(np.hypot(tx, ty), 1e-6)
    lx, ly = -ty / norm, tx / norm
    left = np.stack([cx + lx * LANE_HALF_WIDTH_M, cy + ly * LANE_HALF_WIDTH_M], axis=1)
    right = np.stack([cx - lx * LANE_HALF_WIDTH_M, cy - ly * LANE_HALF_WIDTH_M], axis=1)
    pts = np.concatenate([left, right], axis=0)
    pts3 = np.concatenate([pts, np.zeros((pts.shape[0], 1), dtype=np.float64)], axis=1)

    box = _box_from_points(pose, spec, pts3, require_all_in_front=False)
    if box is None:
        return None

    _, _, depth = project_points(pose, spec, pts3)
    visible_depth = float(np.max(depth[depth > spec.near])) if np.any(depth > spec.near) else 0.0

    fleet = world.fleet
    ox = float(fleet.x[int(slot)])
    oy = float(fleet.y[int(slot)])
    heading = float(fleet.heading[int(slot)])
    cos_h = float(np.cos(heading))
    sin_h = float(np.sin(heading))
    stride = max(1, int(round(cx.size / max(LANE_POLYLINE_POINTS, 1))))
    keep = np.arange(0, cx.size, stride)
    if keep[-1] != cx.size - 1:
        keep = np.append(keep, cx.size - 1)
    ddx = cx[keep] - ox
    ddy = cy[keep] - oy
    lane_points = [
        (float(fx), float(fy))
        for fx, fy in zip(ddx * cos_h + ddy * sin_h, -ddx * sin_h + ddy * cos_h)
    ]

    return (
        0.0,
        Detection(
            cls=DetClass.LANE,
            x0=box[0], y0=box[1], x1=box[2], y1=box[3],
            confidence=1.0,
            distance=min(visible_depth, float(max_distance)),
            lateral=float(world.lateral[int(slot)]),
            lane_points=lane_points,
        ),
    )


def detect_ground_truth(
    world: "World",
    slot: int,
    spec: CameraSpec = DEFAULT_CAMERA,
    weather: Weather = CLEAR,
) -> PerceptionResult:
    """world の真値から「理想の検出結果」を作る。"""
    slot = int(slot)
    result = PerceptionResult(slot=slot)
    if not (0 <= slot < len(world.slots)):
        return result
    if not bool(world.fleet.active[slot]):
        return result

    pose = camera_pose(
        float(world.fleet.x[slot]),
        float(world.fleet.y[slot]),
        float(world.fleet.heading[slot]),
        spec,
    )
    scene = _static_scene(world.map_index)
    reach = weather.visibility_m(float(spec.far))

    per_class: dict[DetClass, list[Detection]] = {c: [] for c in DetClass}
    buckets: list[tuple[DetClass, list[tuple[float, Detection]]]] = [
        (DetClass.TRAFFIC_LIGHT, _detect_signals(world, slot, spec, pose, scene, reach)),
        (DetClass.SPEED_SIGN, _detect_speed_signs(world, slot, spec, pose, scene, reach)),
        (DetClass.VEHICLE, _detect_vehicles(world, slot, spec, pose, reach)),
        (DetClass.OBSTACLE, _detect_obstacles(world, slot, spec, pose, reach)),
        (DetClass.PEDESTRIAN, _detect_pedestrians(world, slot, spec, pose, reach)),
    ]
    lane = _detect_lane(world, slot, spec, pose, reach)
    if lane is not None:
        per_class[DetClass.LANE] = [lane[1]]
    for cls, items in buckets:
        items.sort(key=lambda pair: pair[0])
        per_class[cls] = [det for _, det in items[: CLASS_QUOTA[cls]]]

    result.detections = pack_by_class_quota(
        per_class, int(config.PERCEP_MAX_DETECTIONS), spec
    )
    return result


def detect_ground_truth_batch(
    world: "World",
    slots: Sequence[int],
    spec: CameraSpec = DEFAULT_CAMERA,
    weather: Weather = CLEAR,
) -> list[PerceptionResult]:
    """複数スロットぶんまとめて作る（教師データ収集用の薄いラッパ）。"""
    return [detect_ground_truth(world, int(s), spec, weather) for s in slots]


def freespace_ground_truth(
    world: "World",
    slot: int,
    spec: CameraSpec = DEFAULT_CAMERA,
    max_distance: float = float(config.OBS_FREESPACE_MAX_DISTANCE),
) -> np.ndarray:
    """走行可能領域の真値。前方 ±90 度を `OBS_FREESPACE_DIM` 本に分けた距離 [m]。"""
    slot = int(slot)
    out = np.full(config.OBS_FREESPACE_DIM, float(max_distance), dtype=np.float32)
    if not (0 <= slot < len(world.slots)) or not bool(world.fleet.active[slot]):
        return out

    heading = float(world.fleet.heading[slot])
    pose = camera_pose(
        float(world.fleet.x[slot]), float(world.fleet.y[slot]), heading, spec
    )
    ox, oy = pose.eye_x, pose.eye_y
    angles = (heading + FREESPACE_ANGLES).astype(np.float64)

    try:
        hit = world.map_index.raycast(
            np.array([ox], dtype=np.float32),
            np.array([oy], dtype=np.float32),
            angles.reshape(1, -1).astype(np.float32),
            float(max_distance),
            1.0,
        )
        out[:] = np.asarray(hit, dtype=np.float32).reshape(-1)
    except Exception:
        _warn_once(
            "freespace_raycast",
            "走行可能距離のレイキャストに失敗しました。全方向が "
            "max_distance（＝前方はすべて空いている）のまま観測と教師データに入ります"
            "（初回のみ記録）",
        )

    blockers: list[tuple[float, float, float]] = []
    fleet = world.fleet
    for other in np.flatnonzero(fleet.active):
        if int(other) == slot:
            continue
        blockers.append(
            (float(fleet.x[other]), float(fleet.y[other]), VEHICLE_BLOCK_RADIUS)
        )
    for obstacle in world.obstacles:
        blockers.append((float(obstacle.x), float(obstacle.y), float(obstacle.radius)))
    for px, py in world.pedestrian_xy:
        blockers.append((float(px), float(py), float(config.PEDESTRIAN_RADIUS)))
    if blockers:
        circles = np.asarray(blockers, dtype=np.float64)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        rel = circles[None, :, :2] - np.array([[ox, oy]], dtype=np.float64)[:, None, :]
        along = rel[..., 0] * dirs[:, None, 0] + rel[..., 1] * dirs[:, None, 1]
        perp2 = (rel * rel).sum(axis=2) - along * along
        radius2 = circles[None, :, 2] ** 2
        crosses = (along > 0.0) & (perp2 <= radius2)
        if crosses.any():
            back = np.sqrt(np.maximum(radius2 - perp2, 0.0))
            entry = np.where(crosses, np.maximum(along - back, 0.0), np.inf)
            out[:] = np.minimum(out, entry.min(axis=1).astype(np.float32))

    return np.clip(out, 0.0, float(max_distance)).astype(np.float32)
