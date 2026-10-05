"""4 台のカメラの検出と走行可能距離だけから、自車まわりの見えている所と死角（BEV）を作る。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from app import config
from app.percep.types import (
    CAMERA_RIG,
    DEFAULT_CAMERA,
    LEFT_CAMERA,
    REAR_CAMERA,
    RIGHT_CAMERA,
    SHADOW_DYNAMIC,
    SHADOW_STATIC,
    CameraSpec,
    CameraVisibility,
    DetClass,
    OcclusionResult,
    PerceptionResult,
    ShadowPolygon,
    estimate_distance,
)

__all__ = [
    "CameraInput",
    "HALF_FOV",
    "OCCLUSION_RANGE_M",
    "RAY_ANGLES",
    "RAY_HALF_WIDTH",
    "camera_frustum_polygon",
    "evaluate_occlusion",
    "polygon_area",
    "shadow_polygon",
    "unknown_occlusion",
]

#: 見えている所・死角を考える距離 [m]。走行可能距離が分かるのがここまでなので、4 台とも同じにする
OCCLUSION_RANGE_M = float(config.OBS_FREESPACE_MAX_DISTANCE)
#: 画角を等分する方位の数（1 本あたり約 1.9 度）
RAYS_PER_CAMERA = 35
HALF_FOV = math.radians(DEFAULT_CAMERA.fov_deg) * 0.5
RAY_HALF_WIDTH = HALF_FOV / RAYS_PER_CAMERA
#: カメラの向きから測った各方位の中心 [rad]（左が正）。1 本が [中心 - RAY_HALF_WIDTH, 中心 + RAY_HALF_WIDTH] を受け持つ
RAY_ANGLES = (-HALF_FOV + RAY_HALF_WIDTH * (2.0 * np.arange(RAYS_PER_CAMERA) + 1.0)).astype(np.float64)

#: 走行可能距離の方位（カメラの向きの ±90 度を 9 本）。`percep/encoder.py` と同じ並び
FREESPACE_ANGLES = np.linspace(-math.pi / 2.0, math.pi / 2.0, config.OBS_FREESPACE_DIM)
FREESPACE_HALF = math.pi / max(config.OBS_FREESPACE_DIM - 1, 1) * 0.5
#: 画角（±34 度）に入る走行可能距離の 3 本
FOV_BINS: tuple[int, ...] = tuple(
    int(i) for i in np.flatnonzero(np.abs(FREESPACE_ANGLES) <= HALF_FOV)
)
RAY_BIN = np.array(
    [FOV_BINS[int(np.argmin(np.abs(FREESPACE_ANGLES[list(FOV_BINS)] - a)))] for a in RAY_ANGLES],
    dtype=np.int64,
)

#: 陰を作る検出。パイロン（高さ 0.75m）と人は背後の車や人を隠さない（`groundtruth._blocked_by_fleet` と同じ判断）
SHADOW_CLASSES = frozenset({DetClass.VEHICLE})
#: 走行可能距離の当たりを説明できる検出（その方位の当たりは建物ではなく、この物体）
EXPLAIN_CLASSES = frozenset({DetClass.VEHICLE, DetClass.OBSTACLE, DetClass.PEDESTRIAN})
#: 検出の推定距離（中心まで）と走行可能距離（手前の面まで）の差がこれ以内なら、その物体の当たりとみなす [m]
EXPLAIN_TOL_M = 3.0
#: 検出枠の方位の幅に足す余裕 [rad]（走行可能距離は方位 1 本ぶんの線なので、枠の端を少し広げて当てる）
EXPLAIN_MARGIN_RAD = math.radians(2.0)
#: 隣り合う方位で奥行きがこれだけ跳べば、手前の端を遮蔽の角とみなす [m]
CORNER_JUMP_M = 6.0
#: 角までの距離を観測へ入れるときの上限 [m]。これより遠い・無いときは 1.0
CORNER_RANGE_M = 20.0
#: 走行可能距離がこれより短ければ、その先を死角（静的）とみなす [m]
OPEN_EPS_M = 0.05

#: 観測の欄に並べるセクター（前・後・左・右）
SECTOR_CAMERAS: tuple[CameraSpec, ...] = (DEFAULT_CAMERA, REAR_CAMERA, LEFT_CAMERA, RIGHT_CAMERA)
#: 角を探すカメラ（後ろの角は通り過ぎたもの）
CORNER_CAMERAS: frozenset[str] = frozenset({DEFAULT_CAMERA.key, LEFT_CAMERA.key, RIGHT_CAMERA.key})
_QUADRANT = math.pi / 2.0


@dataclass(frozen=True)
class CameraInput:
    """1 台のカメラの入力。結果が無い（まだ撮っていない）カメラは `result` が None。"""

    spec: CameraSpec
    result: PerceptionResult | None
    freespace: np.ndarray | None


def _edge_bearing(x: float, spec: CameraSpec) -> float:
    """画像の横位置 x（0..1）の方位 [rad]（カメラの向きから測り左が正）。"""
    return -math.atan2((float(x) - 0.5) * spec.width, spec.focal_px)


def _origin(spec: CameraSpec) -> tuple[float, float]:
    """カメラの位置（自車座標: 前方 +x / 左 +y）。"""
    return float(spec.forward), -float(spec.right)


def _at(spec: CameraSpec, angle: float, depth: float) -> tuple[float, float]:
    ox, oy = _origin(spec)
    bearing = spec.yaw + float(angle)
    return ox + depth * math.cos(bearing), oy + depth * math.sin(bearing)


def _arc(spec: CameraSpec, start: float, end: float, depth: float, step: float) -> list[tuple[float, float]]:
    count = max(1, int(math.ceil((end - start) / max(step, 1e-3))))
    return [_at(spec, start + (end - start) * i / count, depth) for i in range(count + 1)]


def camera_frustum_polygon(
    spec: CameraSpec, max_range: float = OCCLUSION_RANGE_M, arc_step: float = math.radians(4.0)
) -> list[tuple[float, float]]:
    """1 台のカメラの画角の扇（自車座標）。先頭がカメラの位置で、そこから見て星形なので扇状に三角形へ分けられる。"""
    return [_origin(spec), *_arc(spec, -HALF_FOV, HALF_FOV, float(max_range), arc_step)]


def shadow_polygon(shadow: ShadowPolygon, arc_step: float = math.radians(4.0)) -> list[tuple[float, float]]:
    """死角の扇（自車座標）。手前は弦・奥は弧の凸多角形で、[手前の始まり, 奥の弧…, 手前の終わり] の順。"""
    spec = next(s for s in CAMERA_RIG if s.key == shadow.camera)
    far = _arc(spec, shadow.start, shadow.end, shadow.far, arc_step)
    return [_at(spec, shadow.start, shadow.near), *far, _at(spec, shadow.end, shadow.near)]


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    """多角形の面積 [m²]（靴紐公式）。"""
    if len(points) < 3:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _camera(
    view: CameraInput, reach: float
) -> tuple[CameraVisibility, list[ShadowPolygon], np.ndarray, list[tuple[float, float]]]:
    """1 台ぶんの見えている奥行き・死角・車両の陰に入った方位・遮蔽の角（自車座標）。"""
    spec = view.spec
    covered = np.zeros(RAYS_PER_CAMERA, dtype=bool)
    if view.result is None:
        return CameraVisibility(spec.key, False, np.zeros(RAYS_PER_CAMERA)), [], covered, []

    if view.freespace is None:
        free = np.full(config.OBS_FREESPACE_DIM, reach, dtype=np.float64)
    else:
        free = np.clip(
            np.nan_to_num(np.asarray(view.freespace, dtype=np.float64).reshape(-1), nan=0.0, posinf=reach),
            0.0,
            reach,
        )

    explained = np.zeros(config.OBS_FREESPACE_DIM, dtype=bool)
    shadows: list[ShadowPolygon] = []
    depths = free[RAY_BIN].copy()
    for det in view.result.detections:
        if det.cls not in EXPLAIN_CLASSES:
            continue
        a, b = _edge_bearing(det.x0, spec), _edge_bearing(det.x1, spec)
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        lo, hi = min(a, b), max(a, b)
        dist = estimate_distance(det, spec)
        for k in FOV_BINS:
            centre = float(FREESPACE_ANGLES[k])
            if lo - EXPLAIN_MARGIN_RAD <= centre <= hi + EXPLAIN_MARGIN_RAD and abs(free[k] - dist) <= EXPLAIN_TOL_M:
                explained[k] = True
        if det.cls not in SHADOW_CLASSES or dist >= reach:
            continue
        start, end = max(lo, -HALF_FOV), min(hi, HALF_FOV)
        if start > end:
            continue
        hit = (RAY_ANGLES >= start) & (RAY_ANGLES <= end)
        if not hit.any():
            # 方位 1 本より細く写った遠くの車は、いちばん近い 1 本だけを陰にする
            hit[int(np.argmin(np.abs(RAY_ANGLES - (start + end) * 0.5)))] = True
        np.minimum(depths, dist, out=depths, where=hit)
        covered |= hit
        shadows.append(ShadowPolygon(spec.key, SHADOW_DYNAMIC, start, end, float(dist), reach))

    corners: list[tuple[float, float]] = []
    for k in FOV_BINS:
        if explained[k] or free[k] >= reach - OPEN_EPS_M:
            continue
        # 画角の端の方位は外側の 1 本に割り当てている（RAY_BIN）ので、扇も画角の端まで広げる
        lo = -HALF_FOV if k == FOV_BINS[0] else float(FREESPACE_ANGLES[k]) - FREESPACE_HALF
        hi = HALF_FOV if k == FOV_BINS[-1] else float(FREESPACE_ANGLES[k]) + FREESPACE_HALF
        shadows.append(ShadowPolygon(spec.key, SHADOW_STATIC, lo, hi, float(free[k]), reach))
    if spec.key in CORNER_CAMERAS:
        for k, k2 in zip(FOV_BINS[:-1], FOV_BINS[1:]):
            if explained[k] or explained[k2] or abs(free[k] - free[k2]) < CORNER_JUMP_M:
                continue
            boundary = (float(FREESPACE_ANGLES[k]) + float(FREESPACE_ANGLES[k2])) * 0.5
            cx, cy = _at(spec, boundary, float(min(free[k], free[k2])))
            if cx >= 0.0:
                corners.append((cx, cy))
    return CameraVisibility(spec.key, True, depths), shadows, covered, corners


def evaluate_occlusion(views: Sequence[CameraInput], reach: float = OCCLUSION_RANGE_M) -> OcclusionResult:
    """4 台のカメラの入力から、1 台ぶんの見えている所・死角・見通し距離・観測の 8 次元を作る。"""
    reach = float(reach)
    cameras: dict[str, CameraVisibility] = {}
    shadows: list[ShadowPolygon] = []
    corner: float | None = None
    front_covered = 0.0
    for view in views:
        vis, cam_shadows, covered, corners = _camera(view, reach)
        cameras[view.spec.key] = vis
        shadows.extend(cam_shadows)
        if view.spec.key == DEFAULT_CAMERA.key:
            front_covered = float(covered.mean())
        for cx, cy in corners:
            d = math.hypot(cx, cy)
            corner = d if corner is None else min(corner, d)

    def los(key: str) -> float | None:
        vis = cameras.get(key)
        return float(vis.depths.max()) if vis is not None and vis.known else None

    # 扇 1 本ぶんの面積は 0.5 d² Δ。90 度の扇（半径 reach）の面積で割る
    sectors = tuple(
        float((cameras[s.key].depths ** 2).sum() * (2.0 * RAY_HALF_WIDTH) / (reach * reach * _QUADRANT))
        if s.key in cameras and cameras[s.key].known
        else 0.0
        for s in SECTOR_CAMERAS
    )
    los_left, los_right = los(LEFT_CAMERA.key), los(RIGHT_CAMERA.key)
    features = np.array(
        [
            0.0 if los_left is None else los_left / reach,
            0.0 if los_right is None else los_right / reach,
            front_covered,
            1.0 if corner is None else min(corner, CORNER_RANGE_M) / CORNER_RANGE_M,
            *sectors,
        ],
        dtype=np.float32,
    )
    np.clip(features, 0.0, 1.0, out=features)
    return OcclusionResult(
        cameras=cameras,
        shadows=shadows,
        los_left=los_left,
        los_right=los_right,
        front_occluded=front_covered,
        corner_distance=corner,
        sectors=(sectors[0], sectors[1], sectors[2], sectors[3]),
        features=features,
        reach=reach,
    )


def unknown_occlusion(reach: float = OCCLUSION_RANGE_M) -> OcclusionResult:
    """どのカメラの結果も無いとき（何も見えていない）。"""
    return evaluate_occlusion([CameraInput(spec, None, None) for spec in CAMERA_RIG], reach)
