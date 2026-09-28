"""運転席カメラの姿勢と透視投影。**描く側とラベルを付ける側の唯一の実装。**"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.percep.types import CameraSpec

__all__ = [
    "CameraPose",
    "NeighborIndex",
    "camera_pose",
    "eye_position",
    "project_components",
    "project_points",
    "view_heading",
]

#: これ以下の点群は索引を作らず、全部を候補として返す
SMALL_POINT_SET = 1500


class NeighborIndex:
    """点群を一様グリッドに入れて、半径内の候補を返す索引（擬似カメラと真値の両方が使う）。"""

    def __init__(self, xs: np.ndarray, ys: np.ndarray, radius: float) -> None:
        self._count = int(xs.size)
        self._all = np.arange(self._count, dtype=np.int64)
        self._small = self._count <= SMALL_POINT_SET
        self._cache: dict[tuple[int, int], np.ndarray] = {}
        if self._small or self._count == 0:
            return

        self._cell = max(float(radius) * 2.0, 1.0)
        ci = np.floor(xs / self._cell).astype(np.int64)
        ri = np.floor(ys / self._cell).astype(np.int64)
        self._buckets: dict[tuple[int, int], np.ndarray] = {}
        order = np.lexsort((ci, ri))
        keys = list(zip(ri[order].tolist(), ci[order].tolist()))
        start = 0
        for i in range(1, len(keys) + 1):
            if i == len(keys) or keys[i] != keys[start]:
                self._buckets[keys[start]] = order[start:i]
                start = i

    def query(self, x: float, y: float) -> np.ndarray:
        """(x, y) から `radius` 以内の点を必ず含む候補の添字（それより遠い点も混ざる）。"""
        if self._small or self._count == 0:
            return self._all
        cr = int(math.floor(y / self._cell))
        cc = int(math.floor(x / self._cell))
        hit = self._cache.get((cr, cc))
        if hit is None:
            parts = [
                self._buckets[(cr + dr, cc + dc)]
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (cr + dr, cc + dc) in self._buckets
            ]
            hit = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
            self._cache[(cr, cc)] = hit
        return hit


@dataclass(frozen=True)
class CameraPose:
    """1 台ぶんのカメラ姿勢。三角関数を毎回引かないよう展開して持つ。"""

    eye_x: float
    eye_y: float
    eye_z: float
    #: 視線の向き（車体の向き + カメラの向き）
    cos_yaw: float
    sin_yaw: float
    cos_pitch: float
    sin_pitch: float

    @property
    def yaw(self) -> float:
        """視線の方位 [rad]。"""
        return math.atan2(self.sin_yaw, self.cos_yaw)


def eye_position(x, y, cos_heading, sin_heading, spec: CameraSpec):
    """車両の位置と**車体の**向きから、カメラの取り付け位置 (x, y) を出す。スカラーでも配列でもよい。"""
    return (
        x + cos_heading * spec.forward + sin_heading * spec.right,
        y + sin_heading * spec.forward - cos_heading * spec.right,
    )


def view_heading(heading, spec: CameraSpec):
    """車体の向きから、そのカメラの視線の方位を出す。スカラーでも配列でもよい。"""
    return heading + spec.yaw


def camera_pose(x: float, y: float, heading: float, spec: CameraSpec) -> CameraPose:
    """車両の姿勢からカメラの姿勢を作る。取り付け位置は車体の向き、視線はそれにカメラの向きを足す。"""
    eye_x, eye_y = eye_position(x, y, math.cos(heading), math.sin(heading), spec)
    look = float(view_heading(float(heading), spec))
    pitch = spec.pitch
    return CameraPose(
        eye_x=float(eye_x),
        eye_y=float(eye_y),
        eye_z=float(spec.eye_height),
        cos_yaw=float(math.cos(look)),
        sin_yaw=float(math.sin(look)),
        cos_pitch=float(math.cos(pitch)),
        sin_pitch=float(math.sin(pitch)),
    )


def project_components(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    eye_x: np.ndarray,
    eye_y: np.ndarray,
    eye_z: float,
    cos_yaw: np.ndarray,
    sin_yaw: np.ndarray,
    cos_pitch: float,
    sin_pitch: float,
    focal: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """世界座標を画像座標へ落とす。戻り値は (u, v, 奥行き)。"""
    rel_x = px - eye_x
    rel_y = py - eye_y
    rel_z = pz - eye_z

    fwd = rel_x * cos_yaw + rel_y * sin_yaw
    right = rel_x * sin_yaw - rel_y * cos_yaw

    depth = cos_pitch * fwd + sin_pitch * rel_z
    up = -sin_pitch * fwd + cos_pitch * rel_z

    safe = np.where(np.abs(depth) < 1e-6, np.float64(1e-6), depth)
    u = cx + focal * (right / safe)
    v = cy - focal * (up / safe)
    return u, v, depth


def project_points(
    pose: CameraPose, spec: CameraSpec, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ワールド座標 (N, 3) をカメラ画像へ透視投影する。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return project_components(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        pose.eye_x,
        pose.eye_y,
        pose.eye_z,
        pose.cos_yaw,
        pose.sin_yaw,
        pose.cos_pitch,
        pose.sin_pitch,
        spec.focal_px,
        spec.width * 0.5,
        spec.height * 0.5,
    )
