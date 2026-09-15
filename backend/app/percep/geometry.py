"""運転席カメラの姿勢と透視投影。**描く側とラベルを付ける側の唯一の実装。**"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.percep.types import CameraSpec

__all__ = [
    "CameraPose",
    "camera_pose",
    "project_components",
    "project_points",
]


@dataclass(frozen=True)
class CameraPose:
    """1 台ぶんのカメラ姿勢。三角関数を毎回引かないよう展開して持つ。"""

    eye_x: float
    eye_y: float
    eye_z: float
    cos_yaw: float
    sin_yaw: float
    cos_pitch: float
    sin_pitch: float


def camera_pose(x: float, y: float, heading: float, spec: CameraSpec) -> CameraPose:
    """車両の姿勢から運転席カメラの姿勢を作る。"""
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    eye_x = x + cos_h * spec.forward + sin_h * spec.right
    eye_y = y + sin_h * spec.forward - cos_h * spec.right
    pitch = spec.pitch
    return CameraPose(
        eye_x=float(eye_x),
        eye_y=float(eye_y),
        eye_z=float(spec.eye_height),
        cos_yaw=float(cos_h),
        sin_yaw=float(sin_h),
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
