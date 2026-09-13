"""運転席カメラの姿勢と透視投影。**描く側とラベルを付ける側の唯一の実装。**

★ ここは `percep/camera.py`（擬似カメラで描く）と `percep/groundtruth.py`
  （真値から正解の箱を作る）の**両方から呼ばれる**。以前は同じ式が
  2 本あり、`groundtruth.py` の冒頭が「camera.py が未完成のあいだ実装が
  止まらないよう自己完結で書いてある（後で共通化する前提）」と明記していた
  （code_review C-06）。実際に符号規約も式も一致していて壊れてはいなかったが、
  **ずれても型でもビルドでも捕まらず、「画像に写っていない場所に正解の箱が付く」
  という形でしか症状が出ない**ので、その前提どおり 1 本にまとめた。

投影の規約（docs/protocol.md 1 章）:
    ENU 平面・メートル・x=東 / y=北 / heading は +x 軸から反時計回り。
    自車座標系は「前方 +x / 左 +y / 上 +z」。カメラは heading 方向を向き、
    `CameraSpec.pitch` だけ下を向く（下向きが負）。
    画像は左上原点で、u は右が正・v は下が正。

**`frontend/src/scene/cameraMath.ts` の `driverEye()` と同じ式**なので、
ここで得た検出結果は Three.js の映像にそのまま重ねられる。片方だけ変えないこと。
"""

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
    """車両の姿勢から運転席カメラの姿勢を作る。

    **`frontend/src/scene/cameraMath.ts` の `driverEye()` と同じ式。**
    日本車は右ハンドルなので進行方向の右へずらす（ENU では heading - 90 度）。
    """
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    # 進行方向の右手 = (sin(h), -cos(h))
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
    """世界座標を画像座標へ落とす。戻り値は (u, v, 奥行き)。

    引数を成分でばらして受けるのは、**カメラ 8 台 × 物体数のブロードキャストを
    そのまま通せるようにするため**（擬似カメラは 1 台ずつのループを持たない）。
    点の配列を渡したいときは `project_points()` を使う。

    奥行きが `spec.near` 以下の点は**呼び出し側で必ず捨てること**。
    カメラの後ろの点を割ると座標が反転し、画面に存在しない物体が現れる。
    ここでは 0 除算だけを避ける（符号は保つので、背後の点は背後のまま出る）。
    """
    rel_x = px - eye_x
    rel_y = py - eye_y
    rel_z = pz - eye_z

    # 自車座標系へ。右手成分 = (sin, -cos) との内積
    fwd = rel_x * cos_yaw + rel_y * sin_yaw
    right = rel_x * sin_yaw - rel_y * cos_yaw

    # 俯角（左軸まわりの回転）。pitch は下向きが負
    depth = cos_pitch * fwd + sin_pitch * rel_z
    up = -sin_pitch * fwd + cos_pitch * rel_z

    safe = np.where(np.abs(depth) < 1e-6, np.float64(1e-6), depth)
    u = cx + focal * (right / safe)
    v = cy - focal * (up / safe)
    return u, v, depth


def project_points(
    pose: CameraPose, spec: CameraSpec, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ワールド座標 (N, 3) をカメラ画像へ透視投影する。

    Returns:
        (u, v, depth)。u / v は**画素**（左上原点）、depth は光軸方向の距離 [m]。
        depth <= 0 はカメラの背後なので、呼び出し側で必ず捨てること。
    """
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
