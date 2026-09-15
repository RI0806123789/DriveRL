"""画像認識パイプラインの共有型。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

from app import config

__all__ = [
    "CameraSpec",
    "CLASS_QUOTA",
    "CLASS_PRIORITY",
    "DEFAULT_CAMERA",
    "DetClass",
    "Detection",
    "FACING_TOLERANCE",
    "LANE_LOOKAHEAD_M",
    "LANE_POLYLINE_POINTS",
    "PerceptionResult",
    "SIGNAL_BEYOND_MARGIN",
    "SIGNAL_HEAD_Z",
    "SIGNAL_HOUSING_H",
    "SIGNAL_HOUSING_W",
    "SIGNAL_LAMP_PITCH",
    "SIGNAL_LAMP_RADIUS",
    "SIGNAL_MOUNT_HEIGHT",
    "SIGNAL_PHASE_NAMES",
    "SIGN_BOARD_Z",
    "SIGN_BOTTOM_HEIGHT",
    "SIGN_RADIUS",
    "facing_viewer",
    "pack_by_class_quota",
]


@dataclass(frozen=True)
class CameraSpec:
    """擬似カメラの内部パラメータ。"""

    width: int = 192
    height: int = 144

    fov_deg: float = 68.0
    forward: float = 0.35
    right: float = 0.36
    eye_height: float = 1.22
    look_ahead: float = 30.0
    look_drop: float = 1.1

    near: float = 0.5
    far: float = 120.0

    @property
    def pitch(self) -> float:
        """カメラの俯角 [rad]（下向きが負）。注視点の下がりから決まる。"""
        return math.atan2(-self.look_drop, max(self.look_ahead - self.forward, 1e-6))

    @property
    def focal_px(self) -> float:
        """水平方向の焦点距離 [px]。透視投影の基準になる。"""
        return (self.width * 0.5) / math.tan(math.radians(self.fov_deg) * 0.5)

    @property
    def aspect(self) -> float:
        return self.width / max(self.height, 1)

    def to_wire(self) -> dict[str, Any]:
        """フロントが投影を再現できるだけの情報を返す。"""
        return {
            "width": self.width,
            "height": self.height,
            "fovDeg": self.fov_deg,
            "forward": self.forward,
            "right": self.right,
            "eyeHeight": self.eye_height,
            "lookAhead": self.look_ahead,
            "lookDrop": self.look_drop,
        }


DEFAULT_CAMERA = CameraSpec()


class DetClass(IntEnum):
    """認識器が出すクラス。"""

    TRAFFIC_LIGHT = 0
    SPEED_SIGN = 1
    VEHICLE = 2
    OBSTACLE = 3
    LANE = 4


NUM_CLASSES = len(DetClass)

SIGNAL_PHASE_NAMES = ("青", "黄", "赤")


@dataclass
class Detection:
    """1 個の検出結果。"""

    cls: DetClass
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float

    phase: int | None = None
    speed_limit: float | None = None

    distance: float | None = None

    lateral: float | None = None

    lane_points: list[tuple[float, float]] | None = None

    @property
    def label(self) -> str:
        """画面に出す日本語名。ログとデバッグにも使う。"""
        if self.cls is DetClass.TRAFFIC_LIGHT:
            if self.phase is not None and 0 <= self.phase < len(SIGNAL_PHASE_NAMES):
                return f"信号機：{SIGNAL_PHASE_NAMES[self.phase]}"
            return "信号機"
        if self.cls is DetClass.SPEED_SIGN:
            if self.speed_limit is not None:
                return f"速度標識：{round(self.speed_limit * 3.6)}km/h"
            return "速度標識"
        if self.cls is DetClass.LANE:
            return "車線"
        if self.cls is DetClass.VEHICLE:
            return "車両"
        return "障害物"

    def to_wire(self) -> dict[str, Any]:
        """WebSocket へ載せる形。**ラベル文字列は送らない**（フロントで組む）。"""
        out: dict[str, Any] = {
            "cls": int(self.cls),
            "box": [
                round(self.x0, 3),
                round(self.y0, 3),
                round(self.x1, 3),
                round(self.y1, 3),
            ],
            "conf": round(self.confidence, 2),
        }
        if self.phase is not None:
            out["phase"] = int(self.phase)
        if self.speed_limit is not None:
            out["speedLimit"] = round(self.speed_limit, 2)
        if self.distance is not None:
            out["distance"] = round(self.distance, 1)
        if self.lateral is not None:
            out["lateral"] = round(self.lateral, 2)
        if self.lane_points:
            out["lanePoints"] = [
                [round(px, 1), round(py, 2)] for px, py in self.lane_points
            ]
        return out


@dataclass
class PerceptionResult:
    """1 台ぶんの認識結果。"""

    slot: int
    detections: list[Detection] = field(default_factory=list)

    def by_class(self, cls: DetClass) -> list[Detection]:
        return [d for d in self.detections if d.cls == cls]

    def best(self, cls: DetClass) -> Detection | None:
        """そのクラスで最も信頼度の高いもの。無ければ None。"""
        for det in self.detections:
            if det.cls == cls:
                return det
        return None

    def to_wire(self) -> list[dict[str, Any]]:
        return [d.to_wire() for d in self.detections]


SIGNAL_MOUNT_HEIGHT = 5.0
SIGNAL_HOUSING_W = 1.16
SIGNAL_HOUSING_H = 0.44
SIGNAL_LAMP_PITCH = 0.35
SIGNAL_LAMP_RADIUS = 0.15
SIGNAL_HEAD_Z = SIGNAL_MOUNT_HEIGHT + SIGNAL_HOUSING_H * 0.5
SIGNAL_BEYOND_MARGIN = 2.0

SIGN_RADIUS = config.SPEED_SIGN_DIAMETER * 0.5
SIGN_BOTTOM_HEIGHT = config.SPEED_SIGN_BOTTOM_HEIGHT
SIGN_BOARD_Z = SIGN_BOTTOM_HEIGHT + SIGN_RADIUS

FACING_TOLERANCE = math.radians(75.0)


def facing_viewer(
    obj_x: Any,
    obj_y: Any,
    obj_heading: Any,
    eye_x: Any,
    eye_y: Any,
    viewer_heading: Any,
    tolerance: float = FACING_TOLERANCE,
) -> np.ndarray:
    """信号・標識が視点に正対しているかを返す（bool の ndarray）。"""
    cos_o = np.cos(obj_heading)
    sin_o = np.sin(obj_heading)
    aligned = (
        cos_o * np.cos(viewer_heading) + sin_o * np.sin(viewer_heading)
        >= math.cos(float(tolerance))
    )
    ahead = (obj_x - eye_x) * cos_o + (obj_y - eye_y) * sin_o > 0.0
    return np.asarray(aligned & ahead, dtype=bool)


LANE_LOOKAHEAD_M = 25.0

LANE_POLYLINE_POINTS = 6

CLASS_QUOTA: dict[DetClass, int] = {
    DetClass.LANE: 1,
    DetClass.TRAFFIC_LIGHT: 2,
    DetClass.SPEED_SIGN: 1,
    DetClass.VEHICLE: 4,
    DetClass.OBSTACLE: 4,
}

CLASS_PRIORITY: tuple[DetClass, ...] = (
    DetClass.LANE,
    DetClass.TRAFFIC_LIGHT,
    DetClass.SPEED_SIGN,
    DetClass.VEHICLE,
    DetClass.OBSTACLE,
)


def pack_by_class_quota(
    per_class: dict[DetClass, list[Detection]], limit: int
) -> list[Detection]:
    """クラス枠つきの優先度ラウンドロビンで `limit` 件へ詰める。"""
    cursor: dict[DetClass, int] = {cls: 0 for cls in CLASS_PRIORITY}
    picked: list[Detection] = []
    while len(picked) < limit:
        added = False
        for cls in CLASS_PRIORITY:
            items = per_class.get(cls) or []
            i = cursor[cls]
            if i < min(len(items), CLASS_QUOTA.get(cls, 0)):
                picked.append(items[i])
                cursor[cls] = i + 1
                added = True
                if len(picked) >= limit:
                    break
        if not added:
            break
    return picked
