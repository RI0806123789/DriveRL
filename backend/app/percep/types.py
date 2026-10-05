"""画像認識パイプラインの共有型。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

from app import config

__all__ = [
    "CAMERAS_BY_KEY",
    "CAMERA_RIG",
    "CameraSpec",
    "CLASS_QUOTA",
    "CLASS_PRIORITY",
    "DEFAULT_CAMERA",
    "LEFT_CAMERA",
    "REAR_CAMERA",
    "RIGHT_CAMERA",
    "SURROUND_CAMERAS",
    "CameraVisibility",
    "DetClass",
    "Detection",
    "OcclusionResult",
    "ShadowPolygon",
    "FACING_TOLERANCE",
    "LANE_LOOKAHEAD_M",
    "LANE_POLYLINE_POINTS",
    "PEDESTRIAN_HALF_WIDTH",
    "PEDESTRIAN_HEAD_RADIUS",
    "PEDESTRIAN_HEAD_Z",
    "PEDESTRIAN_HEIGHT",
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
    "estimate_distance",
    "facing_viewer",
    "pack_by_class_quota",
    "signal_ahead_of_stop",
]


@dataclass(frozen=True)
class CameraSpec:
    """擬似カメラの内部パラメータと、車体への取り付け位置・向き。"""

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

    key: str = "front"
    #: 車両の進行方向から測ったカメラの向き [度]（反時計回りが正。左 = +90 / 後ろ = 180）
    yaw_deg: float = 0.0
    #: 俯角を直接決めるとき [度]（下向きが負）。None なら注視点の下がりから決める
    pitch_deg: float | None = None

    @property
    def pitch(self) -> float:
        """カメラの俯角 [rad]（下向きが負）。"""
        if self.pitch_deg is not None:
            return math.radians(float(self.pitch_deg))
        return math.atan2(-self.look_drop, max(self.look_ahead - self.forward, 1e-6))

    @property
    def yaw(self) -> float:
        """車両の進行方向から測ったカメラの向き [rad]。"""
        return math.radians(float(self.yaw_deg))

    @property
    def focal_px(self) -> float:
        """水平方向の焦点距離 [px]。透視投影の基準になる。"""
        return (self.width * 0.5) / math.tan(math.radians(self.fov_deg) * 0.5)

    def to_wire(self) -> dict[str, Any]:
        """フロントが投影を再現できるだけの情報を返す。"""
        return {
            "key": self.key,
            "width": self.width,
            "height": self.height,
            "fovDeg": self.fov_deg,
            "forward": self.forward,
            "right": self.right,
            "eyeHeight": self.eye_height,
            "yawDeg": self.yaw_deg,
            "pitchDeg": math.degrees(self.pitch),
        }


DEFAULT_CAMERA = CameraSpec()

#: 周囲カメラの共通の俯角 [度]。車のすぐ後ろ・横の路面まで写すため前方より下へ向ける
SURROUND_PITCH_DEG = -10.0

REAR_CAMERA = CameraSpec(
    forward=-0.62,
    right=0.0,
    eye_height=1.38,
    key="rear",
    yaw_deg=180.0,
    pitch_deg=SURROUND_PITCH_DEG,
)
LEFT_CAMERA = CameraSpec(
    forward=0.0,
    right=-0.74,
    eye_height=1.34,
    key="left",
    yaw_deg=90.0,
    pitch_deg=SURROUND_PITCH_DEG,
)
RIGHT_CAMERA = CameraSpec(
    forward=0.0,
    right=0.74,
    eye_height=1.34,
    key="right",
    yaw_deg=-90.0,
    pitch_deg=SURROUND_PITCH_DEG,
)

#: 前方以外の 3 台。並びはワイヤ形式（`frame.surround`）と観測の欄の並びと同じ
SURROUND_CAMERAS: tuple[CameraSpec, ...] = (REAR_CAMERA, LEFT_CAMERA, RIGHT_CAMERA)
CAMERA_RIG: tuple[CameraSpec, ...] = (DEFAULT_CAMERA, *SURROUND_CAMERAS)
CAMERAS_BY_KEY: dict[str, CameraSpec] = {spec.key: spec for spec in CAMERA_RIG}

# 4 台とも同じ認識器に通すので、画の大きさ・画角・奥行きの範囲は揃っていなければならない
assert all(
    (s.width, s.height, s.fov_deg, s.near, s.far)
    == (DEFAULT_CAMERA.width, DEFAULT_CAMERA.height, DEFAULT_CAMERA.fov_deg, DEFAULT_CAMERA.near, DEFAULT_CAMERA.far)
    for s in CAMERA_RIG
), "周囲カメラの内部パラメータが前方カメラと揃っていない"


class DetClass(IntEnum):
    """認識器が出すクラス。"""

    TRAFFIC_LIGHT = 0
    SPEED_SIGN = 1
    VEHICLE = 2
    OBSTACLE = 3
    LANE = 4
    PEDESTRIAN = 5


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

    #: 安全ギミックが付ける危険度。1 = 注意（進路の近く）/ 2 = これで止めた。付けるのは `sim/safety.py` だけ
    hazard: int | None = None

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
        if self.cls is DetClass.PEDESTRIAN:
            return "歩行者"
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
        if self.hazard:
            out["hazard"] = int(self.hazard)
        return out


@dataclass
class PerceptionResult:
    """1 台ぶんの認識結果。"""

    slot: int
    detections: list[Detection] = field(default_factory=list)

    def by_class(self, cls: DetClass) -> list[Detection]:
        return [d for d in self.detections if d.cls == cls]

    def to_wire(self) -> list[dict[str, Any]]:
        return [d.to_wire() for d in self.detections]


SHADOW_DYNAMIC = "dynamic"
SHADOW_STATIC = "static"


@dataclass(frozen=True)
class ShadowPolygon:
    """死角 1 つ。カメラから見た方位 [start, end]（カメラの向きから測り左が正）と、奥行き [near, far] の扇で持つ。"""

    camera: str
    #: dynamic = 検出した車両の陰 / static = 走行可能距離の先（建物の陰・視程の外）
    kind: str
    start: float
    end: float
    near: float
    far: float

    def to_wire(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "kind": self.kind,
            "from": round(self.start, 3),
            "to": round(self.end, 3),
            "near": round(self.near, 1),
        }


@dataclass
class CameraVisibility:
    """1 台のカメラで見えている奥行き。画角を等分した方位ごとに持つ。"""

    key: str
    #: このカメラの結果があるか。無ければ何も見えていないものとして扱う
    known: bool
    #: 方位ごとの見えている奥行き [m]（`percep/occlusion.py` の RAY_ANGLES の順）
    depths: np.ndarray

    def runs(self, angles: np.ndarray, half_width: float) -> list[list[float]]:
        """奥行きが同じ方位をまとめた [始まりの方位, 終わりの方位, 奥行き] の列（ワイヤ形式）。"""
        out: list[list[float]] = []
        for angle, depth in zip(angles, self.depths):
            lo, hi, d = float(angle) - half_width, float(angle) + half_width, round(float(depth), 1)
            if out and abs(out[-1][2] - d) < 0.05:
                out[-1][1] = round(hi, 3)
            else:
                out.append([round(lo, 3), round(hi, 3), d])
        return out


@dataclass
class OcclusionResult:
    """1 台ぶんの見えている所と死角。どれもカメラの検出と走行可能距離だけから作る（`percep/occlusion.py`）。"""

    cameras: dict[str, CameraVisibility]
    shadows: list[ShadowPolygon]
    #: 左右のカメラの見通し距離 [m]。そのカメラの結果が無ければ None
    los_left: float | None
    los_right: float | None
    #: 前方カメラの画角のうち、検出した車両の陰になっている割合
    front_occluded: float
    #: いちばん近い遮蔽の角（奥行きが跳ぶ所の手前）までの距離 [m]。無ければ None
    corner_distance: float | None
    #: 前・後・左・右の 90 度ずつの扇（半径は見る距離）のうち、見えている面積の割合
    sectors: tuple[float, float, float, float]
    #: 観測の末尾 8 次元（`config.OBS_LAYOUT` の occlusion）
    features: np.ndarray
    reach: float = 0.0

    def to_wire(self, angles: np.ndarray, half_width: float) -> dict[str, Any]:
        """frame.occlusion に載せる形。座標は自車座標（前方 +x / 左 +y）、方位は rad。"""
        cams = []
        for key, vis in self.cameras.items():
            if not vis.known:
                continue
            spec = CAMERAS_BY_KEY[key]
            cams.append(
                {
                    "key": key,
                    # + 0.0 は -0.0 を 0.0 にするため
                    "at": [round(float(spec.forward), 2) + 0.0, round(-float(spec.right), 2) + 0.0],
                    "yaw": round(spec.yaw, 4),
                    "seen": vis.runs(angles, half_width),
                }
            )
        out: dict[str, Any] = {
            "range": round(self.reach, 1),
            "los": [
                None if self.los_left is None else round(self.los_left, 1),
                None if self.los_right is None else round(self.los_right, 1),
            ],
            "sectors": [round(float(v), 3) for v in self.sectors],
            "frontOccluded": round(float(self.front_occluded), 3),
            "cameras": cams,
            "shadows": [s.to_wire() for s in self.shadows],
        }
        if self.corner_distance is not None:
            out["corner"] = round(float(self.corner_distance), 1)
        return out


SIGNAL_MOUNT_HEIGHT = 5.0
SIGNAL_HOUSING_W = 1.16
SIGNAL_HOUSING_H = 0.44
SIGNAL_LAMP_PITCH = 0.35
SIGNAL_LAMP_RADIUS = 0.15
SIGNAL_HEAD_Z = SIGNAL_MOUNT_HEIGHT + SIGNAL_HOUSING_H * 0.5
SIGNAL_BEYOND_MARGIN = 2.0

PEDESTRIAN_HEIGHT = config.PEDESTRIAN_HEIGHT
PEDESTRIAN_HALF_WIDTH = config.PEDESTRIAN_RADIUS
PEDESTRIAN_HEAD_Z = PEDESTRIAN_HEIGHT - 0.11
PEDESTRIAN_HEAD_RADIUS = 0.105

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


def signal_ahead_of_stop(
    stop_x: Any, stop_y: Any, signal_heading: Any, eye_x: Any, eye_y: Any
) -> np.ndarray:
    """信号の停止線をまだ越えていないか。越えた灯器は視点を規制しないので、描かずラベルも付けない。"""
    along = (stop_x - eye_x) * np.cos(signal_heading) + (stop_y - eye_y) * np.sin(
        signal_heading
    )
    # 信号無視の判定（`World.signal_violations`）と同じ許容で「越えた」とみなす
    return np.asarray(along > -float(config.SIGNAL_STOP_TOLERANCE_M), dtype=bool)


LANE_LOOKAHEAD_M = 25.0

LANE_POLYLINE_POINTS = 6

CLASS_QUOTA: dict[DetClass, int] = {
    DetClass.LANE: 1,
    DetClass.TRAFFIC_LIGHT: 2,
    DetClass.SPEED_SIGN: 1,
    DetClass.PEDESTRIAN: 3,
    DetClass.VEHICLE: 4,
    DetClass.OBSTACLE: 4,
}

CLASS_PRIORITY: tuple[DetClass, ...] = (
    DetClass.LANE,
    DetClass.TRAFFIC_LIGHT,
    DetClass.SPEED_SIGN,
    DetClass.PEDESTRIAN,
    DetClass.VEHICLE,
    DetClass.OBSTACLE,
)


#: 距離が「手前を優先する」意味を持つクラス。車線だけは面なので信頼度で選ぶ
DISTANCE_ORDERED: frozenset[DetClass] = frozenset(
    {
        DetClass.TRAFFIC_LIGHT,
        DetClass.SPEED_SIGN,
        DetClass.VEHICLE,
        DetClass.OBSTACLE,
        DetClass.PEDESTRIAN,
    }
)

_ASSUMED_WIDTH_M: dict[DetClass, float] = {
    DetClass.TRAFFIC_LIGHT: 1.2,
    DetClass.SPEED_SIGN: float(config.SPEED_SIGN_DIAMETER),
    DetClass.VEHICLE: float(config.VEHICLE_WIDTH),
    DetClass.OBSTACLE: float(config.OBSTACLE_RADIUS) * 2.0,
    DetClass.PEDESTRIAN: float(config.PEDESTRIAN_WIDTH),
}


def estimate_distance(det: Detection, spec: CameraSpec = DEFAULT_CAMERA) -> float:
    """検出までの推定距離 [m]。`near`〜`far` に収める。"""
    given = det.distance
    if given is not None:
        value = float(given)
        if math.isfinite(value) and value > 0.0:
            return min(max(value, float(spec.near)), float(spec.far))

    real = _ASSUMED_WIDTH_M.get(det.cls)
    width_px = (float(det.x1) - float(det.x0)) * spec.width
    if real is None or not math.isfinite(width_px) or width_px <= 1e-3:
        return float(spec.far)
    return min(max(spec.focal_px * real / width_px, float(spec.near)), float(spec.far))


def pack_by_class_quota(
    per_class: dict[DetClass, list[Detection]], limit: int, spec: CameraSpec = DEFAULT_CAMERA
) -> list[Detection]:
    """クラス枠つきの優先度ラウンドロビンで `limit` 件へ詰める。"""
    per_class = {
        cls: (
            sorted(items, key=lambda det: estimate_distance(det, spec))
            if cls in DISTANCE_ORDERED
            else list(items)
        )
        for cls, items in per_class.items()
    }
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
