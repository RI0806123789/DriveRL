"""画像認識パイプラインの共有型。

`percep` パッケージは「擬似カメラ画像 → CNN 認識器 → 検出結果」を担う。
検出結果は 2 か所で使われるので、この型が両者の**唯一の契約**になる:

- `percep/encoder.py` が観測ベクトルへ変換して PPO へ渡す（学習入力）
- `contracts.FrameSnapshot.detections` に載って WebSocket でフロントへ届き、
  運転席カメラのバウンディングボックスとして描かれる（可視化）

**学習が見ているものと画面に出るものが同じ**であることがこの設計の要点で、
片方だけを変えてはいけない。ここを分けると「画面では信号を認識できているのに
学習は別の値を見ている」という、外から絶対に気づけない食い違いが生まれる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

__all__ = [
    "CameraSpec",
    "DEFAULT_CAMERA",
    "DetClass",
    "Detection",
    "PerceptionResult",
    "SIGNAL_PHASE_NAMES",
]


# ---------------------------------------------------------------------------
# カメラ
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraSpec:
    """擬似カメラの内部パラメータ。

    **フロントエンドの運転席カメラ（`frontend/src/scene/cameraMath.ts`）と
    必ず一致させること。** 一致していないと、バックエンドが検出した
    バウンディングボックスを Three.js の映像へ重ねたときにずれる。
    可視化は「学習が見ているもの」を映すのが目的なので、ずれると意味を成さない。

    対応する定数:
        forward   <-> DRIVER_FORWARD    = 0.35
        right     <-> DRIVER_RIGHT      = 0.36
        eye_height<-> DRIVER_EYE_HEIGHT = 1.22
        pitch     <-> DRIVER_LOOK_AHEAD = 30 / DRIVER_LOOK_DROP = 1.1 から決まる
        fov_deg   <-> 運転席カメラの視野角 68 度
    """

    # 認識器へ渡す画像の大きさ [px]。
    # ★ 小さくしすぎると信号の灯色も標識の数字も原理的に読めない。
    #   水平画角 68 度・幅 W px のとき、距離 d [m] にある幅 w [m] の物体は
    #       画素幅 ≈ W * w / (2 * d * tan(34°))
    #   に写る。幅 192px なら 1.2m の灯器が 15m 先で約 11px、0.6m の標識が
    #   10m 先で約 8px。これが読み取れる下限で、これ以上落とすと
    #   「認識できないのはモデルのせい」ではなく**入力に情報が無い**状態になる。
    width: int = 192
    height: int = 144

    fov_deg: float = 68.0          # 水平画角。運転席カメラと同じ
    forward: float = 0.35          # 車体中心からの前方オフセット [m]
    right: float = 0.36            # 右方向オフセット [m]（日本車の右ハンドル）
    eye_height: float = 1.22       # 路面からの視点高さ [m]
    look_ahead: float = 30.0       # 注視点までの水平距離 [m]
    look_drop: float = 1.1         # 注視点の下がり [m]

    near: float = 0.5              # これより近い物は描かない [m]
    far: float = 120.0             # これより遠い物は描かない [m]

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


# ---------------------------------------------------------------------------
# 検出
# ---------------------------------------------------------------------------


class DetClass(IntEnum):
    """認識器が出すクラス。

    **並び順は学習済みモデルの出力チャンネルと直結している。**
    途中に挿入すると既存の認識器の重みが全部ずれるので、追加は末尾に足すこと。
    """

    TRAFFIC_LIGHT = 0   # 信号機（灯色を phase に持つ）
    SPEED_SIGN = 1      # 最高速度標識（規制速度を speed_limit に持つ）
    VEHICLE = 2         # 他車両
    OBSTACLE = 3        # パイロン
    LANE = 4            # 走行車線（横方向偏差を lateral に持つ）


NUM_CLASSES = len(DetClass)

# 灯色の表示名。protocol の signals と同じ並び（0=青 / 1=黄 / 2=赤）
SIGNAL_PHASE_NAMES = ("青", "黄", "赤")


@dataclass
class Detection:
    """1 個の検出結果。

    座標は**画像の正規化座標**（左上 0,0 / 右下 1,1）で持つ。画素で持つと
    解像度を変えたときに全部の利用側を直すことになるため。
    """

    cls: DetClass
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float

    # --- クラスごとの属性（該当しないクラスでは None）---
    phase: int | None = None          # 信号: 0=青 / 1=黄 / 2=赤
    speed_limit: float | None = None  # 標識: 規制速度 [m/s]
    distance: float | None = None     # 推定距離 [m]
    lateral: float | None = None      # 車線: 車線中心からの横方向偏差 [m]

    #: 車線: 認識した車線中心線（**自車座標系** 前方 +x / 左 +y [m]）。
    #: ★ 車線は矩形で囲んでも「正しく認識できているか」が分からない
    #:   （細長い曲線なので、ボックスは道の形をまったく表さない）。
    #:   路面へ直接重ねて描くために点列で持つ。画面ではこれを実際の車線標示の上に
    #:   重ねるので、**ずれていればそのままずれて見える**。
    lane_points: list[tuple[float, float]] | None = None

    @property
    def label(self) -> str:
        """画面に出す日本語名。ログとデバッグにも使う。

        フロントも同じ規則で組み立てる（転送量を減らすため文字列は送らない）。
        **変えるときは `frontend/src/scene/detectionLabels.ts` と両方直すこと。**
        """
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
        """WebSocket へ載せる形。**ラベル文字列は送らない**（フロントで組む）。

        20Hz で全車両分を流すので、桁を落として量を抑える。
        正規化座標は 3 桁あれば 192px 幅で 1px を切る精度になる。
        """
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
            # 前方距離は 10cm、横位置は 1cm の精度があれば路面へ重ねるには足りる。
            # 20Hz で全車両ぶん流すので、これ以上の桁は量に見合わない。
            out["lanePoints"] = [
                [round(px, 1), round(py, 2)] for px, py in self.lane_points
            ]
        return out


@dataclass
class PerceptionResult:
    """1 台ぶんの認識結果。

    `detections` は信頼度の降順。観測ベクトル化も可視化もこの順を前提にする
    （上位 N 件だけを使う箇所があるため）。
    """

    slot: int
    detections: list[Detection] = field(default_factory=list)

    def by_class(self, cls: DetClass) -> list[Detection]:
        return [d for d in self.detections if d.cls is cls]

    def best(self, cls: DetClass) -> Detection | None:
        """そのクラスで最も信頼度の高いもの。無ければ None。"""
        for det in self.detections:
            if det.cls is cls:
                return det
        return None

    def to_wire(self) -> list[dict[str, Any]]:
        return [d.to_wire() for d in self.detections]
