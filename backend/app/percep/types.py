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
    "CLASS_QUOTA",
    "CLASS_PRIORITY",
    "DEFAULT_CAMERA",
    "DetClass",
    "Detection",
    "LANE_LOOKAHEAD_M",
    "LANE_POLYLINE_POINTS",
    "PerceptionResult",
    "SIGNAL_PHASE_NAMES",
    "pack_by_class_quota",
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

    #: 推定距離 [m]。**クラスによって測っているものが違う**ので、
    #: 「対象までの距離」として一律に扱わないこと（code_review Q-07）:
    #:
    #:   - 信号         : **停止線までの水平距離**（灯器までの距離ではない。
    #:                    観測が要求するのが停止線までの距離だから）
    #:   - 車線         : **認識できた車線の前方距離**（＝線の長さ。
    #:                    `detector._lane_polyline()` はこれを長さとして読む）
    #:   - 標識・車両・障害物: 対象そのものまでの距離
    #:
    #: `encoder._distance()` はこの 3 種を区別せず距離として扱い `near`〜`far` へ
    #: クランプする。いま車線がそこを通る経路は無いが、`_pick()` や freespace の
    #: 対象に車線を足すと**線の長さが静かに距離として使われる**。足すときは
    #: 先に `lane_length` のような別フィールドへ分けること。
    distance: float | None = None

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
        # ★ `is` ではなく `==` で比べる（code_review Q-09）。DetClass は IntEnum
        #   なので `2 is DetClass.VEHICLE` は False になり、`cls` に素の int が
        #   入ってくると**静かに空を返す**。同じ契約型の中で 2 つの比較規則が
        #   同居しないよう `encoder._iter_class()` の側へ揃えてある。
        return [d for d in self.detections if d.cls == cls]

    def best(self, cls: DetClass) -> Detection | None:
        """そのクラスで最も信頼度の高いもの。無ければ None。"""
        for det in self.detections:
            if det.cls == cls:
                return det
        return None

    def to_wire(self) -> list[dict[str, Any]]:
        return [d.to_wire() for d in self.detections]


# ---------------------------------------------------------------------------
# 検出の切り詰め方（真値と認識器で共通）
# ---------------------------------------------------------------------------

#: 車線として前方何メートルまでを 1 つの検出として扱うか [m]。
#: 真値（`groundtruth`）はここまで経路をたどって帯を作り、
#: 認識器（`detector`）は描く中心線の長さの上限に使う。
#: ★ 認識器の車線は直線近似なので、長く伸ばすほどカーブで実際の車線から離れる。
#:   未学習のモデルが 57m まで伸ばした実例があるので上限として効かせている。
LANE_LOOKAHEAD_M = 25.0

#: 路面へ重ねて描くために送る車線中心線の点数。
#: **20Hz で全車両ぶん流すので、増やすと配信量にそのまま効く。**
LANE_POLYLINE_POINTS = 6

#: クラスごとに残す上限。`config.PERCEP_MAX_DETECTIONS`（12）へ詰めるとき、
#: 観測化で必要な内訳（車線 1 / 信号 1 / 標識 1 / 車両 3 / 障害物 3）が
#: 必ず生き残るようにするための枠。合計はちょうど 12。
CLASS_QUOTA: dict[DetClass, int] = {
    DetClass.LANE: 1,
    DetClass.TRAFFIC_LIGHT: 2,
    DetClass.SPEED_SIGN: 1,
    DetClass.VEHICLE: 4,
    DetClass.OBSTACLE: 4,
}

#: 枠で詰めるときの優先順位（ラウンドロビンで先に置く順）。
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
    """クラス枠つきの優先度ラウンドロビンで `limit` 件へ詰める。

    ★ **真値（`groundtruth`）と認識器（`detector`）の両方がこれを呼ぶこと**
      （code_review Q-01）。片方だけ「信頼度の上位から一括で 12 件」にすると、
      発火したセルが車両で埋まったときに信号・標識・車線が 1 件も残らず、
      `encoder` が「見えなかった」既定値で埋めるため
      **目の前に赤信号があっても観測は「信号は無い」**になる。
      報酬と終了判定は world の真値なので、そのとき**罰だけが入る**。
      実測では未学習のモデルで 16.4% の画像から信号が丸ごと消えた。
      画面のボックスも同じ検出から描くので、症状は「成績が伸びない」という
      形でしか出ない。`encode_targets` / `decode_detections` を対で扱うのと
      同じ理由で、詰め方は 1 か所にまとめてある。

    各クラスは `CLASS_QUOTA` の件数まで。渡す時点で**各クラス内は残したい順**
    （真値なら近い順、認識器なら信頼度の降順）に並べておくこと。
    """
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
