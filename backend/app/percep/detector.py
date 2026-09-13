# -*- coding: utf-8 -*-
"""擬似カメラ画像から検出結果を作る軽量 CNN 認識器。

Keras 3 を **torch バックエンド**で使う（`app/rl/export.py` と同じ作法）。
この環境に TensorFlow は入っていないし、入れてはいけない。

出力形式は YOLO v1 風のグリッド回帰。画像を `GRID_ROWS` x `GRID_COLS` の
セルに分け、各セルが「物体があるか・その BBox・クラス・属性」を持つ。
NMS もアンカーも使わない**アンカーフリーの 1 セル 1 物体**方式で、
`config.PERCEP_MAX_DETECTIONS`（12）しか使わない用途には十分。

★ **エンコード（教師データ側）とデコード（推論側）を同じファイルに置くこと。**
  この 2 つがずれると、学習は損失が下がっているのに検出結果が全く合わない、
  という**外から絶対に気づけない**壊れ方をする。`encode_targets()` と
  `decode_detections()` は必ず対で直すこと。

計算量の目安（`build_detector()` の既定構成・入力 192x144x3）:
  8 台ぶんのバッチ推論で 20ms 以内に収まる規模にしてある。1 ステップ 20Hz で
  8 台ぶんの描画と推論が乗るので、ここが重いと倍速が出せなくなる。
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app import config
from app.percep.types import (
    DEFAULT_CAMERA,
    LANE_LOOKAHEAD_M,
    LANE_POLYLINE_POINTS,
    CameraSpec,
    DetClass,
    Detection,
    PerceptionResult,
    pack_by_class_quota,
)

logger = logging.getLogger("autoware_sim")

__all__ = [
    "CHANNELS",
    "Detector",
    "GRID_COLS",
    "GRID_ROWS",
    "SPEED_BINS_KMH",
    "build_detector",
    "decode_detections",
    "encode_freespace",
    "encode_targets",
    "speed_bin_index",
]


# ---------------------------------------------------------------------------
# 出力レイアウト
#
# ★ 並びを変えると学習済みの重みが全部ずれる。追加は末尾に足すこと
#   （`DetClass` と同じ約束）。
# ---------------------------------------------------------------------------

GRID_ROWS = 6
GRID_COLS = 8

NUM_CLASSES = len(DetClass)      # 5
NUM_PHASES = 3                   # 0=青 / 1=黄 / 2=赤

#: 規制速度のカテゴリ [km/h]。回帰ではなく分類にしてあるのは、
#: 標識に書かれているのが**離散した数字**だからで、49km/h という標識は無い。
#: 銀座の実データは 20 / 30 / 40 / 50 の 4 種類だが、`loader._DEFAULT_MAXSPEED_KPH`
#: は 60 / 80 も返しうるので余裕を持たせてある。
SPEED_BINS_KMH: tuple[int, ...] = (20, 30, 40, 50, 60, 70, 80, 100)
NUM_SPEED_BINS = len(SPEED_BINS_KMH)

# セルあたりのチャンネル配置
OFF_OBJ = 0                                   # 1ch  物体がある信頼度（sigmoid）
OFF_BOX = OFF_OBJ + 1                         # 4ch  セル内オフセット cx,cy と 画像比 w,h
OFF_CLS = OFF_BOX + 4                         # 5ch  クラス（softmax）
OFF_PHASE = OFF_CLS + NUM_CLASSES             # 3ch  灯色（softmax）
OFF_SPEED = OFF_PHASE + NUM_PHASES            # 8ch  規制速度（softmax）
OFF_DIST = OFF_SPEED + NUM_SPEED_BINS         # 1ch  距離 / CameraSpec.far（sigmoid）
OFF_LATERAL = OFF_DIST + 1                    # 1ch  横偏差 / OBS_LATERAL_RANGE（tanh）
CHANNELS = OFF_LATERAL + 1                    # = 23

#: 走行可能領域の出力本数。`Detection` には入れる場所が無いので独立した出力にする
FREESPACE_DIM = int(config.OBS_FREESPACE_DIM)

#: 出力テンソルの名前。`load()` はこの名前と形で読み込んだモデルを検証する
DET_OUTPUT_NAME = "detections"
FREESPACE_OUTPUT_NAME = "freespace"

#: 横偏差の正規化スケール [m]（観測側と同じ分母を使う）
LATERAL_SCALE = float(config.OBS_LATERAL_RANGE)
#: 走行可能距離の正規化スケール [m]
FREESPACE_SCALE = float(config.OBS_FREESPACE_MAX_DISTANCE)


def _import_keras():
    """Keras 3 を torch バックエンドで読み込む。

    `app/rl/export.py` の `_import_keras()` と同じ作法。**環境変数は
    `import keras` より前に設定しなければ効かない。** Keras 3 は既定で
    TensorFlow を探すが、この環境には入れていない（入れてもいけない）。
    torch は学習で既に使っているのでそれをバックエンドにする。

    export.py 側と実装を共有していないのは、`app.rl.export` を import すると
    PPO 一式（torch のモデル定義）まで読み込まれ、`percep` が `rl` に
    依存してしまうため。**片方を直したらもう片方も直すこと。**
    """
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras  # noqa: PLC0415

    return keras


# ---------------------------------------------------------------------------
# モデル定義
# ---------------------------------------------------------------------------


def build_detector(
    spec: CameraSpec = DEFAULT_CAMERA,
    *,
    width: float = 1.0,
    use_batchnorm: bool = True,
) -> Any:
    """軽量 CNN の認識器を作る（学習スクリプトからも使う）。

    Args:
        spec: 入力画像の大きさを決めるカメラ諸元。
        width: チャンネル数の倍率。速度が足りなければ 0.5 などに下げる。
        use_batchnorm: 畳み込みの後に BatchNormalization を入れるか。
            CPU では 1 層あたりの実行時間より**演算の呼び出し回数**が効くので、
            速度が足りないときはここを False にするのが最初の一手。

    Returns:
        `keras.Model`。出力は 2 つ:
            "detections": (N, GRID_ROWS, GRID_COLS, CHANNELS)
            "freespace" : (N, FREESPACE_DIM) — 0〜1 に正規化した走行可能距離

    ★ 解像度の落とし方は `192 -> 96 -> 48 -> 24 -> 8` / `144 -> 72 -> 36 -> 18 -> 6`。
      最後だけプール 3 なのは、6x8 のグリッドをちょうど作るため。
      **入力の大きさを変えるとここが割り切れなくなる**ので、
      `CameraSpec.width/height` を変えたらこの段数を必ず見直すこと。
    """
    keras = _import_keras()
    layers = keras.layers

    height_px = int(spec.height)
    width_px = int(spec.width)
    if height_px % (2 * 2 * 2 * 3) or width_px % (2 * 2 * 2 * 3):
        raise ValueError(
            f"入力サイズ {width_px}x{height_px} は 24 の倍数である必要があります"
            "（6x8 のグリッドを作れません）"
        )

    def ch(n: int) -> int:
        return max(8, int(round(n * float(width))))

    def conv(x, filters: int, name: str, strides: int = 1):
        x = layers.Conv2D(
            filters,
            3,
            strides=strides,
            padding="same",
            use_bias=not use_batchnorm,
            name=name,
        )(x)
        if use_batchnorm:
            x = layers.BatchNormalization(name=f"{name}_bn")(x)
        return layers.ReLU(name=f"{name}_relu")(x)

    image = keras.Input(shape=(height_px, width_px, 3), dtype="float32", name="image")
    # 正規化はモデルの中に入れる。外に置くと「学習時は /255、推論時は素通し」
    # のような食い違いが起きうるが、症状は「精度が出ない」だけで気づけない。
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(image)

    x = conv(x, ch(16), "stem", strides=2)      # 72 x 96
    x = layers.MaxPooling2D(2, name="pool1")(x)  # 36 x 48
    x = conv(x, ch(32), "block1")
    x = layers.MaxPooling2D(2, name="pool2")(x)  # 18 x 24
    x = conv(x, ch(64), "block2")
    x = layers.MaxPooling2D(3, name="pool3")(x)  # 6 x 8
    features = conv(x, ch(128), "block3")

    # --- 検出ヘッド ---
    # 活性が項目ごとに違うので 1x1 畳み込みを並べて連結する。
    # 1 本の Conv でまとめると sigmoid / softmax / tanh を混ぜられない。
    head = [
        layers.Conv2D(1, 1, activation="sigmoid", name="head_obj")(features),
        layers.Conv2D(4, 1, activation="sigmoid", name="head_box")(features),
        layers.Conv2D(NUM_CLASSES, 1, activation="softmax", name="head_cls")(features),
        layers.Conv2D(NUM_PHASES, 1, activation="softmax", name="head_phase")(features),
        layers.Conv2D(NUM_SPEED_BINS, 1, activation="softmax", name="head_speed")(features),
        layers.Conv2D(1, 1, activation="sigmoid", name="head_dist")(features),
        layers.Conv2D(1, 1, activation="tanh", name="head_lateral")(features),
    ]
    detections = layers.Concatenate(axis=-1, name=DET_OUTPUT_NAME)(head)

    # --- 走行可能領域ヘッド ---
    # 方向ごとの距離なので空間情報を潰してはいけない（GlobalAveragePooling は不可）。
    # 1x1 で絞ってから平坦化する。
    f = layers.Conv2D(ch(16), 1, activation="relu", name="free_reduce")(features)
    f = layers.Flatten(name="free_flatten")(f)
    f = layers.Dense(ch(64), activation="relu", name="free_dense")(f)
    freespace = layers.Dense(
        FREESPACE_DIM, activation="sigmoid", name=FREESPACE_OUTPUT_NAME
    )(f)

    return keras.Model(
        inputs=image, outputs=[detections, freespace], name="driverl_detector"
    )


# ---------------------------------------------------------------------------
# 教師データのエンコード（学習用）
# ---------------------------------------------------------------------------


def speed_bin_index(speed_limit_mps: float) -> int:
    """規制速度 [m/s] を最も近いカテゴリの添字に落とす。"""
    kmh = float(speed_limit_mps) * 3.6
    diffs = [abs(kmh - float(b)) for b in SPEED_BINS_KMH]
    return int(min(range(len(diffs)), key=diffs.__getitem__))


def encode_targets(
    results: Sequence[PerceptionResult], spec: CameraSpec = DEFAULT_CAMERA
) -> np.ndarray:
    """`PerceptionResult` の並びを学習用のターゲットテンソルに変換する。

    Returns:
        shape (N, GRID_ROWS, GRID_COLS, CHANNELS) float32。
        `[..., OFF_OBJ]` が 1 のセルだけが物体を持ち、これが損失のマスクを兼ねる。

    ★ **1 セルに 1 物体しか入らない**（YOLO v1 と同じ制限）。同じセルに
      落ちた検出は近いほうを残す。6x8 = 48 セルに対して検出は最大 12 件なので
      実害は小さいが、遠くの信号が並ぶ場面では取りこぼしうる。
    """
    n = len(results)
    target = np.zeros((n, GRID_ROWS, GRID_COLS, CHANNELS), dtype=np.float32)
    far = float(spec.far)

    for i, result in enumerate(results):
        # セルごとの「今入っている物体の距離」。近いものを優先する
        occupied: dict[tuple[int, int], float] = {}
        for det in result.detections:
            cx = (float(det.x0) + float(det.x1)) * 0.5
            cy = (float(det.y0) + float(det.y1)) * 0.5
            box_w = max(float(det.x1) - float(det.x0), 0.0)
            box_h = max(float(det.y1) - float(det.y0), 0.0)
            if box_w <= 0.0 or box_h <= 0.0:
                continue
            col = min(max(int(cx * GRID_COLS), 0), GRID_COLS - 1)
            row = min(max(int(cy * GRID_ROWS), 0), GRID_ROWS - 1)
            distance = float(det.distance) if det.distance is not None else far
            key = (row, col)
            if key in occupied and occupied[key] <= distance:
                continue
            occupied[key] = distance

            cell = target[i, row, col]
            cell[:] = 0.0
            cell[OFF_OBJ] = 1.0
            cell[OFF_BOX + 0] = np.clip(cx * GRID_COLS - col, 0.0, 1.0)
            cell[OFF_BOX + 1] = np.clip(cy * GRID_ROWS - row, 0.0, 1.0)
            cell[OFF_BOX + 2] = np.clip(box_w, 0.0, 1.0)
            cell[OFF_BOX + 3] = np.clip(box_h, 0.0, 1.0)
            cell[OFF_CLS + int(det.cls)] = 1.0
            if det.cls is DetClass.TRAFFIC_LIGHT and det.phase is not None:
                cell[OFF_PHASE + int(np.clip(int(det.phase), 0, NUM_PHASES - 1))] = 1.0
            if det.cls is DetClass.SPEED_SIGN and det.speed_limit is not None:
                cell[OFF_SPEED + speed_bin_index(float(det.speed_limit))] = 1.0
            cell[OFF_DIST] = np.clip(distance / max(far, 1e-6), 0.0, 1.0)
            if det.lateral is not None:
                cell[OFF_LATERAL] = np.clip(
                    float(det.lateral) / LATERAL_SCALE, -1.0, 1.0
                )
    return target


def encode_freespace(distances: np.ndarray) -> np.ndarray:
    """走行可能距離 [m] を 0〜1 に正規化する（`freespace` ヘッドの教師）。"""
    arr = np.asarray(distances, dtype=np.float32)
    return np.clip(arr / np.float32(FREESPACE_SCALE), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 推論結果のデコード
# ---------------------------------------------------------------------------


# ★ 車線の点数と長さの上限は `percep/types.py` にある（code_review Q-10）。
#   以前は `groundtruth.py` と同じ値をここにも書き、どちらのコメントも
#   「揃えること」と言うだけで、揃っていることを確かめる仕組みが無かった。
#: 線として描く下限の長さ [m]。これを割ったら「車線の長さが認識できていない」
#: とみなして点列を返さない（1 か所に潰れた線を描いても意味が無いため）。
LANE_MIN_DRAW_M = 0.1


def _lane_polyline(
    det: Detection, spec: CameraSpec, points: int = LANE_POLYLINE_POINTS
) -> list[tuple[float, float]]:
    """認識した横偏差と方位から、路面へ重ねる車線中心線を組み立てる。

    ★ **直線で近似する。** CNN が出すのは横偏差・方位・見えている長さまでで、
      車線の曲率は持っていない。そのためカーブでは実際の車線から離れていく。
      真値（`groundtruth`）は経路そのものをたどった曲線を返すので、
      **認識器を有効にすると画面の車線は直線的になる**。これは描画の手抜きでは
      なく、認識器が持っている情報がそこまでだということ。曲線を描きたいなら
      出力チャンネルを増やして曲率を学習させる必要がある。

    座標は自車座標系（前方 +x / 左 +y）。`encoder._bearing` と同じ符号規約で、
    画像の x は右向きに増えるので方位は符号を反転させる。
    """
    lateral = det.lateral
    if lateral is None or not math.isfinite(lateral):
        lateral = 0.0
    cx = (float(det.x0) + float(det.x1)) * 0.5
    bearing = (
        -math.atan2((cx - 0.5) * spec.width, spec.focal_px) if math.isfinite(cx) else 0.0
    )
    length = float(det.distance) if det.distance and math.isfinite(det.distance) else 0.0
    # ★ クランプするのは**上限だけ**（code_review P-05）。以前は下限 5.0m も
    #   掛けていたが、CNN が「1m 先までしか車線を認識できていない」と出しても
    #   線は必ず 5m 伸びるので、「ずれていればそのままずれて見える」という
    #   このオーバーレイの狙いを一部覆い隠していた。上限のほうは未学習の
    #   モデルが 57m まで伸ばした実例への対策なので残す。
    length = min(length, LANE_LOOKAHEAD_M)
    if length <= LANE_MIN_DRAW_M:
        return []

    # 車線中心は自車から見て横偏差のぶん反対側にある（lateral は左が正）
    y0 = -float(lateral)
    cos_b = math.cos(bearing)
    sin_b = math.sin(bearing)
    n = max(2, int(points))
    return [
        (length * (i / (n - 1)) * cos_b, y0 + length * (i / (n - 1)) * sin_b)
        for i in range(n)
    ]


def decode_detections(
    raw: np.ndarray,
    slots: Sequence[int],
    spec: CameraSpec = DEFAULT_CAMERA,
    *,
    conf_threshold: float = float(config.PERCEP_CONF_THRESHOLD),
    max_detections: int = int(config.PERCEP_MAX_DETECTIONS),
) -> list[PerceptionResult]:
    """モデル出力 (N, R, C, CHANNELS) を `PerceptionResult` の並びに戻す。

    `encode_targets()` の逆変換。**片方だけ直さないこと。**
    """
    grid = np.asarray(raw, dtype=np.float32)
    if grid.ndim != 4 or grid.shape[1:] != (GRID_ROWS, GRID_COLS, CHANNELS):
        raise ValueError(
            f"検出テンソルの形が想定と違います: {grid.shape} "
            f"（期待 (N, {GRID_ROWS}, {GRID_COLS}, {CHANNELS})）"
        )
    far = float(spec.far)
    out: list[PerceptionResult] = []

    for i, slot in enumerate(slots):
        cell = grid[i]
        obj = cell[:, :, OFF_OBJ]
        rows, cols = np.nonzero(obj >= np.float32(conf_threshold))
        if rows.size == 0:
            out.append(PerceptionResult(slot=int(slot)))
            continue
        # 信頼度の降順（`PerceptionResult` の約束）。
        # ★ ここで `[:max_detections]` と一括で切ってはいけない（code_review Q-01）。
        #   発火したセルが車両で埋まると信号・標識・車線が 1 件も残らず、
        #   観測が「信号は無い」になる一方で罰だけが真値から入る。
        #   クラス枠つきの詰め込み（`pack_by_class_quota`）は真値パスと共通。
        order = np.argsort(-obj[rows, cols], kind="stable")
        per_class: dict[DetClass, list[Detection]] = {c: [] for c in DetClass}
        for k in order:
            row, col = int(rows[k]), int(cols[k])
            values = cell[row, col]
            cx = (col + float(values[OFF_BOX + 0])) / GRID_COLS
            cy = (row + float(values[OFF_BOX + 1])) / GRID_ROWS
            half_w = float(values[OFF_BOX + 2]) * 0.5
            half_h = float(values[OFF_BOX + 3]) * 0.5
            cls = DetClass(int(np.argmax(values[OFF_CLS:OFF_CLS + NUM_CLASSES])))
            det = Detection(
                cls=cls,
                x0=float(np.clip(cx - half_w, 0.0, 1.0)),
                y0=float(np.clip(cy - half_h, 0.0, 1.0)),
                x1=float(np.clip(cx + half_w, 0.0, 1.0)),
                y1=float(np.clip(cy + half_h, 0.0, 1.0)),
                confidence=float(values[OFF_OBJ]),
                distance=float(values[OFF_DIST]) * far,
            )
            if cls is DetClass.TRAFFIC_LIGHT:
                det.phase = int(np.argmax(values[OFF_PHASE:OFF_PHASE + NUM_PHASES]))
            elif cls is DetClass.SPEED_SIGN:
                bin_index = int(np.argmax(values[OFF_SPEED:OFF_SPEED + NUM_SPEED_BINS]))
                det.speed_limit = float(SPEED_BINS_KMH[bin_index]) / 3.6
            elif cls is DetClass.LANE:
                det.lateral = float(values[OFF_LATERAL]) * LATERAL_SCALE
                # 路面へ重ねて描くための中心線。矩形だけでは認識のずれが見えない
                det.lane_points = _lane_polyline(det, spec)
            per_class[cls].append(det)
        out.append(
            PerceptionResult(
                slot=int(slot),
                detections=pack_by_class_quota(per_class, max_detections),
            )
        )
    return out


# ---------------------------------------------------------------------------
# 認識器
# ---------------------------------------------------------------------------


class Detector:
    """学習済みの CNN で擬似カメラ画像から検出する。

    `load()` が None を返したら、呼び出し側は真値へフォールバックすること
    （`config.PERCEP_FALLBACK_GROUND_TRUTH` / `percep.groundtruth`）。
    """

    def __init__(self, model: Any, spec: CameraSpec = DEFAULT_CAMERA) -> None:
        self.model = model
        self.spec = spec
        self._torch = None
        try:  # torch バックエンドなら no_grad で回す（勾配グラフを作らせない）
            import torch  # noqa: PLC0415

            self._torch = torch
        except ImportError:  # pragma: no cover - 他バックエンド用の逃げ道
            self._torch = None

    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls, path: Path, spec: CameraSpec = DEFAULT_CAMERA
    ) -> "Detector | None":
        """学習済みモデルを読む。無ければ / 形が違えば None。

        ★ **形が違うモデルを黙って受け入れない。** グリッドやチャンネル数が
          変わった重みをそのまま使うと、デコードが別の意味のチャンネルを読み、
          「検出はされるが中身が全部でたらめ」という気づけない壊れ方をする。
          `PPOTrainer.load()` が `hidden_sizes` を検証するのと同じ理由。
        """
        path = Path(path)
        if not path.exists():
            logger.info("認識器が見つかりません（真値へフォールバックします）: %s", path)
            return None
        try:
            keras = _import_keras()
            # compile=False。学習用の損失は train_detector.py 側の自作関数なので、
            # 推論のためだけに読むこちらでは復元しない（custom_objects も不要になる）
            model = keras.models.load_model(path, compile=False)
        except Exception:
            logger.exception(
                "認識器の読み込みに失敗しました（真値へフォールバックします）: %s", path
            )
            return None

        problem = cls._validate(model, spec)
        if problem:
            logger.warning(
                "認識器の形が想定と違うため使いません（真値へフォールバックします）: %s / %s",
                path,
                problem,
            )
            return None
        return cls(model, spec)

    @staticmethod
    def _validate(model: Any, spec: CameraSpec) -> str:
        """入出力の形を検証する。問題があれば理由、無ければ空文字。"""
        try:
            in_shape = tuple(model.input_shape)
            out_shapes = model.output_shape
        except Exception as exc:  # pragma: no cover
            return f"形が取得できません: {exc}"

        expected_in = (None, int(spec.height), int(spec.width), 3)
        if in_shape != expected_in:
            return f"入力 {in_shape} != 期待 {expected_in}"

        if not isinstance(out_shapes, (list, tuple)) or len(out_shapes) != 2:
            return f"出力が 2 本ではありません: {out_shapes}"
        det_shape = tuple(out_shapes[0])
        free_shape = tuple(out_shapes[1])
        if det_shape != (None, GRID_ROWS, GRID_COLS, CHANNELS):
            return (
                f"検出出力 {det_shape} != 期待 "
                f"{(None, GRID_ROWS, GRID_COLS, CHANNELS)}"
            )
        if free_shape != (None, FREESPACE_DIM):
            return f"走行可能領域出力 {free_shape} != 期待 {(None, FREESPACE_DIM)}"
        return ""

    # ------------------------------------------------------------------

    def _forward(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, H, W, 3) uint8 -> (検出テンソル, 走行可能領域)。

        正規化（/255）はモデルの `Rescaling` 層が持っているので、ここでは
        float32 にするだけ。**両方でやると 1/65025 になって真っ黒になる。**
        """
        batch = np.asarray(images)
        if batch.ndim != 4 or batch.shape[3] != 3:
            raise ValueError(f"画像の形が想定と違います: {batch.shape}（(N, H, W, 3) が必要）")
        if batch.shape[1] != int(self.spec.height) or batch.shape[2] != int(self.spec.width):
            raise ValueError(
                f"画像の大きさ {batch.shape[2]}x{batch.shape[1]} が "
                f"CameraSpec {self.spec.width}x{self.spec.height} と違います"
            )
        batch = batch.astype(np.float32, copy=False)

        if self._torch is not None:
            with self._torch.no_grad():
                det, free = self.model(batch, training=False)
        else:  # pragma: no cover - 他バックエンド用
            det, free = self.model(batch, training=False)

        return _to_numpy(det), _to_numpy(free)

    def detect(
        self, images: np.ndarray, slots: Sequence[int]
    ) -> list[PerceptionResult]:
        """(N, H, W, 3) uint8 の画像から検出する。

        `slots` は各画像がどの車両のものかを表すスロット番号で、
        画像の枚数と同じ長さでなければならない。
        """
        slots = list(int(s) for s in slots)
        if len(slots) != int(np.asarray(images).shape[0]):
            raise ValueError(
                f"画像 {np.asarray(images).shape[0]} 枚に対しスロットが {len(slots)} 個です"
            )
        det, _ = self._forward(images)
        return decode_detections(det, slots, self.spec)

    def detect_with_freespace(
        self, images: np.ndarray, slots: Sequence[int]
    ) -> tuple[list[PerceptionResult], np.ndarray]:
        """検出結果と走行可能領域 [m] (N, FREESPACE_DIM) をまとめて返す。

        `PerceptionResult` に走行可能領域を入れる場所が無いので別に返す
        （`percep.groundtruth.freespace_ground_truth()` と同じ約束）。
        """
        slots = list(int(s) for s in slots)
        det, free = self._forward(images)
        results = decode_detections(det, slots, self.spec)
        return results, (np.asarray(free, dtype=np.float32) * np.float32(FREESPACE_SCALE))


def _to_numpy(tensor: Any) -> np.ndarray:
    """バックエンドのテンソルを numpy へ落とす。"""
    if isinstance(tensor, np.ndarray):
        return tensor
    detach = getattr(tensor, "detach", None)
    if detach is not None:
        return detach().cpu().numpy()
    return np.asarray(tensor)
