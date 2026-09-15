# -*- coding: utf-8 -*-
"""擬似カメラ画像から検出結果を作る軽量 CNN 認識器。"""

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

GRID_ROWS = 6
GRID_COLS = 8

NUM_CLASSES = len(DetClass)
NUM_PHASES = 3

SPEED_BINS_KMH: tuple[int, ...] = (20, 30, 40, 50, 60, 70, 80, 100)
NUM_SPEED_BINS = len(SPEED_BINS_KMH)

OFF_OBJ = 0
OFF_BOX = OFF_OBJ + 1
OFF_CLS = OFF_BOX + 4
OFF_PHASE = OFF_CLS + NUM_CLASSES
OFF_SPEED = OFF_PHASE + NUM_PHASES
OFF_DIST = OFF_SPEED + NUM_SPEED_BINS
OFF_LATERAL = OFF_DIST + 1
CHANNELS = OFF_LATERAL + 1

FREESPACE_DIM = int(config.OBS_FREESPACE_DIM)

DET_OUTPUT_NAME = "detections"
FREESPACE_OUTPUT_NAME = "freespace"

LATERAL_SCALE = float(config.OBS_LATERAL_RANGE)
FREESPACE_SCALE = float(config.OBS_FREESPACE_MAX_DISTANCE)


def _import_keras():
    """Keras 3 を torch バックエンドで読み込む。"""
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras  # noqa: PLC0415

    return keras


def build_detector(
    spec: CameraSpec = DEFAULT_CAMERA,
    *,
    width: float = 1.0,
    use_batchnorm: bool = True,
) -> Any:
    """軽量 CNN の認識器を作る（学習スクリプトからも使う）。"""
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
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(image)

    x = conv(x, ch(16), "stem", strides=2)
    x = layers.MaxPooling2D(2, name="pool1")(x)
    x = conv(x, ch(32), "block1")
    x = layers.MaxPooling2D(2, name="pool2")(x)
    x = conv(x, ch(64), "block2")
    x = layers.MaxPooling2D(3, name="pool3")(x)
    features = conv(x, ch(128), "block3")

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

    f = layers.Conv2D(ch(16), 1, activation="relu", name="free_reduce")(features)
    f = layers.Flatten(name="free_flatten")(f)
    f = layers.Dense(ch(64), activation="relu", name="free_dense")(f)
    freespace = layers.Dense(
        FREESPACE_DIM, activation="sigmoid", name=FREESPACE_OUTPUT_NAME
    )(f)

    return keras.Model(
        inputs=image, outputs=[detections, freespace], name="driverl_detector"
    )


def speed_bin_index(speed_limit_mps: float) -> int:
    """規制速度 [m/s] を最も近いカテゴリの添字に落とす。"""
    kmh = float(speed_limit_mps) * 3.6
    diffs = [abs(kmh - float(b)) for b in SPEED_BINS_KMH]
    return int(min(range(len(diffs)), key=diffs.__getitem__))


def encode_targets(
    results: Sequence[PerceptionResult], spec: CameraSpec = DEFAULT_CAMERA
) -> np.ndarray:
    """`PerceptionResult` の並びを学習用のターゲットテンソルに変換する。"""
    n = len(results)
    target = np.zeros((n, GRID_ROWS, GRID_COLS, CHANNELS), dtype=np.float32)
    far = float(spec.far)

    for i, result in enumerate(results):
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


LANE_MIN_DRAW_M = 0.1


def _lane_polyline(
    det: Detection, spec: CameraSpec, points: int = LANE_POLYLINE_POINTS
) -> list[tuple[float, float]]:
    """認識した横偏差と方位から、路面へ重ねる車線中心線を組み立てる。"""
    lateral = det.lateral
    if lateral is None or not math.isfinite(lateral):
        lateral = 0.0
    cx = (float(det.x0) + float(det.x1)) * 0.5
    bearing = (
        -math.atan2((cx - 0.5) * spec.width, spec.focal_px) if math.isfinite(cx) else 0.0
    )
    length = float(det.distance) if det.distance and math.isfinite(det.distance) else 0.0
    length = min(length, LANE_LOOKAHEAD_M)
    if length <= LANE_MIN_DRAW_M:
        return []

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
    """モデル出力 (N, R, C, CHANNELS) を `PerceptionResult` の並びに戻す。"""
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
                det.lane_points = _lane_polyline(det, spec)
            per_class[cls].append(det)
        out.append(
            PerceptionResult(
                slot=int(slot),
                detections=pack_by_class_quota(per_class, max_detections),
            )
        )
    return out


class Detector:
    """学習済みの CNN で擬似カメラ画像から検出する。"""

    def __init__(self, model: Any, spec: CameraSpec = DEFAULT_CAMERA) -> None:
        self.model = model
        self.spec = spec
        self._torch = None
        try:
            import torch  # noqa: PLC0415

            self._torch = torch
        except ImportError:  # pragma: no cover - 他バックエンド用の逃げ道
            self._torch = None

    @classmethod
    def load(
        cls, path: Path, spec: CameraSpec = DEFAULT_CAMERA
    ) -> "Detector | None":
        """学習済みモデルを読む。無ければ / 形が違えば None。"""
        path = Path(path)
        if not path.exists():
            logger.info("認識器が見つかりません（真値へフォールバックします）: %s", path)
            return None
        try:
            keras = _import_keras()
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

    def _forward(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, H, W, 3) uint8 -> (検出テンソル, 走行可能領域)。"""
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
        """(N, H, W, 3) uint8 の画像から検出する。"""
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
        """検出結果と走行可能領域 [m] (N, FREESPACE_DIM) をまとめて返す。"""
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
