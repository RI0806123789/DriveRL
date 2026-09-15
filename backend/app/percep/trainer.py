# -*- coding: utf-8 -*-
"""認識器（CNN）の学習そのもの。教師データの収集・損失・学習・検証。"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

from app import config
from app.contracts import SimParams
from app.percep import detector as det
from app.percep.types import DEFAULT_CAMERA, CameraSpec, DetClass

if TYPE_CHECKING:
    from app.contracts import MapIndex
    from app.sim.env import SimulationEnv


DATASET_FILE = "detector_dataset.npz"

DATASET_VERSION = 1

VALIDATION_SPLIT = 0.1


class TrainingCancelled(Exception):
    """利用者が中断したときに送出する。失敗（エラー）とは区別すること。"""


ProgressFn = Callable[[str], None]


def dataset_meta(spec: CameraSpec = DEFAULT_CAMERA) -> dict[str, np.ndarray]:
    """`np.savez_compressed(path, **data, **dataset_meta(spec))` に渡すメタ情報。"""
    return {
        "version": np.asarray(DATASET_VERSION, dtype=np.int32),
        "camera_width": np.asarray(spec.width, dtype=np.int32),
        "camera_height": np.asarray(spec.height, dtype=np.int32),
        "grid_rows": np.asarray(det.GRID_ROWS, dtype=np.int32),
        "grid_cols": np.asarray(det.GRID_COLS, dtype=np.int32),
        "channels": np.asarray(det.CHANNELS, dtype=np.int32),
        "freespace_bins": np.asarray(config.OBS_FREESPACE_DIM, dtype=np.int32),
    }


def check_dataset(data: dict[str, np.ndarray], spec: CameraSpec = DEFAULT_CAMERA) -> str:
    """読み込んだ教師データが今のコードと噛み合うか調べる。"""
    required = ("images", "detections", "freespace")
    missing = [k for k in required if k not in data]
    if missing:
        return (
            f"教師データに {', '.join(missing)} が入っていません。"
            "収集し直してください"
        )

    version = int(data["version"]) if "version" in data else 0
    if version != DATASET_VERSION:
        return (
            f"教師データの形式が古いか新しすぎます（ファイル: 版 {version} / "
            f"いまのコード: 版 {DATASET_VERSION}）。収集し直してください"
        )

    want = dataset_meta(spec)
    labels = {
        "camera_width": "カメラ幅",
        "camera_height": "カメラ高さ",
        "grid_rows": "検出グリッドの行数",
        "grid_cols": "検出グリッドの列数",
        "channels": "検出チャンネル数",
        "freespace_bins": "走行可能距離の分割数",
    }
    for key, label in labels.items():
        if key not in data:
            return f"教師データに {label} が記録されていません。収集し直してください"
        if int(data[key]) != int(want[key]):
            return (
                f"教師データの{label}が合いません"
                f"（ファイル: {int(data[key])} / いまのコード: {int(want[key])}）。"
                "収集し直してください"
            )

    n = int(data["images"].shape[0])
    if n == 0:
        return "教師データが空です。収集し直してください"
    if int(data["detections"].shape[0]) != n or int(data["freespace"].shape[0]) != n:
        return "教師データの枚数が画像・検出・走行可能距離で揃っていません。収集し直してください"
    return ""


def load_dataset(path: Path, spec: CameraSpec = DEFAULT_CAMERA) -> dict[str, np.ndarray]:
    """保存済みの教師データを読んで照合する。合わなければ `ValueError`。"""
    with np.load(path) as npz:
        data = {k: npz[k] for k in npz.files}
    why = check_dataset(data, spec)
    if why:
        raise ValueError(why)
    return data


def cluster_vehicles(env: "SimulationEnv", rng: np.random.Generator) -> None:
    """車両を互いの視界に入る距離へ寄せ集める。"""
    slots = np.flatnonzero(env.world.fleet.active)
    if slots.size < 2:
        return
    anchor = int(slots[0])
    ax = float(env.world.fleet.x[anchor])
    ay = float(env.world.fleet.y[anchor])
    for raw in slots[1:]:
        slot = int(raw)
        for _ in range(8):
            radius = float(rng.uniform(8.0, 45.0))
            theta = float(rng.uniform(0.0, 2.0 * np.pi))
            at = (ax + radius * np.cos(theta), ay + radius * np.sin(theta))
            if env.relocate_vehicle(slot, at=at):
                break
        else:
            env.relocate_vehicle(slot)


def scatter_obstacles(env: "SimulationEnv", rng: np.random.Generator) -> None:
    """各車両の前方にパイロンを置く。置かないと OBSTACLE の教師が 0 件になる。"""
    env.world.clear_obstacles()
    for raw in np.flatnonzero(env.world.fleet.active):
        slot = int(raw)
        x = float(env.world.fleet.x[slot])
        y = float(env.world.fleet.y[slot])
        heading = float(env.world.fleet.heading[slot])
        ahead = float(rng.uniform(6.0, 28.0))
        side = float(rng.uniform(-3.5, 3.5))
        env.world.add_obstacle(
            x + np.cos(heading) * ahead - np.sin(heading) * side,
            y + np.sin(heading) * ahead + np.cos(heading) * side,
            float(config.OBSTACLE_RADIUS),
        )


SCATTER_EVERY = 40

RESET_COUNT = 16


#: 散らし直しの最短間隔［ステップ］。これより短いと車がほとんど動かないまま撮り続ける。
RESET_MIN_INTERVAL = 5


def _reset_interval(samples: int) -> int:
    """`samples` 枚を集める間に `RESET_COUNT` 回だけ散らし直す間隔［ステップ］。"""
    steps = int(math.ceil(max(1, int(samples)) / max(1, int(config.MAX_VEHICLES))))
    return max(RESET_MIN_INTERVAL, steps // max(1, RESET_COUNT))


def collect_dataset(
    map_index: "MapIndex",
    samples: int,
    *,
    spec: CameraSpec = DEFAULT_CAMERA,
    seed: int = 0,
    reset_every: int | None = None,
    on_progress: Callable[[int, int, float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, np.ndarray]:
    """走らせながら画像と正解を集める。"""
    from app.sim.env import SimulationEnv
    from app.percep.camera import PseudoCamera
    from app.percep.groundtruth import detect_ground_truth_batch, freespace_ground_truth

    params = SimParams()
    params.vehicle_count = config.MAX_VEHICLES
    env = SimulationEnv(map_index, params, seed=seed, compute_observations=False)
    camera = PseudoCamera(map_index, spec)
    rng = np.random.default_rng(seed)
    if reset_every is None:
        reset_every = _reset_interval(samples)

    images: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    frees: list[np.ndarray] = []

    cluster_vehicles(env, rng)
    scatter_obstacles(env, rng)

    started = time.perf_counter()
    step = 0
    collected = 0
    while collected < samples:
        if should_cancel is not None and should_cancel():
            raise TrainingCancelled("教師データの収集を中断しました")

        slots = np.flatnonzero(env.world.fleet.active)
        if slots.size:
            frame = camera.render(env.world, slots)
            results = detect_ground_truth_batch(env.world, slots, spec)
            target = det.encode_targets(results, spec)
            free = np.stack(
                [
                    freespace_ground_truth(
                        env.world, int(s), spec, float(config.OBS_FREESPACE_MAX_DISTANCE)
                    )
                    for s in slots
                ]
            )
            images.append(frame)
            targets.append(target)
            frees.append(det.encode_freespace(free))
            collected += int(frame.shape[0])

        action = np.zeros((config.MAX_VEHICLES, config.ACTION_DIM), dtype=np.float32)
        action[:, 0] = rng.uniform(-0.2, 1.0, size=config.MAX_VEHICLES)
        action[:, 1] = rng.uniform(-0.6, 0.6, size=config.MAX_VEHICLES)
        env.step(action)
        step += 1
        if reset_every > 0 and step % reset_every == 0:
            env.reset_all()
            cluster_vehicles(env, rng)
            scatter_obstacles(env, rng)
        elif step % SCATTER_EVERY == 0:
            scatter_obstacles(env, rng)

        if on_progress is not None and step % 20 == 0:
            on_progress(min(collected, samples), samples, time.perf_counter() - started)

    x = np.concatenate(images, axis=0)[:samples]
    y_det = np.concatenate(targets, axis=0)[:samples]
    y_free = np.concatenate(frees, axis=0)[:samples]

    order = rng.permutation(int(x.shape[0]))
    x = x[order]
    y_det = y_det[order]
    y_free = y_free[order]

    if on_progress is not None:
        on_progress(int(x.shape[0]), samples, time.perf_counter() - started)
    return {"images": x, "detections": y_det, "freespace": y_free, **dataset_meta(spec)}


@dataclass(frozen=True)
class DatasetSummary:
    """集めた教師データに何が写っているか。"""

    samples: int
    object_cell_ratio: float
    objects_per_image: float
    class_counts: dict[str, int]

    def to_wire(self) -> dict[str, Any]:
        return {
            "samples": int(self.samples),
            "objectCellRatio": float(self.object_cell_ratio),
            "objectsPerImage": float(self.objects_per_image),
            "classCounts": {k: int(v) for k, v in self.class_counts.items()},
        }


def summarize_dataset(data: dict[str, np.ndarray]) -> DatasetSummary:
    """教師データの中身を数える（何が写っているか）。"""
    x = data["images"]
    y_det = data["detections"]
    n = max(1, int(x.shape[0]))
    obj = y_det[..., det.OFF_OBJ]
    cls_hist = y_det[..., det.OFF_CLS : det.OFF_CLS + det.NUM_CLASSES].sum(axis=(0, 1, 2))
    return DatasetSummary(
        samples=int(x.shape[0]),
        object_cell_ratio=float(obj.mean()),
        objects_per_image=float(obj.sum()) / n,
        class_counts={DetClass(i).name: int(v) for i, v in enumerate(cls_hist)},
    )


def make_detection_loss(keras, pos_weight: float = 20.0):
    """検出ヘッドの損失（物体のあるセルだけ中身を見る）。"""
    ops = keras.ops
    off_obj, off_box, off_cls = det.OFF_OBJ, det.OFF_BOX, det.OFF_CLS
    off_phase, off_speed = det.OFF_PHASE, det.OFF_SPEED
    off_dist, off_lat = det.OFF_DIST, det.OFF_LATERAL
    n_cls, n_phase, n_speed = det.NUM_CLASSES, det.NUM_PHASES, det.NUM_SPEED_BINS

    def masked_ce(y_true, y_pred, off: int, size: int, mask):
        t = y_true[..., off : off + size]
        p = ops.clip(y_pred[..., off : off + size], 1e-7, 1.0)
        return -ops.sum(t * ops.log(p), axis=-1, keepdims=True) * mask

    def loss(y_true, y_pred):
        mask = y_true[..., off_obj : off_obj + 1]

        obj_t = y_true[..., off_obj : off_obj + 1]
        obj_p = ops.clip(y_pred[..., off_obj : off_obj + 1], 1e-7, 1.0 - 1e-7)
        obj_loss = -(
            obj_t * ops.log(obj_p) * pos_weight + (1.0 - obj_t) * ops.log(1.0 - obj_p)
        )

        box_loss = ops.sum(
            ops.square(y_true[..., off_box : off_box + 4] - y_pred[..., off_box : off_box + 4]),
            axis=-1,
            keepdims=True,
        ) * mask

        cls_loss = masked_ce(y_true, y_pred, off_cls, n_cls, mask)
        phase_loss = masked_ce(y_true, y_pred, off_phase, n_phase, mask)
        speed_loss = masked_ce(y_true, y_pred, off_speed, n_speed, mask)

        dist_loss = ops.square(
            y_true[..., off_dist : off_dist + 1] - y_pred[..., off_dist : off_dist + 1]
        ) * mask
        lat_loss = ops.square(
            y_true[..., off_lat : off_lat + 1] - y_pred[..., off_lat : off_lat + 1]
        ) * mask

        total = (
            obj_loss
            + 5.0 * box_loss
            + cls_loss
            + phase_loss
            + speed_loss
            + 2.0 * dist_loss
            + lat_loss
        )
        return ops.mean(total)

    return loss


@dataclass
class FitResult:
    """学習の結果。画面にもコンソールにも同じものを出す。"""

    path: Path
    param_count: int
    epochs_run: int
    size_bytes: int
    history: list[dict[str, float]] = field(default_factory=list)
    verify_counts: list[int] = field(default_factory=list)
    warning: str = ""


def steps_per_epoch(sample_count: int, batch_size: int) -> int:
    """`model.fit` が 1 エポックで回すバッチ数（進捗表示用）。"""
    train_count = int(sample_count * (1.0 - VALIDATION_SPLIT))
    return max(1, int(math.ceil(train_count / max(1, int(batch_size)))))


def fit_detector(
    data: dict[str, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    out_path: Path,
    spec: CameraSpec = DEFAULT_CAMERA,
    width: float = 1.0,
    on_batch: Callable[[int, int, int], None] | None = None,
    on_epoch: Callable[[int, int, dict[str, float]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    log: ProgressFn | None = None,
    verbose: int = 2,
) -> FitResult:
    """認識器を学習して保存し、**読み直して推論できるところまで**確かめる。"""
    keras = det._import_keras()

    model = det.build_detector(spec, width=width)
    param_count = int(model.count_params())
    if log is not None:
        log(f"パラメータ数 {param_count:,}")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={
            det.DET_OUTPUT_NAME: make_detection_loss(keras),
            det.FREESPACE_OUTPUT_NAME: "mse",
        },
        loss_weights={det.DET_OUTPUT_NAME: 1.0, det.FREESPACE_OUTPUT_NAME: 10.0},
    )

    why = check_dataset(data, spec)
    if why:
        raise ValueError(why)
    x = data["images"]
    y = {
        det.DET_OUTPUT_NAME: data["detections"].astype(np.float32),
        det.FREESPACE_OUTPUT_NAME: data["freespace"].astype(np.float32),
    }

    total_epochs = int(epochs)
    batches = steps_per_epoch(int(x.shape[0]), int(batch_size))
    history: list[dict[str, float]] = []

    class _Reporter(keras.callbacks.Callback):
        """進捗を外へ出し、中断要求を拾う。"""

        def __init__(self) -> None:
            super().__init__()
            self.epoch = 1

        def on_epoch_begin(self, epoch, logs=None):  # noqa: ANN001, ARG002
            self.epoch = int(epoch) + 1

        def on_train_batch_end(self, batch, logs=None):  # noqa: ANN001, ARG002
            if on_batch is not None:
                on_batch(self.epoch, int(batch) + 1, batches)
            if should_cancel is not None and should_cancel():
                self.model.stop_training = True

        def on_epoch_end(self, epoch, logs=None):  # noqa: ANN001
            entry = {
                "epoch": int(epoch) + 1,
                "loss": float((logs or {}).get("loss", float("nan"))),
                "valLoss": float((logs or {}).get("val_loss", float("nan"))),
            }
            history.append(entry)
            if on_epoch is not None:
                on_epoch(int(epoch) + 1, total_epochs, entry)
            if should_cancel is not None and should_cancel():
                self.model.stop_training = True

    model.fit(
        x,
        y,
        epochs=total_epochs,
        batch_size=int(batch_size),
        validation_split=VALIDATION_SPLIT,
        shuffle=True,
        verbose=verbose,
        callbacks=[_Reporter()],
    )

    if should_cancel is not None and should_cancel():
        raise TrainingCancelled("学習を中断しました（モデルは保存していません）")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    size_bytes = out_path.stat().st_size if out_path.exists() else 0
    if log is not None:
        log(f"保存しました: {out_path}")

    result = FitResult(
        path=out_path,
        param_count=param_count,
        epochs_run=len(history),
        size_bytes=int(size_bytes),
        history=history,
    )

    loaded = det.Detector.load(out_path, spec)
    if loaded is None:
        result.warning = "保存したモデルを Detector.load() が受け付けませんでした"
        return result

    sample = data["images"][: min(4, len(data["images"]))]
    results = loaded.detect(sample, list(range(len(sample))))
    result.verify_counts = [len(r.detections) for r in results]
    if not any(result.verify_counts):
        result.warning = (
            "読み直せましたが何も検出しませんでした。エポック数かサンプル数を増やすか、"
            "収集時のクラス内訳が偏っていないか確認してください"
        )
    return result
