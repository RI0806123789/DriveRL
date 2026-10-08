# -*- coding: utf-8 -*-
"""認識器（CNN）の学習そのもの。教師データの収集・損失・学習・検証。"""

from __future__ import annotations

import math
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, TYPE_CHECKING

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

from app import config
from app.contracts import SimParams
from app.percep import detector as det
from app.percep import openvino_backend as ovb
from app.percep.evaluate import _attribute_ok, _match
from app.percep.types import CAMERAS_BY_KEY, DEFAULT_CAMERA, SIGN_CLASSES, SIGN_DIRECTIONS, SIGN_LABELS, CameraSpec, DetClass
from app.percep.weather import PRESETS, Weather
from app.warn import warn_once

if TYPE_CHECKING:
    from app.contracts import MapIndex
    from app.sim.env import SimulationEnv


DATASET_FILE = "detector_dataset.npz"

DATASET_VERSION = 5

VALIDATION_SPLIT = 0.1

#: 教師データの画をどのカメラで撮るか（割合）。同じ認識器を 4 台のカメラに通すので、
#: 前方だけで学習すると後方・左右の画（俯角が違い、写るものも違う）で取りこぼす
CAMERA_MIX: dict[str, float] = {
    "front": 0.5,
    "rear": 1.0 / 6.0,
    "left": 1.0 / 6.0,
    "right": 1.0 / 6.0,
}

#: 周囲カメラ用に車の周りへ置くパイロンの距離 [m]
AROUND_MIN_M = 3.0
AROUND_MAX_M = 18.0


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
    required = ("images", "detections", "freespace", "weather")
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
    if (
        int(data["detections"].shape[0]) != n
        or int(data["freespace"].shape[0]) != n
        or int(data["weather"].shape[0]) != n
    ):
        return "教師データの枚数が画像・検出・走行可能距離・天候で揃っていません。収集し直してください"
    return ""


def load_dataset(path: Path, spec: CameraSpec = DEFAULT_CAMERA) -> dict[str, np.ndarray]:
    """保存済みの教師データを読んで照合する。合わなければ `ValueError`。"""
    with np.load(path) as npz:
        data = {k: npz[k] for k in npz.files}
    why = check_dataset(data, spec)
    if why:
        raise ValueError(why)
    return data


def cluster_vehicles(
    env: "SimulationEnv", rng: np.random.Generator, *, tight: bool = False
) -> None:
    """車両を互いの視界に入る距離へ寄せ集める。"""
    slots = np.flatnonzero(env.world.fleet.active)
    if slots.size < 2:
        return
    far = 22.0 if tight else 45.0
    anchor = int(slots[0])
    ax = float(env.world.fleet.x[anchor])
    ay = float(env.world.fleet.y[anchor])
    for raw in slots[1:]:
        slot = int(raw)
        for _ in range(8):
            radius = float(rng.uniform(8.0, far))
            theta = float(rng.uniform(0.0, 2.0 * np.pi))
            at = (ax + radius * np.cos(theta), ay + radius * np.sin(theta))
            if env.relocate_vehicle(slot, at=at):
                break
        else:
            env.relocate_vehicle(slot)


def scatter_obstacles(
    env: "SimulationEnv", rng: np.random.Generator, *, focus: DetClass | None = None
) -> None:
    """各車両の前方と周り（後ろ・横）にパイロンを置く。置かないと OBSTACLE の教師が 0 件になる。"""
    env.world.clear_obstacles()
    dense = focus is DetClass.OBSTACLE
    per_vehicle = 3 if dense else 1
    reach = 16.0 if dense else 28.0
    for raw in np.flatnonzero(env.world.fleet.active):
        slot = int(raw)
        x = float(env.world.fleet.x[slot])
        y = float(env.world.fleet.y[slot])
        heading = float(env.world.fleet.heading[slot])
        for _ in range(per_vehicle):
            ahead = float(rng.uniform(6.0, reach))
            side = float(rng.uniform(-3.5, 3.5))
            env.world.add_obstacle(
                x + np.cos(heading) * ahead - np.sin(heading) * side,
                y + np.sin(heading) * ahead + np.cos(heading) * side,
                float(config.OBSTACLE_RADIUS),
            )
        # 後方・左右カメラの教師（後退 AEB はこれで後ろのパイロンに気づく）
        bearing = heading + float(rng.uniform(math.pi * 0.35, math.pi * 1.65))
        away = float(rng.uniform(AROUND_MIN_M, AROUND_MAX_M))
        env.world.add_obstacle(
            x + math.cos(bearing) * away,
            y + math.sin(bearing) * away,
            float(config.OBSTACLE_RADIUS),
        )


def scatter_pedestrians(env: "SimulationEnv", *, focus: DetClass | None = None) -> None:
    """歩行者を車両の周りへ散らす。"""
    world = env.world
    slots = np.flatnonzero(world.fleet.active)
    if slots.size == 0:
        world.relocate_pedestrians()
        return
    dense = focus is DetClass.PEDESTRIAN
    world.crowd.gather_near(
        world.fleet.x[slots].astype(np.float64),
        world.fleet.y[slots].astype(np.float64),
        world.fleet.heading[slots].astype(np.float64),
        reach=26.0 if dense else 115.0,
        ahead_only=dense,
    )


def scatter_props(
    env: "SimulationEnv", rng: np.random.Generator, *, focus: DetClass | None = None
) -> None:
    """パイロンと歩行者を置き直す。"""
    scatter_obstacles(env, rng, focus=focus)
    scatter_pedestrians(env, focus=focus)


FACE_TOLERANCE = math.cos(math.radians(55.0))

PLACE_TRIALS = 6

APPROACH_MIN_M = 18.0
APPROACH_MAX_M = 55.0


def _place_facing(
    env: "SimulationEnv",
    rng: np.random.Generator,
    positions: np.ndarray,
    headings: np.ndarray,
) -> None:
    """物体の手前へ、その物体に正対する向きで車両を置き直す。"""
    if positions.shape[0] == 0:
        cluster_vehicles(env, rng)
        return

    for raw in np.flatnonzero(env.world.fleet.active):
        slot = int(raw)
        placed = False
        for _ in range(PLACE_TRIALS):
            i = int(rng.integers(0, positions.shape[0]))
            heading = float(headings[i])
            back = float(rng.uniform(APPROACH_MIN_M, APPROACH_MAX_M))
            side = float(rng.uniform(-2.5, 2.5))
            at = (
                float(positions[i, 0]) - math.cos(heading) * back - math.sin(heading) * side,
                float(positions[i, 1]) - math.sin(heading) * back + math.cos(heading) * side,
            )
            if not env.relocate_vehicle(slot, at=at):
                continue
            mine = float(env.world.fleet.heading[slot])
            if math.cos(mine - heading) >= FACE_TOLERANCE:
                placed = True
                break
        if not placed:
            env.relocate_vehicle(slot)


def _signal_approaches(env: "SimulationEnv") -> tuple[np.ndarray, np.ndarray]:
    """信号の停止線の座標と、そこへ向かう進行方向。"""
    signals = env.world.map_index.data.signals
    if not signals:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)
    xy = np.array([(s.x, s.y) for s in signals], dtype=np.float64)
    heading = np.array([s.heading for s in signals], dtype=np.float64)
    return xy, heading


def _sign_approaches(env: "SimulationEnv", cls: DetClass = DetClass.SPEED_SIGN) -> tuple[np.ndarray, np.ndarray]:
    """指定した種類の標識の座標と、そこへ向かう進行方向。"""
    signs = [s for s in env.world.map_index.data.signs if SIGN_CLASSES.get(getattr(s, "kind", "speed_limit")) == cls]
    if not signs:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)
    xy = np.array([(s.x, s.y) for s in signs], dtype=np.float64)
    heading = np.array([s.heading for s in signs], dtype=np.float64)
    return xy, heading


def arrange_scene(
    env: "SimulationEnv", rng: np.random.Generator, focus: DetClass | None
) -> None:
    """狙うクラスに応じて車両とパイロンを置き直す。"""
    if focus is DetClass.TRAFFIC_LIGHT:
        _place_facing(env, rng, *_signal_approaches(env))
    elif focus in SIGN_CLASSES.values():
        _place_facing(env, rng, *_sign_approaches(env, focus))
    else:
        cluster_vehicles(env, rng, tight=focus is DetClass.VEHICLE)
    scatter_props(env, rng, focus=focus)


def _weighted_choice(
    rng: np.random.Generator, keys: Sequence[Any], weights: dict[Any, float] | None
) -> Any:
    """重みに比例して 1 つ選ぶ。重みが無ければ一様。"""
    if not keys:
        return None
    if not weights:
        return keys[int(rng.integers(0, len(keys)))]
    raw = np.array([max(float(weights.get(k, 1.0)), 0.0) for k in keys], dtype=np.float64)
    total = float(raw.sum())
    if total <= 1e-9:
        return keys[int(rng.integers(0, len(keys)))]
    return keys[int(rng.choice(len(keys), p=raw / total))]


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
    weathers: Sequence[str] | None = None,
    weather_focus: dict[str, float] | None = None,
    class_focus: dict[DetClass, float] | None = None,
    camera_mix: dict[str, float] | None = None,
    on_progress: Callable[[int, int, float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, np.ndarray]:
    """走らせながら画像と正解を集める。画は `camera_mix`（省略時 `CAMERA_MIX`）の割合で 4 台のカメラから撮る。"""
    from app.sim.env import SimulationEnv
    from app.percep.camera import PseudoCamera
    from app.percep.groundtruth import detect_ground_truth_batch, freespace_ground_truth

    params = SimParams()
    params.vehicle_count = config.MAX_VEHICLES
    params.pedestrian_count = config.MAX_PEDESTRIANS
    env = SimulationEnv(map_index, params, seed=seed, compute_observations=False)
    camera = PseudoCamera(map_index, spec)
    rng = np.random.default_rng(seed)
    if reset_every is None:
        reset_every = _reset_interval(samples)

    names = [name for name in (weathers or ("clear",)) if name in PRESETS] or ["clear"]
    classes = list(DetClass)
    mix = {key: weight for key, weight in (camera_mix or CAMERA_MIX).items() if key in CAMERAS_BY_KEY}
    mix_keys = list(mix) or [spec.key]

    images: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    frees: list[np.ndarray] = []
    skies: list[np.ndarray] = []

    def pick_focus() -> DetClass | None:
        """狙うクラスを選ぶ。"""
        return _weighted_choice(rng, classes, class_focus) if class_focus else None

    focus = pick_focus()
    arrange_scene(env, rng, focus)

    started = time.perf_counter()
    step = 0
    collected = 0
    while collected < samples:
        if should_cancel is not None and should_cancel():
            raise TrainingCancelled("教師データの収集を中断しました")

        slots = np.flatnonzero(env.world.fleet.active)
        if slots.size:
            weather: Weather = PRESETS[_weighted_choice(rng, names, weather_focus)]
            reach = min(
                float(config.OBS_FREESPACE_MAX_DISTANCE),
                weather.visibility_m(float(spec.far)),
            )
            # 車ごとにどのカメラで撮るかを選ぶ。同じカメラの車はまとめて描く
            views = [_weighted_choice(rng, mix_keys, mix) for _ in range(int(slots.size))]
            for key in mix_keys:
                group = slots[[i for i, v in enumerate(views) if v == key]]
                if group.size == 0:
                    continue
                view = spec if key == spec.key else CAMERAS_BY_KEY[key]
                frame = camera.render(env.world, group, weather, step, spec=view)
                results = detect_ground_truth_batch(env.world, group, view, weather)
                free = np.stack(
                    [freespace_ground_truth(env.world, int(s), view, reach) for s in group]
                )
                images.append(frame)
                targets.append(det.encode_targets(results, view))
                frees.append(det.encode_freespace(free))
                skies.append(
                    np.tile(
                        np.array([weather.rain, weather.fog], dtype=np.float32),
                        (int(frame.shape[0]), 1),
                    )
                )
                collected += int(frame.shape[0])

        action = np.zeros((config.MAX_VEHICLES, config.ACTION_DIM), dtype=np.float32)
        action[:, 0] = rng.uniform(-0.2, 1.0, size=config.MAX_VEHICLES)
        action[:, 1] = rng.uniform(-0.6, 0.6, size=config.MAX_VEHICLES)
        env.step(action)
        step += 1
        if reset_every > 0 and step % reset_every == 0:
            # 歩行者は直後の `arrange_scene` → `scatter_props` が車の周りへ
            # 寄せ直すので、ここで街へ撒き直さない
            env.reset_all(relocate_walkers=False)
            focus = pick_focus()
            arrange_scene(env, rng, focus)
        elif step % SCATTER_EVERY == 0:
            scatter_props(env, rng, focus=focus)

        if on_progress is not None and step % 20 == 0:
            on_progress(min(collected, samples), samples, time.perf_counter() - started)

    x = np.concatenate(images, axis=0)[:samples]
    y_det = np.concatenate(targets, axis=0)[:samples]
    y_free = np.concatenate(frees, axis=0)[:samples]
    y_sky = np.concatenate(skies, axis=0)[:samples]

    order = rng.permutation(int(x.shape[0]))
    x = x[order]
    y_det = y_det[order]
    y_free = y_free[order]
    y_sky = y_sky[order]

    if on_progress is not None:
        on_progress(int(x.shape[0]), samples, time.perf_counter() - started)
    return {
        "images": x,
        "detections": y_det,
        "freespace": y_free,
        "weather": y_sky,
        **dataset_meta(spec),
    }


@dataclass(frozen=True)
class DatasetSummary:
    """集めた教師データに何が写っているか。"""

    samples: int
    object_cell_ratio: float
    objects_per_image: float
    class_counts: dict[str, int]
    weather_counts: dict[str, int] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        return {
            "samples": int(self.samples),
            "objectCellRatio": float(self.object_cell_ratio),
            "objectsPerImage": float(self.objects_per_image),
            "classCounts": {k: int(v) for k, v in self.class_counts.items()},
            "weatherCounts": {k: int(v) for k, v in self.weather_counts.items()},
        }


def _weather_name(rain: float, fog: float) -> str:
    """収集時の (rain, fog) をいちばん近いプリセット名へ戻す。"""
    best = "clear"
    best_gap = float("inf")
    for name, preset in PRESETS.items():
        gap = abs(preset.rain - float(rain)) + abs(preset.fog - float(fog))
        if gap < best_gap:
            best = name
            best_gap = gap
    return best


def summarize_dataset(data: dict[str, np.ndarray]) -> DatasetSummary:
    """教師データの中身を数える（何が写っているか）。"""
    x = data["images"]
    y_det = data["detections"]
    n = max(1, int(x.shape[0]))
    obj = y_det[..., det.OFF_OBJ]
    cls_hist = y_det[..., det.OFF_CLS : det.OFF_CLS + det.NUM_CLASSES].sum(axis=(0, 1, 2))

    weather_counts: dict[str, int] = {}
    sky = data.get("weather")
    if sky is not None and sky.size:
        for rain, fog in np.asarray(sky, dtype=np.float32).reshape(-1, 2):
            name = _weather_name(float(rain), float(fog))
            weather_counts[name] = weather_counts.get(name, 0) + 1

    return DatasetSummary(
        samples=int(x.shape[0]),
        object_cell_ratio=float(obj.mean()),
        objects_per_image=float(obj.sum()) / n,
        class_counts={DetClass(i).name: int(v) for i, v in enumerate(cls_hist)},
        weather_counts=weather_counts,
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
        direction_loss = masked_ce(y_true, y_pred, det.OFF_DIRECTION, det.NUM_DIRECTIONS, mask)

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
            + direction_loss
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
    #: 検証に通って `path` を差し替えたか。False なら前の認識器がそのまま残っている
    installed: bool = False


def previous_detector_path(path: Path) -> Path:
    """差し替える前の認識器を 1 世代だけ残す先（`detector.prev.keras`）。"""
    return path.with_name(f"{path.stem}.prev{path.suffix}")


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
    # 保存先へ直接書かない。検証に落ちたモデルや書きかけのファイルで、動いている認識器を壊さないため
    staged = out_path.with_name(f"{out_path.stem}.staged{out_path.suffix}")
    result = FitResult(
        path=out_path,
        param_count=param_count,
        epochs_run=len(history),
        size_bytes=0,
        history=history,
    )
    try:
        model.save(staged)
        result.size_bytes = int(staged.stat().st_size)
        result.warning = _verify_saved(staged, data, spec, result)
        if result.warning:
            if log is not None:
                log(f"検証に通らなかったため保存しませんでした（前の認識器のまま）: {result.warning}")
            return result
        if out_path.exists():
            shutil.copy2(out_path, previous_detector_path(out_path))
        os.replace(staged, out_path)
        result.installed = True
    finally:
        staged.unlink(missing_ok=True)
    if log is not None:
        log(f"保存しました: {out_path}")
    result.warning = _refresh_openvino_ir(model, out_path, spec, data, log)
    return result


def _refresh_openvino_ir(
    model: Any,
    path: Path,
    spec: CameraSpec,
    data: dict[str, np.ndarray],
    log: ProgressFn | None,
) -> str:
    """差し替えた認識器の OpenVINO 版を作る。失敗しても Keras 版は動くので、理由を返すだけ（空なら成功か対象外）。"""
    if ovb.preference() == "keras" or not ovb.is_installed():
        return ""
    try:
        ir = ovb.build_ir(
            model,
            path,
            det.openvino_shapes(spec),
            det._keras_forward_fn(model),
            images=data["images"][:4],
        )
    except Exception:
        warn_once(
            "percep.openvino_build_ir",
            "学習した認識器を OpenVINO へ変換できませんでした。Keras の CPU 推論で動かします",
        )
        try:
            ovb.prune_stale(path, ovb.ir_path(path, ovb.fingerprint(path)))
        except OSError:
            warn_once("percep.openvino_prune_after_failure", "変換に失敗した後、古い IR を消せませんでした")
        return "OpenVINO への変換に失敗したため、認識器は Keras の CPU 推論で動かします（ログを見てください）"
    if log is not None:
        log(f"OpenVINO 版を作りました: {ir.name}")
    return ""


def _verify_saved(
    path: Path, data: dict[str, np.ndarray], spec: CameraSpec, result: FitResult
) -> str:
    """保存したモデルを読み直して推論する。問題があれば理由を返す（無ければ空文字）。"""
    loaded = det.Detector.load(path, spec, accelerate=False)
    if loaded is None:
        return "保存したモデルを Detector.load() が受け付けませんでした"
    class_targets = data["detections"][..., det.OFF_CLS : det.OFF_CLS + det.NUM_CLASSES]
    class_present = class_targets.sum(axis=(1, 2)) > 0
    indices = set(range(min(4, len(data["images"]))))
    for cls in DetClass:
        indices.update(np.flatnonzero(class_present[:, int(cls)])[:4].tolist())
    required_directions: set[tuple[DetClass, str]] = set()
    for cls in (DetClass.MANDATORY_DIRECTION_SIGN, DetClass.ONE_WAY_SIGN):
        class_mask = class_targets[..., int(cls)] > 0
        for i, direction in enumerate(SIGN_DIRECTIONS):
            present = np.any(class_mask & (data["detections"][..., det.OFF_DIRECTION + i] > 0), axis=(1, 2))
            if present.any():
                required_directions.add((cls, direction))
                indices.update(np.flatnonzero(present)[:4].tolist())
    seen: set[DetClass] = set()
    matched_signs: set[DetClass] = set()
    matched_directions: set[tuple[DetClass, str]] = set()
    result.verify_counts = []
    order = sorted(indices)
    for start in range(0, len(order), 4):
        selected = order[start : start + 4]
        results = loaded.detect(data["images"][selected], selected)
        truths = det.decode_detections(data["detections"][selected], selected, spec)
        result.verify_counts.extend(len(r.detections) for r in results)
        for truth, perception in zip(truths, results):
            seen.update(d.cls for d in perception.detections)
            for cls in SIGN_LABELS:
                pairs = _match(truth.by_class(cls), perception.by_class(cls))
                if pairs:
                    matched_signs.add(cls)
                for expected, predicted in pairs:
                    if _attribute_ok(cls, expected, predicted) is True:
                        matched_directions.add((cls, expected.direction))
    if not any(result.verify_counts):
        return (
            "読み直せましたが何も検出しませんでした。エポック数かサンプル数を増やすか、"
            "収集時のクラス内訳が偏っていないか確認してください"
        )
    missing = [cls.name for cls in DetClass if np.any(class_present[:, int(cls)]) and cls not in (matched_signs if cls in SIGN_LABELS else seen)]
    if missing:
        return f"教師データにあるクラスを検出できませんでした: {', '.join(missing)}。エポック数かサンプル数を増やしてください"
    missing_directions = required_directions - matched_directions
    if missing_directions:
        names = ", ".join(f"{cls.name}:{direction}" for cls, direction in sorted(missing_directions))
        return f"教師データにある標識の矢印を正しく認識できませんでした: {names}。エポック数かサンプル数を増やしてください"
    return ""
