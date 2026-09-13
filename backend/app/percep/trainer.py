# -*- coding: utf-8 -*-
"""認識器（CNN）の学習そのもの。教師データの収集・損失・学習・検証。

★ **ここは CLI（`backend/train_detector.py`）と Web UI（操作パネルの「モデル作成」
  タブ）が共有する唯一の実装。** どちらから走らせても同じ教師データ・同じ損失・
  同じ検証を通る。片方にだけ手を入れると「コマンドでは学習できるのに画面から
  やると結果が違う」という、外から切り分けようのない食い違いが生まれる。

違うのは**進捗の出し方だけ**なので、そこはコールバックで外へ出してある
（CLI は標準出力へ print、Web UI は `detector` メッセージとして WebSocket へ流す）。

流れ:

    1. マップ（`MapIndex`）の上で `SimulationEnv` を走らせる
    2. 各ステップで擬似カメラ画像と「理想の検出結果」（world の真値）を集める
    3. `build_detector()` を学習して `.keras` へ保存する
    4. 保存したものを `Detector.load()` で読み直し、実際に推論できるか確かめる

★ 教師データは `percep/groundtruth.py` が world の真値から作る。これは
  認識器が未学習のときのフォールバックと**同じ関数**で、だからこそ
  「まず走らせる → 教師データを集める → 学習する → 差し替える」という
  順序で立ち上げられる。

★ **`app.sim` の import はすべて関数の中で行う。** `sim.env` は `percep.encoder` を
  読むので、モジュールの先頭で import すると percep -> sim -> percep の循環になる。
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

# `import keras` より前に置かないと効かない（app/rl/export.py と同じ作法）
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

from app import config
from app.contracts import SimParams
from app.percep import detector as det
from app.percep.types import DEFAULT_CAMERA, CameraSpec, DetClass

if TYPE_CHECKING:  # 実行時に import しない（循環を避けるため）
    from app.contracts import MapIndex
    from app.sim.env import SimulationEnv

#: 教師データの保存ファイル名（`config.DETECTOR_DATASET_DIR` の下）
DATASET_FILE = "detector_dataset.npz"

#: 学習に使う検証データの割合。`model.fit(validation_split=...)` に渡す
VALIDATION_SPLIT = 0.1


class TrainingCancelled(Exception):
    """利用者が中断したときに送出する。失敗（エラー）とは区別すること。"""


#: 文字列を外へ出すためのコールバック（CLI は print、Web UI は状態メッセージ）
ProgressFn = Callable[[str], None]


# ---------------------------------------------------------------------------
# 収集の下ごしらえ
#
# ★ **放っておくと車両と障害物の教師が 1 件も集まらない。**
#   8 台が銀座（道路総延長 19km）へ散ると互いに一度も視界に入らず、
#   障害物は誰も置かないので実測でどちらも 0 件だった。
#   教師に無いクラスは当然検出できるようにならないが、症状は
#   「走らせてみたら前の車を認識しない」という形でしか出ない。
# ---------------------------------------------------------------------------


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
        # ★ `env.world` を直接触らない（code_review Q-11）。スロットを起こす経路は
        #   必ず `SimulationEnv` を通す約束で、そこでエピソード統計が落ちる
        for _ in range(8):
            radius = float(rng.uniform(8.0, 45.0))
            theta = float(rng.uniform(0.0, 2.0 * np.pi))
            at = (ax + radius * np.cos(theta), ay + radius * np.sin(theta))
            if env.relocate_vehicle(slot, at=at):
                break
        else:
            env.relocate_vehicle(slot)  # 近くに道が無ければ通常のスポーンで妥協する


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


# ---------------------------------------------------------------------------
# 1. データ収集
# ---------------------------------------------------------------------------


def collect_dataset(
    map_index: "MapIndex",
    samples: int,
    *,
    spec: CameraSpec = DEFAULT_CAMERA,
    seed: int = 0,
    reset_every: int = 400,
    on_progress: Callable[[int, int, float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, np.ndarray]:
    """走らせながら画像と正解を集める。

    ★ **行動をランダムにする。** 学習済みの方策で走ると、通った場所の画だけが
      集まって偏る（信号の手前で止まっている画ばかりになる）。認識器には
      「車線から外れた画」「標識を斜めから見た画」も要る。

    `reset_every` ごとに全車を再スポーンして、同じ交差点に張り付くのを防ぐ。

    Args:
        map_index: 走らせるマップ。**呼び出し側が用意する**（CLI はプリセットから
            読み、Web UI は読み込み済みのものをそのまま使う）。
        on_progress: `(集まった枚数, 目標枚数, 経過秒)` を受け取る。
        should_cancel: True を返したら `TrainingCancelled` を送出して抜ける。
    """
    from app.sim.env import SimulationEnv  # 循環 import を避けるため関数の中で読む
    from app.percep.camera import PseudoCamera
    from app.percep.groundtruth import detect_ground_truth_batch, freespace_ground_truth

    params = SimParams()
    params.vehicle_count = config.MAX_VEHICLES
    # ★ 観測を作らせない（code_review Q-02）。ここは画像も真値も自前で作るので、
    #   env にもう一度同じことをさせると擬似カメラ描画が二重になり、
    #   `detector.keras` が既にあると**捨てるためだけの CNN 推論**まで毎ステップ
    #   走る（実測で収集時間がおおむね倍になる）。`env.step()` の戻り値は
    #   使っていないので、観測が無くても収集結果は何も変わらない
    env = SimulationEnv(map_index, params, seed=seed, compute_observations=False)
    camera = PseudoCamera(map_index, spec)
    rng = np.random.default_rng(seed)

    images: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    frees: list[np.ndarray] = []

    # 走り出す前に寄せ集めてパイロンを置く（そうしないと車両・障害物が写らない）
    cluster_vehicles(env, rng)
    scatter_obstacles(env, rng)

    started = time.perf_counter()
    step = 0
    collected = 0  # 1 回の render で車両台数ぶん貯まるので、リスト長では数えない
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

        # 一様ランダムだとほとんど直進しないので、加速側に寄せて前へ進ませる
        action = np.zeros((config.MAX_VEHICLES, config.ACTION_DIM), dtype=np.float32)
        action[:, 0] = rng.uniform(-0.2, 1.0, size=config.MAX_VEHICLES)
        action[:, 1] = rng.uniform(-0.6, 0.6, size=config.MAX_VEHICLES)
        env.step(action)
        step += 1
        # 走っているうちに散っていくので、定期的に寄せ直してパイロンも置き直す
        if reset_every > 0 and step % reset_every == 0:
            env.reset_all()
            cluster_vehicles(env, rng)
        if step % 40 == 0:
            scatter_obstacles(env, rng)

        if on_progress is not None and step % 20 == 0:
            on_progress(min(collected, samples), samples, time.perf_counter() - started)

    x = np.concatenate(images, axis=0)[:samples]
    y_det = np.concatenate(targets, axis=0)[:samples]
    y_free = np.concatenate(frees, axis=0)[:samples]
    if on_progress is not None:
        on_progress(int(x.shape[0]), samples, time.perf_counter() - started)
    return {"images": x, "detections": y_det, "freespace": y_free}


@dataclass(frozen=True)
class DatasetSummary:
    """集めた教師データに何が写っているか。

    **ここが偏っていると、学習しても偏ったぶんは検出できるようにならない。**
    件数 0 のクラスは「走らせてみたら前の車を認識しない」という形でしか
    症状が出ないので、収集のたびに必ず外へ出す。
    """

    samples: int
    #: 物体があるセルの割合（0.0〜1.0）
    object_cell_ratio: float
    #: 1 枚あたりの物体数
    objects_per_image: float
    #: クラス名 -> 件数
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


# ---------------------------------------------------------------------------
# 2. 損失
# ---------------------------------------------------------------------------


def make_detection_loss(keras, pos_weight: float = 20.0):
    """検出ヘッドの損失（物体のあるセルだけ中身を見る）。

    ★ 物体があるセルは全体の数 % しかない。素の binary crossentropy だと
      **「どのセルにも何も無い」と答えるのが最適解**になり、学習は進んだのに
      何も検出しないモデルができあがる。正例に重みを掛けて釣り合わせる。
    """
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
        mask = y_true[..., off_obj : off_obj + 1]  # (B, R, C, 1)

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
        # 灯色と規制速度は該当クラスのセルでしか教師が立たない（他は全 0）。
        # 全 0 なら CE は 0 になるので、ここで別扱いする必要はない。
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


# ---------------------------------------------------------------------------
# 3. 学習
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    """学習の結果。画面にもコンソールにも同じものを出す。"""

    path: Path
    param_count: int
    epochs_run: int
    size_bytes: int
    #: エポックごとの `{"epoch", "loss", "valLoss"}`
    history: list[dict[str, float]] = field(default_factory=list)
    #: 読み直して推論したときの検出数（空なら検証に失敗）
    verify_counts: list[int] = field(default_factory=list)
    #: 検証で気づいたこと（何も検出しない等）。空なら問題なし
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
    """認識器を学習して保存し、**読み直して推論できるところまで**確かめる。

    Args:
        on_batch: `(エポック番号(1 始まり), 済んだバッチ数, 1 エポックのバッチ数)`。
        on_epoch: `(エポック番号, 総エポック数, {"loss", "valLoss"})`。
        should_cancel: True を返したら学習を打ち切り `TrainingCancelled` を送出する。
    """
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

    x = data["images"].astype(np.float32)
    y = {
        det.DET_OUTPUT_NAME: data["detections"].astype(np.float32),
        det.FREESPACE_OUTPUT_NAME: data["freespace"].astype(np.float32),
    }

    total_epochs = int(epochs)
    batches = steps_per_epoch(int(x.shape[0]), int(batch_size))
    history: list[dict[str, float]] = []

    class _Reporter(keras.callbacks.Callback):
        """進捗を外へ出し、中断要求を拾う。

        ★ 中断はバッチ境界でも見る。1 エポックは銀座 2,400 枚で十数秒かかるので、
          エポック境界だけで見ていると「中止」を押してから止まるまでがそれだけ空く。
        """

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
        # ★ 中断したモデルは**保存しない**。中途半端な重みで `detector.keras` を
        #   上書きすると、いま走っている認識器が黙って悪くなる。
        #   悪くなったことは「走らせてみたら認識が雑」という形でしか出ない
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

    # 実際に読み直せるか確かめる。`Detector.load()` は形も検証するので、
    # ここが通れば次回のサーバー起動でそのまま使われる。
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
