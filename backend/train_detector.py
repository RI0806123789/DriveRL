"""擬似カメラ画像から信号・標識・車線・車両・障害物を検出する CNN を学習する。

    cd backend
    .venv\\Scripts\\python.exe train_detector.py --samples 2400 --epochs 12

流れ:

    1. マップを読み込んで `SimulationEnv` を走らせる
    2. 各ステップで擬似カメラ画像と「理想の検出結果」（world の真値）を集める
    3. データセットを `data/detector/dataset/` へ保存する
    4. `build_detector()` を学習して `data/detector/detector.keras` へ保存する

学習が終わると、次回サーバーを起動したときに `SimulationEnv` が自動で
このモデルを読み込み、**観測が真値フォールバックから実際の CNN の出力へ切り替わる**
（`app/sim/env.py` の `_ensure_percep()`）。

★ 教師データは `percep/groundtruth.py` が world の真値から作る。これは
  認識器が未学習のときのフォールバックと**同じ関数**で、だからこそ
  「まず走らせる → 教師データを集める → 学習する → 差し替える」という
  順序で立ち上げられる。

★ **学習を回している間はサーバーを止めておくこと。** どちらも CPU を使い切るので、
  同時に動かすと学習ループのステップ時間が跳ね上がる。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# `import keras` より前に置かないと効かない（app/rl/export.py と同じ作法）
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

from app import config
from app.contracts import SimParams
from app.map import build_map_index, get_preset, list_presets, load_map
from app.percep import detector as det
from app.percep.camera import PseudoCamera
from app.percep.groundtruth import detect_ground_truth_batch, freespace_ground_truth
from app.percep.types import DEFAULT_CAMERA, CameraSpec
from app.sim.env import SimulationEnv

DATASET_FILE = "detector_dataset.npz"


# ---------------------------------------------------------------------------
# 収集の下ごしらえ
#
# ★ **放っておくと車両と障害物の教師が 1 件も集まらない。**
#   8 台が銀座（道路総延長 19km）へ散ると互いに一度も視界に入らず、
#   障害物は誰も置かないので実測でどちらも 0 件だった。
#   教師に無いクラスは当然検出できるようにならないが、症状は
#   「走らせてみたら前の車を認識しない」という形でしか出ない。
# ---------------------------------------------------------------------------


def _cluster_vehicles(env: SimulationEnv, rng: np.random.Generator) -> None:
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


def _scatter_obstacles(env: SimulationEnv, rng: np.random.Generator) -> None:
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


def collect(
    preset_id: str,
    samples: int,
    *,
    spec: CameraSpec = DEFAULT_CAMERA,
    seed: int = 0,
    reset_every: int = 400,
) -> dict[str, np.ndarray]:
    """走らせながら画像と正解を集める。

    ★ **行動をランダムにする。** 学習済みの方策で走ると、通った場所の画だけが
      集まって偏る（信号の手前で止まっている画ばかりになる）。認識器には
      「車線から外れた画」「標識を斜めから見た画」も要る。

    `reset_every` ごとに全車を再スポーンして、同じ交差点に張り付くのを防ぐ。
    """
    print(f"[収集] マップ {preset_id} を読み込みます")
    index = build_map_index(load_map(get_preset(preset_id)))

    params = SimParams()
    params.vehicle_count = config.MAX_VEHICLES
    # ★ 観測を作らせない（code_review Q-02）。ここは画像も真値も自前で作るので、
    #   env にもう一度同じことをさせると擬似カメラ描画が二重になり、
    #   `detector.keras` が既にあると**捨てるためだけの CNN 推論**まで毎ステップ
    #   走る（実測で収集時間がおおむね倍になる）。`env.step()` の戻り値は
    #   使っていないので、観測が無くても収集結果は何も変わらない
    env = SimulationEnv(index, params, seed=seed, compute_observations=False)
    camera = PseudoCamera(index, spec)
    rng = np.random.default_rng(seed)

    images: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    frees: list[np.ndarray] = []

    # 走り出す前に寄せ集めてパイロンを置く（そうしないと車両・障害物が写らない）
    _cluster_vehicles(env, rng)
    _scatter_obstacles(env, rng)

    started = time.perf_counter()
    step = 0
    collected = 0  # 1 回の render で車両台数ぶん貯まるので、リスト長では数えない
    while collected < samples:
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
            _cluster_vehicles(env, rng)
        if step % 40 == 0:
            _scatter_obstacles(env, rng)

        if step % 20 == 0:
            print(
                f"  {collected} 枚 / {samples}  ({time.perf_counter() - started:.0f}s)",
                end="\r",
            )

    x = np.concatenate(images, axis=0)[:samples]
    y_det = np.concatenate(targets, axis=0)[:samples]
    y_free = np.concatenate(frees, axis=0)[:samples]
    print(f"\n[収集] 完了: {x.shape[0]} 枚 ({time.perf_counter() - started:.0f}s)")

    # 何が写っているかを出す。**ここが偏っていると学習しても検出できない。**
    obj = y_det[..., det.OFF_OBJ]
    print(f"  物体のあるセル: {float(obj.mean()) * 100:.2f}%（1 枚あたり {obj.sum() / len(x):.1f} 個）")
    cls_hist = y_det[..., det.OFF_CLS : det.OFF_CLS + det.NUM_CLASSES].sum(axis=(0, 1, 2))
    from app.percep.types import DetClass

    print("  クラス内訳:", {DetClass(i).name: int(v) for i, v in enumerate(cls_hist)})
    return {"images": x, "detections": y_det, "freespace": y_free}


# ---------------------------------------------------------------------------
# 2. 損失
# ---------------------------------------------------------------------------


def _make_detection_loss(keras, pos_weight: float = 20.0):
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


def train(
    data: dict[str, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    spec: CameraSpec = DEFAULT_CAMERA,
    out_path: Path,
    width: float = 1.0,
) -> None:
    keras = det._import_keras()

    model = det.build_detector(spec, width=width)
    print(f"[学習] パラメータ数 {model.count_params():,}")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={
            det.DET_OUTPUT_NAME: _make_detection_loss(keras),
            det.FREESPACE_OUTPUT_NAME: "mse",
        },
        loss_weights={det.DET_OUTPUT_NAME: 1.0, det.FREESPACE_OUTPUT_NAME: 10.0},
    )

    x = data["images"].astype(np.float32)
    y = {
        det.DET_OUTPUT_NAME: data["detections"].astype(np.float32),
        det.FREESPACE_OUTPUT_NAME: data["freespace"].astype(np.float32),
    }
    model.fit(
        x,
        y,
        epochs=int(epochs),
        batch_size=int(batch_size),
        validation_split=0.1,
        shuffle=True,
        verbose=2,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"[学習] 保存しました: {out_path}")

    # 実際に読み直せるか確かめる。`Detector.load()` は形も検証するので、
    # ここが通れば次回のサーバー起動でそのまま使われる。
    loaded = det.Detector.load(out_path, spec)
    if loaded is None:
        print("！ 保存したモデルを Detector.load() が受け付けませんでした")
        return
    sample = data["images"][: min(4, len(data["images"]))]
    results = loaded.detect(sample, list(range(len(sample))))
    counts = [len(r.detections) for r in results]
    print(f"[確認] 読み直して推論できました。検出数 {counts}")
    if not any(counts):
        print(
            "！ 何も検出しませんでした。エポック数かサンプル数を増やすか、"
            "収集時のクラス内訳が偏っていないか確認してください"
        )


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    presets = ", ".join(p.id for p in list_presets())
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", default="ginza", help=f"マップ（{presets}）")
    parser.add_argument("--samples", type=int, default=2400, help="集める画像の枚数")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--width", type=float, default=1.0, help="モデルのチャンネル倍率（速度が足りなければ 0.5）"
    )
    parser.add_argument("--collect-only", action="store_true", help="収集だけして終わる")
    parser.add_argument("--train-only", action="store_true", help="保存済みデータで学習だけする")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=config.DETECTOR_DATASET_DIR / DATASET_FILE,
        help="データセットの保存先",
    )
    parser.add_argument("--out", type=Path, default=config.DETECTOR_PATH)
    args = parser.parse_args(argv)

    if get_preset(args.preset) is None:
        print(f"未知のプリセットです: {args.preset}（{presets}）")
        return 2

    dataset_path = Path(args.dataset)
    if args.train_only:
        if not dataset_path.exists():
            print(f"データセットがありません: {dataset_path}")
            return 2
        print(f"[読込] {dataset_path}")
        with np.load(dataset_path) as npz:
            data = {k: npz[k] for k in npz.files}
    else:
        data = collect(args.preset, int(args.samples))
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dataset_path, **data)
        size_mb = dataset_path.stat().st_size / 1024 / 1024
        print(f"[保存] {dataset_path}（{size_mb:.0f} MB）")

    if args.collect_only:
        return 0

    train(
        data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        out_path=Path(args.out),
        width=float(args.width),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
