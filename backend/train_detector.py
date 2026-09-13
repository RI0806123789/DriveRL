"""擬似カメラ画像から信号・標識・車線・車両・障害物を検出する CNN を学習する。

    cd backend
    .venv\\Scripts\\python.exe train_detector.py --samples 2400 --epochs 12

★ **中身は `app/percep/trainer.py` にある。ここはその CLI の皮でしかない。**
  同じ処理を Web アプリの操作パネル「モデル作成」タブからも走らせるため、
  収集・損失・学習・検証の実装は両方から呼べる場所へ移してある。
  **アルゴリズムを直すときは `app/percep/trainer.py` を直すこと。**
  ここに書き足すと、画面から回したときにだけ効かない変更になる。

流れ:

    1. マップを読み込んで `SimulationEnv` を走らせる
    2. 各ステップで擬似カメラ画像と「理想の検出結果」（world の真値）を集める
    3. データセットを `data/detector/dataset/` へ保存する
    4. `build_detector()` を学習して `data/detector/detector.keras` へ保存する

学習が終わると、次回サーバーを起動したときに `SimulationEnv` が自動で
このモデルを読み込み、**観測が真値フォールバックから実際の CNN の出力へ切り替わる**
（`app/sim/env.py` の `_ensure_percep()`）。画面から回した場合は、その場で
載せ替えるので再起動は要らない（`runtime/detector_job.py`）。

★ 教師データは `percep/groundtruth.py` が world の真値から作る。これは
  認識器が未学習のときのフォールバックと**同じ関数**で、だからこそ
  「まず走らせる → 教師データを集める → 学習する → 差し替える」という
  順序で立ち上げられる。

★ **CLI から学習を回している間はサーバーを止めておくこと。** どちらも CPU を
  使い切るので、同時に動かすと学習ループのステップ時間が跳ね上がる。
  「モデル作成」タブから回す場合は、サーバー側が自分でシミュレーションを
  止めてから始めるので気にしなくてよい。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from app import config
from app.map import build_map_index, get_preset, list_presets, load_map
from app.percep import trainer
from app.percep.trainer import DATASET_FILE

__all__ = ["DATASET_FILE", "main"]


def _print_collect_progress(collected: int, samples: int, elapsed: float) -> None:
    print(f"  {collected} 枚 / {samples}  ({elapsed:.0f}s)", end="\r")


def _print_summary(summary: trainer.DatasetSummary) -> None:
    """何が写っているかを出す。**ここが偏っていると学習しても検出できない。**"""
    print(
        f"  物体のあるセル: {summary.object_cell_ratio * 100:.2f}%"
        f"（1 枚あたり {summary.objects_per_image:.1f} 個）"
    )
    print("  クラス内訳:", summary.class_counts)


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
        print(f"[収集] マップ {args.preset} を読み込みます")
        index = build_map_index(load_map(get_preset(args.preset)))
        started = time.perf_counter()
        data = trainer.collect_dataset(
            index, int(args.samples), on_progress=_print_collect_progress
        )
        print(
            f"\n[収集] 完了: {data['images'].shape[0]} 枚"
            f"（{time.perf_counter() - started:.0f}s）"
        )
        _print_summary(trainer.summarize_dataset(data))

        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dataset_path, **data)
        size_mb = dataset_path.stat().st_size / 1024 / 1024
        print(f"[保存] {dataset_path}（{size_mb:.0f} MB）")

    if args.collect_only:
        return 0

    result = trainer.fit_detector(
        data,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        out_path=Path(args.out),
        width=float(args.width),
        log=lambda message: print(f"[学習] {message}"),
    )
    if result.verify_counts:
        print(f"[確認] 読み直して推論できました。検出数 {result.verify_counts}")
    if result.warning:
        print(f"！ {result.warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
