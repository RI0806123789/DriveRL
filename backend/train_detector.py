"""擬似カメラ画像から信号・標識・車線・車両・障害物を検出する CNN を学習する。"""

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

from app.runtime.detector_job import (
    BATCH_MAX,
    BATCH_MIN,
    EPOCHS_MAX,
    EPOCHS_MIN,
    SAMPLES_MAX,
    SAMPLES_MIN,
    SEED_MAX,
    SEED_MIN,
    WIDTH_MAX,
    WIDTH_MIN,
)

__all__ = ["DATASET_FILE", "main"]


def _out_of_range(name: str, value: float, lo: float, hi: float) -> str:
    """値域外なら理由を返す（`parse_request` と同じ文面）。合っていれば空文字。"""
    if lo <= value <= hi:
        return ""
    return f"{name} は {lo}〜{hi} の範囲で指定してください（受け取った値: {value}）"


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
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="収集の乱数種。変えると別の教師データが集まる（同じ種なら毎回同じ）",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--collect-only", action="store_true", help="収集だけして終わる")
    mode_group.add_argument("--train-only", action="store_true", help="保存済みデータで学習だけする")
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

    for why in (
        _out_of_range("--samples", int(args.samples), SAMPLES_MIN, SAMPLES_MAX),
        _out_of_range("--epochs", int(args.epochs), EPOCHS_MIN, EPOCHS_MAX),
        _out_of_range("--batch-size", int(args.batch_size), BATCH_MIN, BATCH_MAX),
        _out_of_range("--width", float(args.width), WIDTH_MIN, WIDTH_MAX),
        _out_of_range("--seed", int(args.seed), SEED_MIN, SEED_MAX),
    ):
        if why:
            print(why)
            return 2

    dataset_path = Path(args.dataset)
    if args.train_only:
        if not dataset_path.exists():
            print(f"データセットがありません: {dataset_path}")
            return 2
        print(f"[読込] {dataset_path}")
        try:
            data = trainer.load_dataset(dataset_path)
        except ValueError as exc:
            print(f"！ {exc}")
            return 2
    else:
        print(f"[収集] マップ {args.preset} を読み込みます")
        index = build_map_index(load_map(get_preset(args.preset)))
        started = time.perf_counter()
        data = trainer.collect_dataset(
            index,
            int(args.samples),
            seed=int(args.seed),
            on_progress=_print_collect_progress,
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
