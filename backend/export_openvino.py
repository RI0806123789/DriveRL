"""学習済みの認識器（detector.keras）を OpenVINO の IR へ変換し、Keras / OpenVINO CPU / NPU の速さと出力の差を測る。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

from app import config
from app.percep import detector as det
from app.percep import openvino_backend as ovb
from app.percep.types import DEFAULT_CAMERA

__all__ = ["main"]


def _time(detector: det.Detector, images: np.ndarray, repeat: int) -> float:
    """中央値 [ms]。最初の 3 回は暖機として捨てる。"""
    for _ in range(3):
        detector._forward(images)
    samples = []
    for _ in range(repeat):
        started = time.perf_counter()
        detector._forward(images)
        samples.append((time.perf_counter() - started) * 1000.0)
    return float(np.median(samples))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keras", type=Path, default=config.DETECTOR_PATH, help="変換する認識器")
    parser.add_argument("--force", action="store_true", help="IR があっても作り直す")
    parser.add_argument("--bench", action="store_true", help="Keras / OpenVINO CPU / NPU の速さと差を測る")
    parser.add_argument("--batches", default="1,4,8,11", help="測るバッチ数（カンマ区切り）")
    parser.add_argument("--repeat", type=int, default=30)
    args = parser.parse_args(argv)

    if not ovb.is_installed():
        print("openvino が入っていません: backend\\.venv\\Scripts\\python.exe -m pip install openvino")
        return 2
    path = Path(args.keras)
    if not path.exists():
        print(f"認識器がありません: {path}（train_detector.py か「モデル作成」タブで学習してください）")
        return 2

    import openvino as ov  # noqa: PLC0415

    core = ov.Core()
    print(f"[OpenVINO] {ov.__version__}")
    for name in core.available_devices:
        print(f"  {name:6s} {core.get_property(name, 'FULL_DEVICE_NAME')}")

    shapes = det.openvino_shapes(DEFAULT_CAMERA)
    xml = ovb.ir_path(path, ovb.fingerprint(path))
    if args.force or not xml.exists():
        reference = det.Detector.load(path, DEFAULT_CAMERA, accelerate=False)
        if reference is None:
            print("Keras の認識器を読めませんでした（ログを見てください）")
            return 1
        started = time.perf_counter()
        try:
            xml = ovb.build_ir(reference.model, path, shapes, det._keras_forward_fn(reference.model))
        except Exception as exc:  # noqa: BLE001
            print(f"変換に失敗しました: {exc}")
            return 1
        print(f"[変換] {xml.name}（{time.perf_counter() - started:.1f}s・Keras との差は {ovb.CONVERT_TOLERANCE:g} 以内）")
    else:
        print(f"[変換] 作成済み: {xml.name}（作り直すなら --force）")

    if not args.bench:
        return 0

    batches = [int(b) for b in str(args.batches).split(",") if b.strip()]
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(max(batches), shapes.height, shapes.width, 3), dtype=np.uint8)
    keras_detector = det.Detector.load(path, DEFAULT_CAMERA, accelerate=False)
    assert keras_detector is not None
    want = keras_detector._forward(images)

    rows: list[tuple[str, det.Detector]] = [("keras", keras_detector)]
    for pref in ("cpu", "auto"):
        config.PERCEP_DEVICE = pref
        loaded = det.Detector.load(path, DEFAULT_CAMERA)
        if loaded is not None and all(loaded.backend != d.backend for _p, d in rows):
            rows.append((pref, loaded))

    print("[計測] 1 回の推論の中央値 [ms]")
    print("  " + "動かし先".ljust(48) + "".join(f"N={n:<6d}" for n in batches) + "Keras との最大差")
    for _pref, detector in rows:
        cells = "".join(f"{_time(detector, images[:n], int(args.repeat)):<8.2f}" for n in batches)
        got = detector._forward(images)
        diff = max(float(np.abs(got[0] - want[0]).max()), float(np.abs(got[1] - want[1]).max()))
        print(f"  {detector.backend[:46].ljust(48)}{cells}{diff:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
