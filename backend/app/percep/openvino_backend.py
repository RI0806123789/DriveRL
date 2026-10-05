# -*- coding: utf-8 -*-
"""認識器の推論を OpenVINO（NPU / CPU）で動かす部品。openvino が無くても import できる。"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import shutil
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from app import config
from app.warn import warn_once

logger = logging.getLogger("autoware_sim")

__all__ = [
    "BucketedEngine",
    "OpenVinoRunner",
    "Shapes",
    "bucket_for",
    "bucket_sizes",
    "build_ir",
    "choose_device",
    "fingerprint",
    "ir_path",
    "is_installed",
    "open_runner",
    "preference",
    "prune_stale",
]

#: 変換のやり方を変えたら上げる（IR のファイル名に入るので、古い IR は読まれなくなる）
IR_TAG = "ir1"

DEVICE_CHOICES = ("auto", "cpu", "keras")
BUCKET_STEP = 4

#: 変換した CPU の fp32 が Keras の出力とずれてよい上限（実測 3.4e-6）
CONVERT_TOLERANCE = 1e-3
#: NPU（fp16）が CPU の fp32 とずれてよい上限。壊れた出力を弾くための値で、fp16 の丸め（実測 9e-3）は許す
NPU_TOLERANCE = 0.1
PROBE_SEED = 20261005

ForwardFn = Callable[[np.ndarray], "tuple[np.ndarray, np.ndarray]"]


@dataclass(frozen=True)
class Shapes:
    """認識器の入出力の形（バッチ軸を除く）。"""

    height: int
    width: int
    detections: tuple[int, ...]
    freespace: tuple[int, ...]


def is_installed() -> bool:
    """openvino が入っているか（import はしない）。"""
    return importlib.util.find_spec("openvino") is not None


def preference() -> str:
    """`config.PERCEP_DEVICE` を正規化する。知らない値は auto にして初回だけ知らせる。"""
    value = str(config.PERCEP_DEVICE).strip().lower()
    if value in DEVICE_CHOICES:
        return value
    warn_once(
        "percep.openvino_device_setting",
        f"DRIVERL_PERCEP_DEVICE={value!r} は使えない値です（{' / '.join(DEVICE_CHOICES)}）。auto として扱います",
    )
    return "auto"


def choose_device(available: Sequence[str], pref: str) -> str:
    """NPU があれば NPU、無ければ CPU。内蔵 GPU は 3D 描画と取り合うので選ばない。"""
    if pref == "cpu":
        return "CPU"
    if any(name == "NPU" or name.startswith("NPU.") for name in available):
        return "NPU"
    return "CPU"


def bucket_sizes(max_batch: int) -> tuple[int, ...]:
    """NPU は形を固定してコンパイルするので、使うバッチ数を `BUCKET_STEP` 刻みの少数に絞る。"""
    top = max(BUCKET_STEP, -(-int(max_batch) // BUCKET_STEP) * BUCKET_STEP)
    return tuple(range(BUCKET_STEP, top + 1, BUCKET_STEP))


def bucket_for(count: int, buckets: Sequence[int]) -> int:
    """`count` 枚が収まる最小のバケット。収まらなければ最大。"""
    for size in buckets:
        if count <= size:
            return int(size)
    return int(buckets[-1])


def fingerprint(path: Path) -> str:
    """モデルファイルの内容の短い指紋。IR のファイル名に入れて、取り違えを構造的に防ぐ。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def ir_path(keras_path: Path, fp: str) -> Path:
    """`detector.keras` の指紋 `fp` に対応する IR（`.xml`。重みは同名の `.bin`）。"""
    keras_path = Path(keras_path)
    return keras_path.with_name(f"{keras_path.stem}.{IR_TAG}-{fp}.xml")


def prune_stale(keras_path: Path, keep: Path) -> None:
    """`keep` の組以外の IR と、NPU のコンパイル済みキャッシュを消す（モデルを替えたあとの掃除）。"""
    keras_path = Path(keras_path)
    keep_names = {keep.name, keep.with_suffix(".bin").name}
    for stale in keras_path.parent.glob(f"{keras_path.stem}.ir*"):
        if stale.name not in keep_names:
            try:
                stale.unlink()
            except OSError:
                warn_once("percep.openvino_prune", f"古い IR を消せませんでした: {stale.name}")
    shutil.rmtree(cache_dir(), ignore_errors=True)


def cache_dir() -> Path:
    """NPU のコンパイル済みモデルを置く場所（次回以降のコンパイルを省く）。"""
    return Path(config.DETECTOR_DIR) / "ov_cache"


_CORE: Any = None
_CORE_LOCK = threading.Lock()


def _core() -> Any:
    global _CORE
    with _CORE_LOCK:
        if _CORE is None:
            import openvino as ov  # noqa: PLC0415

            _CORE = ov.Core()
        return _CORE


def _opset() -> Any:
    try:
        from openvino import opset13  # noqa: PLC0415
    except ImportError:  # pragma: no cover - 古い openvino
        from openvino.runtime import opset13  # noqa: PLC0415
    return opset13


def convert_keras(model: Any, shapes: Shapes) -> Any:
    """Keras（torch バックエンド）のモデルを、バッチ軸だけ動的な OpenVINO のモデルへ変換する。"""
    import openvino as ov  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from openvino.frontend.pytorch import ConversionExtension  # noqa: PLC0415

    opset = _opset()

    class _Forward(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, images):  # noqa: ANN001, ANN202
            detections, freespace = self.inner(images, training=False)
            return detections, freespace

    def subtract(context):  # noqa: ANN001, ANN202
        # aten::subtract には変換規則が無い（aten::sub の別名で、Keras の BatchNormalization が使う）
        return [opset.subtract(context.get_input(0), context.get_input(1)).output(0)]

    example = torch.zeros(1, shapes.height, shapes.width, 3, dtype=torch.float32)
    # Keras の形の判定がトレースで定数になる旨の警告。形はバッチ以外固定なので害は無く、一致は build_ir が確かめる
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        converted = ov.convert_model(
            _Forward(model).eval(),
            example_input=example,
            extension=[ConversionExtension("aten::subtract", subtract)],
        )
    converted.reshape({converted.input(0): ov.PartialShape([-1, shapes.height, shapes.width, 3])})
    return converted


def check_shapes(model: Any, shapes: Shapes) -> None:
    """読んだ（または変換した）モデルの入出力が、想定の形かを確かめる。違えば ValueError。"""
    if len(model.inputs) != 1 or len(model.outputs) != 2:
        raise ValueError(f"入力 {len(model.inputs)} 本・出力 {len(model.outputs)} 本（期待 1 本・2 本）")
    shape = model.input(0).partial_shape
    if shape.rank.get_length() != 4 or [str(d) for d in list(shape)[1:]] != [
        str(shapes.height),
        str(shapes.width),
        "3",
    ]:
        raise ValueError(f"入力の形 {shape} が期待 (?, {shapes.height}, {shapes.width}, 3) と違います")
    for index, expected in enumerate((shapes.detections, shapes.freespace)):
        out = model.output(index).partial_shape
        if [str(d) for d in list(out)[1:]] != [str(d) for d in expected]:
            raise ValueError(f"出力 {index} の形 {out} が期待 (?, {', '.join(map(str, expected))}) と違います")


def probe_images(shapes: Shapes, count: int = 3) -> np.ndarray:
    """変換と NPU の確認に使う、毎回同じ雑音の画像（0〜255 の float32）。"""
    rng = np.random.default_rng(PROBE_SEED)
    return rng.integers(0, 256, size=(count, shapes.height, shapes.width, 3)).astype(np.float32)


class _CpuEngine:
    """OpenVINO の CPU（fp32）。バッチ数は何枚でもそのまま通す。"""

    def __init__(self, model: Any) -> None:
        compiled = _core().compile_model(
            model, "CPU", {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"}
        )
        self._request = compiled.create_infer_request()

    def __call__(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result = self._request.infer({0: np.ascontiguousarray(batch, dtype=np.float32)})
        return np.array(result[0]), np.array(result[1])


class BucketedEngine:
    """バッチ数を固定した推論器の束。少ない枚数は 0 で埋めて次のバケットへ、多い枚数は最大のバケットで刻む。"""

    def __init__(self, engines: dict[int, ForwardFn], shapes: Shapes) -> None:
        if not engines:
            raise ValueError("バケットがありません")
        self._engines = dict(sorted(engines.items()))
        self.buckets = tuple(self._engines)
        self._buffers = {
            size: np.zeros((size, shapes.height, shapes.width, 3), dtype=np.float32)
            for size in self.buckets
        }

    def __call__(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        top = self.buckets[-1]
        dets: list[np.ndarray] = []
        frees: list[np.ndarray] = []
        for start in range(0, int(batch.shape[0]), top):
            part = batch[start : start + top]
            count = int(part.shape[0])
            size = bucket_for(count, self.buckets)
            buffer = self._buffers[size]
            buffer[:count] = part
            det, free = self._engines[size](buffer)
            dets.append(np.asarray(det)[:count])
            frees.append(np.asarray(free)[:count])
        return np.concatenate(dets, axis=0), np.concatenate(frees, axis=0)


def _compile_npu(model: Any, shapes: Shapes, buckets: Sequence[int]) -> BucketedEngine:
    import openvino as ov  # noqa: PLC0415

    core = _core()
    properties: dict[str, Any] = {"PERFORMANCE_HINT": "LATENCY"}
    try:
        cache = cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        properties["CACHE_DIR"] = str(cache)
    except OSError:
        warn_once("percep.openvino_cache", "NPU のコンパイル済みキャッシュの置き場を作れませんでした（毎回コンパイルします）")

    engines: dict[int, ForwardFn] = {}
    for size in buckets:
        fixed = model.clone()
        fixed.reshape({fixed.input(0): ov.PartialShape([int(size), shapes.height, shapes.width, 3])})
        request = core.compile_model(fixed, "NPU", properties).create_infer_request()

        def run(batch: np.ndarray, request: Any = request) -> tuple[np.ndarray, np.ndarray]:
            result = request.infer({0: batch})
            return np.array(result[0]), np.array(result[1])

        run(np.zeros((int(size), shapes.height, shapes.width, 3), dtype=np.float32))
        engines[int(size)] = run
    return BucketedEngine(engines, shapes)


class OpenVinoRunner:
    """OpenVINO で推論する。NPU が落ちたら、その場で CPU へ切り替えて続ける。"""

    def __init__(
        self,
        primary: ForwardFn,
        fallback: ForwardFn,
        *,
        device: str,
        description: str,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._lock = threading.Lock()
        self.device = device
        self.description = description

    def __call__(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if self.device == "NPU":
                try:
                    return self._primary(batch)
                except Exception:
                    warn_once(
                        "percep.openvino_npu_failed",
                        "NPU での推論に失敗しました。以後は OpenVINO の CPU で続けます"
                        "（NPU を使うには「モデル作成」タブで読み込み直すか、再起動してください）",
                    )
                    self.degrade()
            return self._fallback(batch)

    def degrade(self) -> None:
        """NPU をやめて CPU に切り替える。"""
        self.device = "CPU"
        self.description = "OpenVINO CPU"


_BUILD_LOCK = threading.Lock()


def build_ir(
    keras_model: Any,
    keras_path: Path,
    shapes: Shapes,
    keras_forward: ForwardFn,
    *,
    images: np.ndarray | None = None,
) -> Path:
    """変換 → Keras と突き合わせ → 書き出し → 古い IR の掃除。通ったときだけ IR のパスを返す。"""
    # エンジンと学習ジョブが同時に作ると、一時ファイルと掃除が互いを壊すので 1 本ずつ
    with _BUILD_LOCK:
        return _build_ir(keras_model, keras_path, shapes, keras_forward, images)


def _build_ir(
    keras_model: Any,
    keras_path: Path,
    shapes: Shapes,
    keras_forward: ForwardFn,
    images: np.ndarray | None,
) -> Path:
    import openvino as ov  # noqa: PLC0415

    keras_path = Path(keras_path)
    fp = fingerprint(keras_path)
    converted = convert_keras(keras_model, shapes)
    check_shapes(converted, shapes)

    probe = probe_images(shapes)
    if images is not None and len(images):
        probe = np.concatenate([np.asarray(images[:4], dtype=np.float32), probe], axis=0)
    expected_det, expected_free = keras_forward(probe)
    engine = _CpuEngine(converted)
    # 動的バッチの確認を兼ねて、枚数を変えて 2 回通す
    for part in (probe, probe[:1]):
        det, free = engine(part)
        worst = max(
            float(np.max(np.abs(det - expected_det[: len(part)]))),
            float(np.max(np.abs(free - expected_free[: len(part)]))),
        )
        if not np.isfinite(worst) or worst > CONVERT_TOLERANCE:
            raise ValueError(f"変換したモデルが Keras の出力と食い違います（最大差 {worst:.3g}）")

    final = ir_path(keras_path, fp)
    staged = final.with_name(f"{final.stem}.staged.xml")
    try:
        ov.save_model(converted, str(staged), compress_to_fp16=False)
        os.replace(staged.with_suffix(".bin"), final.with_suffix(".bin"))
        os.replace(staged, final)
    finally:
        staged.unlink(missing_ok=True)
        staged.with_suffix(".bin").unlink(missing_ok=True)
    prune_stale(keras_path, final)
    return final


def _runner_from_model(model: Any, shapes: Shapes, pref: str, max_batch: int) -> OpenVinoRunner:
    """読んだ（または変換した）モデルから、NPU → CPU の順で動く推論器を作る。"""
    core = _core()
    available = list(core.available_devices)
    cpu = _CpuEngine(model)
    device = choose_device(available, pref)
    probe = probe_images(shapes, 2)
    reference = cpu(probe)

    if device == "NPU":
        try:
            npu = _compile_npu(model, shapes, bucket_sizes(max_batch))
            det, free = npu(probe)
            worst = max(
                float(np.max(np.abs(det - reference[0]))), float(np.max(np.abs(free - reference[1])))
            )
            if not np.isfinite(worst) or worst > NPU_TOLERANCE:
                raise ValueError(f"NPU の出力が CPU と食い違います（最大差 {worst:.3g}）")
            name = str(core.get_property("NPU", "FULL_DEVICE_NAME"))
            return OpenVinoRunner(
                npu, cpu, device="NPU", description=f"OpenVINO NPU（{name}・バッチ {'/'.join(map(str, npu.buckets))}）"
            )
        except Exception:
            warn_once(
                "percep.openvino_npu_unavailable",
                "NPU で認識器を動かせませんでした。OpenVINO の CPU で動かします",
            )
    name = str(core.get_property("CPU", "FULL_DEVICE_NAME"))
    return OpenVinoRunner(cpu, cpu, device="CPU", description=f"OpenVINO CPU（{name}）")


def open_runner(
    keras_path: Path,
    shapes: Shapes,
    *,
    pref: str,
    max_batch: int,
    keras_loader: Callable[[], Any],
    keras_forward: Callable[[Any], ForwardFn],
) -> OpenVinoRunner | None:
    """IR があればそれを読み、無ければ Keras から作ってから読む。Keras を読むのは作るときだけ。"""
    keras_path = Path(keras_path)
    fp = fingerprint(keras_path)
    xml = ir_path(keras_path, fp)

    model = None
    if xml.exists() and xml.with_suffix(".bin").exists():
        try:
            model = _core().read_model(str(xml))
            check_shapes(model, shapes)
        except Exception:
            warn_once("percep.openvino_ir_unreadable", f"IR を読めませんでした。作り直します: {xml.name}")
            model = None

    if model is None:
        keras_model = keras_loader()
        if keras_model is None:
            return None
        try:
            xml = build_ir(keras_model, keras_path, shapes, keras_forward(keras_model))
            model = _core().read_model(str(xml))
        except Exception:
            warn_once(
                "percep.openvino_convert",
                "認識器を OpenVINO へ変換できませんでした。Keras の CPU 推論で動かします",
            )
            return None
    return _runner_from_model(model, shapes, pref, max_batch)
