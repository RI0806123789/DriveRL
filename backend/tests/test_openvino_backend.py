"""認識器の OpenVINO（NPU / CPU）推論。openvino が無い環境でも、従来の Keras へ戻ることまでは検査する。"""
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")

from app import config, warn
from app.percep import detector as det
from app.percep import openvino_backend as ovb
from app.percep import trainer
from app.percep.types import DEFAULT_CAMERA

SMALL = ovb.Shapes(height=4, width=4, detections=(6, 8, 24), freespace=(9,))
HAS_OPENVINO = importlib.util.find_spec("openvino") is not None


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """警告の既出の記録と、コンパイル済みキャッシュの置き場をテストごとに分ける。"""
    warn._WARNED.clear()
    monkeypatch.setattr(config, "DETECTOR_DIR", tmp_path)
    monkeypatch.setattr(config, "PERCEP_DEVICE", "auto")


def _row_model(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """行ごとに決まる値（画素平均）を返す偽の推論器。詰め物の行が本物の行に混ざらないかを見る。"""
    means = batch.reshape(batch.shape[0], -1).mean(axis=1).astype(np.float32)
    det_out = np.broadcast_to(means[:, None, None, None], (batch.shape[0], 6, 8, 24)).copy()
    free_out = np.broadcast_to(means[:, None], (batch.shape[0], 9)).copy()
    return det_out, free_out


# ---------- OpenVINO が無くても検査できる部分 ----------


@pytest.mark.parametrize(
    ("available", "pref", "expected"),
    [
        (["CPU", "GPU", "NPU"], "auto", "NPU"),
        (["CPU", "GPU", "NPU.0"], "auto", "NPU"),
        (["CPU", "GPU"], "auto", "CPU"),
        (["CPU", "GPU", "NPU"], "cpu", "CPU"),
        ([], "auto", "CPU"),
    ],
)
def test_choose_device_never_picks_the_igpu(available: list[str], pref: str, expected: str) -> None:
    assert ovb.choose_device(available, pref) == expected


@pytest.mark.parametrize(
    ("max_batch", "expected"),
    [(1, (4,)), (4, (4,)), (5, (4, 8)), (11, (4, 8, 12)), (12, (4, 8, 12)), (13, (4, 8, 12, 16))],
)
def test_bucket_sizes_cover_the_largest_batch(max_batch: int, expected: tuple[int, ...]) -> None:
    assert ovb.bucket_sizes(max_batch) == expected


def test_default_buckets_cover_every_batch_the_engine_sends() -> None:
    assert ovb.bucket_sizes(config.PERCEP_MAX_BATCH)[-1] >= config.MAX_VEHICLES + config.SURROUND_CNN_IMAGES_PER_STEP


@pytest.mark.parametrize("count", [1, 2, 4, 5, 8, 11, 12, 13, 25])
def test_bucketed_engine_pads_and_returns_only_real_rows(count: int) -> None:
    calls: list[int] = []

    def make(size: int):
        def run(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            assert batch.shape[0] == size
            calls.append(size)
            return _row_model(batch)

        return run

    engine = ovb.BucketedEngine({size: make(size) for size in (4, 8, 12)}, SMALL)
    rng = np.random.default_rng(count)
    batch = rng.uniform(1, 255, size=(count, 4, 4, 3)).astype(np.float32)

    got_det, got_free = engine(batch)
    want_det, want_free = _row_model(batch)

    assert got_det.shape == (count, 6, 8, 24) and got_free.shape == (count, 9)
    np.testing.assert_allclose(got_det, want_det, rtol=1e-6)
    np.testing.assert_allclose(got_free, want_free, rtol=1e-6)
    assert all(size in (4, 8, 12) for size in calls)
    if count <= 12:
        assert calls == [ovb.bucket_for(count, (4, 8, 12))]


def test_bucketed_engine_ignores_stale_rows_left_in_the_buffer() -> None:
    engine = ovb.BucketedEngine({4: _row_model}, SMALL)
    big = np.full((4, 4, 4, 3), 200.0, dtype=np.float32)
    small = np.full((1, 4, 4, 3), 7.0, dtype=np.float32)
    engine(big)
    det_out, _ = engine(small)
    assert det_out.shape[0] == 1
    np.testing.assert_allclose(det_out[0, 0, 0, 0], 7.0)


def test_runner_switches_to_cpu_when_the_npu_fails(caplog: pytest.LogCaptureFixture) -> None:
    npu_calls: list[int] = []

    def broken_npu(batch: np.ndarray):
        npu_calls.append(1)
        raise RuntimeError("device lost")

    runner = ovb.OpenVinoRunner(broken_npu, _row_model, device="NPU", description="NPU")
    batch = np.full((2, 4, 4, 3), 3.0, dtype=np.float32)
    with caplog.at_level(logging.ERROR, logger="autoware_sim"):
        first = runner(batch)
        second = runner(batch)

    assert runner.device == "CPU"
    assert len(npu_calls) == 1, "切り替えた後は NPU を呼び直さない"
    np.testing.assert_allclose(first[0], second[0])
    assert sum("NPU での推論に失敗" in r.message for r in caplog.records) == 1


def test_runner_lets_cpu_errors_through_so_the_env_can_drop_the_detector() -> None:
    def broken(_batch: np.ndarray):
        raise RuntimeError("cpu failed")

    runner = ovb.OpenVinoRunner(broken, broken, device="CPU", description="CPU")
    with pytest.raises(RuntimeError, match="cpu failed"):
        runner(np.zeros((1, 4, 4, 3), dtype=np.float32))


def test_ir_name_follows_the_model_file_content(tmp_path: Path) -> None:
    keras = tmp_path / "detector.keras"
    keras.write_bytes(b"model-a")
    first = ovb.ir_path(keras, ovb.fingerprint(keras))
    keras.write_bytes(b"model-b")
    second = ovb.ir_path(keras, ovb.fingerprint(keras))
    assert first != second
    assert first.parent == keras.parent and first.suffix == ".xml"
    assert first.name.startswith("detector.") and ovb.IR_TAG in first.name


def test_prune_stale_keeps_only_the_current_pair_and_leaves_other_files(tmp_path: Path) -> None:
    keras = tmp_path / "detector.keras"
    keras.write_bytes(b"x")
    prev = tmp_path / "detector.prev.keras"
    prev.write_bytes(b"x")
    keep = ovb.ir_path(keras, "aaaa")
    stale = ovb.ir_path(keras, "bbbb")
    for path in (keep, keep.with_suffix(".bin"), stale, stale.with_suffix(".bin")):
        path.write_bytes(b"x")
    leftover = tmp_path / "detector.ir1-aaaa.staged.xml"
    leftover.write_bytes(b"x")
    cache = ovb.cache_dir()
    cache.mkdir()
    (cache / "blob").write_bytes(b"x")

    ovb.prune_stale(keras, keep)

    assert keep.exists() and keep.with_suffix(".bin").exists()
    assert not stale.exists() and not stale.with_suffix(".bin").exists()
    assert not leftover.exists()
    assert keras.exists() and prev.exists()
    assert not cache.exists()


def test_unknown_device_setting_falls_back_to_auto(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(config, "PERCEP_DEVICE", "tpu")
    with caplog.at_level(logging.ERROR, logger="autoware_sim"):
        assert ovb.preference() == "auto"
        assert ovb.preference() == "auto"
    assert sum("DRIVERL_PERCEP_DEVICE" in r.message for r in caplog.records) == 1


@pytest.fixture(scope="module")
def keras_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("keras")
    path = tmp_path_factory.mktemp("detector_model") / "detector.keras"
    det.build_detector(DEFAULT_CAMERA, width=0.25).save(path)
    return path


def _images(count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(count, int(DEFAULT_CAMERA.height), int(DEFAULT_CAMERA.width), 3), dtype=np.uint8)


def test_load_uses_keras_when_openvino_is_not_installed(monkeypatch: pytest.MonkeyPatch, keras_model_path: Path) -> None:
    monkeypatch.setattr(ovb, "is_installed", lambda: False)
    detector = det.Detector.load(keras_model_path, DEFAULT_CAMERA)
    assert detector is not None and detector.model is not None
    assert detector.backend.startswith("Keras")
    results = detector.detect(_images(2), [0, 1])
    assert len(results) == 2


def test_load_uses_keras_when_the_setting_says_so(monkeypatch: pytest.MonkeyPatch, keras_model_path: Path) -> None:
    monkeypatch.setattr(config, "PERCEP_DEVICE", "keras")
    monkeypatch.setattr(ovb, "is_installed", lambda: True)

    def must_not_run(*_a, **_k):
        raise AssertionError("keras 指定のときに OpenVINO へ触れてはいけない")

    monkeypatch.setattr(ovb, "open_runner", must_not_run)
    detector = det.Detector.load(keras_model_path, DEFAULT_CAMERA)
    assert detector is not None and detector.backend.startswith("Keras")


def test_load_falls_back_to_keras_when_openvino_blows_up(
    monkeypatch: pytest.MonkeyPatch, keras_model_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(ovb, "is_installed", lambda: True)

    def explode(*_a, **_k):
        raise RuntimeError("driver missing")

    monkeypatch.setattr(ovb, "open_runner", explode)
    with caplog.at_level(logging.ERROR, logger="autoware_sim"):
        detector = det.Detector.load(keras_model_path, DEFAULT_CAMERA)
    assert detector is not None and detector.backend.startswith("Keras")
    assert any("OpenVINO での読み込みに失敗" in r.message for r in caplog.records)


def test_verification_load_never_touches_openvino(monkeypatch: pytest.MonkeyPatch, keras_model_path: Path) -> None:
    monkeypatch.setattr(ovb, "is_installed", lambda: True)
    monkeypatch.setattr(ovb, "open_runner", lambda *_a, **_k: pytest.fail("accelerate=False のとき OpenVINO を使った"))
    detector = det.Detector.load(keras_model_path, DEFAULT_CAMERA, accelerate=False)
    assert detector is not None and detector.model is not None


def test_refresh_ir_is_skipped_without_openvino(monkeypatch: pytest.MonkeyPatch, keras_model_path: Path) -> None:
    monkeypatch.setattr(ovb, "is_installed", lambda: False)
    data = {"images": _images(4)}
    assert trainer._refresh_openvino_ir(object(), keras_model_path, DEFAULT_CAMERA, data, None) == ""


def test_refresh_ir_reports_a_failed_conversion_and_drops_stale_ir(
    monkeypatch: pytest.MonkeyPatch, keras_model_path: Path
) -> None:
    monkeypatch.setattr(ovb, "is_installed", lambda: True)
    stale = ovb.ir_path(keras_model_path, "0000000000000000")
    stale.write_bytes(b"old")
    stale.with_suffix(".bin").write_bytes(b"old")

    def explode(*_a, **_k):
        raise ValueError("conversion failed")

    monkeypatch.setattr(ovb, "build_ir", explode)
    data = {"images": _images(4)}
    note = trainer._refresh_openvino_ir(object(), keras_model_path, DEFAULT_CAMERA, data, None)
    assert "OpenVINO" in note and "Keras" in note
    assert not stale.exists() and not stale.with_suffix(".bin").exists()


def _tiny_dataset(count: int = 8) -> dict[str, np.ndarray]:
    data = {
        "images": _images(count, seed=11),
        "detections": np.zeros((count, det.GRID_ROWS, det.GRID_COLS, det.CHANNELS), dtype=np.float32),
        "freespace": np.zeros((count, config.OBS_FREESPACE_DIM), dtype=np.float32),
        "weather": np.zeros((count, 2), dtype=np.float32),
    }
    data.update(trainer.dataset_meta(DEFAULT_CAMERA))
    return data


def _fit_tiny(monkeypatch: pytest.MonkeyPatch, out: Path) -> trainer.FitResult:
    pytest.importorskip("keras")
    # 1 エポックでは何も検出しないので、検証（何か検出するか）だけ通して差し替えの後ろを見る
    monkeypatch.setattr(trainer, "_verify_saved", lambda *_a, **_k: "")
    return trainer.fit_detector(_tiny_dataset(), epochs=1, batch_size=4, out_path=out, width=0.25, verbose=0)


def test_fit_detector_installs_keras_only_without_openvino(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ovb, "is_installed", lambda: False)
    result = _fit_tiny(monkeypatch, tmp_path / "detector.keras")
    assert result.installed and result.warning == ""
    assert not list(tmp_path.glob("detector.ir*"))


def test_fit_detector_keeps_the_new_model_when_conversion_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ovb, "is_installed", lambda: True)

    def explode(*_a, **_k):
        raise ValueError("conversion failed")

    monkeypatch.setattr(ovb, "build_ir", explode)
    result = _fit_tiny(monkeypatch, tmp_path / "detector.keras")
    assert result.installed, "変換に失敗しても Keras の認識器は差し替える"
    assert "OpenVINO" in result.warning
    assert (tmp_path / "detector.keras").exists()


# ---------- openvino があるときだけ回す部分 ----------

needs_openvino = pytest.mark.skipif(not HAS_OPENVINO, reason="openvino が入っていない")


def _npu_available() -> bool:
    if not HAS_OPENVINO:
        return False
    import openvino as ov  # noqa: PLC0415

    return any(d == "NPU" or d.startswith("NPU.") for d in ov.Core().available_devices)


def _keras_outputs(path: Path, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = det.Detector.load(path, DEFAULT_CAMERA, accelerate=False)
    assert reference is not None
    return reference._forward(images)


@needs_openvino
def test_cpu_conversion_matches_keras_and_the_next_load_skips_keras(
    monkeypatch: pytest.MonkeyPatch, keras_model_path: Path
) -> None:
    monkeypatch.setattr(config, "PERCEP_DEVICE", "cpu")
    images = _images(5, seed=3)
    want_det, want_free = _keras_outputs(keras_model_path, images)

    detector = det.Detector.load(keras_model_path, DEFAULT_CAMERA)
    assert detector is not None and detector.backend.startswith("OpenVINO CPU")
    got_det, got_free = detector._forward(images)
    assert np.abs(got_det - want_det).max() < 1e-4 and np.abs(got_free - want_free).max() < 1e-4
    xml = ovb.ir_path(keras_model_path, ovb.fingerprint(keras_model_path))
    assert xml.exists() and xml.with_suffix(".bin").exists()

    monkeypatch.setattr(det.Detector, "_read_keras", staticmethod(lambda _p: pytest.fail("IR があるのに Keras を読んだ")))
    again = det.Detector.load(keras_model_path, DEFAULT_CAMERA)
    assert again is not None and again.model is None
    np.testing.assert_allclose(again._forward(images)[0], got_det, atol=1e-6)


@needs_openvino
def test_ir_of_an_older_model_is_never_used_for_a_replaced_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    keras = pytest.importorskip("keras")
    monkeypatch.setattr(config, "PERCEP_DEVICE", "cpu")
    path = tmp_path / "detector.keras"
    images = _images(3, seed=9)

    keras.utils.set_random_seed(1)
    det.build_detector(DEFAULT_CAMERA, width=0.25).save(path)
    first = det.Detector.load(path, DEFAULT_CAMERA)
    assert first is not None
    first_out = first._forward(images)[0]

    keras.utils.set_random_seed(2)
    det.build_detector(DEFAULT_CAMERA, width=0.25).save(path)
    want = _keras_outputs(path, images)[0]
    second = det.Detector.load(path, DEFAULT_CAMERA)
    assert second is not None
    got = second._forward(images)[0]

    assert np.abs(first_out - want).max() > 1e-3, "2 つのモデルが区別できない（テストの前提が崩れている）"
    assert np.abs(got - want).max() < 1e-4
    assert len(list(tmp_path.glob("detector.ir*.xml"))) == 1


@needs_openvino
def test_fit_detector_builds_the_ir_of_the_installed_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "PERCEP_DEVICE", "cpu")
    out = tmp_path / "detector.keras"
    result = _fit_tiny(monkeypatch, out)
    assert result.installed and result.warning == ""
    xml = ovb.ir_path(out, ovb.fingerprint(out))
    assert xml.exists() and xml.with_suffix(".bin").exists()
    assert not list(tmp_path.glob("detector.staged*")), "検証用の一時ファイルの IR を作ってはいけない"

    monkeypatch.setattr(det.Detector, "_read_keras", staticmethod(lambda _p: pytest.fail("IR があるのに Keras を読んだ")))
    loaded = det.Detector.load(out, DEFAULT_CAMERA)
    assert loaded is not None and loaded.backend.startswith("OpenVINO CPU")


@needs_openvino
def test_a_corrupt_ir_is_rebuilt_from_keras(monkeypatch: pytest.MonkeyPatch, keras_model_path: Path) -> None:
    monkeypatch.setattr(config, "PERCEP_DEVICE", "cpu")
    assert det.Detector.load(keras_model_path, DEFAULT_CAMERA) is not None
    xml = ovb.ir_path(keras_model_path, ovb.fingerprint(keras_model_path))
    xml.write_text("not xml", encoding="utf-8")

    detector = det.Detector.load(keras_model_path, DEFAULT_CAMERA)
    assert detector is not None and detector.backend.startswith("OpenVINO CPU")
    assert detector._forward(_images(2))[0].shape == (2, 6, 8, 24)


@needs_openvino
@pytest.mark.skipif(not _npu_available(), reason="NPU が無い")
def test_npu_output_is_close_to_keras_for_every_batch_size(
    monkeypatch: pytest.MonkeyPatch, keras_model_path: Path
) -> None:
    monkeypatch.setattr(config, "PERCEP_DEVICE", "auto")
    detector = det.Detector.load(keras_model_path, DEFAULT_CAMERA)
    assert detector is not None and detector.backend.startswith("OpenVINO NPU")
    images = _images(13, seed=5)
    want_det, want_free = _keras_outputs(keras_model_path, images)
    for count in (1, 3, 4, 5, 8, 11, 12, 13):
        got_det, got_free = detector._forward(images[:count])
        assert got_det.shape[0] == count
        assert np.abs(got_det - want_det[:count]).max() < 0.05
        assert np.abs(got_free - want_free[:count]).max() < 0.05
