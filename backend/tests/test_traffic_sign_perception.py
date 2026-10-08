"""標識の描画・教師・CNN の種別と矢印を検査する。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.contracts import MapSign
from app.percep import camera, detector, groundtruth, trainer
from app.percep.benchmark import _label, _sample_digest, ImageSample
from app.percep.evaluate import _attribute_ok
from app.percep.geometry import camera_pose
from app.percep.types import (
    DEFAULT_CAMERA, SIGN_CLASSES, SIGN_DIRECTIONS, DetClass, Detection, PerceptionResult,
)


def _sign(kind, *, heading=0.0, direction="straight"):
    return MapSign(1, 1, 1, 8.0, 0.0, heading, 30.0 / 3.6, kind, direction)


def _scene(sign):
    return SimpleNamespace(data=SimpleNamespace(signs=[sign], signals=[], nodes=[]))


def _camera(monkeypatch, sign):
    monkeypatch.setattr(camera.PseudoCamera, "_build_road_raster", lambda self: None)
    return camera.PseudoCamera(_scene(sign))


def _draw(monkeypatch, sign):
    view = _camera(monkeypatch, sign)
    pose = camera_pose(0.0, 0.0, 0.0, DEFAULT_CAMERA)
    part = view._collect_signs(
        np.array([pose.eye_x]), np.array([pose.eye_y]), np.ones(1), np.zeros(1), np.zeros(1),
    )
    labels = np.zeros((1, DEFAULT_CAMERA.height, DEFAULT_CAMERA.width), dtype=np.uint8)
    if part is not None:
        view._draw_shapes(labels, None, [part])
    return labels[0]


@pytest.mark.parametrize("kind,cls", SIGN_CLASSES.items())
def test_sign_board_and_teacher_share_visible_bounds(monkeypatch, kind, cls):
    sign = _sign(kind)
    labels = _draw(monkeypatch, sign)
    static = groundtruth._build_static_scene(_scene(sign))
    world = SimpleNamespace(map_index=_scene(sign))
    found = groundtruth._detect_signs(
        world, DEFAULT_CAMERA, camera_pose(0.0, 0.0, 0.0, DEFAULT_CAMERA), 0.0,
        static, np.array([0]), DEFAULT_CAMERA.far,
    )
    assert len(found) == 1
    truth = found[0][1]
    assert truth.cls == cls
    assert (truth.speed_limit is not None) == (kind == "speed_limit")
    assert (truth.direction is not None) == (kind in ("one_way", "mandatory_direction"))
    ys, xs = np.nonzero((labels != 0) & (labels != camera.LBL_POLE))
    assert len(xs) > 0
    assert abs(xs.min() - truth.x0 * DEFAULT_CAMERA.width) < 2.0
    assert abs(xs.max() - truth.x1 * DEFAULT_CAMERA.width) < 2.0
    assert abs(ys.min() - truth.y0 * DEFAULT_CAMERA.height) < 2.0
    assert abs(ys.max() - truth.y1 * DEFAULT_CAMERA.height) < 2.0


@pytest.mark.parametrize("kind", SIGN_CLASSES)
def test_sign_back_is_neither_drawn_nor_labelled(monkeypatch, kind):
    sign = _sign(kind, heading=np.pi)
    assert not _draw(monkeypatch, sign).any()
    static = groundtruth._build_static_scene(_scene(sign))
    assert not groundtruth._detect_signs(
        SimpleNamespace(map_index=_scene(sign)), DEFAULT_CAMERA,
        camera_pose(0.0, 0.0, 0.0, DEFAULT_CAMERA), 0.0, static, np.array([0]), DEFAULT_CAMERA.far,
    )


def test_every_sign_kind_has_a_distinct_camera_image(monkeypatch):
    images = [_draw(monkeypatch, _sign(kind)).tobytes() for kind in SIGN_CLASSES]
    assert len(set(images)) == len(images)


def test_mandatory_direction_arrows_have_distinct_camera_images(monkeypatch):
    images = [_draw(monkeypatch, _sign("mandatory_direction", direction=d)).tobytes() for d in SIGN_DIRECTIONS]
    assert len(set(images)) == len(images)


@pytest.mark.parametrize("kind,cls", SIGN_CLASSES.items())
@pytest.mark.parametrize("direction", SIGN_DIRECTIONS)
def test_cnn_targets_decode_each_sign_without_speed_leakage(kind, cls, direction):
    original = Detection(cls, 0.2, 0.2, 0.3, 0.3, 1.0, distance=15.0,
                         speed_limit=30.0 / 3.6 if kind == "speed_limit" else None,
                         direction=direction if kind in ("one_way", "mandatory_direction") else None)
    raw = detector.encode_targets([PerceptionResult(0, [original])])
    decoded = detector.decode_detections(raw, [0])[0].detections
    assert len(decoded) == 1
    assert decoded[0].cls == original.cls
    assert decoded[0].speed_limit == original.speed_limit
    assert decoded[0].direction == original.direction
    assert decoded[0].distance == pytest.approx(original.distance)


def test_old_cnn_shape_is_rejected_with_fallback_log(monkeypatch, tmp_path, caplog):
    path = tmp_path / "old.keras"
    path.write_bytes(b"old")
    old_model = SimpleNamespace(input_shape=(None, 144, 192, 3), output_shape=[(None, 6, 8, 24), (None, detector.FREESPACE_DIM)])
    monkeypatch.setattr(detector.Detector, "_read_keras", lambda path: old_model)
    assert detector.Detector.load(path, accelerate=False) is None
    assert "真値へフォールバック" in caplog.text


def test_old_dataset_is_rejected():
    data = {"images": np.zeros((1, 144, 192, 3)), "detections": np.zeros((1, 6, 8, detector.CHANNELS)),
            "freespace": np.zeros((1, detector.FREESPACE_DIM)), "weather": np.zeros((1, 2)),
            **trainer.dataset_meta()}
    data["version"] = np.asarray(4)
    assert "収集し直してください" in trainer.check_dataset(data)


def test_collection_focus_filters_sign_kind():
    signs = [_sign(kind) for kind in SIGN_CLASSES]
    signs[1] = replace(signs[1], x=21.0)
    env = SimpleNamespace(world=SimpleNamespace(map_index=SimpleNamespace(data=SimpleNamespace(signs=signs))))
    xy, _ = trainer._sign_approaches(env, DetClass.STOP_SIGN)
    assert xy.tolist() == [[21.0, 0.0]]


def test_direction_accuracy_is_scored_independently():
    truth = Detection(DetClass.MANDATORY_DIRECTION_SIGN, 0.2, 0.2, 0.3, 0.3, 1.0, direction="left")
    assert _attribute_ok(truth.cls, truth, replace(truth, direction="left")) is True
    assert _attribute_ok(truth.cls, truth, replace(truth, direction="right")) is False


def test_external_teacher_preserves_and_validates_arrow_direction():
    label = [{"cls": "MANDATORY_DIRECTION_SIGN", "box": [0.2, 0.2, 0.3, 0.3], "direction": "left"}]
    loaded = _label(label)
    assert loaded.detections[0].direction == "left"
    image = np.zeros((144, 192, 3), dtype=np.uint8)
    digest = _sample_digest(ImageSample(image, loaded))
    label[0]["direction"] = "right"
    assert digest != _sample_digest(ImageSample(image, _label(label)))
    label[0]["direction"] = "u_turn"
    with pytest.raises(ValueError, match="方向"):
        _label(label)


def test_staged_verification_rejects_missing_class(monkeypatch):
    truth = [PerceptionResult(0, [Detection(DetClass.STOP_SIGN, 0.2, 0.2, 0.3, 0.3, 1.0)])]
    data = {"images": np.zeros((1, 144, 192, 3)), "detections": detector.encode_targets(truth)}
    loaded = SimpleNamespace(detect=lambda images, slots: [PerceptionResult(s, [Detection(DetClass.VEHICLE, 0.2, 0.2, 0.3, 0.3, 1.0)]) for s in slots])
    monkeypatch.setattr(detector.Detector, "load", lambda *args, **kwargs: loaded)
    result = trainer.FitResult(Path("detector.keras"), 1, 1, 0, [])
    assert "STOP_SIGN" in trainer._verify_saved(Path("staged.keras"), data, DEFAULT_CAMERA, result)


def test_staged_verification_rejects_wrong_arrow_directions(monkeypatch):
    truth = Detection(DetClass.MANDATORY_DIRECTION_SIGN, 0.2, 0.2, 0.3, 0.3, 1.0, direction="left")
    data = {"images": np.zeros((1, 144, 192, 3)), "detections": detector.encode_targets([PerceptionResult(0, [truth])])}
    loaded = SimpleNamespace(detect=lambda images, slots: [PerceptionResult(s, [replace(truth, direction="right")]) for s in slots])
    monkeypatch.setattr(detector.Detector, "load", lambda *args, **kwargs: loaded)
    result = trainer.FitResult(Path("detector.keras"), 1, 1, 0, [])
    assert "矢印" in trainer._verify_saved(Path("staged.keras"), data, DEFAULT_CAMERA, result)


def test_staged_verification_rejects_a_sign_at_the_wrong_position(monkeypatch):
    truth = Detection(DetClass.STOP_SIGN, 0.2, 0.2, 0.3, 0.3, 1.0)
    data = {"images": np.zeros((1, 144, 192, 3)), "detections": detector.encode_targets([PerceptionResult(0, [truth])])}
    loaded = SimpleNamespace(detect=lambda images, slots: [PerceptionResult(s, [replace(truth, x0=0.7, x1=0.8)]) for s in slots])
    monkeypatch.setattr(detector.Detector, "load", lambda *args, **kwargs: loaded)
    result = trainer.FitResult(Path("detector.keras"), 1, 1, 0, [])
    assert "STOP_SIGN" in trainer._verify_saved(Path("staged.keras"), data, DEFAULT_CAMERA, result)
