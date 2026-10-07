"""画像評価の対応づけ・条件変換・安全な外部教師読み込みを確かめる。"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from app import config
from app.percep.benchmark import ImageCondition, ImageSample, evaluate_images, load_external_dataset, perturb_sample
from app.percep.types import CameraSpec, DetClass, Detection, PerceptionResult


def _box(cls=DetClass.VEHICLE, *, x0=0.2, x1=0.4, confidence=1.0, **attributes):
    return Detection(cls, x0, 0.2, x1, 0.4, confidence, **attributes)


def _sample(detections=None, *, labelled=True, **kwargs):
    image = np.full((8, 12, 3), 100, dtype=np.uint8)
    truth = PerceptionResult(0, [_box()] if detections is None else detections) if labelled else None
    return ImageSample(image, truth, **kwargs)


class FixedDetector:
    spec = CameraSpec(width=12, height=8)

    def __init__(self, predictions, *, free=10):
        self.predictions = predictions
        self.free = free
        self.images = []

    def detect_with_freespace(self, images, slots):
        self.images.append(images.copy())
        return [PerceptionResult(slots[0], self.predictions)], np.full((1, config.OBS_FREESPACE_DIM), self.free, dtype=np.float32)


BASE = [ImageCondition("baseline")]


def test_matching_counts_duplicate_predictions_and_empty_truth_as_false_positives():
    detector = FixedDetector([_box(), _box(), _box(DetClass.PEDESTRIAN)])
    report = evaluate_images(detector, [_sample(), _sample([])], BASE)
    overall = report["conditions"][0]["overall"]
    assert overall["truth"] == 1
    assert overall["predicted"] == 6
    assert overall["matched"] == 1
    assert overall["falsePositives"] == 5
    assert overall["recall"] == 1
    assert overall["precision"] == pytest.approx(1 / 6)
    assert overall["falsePositiveRate"] == pytest.approx(5 / 6)
    assert report["sources"] == {"synthetic": 2}
    json.dumps(report, allow_nan=False)


def test_wrong_class_is_false_positive_and_missed_truth_even_with_identical_box():
    row = evaluate_images(FixedDetector([_box(DetClass.PEDESTRIAN)]), [_sample()], BASE)["conditions"][0]
    assert row["overall"]["matched"] == 0
    assert row["overall"]["falsePositives"] == 1
    assert row["overall"]["precision"] == 0
    assert row["overall"]["recall"] == 0


def test_attribute_accuracy_uses_matched_supplied_attributes():
    truths = [_box(DetClass.TRAFFIC_LIGHT, phase=2), _box(DetClass.TRAFFIC_LIGHT, x0=0.7, x1=0.9), _box(DetClass.SPEED_SIGN, speed_limit=10)]
    preds = [_box(DetClass.TRAFFIC_LIGHT, phase=0), _box(DetClass.TRAFFIC_LIGHT, x0=0.7, x1=0.9, phase=1), _box(DetClass.SPEED_SIGN, speed_limit=10.2)]
    row = evaluate_images(FixedDetector(preds), [_sample(truths)], BASE)["conditions"][0]
    assert row["overall"]["attributeTotal"] == 2
    assert row["overall"]["attributeAccuracy"] == 0.5
    assert row["classes"][0]["attributeAccuracy"] == 0
    assert row["classes"][1]["attributeAccuracy"] == 1


def test_unlabelled_images_do_not_turn_predictions_into_false_positives():
    report = evaluate_images(FixedDetector([_box()]), [_sample(labelled=False, source="external")], BASE)
    row = report["conditions"][0]
    assert row["predictions"] == 1
    assert row["labelledSamples"] == 0
    assert row["overall"]["falsePositives"] is None
    assert row["overall"]["precision"] is None
    assert row["overall"]["recall"] is None


def test_mixed_labels_only_score_known_images():
    row = evaluate_images(FixedDetector([_box()]), [_sample(), _sample(labelled=False)], BASE)["conditions"][0]
    assert row["predictions"] == 2
    assert row["overall"]["predicted"] == 1
    assert row["overall"]["precision"] == 1
    assert row["labelledSamples"] == 1


def test_source_conditions_have_metrics_without_extra_inference():
    detector = FixedDetector([_box()])
    report = evaluate_images(detector, [_sample(condition="clear"), _sample([], condition="rain"), _sample(labelled=False, condition="external")], BASE)
    groups = {item["name"]: item for item in report["conditions"][0]["sourceConditions"]}
    assert groups["clear"]["overall"]["recall"] == 1
    assert groups["rain"]["overall"]["falsePositives"] == 1
    assert groups["external"]["overall"]["precision"] is None
    assert len(detector.images) == 3


def test_fingerprint_tracks_image_and_teacher_contents():
    sample = _sample()
    a = evaluate_images(FixedDetector([]), [sample], BASE)["datasetFingerprint"]
    b = evaluate_images(FixedDetector([]), [replace(sample, image=sample.image + np.uint8(1))], BASE)["datasetFingerprint"]
    c = evaluate_images(FixedDetector([]), [_sample([])], BASE)["datasetFingerprint"]
    assert len(a) == 64
    assert len({a, b, c}) == 3
    assert a == evaluate_images(FixedDetector([]), [sample], BASE)["datasetFingerprint"]


def test_freespace_errors_and_crop_omission():
    sample = _sample(freespace=np.arange(config.OBS_FREESPACE_DIM))
    report = evaluate_images(FixedDetector([], free=5), [sample], [*BASE, ImageCondition("crop", crop_fraction=0.5)])
    free = report["conditions"][0]["freespace"]
    errors = 5 - np.arange(config.OBS_FREESPACE_DIM)
    assert free["maeMeters"] == pytest.approx(np.mean(np.abs(errors)))
    assert free["rmseMeters"] == pytest.approx(np.sqrt(np.mean(errors ** 2)))
    assert free["values"] == config.OBS_FREESPACE_DIM
    cropped = report["conditions"][1]["freespace"]
    assert cropped["values"] == 0
    assert cropped["maeMeters"] is None
    assert cropped["omittedReason"]


def test_crop_transforms_clips_and_removes_labels_using_actual_pixel_bounds():
    image = np.broadcast_to(np.arange(12, dtype=np.uint8)[None, :, None], (8, 12, 3)).copy()
    boxes = [Detection(DetClass.VEHICLE, 0.2, 0.3, 0.6, 0.6, 1), Detection(DetClass.PEDESTRIAN, 0, 0, 0.1, 0.1, 1)]
    sample = ImageSample(image, PerceptionResult(0, boxes))
    changed = perturb_sample(sample, ImageCondition("crop", crop_fraction=0.5))
    assert len(changed.truth.detections) == 1
    box = changed.truth.detections[0]
    assert (box.x0, box.y0, box.x1, box.y1) == pytest.approx((0, 0.1, 0.7, 0.7))
    assert changed.image[0, :, 0].tolist() == [3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    np.testing.assert_array_equal(sample.image, image)
    assert boxes[0].x0 == 0.2


def test_brightness_clips_without_uint8_overflow_and_resolution_preserves_labels():
    sample = replace(_sample(), image=np.full((8, 12, 3), 200, dtype=np.uint8))
    changed = perturb_sample(sample, ImageCondition("bright", brightness=2, resolution_scale=0.5))
    assert changed.image.dtype == np.uint8
    assert np.all(changed.image == 255)
    assert changed.truth.detections == sample.truth.detections
    assert np.all(sample.image == 200)


def test_noise_is_seeded_and_condition_order_independent():
    noise = ImageCondition("noise", noise_std=20)
    a = FixedDetector([])
    b = FixedDetector([])
    evaluate_images(a, [_sample()], [noise, *BASE], seed=14)
    evaluate_images(b, [_sample()], [*BASE, noise], seed=14)
    np.testing.assert_array_equal(a.images[0], b.images[1])
    changed = perturb_sample(_sample(), noise, seed=15)
    assert not np.array_equal(a.images[0][0], changed.image)
    assert np.any(a.images[0] != 100)


@pytest.mark.parametrize("kwargs", [{"brightness": float("nan")}, {"noise_std": -1}, {"resolution_scale": 0}, {"crop_fraction": 2}, {"brightness": True}, {"name": ""}])
def test_invalid_conditions_rejected(kwargs):
    with pytest.raises(ValueError):
        ImageCondition(**{"name": "invalid", **kwargs})


def test_npz_loader_reads_rgb_and_unicode_labels_without_pickle(tmp_path):
    path = tmp_path / "dataset.npz"
    labels = np.array([json.dumps([{"cls": "TRAFFIC_LIGHT", "box": [0.1, 0.2, 0.3, 0.4], "phase": 2}]), "[]"])
    np.savez(path, images=np.full((2, 8, 12, 3), 42, dtype=np.uint8), labels=labels, freespace=np.full((2, config.OBS_FREESPACE_DIM), 20))
    samples = load_external_dataset(path)
    assert len(samples) == 2
    assert samples[0].source == "external"
    assert samples[0].truth.detections[0].phase == 2
    assert samples[1].truth.detections == []
    np.testing.assert_array_equal(samples[0].freespace, np.full(config.OBS_FREESPACE_DIM, 20))


def test_npz_missing_labels_are_unknown_and_object_labels_are_refused(tmp_path):
    path = tmp_path / "dataset.npz"
    images = np.zeros((1, 8, 12, 3), dtype=np.uint8)
    np.savez(path, images=images)
    assert load_external_dataset(path)[0].truth is None
    np.savez(path, images=images, labels=np.array([[]], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        load_external_dataset(path)


def _manifest(tmp_path, entries):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"samples": entries}), encoding="utf-8")
    return path


def test_manifest_uses_local_arrays_with_explicit_negative_labels(tmp_path):
    np.save(tmp_path / "frame.npy", np.full((8, 12, 3), 70, dtype=np.uint8))
    np.savez(tmp_path / "frame.npz", image=np.full((8, 12, 3), 80, dtype=np.uint8))
    path = _manifest(tmp_path, [{"image": "frame.npy", "detections": [], "condition": "rain", "id": "held-out-1"}, {"image": "frame.npz"}])
    samples = load_external_dataset(path)
    assert samples[0].truth.detections == []
    assert samples[0].condition == "rain"
    assert samples[0].sample_id == "held-out-1"
    assert samples[1].truth is None


@pytest.mark.parametrize("label", [{"cls": 2, "box": [0, 0, 1, float("nan")]}, {"cls": True, "box": [0, 0, 1, 1]}, {"cls": 2, "box": [0.5, 0, 0.2, 1]}, {"cls": 0, "box": [0, 0, 1, 1], "phase": 3}, {"cls": 1, "box": [0, 0, 1, 1], "speedLimit": -1}, {"cls": 2, "box": [0, 0, 1, 1], "unexpected": 1}])
def test_manifest_refuses_invalid_labels(tmp_path, label):
    np.save(tmp_path / "frame.npy", np.zeros((8, 12, 3), dtype=np.uint8))
    path = _manifest(tmp_path, [{"image": "frame.npy", "detections": [label]}])
    with pytest.raises(ValueError):
        load_external_dataset(path)


def test_manifest_refuses_path_escape_before_reading(tmp_path):
    path = _manifest(tmp_path, [{"image": "../outside.npy"}])
    with pytest.raises(ValueError, match="ディレクトリ"):
        load_external_dataset(path)


@pytest.mark.parametrize("free", [np.zeros(8), np.full(config.OBS_FREESPACE_DIM, float("nan")), np.full(config.OBS_FREESPACE_DIM, -1)])
def test_invalid_freespace_rejected(free):
    with pytest.raises(ValueError):
        evaluate_images(FixedDetector([]), [_sample(freespace=free)], BASE)


def test_invalid_detector_output_fails_without_truth_fallback():
    with pytest.raises(ValueError):
        evaluate_images(FixedDetector([_box(confidence=float("nan"))]), [_sample()], BASE)


def test_images_resized_to_detector_input_and_invalid_images_rejected():
    detector = FixedDetector([])
    sample = replace(_sample(), image=np.zeros((16, 24, 3), dtype=np.uint8))
    evaluate_images(detector, [sample], BASE)
    assert detector.images[0].shape == (1, 8, 12, 3)
    with pytest.raises(ValueError):
        evaluate_images(detector, [replace(sample, image=sample.image.astype(np.float32))], BASE)
