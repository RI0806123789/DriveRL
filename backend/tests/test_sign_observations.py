"""新標識の観測と旧観測・旧方策の互換性を検査する。"""
from __future__ import annotations

import numpy as np
import pytest

from app import config
from app.percep.encoder import OBS_OFFSETS, _encode_camera, _encode_traffic_signs
from app.percep.types import DEFAULT_CAMERA, SIGN_DIRECTIONS, DetClass, Detection, PerceptionResult

CLASSES = (DetClass.STOP_SIGN, DetClass.CROSSWALK_SIGN, DetClass.ONE_WAY_SIGN,
           DetClass.MANDATORY_DIRECTION_SIGN, DetClass.NO_PARKING_SIGN, DetClass.NO_STOPPING_SIGN)


def box(cls, distance=12.0, direction=None):
    return Detection(cls, 0.45, 0.2, 0.55, 0.4, 0.8, distance=distance, direction=direction)


def test_layout_preserves_all_existing_offsets():
    assert config.OBS_DIM == 111
    assert OBS_OFFSETS["surround"] == 66
    assert OBS_OFFSETS["v2x"] == 75
    assert OBS_OFFSETS["occlusion"] == 79
    assert OBS_OFFSETS["traffic_signs"] == 87
    assert config.OBS_WIDENABLE_DIMS == (66, 75, 79, 87)
    assert config.OBS_SIGN_DIRECTIONS == SIGN_DIRECTIONS


@pytest.mark.parametrize("cls", CLASSES)
def test_sign_detection_only_changes_appended_block(cls):
    row = np.zeros(config.OBS_DIM, dtype=np.float32)
    baseline = row.copy()
    _encode_camera(baseline, PerceptionResult(0, []), DEFAULT_CAMERA, 2.0, 15.0)
    result = PerceptionResult(0, [box(cls)])
    _encode_camera(row, result, DEFAULT_CAMERA, 2.0, 15.0)
    _encode_traffic_signs(row, result, DEFAULT_CAMERA)
    np.testing.assert_array_equal(row[:87], baseline[:87])
    tail = row[87:105].reshape(6, 3)
    assert tail[CLASSES.index(cls), 0] == pytest.approx(12 / 60)
    assert tail[CLASSES.index(cls), 2] == pytest.approx(0.8)
    assert np.count_nonzero(tail[:, 2]) == 1


@pytest.mark.parametrize("direction", SIGN_DIRECTIONS)
def test_direction_is_only_from_detected_mandatory_sign(direction):
    row = np.zeros(config.OBS_DIM, dtype=np.float32)
    _encode_traffic_signs(row, PerceptionResult(0, [box(DetClass.MANDATORY_DIRECTION_SIGN, direction=direction)]), DEFAULT_CAMERA)
    assert row[105 + SIGN_DIRECTIONS.index(direction)] == 1
    assert row[105:].sum() == 1


def test_nearest_sign_and_missing_signs():
    row = np.zeros(config.OBS_DIM, dtype=np.float32)
    _encode_traffic_signs(row, PerceptionResult(0, [box(DetClass.STOP_SIGN, 30), box(DetClass.STOP_SIGN, 6)]), DEFAULT_CAMERA)
    assert row[87] == pytest.approx(0.1)
    row[:] = 0
    _encode_traffic_signs(row, PerceptionResult(0, [box(DetClass.MANDATORY_DIRECTION_SIGN, 61, "left")]), DEFAULT_CAMERA)
    np.testing.assert_array_equal(row, 0)
    _encode_traffic_signs(row, None, DEFAULT_CAMERA)
    np.testing.assert_array_equal(row, 0)
