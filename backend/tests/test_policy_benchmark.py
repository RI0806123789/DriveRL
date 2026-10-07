"""認識の分離評価・安全な読取・未完了を含む指標の定義を確かめる。"""

from __future__ import annotations

import hashlib
from dataclasses import asdict

import numpy as np
import pytest
import torch

from app import config
from app.contracts import DriveState, EpisodeResult, SimParams, StepResult
from app.map import build_map_index
from app.percep.types import DEFAULT_CAMERA, DetClass, Detection, PerceptionResult
from app.percep import groundtruth
from app.runtime.policy_benchmark import NoiseConfig, PerceptionNoise, PolicyMetrics, evaluate_policy, load_policy
from app.rl.ppo import CHECKPOINT_FORMAT, PPOTrainer
from tune_hyperparams import synthetic_grid


@pytest.fixture
def policy():
    return PPOTrainer(config.OBS_DIM, config.ACTION_DIM, SimParams(), config.MAX_VEHICLES, seed=31, hidden_sizes=(16, 16)).policy


@pytest.fixture
def grid(monkeypatch):
    monkeypatch.delenv("DRIVERL_SCENARIO", raising=False)
    return build_map_index(synthetic_grid())


def test_noise_is_repeatable_and_never_changes_source_labels():
    source = PerceptionResult(2, [
        Detection(DetClass.TRAFFIC_LIGHT, 0.2, 0.2, 0.3, 0.3, 1.0, distance=12, phase=2),
        Detection(DetClass.VEHICLE, 0.3, 0.3, 0.5, 0.5, 1.0, distance=5),
        Detection(DetClass.LANE, 0.1, 0.4, 0.9, 0.9, 1.0, lane_points=[(0.1, 0.4), (0.2, 0.9)]),
    ])
    before = asdict(source)
    settings = NoiseConfig(drop_probability=0, distance_std=0.2, phase_error_probability=1)
    first = PerceptionNoise(9, settings)(source, DEFAULT_CAMERA)
    second = PerceptionNoise(9, settings)(source, DEFAULT_CAMERA)
    assert asdict(first) == asdict(second)
    assert first.detections[0].phase != 2
    assert first.detections[0].distance != 12
    first.detections[2].lane_points.append((0.7, 0.8))
    assert asdict(source) == before
    dropped = PerceptionNoise(9, NoiseConfig(1, 0, 0))(source, DEFAULT_CAMERA)
    assert dropped.slot == source.slot and not dropped.detections


@pytest.mark.parametrize("settings", [{"drop_probability": -0.1}, {"drop_probability": 1.1}, {"distance_std": float("nan")}, {"phase_error_probability": float("inf")}])
def test_noise_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        NoiseConfig(**settings)


def test_safe_checkpoint_read_preserves_source_file_and_uses_existing_migration(tmp_path, policy, monkeypatch):
    checkpoint = tmp_path / "policy.pt"
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "obs_dim": config.OBS_DIM,
        "action_dim": config.ACTION_DIM,
        "hidden_sizes": [16, 16],
        "policy": policy.state_dict(),
        "updates": 7,
    }, checkpoint)
    original = checkpoint.read_bytes()
    original_load = torch.load
    loads = []

    def guarded_load(*args, **kwargs):
        loads.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    def reject_save(*args, **kwargs):
        pytest.fail("評価の読み込みはモデルを保存しない")

    monkeypatch.setattr(torch, "load", guarded_load)
    monkeypatch.setattr(torch, "save", reject_save)
    loaded, metadata = load_policy(checkpoint)
    assert loads and all(loads)
    assert metadata["checkpoint_sha256"] == hashlib.sha256(original).hexdigest()
    assert metadata["training_updates"] == 7
    assert not loaded.training
    assert checkpoint.read_bytes() == original
    assert list(tmp_path.iterdir()) == [checkpoint]
    obs = torch.zeros((3, config.OBS_DIM))
    assert torch.equal(loaded.mean_action(obs), policy.mean_action(obs))


def test_invalid_checkpoint_is_rejected_without_fallback(tmp_path):
    checkpoint = tmp_path / "broken.pt"
    checkpoint.write_bytes(b"invalid checkpoint")
    with pytest.raises(ValueError, match="安全"):
        load_policy(checkpoint)


def test_metrics_separate_completed_rates_censored_events_and_driving_time():
    learners = np.ones(4, dtype=bool)
    active = np.array([True, True, True, False])
    result = StepResult(
        np.zeros((4, config.OBS_DIM)), np.zeros(4), np.array([True, True, True, False]), active,
        episodes=[
            EpisodeResult(0, 0, 20, "goal", signal_violations=1),
            EpisodeResult(1, 0, 40, "collision", lane_departures=2),
            EpisodeResult(2, 0, 60, "offroad"),
        ],
        drive=DriveState(np.array([0, 3, 6, 100]), np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)),
    )
    metrics = PolicyMetrics()
    metrics.observe(result, learners)
    output = metrics.to_dict([(10, 2, 1)])
    assert output["arrival_rate"] == output["collision_rate"] == output["offroad_rate"] == pytest.approx(1 / 3)
    assert output["signal_violation_rate"] == output["lane_departure_rate"] == pytest.approx(1 / 3)
    assert output["mean_completed_episode_seconds"] == pytest.approx(40 * config.DT)
    assert output["mean_arrival_seconds"] == pytest.approx(20 * config.DT)
    assert output["unfinished_episodes"] == 1
    assert output["unfinished_episode_seconds"] == pytest.approx(10 * config.DT)
    assert output["signal_violation_events"] == output["lane_departure_events"] == 3
    assert output["vehicle_steps"] == 3
    assert output["vehicle_seconds"] == pytest.approx(3 * config.DT)
    assert output["mean_speed_m_s"] == 3
    empty = PolicyMetrics().to_dict([])
    assert empty["arrival_rate"] is None
    assert empty["collision_rate"] is None
    assert empty["mean_completed_episode_seconds"] is None
    assert empty["signal_violations_per_vehicle_hour"] is None


def without_timings(output):
    return {key: value for key, value in output.items() if key not in {"timing", "steps_per_second", "vehicle_steps_per_second", "mode", "noise", "noise_scope"}}


def test_evaluation_repeats_initial_world_without_learning_and_zero_noise_matches_oracle(grid, policy, monkeypatch):
    params = SimParams(vehicle_count=2, pedestrian_count=0, weather_auto=False)
    before_params = asdict(params)
    before_policy = {key: value.clone() for key, value in policy.state_dict().items()}

    def reject_learning(*args, **kwargs):
        pytest.fail("評価では学習・保存を呼ばない")

    monkeypatch.setattr(PPOTrainer, "store", reject_learning)
    monkeypatch.setattr(PPOTrainer, "maybe_update", reject_learning)
    monkeypatch.setattr(PPOTrainer, "save", reject_learning)
    first = evaluate_policy(grid, policy, params, mode="oracle", seed=3, steps=12)
    second = evaluate_policy(grid, policy, params, mode="oracle", seed=3, steps=12)
    noisy = evaluate_policy(grid, policy, params, mode="noisy", seed=3, steps=12, noise=NoiseConfig(0, 0, 0))
    assert without_timings(first) == without_timings(second) == without_timings(noisy)
    assert first["completed_episodes"] == 0
    assert first["arrival_rate"] is None
    assert first["unfinished_episodes"] == 2
    assert first["vehicle_steps"] == 24
    assert first["mean_speed_m_s"] is not None
    assert first["learning_updates"] == 0
    assert first["timing"]["total"]["samples"] == 12
    assert policy.training
    assert asdict(params) == before_params
    assert all(torch.equal(before_policy[key], value) for key, value in policy.state_dict().items())


def test_noisy_labels_do_not_change_true_world_with_identical_actions(grid, policy, monkeypatch):
    def mean_action(obs, options):
        return torch.zeros((config.MAX_VEHICLES, config.ACTION_DIM))

    monkeypatch.setattr(policy, "mean_action", mean_action)
    params = SimParams(vehicle_count=2, pedestrian_count=0, weather_auto=False)
    first = evaluate_policy(grid, policy, params, mode="oracle", seed=3, steps=10)
    noisy = evaluate_policy(grid, policy, params, mode="noisy", seed=3, steps=10, noise=NoiseConfig(1, 1, 1))
    assert without_timings(first) == without_timings(noisy)


class FixedDetector:
    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after

    def detect_with_freespace(self, images, slots):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("注入した認識器の推論失敗")
        assert images.dtype == np.uint8
        assert len(images) == len(slots)
        return [PerceptionResult(int(slot)) for slot in slots], np.full((len(slots), config.OBS_FREESPACE_DIM), 30, dtype=np.float32)


def test_cnn_evaluation_uses_injected_detector_for_every_new_camera_batch(grid, policy, monkeypatch):
    detector = FixedDetector()

    def reject_labels(*args, **kwargs):
        pytest.fail("CNN の評価に正解ラベルを混ぜない")

    monkeypatch.setattr(groundtruth, "detect_ground_truth_views", reject_labels)
    output = evaluate_policy(grid, policy, SimParams(vehicle_count=1, pedestrian_count=0), mode="cnn", detector=detector, seed=0, steps=3)
    assert output["cnn_active"]
    assert detector.calls == 4
    assert output["vehicle_steps"] == 3


def test_cnn_runtime_inference_failure_aborts_and_restores_profiler_and_policy(grid, policy, monkeypatch):
    detector = FixedDetector(fail_after=1)

    class Probe:
        closed = False

        def close(self):
            self.closed = True

    probe = Probe()

    def attach(env):
        assert env.detector_active
        return probe

    def reject_labels(*args, **kwargs):
        pytest.fail("CNN の推論失敗を正解ラベルで補わない")

    monkeypatch.setattr(groundtruth, "detect_ground_truth_views", reject_labels)
    with pytest.raises(RuntimeError, match="推論失敗"):
        evaluate_policy(grid, policy, SimParams(vehicle_count=1, pedestrian_count=0), mode="cnn", detector=detector, seed=0, steps=3, profiler=attach)
    assert detector.calls == 2
    assert probe.closed
    assert policy.training
    assert groundtruth._STATIC_CACHE is None
