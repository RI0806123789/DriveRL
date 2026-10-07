"""通常運転に計測を残さず、予算と学習更新を正しく数えることを検証する。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app import config
from app.map import build_map_index
from app.percep.detector import CHANNELS, GRID_COLS, GRID_ROWS, decode_detections
from app.runtime.pipeline_benchmark import PipelineProfiler, benchmark_speed, timing_summary, weather_params
from tune_hyperparams import synthetic_grid


def test_profiler_restores_owned_and_inherited_methods_after_failure():
    class Target:
        def run(self):
            raise ValueError("意図した失敗")

    instance = Target()
    second = SimpleNamespace(run=lambda: 12)
    original = second.run
    profile = PipelineProfiler()
    profile.attach(instance, "run", "failing")
    profile.attach(second, "run", "owned")
    try:
        with pytest.raises(ValueError):
            instance.run()
        assert second.run() == 12
    finally:
        profile.close()
    assert "run" not in vars(instance)
    assert second.run is original
    assert profile.to_dict()["failing"]["calls"] == 1
    assert profile.to_dict()["owned"]["calls"] == 1


def test_empty_timings_are_not_reported_as_zero_latency():
    assert timing_summary([])["median_ms"] is None
    assert timing_summary([10, 20, 30])["median_ms"] == 20


@pytest.mark.parametrize("publish", [False, True])
def test_speed_benchmark_runs_actual_updates_without_saving(monkeypatch, publish):
    from app.rl.ppo import PPOTrainer

    monkeypatch.delenv("DRIVERL_SCENARIO", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("ベンチマークは本番重みを保存しない")

    monkeypatch.setattr(PPOTrainer, "save", forbidden)
    monkeypatch.setattr(PPOTrainer, "save_state", forbidden)
    index = build_map_index(synthetic_grid())
    params = weather_params("clear", 2, pedestrians=0, rollout_length=4)
    result = benchmark_speed(index, params, mode="oracle", steps=24, warmup=2, publish_frame=publish)
    assert result["step"]["calls"] == 24
    assert result["vehicle_steps"] == 48
    assert result["training_updates"] > 0
    assert result["phases"]["policy_action"]["calls"] == 24
    assert result["phases"]["ppo_update"]["calls"] == 24
    assert result["headless"] is not publish
    assert (result["frame_bytes"] > 0) is publish
    assert 0 <= result["budget_exceeded_fraction"] <= 1


def test_decode_rejects_mismatched_slots():
    raw = np.zeros((2, GRID_ROWS, GRID_COLS, CHANNELS), dtype=np.float32)
    with pytest.raises(ValueError, match="スロット"):
        decode_detections(raw, [0])


def test_decode_preserves_float64_box_math_and_stable_ties():
    from app.percep.detector import OFF_BOX, OFF_CLS, OFF_DIST, OFF_OBJ
    from app.percep.types import DetClass

    raw = np.zeros((1, GRID_ROWS, GRID_COLS, CHANNELS), dtype=np.float32)
    values = raw[0, 1, 2]
    values[OFF_OBJ] = 0.9
    values[OFF_BOX:OFF_BOX + 4] = [0.23, 0.48, 0.1, 0.2]
    values[OFF_CLS + int(DetClass.VEHICLE)] = 1
    values[OFF_DIST] = 0.2
    found = decode_detections(raw, [3])[0]
    det = found.detections[0]
    assert det.x0 == (2 + float(values[OFF_BOX])) / GRID_COLS - float(values[OFF_BOX + 2]) / 2
    assert det.y1 == (1 + float(values[OFF_BOX + 1])) / GRID_ROWS + float(values[OFF_BOX + 3]) / 2
    assert found.slot == 3


def test_cli_rejects_overwrite_and_unbounded_agent_count(tmp_path):
    from benchmark_pipeline import main

    existing = tmp_path / "existing.json"
    existing.write_text("元の内容", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--output", str(existing)])
    assert existing.read_text(encoding="utf-8") == "元の内容"
    with pytest.raises(SystemExit):
        main(["--vehicles", str(config.MAX_VEHICLES + 1), "--output", str(tmp_path / "other.json")])
