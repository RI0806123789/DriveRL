"""通常起動で使う詳細物理・背景交通・信号設定の統合検査。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app import config
from app.contracts import InterventionEvent, SimParams
from app.map.index import build_map_index
from app.percep.detector import Detector
from app.rl.buffer import RolloutBuffer
from app.rl.ppo import PPOTrainer
from app.runtime.engine import SimulationEngine
from tune_hyperparams import synthetic_grid


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"model": "kinematic", "learner_vehicles": 8}), encoding="utf-8")
    monkeypatch.setenv("DRIVERL_SCENARIO", str(baseline))
    monkeypatch.setattr(config, "CHECKPOINT_PATH", tmp_path / "shared_policy.pt")
    monkeypatch.setattr(Detector, "load", classmethod(lambda cls, *_args, **_kwargs: None))
    instance = SimulationEngine(param_store=None)
    instance._params.pedestrian_count = 0
    instance._trainer = PPOTrainer(
        config.OBS_DIM, config.ACTION_DIM, SimParams(rollout_length=16),
        config.MAX_VEHICLES, seed=0, hidden_sizes=(32, 32),
    )
    instance._install_map(build_map_index(synthetic_grid()), "grid", "碁盤の目")
    yield instance
    instance._autotune.shutdown()


def _assert_standard(instance: SimulationEngine) -> None:
    env = instance._env
    assert env is not None
    assert env.world.fleet.dynamics is not None
    assert env.world.signals.plan.mode == "adaptive"
    assert env.traffic is not None and env.traffic.mask.any()
    assert env.fixed_vehicle_count == 4
    assert env.params.vehicle_count == 4


def test_startup_overrides_baseline_environment_and_stale_vehicle_count(engine) -> None:
    _assert_standard(engine)
    engine._params.vehicle_count = 0
    engine._install_map(build_map_index(synthetic_grid()), "new-grid", "次の地図")
    _assert_standard(engine)
    assert engine.snapshot_params().vehicle_count == 4
    assert engine._env.active_mask[0]


def test_map_switch_cannot_disable_physics_with_missing_environment_file(engine, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DRIVERL_SCENARIO", str(tmp_path / "missing.json"))
    previous = engine._env
    engine._install_map(build_map_index(synthetic_grid()), "other-grid", "別の地図")
    assert engine._env is not previous
    _assert_standard(engine)


def test_reset_and_practical_mode_preserve_standard_environment(engine) -> None:
    engine._apply_app_mode(True)
    env = engine._env
    assert env.autopilot_all
    _assert_standard(engine)
    env.world.fleet.lateral_speed[0] = 2.0
    env.world.fleet.yaw_rate[0] = 0.5
    engine._handle_inbox_item("event", InterventionEvent("reset_episode", {}))
    assert env.world.fleet.lateral_speed[0] == 0
    assert env.world.fleet.yaw_rate[0] == 0
    _assert_standard(engine)
    engine._apply_app_mode(False)
    assert not env.autopilot_all
    assert engine._env is env
    _assert_standard(engine)


def test_vehicle_count_patch_and_manual_spawn_cannot_change_reserved_fleet(engine) -> None:
    before = engine._env.active_mask.copy()
    params, patch = engine.update_params({"vehicleCount": 0, "simSpeed": 2})
    engine._drain_inbox()
    assert "vehicleCount" in patch.rejected
    assert params.vehicle_count == 4 and params.sim_speed == 2
    np.testing.assert_array_equal(engine._env.active_mask, before)
    assert engine._env.apply_event(InterventionEvent("spawn_vehicle", {"x": 0, "y": 0}))
    assert engine._env.apply_event(InterventionEvent("despawn_vehicle", {"id": 0}))
    _assert_standard(engine)


def test_background_traffic_stays_active_without_policy_value_or_imitation_samples(engine) -> None:
    env = engine._env
    env.params.safety_assist = False
    result = env.step(np.zeros((config.MAX_VEHICLES, config.ACTION_DIM)), expert=np.ones(config.MAX_VEHICLES, dtype=bool))
    background = env.traffic.mask & result.active
    assert background.any()
    assert not result.learn[background].any()
    assert not result.assisted[background].any()
    buffer = RolloutBuffer(1, config.MAX_VEHICLES, config.OBS_DIM, config.ACTION_DIM)
    buffer.add(
        result.obs, np.zeros((config.MAX_VEHICLES, config.ACTION_DIM)),
        np.zeros(config.MAX_VEHICLES), np.zeros(config.MAX_VEHICLES),
        result.rewards, result.dones, result.active, learn=result.learn,
        assisted=result.assisted, expert_actions=result.expert_actions,
    )
    assert buffer.active[0, background].all()
    assert not buffer.learn[0, background].any()
    assert not buffer.assisted[0, background].any()


def test_autotune_trial_retains_detailed_physics_and_background_traffic(engine) -> None:
    env = engine._env
    state = engine._trainer.snapshot_state()
    engine._tune_begin_trial({"learningRate": 0.0002}, state)
    assert engine._env is env
    assert engine.snapshot_params().learning_rate == pytest.approx(0.0002)
    _assert_standard(engine)
