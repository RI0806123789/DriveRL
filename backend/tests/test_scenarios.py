"""背景交通・道路イベント・同じ方策の比較評価を検査する。"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from app import config
from app.contracts import DriveState, EpisodeResult, InterventionEvent, MapPreset, SimParams, StepResult
from app.map import build_map_index
from app.map.loader import _build_signals, _from_cache_dict, _to_cache_dict
from app.rl.buffer import RolloutBuffer
from app.sim.dynamics import VehicleDynamics
from app.sim.env import SimulationEnv
from app.sim.evaluation import EvaluationMetrics
from app.sim.scenario import DriverBehavior, RoadEvent, Scenario, TrafficPeriod, load_scenario
from evaluate_scenarios import maneuvers
from tune_hyperparams import synthetic_grid


@pytest.fixture
def grid():
    return build_map_index(synthetic_grid())


@pytest.fixture(autouse=True)
def no_scenario_env(monkeypatch):
    monkeypatch.delenv("DRIVERL_SCENARIO", raising=False)


def environment(grid, scenario: Scenario | None = None, seed: int = 0):
    return SimulationEnv(grid, SimParams(vehicle_count=3, pedestrian_count=0), seed=seed, compute_observations=False, scenario=scenario)


def test_background_drivers_never_become_ppo_samples(grid):
    env = environment(grid, Scenario(background_vehicles=2))
    result = env.step(np.zeros((config.MAX_VEHICLES, 2)), expert=np.ones(config.MAX_VEHICLES, dtype=bool))
    assert result.active[:3].all()
    assert not result.learn[1:3].any()
    assert not result.assisted[1:3].any()
    assert result.drive.speed[1:3].min() > 0
    buffer = RolloutBuffer(1, config.MAX_VEHICLES, config.OBS_DIM, 2)
    buffer.add(result.obs, np.zeros((config.MAX_VEHICLES, 2)), np.zeros(config.MAX_VEHICLES), np.zeros(config.MAX_VEHICLES), result.rewards, result.dones, result.active, learn=result.learn)
    assert not buffer.learn[0, 1:3].any()
    assert buffer.active[0, 1:3].all()


def test_time_of_day_volume_updates_before_next_observation(grid):
    scenario = Scenario(background_vehicles=2, start_hour=0, periods=(TrafficPeriod(0, 0), TrafficPeriod(1, 2), TrafficPeriod(2, 0)))
    env = environment(grid, scenario)
    assert env.active_mask.sum() == 1
    env.world.sim_time = 3600 - config.DT
    env.step(np.zeros((config.MAX_VEHICLES, 2)))
    env.step(np.zeros((config.MAX_VEHICLES, 2)))
    assert env.active_mask[:3].all()
    env.world.sim_time = 7200 - config.DT
    env.step(np.zeros((config.MAX_VEHICLES, 2)))
    assert env.active_mask.sum() == 1
    env.apply_params(replace(env.params, vehicle_count=8))
    assert env.params.vehicle_count == 3
    assert env.active_mask.sum() == 1


def test_driver_variations_are_seeded_independently(grid):
    scenario = Scenario(background_vehicles=2, driver_spread=0.4)
    first = environment(grid, scenario, seed=7)
    second = environment(grid, scenario, seed=7)
    assert first.traffic.drivers == second.traffic.drivers
    assert first.traffic.drivers[1] != first.traffic.drivers[2]
    assert first.rng.bit_generator.state == second.rng.bit_generator.state
    for _ in range(10):
        first.traffic.adjust_commands(1, 0.5, 0.0)
    assert first.rng.bit_generator.state == second.rng.bit_generator.state


@pytest.mark.parametrize("kind", ["parked", "construction", "closure"])
def test_events_use_visible_collision_obstacles_and_expire(grid, kind):
    scenario = Scenario(events=(RoadEvent(0, kind, 1, 2),))
    env = environment(grid, scenario)
    manual = env.world.add_obstacle(400, 400, 0.5)
    assert len(env.world.obstacles) == 1
    env.world.sim_time = 1
    env.traffic.prepare(env)
    assert len(env.world.obstacles) > 1
    assert len(env.world.obstacle_xy) == len(env.world.obstacles)
    obstacle = env.world.obstacles[1]
    env.world.fleet.reset_slot(0, obstacle.x, obstacle.y, 0)
    assert env.world.check_collisions()[0]
    env.world.sim_time = 3
    env.traffic.prepare(env)
    assert [o.id for o in env.world.obstacles] == [manual]
    env.world.clear_obstacles()
    env.traffic.prepare(env)
    assert not env.world.obstacles


def test_weighted_spawn_passes_road_direction_and_does_not_double_spawn(grid, monkeypatch):
    env = environment(grid, Scenario(background_vehicles=1, road_weights={0: 1}))
    calls = []
    original = env.world.activate_at

    def record(slot, at, *, clearance_m, heading=None):
        calls.append((slot, at, heading))
        return original(slot, at, clearance_m=clearance_m, heading=heading)

    monkeypatch.setattr(env.world, "activate_at", record)
    env.world.deactivate(1)
    env.traffic.prepare(env)
    assert len(calls) == 1
    assert calls[0][2] == pytest.approx(0)
    assert calls[0][1][1] == pytest.approx(1.75)
    serial = env.world.route_serial[1]
    env.traffic.prepare(env)
    assert env.world.route_serial[1] == serial


def test_detailed_state_clears_when_deactivated(grid):
    env = environment(grid, Scenario(dynamics=VehicleDynamics()))
    fleet = env.world.fleet
    fleet.lateral_speed[0] = 3
    fleet.yaw_rate[0] = 1
    fleet.throttle[0] = 1
    env.world.deactivate(0)
    assert fleet.lateral_speed[0] == fleet.yaw_rate[0] == fleet.throttle[0] == 0


def test_reset_retains_reserved_capacity_and_uses_weighted_spawns(grid, monkeypatch):
    env = environment(grid, Scenario(background_vehicles=1, road_weights={0: 1}))
    env.world.deactivate(1)
    env._respawn_queue.append(1)
    calls = []
    original = env.traffic.spawn

    def record(current, slot):
        calls.append(slot)
        return original(current, slot)

    monkeypatch.setattr(env.traffic, "spawn", record)
    env.reset_all()
    assert 1 in calls
    assert env.params.vehicle_count == 2
    assert env.apply_event(InterventionEvent("spawn_vehicle", {"x": 0, "y": 0})) is not None


def test_reserved_taxi_is_kept_when_background_volume_falls(grid):
    scenario = Scenario(background_vehicles=1, start_hour=0, periods=(TrafficPeriod(0, 1), TrafficPeriod(1, 0)))
    env = environment(grid, scenario)
    env.commandeer_vehicle(1)
    env.world.sim_time = 3600
    env.traffic.prepare(env)
    assert env.active_mask[1]


def test_engine_preserves_scenario_capacity_and_rejects_count_changes(grid):
    from app.runtime.engine import SimulationEngine

    env = environment(grid, Scenario(background_vehicles=2, start_hour=0, periods=(TrafficPeriod(0, 0),)))
    engine = SimulationEngine(param_store=None)
    engine._env = env
    engine._sync_vehicle_count()
    assert engine.snapshot_params().vehicle_count == 3
    params, patch = engine.update_params({"vehicleCount": 8, "simSpeed": 2})
    assert "vehicleCount" in patch.rejected
    assert params.vehicle_count == 3
    assert params.sim_speed == 2


def test_background_episodes_do_not_pollute_runtime_metrics(grid):
    env = environment(grid, Scenario(background_vehicles=1))
    env.world.slots[1].steps = config.MAX_EPISODE_STEPS
    result = env.step(np.zeros((config.MAX_VEHICLES, 2)))
    assert result.dones[1]
    assert all(episode.slot != 1 for episode in result.episodes)


def test_default_scenario_does_not_consume_extra_rng(grid):
    first = environment(grid)
    second = environment(grid)
    assert first.traffic is second.traffic is None
    for _ in range(10):
        a = first.step(np.zeros((config.MAX_VEHICLES, 2)))
        b = second.step(np.zeros((config.MAX_VEHICLES, 2)))
        np.testing.assert_array_equal(a.rewards, b.rewards)
        np.testing.assert_array_equal(first.world.fleet.x, second.world.fleet.x)
    assert first.rng.bit_generator.state == second.rng.bit_generator.state


@pytest.mark.parametrize("values", [
    {"learner_vehicles": 0}, {"background_vehicles": 8}, {"driver_spread": float("nan")},
    {"periods": (TrafficPeriod(2, 0), TrafficPeriod(1, 0))},
    {"road_weights": {0: 0}}, {"drivers": (DriverBehavior(),)},
])
def test_invalid_scenario_settings_are_rejected(values):
    with pytest.raises(ValueError):
        Scenario(**values)


def test_missing_road_is_rejected(grid):
    with pytest.raises(ValueError, match="道路 ID"):
        environment(grid, Scenario(events=(RoadEvent(9999),)))


def test_json_settings_and_environment_configuration(tmp_path, grid, monkeypatch):
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps({"model": "dynamic", "physics": {"friction": 0.3}, "signals": {"mode": "adaptive"}, "background_vehicles": 1}), encoding="utf-8")
    scenario = load_scenario(path)
    assert scenario.dynamics.friction == pytest.approx(0.3)
    assert scenario.signals.start_hour == pytest.approx(scenario.start_hour)
    monkeypatch.setenv("DRIVERL_SCENARIO", str(path))
    env = environment(grid)
    assert env.world.fleet.dynamics.friction == pytest.approx(0.3)
    assert env.world.signals.plan.mode == "adaptive"
    path.write_text('{"unknown_key":1}', encoding="utf-8")
    with pytest.raises(TypeError):
        load_scenario(path)


def test_supplied_scenarios_load():
    from pathlib import Path

    paths = Path(__file__).resolve().parents[1] / "scenarios"
    for path in paths.glob("*.json"):
        assert load_scenario(path).name


def test_signal_source_cache_and_wire_roundtrip():
    data = synthetic_grid()
    data.signals = _build_signals({101}, {101: 6}, data.edges, True)
    assert {s.source for s in data.signals if s.node_id == 6} == {"osm"}
    assert {s.source for s in data.signals if s.node_id != 6} == {"synthetic"}
    preset = MapPreset("grid", "grid", "", 0, 0, data.radius_m)
    restored = _from_cache_dict(_to_cache_dict(data), preset)
    assert [s.source for s in restored.signals] == [s.source for s in data.signals]
    assert [s["source"] for s in restored.to_wire()["signals"]] == [s.source for s in data.signals]


def test_evaluation_excludes_background_and_unfinished_episodes():
    metrics = EvaluationMetrics()
    assert metrics.to_dict()["arrival_rate"] is None
    result = StepResult(
        np.zeros((2, 1)), np.zeros(2), np.ones(2, dtype=bool), np.ones(2, dtype=bool),
        episodes=[EpisodeResult(0, 0, 5, "goal", signal_violations=1, lane_departures=1), EpisodeResult(1, 0, 5, "collision")],
        drive=DriveState(np.array([4, 10]), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)),
    )
    metrics.observe(result, np.array([True, False]))
    output = metrics.to_dict()
    assert output["completed_episodes"] == 1
    assert output["arrival_rate"] == output["lane_departure_rate"] == output["signal_violation_rate"] == 1
    assert output["collision_rate"] == 0
    assert output["mean_speed_m_s"] == 4
    result.drive.ground_speed = np.array([5, 20])
    next_metrics = EvaluationMetrics()
    next_metrics.observe(result, np.array([True, False]))
    assert next_metrics.to_dict()["mean_speed_m_s"] == 5


def test_low_friction_maneuver_stops_later():
    simple = maneuvers(Scenario())
    slippery = maneuvers(Scenario(dynamics=VehicleDynamics(friction=0.2)))
    assert simple["braking"]["stopped"] and slippery["braking"]["stopped"]
    assert slippery["braking"]["distance_m"] > simple["braking"]["distance_m"] * 2


def test_cli_compares_conditions_and_restores_training_state(tmp_path, monkeypatch):
    import sys
    import evaluate_scenarios as cli
    from app.rl.ppo import PPOTrainer

    checkpoint = tmp_path / "policy.pt"
    trainer = PPOTrainer(config.OBS_DIM, 2, params=SimParams(), num_agents=config.MAX_VEHICLES, hidden_sizes=(32, 32))
    trainer.save(checkpoint)
    scenario = tmp_path / "scenario.json"
    scenario.write_text('{"name":"test","background_vehicles":1}', encoding="utf-8")
    output = tmp_path / "report.json"
    real_make_env = cli.make_env

    def small_env(index, setting, weather, seed, max_speed):
        env = real_make_env(index, setting, weather, seed, max_speed)
        env.params.rollout_length = 16
        return env

    monkeypatch.setattr(cli, "make_env", small_env)
    monkeypatch.setattr(sys, "argv", ["evaluate_scenarios.py", "--checkpoint", str(checkpoint), "--scenarios", str(scenario), "--presets", "grid", "--weather", "clear", "rain", "--steps", "2", "--train-steps", "32", "--output", str(output)])
    before = checkpoint.read_bytes()
    cli.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["results"]) == 2
    assert all(result["training_updates"] > 0 for result in report["results"])
    assert all(result["obey_signals"] for result in report["results"])
    assert checkpoint.read_bytes() == before
    with pytest.raises(SystemExit):
        cli.main()
