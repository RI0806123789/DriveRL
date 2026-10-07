"""詳細車両物理の解析解と安定性を検査する。"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from app import config
from app.sim.dynamics import VehicleDynamics
from app.sim.vehicle import VehicleFleet


def fleet(parameters: VehicleDynamics | None = None, size: int = 1) -> VehicleFleet:
    vehicle = VehicleFleet(size, dynamics=parameters or VehicleDynamics())
    vehicle.active[:] = True
    return vehicle


def advance(vehicle: VehicleFleet, accel: float, steer: float, seconds: float, dt: float = 0.01) -> None:
    for _ in range(round(seconds / dt)):
        vehicle.step(np.full(vehicle.size, accel), np.full(vehicle.size, steer), dt, 40)


@pytest.mark.parametrize("name,value", [
    ("mass_kg", 0), ("yaw_inertia_kg_m2", -1), ("friction", float("nan")),
    ("throttle_tau", float("inf")), ("brake_tau", False),
    ("front_axle_m", config.WHEELBASE), ("substep_s", 0.1),
    ("substep_s", 1e-12), ("mass_kg", 1e30), ("yaw_inertia_kg_m2", 1e30),
    ("friction", 50), ("front_cornering_stiffness", 1e30), ("low_speed_m_s", 1e-20),
    ("throttle_tau", 1e30), ("cg_height_m", -0.1),
])
def test_invalid_parameters(name: str, value: float) -> None:
    with pytest.raises(ValueError):
        VehicleDynamics(**{name: value})


def test_parameters_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        VehicleDynamics().friction = 0.5


def test_actuator_response_matches_exponential() -> None:
    parameters = VehicleDynamics()
    vehicle = fleet(parameters)
    advance(vehicle, 0.4, 0.1, 0.3)
    assert vehicle.throttle[0] == pytest.approx(0.4 * (1 - np.exp(-0.3 / parameters.throttle_tau)), rel=1e-6)
    assert vehicle.steer[0] == pytest.approx(0.1 * config.MAX_STEER * (1 - np.exp(-0.3 / parameters.steering_tau)), rel=1e-6)
    vehicle = fleet(parameters)
    vehicle.speed[:] = 10
    advance(vehicle, -1, 0, 0.3)
    assert vehicle.brake[0] == pytest.approx(1 - np.exp(-0.3 / parameters.brake_tau), rel=1e-6)


def test_straight_motion_and_throttle_integral() -> None:
    parameters = VehicleDynamics()
    vehicle = fleet(parameters)
    advance(vehicle, 1, 0, 1, dt=0.001)
    expected = config.MAX_ACCEL * (1 - parameters.throttle_tau * (1 - np.exp(-1 / parameters.throttle_tau)))
    assert vehicle.speed[0] == pytest.approx(expected, abs=0.002)
    assert vehicle.x[0] > 0
    assert vehicle.y[0] == vehicle.yaw_rate[0] == vehicle.lateral_speed[0] == 0


def test_zero_response_delays_apply_commands_immediately() -> None:
    vehicle = fleet(VehicleDynamics(throttle_tau=0, brake_tau=0, steering_tau=0, cg_height_m=0))
    vehicle.step(np.full(1, 0.5), np.full(1, 0.01), 0.01, 40)
    assert vehicle.throttle[0] == 0.5
    assert vehicle.steer[0] == pytest.approx(0.01 * config.MAX_STEER)
    assert vehicle.speed[0] == pytest.approx(0.5 * config.MAX_ACCEL * 0.01, rel=1e-4)
    vehicle.step(-np.ones(1), np.zeros(1), 0.01, 40)
    assert vehicle.throttle[0] == 0
    assert vehicle.brake[0] == 1
    assert vehicle.speed[0] == pytest.approx(0, abs=1e-6)


def test_initial_small_angle_response_obeys_yaw_moment() -> None:
    parameters = VehicleDynamics()
    vehicle = fleet(parameters)
    vehicle.speed[:] = 15
    delta = 1e-4
    vehicle.steer[:] = delta
    dt = 1e-4
    vehicle.step(np.zeros(1), np.full(1, delta / config.MAX_STEER), dt, 40)
    expected_yaw = parameters.front_axle_m * parameters.front_cornering_stiffness * delta / parameters.yaw_inertia_kg_m2 * dt
    expected_lateral = parameters.front_cornering_stiffness * delta / parameters.mass_kg * dt
    assert vehicle.yaw_rate[0] == pytest.approx(expected_yaw, rel=1e-4)
    assert vehicle.lateral_speed[0] == pytest.approx(expected_lateral, abs=1e-9)


def test_steering_response_is_symmetric_and_can_slide() -> None:
    vehicle = fleet(size=2)
    vehicle.speed[:] = 18
    for _ in range(100):
        vehicle.step(np.zeros(2), np.array([0.6, -0.6]), 0.01, 40)
    assert vehicle.y[0] > 0
    assert abs(vehicle.lateral_speed[0]) > 0.1
    assert vehicle.x[0] == pytest.approx(vehicle.x[1], rel=1e-6)
    assert vehicle.y[0] == pytest.approx(-vehicle.y[1], rel=1e-6)
    assert vehicle.yaw_rate[0] == pytest.approx(-vehicle.yaw_rate[1], rel=1e-6)


def test_small_angle_steady_turn_matches_linear_bicycle_solution() -> None:
    parameters = VehicleDynamics()
    vehicle = fleet(parameters)
    vehicle.speed[:] = 15
    delta = 0.01
    advance(vehicle, 0, delta / config.MAX_STEER, 5)
    front = parameters.front_axle_m
    rear = config.WHEELBASE - front
    understeer = parameters.mass_kg / config.WHEELBASE * (
        rear / parameters.front_cornering_stiffness - front / parameters.rear_cornering_stiffness
    )
    speed = float(vehicle.speed[0])
    expected_yaw = speed * delta / (config.WHEELBASE + understeer * speed ** 2)
    expected_lateral_speed = rear * expected_yaw - (
        front * parameters.mass_kg * speed ** 2 * expected_yaw
        / (config.WHEELBASE * parameters.rear_cornering_stiffness)
    )
    assert vehicle.yaw_rate[0] == pytest.approx(expected_yaw, rel=0.01)
    assert vehicle.lateral_speed[0] == pytest.approx(expected_lateral_speed, rel=0.01)


def test_low_friction_increases_braking_distance() -> None:
    stopping_distances = []
    for friction in (0.9, 0.2):
        vehicle = fleet(VehicleDynamics(friction=friction))
        vehicle.speed[:] = 15
        advance(vehicle, -1, 0, 10)
        assert vehicle.speed[0] == 0
        stopping_distances.append(vehicle.x[0])
    assert stopping_distances[1] > stopping_distances[0] * 2


def test_combined_braking_reduces_lateral_force_budget() -> None:
    parameters = VehicleDynamics(friction=0.5)
    turning = fleet(parameters)
    braking = fleet(parameters)
    for vehicle in (turning, braking):
        vehicle.speed[:] = 20
        vehicle.steer[:] = 0.3
    braking.brake[:] = 1
    turning.step(np.zeros(1), np.full(1, 0.3 / config.MAX_STEER), 0.001, 40)
    braking.step(-np.ones(1), np.full(1, 0.3 / config.MAX_STEER), 0.001, 40)
    assert braking.lateral_accel[0] < turning.lateral_accel[0] * 0.1
    assert np.hypot(braking.longitudinal_accel[0], braking.lateral_accel[0]) <= parameters.friction * 9.81 * 1.01


def test_load_transfer_changes_front_tire_response() -> None:
    yaw_rates = []
    for height in (0.1, 1.0):
        vehicle = fleet(VehicleDynamics(cg_height_m=height))
        vehicle.speed[:] = 20
        vehicle.steer[:] = 0.3
        vehicle.throttle[:] = 1
        vehicle.step(np.ones(1), np.full(1, 0.3 / config.MAX_STEER), 0.001, 40)
        yaw_rates.append(vehicle.yaw_rate[0])
    assert yaw_rates[1] < yaw_rates[0]


def test_low_speed_reverse_and_stop_are_stable() -> None:
    vehicle = fleet()
    assert vehicle.set_gear(0, -1)
    advance(vehicle, 1, 1, 3)
    assert -config.REVERSE_MAX_SPEED <= vehicle.speed[0] < 0
    assert vehicle.yaw_rate[0] < 0
    advance(vehicle, -1, -1, 3)
    assert abs(vehicle.speed[0]) < 1e-6
    assert abs(vehicle.yaw_rate[0]) < 1e-6
    assert abs(vehicle.lateral_speed[0]) < 1e-6
    assert np.isfinite(vehicle.corners()).all()
    assert vehicle.set_gear(0, 1)


def test_inactive_and_reset_transients() -> None:
    vehicle = fleet(size=2)
    vehicle.active[1] = False
    vehicle.lateral_speed[1] = 0.75
    advance(vehicle, 1, 0.5, 0.5)
    assert vehicle.throttle[1] == vehicle.speed[1] == vehicle.x[1] == 0
    assert vehicle.lateral_speed[1] == 0.75
    vehicle.reset_slot(0, 3, 4, 0.5)
    for name in ("speed", "steer", "lateral_speed", "yaw_rate", "throttle", "brake", "longitudinal_accel", "lateral_accel"):
        assert getattr(vehicle, name)[0] == 0
    vehicle.throttle[0] = 1
    vehicle.steer[0] = 0.2
    assert vehicle.set_gear(0, -1)
    assert vehicle.throttle[0] == vehicle.steer[0] == 0
    vehicle.speed[0] = -1
    assert not vehicle.set_gear(0, 1)


@pytest.mark.parametrize("dt", [0, -0.1, float("nan"), float("inf"), 1e30, 0.100001])
def test_invalid_timestep_does_not_mutate(dt: float) -> None:
    vehicle = fleet()
    with pytest.raises(ValueError):
        vehicle.step(np.ones(1), np.zeros(1), dt, 40)
    assert vehicle.speed[0] == vehicle.throttle[0] == 0


@pytest.mark.parametrize("accel,steer,limit", [
    (float("nan"), 0, 40), (0, float("nan"), 40), (0, 0, float("inf")), (0, 0, -1),
])
def test_invalid_command_and_speed_limit(accel: float, steer: float, limit: float) -> None:
    vehicle = fleet()
    with pytest.raises(ValueError):
        vehicle.step(np.array([accel]), np.array([steer]), 0.05, limit)
    assert vehicle.speed[0] == vehicle.throttle[0] == 0


def test_forward_speed_limit_and_steering_rate() -> None:
    vehicle = fleet()
    vehicle.speed[:] = 9
    vehicle.throttle[:] = 1
    vehicle.step(np.ones(1), np.ones(1), 0.01, 9)
    assert vehicle.speed[0] <= 9
    assert 0 < vehicle.steer[0] <= config.STEER_RATE * 0.01


def inertial_velocity(vehicle: VehicleFleet) -> np.ndarray:
    heading = float(vehicle.heading[0])
    vx, vy = float(vehicle.speed[0]), float(vehicle.lateral_speed[0])
    return np.array([vx * np.cos(heading) - vy * np.sin(heading), vx * np.sin(heading) + vy * np.cos(heading)])


def test_slalom_inertial_acceleration_respects_friction_through_stopping() -> None:
    parameters = VehicleDynamics(friction=0.35, brake_tau=0.2, steering_tau=0.2)
    vehicle = fleet(parameters)
    vehicle.speed[:] = 20
    visited_low_speed = False
    peak_acceleration = 0.0
    for i in range(500):
        previous = inertial_velocity(vehicle)
        accel = 0 if i < 200 else -1
        steer = 0.5 * np.sin(i * 0.05 * np.pi)
        vehicle.step(np.array([accel]), np.array([steer]), 0.05, 40)
        measured = np.linalg.norm(inertial_velocity(vehicle) - previous) / 0.05
        peak_acceleration = max(peak_acceleration, measured)
        visited_low_speed |= abs(vehicle.speed[0]) < parameters.low_speed_m_s
    assert visited_low_speed
    assert peak_acceleration <= parameters.friction * 9.81 + 0.002
    assert np.linalg.norm(inertial_velocity(vehicle)) < 1e-3


def test_sideways_slide_keeps_momentum_and_prevents_gear_change() -> None:
    parameters = VehicleDynamics(friction=0.35)
    vehicle = fleet(parameters)
    vehicle.lateral_speed[:] = 14
    assert not vehicle.set_gear(0, -1)
    before = inertial_velocity(vehicle)
    vehicle.step(-np.ones(1), np.zeros(1), 0.01, 40)
    acceleration = np.linalg.norm(inertial_velocity(vehicle) - before) / 0.01
    assert vehicle.lateral_speed[0] > 13.9
    assert acceleration <= parameters.friction * 9.81 + 0.002
    advance(vehicle, -1, 0, 6)
    assert np.linalg.norm(inertial_velocity(vehicle)) < 1e-3
    assert vehicle.set_gear(0, -1)
    vehicle.yaw_rate[0] = 0.5
    assert not vehicle.set_gear(0, 1)


def test_lower_speed_limit_does_not_remove_momentum_instantly() -> None:
    parameters = VehicleDynamics(friction=0.35)
    vehicle = fleet(parameters)
    vehicle.speed[:] = 20
    before = inertial_velocity(vehicle)
    vehicle.step(-np.ones(1), np.zeros(1), 0.05, 5)
    acceleration = np.linalg.norm(inertial_velocity(vehicle) - before) / 0.05
    assert vehicle.speed[0] > 19
    assert acceleration <= parameters.friction * 9.81 + 0.002


def test_substep_convergence_and_high_speed_stability() -> None:
    trajectories = []
    for substep in (0.01, 0.005):
        vehicle = fleet(VehicleDynamics(substep_s=substep, friction=0.4))
        vehicle.speed[:] = 30
        advance(vehicle, 0.1, 0.8, 5, dt=0.05)
        assert np.isfinite(vehicle.corners()).all()
        assert np.isfinite(vehicle.lateral_speed).all()
        assert 0 <= vehicle.speed[0] <= 40
        trajectories.append(np.array([vehicle.x[0], vehicle.y[0]]))
    assert np.linalg.norm(trajectories[0] - trajectories[1]) < 1
