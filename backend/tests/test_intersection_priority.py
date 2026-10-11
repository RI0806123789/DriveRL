"""交差点のカメラ追跡、優先判断、操作介入の境界を検査する。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app import config
from app.contracts import IntersectionDecision, SimParams
from app.map.index import build_map_index
from app.percep.types import CAMERA_RIG, DetClass, Detection, PerceptionResult
from app.sim.env import SimulationEnv
from app.sim.intersection import IntersectionSupervisor
from app.sim.world import Route, World
from tests.test_map_intersections import cross_map


def fixture_world(*, signals: bool = True, path: list[int] | None = None) -> World:
    index = build_map_index(cross_map(signals=signals))
    world = World(index, np.random.default_rng(134))
    points, legs = index.lane_route(path or [1, 0, 4])
    world.install_route(0, Route(np.asarray(points, dtype=np.float32), tuple(legs)))
    world.fleet.active[0] = True
    place(world, 0, world.slots[0].intersections[0].entry_arc - 4.0)
    return world


def place(world: World, slot: int, arc: float, speed: float = 0.0) -> None:
    state = world.slots[slot]
    world.arc[slot] = state.arc_position = arc
    world.fleet.x[slot] = np.interp(arc, state.route_cum, state.route[:, 0])
    world.fleet.y[slot] = np.interp(arc, state.route_cum, state.route[:, 1])
    x0 = np.interp(max(0, arc - 0.2), state.route_cum, state.route[:, 0])
    y0 = np.interp(max(0, arc - 0.2), state.route_cum, state.route[:, 1])
    x1 = np.interp(arc + 0.2, state.route_cum, state.route[:, 0])
    y1 = np.interp(arc + 0.2, state.route_cum, state.route[:, 1])
    world.fleet.heading[slot] = math.atan2(y1 - y0, x1 - x0)
    world.fleet.speed[slot] = speed


def camera_views(world: World, actors: list[tuple[DetClass, float, float]]) -> tuple:
    x, y, heading = float(world.fleet.x[0]), float(world.fleet.y[0]), float(world.fleet.heading[0])
    views = {camera.key: PerceptionResult(0) for camera in CAMERA_RIG}
    for cls, wx, wy in actors:
        dx, dy = wx - x, wy - y
        fx, fy = dx * math.cos(heading) + dy * math.sin(heading), -dx * math.sin(heading) + dy * math.cos(heading)
        for camera in CAMERA_RIG:
            cx, cy = fx - camera.forward, fy + camera.right
            angle = math.atan2(cy, cx) - camera.yaw
            angle = math.atan2(math.sin(angle), math.cos(angle))
            if abs(angle) >= math.atan(camera.width / (2.0 * camera.focal_px)):
                continue
            centre = 0.5 - math.tan(angle) * camera.focal_px / camera.width
            views[camera.key].detections.append(Detection(cls, centre - 0.02, 0.4, centre + 0.02, 0.6, 1.0, distance=math.hypot(cx, cy)))
    surround = {0: {k: v for k, v in views.items() if k != "front"}}
    ages = {0: {k: 0.0 for k in surround[0]}}
    poses = {0: {k: (x, y, heading) for k in surround[0]}}
    return {0: views["front"]}, surround, ages, poses


def evaluate_camera(controller: IntersectionSupervisor, world: World, now: float, actors: list[tuple[DetClass, float, float]]) -> IntersectionDecision:
    front, surround, ages, poses = camera_views(world, actors)
    controller.evaluate(world, now, front, surround, ages, poses, {})
    return controller.decisions[0]


def test_camera_oncoming_speed_and_clear_gap() -> None:
    world = fixture_world()
    controller = IntersectionSupervisor()
    evaluate_camera(controller, world, 0.0, [(DetClass.VEHICLE, 20.0, -1.75)])
    decision = evaluate_camera(controller, world, 0.1, [(DetClass.VEHICLE, 19.0, -1.75)])
    assert decision.source == "camera" and decision.reason == "oncoming"
    assert math.isfinite(decision.speed_cap) and decision.gap_seconds is not None
    decision = evaluate_camera(controller, world, 1.3, [])
    assert decision.reason == "" and math.isinf(decision.speed_cap)


def test_speed_difference_changes_temporal_conflict() -> None:
    world = fixture_world()
    reasons = []
    for speed in (1.0, 12.0):
        controller = IntersectionSupervisor()
        evaluate_camera(controller, world, 0.0, [(DetClass.VEHICLE, 29.0, -1.75)])
        decision = evaluate_camera(controller, world, 0.1, [(DetClass.VEHICLE, 29.0 - speed * 0.1, -1.75)])
        reasons.append(decision.reason)
    assert reasons == ["", "oncoming"]


def test_hidden_truth_never_changes_camera_decision() -> None:
    world = fixture_world()
    controller = IntersectionSupervisor()
    a = evaluate_camera(controller, world, 0.0, []).to_wire()
    world.fleet.active[1] = True
    world.fleet.x[1], world.fleet.y[1], world.fleet.speed[1] = 12.0, -1.75, 15.0
    world.set_player((0.0, 0.0))
    b = evaluate_camera(controller, world, 0.1, []).to_wire()
    assert a == b


def test_missing_and_stale_camera_require_visibility() -> None:
    world = fixture_world()
    controller = IntersectionSupervisor()
    front, surround, ages, poses = camera_views(world, [])
    ages[0] = {key: 0.8 for key in ages[0]}
    controller.evaluate(world, 0.0, front, surround, ages, poses, {})
    assert controller.decisions[0].reason == "visibility"
    assert math.isfinite(controller.decisions[0].speed_cap)


def test_short_occlusion_keeps_approaching_vehicle_without_timeout_release() -> None:
    world = fixture_world()
    controller = IntersectionSupervisor()
    evaluate_camera(controller, world, 0.0, [(DetClass.VEHICLE, 15.0, -1.75)])
    evaluate_camera(controller, world, 0.1, [(DetClass.VEHICLE, 14.5, -1.75)])
    assert evaluate_camera(controller, world, 0.4, []).reason == "oncoming"
    decision = evaluate_camera(controller, world, 14.0, [(DetClass.VEHICLE, 13.5, -1.75)])
    assert decision.reason == "oncoming"
    assert decision.waiting_seconds > 6.0


def test_exit_queue_and_curved_exit_pedestrian() -> None:
    world = fixture_world()
    p = world.slots[0].intersections[0]
    place(world, 0, p.entry_arc - 15.0)
    state = world.slots[0]
    for cls, offset, reason in ((DetClass.VEHICLE, 3.0, "exit_blocked"), (DetClass.PEDESTRIAN, 1.0, "pedestrian")):
        controller = IntersectionSupervisor()
        arc = p.exit_arc + offset
        x = float(np.interp(arc, state.route_cum, state.route[:, 0]))
        y = float(np.interp(arc, state.route_cum, state.route[:, 1]))
        assert evaluate_camera(controller, world, offset, [(cls, x, y)]).reason == reason


def test_stationary_crossing_vehicle_still_blocks_and_route_reset_discards_tracks() -> None:
    world = fixture_world()
    controller = IntersectionSupervisor()
    evaluate_camera(controller, world, 0.0, [(DetClass.VEHICLE, 0.0, -1.75)])
    assert evaluate_camera(controller, world, 0.1, [(DetClass.VEHICLE, 0.0, -1.75)]).reason == "oncoming"
    world.route_serial[0] += 1
    assert evaluate_camera(controller, world, 0.2, []).reason == ""


def test_background_includes_player_but_labels_source() -> None:
    world = fixture_world()
    p = world.slots[0].intersections[0]
    state = world.slots[0]
    world.set_player((float(np.interp(p.exit_arc, state.route_cum, state.route[:, 0])), float(np.interp(p.exit_arc, state.route_cum, state.route[:, 1]))))
    controller = IntersectionSupervisor()
    controller.evaluate(world, 0.0, {}, {}, {}, {}, {}, frozenset([0]))
    assert controller.decisions[0].source == "background"
    assert controller.decisions[0].reason == "pedestrian"


def test_stop_requires_actual_stationary_second() -> None:
    world = fixture_world()
    controller = IntersectionSupervisor()
    p = world.slots[0].intersections[0]
    place(world, 0, p.entry_arc - config.VEHICLE_LENGTH / 2 - 0.6)
    x, y = float(world.fleet.x[0]) + 8.0, float(world.fleet.y[0]) + 3.0
    assert evaluate_camera(controller, world, 0.0, [(DetClass.STOP_SIGN, x, y)]).reason == "stop"
    assert evaluate_camera(controller, world, 0.5, []).reason == "stop"
    place(world, 0, float(world.arc[0]), speed=0.2)
    assert evaluate_camera(controller, world, 0.9, []).reason == "stop"
    place(world, 0, float(world.arc[0]))
    assert evaluate_camera(controller, world, 1.0, []).reason == "stop"
    assert evaluate_camera(controller, world, 2.1, []).reason == ""


@pytest.mark.parametrize("assist,expert,practical", [(False, False, False), (True, False, False), (False, True, False), (False, False, True)])
def test_environment_intervention_and_learning_masks(assist: bool, expert: bool, practical: bool) -> None:
    env = SimulationEnv(build_map_index(cross_map(signals=True)), SimParams(vehicle_count=1, pedestrian_count=0, safety_assist=assist), compute_observations=False)
    env.autopilot_all = practical
    env.intersections.decisions[0] = IntersectionDecision(0, reason="oncoming", speed_cap=0.0)
    env._run_safety = lambda: None
    env.world.fleet.speed[0] = 0.1
    action = np.zeros((config.MAX_VEHICLES, config.ACTION_DIM), dtype=np.float32)
    action[0, 0] = 1.0
    mask = np.zeros(config.MAX_VEHICLES, dtype=bool)
    mask[0] = expert
    result = env.step(action, expert=mask)
    if assist or expert or practical:
        assert env.intersections.decisions[0].imposed
        assert not result.learn[0] or result.assisted[0]
    else:
        assert result.learn[0] and not env.intersections.decisions[0].imposed
    assert result.assisted[0] == expert
    assert env.world.fleet.active[0]


@pytest.mark.parametrize("existing_limit", ["curve", "signal"])
def test_existing_constraint_with_same_braking_keeps_policy_step_learnable(
    existing_limit: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = []
    for cap in (math.inf, 4.0):
        env = SimulationEnv(
            build_map_index(cross_map(signals=True)),
            SimParams(vehicle_count=1, pedestrian_count=0, safety_assist=True),
            compute_observations=False,
        )
        env.world = fixture_world()
        passage = env.world.slots[0].intersections[0]
        place(env.world, 0, passage.entry_arc - 15.0, speed=8.0)
        env.intersections.decisions[0] = IntersectionDecision(0, reason="oncoming", speed_cap=cap)
        monkeypatch.setattr(env, "_run_safety", lambda: None)
        for name in ("curve", "signal", "posted", "stop"):
            value = 5.0 if name == existing_limit else math.inf
            monkeypatch.setattr(
                env.world, f"{name}_speed_limits",
                lambda value=value: np.full(config.MAX_VEHICLES, value, dtype=np.float64),
            )
        action = np.zeros((config.MAX_VEHICLES, config.ACTION_DIM), dtype=np.float32)
        result = env.step(action)
        assert result.learn[0] and not result.assisted[0]
        assert not env.intersections.decisions[0].imposed
        assert env.world.fleet.active[0]
        commands.append((float(env.world.throttle[0]), float(env.world.fleet.speed[0])))
    assert commands[0][0] == commands[1][0] == -1.0
    assert commands[0][1] == pytest.approx(commands[1][1])


def test_expert_intervention_remains_visible_after_step_end_reassessment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = SimulationEnv(
        build_map_index(cross_map(signals=True)),
        SimParams(vehicle_count=1, pedestrian_count=0, safety_assist=False),
        compute_observations=False,
    )
    env.world = fixture_world()
    env.intersections.decisions[0] = IntersectionDecision(0, reason="oncoming", speed_cap=0.0)
    env.world.fleet.speed[0] = 0.1
    reassessed: list[IntersectionDecision] = []

    def reassess(*args, **kwargs) -> None:
        next_decision = IntersectionDecision(0, state="waiting", reason="oncoming", speed_cap=0.0)
        reassessed.append(next_decision)
        env.intersections.decisions = {0: next_decision}

    monkeypatch.setattr(env.intersections, "evaluate", reassess)
    action = np.zeros((config.MAX_VEHICLES, config.ACTION_DIM), dtype=np.float32)
    expert = np.zeros(config.MAX_VEHICLES, dtype=bool)
    expert[0] = True
    result = env.step(action, expert=expert)
    assert reassessed and env.intersections.decisions[0] is reassessed[-1]
    assert result.assisted[0] and result.learn[0]
    assert reassessed[-1].imposed
    vehicle = env.snapshot(1, env.sim_time).to_wire()["vehicles"][0]
    assert vehicle["intersection"]["imposed"] is True
    assert vehicle["assist"] == "intersection_yield"
    result = env.step(action)
    vehicle = env.snapshot(2, env.sim_time).to_wire()["vehicles"][0]
    assert result.learn[0] and not result.assisted[0]
    assert vehicle["intersection"]["imposed"] is False
    assert vehicle["assist"] == ""


def test_captured_ego_pose_keeps_stationary_other_vehicle_velocity_zero() -> None:
    world = fixture_world()
    controller = IntersectionSupervisor()
    actor_xy = (float(world.fleet.x[0]) + 1.0, float(world.fleet.y[0]) + 12.0)
    evaluate_camera(controller, world, 0.0, [(DetClass.VEHICLE, *actor_xy)])
    world.fleet.x[0] += 0.8
    world.fleet.y[0] += 0.2
    world.fleet.heading[0] = 0.15
    front, surround, ages, poses = camera_views(world, [(DetClass.VEHICLE, *actor_xy)])
    assert surround[0]["left"].detections and not front[0].detections
    world.fleet.x[0] += 1.1
    world.fleet.y[0] += 0.4
    world.fleet.heading[0] = 0.3
    ages[0] = {key: 0.1 for key in ages[0]}
    controller.evaluate(world, 0.2, front, surround, ages, poses, {})
    actors = controller._states[0].actors
    assert len(actors) == 1
    actor = actors[0]
    assert actor.measured and actor.at == pytest.approx(0.1)
    assert (actor.x, actor.y) == pytest.approx(actor_xy, abs=1e-5)
    assert (actor.vx, actor.vy) == pytest.approx((0.0, 0.0), abs=1e-4)


def test_reused_surround_image_never_creates_new_velocity_sample() -> None:
    world = fixture_world()
    controller = IntersectionSupervisor()
    actor_xy = (float(world.fleet.x[0]) + 1.0, float(world.fleet.y[0]) + 12.0)
    evaluate_camera(controller, world, 0.0, [(DetClass.VEHICLE, *actor_xy)])
    actor_xy = (actor_xy[0] + 0.5, actor_xy[1])
    front, surround, ages, poses = camera_views(world, [(DetClass.VEHICLE, *actor_xy)])
    assert surround[0]["left"].detections and not front[0].detections
    controller.evaluate(world, 0.1, front, surround, ages, poses, {})
    actor = controller._states[0].actors[0]
    assert actor.measured
    assert (actor.vx, actor.vy) == pytest.approx((5.0, 0.0), abs=1e-4)
    for now in (0.15, 0.3, 0.5):
        world.fleet.x[0] += 1.0
        world.fleet.heading[0] += 0.1
        ages[0] = {key: now - 0.1 for key in ages[0]}
        controller.evaluate(world, now, front, surround, ages, poses, {})
        actors = controller._states[0].actors
        assert actors == [actor]
        assert actor.at == pytest.approx(0.1)
        assert (actor.vx, actor.vy) == pytest.approx((5.0, 0.0), abs=1e-4)
    ages[0] = {key: 1.1 for key in ages[0]}
    controller.evaluate(world, 1.2, front, surround, ages, poses, {})
    assert not controller._states[0].actors


def test_same_time_replacement_updates_position_without_inventing_velocity() -> None:
    world = fixture_world()
    controller = IntersectionSupervisor()
    evaluate_camera(controller, world, 0.0, [(DetClass.VEHICLE, 20.0, -1.75)])
    evaluate_camera(controller, world, 0.0, [(DetClass.VEHICLE, 19.5, -1.75)])
    actors = controller._states[0].actors
    assert len(actors) == 1
    assert actors[0].x == pytest.approx(19.5)
    assert not actors[0].measured
    evaluate_camera(controller, world, 0.1, [(DetClass.VEHICLE, 19.0, -1.75)])
    actors = controller._states[0].actors
    assert len(actors) == 1 and actors[0].measured
    assert (actors[0].vx, actors[0].vy) == pytest.approx((-5.0, 0.0), abs=1e-4)


def test_nearby_distinct_vehicles_in_one_camera_keep_separate_tracks() -> None:
    world = fixture_world()
    controller = IntersectionSupervisor()
    x, y = float(world.fleet.x[0]), float(world.fleet.y[0])
    positions = [(x + 20.0, y - 1.3), (x + 20.0, y + 1.3)]
    front, surround, ages, poses = camera_views(world, [(DetClass.VEHICLE, *point) for point in positions])
    assert len(front[0].detections) == 2
    controller.evaluate(world, 0.0, front, surround, ages, poses, {})
    actors = controller._states[0].actors
    assert len(actors) == 2
    tracked = sorted(((a.x, a.y) for a in actors), key=lambda point: point[1])
    np.testing.assert_allclose(tracked, sorted(positions, key=lambda point: point[1]), atol=1e-5)


def test_two_cameras_of_same_vehicle_merge_without_duplicate_velocity_tracks() -> None:
    world = fixture_world()
    controller = IntersectionSupervisor()
    x, y = float(world.fleet.x[0]), float(world.fleet.y[0])
    for now, shift in ((0.0, 0.0), (0.1, -0.2)):
        front, _, _, _ = camera_views(world, [(DetClass.VEHICLE, x + 4.0 + shift, y - 2.0)])
        _, surround, ages, poses = camera_views(world, [(DetClass.VEHICLE, x + 2.0 + shift, y - 4.0)])
        assert len(front[0].detections) == len(surround[0]["right"].detections) == 1
        controller.evaluate(world, now, front, surround, ages, poses, {})
        actors = controller._states[0].actors
        assert len(actors) == 1
        assert (actors[0].x, actors[0].y) == pytest.approx((x + 3.0 + shift, y - 3.0), abs=1e-5)
    assert actors[0].measured
    assert (actors[0].vx, actors[0].vy) == pytest.approx((-2.0, 0.0), abs=1e-4)


@pytest.mark.parametrize("expert,speed,command,cap", [(False, 2.0, -1.0, 1.0), (True, 0.0, 0.0, 4.0)])
def test_lower_intersection_limit_without_command_change_is_not_intervention(
    expert: bool, speed: float, command: float, cap: float, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = []
    for intersection_cap in (math.inf, cap):
        env = SimulationEnv(
            build_map_index(cross_map(signals=True)),
            SimParams(vehicle_count=1, pedestrian_count=0, safety_assist=not expert),
            compute_observations=False,
        )
        env.world = fixture_world()
        passage = env.world.slots[0].intersections[0]
        place(env.world, 0, passage.entry_arc - 15.0, speed=speed)
        env.intersections.decisions[0] = IntersectionDecision(0, reason="oncoming", speed_cap=intersection_cap)
        monkeypatch.setattr(env, "_run_safety", lambda: None)
        for name in ("curve", "signal", "posted", "stop"):
            monkeypatch.setattr(env.world, f"{name}_speed_limits", lambda: np.full(config.MAX_VEHICLES, math.inf))
        action = np.zeros((config.MAX_VEHICLES, config.ACTION_DIM), dtype=np.float32)
        action[0, 0] = command
        mask = np.zeros(config.MAX_VEHICLES, dtype=bool)
        mask[0] = expert
        result = env.step(action, expert=mask)
        assert result.active[0] and result.learn[0] and result.assisted[0] == expert
        assert not env.intersections.decisions[0].imposed
        commands.append((float(env.world.throttle[0]), float(env.world.fleet.speed[0])))
    assert commands[0] == pytest.approx(commands[1])


def test_snapshot_keeps_intersection_decision_before_next_step_mutates_it() -> None:
    env = SimulationEnv(build_map_index(cross_map()), SimParams(vehicle_count=0, pedestrian_count=0), compute_observations=False)
    env.world = fixture_world()
    decision = IntersectionDecision(0, reason="oncoming", imposed=True)
    env.intersections.decisions[0] = decision
    frame = env.snapshot(1, 0.05)
    decision.imposed = False
    decision.reason = "visibility"
    wire = frame.to_wire()
    vehicle = next(v for v in wire["vehicles"] if v["id"] == 0)
    assert vehicle["intersection"]["imposed"] is True
    assert vehicle["intersection"]["reason"] == "oncoming"
