"""変更の許可・予測車間・撮影時刻・実制御の区別を検証する。"""

import math

import numpy as np
import pytest

from app import config
from app.contracts import Bounds, LaneCorridor, MapData, MapEdge, MapNode, MapPreset, SimParams
from app.map.index import build_map_index
from app.percep.types import CAMERA_RIG, CameraVisibility, DetClass, OcclusionResult
from app.sim.env import SimulationEnv
from app.sim.lane_change import LaneChangeSupervisor, _LaneState, avoidance_path
from app.sim.world import Route, World
from tests.test_intersection_priority import camera_views, place


def lane_map(allowed: bool | None = True):
    return build_map_index(MapData(
        preset_id="lane", name="車線変更", center_lat=0, center_lon=0, radius_m=300,
        bounds=Bounds(-20, -20, 320, 20),
        nodes=[MapNode(0, 0, 0), MapNode(1, 300, 0)],
        edges=[MapEdge(0, 0, 1, 2, 7, True, 13.9, [(0, 0), (300, 0)], 300, lane_change_allowed=allowed)],
        buildings=[], signals=[], signs=[],
    ))


def lane_world(allowed: bool | None = True):
    index = lane_map(allowed)
    world = World(index, np.random.default_rng(135))
    points, legs = index.lane_route([0, 1])
    world.install_route(0, Route(np.array(points, dtype=np.float32), tuple(legs)))
    world.fleet.active[0] = True
    place(world, 0, 20, 4)
    return world


def visibility(depth: float = 30.0):
    return OcclusionResult({cam.key: CameraVisibility(cam.key, True, np.full(35, depth)) for cam in CAMERA_RIG}, [], 30, 30, 0, None, (1, 1, 1, 1), np.zeros(8), 30)


def sensed(controller, world, now, actors, *, ages_override=None, sight=None):
    front, surround, ages, poses = camera_views(world, actors)
    if ages_override is not None:
        ages[0].update(ages_override)
    controller.evaluate(world, now, front, surround, ages, poses, {0: sight or visibility()})
    return controller.decisions[0]


def test_empty_neighbor_proposes_continuous_round_trip():
    world = lane_world()
    controller = LaneChangeSupervisor()
    decision = sensed(controller, world, 0, [(DetClass.OBSTACLE, 70, 1.75)])
    assert decision.state == "changing" and decision.direction == -1
    assert controller.demand(0) == ("rear", "right")
    assert decision.source == "camera" and not decision.imposed
    points, legs = controller.commands[0].plan
    assert np.max(points[:, 1]) == pytest.approx(1.75)
    assert np.min(points[:, 1]) == pytest.approx(-1.75)
    assert np.array_equal(points[0], world.slots[0].route[0])
    assert np.array_equal(points[-1], world.slots[0].route[-1])
    assert all(leg.edge_id == 0 for leg in legs)
    assert np.max(np.linalg.norm(np.diff(points, axis=0), axis=1)) < 2.1


@pytest.mark.parametrize("allowed,reason", [(None, "unknown_regulation"), (False, "regulation")])
def test_regulation_requires_positive_evidence(allowed, reason):
    world = lane_world(allowed)
    controller = LaneChangeSupervisor()
    decision = sensed(controller, world, 0, [(DetClass.OBSTACLE, 70, 1.75)])
    assert decision.reason == reason
    assert controller.commands[0].plan is None


def test_fast_rear_vehicle_blocks_change_after_speed_measurement():
    world = lane_world()
    place(world, 0, 70, 10)
    controller = LaneChangeSupervisor()
    sensed(controller, world, 0, [(DetClass.OBSTACLE, 120, 1.75), (DetClass.VEHICLE, 20, -1.75)])
    decision = sensed(controller, world, 0.1, [(DetClass.OBSTACLE, 120, 1.75), (DetClass.VEHICLE, 22, -1.75)])
    assert decision.reason == "gap"
    assert decision.rear_decel > 2.0
    assert controller.commands[0].plan is None


@pytest.mark.parametrize("actors", [[(DetClass.VEHICLE, 42, -1.75)], [(DetClass.OBSTACLE, 70, -1.75)], [(DetClass.VEHICLE, 20, -1.75)]])
def test_front_rear_and_all_lane_blockers(actors):
    world = lane_world()
    controller = LaneChangeSupervisor()
    decision = sensed(controller, world, 0, [(DetClass.OBSTACLE, 70, 1.75), *actors])
    assert decision.reason == "gap" and controller.commands[0].plan is None


def test_missing_stale_and_occluded_views_stop_start():
    world = lane_world()
    for ages, sight in [({"rear": 1.0}, None), ({"right": math.inf}, None), ({}, visibility(10))]:
        controller = LaneChangeSupervisor()
        decision = sensed(controller, world, 0, [(DetClass.OBSTACLE, 70, 1.75)], ages_override=ages, sight=sight)
        assert decision.reason == "visibility" and controller.commands[0].plan is None


def test_waiting_vehicle_is_never_classified_by_stop_duration():
    world = lane_world()
    controller = LaneChangeSupervisor()
    for now in (0, 0.1, 30):
        decision = sensed(controller, world, now, [(DetClass.VEHICLE, 50, 1.75)])
        assert decision.reason == "traffic" and 0 not in controller.commands


def test_hidden_world_actors_do_not_affect_camera_plan():
    world = lane_world()
    def outcome():
        controller = LaneChangeSupervisor()
        return sensed(controller, world, 0, [(DetClass.OBSTACLE, 70, 1.75)]).to_wire(), controller.commands[0].plan[0]
    original, path = outcome()
    world.fleet.active[1] = True
    world.fleet.x[1], world.fleet.y[1], world.fleet.speed[1] = 15, -1.75, 25
    world.set_player((40, -1.75))
    world.add_obstacle(50, -1.75, 2)
    changed, newer_path = outcome()
    assert original == changed and np.array_equal(path, newer_path)


def test_start_is_cancelled_when_gap_disappears():
    world = lane_world()
    controller = LaneChangeSupervisor()
    sensed(controller, world, 0, [(DetClass.OBSTACLE, 70, 1.75)])
    assert controller.commands[0].plan is not None
    decision = sensed(controller, world, 0.1, [(DetClass.OBSTACLE, 70, 1.75), (DetClass.VEHICLE, 20, -1.75)])
    assert decision.reason == "gap" and controller.commands[0].plan is None


def test_mid_change_gap_loss_keeps_path_and_stops():
    world = lane_world()
    controller = LaneChangeSupervisor()
    sensed(controller, world, 0, [(DetClass.OBSTACLE, 70, 1.75)])
    points, legs = controller.commands[0].plan
    world.revise_route(0, Route(points, legs))
    controller.activate(0)
    controller.installed(world, 0)
    place(world, 0, 30, 3)
    original = world.slots[0].route.copy()
    decision = sensed(controller, world, 0.1, [(DetClass.VEHICLE, 32, -1.75)])
    assert decision.state == "changing" and decision.reason == "gap"
    assert controller.commands[0].speed_cap == 0 and controller.commands[0].plan is None
    assert np.array_equal(original, world.slots[0].route)


def test_return_requires_opposite_side_camera_and_rear_clearance():
    world = lane_world()
    controller = LaneChangeSupervisor()
    sensed(controller, world, 0, [(DetClass.OBSTACLE, 70, 1.75)])
    points, legs = controller.commands[0].plan
    world.revise_route(0, Route(points, legs))
    controller.activate(0)
    controller.installed(world, 0)
    place(world, 0, 84, 3)
    decision = sensed(controller, world, 0.1, [], ages_override={"left": 1.0})
    assert decision.reason == "visibility" and decision.direction == 1
    assert controller.demand(0) == ("rear", "left")
    decision = sensed(controller, world, 0.2, [(DetClass.VEHICLE, 84, 1.75)])
    assert decision.reason == "gap" and controller.commands[0].speed_cap == 0


def test_external_route_generation_discards_incomplete_maneuver():
    world = lane_world()
    controller = LaneChangeSupervisor()
    sensed(controller, world, 0, [(DetClass.OBSTACLE, 70, 1.75)])
    points, legs = controller.commands[0].plan
    world.revise_route(0, Route(points, legs))
    controller.activate(0)
    controller.installed(world, 0)
    base, base_legs = world.map_index.lane_route([0, 1])
    world.install_route(0, Route(np.array(base, dtype=np.float32), tuple(base_legs)))
    place(world, 0, 20)
    front, surround, ages, poses = camera_views(world, [])
    controller.evaluate(world, 0.1, front, surround, ages, poses, {0: visibility()})
    assert 0 not in controller.decisions and not controller._states[0].active


def test_completion_count_survives_revision_but_not_new_episode():
    world = lane_world()
    controller = LaneChangeSupervisor()
    sensed(controller, world, 0, [(DetClass.OBSTACLE, 70, 1.75)])
    points, legs = controller.commands[0].plan
    world.revise_route(0, Route(points, legs))
    controller.activate(0)
    controller.installed(world, 0)
    place(world, 0, controller._states[0].end + 5, 3)
    world.slots[0].steps = 100
    assert sensed(controller, world, 0.1, []).completed == 1
    base, base_legs = world.map_index.lane_route([0, 1])
    route = Route(np.array(base, dtype=np.float32), tuple(base_legs))
    world.revise_route(0, route)
    place(world, 0, 20)
    assert sensed(controller, world, 0.2, [(DetClass.OBSTACLE, 70, 1.75)]).completed == 1
    world.install_route(0, route)
    place(world, 0, 20)
    assert sensed(controller, world, 0.3, [(DetClass.OBSTACLE, 70, 1.75)]).completed == 0


def test_new_obstacle_in_original_lane_stops_ongoing_change():
    world = lane_world()
    controller = LaneChangeSupervisor()
    sensed(controller, world, 0, [(DetClass.OBSTACLE, 70, 1.75)])
    points, legs = controller.commands[0].plan
    world.revise_route(0, Route(points, legs))
    controller.activate(0)
    controller.installed(world, 0)
    place(world, 0, 21, 4)
    sensed(controller, world, 0.1, [(DetClass.OBSTACLE, 24, 1.75)])
    assert controller.commands[0].speed_cap == 0


def test_braking_front_vehicle_changes_prediction():
    world = lane_world()
    controller = LaneChangeSupervisor()
    sensed(controller, world, 0, [(DetClass.OBSTACLE, 70, 1.75), (DetClass.VEHICLE, 45, -1.75)])
    sensed(controller, world, 0.1, [(DetClass.OBSTACLE, 70, 1.75), (DetClass.VEHICLE, 45.8, -1.75)])
    assert controller.commands[0].plan is not None
    for step in range(2, 12):
        decision = sensed(controller, world, step * 0.1, [(DetClass.OBSTACLE, 70, 1.75), (DetClass.VEHICLE, 45.8, -1.75)])
    assert decision.reason == "gap" and controller.commands[0].plan is None


def test_simultaneous_opposite_changes_wait_without_timeout():
    world = lane_world()
    points = world.slots[0].route.copy()
    points[:, 1] = -1.75
    world.install_route(1, Route(points, world.slots[0].route_legs))
    world.fleet.active[1] = True
    place(world, 1, 20, 4)
    world.add_obstacle(70, 1.75, 0.9)
    world.add_obstacle(80, -1.75, 0.9)
    controller = LaneChangeSupervisor()
    for now in (0, 30):
        controller.evaluate(world, now, {}, {}, {}, {}, {}, frozenset({0, 1}))
        assert controller.decisions[0].reason == "gap"
        assert controller.decisions[1].reason == "gap"
        assert controller.commands[0].plan is None and controller.commands[1].plan is None


def test_plain_policy_preview_does_not_take_over():
    env = SimulationEnv(lane_map(), SimParams(vehicle_count=1, pedestrian_count=0, safety_assist=False), seed=135, perception_mode="oracle")
    points, legs = env.map_index.lane_route([0, 1])
    env.world.install_route(0, Route(np.asarray(points, dtype=np.float32), tuple(legs)))
    env.world.fleet.active[0] = True
    place(env.world, 0, 20, 4)
    env.world.add_obstacle(70, 1.75, 0.9)
    env._obs_stale = True
    serial = env.world.route_serial[0]
    proposed = False
    for _ in range(20):
        result = env.step(np.zeros((config.MAX_VEHICLES, 2), dtype=np.float32))
        command = env.lane_changes.commands.get(0)
        proposed |= command is not None and command.plan is not None
        assert result.learn[0] and not result.assisted[0]
    assert proposed
    assert env.world.route_serial[0] == serial


def test_active_change_keeps_needed_cameras_fresh_under_competing_demand(monkeypatch):
    env = SimulationEnv(lane_map(), SimParams(vehicle_count=1, pedestrian_count=0), seed=135, compute_observations=False)
    points, legs = env.map_index.lane_route([0, 1])
    env.world.install_route(0, Route(np.asarray(points, dtype=np.float32), tuple(legs)))
    env.world.fleet.active[:8] = True
    place(env.world, 0, 20, 4)
    sensed(env.lane_changes, env.world, 0, [(DetClass.OBSTACLE, 70, 1.75)])
    env.lane_changes.activate(0)
    monkeypatch.setattr(env.intersections, "demand", lambda slot: ("rear", "left", "right"))
    for key in ("rear", "left", "right"):
        env._surround_taken.setdefault(0, {})[key] = (0.0, int(env.world.route_serial[0]))
    last = {"rear": 0.0, "right": 0.0}
    for step in range(1, 21):
        env.world.sim_time = step * config.DT
        camera, slots = env._surround_schedule(np.arange(8))
        assert len(slots) <= config.SURROUND_CNN_IMAGES_PER_STEP
        for slot in slots:
            env._surround_taken.setdefault(slot, {})[camera.key] = (env.world.sim_time, int(env.world.route_serial[slot]))
        if 0 in slots and camera.key in last:
            last[camera.key] = env.world.sim_time
        assert all(env.world.sim_time - at <= 0.25 + 1e-6 for at in last.values())


@pytest.mark.parametrize("allowed", [None, False])
def test_unknown_or_prohibited_regulation_never_starts_old_cutback(allowed):
    env = SimulationEnv(lane_map(allowed), SimParams(vehicle_count=1, pedestrian_count=0), seed=135, perception_mode="oracle")
    points, legs = env.map_index.lane_route([0, 1])
    env.world.install_route(0, Route(np.asarray(points, dtype=np.float32), tuple(legs)))
    env.world.fleet.active[0] = True
    place(env.world, 0, 20, 4)
    env.world.add_obstacle(70, 1.75, 0.9)
    env.commandeered_slot = 0
    env._obs_stale = True
    serial = int(env.world.route_serial[0])
    for _ in range(500):
        env.step(np.zeros((config.MAX_VEHICLES, 2), dtype=np.float32))
        assert not env.world.collided_flags[0] and env.world.fleet.gear[0] >= 0
        assert env.safety.assist(0) not in ("reverse_check", "reversing", "detour")
    assert env.world.fleet.speed[0] < 0.01
    assert env.world.route_serial[0] == serial
    assert env.lane_changes.decisions[0].reason == ("unknown_regulation" if allowed is None else "regulation")


@pytest.mark.parametrize("begin,end", [(29.0, 30.0), (31.0, 32.0), (-2.0, -1.0), (100.0, 0.0)])
def test_lane_frame_short_window_keeps_segment_heading(begin, end):
    state = _LaneState(0, points=np.array([(5.0, 0.0), (5.0, 10.0), (5.0, 20.0), (5.0, 30.0)]), cum=np.array([0.0, 10.0, 20.0, 30.0]),
                       corridor=LaneCorridor(0, 0, 3.5, -3.5, begin, end, True, ""))
    arc, lateral, heading = LaneChangeSupervisor._frame(state, 7.0, 31.0)
    assert (arc, lateral, heading) == pytest.approx((10.0 if begin < 0.0 else 30.0, -2.0, math.pi / 2))


@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("active", [False, True])
def test_lane_change_without_route_segment_never_proposes_path(count, active):
    world = lane_world()
    corridor = world.map_index.lane_corridor(world.slots[0].route, world.slots[0].route_legs, 20.0)
    state = _LaneState(int(world.route_serial[0]), corridor=corridor,
                       points=world.slots[0].route[:count], cum=world.slots[0].route_cum[:count],
                       active=active, completed=2)
    controller = LaneChangeSupervisor()
    controller._evaluate_slot(world, 0, state, [], 0.0, False, None, {}, {}, None)
    decision, command = controller.decisions[0], controller.commands[0]
    assert decision.state == "waiting" and decision.reason == "visibility"
    assert decision.completed == 2 and decision.source == "camera"
    assert command.speed_cap == 0.0 and command.plan is None and not command.driving
    assert state.active == active
