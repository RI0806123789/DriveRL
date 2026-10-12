"""閉鎖の非同期迂回・古い結果の破棄と、経路変更後の物理・採点・配車を検査する。"""

from __future__ import annotations

import math
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from app.contracts import Bounds, MapBuilding, MapData, MapEdge, MapNode, TaxiStatus
from app.map.index import build_map_index
from app.runtime.taxi import TaxiService
from app.sim import road_closure
from app.sim.road_closure import RoadClosureSupervisor
from app.sim.world import Route, World


class ControlledSearch:
    """実時間を待たず、検索の完了と取消不能な古い結果を順に再現する。"""

    def __init__(self) -> None:
        self.jobs: list[tuple[Future, object, tuple, dict]] = []

    def submit(self, fn, *args, **kwargs) -> Future:
        future = Future()
        future.set_running_or_notify_cancel()
        self.jobs.append((future, fn, args, kwargs))
        return future

    def finish(self, index: int = -1) -> Route | None:
        future, fn, args, kwargs = self.jobs[index]
        result = fn(*args, **kwargs)
        future.set_result(result)
        return result


def network(*, buildings: list[MapBuilding] | None = None):
    coordinates = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (200.0, 0.0)]
    nodes = [MapNode(i, x, y) for i, (x, y) in enumerate(coordinates)]
    edges = [
        MapEdge(i, u, v, 2, 6.5, False, 11.1, [coordinates[u], coordinates[v]], math.dist(coordinates[u], coordinates[v]))
        for i, (u, v) in enumerate([(0, 1), (1, 3), (1, 2), (2, 3)])
    ]
    return build_map_index(MapData(
        "detour", "閉鎖と迂回の合成道路", 0.0, 0.0, 220.0, Bounds(-15, -15, 215, 115),
        nodes, edges, buildings or [],
    ))


def world_on_route(*, buildings: list[MapBuilding] | None = None) -> World:
    world = World(network(buildings=buildings), np.random.default_rng(135))
    points, legs = world.map_index.lane_route([0, 1, 3])
    assert world.install_route(0, Route(np.asarray(points, dtype=np.float32), tuple(legs)))
    world.fleet.active[0] = True
    world.fleet.x[0] = 20.0
    world.arc[0] = 20.0
    world.slots[0].arc_position = 20.0
    world.slots[0].route_progress = 10
    return world


@pytest.fixture
def searches(monkeypatch) -> ControlledSearch:
    executor = ControlledSearch()
    monkeypatch.setattr(road_closure, "_SEARCH", executor)
    return executor


def close_ahead(world: World) -> None:
    world.map_index.set_closed_edges(frozenset((1,)))


def test_pending_search_holds_pose_and_installs_connected_detour_once(searches) -> None:
    world = world_on_route()
    close_ahead(world)
    supervisor = RoadClosureSupervisor()
    pose = (float(world.fleet.x[0]), float(world.fleet.y[0]), float(world.fleet.heading[0]))
    serial = int(world.route_serial[0])
    assert supervisor.evaluate(world, frozenset((0,))) == ()
    assert len(searches.jobs) == 1
    assert supervisor.caps[0] == 0.0
    assert supervisor.decisions[0].source == "road_event"
    assert not supervisor.decisions[0].imposed
    assert supervisor.evaluate(world, frozenset((0,))) == ()
    assert len(searches.jobs) == 1 and supervisor.caps[0] == 0.0
    detour = searches.finish()
    assert detour is not None
    assert all(leg.edge_id not in world.map_index.closed_edges for leg in detour.legs)
    assert any(leg.edge_id == 2 for leg in detour.legs)
    assert supervisor.evaluate(world, frozenset((0,))) == (0,)
    assert int(world.route_serial[0]) == serial + 1
    assert tuple(float(getattr(world.fleet, name)[0]) for name in ("x", "y", "heading")) == pose
    assert supervisor.decisions[0].state == "rerouting" and supervisor.decisions[0].imposed
    assert supervisor.evaluate(world, frozenset((0,))) == ()
    assert not supervisor.decisions and not supervisor.caps
    assert len(searches.jobs) == 1


def test_reopening_discards_running_result_and_clears_waiting(searches) -> None:
    world = world_on_route()
    close_ahead(world)
    supervisor = RoadClosureSupervisor()
    supervisor.evaluate(world, frozenset((0,)))
    original, serial = world.slots[0].route.copy(), int(world.route_serial[0])
    assert searches.finish() is not None
    world.map_index.set_closed_edges(frozenset())
    assert supervisor.evaluate(world, frozenset((0,))) == ()
    assert not supervisor.caps and not supervisor.decisions
    np.testing.assert_array_equal(world.slots[0].route, original)
    assert int(world.route_serial[0]) == serial
    close_ahead(world)
    supervisor.evaluate(world, frozenset((0,)))
    assert len(searches.jobs) == 2


def test_changed_closure_generation_rejects_old_finished_route(searches) -> None:
    world = world_on_route()
    close_ahead(world)
    supervisor = RoadClosureSupervisor()
    supervisor.evaluate(world, frozenset((0,)))
    assert searches.finish() is not None
    world.map_index.set_closed_edges(frozenset((1, 2)))
    serial = int(world.route_serial[0])
    assert supervisor.evaluate(world, frozenset((0,))) == ()
    assert int(world.route_serial[0]) == serial
    assert len(searches.jobs) == 2
    assert searches.finish() is None
    supervisor.evaluate(world, frozenset((0,)))
    assert supervisor.decisions[0].state == "blocked" and supervisor.decisions[0].reason == "no_detour"
    supervisor.evaluate(world, frozenset((0,)))
    assert len(searches.jobs) == 2


def test_route_generation_change_rejects_old_result(searches) -> None:
    world = world_on_route()
    close_ahead(world)
    supervisor = RoadClosureSupervisor()
    supervisor.evaluate(world, frozenset((0,)))
    assert searches.finish() is not None
    current = world.slots[0]
    assert world.revise_route(0, Route(current.route.copy(), current.route_legs))
    serial = int(world.route_serial[0])
    assert supervisor.evaluate(world, frozenset((0,))) == ()
    assert int(world.route_serial[0]) == serial
    assert len(searches.jobs) == 2


def test_new_map_rejects_result_even_when_serial_and_closure_generation_match(searches) -> None:
    old_world, new_world = world_on_route(), world_on_route()
    close_ahead(old_world)
    close_ahead(new_world)
    assert old_world.route_serial[0] == new_world.route_serial[0]
    assert old_world.map_index.closure_generation == new_world.map_index.closure_generation
    supervisor = RoadClosureSupervisor()
    supervisor.evaluate(old_world, frozenset((0,)))
    assert searches.finish() is not None
    serial = int(new_world.route_serial[0])
    original = new_world.slots[0].route.copy()
    assert supervisor.evaluate(new_world, frozenset((0,))) == ()
    assert int(new_world.route_serial[0]) == serial
    np.testing.assert_array_equal(new_world.slots[0].route, original)
    assert len(searches.jobs) == 2


def test_new_map_does_not_keep_failed_search_from_previous_map(searches) -> None:
    old_world, new_world = world_on_route(), world_on_route()
    old_world.map_index.set_closed_edges(frozenset((1, 2)))
    close_ahead(new_world)
    supervisor = RoadClosureSupervisor()
    supervisor.evaluate(old_world, frozenset((0,)))
    assert searches.finish() is None
    supervisor.evaluate(old_world, frozenset((0,)))
    assert supervisor.decisions[0].reason == "no_detour"
    assert supervisor.evaluate(new_world, frozenset((0,))) == ()
    assert len(searches.jobs) == 2
    assert supervisor.decisions[0].reason == "blocked"
    assert searches.finish() is not None
    assert supervisor.evaluate(new_world, frozenset((0,))) == (0,)


def test_moving_from_search_origin_discards_route_and_retries_from_current_pose(searches) -> None:
    world = world_on_route()
    close_ahead(world)
    supervisor = RoadClosureSupervisor()
    supervisor.evaluate(world, frozenset((0,)))
    assert searches.finish() is not None
    world.fleet.x[0], world.arc[0] = 22.0, 22.0
    serial = int(world.route_serial[0])
    assert supervisor.evaluate(world, frozenset((0,))) == ()
    assert int(world.route_serial[0]) == serial and len(searches.jobs) == 2
    assert supervisor.caps[0] == 0.0 and supervisor.decisions[0].reason == "blocked"
    route = searches.finish()
    assert route is not None and route.points[0, 0] == pytest.approx(22.0)
    assert supervisor.evaluate(world, frozenset((0,))) == (0,)


@pytest.mark.parametrize("clearance", ["center", "body"])
def test_detour_crossing_building_or_sweeping_body_is_rejected(clearance, searches, monkeypatch) -> None:
    outline = [(130, 30), (145, 30), (145, 45), (130, 45)] if clearance == "center" else [(120, 70.5), (150, 70.5), (150, 73), (120, 73)]
    world = world_on_route(buildings=[MapBuilding(0, 8.0, outline)])
    close_ahead(world)
    good = world.route_between((20.0, float(world.fleet.y[0])), world.slots[0].goal, heading=0.0)
    assert good is not None
    points = [(20, float(world.fleet.y[0])), (140, 40), (200, 1.625)] if clearance == "center" else [(20, float(world.fleet.y[0])), (100, 70), (160, 70), (200, 1.625)]
    candidate = Route(np.asarray(points, dtype=np.float32), good.legs)
    if clearance == "body":
        assert not np.any(world.map_index.occupancy.sample_building(candidate.points[:, 0], candidate.points[:, 1]))
    monkeypatch.setattr(world, "route_between", lambda *args, **kwargs: candidate)
    supervisor = RoadClosureSupervisor()
    serial, original = int(world.route_serial[0]), world.slots[0].route.copy()
    supervisor.evaluate(world, frozenset((0,)))
    assert searches.finish() is None
    assert supervisor.evaluate(world, frozenset((0,))) == ()
    assert supervisor.decisions[0].state == "blocked" and supervisor.decisions[0].reason == "no_detour"
    assert int(world.route_serial[0]) == serial
    np.testing.assert_array_equal(world.slots[0].route, original)


def test_disabling_vehicle_or_respawning_discards_pending_search(searches) -> None:
    world = world_on_route()
    close_ahead(world)
    supervisor = RoadClosureSupervisor()
    supervisor.evaluate(world, frozenset((0,)))
    serial = int(world.route_serial[0])
    assert searches.finish() is not None
    assert supervisor.evaluate(world, frozenset()) == ()
    assert int(world.route_serial[0]) == serial and not supervisor.decisions
    supervisor.evaluate(world, frozenset((0,)))
    world.fleet.active[0] = False
    searches.finish()
    assert supervisor.evaluate(world, frozenset((0,))) == ()
    assert int(world.route_serial[0]) == serial and not supervisor.decisions


@pytest.mark.parametrize("stop_kind", ["goal", "intermediate", "none"])
def test_revise_route_preserves_physics_scores_and_stop_target(stop_kind) -> None:
    world = world_on_route()
    state = world.slots[0]
    values = {"steps": 57, "violations": 2, "lane_departures": 3, "outside_lane": True,
              "speed_violations": 4, "over_speed": True, "collided": True, "reached_goal": False}
    for name, value in values.items():
        setattr(state, name, value)
    world.collided_flags[0] = True
    world.fleet.speed[0], world.fleet.steer[0], world.fleet.lateral_speed[0], world.fleet.yaw_rate[0] = 2.0, 0.1, 0.2, -0.3
    names = ("x", "y", "heading", "speed", "steer", "gear", "lateral_speed", "yaw_rate", "active")
    before = {name: getattr(world.fleet, name)[0].item() for name in names}
    serial = int(world.route_serial[0])
    if stop_kind != "none":
        world.set_stop_target(0, float(world.route_total[0]) if stop_kind == "goal" else 50.0)
    old_stop_point = tuple(float(np.interp(world.stop_arc[0], state.route_cum, state.route[:, i])) for i in (0, 1))
    close_ahead(world)
    revised = world.route_between((float(world.fleet.x[0]), float(world.fleet.y[0])), state.goal, heading=float(world.fleet.heading[0]))
    assert revised is not None and world.revise_route(0, revised)
    assert {name: getattr(world.fleet, name)[0].item() for name in names} == before
    assert {name: getattr(state, name) for name in values} == values
    assert world.collided_flags[0] and not world.reached_flags[0]
    assert int(world.route_serial[0]) == serial + 1 and state.route_dirty
    assert state.route_legs == revised.legs
    if stop_kind == "goal":
        assert world.stop_arc[0] == pytest.approx(world.route_total[0])
    elif stop_kind == "intermediate":
        new_stop_point = tuple(float(np.interp(world.stop_arc[0], state.route_cum, state.route[:, i])) for i in (0, 1))
        assert new_stop_point == pytest.approx(old_stop_point, abs=0.01)
    else:
        assert math.isinf(world.stop_arc[0])


def test_taxi_route_revision_tracks_revised_world_once_without_changing_ride() -> None:
    world = world_on_route()
    service = TaxiService()
    service.status = TaxiStatus(
        phase="riding", ride_id="synthetic-ride", vehicle_id=0, pickup=(0, 0), dropoff=(200, 0),
        route=[tuple(float(v) for v in point) for point in world.slots[0].route], route_revision=7,
    )
    service._route_serial = int(world.route_serial[0])
    env = SimpleNamespace(world=world)
    close_ahead(world)
    revised = world.route_between((20.0, float(world.fleet.y[0])), world.slots[0].goal, heading=0.0)
    assert revised is not None and world.revise_route(0, revised)
    service.update(env)
    assert service.status.route_revision == 8
    assert service.status.route == [tuple(float(v) for v in point) for point in world.slots[0].route]
    assert service.status.ride_id == "synthetic-ride" and service.status.phase == "riding"
    service.update(env)
    assert service.status.route_revision == 8
    assert service.status.to_wire(include_route=True)["routeRevision"] == 8
