"""交差点の静的な候補進路、交錯区間、経路上の位置を検査する。"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from shapely.geometry import LineString

from app import config
from app.contracts import Bounds, MapData, MapEdge, MapNode, MapSign, MapSignal
from app.map.index import build_map_index
from app.map.intersections import CONFLICT_MARGIN_M, _overlap_span
from app.sim.intersection import IntersectionSupervisor
from app.sim.world import Route, World


def cross_map(*, signals: bool = False) -> MapData:
    nodes = [MapNode(i, x, y) for i, (x, y) in enumerate([
        (0.0, 0.0), (-50.0, 0.0), (50.0, 0.0), (0.0, 50.0), (0.0, -50.0),
    ])]
    edges = [MapEdge(i, 0, n.id, 2, 7.0, False, 11.1, [(0.0, 0.0), (n.x, n.y)], 50.0)
             for i, n in enumerate(nodes[1:])]
    lights = [MapSignal(0, 0, -10.0, 5.0, 0.0, 0, 7.0, 0, 0)] if signals else []
    return MapData("cross", "cross", 0.0, 0.0, 60.0, Bounds(-60.0, -60.0, 60.0, 60.0),
                   nodes, edges, [], signals=lights)


@pytest.mark.parametrize("signals", [False, True])
def test_right_turn_includes_oncoming_straight_and_left_turn(signals: bool) -> None:
    index = build_map_index(cross_map(signals=signals))
    points, legs = index.lane_route([1, 0, 4])
    junctions = index.intersections_on_route(points, legs)
    assert len(junctions) == 1
    junction = junctions[0]
    assert junction.node_id == 0 and junction.turn == "right"
    assert junction.entry_arc == legs[0].end_arc
    assert junction.exit_arc == legs[1].start_arc
    assert junction.entry_heading == pytest.approx(0.0)
    assert junction.exit_heading == pytest.approx(-math.pi / 2)
    assert junction.road_width == junction.exit_road_width == 7.0
    routes = {(c.movement.entry_node, c.movement.exit_node): c for c in junction.conflicts}
    assert routes[2, 1].movement.turn == "straight"
    assert routes[2, 4].movement.turn == "left"
    for conflict in junction.conflicts:
        movement = conflict.movement
        assert movement.entry_node != 1
        assert movement.entry_node != movement.exit_node
        assert 0 <= movement.entry_arc <= movement.exit_arc <= movement.cum[-1]
        assert len(movement.points) == len(movement.cum)
        assert np.diff(movement.cum).min() > 0
        assert 0 <= conflict.ego_start_arc <= conflict.ego_end_arc <= LineString(points).length
        assert 0 <= conflict.other_start_arc <= conflict.other_end_arc <= movement.cum[-1]


def test_straight_opposing_lanes_do_not_conflict() -> None:
    index = build_map_index(cross_map())
    points, legs = index.lane_route([1, 0, 2])
    junction = index.intersections_on_route(points, legs)[0]
    assert junction.turn == "straight"
    assert not any(c.movement.entry_node == 2 and c.movement.exit_node == 1 for c in junction.conflicts)
    assert any(c.movement.entry_node == 3 and c.movement.exit_node == 4 for c in junction.conflicts)


def test_oneway_outgoing_arm_never_becomes_incoming_candidate() -> None:
    data = cross_map()
    data.edges[3] = replace(data.edges[3], oneway=True)
    index = build_map_index(data)
    points, legs = index.lane_route([1, 0, 4])
    junction = index.intersections_on_route(points, legs)[0]
    assert not any(c.movement.entry_node == 4 for c in junction.conflicts)
    assert any(c.movement.exit_node == 4 for c in junction.conflicts)


def test_turn_direction_and_incoming_width_follow_actual_edges() -> None:
    data = cross_map()
    data.edges[0] = replace(data.edges[0], width=14.0, lanes=4)
    index = build_map_index(data)
    points, legs = index.lane_route([1, 0, 3])
    junction = index.intersections_on_route(points, legs)[0]
    assert junction.turn == "left"
    assert junction.road_width == 14.0 and junction.exit_road_width == 7.0
    assert all(c.movement.road_width == 7.0 for c in junction.conflicts)


def test_route_endpoints_are_not_crossed_junctions() -> None:
    index = build_map_index(cross_map())
    for path in ([1, 0], [0, 2]):
        points, legs = index.lane_route(path)
        assert index.intersections_on_route(points, legs) == []
    assert index.intersections_on_route([], []) == []
    assert index._intersections._movements == {}


def test_repeated_junction_has_separate_route_arc_windows() -> None:
    index = build_map_index(cross_map())
    points, legs = index.lane_route([1, 0, 2, 0, 4])
    first, second = index.intersections_on_route(points, legs)
    assert first.node_id == second.node_id == 0
    assert first.entry_arc < first.exit_arc < second.entry_arc < second.exit_arc
    assert first.conflicts and second.conflicts
    assert max(c.ego_end_arc for c in first.conflicts) < min(c.ego_start_arc for c in second.conflicts)


def test_candidates_are_reused_without_rebuilding_lane_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    index = build_map_index(cross_map())
    points, legs = index.lane_route([1, 0, 4])
    first = index.intersections_on_route(points, legs)
    assert set(index._intersections._movements) == {0}

    def unexpected_build(*args, **kwargs):
        raise AssertionError("候補進路を再生成してはいけません")

    monkeypatch.setattr("app.map.intersections.build_lane_route", unexpected_build)
    second = index.intersections_on_route(points, legs)
    assert second == first
    assert second[0].conflicts[0].movement is first[0].conflicts[0].movement


def test_vehicle_width_margin_covers_near_miss_without_centerline_crossing() -> None:
    ego = LineString([(0.0, 0.0), (10.0, 0.0)])
    threshold = config.VEHICLE_WIDTH + CONFLICT_MARGIN_M
    close = LineString([(2.0, threshold - 0.05), (8.0, threshold - 0.05)])
    separated = LineString([(2.0, threshold + 0.05), (8.0, threshold + 0.05)])
    assert not ego.intersects(close)
    span = _overlap_span(ego, close.buffer(threshold))
    assert span is not None and span[0] < 2.0 and span[1] > 8.0
    assert _overlap_span(ego, separated.buffer(threshold)) is None


def test_only_visited_junctions_are_generated() -> None:
    data = cross_map()
    data.nodes.extend([MapNode(5, 100.0, 0.0), MapNode(6, 100.0, 50.0), MapNode(7, 100.0, -50.0)])
    for a, b in ((2, 5), (5, 6), (5, 7)):
        start, end = data.nodes[a], data.nodes[b]
        data.edges.append(MapEdge(len(data.edges), a, b, 2, 7.0, False, 11.1,
                                  [(start.x, start.y), (end.x, end.y)], 50.0))
    data.bounds = Bounds(-60.0, -60.0, 110.0, 60.0)
    index = build_map_index(data)
    points, legs = index.lane_route([1, 0, 2])
    index.intersections_on_route(points, legs)
    assert index.is_intersection(5)
    assert set(index._intersections._movements) == {0}


def test_arc_window_does_not_generate_distant_candidates() -> None:
    index = build_map_index(cross_map())
    points, legs = index.lane_route([1, 0, 2, 0, 4])
    assert index.intersections_on_route(points, legs, end_arc=30.0) == []
    assert index._intersections._movements == {}
    all_passages = index.intersections_on_route(points, legs)
    first, second = all_passages
    ahead = index.intersections_on_route(points, legs, start_arc=first.exit_arc + 1.0, end_arc=second.exit_arc + 1.0)
    assert ahead == [second]
    assert index.intersections_on_route(points, legs, start_arc=second.exit_arc + 1.0) == []


def test_arc_window_keeps_partly_passed_intersection() -> None:
    index = build_map_index(cross_map())
    points, legs = index.lane_route([1, 0, 4])
    junction = index.intersections_on_route(points, legs)[0]
    midpoint = (junction.entry_arc + junction.exit_arc) / 2.0
    assert index.intersections_on_route(points, legs, start_arc=midpoint, end_arc=midpoint) == [junction]
    assert index.intersections_on_route(points, legs, start_arc=midpoint, end_arc=midpoint - 1.0) == []


def routed_world(data: MapData, path: list[int]) -> World:
    index = build_map_index(data)
    world = World(index, np.random.default_rng(134))
    points, legs = index.lane_route(path)
    assert world.install_route(0, Route(np.asarray(points, dtype=np.float32), tuple(legs)))
    return world


def linked_junctions(separation: float = 50.0) -> MapData:
    coords = [(0.0, 0.0), (separation, 0.0), (-50.0, 0.0), (0.0, 50.0),
              (separation + 50.0, 0.0), (separation, 50.0)]
    nodes = [MapNode(i, x, y) for i, (x, y) in enumerate(coords)]
    edges = [MapEdge(i, a, b, 2, 7.0, False, 11.1, [coords[a], coords[b]], math.dist(coords[a], coords[b]))
             for i, (a, b) in enumerate([(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)])]
    return MapData("linked", "linked", 0.0, 0.0, separation + 60.0,
                   Bounds(-60.0, -60.0, separation + 60.0, 60.0), nodes, edges, [])


def test_world_window_refreshes_before_revisited_junction_and_resets_with_route() -> None:
    world = routed_world(cross_map(), [1, 0, 2, 0, 4])
    state = world.slots[0]
    passages = world.map_index.intersections_on_route(state.route, state.route_legs)
    first, second = passages
    assert state.intersections == (first,)
    world.arc[0] = 25.0
    world.prepare_intersections(0)
    assert state.intersections_until == 60.0
    world.arc[0] = 25.1
    world.prepare_intersections(0)
    assert state.intersections_until == pytest.approx(85.1)
    for arc in np.arange(26.0, second.exit_arc + config.VEHICLE_LENGTH, 1.0):
        world.arc[0] = arc
        world.prepare_intersections(0)
        nearest = next((p for p in passages if p.exit_arc + config.VEHICLE_LENGTH / 2 >= arc), None)
        cached = next((p for p in state.intersections if p.exit_arc + config.VEHICLE_LENGTH / 2 >= arc), None)
        if nearest is not None and nearest.entry_arc - arc <= 35.0:
            assert cached == nearest
    assert second.entry_arc != first.entry_arc and second.node_id == first.node_id
    serial = int(world.route_serial[0])
    points, legs = world.map_index.lane_route([3, 0, 4])
    assert world.install_route(0, Route(np.asarray(points, dtype=np.float32), tuple(legs)))
    assert int(world.route_serial[0]) == serial + 1
    assert state.intersections_until == 60.0
    assert state.intersections == tuple(world.map_index.intersections_on_route(state.route, state.route_legs))
    assert state.intersections[0].entry_heading != first.entry_heading


def test_world_window_restores_passed_junction_after_reversing() -> None:
    world = routed_world(cross_map(), [1, 0, 2, 0, 4])
    state = world.slots[0]
    first = state.intersections[0]
    world.arc[0] = first.exit_arc + config.VEHICLE_LENGTH / 2 + 12.0
    world.prepare_intersections(0)
    assert first not in state.intersections
    world.arc[0] = first.entry_arc - 5.0
    world.prepare_intersections(0)
    assert first in state.intersections
    refreshed_until = state.intersections_until
    world.arc[0] -= 1.0
    world.prepare_intersections(0)
    assert state.intersections_until == refreshed_until
    assert first in state.intersections


@pytest.mark.parametrize("sign_edge, expected", [(0, [True, False]), (1, [False, True])])
def test_background_stop_matches_revisited_junction_approach(sign_edge: int, expected: list[bool]) -> None:
    data = cross_map()
    forward = sign_edge == 0
    data.signs = [MapSign(0, 0, sign_edge, -3.0 if forward else 3.0, 4.0 if forward else -4.0,
                          0.0 if forward else math.pi, kind="stop")]
    world = routed_world(data, [1, 0, 2, 0, 4])
    state = world.slots[0]
    passages = world.map_index.intersections_on_route(state.route, state.route_legs)
    controller = IntersectionSupervisor()
    assert [controller._background_stop(world, 0, p) for p in passages] == expected


@pytest.mark.parametrize("path, expected", [([2, 0, 1, 4], [False, True]), ([4, 1, 0, 2], [False, False])])
def test_background_stop_does_not_apply_at_opposite_end_of_same_edge(path: list[int], expected: list[bool]) -> None:
    data = linked_junctions()
    data.signs = [MapSign(0, 1, 0, 47.0, 4.0, 0.0, kind="stop")]
    world = routed_world(data, path)
    state = world.slots[0]
    passages = world.map_index.intersections_on_route(state.route, state.route_legs)
    assert len(passages) == 2
    controller = IntersectionSupervisor()
    assert [controller._background_stop(world, 0, p) for p in passages] == expected


@pytest.mark.parametrize("separation", [6.0, 10.0, 12.0])
def test_background_intention_includes_next_close_junction_before_previous_is_cleared(separation: float) -> None:
    data = linked_junctions(separation)
    data.nodes[3] = replace(data.nodes[3], y=-50.0)
    data.edges[2] = replace(data.edges[2], polyline=[(0.0, 0.0), (0.0, -50.0)])
    world = routed_world(data, [2, 0, 3])
    points, legs = world.map_index.lane_route([4, 1, 0, 2])
    assert world.install_route(1, Route(np.asarray(points, dtype=np.float32), tuple(legs)))
    ego = world.slots[0].intersections[0]
    previous, following = world.slots[1].intersections
    assert previous.node_id == 1 and following.node_id == ego.node_id == 0
    other_arc = previous.exit_arc + 1.5
    assert other_arc < previous.exit_arc + config.VEHICLE_LENGTH / 2
    assert following.entry_arc - other_arc < config.VEHICLE_LENGTH / 2
    for slot, arc, speed, heading in [(0, ego.entry_arc - 4.0, 0.0, 0.0), (1, other_arc, 3.0, math.pi)]:
        state = world.slots[slot]
        world.arc[slot] = arc
        world.fleet.x[slot] = np.interp(arc, state.route_cum, state.route[:, 0])
        world.fleet.y[slot] = np.interp(arc, state.route_cum, state.route[:, 1])
        world.fleet.speed[slot] = speed
        world.fleet.heading[slot] = heading
        world.fleet.active[slot] = True
    controller = IntersectionSupervisor()
    controller.evaluate(world, 0.0, {}, {}, {}, {}, {}, frozenset({0, 1}))
    decision = controller.decisions[0]
    assert decision.source == "background" and decision.reason == "oncoming"
    assert math.isfinite(decision.speed_cap)
