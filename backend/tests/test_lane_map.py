"""同方向車線の変更範囲、OSM規制、全道路閉鎖の探索と期間を検査する。"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from app import config
from app.contracts import Bounds, MapData, MapEdge, MapNode, MapPreset, RoadLead, SimParams
from app.map import build_map_index
from app.map.loader import (
    CACHE_VERSION,
    _from_cache_dict,
    _parse_lane_change,
    _read_cache,
    _to_cache_dict,
)
from app.sim.env import SimulationEnv
from app.sim.scenario import RoadEvent, Scenario
from tune_hyperparams import synthetic_grid


def straight(*, oneway=True, allowed=True, lanes=None, width=None, reverse_alias=False):
    count = (2 if oneway else 4) if lanes is None else lanes
    width = count * 3.25 if width is None else width
    edge = MapEdge(0, 0, 1, count, width, oneway, 11.1, [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)], 100.0, allowed)
    edges = [edge]
    if reverse_alias:
        edges.append(replace(edge, id=1, u=1, v=0, polyline=edge.polyline[::-1]))
    return build_map_index(MapData(
        "straight", "合成直線", 0, 0, 100, Bounds(-10, -15, 110, 15),
        [MapNode(0, 0, 0), MapNode(1, 100, 0)], edges, [],
    ))


@pytest.mark.parametrize("oneway,reverse", [(True, False), (False, False), (False, True)])
def test_neighbor_stays_in_same_direction(oneway, reverse):
    index = straight(oneway=oneway)
    points, legs = index.lane_route([1, 0] if reverse else [0, 1])
    corridor = index.lane_corridor(points, legs, 40)
    assert corridor is not None and corridor.allowed
    assert corridor.lane_index == 0
    assert corridor.neighbor_offset == pytest.approx(-3.25)
    assert corridor.begin_arc == pytest.approx(0)
    assert corridor.end_arc == pytest.approx(100)
    direction = np.array([-1.0, 0.0] if reverse else [1.0, 0.0])
    left = np.array([-direction[1], direction[0]])
    neighbor = np.asarray(points[20]) + left * corridor.neighbor_offset
    lateral = float(neighbor @ left)
    assert lateral == pytest.approx(-1.625 if oneway else 1.625)


def test_current_right_lane_can_return_left_without_crossing_center():
    index = straight(oneway=False)
    lead = RoadLead(1, 0, [0], [[(0, 0), (50, 0), (100, 0)]], 100, entry_offset=1.625)
    points, legs = index.lane_route([1], lead=lead)
    corridor = index.lane_corridor(points, legs, 20)
    assert corridor is not None
    assert corridor.lane_index == 1 and corridor.neighbor_offset == pytest.approx(3.25)
    assert index.lane_corridor(points, legs, 75) is None


@pytest.mark.parametrize("allowed,reason", [(None, "unknown_regulation"), (False, "regulation")])
def test_missing_or_prohibited_regulation_never_allows_change(allowed, reason):
    index = straight(allowed=allowed)
    points, legs = index.lane_route([0, 1])
    corridor = index.lane_corridor(points, legs, 40)
    assert corridor is not None and not corridor.allowed and corridor.reason == reason


@pytest.mark.parametrize("oneway,lanes,width", [(False, 2, 6.5), (False, 5, 16.25), (True, 3, 9.75), (True, 2, 4)])
def test_opposing_shared_extra_or_narrow_lanes_are_out_of_scope(oneway, lanes, width):
    index = straight(oneway=oneway, lanes=lanes, width=width)
    points, legs = index.lane_route([0, 1])
    assert index.lane_corridor(points, legs, 40) is None


def test_curved_road_and_transitioning_route_are_not_straight_corridors():
    index = straight()
    index.data.edges[0].polyline[1] = (50, 2)
    points, legs = index.lane_route([0, 1])
    assert index.lane_corridor(points, legs, 40) is None
    index = straight()
    points, legs = index.lane_route([0, 1])
    shifted = [(x, y + 0.05 * x) for x, y in points]
    assert index.lane_corridor(shifted, legs, 40) is None
    assert index.lane_corridor(points, legs, 40) is not None


def test_corridor_cache_does_not_share_across_new_route_and_reads_current_closure():
    index = straight()
    points, legs = index.lane_route([0, 1])
    first = index.lane_corridor(points, legs, 40)
    assert index.lane_corridor(points, legs, 40) is first
    index.set_closed_edges(frozenset((0,)))
    closed = index.lane_corridor(points, legs, 40)
    assert closed is not None and not closed.allowed and closed.reason == "blocked"
    index.set_closed_edges(frozenset())
    assert index.lane_corridor(points, legs, 40) is first
    shifted = [(x, y - 3.25) for x, y in points]
    other = index.lane_corridor(shifted, legs, 40)
    assert other is not None and other.lane_index == 1


@pytest.mark.parametrize("tags,oneway,lanes,expected", [
    ({}, True, 2, None),
    ({"change:lanes": "yes|yes"}, True, 2, True),
    ({"change:lanes": "not_left|not_right"}, True, 2, True),
    ({"change:lanes": "not_right|not_left"}, True, 2, False),
    ({"change:lanes": "no|no"}, True, 2, False),
    ({"change:lanes": "yes|unknown"}, True, 2, None),
    ({"change:lanes": "yes|yes", "change:lanes:forward": "unknown"}, True, 2, None),
    ({"change:lanes": ["yes|yes", "no|no"]}, True, 2, False),
    ({"change:lanes": "yes|yes"}, False, 4, None),
    ({"change:lanes:forward": "yes|yes", "change:lanes:backward": "yes|yes"}, False, 4, True),
    ({"change:lanes:forward": "yes|yes", "change:lanes:backward": "no|no"}, False, 4, False),
    ({"change:lanes:forward": "yes|yes", "change:lanes:backward": "no|no", "reversed": True}, True, 2, False),
    ({"change:lanes": "yes|yes"}, False, 5, None),
])
def test_osm_permission_requires_known_bidirectional_lane_boundary(tags, oneway, lanes, expected):
    assert _parse_lane_change(tags, oneway, lanes) is expected


@pytest.mark.parametrize("allowed", [True, False, None])
def test_permission_cache_and_wire_roundtrip(allowed):
    data = synthetic_grid()
    data.edges[0].lane_change_allowed = allowed
    payload = _to_cache_dict(data)
    preset = MapPreset(data.preset_id, data.name, "", 0, 0, data.radius_m)
    restored = _from_cache_dict(payload, preset)
    assert payload["version"] == CACHE_VERSION == 14
    assert restored.edges[0].lane_change_allowed is allowed
    assert restored.to_wire()["edges"][0]["laneChangeAllowed"] is allowed


def test_version12_is_read_without_rewrite_and_keeps_permission_unknown(tmp_path):
    data = synthetic_grid()
    data.edges[0].lane_change_allowed = True
    payload = _to_cache_dict(data)
    payload["version"] = 12
    payload["signalsAtAllIntersections"] = True
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    before = path.read_bytes()
    preset = MapPreset(data.preset_id, data.name, "", 0, 0, data.radius_m)
    restored = _read_cache(path, preset)
    assert restored is not None and restored.edges[0].lane_change_allowed is None
    assert path.read_bytes() == before
    payload["version"] = 11
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _read_cache(path, preset) is None


def test_invalid_cache_permission_is_not_truthy_string():
    data = synthetic_grid()
    payload = _to_cache_dict(data)
    payload["edges"][0]["laneChangeAllowed"] = "false"
    preset = MapPreset(data.preset_id, data.name, "", 0, 0, data.radius_m)
    with pytest.raises(ValueError, match="車線変更規制"):
        _from_cache_dict(payload, preset)


def test_closure_blocks_both_directions_and_reverse_edge_aliases():
    index = straight(oneway=False, reverse_alias=True)
    assert index.shortest_path(0, 1) == [0, 1]
    assert index.set_closed_edges(frozenset((0,)))
    assert index.closed_edges == frozenset((0, 1))
    assert index.closure_generation == 1
    assert index.shortest_path(0, 1, arrive_heading=0) is None
    assert index.shortest_path(1, 0) is None
    assert not index.set_closed_edges(frozenset((1,)))
    assert index.closure_generation == 1
    assert index.set_closed_edges(frozenset())
    assert index.shortest_path(1, 0) == [1, 0]
    assert index.closure_generation == 2


def test_closure_selects_real_connected_detour_and_keeps_geometric_graph():
    index = build_map_index(synthetic_grid())
    path = index.shortest_path(0, 2)
    first_edge = index.graph[path[0]][path[1]]["edge_id"]
    index.set_closed_edges(frozenset((first_edge,)))
    detour = index.shortest_path(0, 2, arrive_heading=0)
    assert detour is not None and len(detour) > len(path)
    assert all(index.graph[a][b]["edge_id"] not in index.closed_edges for a, b in zip(detour[:-1], detour[1:]))
    assert index.graph.has_edge(path[0], path[1])
    with pytest.raises(ValueError, match="道路ID"):
        index.set_closed_edges(frozenset((999999,)))


def test_overlapping_closure_periods_clear_only_after_last_event(monkeypatch):
    monkeypatch.delenv("DRIVERL_SCENARIO", raising=False)
    index = build_map_index(synthetic_grid())
    scenario = Scenario(events=(RoadEvent(0, "closure", 1, 2), RoadEvent(0, "closure", 2, 3)))
    env = SimulationEnv(index, SimParams(vehicle_count=1, pedestrian_count=0), compute_observations=False, scenario=scenario)
    for now, closed, generation in [(0.5, False, 0), (1, True, 1), (2, True, 1), (3, True, 1), (5, False, 2)]:
        env.world.sim_time = now
        env.traffic.prepare(env)
        assert bool(index.closed_edges) == closed
        assert index.closure_generation == generation
    assert not env.world.obstacles


@pytest.mark.parametrize("kind", ["construction", "parked"])
def test_partial_events_preserve_routing(kind, monkeypatch):
    monkeypatch.delenv("DRIVERL_SCENARIO", raising=False)
    index = build_map_index(synthetic_grid())
    before = index.shortest_path(0, 2)
    scenario = Scenario(events=(RoadEvent(0, kind, 0, 2),))
    env = SimulationEnv(index, SimParams(vehicle_count=1, pedestrian_count=0), compute_observations=False, scenario=scenario)
    assert env.world.obstacles
    assert not index.closed_edges and index.closure_generation == 0
    assert index.shortest_path(0, 2) == before


def test_weighted_background_waits_while_all_selected_roads_are_closed(monkeypatch):
    monkeypatch.delenv("DRIVERL_SCENARIO", raising=False)
    index = build_map_index(synthetic_grid())
    scenario = Scenario(background_vehicles=1, road_weights={0: 1}, events=(RoadEvent(0, "closure", 0, 2),))
    env = SimulationEnv(index, SimParams(vehicle_count=2, pedestrian_count=0), compute_observations=False, scenario=scenario)
    assert not env.world.fleet.active[1]
    env.world.sim_time = 2
    env.traffic.prepare(env)
    assert not index.closed_edges
    assert env.world.fleet.active[1]
