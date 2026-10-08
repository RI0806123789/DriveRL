"""OSM 標識の方向、保存形式、経路の最高速度への影響を検査する。"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from app.contracts import MapNode, MapPreset, MapSign, RouteLeg
from app.map import build_map_index
from app.map.loader import (
    _build_speed_signs,
    _build_traffic_signs,
    _collect_edges,
    _fetch_and_normalize,
    _from_cache_dict,
    _node_has_sign,
    _sign_applies,
    _sign_specs,
    _sign_tokens,
    _to_cache_dict,
)


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def iterrows(self):
        return iter(self.rows)


def traffic_signs(tiny_map, node_tags=None, way_tags=None, *, oneway=False, reverse=False):
    edge = replace(tiny_map.edges[0], oneway=oneway)
    raw = {(10, 11): {"sign_tags": way_tags or {}, "reversed": reverse}}
    rows = Rows([(11, node_tags or {})])
    return _build_traffic_signs(rows, raw, {10: 0, 11: 1}, [edge], start_id=10)


@pytest.mark.parametrize("code,kind", [
    ("JP:330-A", "stop"), ("JP:330-B", "stop"),
    ("JP:407-A", "crosswalk"), ("JP:407-B", "crosswalk"),
    ("JP:326-A", "one_way"), ("JP:326-B", "one_way"),
    ("JP:315", "no_stopping"), ("JP:316", "no_parking"),
])
def test_japanese_sign_ids(code, kind):
    assert _sign_specs({"traffic_sign": code}, True) == [(kind, "straight")]


def test_multiple_codes_inherit_country_and_ignore_supplementary_values():
    assert _sign_tokens("JP:330-A,316[20];407-A") == {"jp:330-a", "jp:316", "jp:407-a"}


def test_tag_order_does_not_change_signs():
    first = {"traffic_sign": ["JP:315", "JP:316", "only_left_turn"]}
    second = {"traffic_sign": list(reversed(first["traffic_sign"]))}
    assert _sign_specs(first, True) == _sign_specs(second, True)
    assert ("mandatory_direction", "left") in _sign_specs(first, True)


def test_unknown_japanese_mandatory_arrow_does_not_become_straight():
    assert _sign_specs({"traffic_sign": "JP:311-A"}, True) == []


def test_stop_point_faces_approaching_vehicle(tiny_map):
    signs = traffic_signs(tiny_map, {"highway": "stop", "direction": "forward"})
    assert len(signs) == 1
    sign = signs[0]
    assert sign.id == 10 and sign.kind == "stop" and sign.node_id == 1
    assert sign.heading == pytest.approx(0)
    assert sign.x == pytest.approx(37)
    assert sign.y > tiny_map.edges[0].width / 2
    assert sign.speed_limit == 0


def test_forward_sign_tag_does_not_create_reverse_stop(tiny_map):
    tags = {"highway": "stop", "traffic_sign:forward": "stop"}
    assert _sign_specs(tags, False, node=True) == []
    assert len(traffic_signs(tiny_map, tags)) == 1


def test_road_order_reversal_swaps_directional_tags(tiny_map):
    assert not traffic_signs(tiny_map, {"highway": "stop", "direction": "forward"}, reverse=True)
    signs = traffic_signs(tiny_map, {"highway": "stop", "direction": "backward"}, reverse=True)
    assert len(signs) == 1 and signs[0].kind == "stop"


def test_compass_direction_means_sign_face_not_travel_direction():
    assert _sign_applies({"direction": "W"}, True, 0)
    assert _sign_applies({"direction": "270"}, True, 0)
    assert not _sign_applies({"direction": "E"}, True, 0)


def test_crossing_without_crossing_is_not_a_sign(tiny_map):
    assert not traffic_signs(tiny_map, {"highway": "crossing", "crossing": "no"})
    signs = traffic_signs(tiny_map, {"highway": "crossing"})
    assert len(signs) == 1 and signs[0].kind == "crosswalk"


def test_one_way_sign_has_only_legal_direction(tiny_map):
    signs = traffic_signs(tiny_map, oneway=True)
    assert len(signs) == 1
    assert signs[0].kind == "one_way" and signs[0].heading == pytest.approx(0)


def test_normalized_one_way_direction_overrides_way_sign_direction(tiny_map):
    signs = traffic_signs(tiny_map, way_tags={"direction": "backward"}, oneway=True)
    assert len(signs) == 1 and signs[0].kind == "one_way"


def test_different_sign_boards_are_separated_without_moving_speed_signs(tiny_map):
    edge = replace(tiny_map.edges[0], oneway=True)
    speed_signs = _build_speed_signs([edge])
    original = [(sign.x, sign.y, sign.heading) for sign in speed_signs]
    raw = {(10, 11): {"sign_tags": {"traffic_sign": "no_parking;no_stopping"}}}
    added = _build_traffic_signs(Rows([]), raw, {10: 0, 11: 1}, [edge], start_id=len(speed_signs), existing_signs=speed_signs)
    assert [(sign.x, sign.y, sign.heading) for sign in speed_signs] == original
    assert {sign.kind for sign in added} == {"one_way", "no_parking", "no_stopping"}
    all_signs = speed_signs + added
    for index, first in enumerate(all_signs):
        for second in all_signs[index + 1:]:
            assert math.hypot(first.x - second.x, first.y - second.y) >= 1 - 1e-9
    assert all(sign.x == pytest.approx(speed_signs[0].x) for sign in added)


def test_duplicate_signs_remain_deduplicated_after_board_separation(tiny_map):
    edge = replace(tiny_map.edges[0], oneway=True)
    speeds = _build_speed_signs([edge])
    raw = {(10, 11): {"sign_tags": {"traffic_sign": "one_way;no_parking"}}}
    added = _build_traffic_signs(Rows([]), raw, {10: 0, 11: 1}, [edge], existing_signs=speeds)
    assert len(added) == 2
    assert len({(sign.x, sign.y) for sign in added}) == 2


def test_right_parking_restriction_remains_on_right_of_one_way(tiny_map):
    signs = traffic_signs(tiny_map, way_tags={"parking:right:restriction": "no_stopping"}, oneway=True)
    parking = [sign for sign in signs if sign.kind == "no_stopping"]
    assert len(parking) == 1 and parking[0].y < -tiny_map.edges[0].width / 2


def test_duplicate_directed_edges_do_not_double_signs(tiny_map):
    edge = tiny_map.edges[0]
    reverse = replace(edge, id=1, u=1, v=0, polyline=list(reversed(edge.polyline)))
    raw = {(10, 11): {"sign_tags": {"traffic_sign": "no_parking"}}, (11, 10): {"sign_tags": {"traffic_sign": "no_parking"}}}
    signs = _build_traffic_signs(Rows([]), raw, {10: 0, 11: 1}, [edge, reverse])
    assert len(signs) == 2
    assert len({(sign.x, sign.y, sign.heading) for sign in signs}) == 2


def test_ambiguous_stop_is_skipped_but_all_way_stop_is_kept(tiny_map):
    edge = tiny_map.edges[0]
    second = replace(edge, id=1, u=1, v=2, polyline=[(40, 0), (80, 0)])
    raw = {(10, 11): {}, (11, 12): {}}
    remap = {10: 0, 11: 1, 12: 2}
    assert not _build_traffic_signs(Rows([(11, {"highway": "stop"})]), raw, remap, [edge, second])
    signs = _build_traffic_signs(Rows([(11, {"highway": "stop", "stop": "all"})]), raw, remap, [edge, second])
    assert len(signs) == 2
    assert math.cos(signs[0].heading - signs[1].heading) == pytest.approx(-1)


def test_osm_edge_collection_retains_sign_tags_and_direction():
    tags = {"highway": "residential", "traffic_sign:backward": "only_right_turn", "parking:both:restriction": "no_parking", "reversed": True}
    result = _collect_edges(Rows([((10, 11, 0), tags)]), {10: (0, 0), 11: (40, 0)}, 0, 0)
    attrs = result[(10, 11)]
    assert attrs["sign_tags"]["traffic_sign:backward"] == "only_right_turn"
    assert attrs["sign_tags"]["parking:both:restriction"] == "no_parking"
    assert attrs["reversed"] is True


def test_sign_marker_catches_points_before_osm_simplification():
    assert _node_has_sign({"highway": "stop"})
    assert _node_has_sign({"highway": "crossing"})
    assert _node_has_sign({"traffic_sign:backward": "JP:316"})
    assert not _node_has_sign({"highway": "traffic_signals"})


def test_fetch_preserves_sign_points_and_parking_boundaries_without_network(monkeypatch):
    import networkx as nx
    import osmnx as ox

    graph = nx.MultiDiGraph(crs="EPSG:3857")
    coordinates = {0: (0, 0), 1: (20, 0), 2: (40, 0), 3: (-40, 0), 4: (0, 40), 5: (0, -40), 6: (-20, 0)}
    for node_id, (x, y) in coordinates.items():
        attrs = {"x": x, "y": y}
        if node_id == 1:
            attrs.update(highway="stop", direction="forward")
        graph.add_node(node_id, **attrs)
    for u, v in ((0, 1), (1, 2), (0, 6), (6, 3), (0, 4), (0, 5)):
        attrs = {"length": 20.0, "oneway": False, "highway": "residential", "maxspeed": "30", "osmid": 1}
        if (u, v) == (6, 3):
            attrs["parking:both:restriction"] = "no_parking"
        graph.add_edge(u, v, reversed=False, **attrs)
        graph.add_edge(v, u, reversed=True, **attrs)
    for setting in ("useful_tags_node", "useful_tags_way", "use_cache", "cache_folder"):
        monkeypatch.setattr(ox.settings, setting, getattr(ox.settings, setting))
    monkeypatch.setattr(ox, "graph_from_point", lambda *_args, **_kwargs: graph)
    monkeypatch.setattr(ox, "project_graph", lambda value: value)
    monkeypatch.setattr(ox, "features_from_point", lambda *_args, **_kwargs: [])
    data = _fetch_and_normalize(MapPreset("fake", "合成", "", 0, 0, 100, False))
    assert len(data.nodes) == 7
    assert len(data.edges) == 12
    assert any(sign.kind == "stop" for sign in data.signs)
    parking = [sign for sign in data.signs if sign.kind == "no_parking"]
    assert len(parking) == 2
    for sign in parking:
        edge = data.edges[sign.edge_id]
        assert max(x for x, _y in edge.polyline) <= -20


def test_sign_cache_roundtrip_and_legacy_defaults(tiny_map):
    data = replace(tiny_map, nodes=tiny_map.nodes + [MapNode(i, i * 40, 0) for i in range(2, 5)])
    data.signs.append(MapSign(1, 1, 0, 37, 4, 0, kind="mandatory_direction", direction="left_or_right"))
    preset = MapPreset("tiny", "最小", "", 35, 139, 50)
    payload = _to_cache_dict(data)
    restored = _from_cache_dict(payload, preset)
    assert restored.signs == data.signs
    assert restored.to_wire()["signs"][1]["kind"] == "mandatory_direction"
    assert restored.to_wire()["signs"][1]["direction"] == "left_or_right"
    payload["signs"] = [payload["signs"][0][:7]]
    legacy = _from_cache_dict(payload, preset)
    assert legacy.signs[0].kind == "speed_limit" and legacy.signs[0].direction == "straight"


def test_non_speed_signs_never_shift_route_speed_change(tiny_map):
    edge = tiny_map.edges[0]
    next_edge = replace(edge, id=1, u=1, v=2, speed_limit=5, polyline=[(40, 0), (80, 0)])
    data = replace(tiny_map, nodes=tiny_map.nodes + [MapNode(2, 80, 0)], edges=[edge, next_edge], signs=_build_speed_signs([edge, next_edge]))
    points = [(float(x), 1.75) for x in np.arange(0, 81, 2)]
    legs = [RouteLeg(0, 0, 1, 0, 40), RouteLeg(1, 1, 2, 40, 80)]
    original = build_map_index(data).speed_limits_on_route(points, legs)
    data.signs.append(MapSign(len(data.signs), 1, 1, 42, 4, 0, kind="no_parking"))
    current = build_map_index(data).speed_limits_on_route(points, legs)
    assert current == original
    assert current[1][1] == 5
