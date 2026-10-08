"""OpenStreetMap から道路網・建物を取得し、`MapData` へ正規化する。"""

from __future__ import annotations

import json
import math
import logging
import re
import time
from pathlib import Path
from dataclasses import replace
from typing import Any, Iterable, Sequence

import numpy as np

from app import config
from app.contracts import (
    SIGNAL_MERGE_M,
    Bounds,
    MapBuilding,
    MapData,
    MapEdge,
    MapNode,
    MapPreset,
    MapSign,
    MapSignal,
)
from app.warn import warn_once

logger = logging.getLogger("autoware_sim")


class MapLoadError(RuntimeError):
    """OSM の取得・正規化に失敗したときに投げる。"""


CACHE_VERSION = 12

_SIGN_TAGS = (
    "traffic_sign", "traffic_sign:forward", "traffic_sign:backward",
    "traffic_sign:direction", "direction", "crossing", "stop", "stop:direction",
)
_PARKING_SIGN_TAGS = tuple(
    f"parking:{side}:restriction" for side in ("left", "right", "both")
) + tuple(f"parking:condition:{side}" for side in ("left", "right", "both"))
SIGN_BOARD_SPACING_M = 1.0

_DEFAULT_LANES: dict[str, int] = {
    "motorway": 3,
    "motorway_link": 1,
    "trunk": 3,
    "trunk_link": 1,
    "primary": 2,
    "primary_link": 1,
    "secondary": 2,
    "secondary_link": 1,
    "tertiary": 2,
    "tertiary_link": 1,
    "unclassified": 1,
    "residential": 1,
    "living_street": 1,
    "service": 1,
}
_FALLBACK_LANES = 1

_DEFAULT_MAXSPEED_KPH: dict[str, float] = {
    "motorway": 80.0,
    "motorway_link": 50.0,
    "trunk": 60.0,
    "trunk_link": 40.0,
    "primary": 50.0,
    "primary_link": 40.0,
    "secondary": 50.0,
    "secondary_link": 40.0,
    "tertiary": 40.0,
    "tertiary_link": 30.0,
    "unclassified": 30.0,
    "residential": 30.0,
    "living_street": 20.0,
    "service": 20.0,
}
_FALLBACK_MAXSPEED_KPH = 40.0

# osmnx がまとめた道のタグはリストで来て、並びは実行ごとに変わる（集合から戻すため）。
# `highway` はこの順で幹線に近いものを採る（表に無いものは後ろ・同順位は名前順）
_HIGHWAY_PRIORITY: tuple[str, ...] = (
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
)
_HIGHWAY_RANK = {name: rank for rank, name in enumerate(_HIGHWAY_PRIORITY)}

_KPH_TO_MPS = 1.0 / 3.6
_MPH_TO_MPS = 0.44704

_MIN_LANES = 1
_MAX_LANES = 10


def load_map(preset: MapPreset, *, force_refresh: bool = False) -> MapData:
    """プリセット 1 件分のマップを読み込む。"""
    cache_path = _cache_path(preset)

    if not force_refresh:
        cached = _read_cache(cache_path, preset)
        if cached is not None:
            return cached

    data = _fetch_and_normalize(preset)
    _write_cache(cache_path, data, preset)
    return data


def cache_path_for(preset: MapPreset) -> Path:
    """プリセットに対応する JSON キャッシュのパス（prefetch などの表示用）。"""
    return _cache_path(preset)


def _fetch_and_normalize(preset: MapPreset) -> MapData:
    try:
        import osmnx as ox
        from pyproj import Transformer
    except Exception as exc:  # pragma: no cover - 環境不備
        raise MapLoadError(f"地図処理ライブラリの読み込みに失敗しました: {exc}") from exc

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(config.OSMNX_CACHE_DIR)
    ox.settings.useful_tags_node = list(dict.fromkeys(ox.settings.useful_tags_node + list(_SIGN_TAGS)))
    ox.settings.useful_tags_way = list(dict.fromkeys(ox.settings.useful_tags_way + list(_SIGN_TAGS) + list(_PARKING_SIGN_TAGS)))

    center = (preset.center_lat, preset.center_lon)
    dist = float(preset.radius_m)

    try:
        graph = ox.graph_from_point(
            center,
            dist=dist,
            network_type="drive",
            simplify=False,
            truncate_by_edge=False,
            retain_all=False,
        )
    except Exception as exc:
        raise MapLoadError(
            f"道路網の取得に失敗しました（preset={preset.id}）: {exc}"
        ) from exc

    try:
        for _node_id, attrs in graph.nodes(data=True):
            if _node_has_sign(attrs):
                attrs["driverl_sign"] = True
        graph = ox.simplify_graph(
            graph,
            node_attrs_include=["driverl_sign"],
            edge_attrs_differ=["traffic_sign", "traffic_sign:forward", "traffic_sign:backward", "traffic_sign:direction", "stop:direction", "direction", *_PARKING_SIGN_TAGS],
        )
        projected = ox.project_graph(graph)
        crs = projected.graph["crs"]
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(projected)
    except Exception as exc:
        raise MapLoadError(
            f"道路網の投影・変換に失敗しました（preset={preset.id}）: {exc}"
        ) from exc

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    origin_x, origin_y = transformer.transform(preset.center_lon, preset.center_lat)

    buildings_gdf = None
    try:
        buildings_gdf = ox.features_from_point(center, tags={"building": True}, dist=dist)
        if len(buildings_gdf) > 0:
            buildings_gdf = buildings_gdf.to_crs(crs)
        else:
            buildings_gdf = None
            logger.warning(
                "建物が 1 件も取得できませんでした（preset=%s）。"
                "建物との衝突が起きないマップになります。"
                "Overpass の一時的な失敗であれば force_refresh で取り直してください",
                preset.id,
            )
    except Exception:
        buildings_gdf = None
        logger.exception(
            "建物の取得に失敗しました（preset=%s）。建物なしで続行しますが、"
            "建物との衝突が起きないマップになります。"
            "force_refresh で取り直してください",
            preset.id,
        )

    raw_node_xy = _collect_node_xy(nodes_gdf, origin_x, origin_y)
    raw_edges = _collect_edges(edges_gdf, raw_node_xy, origin_x, origin_y)

    if not raw_edges:
        raise MapLoadError(f"走行可能なエッジが 1 本も得られませんでした（preset={preset.id}）")

    nodes, edges, remap = _renumber(raw_node_xy, raw_edges)

    if len(nodes) < 5:
        raise MapLoadError(
            f"ノード数が少なすぎます（preset={preset.id}, nodes={len(nodes)}）"
        )
    if not edges:
        raise MapLoadError(f"走行可能なエッジが 1 本も得られませんでした（preset={preset.id}）")

    buildings = _collect_buildings(buildings_gdf, origin_x, origin_y)
    signals = _build_signals(
        _collect_signal_osmids(nodes_gdf),
        remap,
        edges,
        at_all_intersections=preset.signals_at_all_intersections,
    )
    signs = _build_speed_signs(edges)
    signs.extend(_build_traffic_signs(nodes_gdf, raw_edges, remap, edges, start_id=len(signs), existing_signs=signs))

    bounds = _compute_bounds(nodes, edges, buildings)

    return MapData(
        preset_id=preset.id,
        name=preset.name,
        center_lat=preset.center_lat,
        center_lon=preset.center_lon,
        radius_m=preset.radius_m,
        bounds=bounds,
        nodes=nodes,
        edges=edges,
        buildings=buildings,
        signals=signals,
        signs=signs,
    )


def _collect_node_xy(
    nodes_gdf: Any, origin_x: float, origin_y: float
) -> dict[int, tuple[float, float]]:
    """osmid -> ENU 座標 の辞書を作る。"""
    out: dict[int, tuple[float, float]] = {}
    xs = nodes_gdf["x"].to_numpy(dtype=float)
    ys = nodes_gdf["y"].to_numpy(dtype=float)
    for osmid, px, py in zip(nodes_gdf.index, xs, ys):
        out[int(osmid)] = (float(px) - origin_x, float(py) - origin_y)
    return out


def _collect_edges(
    edges_gdf: Any,
    node_xy: dict[int, tuple[float, float]],
    origin_x: float,
    origin_y: float,
) -> dict[tuple[int, int], dict[str, Any]]:
    """(u, v) -> エッジ属性 の辞書を作る。"""
    best: dict[tuple[int, int], dict[str, Any]] = {}

    for (u, v, _key), row in edges_gdf.iterrows():
        u = int(u)
        v = int(v)
        if u not in node_xy or v not in node_xy:
            continue

        polyline = _edge_polyline(row, node_xy[u], node_xy[v], origin_x, origin_y)
        if polyline is None or len(polyline) < 2:
            continue

        length = _polyline_length(polyline)
        if length <= 0.0:
            continue

        key = (u, v)
        previous = best.get(key)
        if previous is not None and previous["length"] <= length:
            continue

        highway = _highway_class(row.get("highway"))
        oneway = _parse_oneway(row.get("oneway"))
        lanes_value, lanes_from_tag = _parse_lanes(row.get("lanes"), highway)

        total_lanes = lanes_value if (lanes_from_tag or oneway) else lanes_value * 2
        total_lanes = int(min(max(total_lanes, _MIN_LANES), _MAX_LANES))
        width = max(total_lanes * config.DEFAULT_LANE_WIDTH, config.MIN_ROAD_WIDTH)

        speed_limit = _parse_maxspeed(row.get("maxspeed"), highway)

        best[key] = {
            "u": u,
            "v": v,
            "lanes": total_lanes,
            "width": float(width),
            "oneway": bool(oneway),
            "speed_limit": float(speed_limit),
            "polyline": polyline,
            "length": float(length),
            "sign_tags": {tag: row.get(tag) for tag in _SIGN_TAGS + _PARKING_SIGN_TAGS},
            "reversed": _tag_is_reversed(row.get("reversed")),
        }

    return best


def _edge_polyline(
    row: Any,
    u_xy: tuple[float, float],
    v_xy: tuple[float, float],
    origin_x: float,
    origin_y: float,
) -> list[tuple[float, float]] | None:
    """エッジのポリラインを ENU 座標で取り出し、始点が u・終点が v になるよう揃える。"""
    geom = row.get("geometry")
    coords: list[tuple[float, float]]

    if geom is not None and getattr(geom, "geom_type", None) == "LineString":
        coords = [(float(px) - origin_x, float(py) - origin_y) for px, py in geom.coords]
    else:
        coords = [u_xy, v_xy]

    coords = _dedupe_consecutive(coords)
    if len(coords) < 2:
        return None

    head_to_u = _sq_dist(coords[0], u_xy)
    head_to_v = _sq_dist(coords[0], v_xy)
    if head_to_v < head_to_u:
        coords.reverse()

    coords[0] = u_xy
    coords[-1] = v_xy
    coords = _dedupe_consecutive(coords)
    return coords if len(coords) >= 2 else None


def _renumber(
    node_xy: dict[int, tuple[float, float]],
    raw_edges: dict[tuple[int, int], dict[str, Any]],
) -> tuple[list[MapNode], list[MapEdge], dict[int, int]]:
    """osmid を 0 始まりの連番に振り直す。"""
    used: set[int] = set()
    for u, v in raw_edges:
        used.add(u)
        used.add(v)

    ordered = sorted(used)
    remap = {osmid: i for i, osmid in enumerate(ordered)}

    nodes = [
        MapNode(id=remap[osmid], x=node_xy[osmid][0], y=node_xy[osmid][1])
        for osmid in ordered
    ]

    edges: list[MapEdge] = []
    for (u, v), attrs in sorted(raw_edges.items()):
        edges.append(
            MapEdge(
                id=len(edges),
                u=remap[u],
                v=remap[v],
                lanes=attrs["lanes"],
                width=attrs["width"],
                oneway=attrs["oneway"],
                speed_limit=attrs["speed_limit"],
                polyline=attrs["polyline"],
                length=attrs["length"],
            )
        )
    return nodes, edges, remap


SIGNALS_AT_ALL_INTERSECTIONS = True

SIGNAL_MIN_STREETS = 3

SIGNAL_CROSSWALK_M = 4.0

SIGNAL_STOPLINE_MARGIN_M = 1.0

SIGNAL_CONFLICT_ANGLE = math.radians(30.0)


def _collect_signal_osmids(nodes_gdf: Any) -> set[int]:
    """`highway=traffic_signals` が付いたノードの osmid を集める。"""
    if nodes_gdf is None:
        return set()
    columns = getattr(nodes_gdf, "columns", [])
    if "highway" not in columns:
        return set()

    out: set[int] = set()
    for osmid, value in zip(nodes_gdf.index, nodes_gdf["highway"]):
        for tag in _iter_tag_values(value):
            if str(tag).strip() == "traffic_signals":
                out.add(int(osmid))
                break
    return out


def _point_before_end(
    polyline: Sequence[tuple[float, float]], setback: float
) -> tuple[tuple[float, float], float]:
    """終点から `setback` だけ手前に戻った点と、そこでの進行方向を返す。"""
    remaining = float(setback)
    for i in range(len(polyline) - 1, 0, -1):
        x1, y1 = polyline[i]
        x0, y0 = polyline[i - 1]
        dx, dy = x1 - x0, y1 - y0
        seg = math.hypot(dx, dy)
        if seg <= 1e-9:
            continue
        heading = math.atan2(dy, dx)
        if remaining <= seg:
            t = remaining / seg
            return (x1 - dx * t, y1 - dy * t), heading
        remaining -= seg

    x0, y0 = polyline[0]
    x1, y1 = polyline[-1]
    return (x0, y0), math.atan2(y1 - y0, x1 - x0)


def _axis_angle(a: float, b: float) -> float:
    """2 つの方位を「軸」として比べた角度差 [0, pi/2]。向きの正負は無視する。"""
    d = abs(math.atan2(math.sin(a - b), math.cos(a - b)))
    return min(d, math.pi - d)


def _phase_groups(
    headings: Sequence[float], thresh: float = SIGNAL_CONFLICT_ANGLE
) -> list[int]:
    """進入路を、軸のそろった組（同時に青にしてよい流れ）へ分けた群番号を返す。"""
    n = len(headings)
    if n <= 1:
        return [0] * n

    axes = sorted((h % math.pi, i) for i, h in enumerate(headings))
    gaps = [(axes[(k + 1) % n][0] - axes[k][0]) % math.pi for k in range(n)]
    start = (max(range(n), key=lambda k: gaps[k]) + 1) % n

    out = [0] * n
    group = 0
    base = axes[start][0]
    for step in range(n):
        axis, idx = axes[(start + step) % n]
        if step and (axis - base) % math.pi >= thresh:
            group += 1
            base = axis
        out[idx] = group
    return out


def _build_signals(
    signal_osmids: set[int],
    remap: dict[int, int],
    edges: Sequence[MapEdge],
    at_all_intersections: bool | None = None,
) -> list[MapSignal]:
    """交差点ごとに、進入路 1 本につき 1 基の車両用信号機を作る。"""
    incident: dict[int, list[MapEdge]] = {}
    for e in edges:
        if e.u == e.v:
            continue
        incident.setdefault(e.u, []).append(e)
        incident.setdefault(e.v, []).append(e)

    if at_all_intersections is None:
        at_all_intersections = SIGNALS_AT_ALL_INTERSECTIONS
    if at_all_intersections:
        candidates = sorted(incident)
    else:
        if not signal_osmids:
            return []
        candidates = sorted(
            {remap[o] for o in signal_osmids if o in remap}
        )

    signals: list[MapSignal] = []
    tagged_nodes = {remap[o] for o in signal_osmids if o in remap}
    for node_id in candidates:
        around = incident.get(node_id, [])
        if not around:
            continue

        neighbours = {(e.u if e.v == node_id else e.v) for e in around}
        if len(neighbours) < SIGNAL_MIN_STREETS:
            continue

        half_width = max(e.width for e in around) / 2.0
        setback = half_width + SIGNAL_CROSSWALK_M + SIGNAL_STOPLINE_MARGIN_M

        approaches: dict[int, tuple[float, tuple[float, float], float, int]] = {}
        for e in around:
            if e.v == node_id:
                neighbour = e.u
                point, heading = _point_before_end(e.polyline, setback)
            elif e.u == node_id and not e.oneway:
                neighbour = e.v
                point, heading = _point_before_end(list(reversed(e.polyline)), setback)
            else:
                continue
            approaches.setdefault(neighbour, (heading, point, e.width, e.id))

        if not approaches:
            continue

        for heading, (px, py), width, edge_id in (
            approaches[k] for k in sorted(approaches)
        ):
            signals.append(
                MapSignal(
                    id=len(signals),
                    node_id=node_id,
                    x=px,
                    y=py,
                    heading=heading,
                    group=0,
                    road_width=width,
                    phase_key=node_id,
                    edge_id=edge_id,
                    source="osm" if node_id in tagged_nodes else "synthetic",
                )
            )

    return _merge_phases(signals)


def _merge_phases(signals: list[MapSignal]) -> list[MapSignal]:
    """近接した灯器を 1 つの交差点とみなし、現示（位相と群）を揃える。"""
    n = len(signals)
    if n == 0:
        return signals

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_node: dict[int, list[int]] = {}
    for i, sg in enumerate(signals):
        by_node.setdefault(sg.node_id, []).append(i)
    for members in by_node.values():
        for i in members[1:]:
            union(members[0], i)

    cells: dict[tuple[int, int], list[int]] = {}
    for i, sg in enumerate(signals):
        key = (int(sg.x // SIGNAL_MERGE_M), int(sg.y // SIGNAL_MERGE_M))
        cells.setdefault(key, []).append(i)
    reach = SIGNAL_MERGE_M * SIGNAL_MERGE_M
    for (cx, cy), members in cells.items():
        for dx, dy in ((0, 0), (0, 1), (1, -1), (1, 0), (1, 1)):
            other = cells.get((cx + dx, cy + dy))
            if not other:
                continue
            same_cell = dx == 0 and dy == 0
            for i in members:
                for j in other:
                    if same_cell and j <= i:
                        continue
                    a, b = signals[i], signals[j]
                    if (a.x - b.x) ** 2 + (a.y - b.y) ** 2 <= reach:
                        union(i, j)

    leader: dict[int, int] = {}
    members: dict[int, list[int]] = {}
    for i, sg in enumerate(signals):
        root = find(i)
        members.setdefault(root, []).append(i)
        best = leader.get(root)
        if best is None or (sg.node_id, sg.id) < (signals[best].node_id, signals[best].id):
            leader[root] = i

    out = list(signals)
    for root, idxs in members.items():
        phase_key = signals[leader[root]].node_id
        groups = _phase_groups([signals[i].heading for i in idxs])
        for i, group in zip(idxs, groups):
            out[i] = replace(signals[i], phase_key=phase_key, group=group)
    return out


SIGNS_ONLY_WHERE_LIMIT_CHANGES = True

SIGN_LIMIT_EPSILON_MPS = 1.0 / 3.6


def _point_after_start(
    polyline: Sequence[tuple[float, float]], setback: float
) -> tuple[tuple[float, float], float]:
    """始点から `setback` だけ進んだ点と、そこでの進行方向を返す。"""
    remaining = float(setback)
    for i in range(len(polyline) - 1):
        x0, y0 = polyline[i]
        x1, y1 = polyline[i + 1]
        dx, dy = x1 - x0, y1 - y0
        seg = math.hypot(dx, dy)
        if seg <= 1e-9:
            continue
        heading = math.atan2(dy, dx)
        if remaining <= seg:
            t = remaining / seg
            return (x0 + dx * t, y0 + dy * t), heading
        remaining -= seg

    x0, y0 = polyline[0]
    x1, y1 = polyline[-1]
    return (x1, y1), math.atan2(y1 - y0, x1 - x0)


def _build_speed_signs(edges: Sequence[MapEdge]) -> list[MapSign]:
    """規制速度が変わる進入口に、最高速度標識を 1 基ずつ立てる。"""
    arriving: dict[int, dict[int, float]] = {}
    leaving: dict[int, dict[int, tuple[MapEdge, list[tuple[float, float]]]]] = {}

    for e in edges:
        if e.u == e.v or len(e.polyline) < 2:
            continue
        forward = [(float(px), float(py)) for px, py in e.polyline]
        arriving.setdefault(e.v, {}).setdefault(e.u, e.speed_limit)
        leaving.setdefault(e.u, {}).setdefault(e.v, (e, forward))
        if not e.oneway:
            arriving.setdefault(e.u, {}).setdefault(e.v, e.speed_limit)
            leaving.setdefault(e.v, {}).setdefault(e.u, (e, list(reversed(forward))))

    signs: list[MapSign] = []
    for node_id in sorted(leaving):
        for neighbour in sorted(leaving[node_id]):
            edge, points = leaving[node_id][neighbour]
            if SIGNS_ONLY_WHERE_LIMIT_CHANGES:
                incoming = [
                    limit
                    for other, limit in arriving.get(node_id, {}).items()
                    if other != neighbour
                ]
                if incoming and all(
                    abs(limit - edge.speed_limit) < SIGN_LIMIT_EPSILON_MPS
                    for limit in incoming
                ):
                    continue

            (px, py), heading = _point_after_start(points, config.SPEED_SIGN_SETBACK_M)
            offset = edge.width / 2.0 + config.SPEED_SIGN_SIDE_MARGIN
            px += -math.sin(heading) * offset
            py += math.cos(heading) * offset

            signs.append(
                MapSign(
                    id=len(signs),
                    node_id=node_id,
                    edge_id=edge.id,
                    x=px,
                    y=py,
                    heading=heading,
                    speed_limit=float(edge.speed_limit),
                )
            )

    return signs


def _sign_tokens(value: Any) -> set[str]:
    """国別コードの接頭辞を補い、複数の標識タグを分ける。"""
    tokens: set[str] = set()
    for item in _iter_tag_values(value):
        country = ""
        for token in re.split(r"[;,]", str(item).strip().lower()):
            token = token.strip().split("[", 1)[0]
            if ":" in token:
                country = token.split(":", 1)[0]
            elif country and token[:1].isdigit():
                token = f"{country}:{token}"
            if token:
                tokens.add(token)
    return tokens


def _tag_is_reversed(value: Any) -> bool:
    return any(str(item).strip().lower() == "true" for item in _iter_tag_values(value))


def _node_has_sign(tags: Any) -> bool:
    return bool(_sign_tokens(tags.get("highway")) & {"stop", "crossing"}) or any(
        _sign_tokens(tags.get(key)) for key in ("traffic_sign", "traffic_sign:forward", "traffic_sign:backward")
    )


def _sign_specs(tags: Any, forward: bool, *, node: bool = False) -> list[tuple[str, str]]:
    """OSM タグを標識の種類と矢印方向へ正規化する。"""
    values = _sign_tokens(tags.get("traffic_sign")) | _sign_tokens(
        tags.get("traffic_sign:forward" if forward else "traffic_sign:backward")
    )
    directional_signs = any(_sign_tokens(tags.get(key)) for key in ("traffic_sign:forward", "traffic_sign:backward"))
    if node and not directional_signs:
        if "stop" in _sign_tokens(tags.get("highway")):
            values.add("stop")
        if "crossing" in _sign_tokens(tags.get("highway")) and "no" not in _sign_tokens(tags.get("crossing")):
            values.add("crosswalk")
    aliases = {
        "stop": "stop", "jp:330": "stop", "jp:330-a": "stop", "jp:330-b": "stop",
        "crosswalk": "crosswalk", "pedestrian_crossing": "crosswalk", "jp:407-a": "crosswalk", "jp:407-b": "crosswalk",
        "one_way": "one_way", "oneway": "one_way", "jp:326": "one_way", "jp:326-a": "one_way", "jp:326-b": "one_way",
        "no_parking": "no_parking", "jp:316": "no_parking",
        "no_stopping": "no_stopping", "jp:315": "no_stopping",
    }
    arrows = {
        "only_straight_on": "straight", "straight_only": "straight",
        "only_left_turn": "left", "mandatory_left": "left",
        "only_right_turn": "right", "mandatory_right": "right",
        "left_or_straight": "left_or_straight", "right_or_straight": "right_or_straight",
        "left_or_right": "left_or_right",
    }
    specs = {(aliases[value], "straight") for value in values if value in aliases}
    specs.update(("mandatory_direction", arrows[value]) for value in values if value in arrows)
    if any(value.startswith("jp:311") for value in values) and not any(kind == "mandatory_direction" for kind, _direction in specs):
        warn_once("map.mandatory_direction", "指定方向外進行禁止の矢印が不明な OSM タグは配置を省略します")
    return sorted(specs)


def _sign_applies(tags: Any, forward: bool, heading: float) -> bool:
    """道路の順方向・逆方向または標識の正面方位で対象の進入路を選ぶ。"""
    values = set()
    for key in ("traffic_sign:direction", "stop:direction", "direction"):
        values = _sign_tokens(tags.get(key))
        if values:
            break
    if not values or "both" in values:
        return True
    if "forward" in values or "backward" in values:
        return ("forward" in values and forward) or ("backward" in values and not forward)
    cardinal = {"n": 0, "ne": 45, "e": 90, "se": 135, "s": 180, "sw": 225, "w": 270, "nw": 315}
    for value in values:
        try:
            degrees = float(cardinal[value] if value in cardinal else value)
        except ValueError:
            continue
        travel = math.pi / 2 - math.radians(degrees) + math.pi
        if math.cos(travel - heading) > math.cos(math.pi / 4):
            return True
    return False


def _build_traffic_signs(
    nodes_gdf: Any,
    raw_edges: dict[tuple[int, int], dict[str, Any]],
    remap: dict[int, int],
    edges: Sequence[MapEdge],
    *,
    start_id: int = 0,
    existing_signs: Sequence[MapSign] = (),
) -> list[MapSign]:
    """OSM の道路・ノードの標識を進入方向ごとに生成し、同じ位置の重複を除く。"""
    raw_by_hop = {(remap[u], remap[v]): attrs for (u, v), attrs in raw_edges.items()}
    incoming: dict[int, dict[int, tuple[MapEdge, list[tuple[float, float]], bool]]] = {}
    outgoing: dict[int, dict[int, tuple[MapEdge, list[tuple[float, float]], bool]]] = {}
    directed = []
    for edge in edges:
        if edge.u == edge.v or len(edge.polyline) < 2:
            continue
        attrs = raw_by_hop.get((edge.u, edge.v), {})
        for forward in (True,) if edge.oneway else (True, False):
            entry, exit_node = (edge.u, edge.v) if forward else (edge.v, edge.u)
            points = list(edge.polyline) if forward else list(reversed(edge.polyline))
            way_forward = forward != bool(attrs.get("reversed", False))
            item = (edge, points, way_forward)
            incoming.setdefault(exit_node, {}).setdefault(entry, item)
            outgoing.setdefault(entry, {}).setdefault(exit_node, item)
            directed.append((entry, exit_node, item, attrs.get("sign_tags", {})))

    signs: list[MapSign] = []
    seen: set[tuple] = set()
    occupied: dict[tuple[int, int], list[tuple[float, float, float]]] = {}

    def reserve(px, py, heading):
        occupied.setdefault((math.floor(px), math.floor(py)), []).append((px, py, heading))

    def overlaps(px, py, heading):
        col, row = math.floor(px), math.floor(py)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for sx, sy, sh in occupied.get((col + dx, row + dy), ()):
                    if (px - sx) ** 2 + (py - sy) ** 2 < (SIGN_BOARD_SPACING_M - 1e-6) ** 2 and math.cos(heading - sh) > 0.95:
                        return True
        return False

    for sign in existing_signs:
        reserve(sign.x, sign.y, sign.heading)

    def add(edge, points, node_id, kind, direction, *, at_exit=False, side=1):
        path = list(reversed(points)) if at_exit else points
        setback = 3.0 if at_exit else config.SPEED_SIGN_SETBACK_M
        (px, py), heading = _point_after_start(path, min(setback, edge.length / 2))
        if at_exit:
            heading = math.atan2(-math.sin(heading), -math.cos(heading))
        offset = side * (edge.width / 2 + config.SPEED_SIGN_SIDE_MARGIN)
        px -= math.sin(heading) * offset
        py += math.cos(heading) * offset
        key = (round(px, 3), round(py, 3), round(heading, 3), kind, direction)
        if key in seen:
            return
        seen.add(key)
        while overlaps(px, py, heading):
            px -= side * math.sin(heading) * SIGN_BOARD_SPACING_M
            py += side * math.cos(heading) * SIGN_BOARD_SPACING_M
        reserve(px, py, heading)
        signs.append(MapSign(start_id + len(signs), node_id, edge.id, px, py, heading, kind=kind, direction=direction))

    for entry, exit_node, (edge, points, forward), tags in directed:
        specs = _sign_specs(tags, forward)
        if edge.oneway:
            add(edge, points, entry, "one_way", "straight")
        for kind, direction in specs:
            at_exit = kind in {"stop", "crosswalk", "mandatory_direction"}
            path = list(reversed(points)) if at_exit else points
            _, heading = _point_after_start(path, min(3.0, edge.length / 2))
            if at_exit:
                heading += math.pi
            if _sign_applies(tags, forward, heading):
                add(edge, points, exit_node if at_exit else entry, kind, direction, at_exit=at_exit)
        for side in ("left", "right", "both"):
            restrictions = _sign_tokens(tags.get(f"parking:{side}:restriction")) | _sign_tokens(tags.get(f"parking:condition:{side}"))
            for kind in sorted(restrictions & {"no_parking", "no_stopping"}):
                relative_side = 1 if side == "both" or (side == "left") == forward else -1
                add(edge, points, entry, kind, "straight", side=relative_side)

    for osmid, tags in nodes_gdf.iterrows():
        node_id = remap.get(int(osmid))
        if node_id is None or not _node_has_sign(tags):
            continue
        approaches = incoming.get(node_id, {})
        has_direction = any(_sign_tokens(tags.get(key)) for key in ("direction", "stop:direction", "traffic_sign:direction", "traffic_sign:forward", "traffic_sign:backward"))
        for edge, points, forward in approaches.values():
            _, heading = _point_after_start(list(reversed(points)), min(3.0, edge.length / 2))
            heading += math.pi
            if not _sign_applies(tags, forward, heading):
                continue
            for kind, direction in _sign_specs(tags, forward, node=True):
                if kind == "one_way":
                    continue
                if kind == "stop" and not has_direction and "all" not in _sign_tokens(tags.get("stop")) and len(approaches) > 1:
                    warn_once("map.stop_direction", "一時停止標識の進入方向が不明な OSM ノードは配置を省略します")
                    continue
                add(edge, points, node_id, kind, direction, at_exit=True)
        for edge, points, forward in outgoing.get(node_id, {}).values():
            for kind, direction in _sign_specs(tags, forward, node=True):
                if kind == "one_way":
                    _, heading = _point_after_start(points, min(3.0, edge.length / 2))
                    if _sign_applies(tags, forward, heading):
                        add(edge, points, node_id, kind, direction)
    return signs


def _collect_buildings(
    buildings_gdf: Any, origin_x: float, origin_y: float
) -> list[MapBuilding]:
    """建物フットプリントを正規化する。"""
    if buildings_gdf is None or len(buildings_gdf) == 0:
        return []

    heights = _column_or_none(buildings_gdf, "height")
    levels = _column_or_none(buildings_gdf, "building:levels")

    out: list[MapBuilding] = []
    for i, geom in enumerate(buildings_gdf.geometry):
        polygon = _largest_polygon(geom)
        if polygon is None:
            continue

        try:
            polygon = polygon.simplify(config.BUILDING_SIMPLIFY_TOLERANCE, preserve_topology=True)
        except Exception:
            warn_once(
                "map.loader.building_simplify",
                "建物の輪郭を簡略化できませんでした。簡略化せずに使います（初回のみ記録）",
            )
        if polygon is None or polygon.is_empty:
            continue
        polygon = _largest_polygon(polygon)
        if polygon is None or polygon.area < config.BUILDING_MIN_AREA:
            continue

        outline = [
            (float(px) - origin_x, float(py) - origin_y) for px, py in polygon.exterior.coords
        ]
        if len(outline) >= 2 and _sq_dist(outline[0], outline[-1]) < 1e-12:
            outline.pop()
        outline = _dedupe_consecutive(outline)
        if len(outline) < 3:
            continue

        height = _resolve_height(
            heights[i] if heights is not None else None,
            levels[i] if levels is not None else None,
        )

        out.append(MapBuilding(id=len(out), height=height, outline=outline))

    return out


def _resolve_height(height_tag: Any, levels_tag: Any) -> float:
    """建物高さを決める（memo 5章「情報がなければ一律の固定高さ」）。"""
    height = _parse_float_tag(height_tag)
    if height is not None and 1.0 <= height <= 700.0:
        return float(height)

    levels = _parse_float_tag(levels_tag)
    if levels is not None and 1.0 <= levels <= 200.0:
        return float(levels) * config.BUILDING_LEVEL_HEIGHT

    return float(config.DEFAULT_BUILDING_HEIGHT)


def _compute_bounds(
    nodes: Sequence[MapNode],
    edges: Sequence[MapEdge],
    buildings: Sequence[MapBuilding],
) -> Bounds:
    """全ノード・全エッジ頂点・全建物頂点から外接矩形を求める。"""
    xs: list[float] = [n.x for n in nodes]
    ys: list[float] = [n.y for n in nodes]
    for e in edges:
        for px, py in e.polyline:
            xs.append(px)
            ys.append(py)
    for b in buildings:
        for px, py in b.outline:
            xs.append(px)
            ys.append(py)

    if not xs:
        raise MapLoadError("bounds を計算できる座標がありません")

    return Bounds(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _iter_tag_values(value: Any) -> Iterable[Any]:
    """タグ値を平坦化して 1 個ずつ返す（`['2', '3']` のようなリスト対策）。"""
    if _is_missing(value):
        return
    if isinstance(value, (list, tuple, set, np.ndarray)):
        for item in value:
            if not _is_missing(item):
                yield item
        return
    yield value


def _highway_class(value: Any) -> str:
    """highway から代表値 1 個を取り出す。リストなら並びに依らず `_HIGHWAY_PRIORITY` で選ぶ。"""
    names = {str(item).strip().lower() for item in _iter_tag_values(value)}
    names.discard("")
    if not names:
        return ""
    return min(names, key=lambda name: (_HIGHWAY_RANK.get(name, len(_HIGHWAY_RANK)), name))


def _parse_float_tag(value: Any) -> float | None:
    """`'12'` / `'12 m'` / `'12;15'` / `['12', '9']` などから最初の数値を取り出す。"""
    for item in _iter_tag_values(value):
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            if math.isnan(float(item)):
                continue
            return float(item)
        text = str(item).strip().replace(",", ".")
        number = ""
        seen_dot = False
        for ch in text:
            if ch.isdigit():
                number += ch
            elif ch == "." and number and not seen_dot:
                number += ch
                seen_dot = True
            elif number:
                break
            elif ch in "+-" and not number:
                continue
            else:
                continue
        if number:
            try:
                return float(number)
            except ValueError:
                continue
    return None


def _parse_lanes(value: Any, highway: str) -> tuple[int, bool]:
    """車線数を頑健にパースする。"""
    candidates: list[int] = []
    for item in _iter_tag_values(value):
        parsed = _parse_float_tag(item)
        if parsed is not None and parsed >= 1.0:
            candidates.append(int(parsed))
    if candidates:
        return int(min(max(max(candidates), _MIN_LANES), _MAX_LANES)), True
    return _DEFAULT_LANES.get(highway, _FALLBACK_LANES), False


def _parse_maxspeed(value: Any, highway: str) -> float:
    """制限速度を m/s で返す。`'50'` / `'50 km/h'` / `'30 mph'` / リストに対応（リストは最も低い値）。"""
    speeds: list[float] = []
    for item in _iter_tag_values(value):
        text = str(item).strip().lower()
        number = _parse_float_tag(text)
        if number is None or number <= 0.0:
            continue
        if "mph" in text:
            speeds.append(float(number) * _MPH_TO_MPS)
        elif "knot" in text:
            speeds.append(float(number) * 0.514444)
        else:
            speeds.append(float(number) * _KPH_TO_MPS)
    if speeds:
        return min(speeds)

    kph = _DEFAULT_MAXSPEED_KPH.get(highway, _FALLBACK_MAXSPEED_KPH)
    return float(kph) * _KPH_TO_MPS


def _parse_oneway(value: Any) -> bool:
    """oneway を bool に正規化する。`True` / `'yes'` / `'-1'` / リストに対応（リストはどれかが一方通行なら一方通行）。"""
    for item in _iter_tag_values(value):
        if isinstance(item, (bool, np.bool_)):
            if bool(item):
                return True
            continue
        if str(item).strip().lower() in ("yes", "true", "1", "-1", "reversible"):
            return True
    return False


def _sq_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _dedupe_consecutive(
    points: Sequence[tuple[float, float]], eps: float = 1e-9
) -> list[tuple[float, float]]:
    """連続する重複点を取り除く。"""
    out: list[tuple[float, float]] = []
    for p in points:
        if out and _sq_dist(out[-1], p) <= eps:
            continue
        out.append((float(p[0]), float(p[1])))
    return out


def _polyline_length(points: Sequence[tuple[float, float]]) -> float:
    """ポリラインの実長 [m]。"""
    total = 0.0
    for i in range(1, len(points)):
        total += math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
    return total


def _largest_polygon(geom: Any) -> Any | None:
    """Polygon をそのまま、MultiPolygon なら最大面積の 1 枚を返す。面が無ければ None。"""
    if geom is None or getattr(geom, "is_empty", True):
        return None
    geom_type = getattr(geom, "geom_type", None)
    if geom_type == "Polygon":
        return geom
    if geom_type in ("MultiPolygon", "GeometryCollection"):
        best = None
        best_area = 0.0
        for part in geom.geoms:
            if getattr(part, "geom_type", None) != "Polygon" or part.is_empty:
                continue
            if part.area > best_area:
                best = part
                best_area = part.area
        return best
    return None


def _column_or_none(gdf: Any, name: str) -> list[Any] | None:
    """GeoDataFrame の列を Python のリストとして取り出す。無い列は None。"""
    if name not in gdf.columns:
        return None
    return list(gdf[name])


def _cache_path(preset: MapPreset) -> Path:
    return config.MAP_CACHE_DIR / f"{preset.id}.json"


def _read_cache(path: Path, preset: MapPreset) -> MapData | None:
    """キャッシュを読む。壊れている／版が古い／プリセット定義が変わっていれば None。"""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("version") != CACHE_VERSION:
        return None
    if payload.get("presetId") != preset.id:
        return None

    if not _close(payload.get("centerLat"), preset.center_lat, 1e-6):
        return None
    if not _close(payload.get("centerLon"), preset.center_lon, 1e-6):
        return None
    if not _close(payload.get("radiusM"), preset.radius_m, 1e-3):
        return None
    # 信号の置き方もプリセット定義の一部。変えたら古い配置を読まない
    if payload.get("signalsAtAllIntersections") != bool(preset.signals_at_all_intersections):
        return None

    try:
        return _from_cache_dict(payload, preset)
    except (KeyError, TypeError, ValueError):
        return None


def _write_cache(path: Path, data: MapData, preset: MapPreset) -> None:
    """キャッシュを書き出す。書き込みに失敗しても処理は続行する（読み込みは成功済み）。"""
    tmp = path.with_suffix(".json.tmp")
    payload = _to_cache_dict(data)
    payload["signalsAtAllIntersections"] = bool(preset.signals_at_all_intersections)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, separators=(",", ":"))
        tmp.replace(path)
    except OSError:
        return


def _to_cache_dict(data: MapData) -> dict[str, Any]:
    """MapData を JSON へ落とす。to_wire() と違い length など内部値も保存する。"""
    return {
        "version": CACHE_VERSION,
        "generatedAt": time.time(),
        "presetId": data.preset_id,
        "name": data.name,
        "centerLat": data.center_lat,
        "centerLon": data.center_lon,
        "radiusM": data.radius_m,
        "bounds": {
            "minX": data.bounds.min_x,
            "minY": data.bounds.min_y,
            "maxX": data.bounds.max_x,
            "maxY": data.bounds.max_y,
        },
        "nodes": [[n.id, round(n.x, 3), round(n.y, 3)] for n in data.nodes],
        "edges": [
            {
                "id": e.id,
                "u": e.u,
                "v": e.v,
                "lanes": e.lanes,
                "width": round(e.width, 3),
                "oneway": e.oneway,
                "speedLimit": round(e.speed_limit, 3),
                "length": round(e.length, 3),
                "polyline": [[round(px, 3), round(py, 3)] for px, py in e.polyline],
            }
            for e in data.edges
        ],
        "buildings": [
            {
                "id": b.id,
                "height": round(b.height, 2),
                "outline": [[round(px, 3), round(py, 3)] for px, py in b.outline],
            }
            for b in data.buildings
        ],
        "signals": [
            [
                sg.id,
                sg.node_id,
                round(sg.x, 3),
                round(sg.y, 3),
                round(sg.heading, 4),
                sg.group,
                round(sg.road_width, 2),
                sg.phase_key,
                sg.edge_id,
                sg.source,
            ]
            for sg in data.signals
        ],
        "signs": [
            [
                sn.id,
                sn.node_id,
                sn.edge_id,
                round(sn.x, 3),
                round(sn.y, 3),
                round(sn.heading, 4),
                round(sn.speed_limit, 3),
                sn.kind,
                sn.direction,
            ]
            for sn in data.signs
        ],
    }


def _from_cache_dict(payload: dict[str, Any], preset: MapPreset) -> MapData:
    bounds_raw = payload["bounds"]
    bounds = Bounds(
        min_x=float(bounds_raw["minX"]),
        min_y=float(bounds_raw["minY"]),
        max_x=float(bounds_raw["maxX"]),
        max_y=float(bounds_raw["maxY"]),
    )

    nodes = [MapNode(id=int(n[0]), x=float(n[1]), y=float(n[2])) for n in payload["nodes"]]

    edges: list[MapEdge] = []
    for e in payload["edges"]:
        polyline = [(float(px), float(py)) for px, py in e["polyline"]]
        edges.append(
            MapEdge(
                id=int(e["id"]),
                u=int(e["u"]),
                v=int(e["v"]),
                lanes=int(e["lanes"]),
                width=float(e["width"]),
                oneway=bool(e["oneway"]),
                speed_limit=float(e["speedLimit"]),
                polyline=polyline,
                length=float(e["length"]),
            )
        )

    buildings = [
        MapBuilding(
            id=int(b["id"]),
            height=float(b["height"]),
            outline=[(float(px), float(py)) for px, py in b["outline"]],
        )
        for b in payload["buildings"]
    ]

    signals = [
        MapSignal(
            id=int(sg[0]),
            node_id=int(sg[1]),
            x=float(sg[2]),
            y=float(sg[3]),
            heading=float(sg[4]),
            group=int(sg[5]),
            road_width=float(sg[6]),
            phase_key=int(sg[7]),
            edge_id=int(sg[8]),
            source=str(sg[9]) if len(sg) > 9 else "unknown",
        )
        for sg in payload.get("signals", [])
    ]

    signs = [
        MapSign(
            id=int(sn[0]),
            node_id=int(sn[1]),
            edge_id=int(sn[2]),
            x=float(sn[3]),
            y=float(sn[4]),
            heading=float(sn[5]),
            speed_limit=float(sn[6]),
            kind=str(sn[7]) if len(sn) > 7 else "speed_limit",
            direction=str(sn[8]) if len(sn) > 8 else "straight",
        )
        for sn in payload.get("signs", [])
    ]

    if len(nodes) < 5 or not edges:
        raise ValueError("キャッシュの内容が最小要件を満たしていません")

    return MapData(
        preset_id=str(payload["presetId"]),
        name=str(payload.get("name", preset.name)),
        center_lat=float(payload["centerLat"]),
        center_lon=float(payload["centerLon"]),
        radius_m=float(payload["radiusM"]),
        bounds=bounds,
        nodes=nodes,
        edges=edges,
        buildings=buildings,
        signals=signals,
        signs=signs,
    )


def _close(value: Any, expected: float, tol: float) -> bool:
    try:
        return abs(float(value) - float(expected)) <= tol
    except (TypeError, ValueError):
        return False
