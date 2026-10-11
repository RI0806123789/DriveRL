"""車線経路から交差点の通過区間と候補進路の交錯区間を作る。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np
import shapely
from shapely.geometry import LineString
from shapely.ops import substring

from app import config
from app.contracts import (
    IntersectionConflict,
    IntersectionMovement,
    MapEdge,
    RouteIntersection,
    RouteLeg,
)
from app.map.lanes import RouteSegment, TURN_ANGLE_THRESHOLD, build_lane_route

APPROACH_SPAN_M = 35.0
CONFLICT_EXTENT_M = 3.0
CONFLICT_MARGIN_M = 0.6
UTURN_THRESHOLD_RAD = math.radians(150.0)
HEADING_SPAN_M = 3.0


@dataclass(frozen=True)
class _MovementGeometry:
    movement: IntersectionMovement
    line: LineString
    swept: shapely.Geometry


class IntersectionGeometry:
    """通過する交差点だけの候補進路を遅延生成して再利用する。"""

    def __init__(self, graph: nx.DiGraph, edges: dict[int, MapEdge], junctions: frozenset[int]) -> None:
        self._graph = graph
        self._edges = edges
        self._junctions = junctions
        self._movements: dict[int, tuple[_MovementGeometry, ...]] = {}

    def on_route(
        self, points: Sequence[tuple[float, float]], legs: Sequence[RouteLeg],
        *, start_arc: float = 0.0, end_arc: float = math.inf,
    ) -> list[RouteIntersection]:
        """辺のつながりごとに交差点と交錯する候補進路を返す。"""
        if len(points) < 2 or len(legs) < 2 or start_arc > end_arc:
            return []
        coords = np.asarray(points, dtype=np.float64)
        cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
        total = float(cum[-1])
        if total <= 1e-6:
            return []
        result: list[RouteIntersection] = []
        for incoming, outgoing in zip(legs[:-1], legs[1:]):
            node = int(incoming.exit_node)
            if node != int(outgoing.entry_node) or node not in self._junctions:
                continue
            entry_arc, exit_arc = float(incoming.end_arc), float(outgoing.start_arc)
            if not 0.0 <= entry_arc <= exit_arc <= total + 0.01:
                continue
            if exit_arc < start_arc or entry_arc > end_arc:
                continue
            first = self._edges.get(int(incoming.edge_id))
            second = self._edges.get(int(outgoing.edge_id))
            if first is None or second is None or len(first.polyline) < 2 or len(second.polyline) < 2:
                continue
            entry_heading = _heading(_directed(first, int(incoming.entry_node)), at_end=True)
            exit_heading = _heading(_directed(second, node), at_end=False)
            start = max(0.0, entry_arc - CONFLICT_EXTENT_M)
            end = min(total, exit_arc + CONFLICT_EXTENT_M)
            local = _route_window(coords, cum, start, end)
            if local.length <= 1e-6:
                continue
            swept = local.buffer(config.VEHICLE_WIDTH + CONFLICT_MARGIN_M)
            conflicts: list[IntersectionConflict] = []
            for geometry in self._at_node(node):
                movement = geometry.movement
                if movement.entry_node == int(incoming.entry_node):
                    continue
                ego_span = _overlap_span(local, geometry.swept)
                if ego_span is None:
                    continue
                other_span = _overlap_span(geometry.line, swept)
                if other_span is None:
                    continue
                conflicts.append(IntersectionConflict(
                    movement=movement,
                    ego_start_arc=start + ego_span[0],
                    ego_end_arc=start + ego_span[1],
                    other_start_arc=other_span[0],
                    other_end_arc=other_span[1],
                ))
            result.append(RouteIntersection(
                node_id=node,
                entry_arc=entry_arc,
                exit_arc=min(exit_arc, total),
                entry_heading=entry_heading,
                exit_heading=exit_heading,
                turn=_turn(entry_heading, exit_heading),
                road_width=float(first.width),
                exit_road_width=float(second.width),
                conflicts=tuple(conflicts),
                movements=tuple(g.movement for g in self._at_node(node)),
            ))
        return result

    def _at_node(self, node: int) -> tuple[_MovementGeometry, ...]:
        """このノードへ合法に進入・退出する短い車線経路をキャッシュする。"""
        cached = self._movements.get(node)
        if cached is not None:
            return cached
        result: list[_MovementGeometry] = []
        incoming = [(int(pred), self._edge(int(pred), node)) for pred in self._graph.predecessors(node)]
        outgoing = [(int(succ), self._edge(node, int(succ))) for succ in self._graph.successors(node)]
        for entry_node, first in incoming:
            if first is None or len(first.polyline) < 2:
                continue
            enter = _near_node(_directed(first, entry_node), at_end=True)
            entry_heading = _heading(enter, at_end=True)
            for exit_node, second in outgoing:
                if entry_node == exit_node or second is None or len(second.polyline) < 2:
                    continue
                leave = _near_node(_directed(second, node), at_end=False)
                exit_heading = _heading(leave, at_end=False)
                if abs(_wrap(exit_heading - entry_heading)) > UTURN_THRESHOLD_RAD:
                    continue
                points, spans = build_lane_route([
                    RouteSegment(edge=first, points=enter),
                    RouteSegment(edge=second, points=leave),
                ], 1.0)
                if len(points) < 2 or spans[0] is None or spans[1] is None:
                    continue
                line = LineString(points)
                if line.length <= 1e-6:
                    continue
                coords = np.asarray(points, dtype=np.float64)
                cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
                movement = IntersectionMovement(
                    node_id=node,
                    entry_node=entry_node,
                    exit_node=exit_node,
                    incoming_edge_id=int(first.id),
                    outgoing_edge_id=int(second.id),
                    entry_heading=entry_heading,
                    exit_heading=exit_heading,
                    turn=_turn(entry_heading, exit_heading),
                    road_width=float(first.width),
                    exit_road_width=float(second.width),
                    points=tuple(points),
                    cum=tuple(float(arc) for arc in cumulative),
                    entry_arc=float(spans[0][1]),
                    exit_arc=float(spans[1][0]),
                )
                result.append(_MovementGeometry(
                    movement, line, line.buffer(config.VEHICLE_WIDTH + CONFLICT_MARGIN_M),
                ))
        cached = tuple(result)
        self._movements[node] = cached
        return cached

    def _edge(self, entry: int, exit_: int) -> MapEdge | None:
        attrs = self._graph.get_edge_data(entry, exit_)
        return None if attrs is None else self._edges.get(int(attrs["edge_id"]))


def _overlap_span(line: LineString, swept: shapely.Geometry) -> tuple[float, float] | None:
    """車体幅と余裕を含む進路に重なる弧長区間を返す。"""
    overlap = line.intersection(swept)
    coords = shapely.get_coordinates(overlap)
    if coords.shape[0] == 0:
        return None
    arc = shapely.line_locate_point(line, shapely.points(coords))
    return float(np.min(arc)), float(np.max(arc))


def _route_window(points: np.ndarray, cum: np.ndarray, start: float, end: float) -> LineString:
    first = min(max(int(np.searchsorted(cum, start, side="right")) - 1, 0), len(cum) - 2)
    last = min(max(int(np.searchsorted(cum, end, side="left")), first + 1), len(cum) - 1)
    lo_span, hi_span = cum[first + 1] - cum[first], cum[last] - cum[last - 1]
    lo = points[first] + (points[first + 1] - points[first]) * ((start - cum[first]) / max(lo_span, 1e-12))
    hi = points[last - 1] + (points[last] - points[last - 1]) * ((end - cum[last - 1]) / max(hi_span, 1e-12))
    return LineString(np.vstack((lo, points[first + 1:last], hi)))


def _directed(edge: MapEdge, entry_node: int) -> list[tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y in edge.polyline]
    return pts if int(edge.u) == entry_node else pts[::-1]


def _near_node(points: list[tuple[float, float]], at_end: bool) -> list[tuple[float, float]]:
    line = LineString(points)
    total = float(line.length)
    if total <= APPROACH_SPAN_M:
        return points
    part = substring(line, total - APPROACH_SPAN_M, total) if at_end else substring(line, 0, APPROACH_SPAN_M)
    return [(float(x), float(y)) for x, y in part.coords]


def _heading(points: Sequence[tuple[float, float]], at_end: bool) -> float:
    line = LineString(points)
    total = float(line.length)
    start, end = (max(0.0, total - HEADING_SPAN_M), total) if at_end else (0.0, min(total, HEADING_SPAN_M))
    a, b = line.interpolate(start), line.interpolate(end)
    return math.atan2(b.y - a.y, b.x - a.x)


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _turn(entry_heading: float, exit_heading: float) -> str:
    delta = _wrap(exit_heading - entry_heading)
    return "left" if delta > TURN_ANGLE_THRESHOLD else "right" if delta < -TURN_ANGLE_THRESHOLD else "straight"
