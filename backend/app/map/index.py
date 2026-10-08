"""`MapData` から実行時インデックス（`contracts.MapIndex`）を組み立てる。"""

from __future__ import annotations

import math
from typing import Any, Sequence

import networkx as nx
import logging

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring

from app import config
from app.contracts import (
    BRANCH_LEFT,
    BRANCH_RIGHT,
    SIGNAL_MERGE_M,
    MapData,
    MapEdge,
    OccupancyGrid,
    RoadLead,
    RouteLeg,
)

__all__ = ["MapIndexImpl", "build_map_index"]

from app.map.lanes import RouteSegment, build_lane_route
from app.warn import warn_once

logger = logging.getLogger("autoware_sim")

#: 建物の中を通るエッジの経路探索の重みに、中の長さ 1m あたり足す値 [m]
ROUTE_BUILDING_PENALTY = 1000.0
#: A* の見積もり（目的地までの直線距離）に掛ける係数。辺の重みはキャッシュの丸めで両端の直線距離を
#: 最大 2mm 下回るので、見積もりが過大にならないよう控えめにする
ROUTE_HEURISTIC_SCALE = 0.99
#: これより短い重なりは、建物の角をかすめただけとみなす [m]
BUILDING_OVERLAP_MIN_M = 1.0
#: いま走っている道路を探す範囲 [m]。幅 32.5m の道路では車線の中心が中心線から 15m 近く離れる
LEAD_SEARCH_RADIUS_M = 25.0
#: 道路の幅の内側にいる候補どうしで、中心線に近いほうを選ぶための重み
LEAD_CENTER_WEIGHT = 0.1
#: 道路を選ぶとき、向きのずれ 1rad を距離何 m ぶんとみなすか
LEAD_ANGLE_WEIGHT_M = 10.0
#: 交差点を「まっすぐ抜けられる」とみなす向きのずれ [rad]
LEAD_STRAIGHT_RAD = math.radians(35.0)
#: まっすぐ抜ける道をたどる上限
LEAD_MAX_HOPS = 8
#: エッジの向きを測る長さ [m]。端の 1 区間だけだと短すぎて向きが暴れる
LEAD_DIRECTION_SPAN_M = 3.0
#: 経路の出だしで、着いた向きからこれより大きく曲がる辺は折り返し（U ターン）とみなす [rad]
ROUTE_MAX_START_TURN_RAD = math.radians(120.0)
#: 地物が経路の始点より後ろ・終点より先にあるとみなす距離 [m]。経路点の float32 の丸めを吸収する
ROUTE_END_EPS_M = 0.01
#: 停止線の位置で、経路の向きと灯器の向きがこれ以上ずれていれば、その灯器に従わない [rad]
SIGNAL_HEADING_TOLERANCE_RAD = math.radians(35.0)
#: 交差点に着いた向きから、この範囲に入ってくる道を「左（右）から来る道」とみなす [rad]
BRANCH_SIDE_MIN_RAD = math.radians(30.0)
BRANCH_SIDE_MAX_RAD = math.radians(150.0)


class MapIndexImpl:
    """`contracts.MapIndex` プロトコルの実装。"""

    def __init__(self, data: MapData) -> None:
        self.data = data
        self._batch_collision_failed = False

        nodes = sorted(data.nodes, key=lambda n: n.id)
        if any(int(n.id) != row for row, n in enumerate(nodes)):
            raise ValueError(
                "道路ノードの ID は 0 からの連番にしてください（ID を行番号として引くため。"
                "map/loader.py の _renumber と同じ形）"
            )
        self._node_xy = np.array([[n.x, n.y] for n in nodes], dtype=np.float64)
        if self._node_xy.size == 0:
            self._node_xy = np.zeros((0, 2), dtype=np.float64)
        self._node_pts: dict[int, tuple[float, float]] = {
            int(n.id): (float(n.x), float(n.y)) for n in nodes
        }

        self._edges_by_id: dict[int, MapEdge] = {e.id: e for e in data.edges}

        self.graph = self._build_graph(data)
        # 3 方向以上に道がつながるノード（向きを問わず、隣のノードの数で数える）
        self._junctions = frozenset(
            int(node)
            for node in self.graph.nodes
            if len(set(self.graph.predecessors(node)) | set(self.graph.successors(node))) >= 3
        )
        self._reachable = self._build_reachable_nodes()
        self._reachable_mask = np.zeros(self._node_xy.shape[0], dtype=bool)
        self._reachable_mask[self._reachable[self._reachable < self._reachable_mask.size]] = True

        self._edge_lines: list[LineString] = []
        self._edge_line_ids: list[int] = []
        for e in data.edges:
            if len(e.polyline) < 2:
                continue
            self._edge_lines.append(LineString(e.polyline))
            self._edge_line_ids.append(e.id)
        self._edge_tree = shapely.STRtree(self._edge_lines) if self._edge_lines else None

        self._building_polys: list[Polygon] = []
        for b in data.buildings:
            if len(b.outline) < 3:
                continue
            poly = Polygon(b.outline)
            if not poly.is_valid:
                poly = poly.buffer(0)
                if poly.is_empty or poly.geom_type not in ("Polygon", "MultiPolygon"):
                    continue
            self._building_polys.append(poly)
        self._building_tree = (
            shapely.STRtree(self._building_polys) if self._building_polys else None
        )

        # 当たり判定は 2 次元なので、建物の中を通る道（高架・建物の下をくぐる道）を走れば
        #   建物にぶつかる。ほかに道があれば避けるよう、経路探索の重みだけを重くする
        self.edge_inside_building_m = self._edge_building_overlap()
        for _u, _v, attrs in self.graph.edges(data=True):
            inside = self.edge_inside_building_m.get(int(attrs["edge_id"]), 0.0)
            attrs["cost"] = float(attrs["length"]) + ROUTE_BUILDING_PENALTY * inside

        self._signal_xy = _xy_of(data.signals)
        self._signal_heading = np.array([s.heading for s in data.signals], dtype=np.float64)
        self._sign_xy = _xy_of(data.signs)
        self._signals_by_hop = self._by_hop(data.signals, at_exit=True)
        self._signs_by_hop = self._by_hop(data.signs, at_exit=False, kind="speed_limit")

        self.occupancy = self._build_occupancy(data)

    @staticmethod
    def _build_graph(data: MapData) -> nx.DiGraph:
        """oneway を尊重した有向グラフを作る。weight は "length"。"""
        graph = nx.DiGraph()
        for node in data.nodes:
            graph.add_node(node.id, x=node.x, y=node.y)

        for edge in data.edges:
            if edge.u == edge.v:
                continue
            graph.add_edge(edge.u, edge.v, length=edge.length, edge_id=edge.id)
            if not edge.oneway:
                graph.add_edge(edge.v, edge.u, length=edge.length, edge_id=edge.id)
        return graph

    def _build_reachable_nodes(self) -> np.ndarray:
        """互いに行き来できるノードの添字（最大の強連結成分）。"""
        if self.graph.number_of_nodes() == 0:
            return np.zeros(0, dtype=np.int64)
        try:
            largest = max(nx.strongly_connected_components(self.graph), key=len)
        except ValueError:
            return np.zeros(0, dtype=np.int64)
        return np.fromiter(sorted(largest), dtype=np.int64, count=len(largest))

    def _edge_building_overlap(self) -> dict[int, float]:
        """エッジの中心線が建物の中を通る長さ [m]。短い重なりは数えない。"""
        if self._building_tree is None or not self._edge_lines:
            return {}
        try:
            lines = np.empty(len(self._edge_lines), dtype=object)
            lines[:] = self._edge_lines
            polys = np.empty(len(self._building_polys), dtype=object)
            polys[:] = self._building_polys
            pairs = self._building_tree.query(lines, predicate="intersects")
            if pairs.shape[1] == 0:
                return {}
            parts = shapely.intersection(lines[pairs[0]], polys[pairs[1]])
            inside = np.zeros(len(self._edge_lines), dtype=np.float64)
            np.add.at(inside, pairs[0], shapely.length(parts))
        except Exception:
            warn_once(
                "map.index.edge_building_overlap",
                "建物の中を通る道路の判定に失敗しました。経路は建物を避けずに作ります",
            )
            return {}
        return {
            int(self._edge_line_ids[i]): float(inside[i])
            for i in np.flatnonzero(inside >= BUILDING_OVERLAP_MIN_M)
        }

    def _build_occupancy(self, data: MapData) -> OccupancyGrid:
        """建物レイヤと道路レイヤをラスタ化した占有グリッドを作る。"""
        cell = float(config.GRID_CELL_SIZE)
        margin = float(config.GRID_MARGIN)

        min_x = data.bounds.min_x - margin
        min_y = data.bounds.min_y - margin
        max_x = data.bounds.max_x + margin
        max_y = data.bounds.max_y + margin

        width = int(math.ceil((max_x - min_x) / cell)) + 1
        height = int(math.ceil((max_y - min_y) / cell)) + 1
        width = max(width, 1)
        height = max(height, 1)

        grid = OccupancyGrid(
            origin_x=min_x,
            origin_y=min_y,
            cell_size=cell,
            width=width,
            height=height,
            building=np.zeros((height, width), dtype=bool),
        )

        _rasterize_into(grid.building, self._building_polys, grid)

        return grid

    def is_intersection(self, node_id: int) -> bool:
        """3 方向以上に道がつながるノードか（交差点の入口で左右を確かめるのに使う）。"""
        return int(node_id) in self._junctions

    def branch_sides(self, node_id: int, arrive_heading: float) -> int:
        """`arrive_heading` で交差点に着いたとき、左（BRANCH_LEFT）・右（BRANCH_RIGHT）から車が入ってくる道があるか。"""
        node = int(node_id)
        if not self.graph.has_node(node):
            return 0
        sides = 0
        # 車が入ってくるのは、このノードへ向かう辺がある道だけ（出ていくだけの一方通行からは来ない）
        for pred in self.graph.predecessors(node):
            attrs = self.graph.get_edge_data(pred, node) or {}
            edge = self._edges_by_id.get(int(attrs.get("edge_id", -1)))
            if edge is None or len(edge.polyline) < 2:
                continue
            pts = [(float(px), float(py)) for px, py in edge.polyline]
            if int(edge.u) != node:
                pts.reverse()
            rel = _wrap(_direction(pts, at_end=False) - float(arrive_heading))
            if BRANCH_SIDE_MIN_RAD < rel < BRANCH_SIDE_MAX_RAD:
                sides |= BRANCH_LEFT
            elif -BRANCH_SIDE_MAX_RAD < rel < -BRANCH_SIDE_MIN_RAD:
                sides |= BRANCH_RIGHT
        return sides

    def nearest_node(self, x: float, y: float) -> int:
        """指定座標に最も近い道路ノード ID を返す。"""
        if self._node_xy.shape[0] == 0:
            return 0
        dx = self._node_xy[:, 0] - float(x)
        dy = self._node_xy[:, 1] - float(y)
        return int(np.argmin(dx * dx + dy * dy))

    def nearest_road_point(self, x: float, y: float) -> tuple[float, float, int, float]:
        """指定座標を最寄りの道路中心線上へスナップする。"""
        if self._edge_tree is None or not self._edge_lines:
            return float(x), float(y), -1, 0.0

        point = Point(float(x), float(y))
        indices = self._edge_tree.query_nearest(point)
        idx = int(np.atleast_1d(np.asarray(indices)).ravel()[0])

        line = self._edge_lines[idx]
        edge_id = self._edge_line_ids[idx]

        distance = float(line.project(point))
        snapped = line.interpolate(distance)

        heading = self._tangent_heading(line, distance)
        return float(snapped.x), float(snapped.y), int(edge_id), float(heading)

    @staticmethod
    def _tangent_heading(line: LineString, distance: float, eps: float = 0.5) -> float:
        """ポリライン上の距離 `distance` における接線方向 [rad]。"""
        total = float(line.length)
        if total <= 1e-9:
            return 0.0
        back = max(0.0, min(total, distance - eps))
        fore = max(0.0, min(total, distance + eps))
        if fore - back < 1e-9:
            back, fore = 0.0, min(total, eps)
        p0 = line.interpolate(back)
        p1 = line.interpolate(fore)
        return math.atan2(p1.y - p0.y, p1.x - p0.x)

    def shortest_path(
        self, src_node: int, dst_node: int, arrive_heading: float | None = None
    ) -> list[int] | None:
        """ノード ID 列で最短経路を返す。到達不能なら None。"""
        src = int(src_node)
        dst = int(dst_node)
        if src == dst:
            return [src] if self.graph.has_node(src) else None
        if arrive_heading is not None:
            # 出だしで折り返す辺（同じ道の U ターン・中央分離帯の切れ目での転回）だけを外して探す。
            #   行き止まりで折り返すしかないときは下で普通に探す
            heading = float(arrive_heading)

            def weight(u: int, v: int, attrs: dict) -> float | None:
                if u == src and self._turns_back(u, attrs, heading):
                    return None
                return attrs["cost"]

            try:
                return self._search(src, dst, weight)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
        try:
            return self._search(src, dst, "cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _search(self, src: int, dst: int, weight: Any) -> list[int]:
        """主成分の中どうしは A*、それ以外は双方向 Dijkstra で探す。"""
        mask = self._reachable_mask
        if 0 <= src < mask.size and 0 <= dst < mask.size and mask[src] and mask[dst]:
            return list(nx.astar_path(self.graph, src, dst, heuristic=self._heuristic, weight=weight))
        # 主成分の外へは届かないことがあり、片側から探す A* は届く範囲を全部なめる（金沢で 100ms 超）
        return list(nx.shortest_path(self.graph, src, dst, weight=weight))

    def _heuristic(self, a: int, b: int) -> float:
        """A* の見積もり。辺の重み（長さ + 建物の罰）は両端の直線距離より小さくならない。"""
        pa = self._node_pts[a]
        pb = self._node_pts[b]
        return ROUTE_HEURISTIC_SCALE * math.hypot(pa[0] - pb[0], pa[1] - pb[1])

    def _turns_back(self, node: int, attrs: dict, heading: float) -> bool:
        """向き `heading` で着いたノード `node` から、このエッジへ出ると折り返しになるか。"""
        edge = self._edges_by_id.get(int(attrs["edge_id"]))
        if edge is None or len(edge.polyline) < 2:
            return False
        pts = edge.polyline if int(edge.u) == int(node) else edge.polyline[::-1]
        return abs(_wrap(_direction(pts, at_end=False) - heading)) > ROUTE_MAX_START_TURN_RAD

    def route_polyline(
        self, node_path: Sequence[int], resample_m: float = 2.0
    ) -> list[tuple[float, float]]:
        """ノード列をエッジのポリラインへ展開し、等間隔にリサンプルして返す。"""
        path = [int(n) for n in node_path]
        if not path:
            return []
        if len(path) == 1:
            if 0 <= path[0] < self._node_xy.shape[0]:
                px, py = self._node_xy[path[0]]
                return [(float(px), float(py))]
            return []

        points: list[tuple[float, float]] = []
        for a, b in zip(path[:-1], path[1:]):
            attrs = self.graph.get_edge_data(a, b)
            if attrs is None:
                segment = [self._node_point(a), self._node_point(b)]
            else:
                edge = self._edges_by_id.get(int(attrs["edge_id"]))
                if edge is None:
                    segment = [self._node_point(a), self._node_point(b)]
                else:
                    segment = list(edge.polyline)
                    if edge.u != a:
                        segment.reverse()
            for p in segment:
                if points and _sq_dist(points[-1], p) <= 1e-12:
                    continue
                points.append((float(p[0]), float(p[1])))

        if len(points) < 2:
            return points
        return _resample_polyline(points, float(resample_m))

    def lane_route(
        self,
        node_path: Sequence[int],
        resample_m: float = 2.0,
        lead: RoadLead | None = None,
    ) -> tuple[list[tuple[float, float]], list[RouteLeg]]:
        """左側通行の車線に沿った走行経路と、それが通る辺の列を返す（道交法 17 条 4 項 / 34 条）。"""
        path = [int(n) for n in node_path]
        segments: list[RouteSegment] = []
        hops: list[tuple[int, int]] = []
        if lead is not None:
            for edge_id, pts, hop in zip(lead.edge_ids, lead.polylines, self._lead_hops(lead)):
                edge = self._edges_by_id.get(int(edge_id))
                if edge is not None and len(pts) >= 2:
                    segments.append(
                        RouteSegment(
                            edge=edge,
                            points=[(float(px), float(py)) for px, py in pts],
                            entry_offset=None if segments else lead.entry_offset,
                        )
                    )
                    hops.append(hop)
        elif len(path) < 2:
            return self._centerline_route(path, resample_m)

        for a, b in zip(path[:-1], path[1:]):
            attrs = self.graph.get_edge_data(a, b)
            if attrs is None:
                continue
            edge = self._edges_by_id.get(int(attrs["edge_id"]))
            if edge is None or len(edge.polyline) < 2:
                continue
            pts = [(float(px), float(py)) for px, py in edge.polyline]
            if edge.u != a:
                pts.reverse()
            segments.append(RouteSegment(edge=edge, points=pts))
            hops.append((a, b))

        if not segments:
            return self._centerline_route(path, resample_m)

        route, spans = build_lane_route(segments, float(resample_m))
        if len(route) < 2:
            return self._centerline_route(path, resample_m)
        legs = [
            RouteLeg(
                edge_id=int(seg.edge.id),
                entry_node=int(entry),
                exit_node=int(exit_),
                start_arc=float(span[0]),
                end_arc=float(span[1]),
            )
            for seg, (entry, exit_), span in zip(segments, hops, spans)
            if span is not None
        ]
        return route, legs

    def _lead_hops(self, lead: RoadLead) -> list[tuple[int, int]]:
        """道なりの出だしが通る辺ごとの (入口のノード, 出口のノード)。"""
        hops: list[tuple[int, int]] = []
        node = int(lead.exit_node)
        for edge_id in reversed(lead.edge_ids):
            edge = self._edges_by_id.get(int(edge_id))
            if edge is None:
                hops.append((-1, -1))
                continue
            entry = int(edge.u) if int(edge.v) == node else int(edge.v)
            hops.append((entry, node))
            node = entry
        hops.reverse()
        return hops

    def _centerline_route(
        self, path: Sequence[int], resample_m: float
    ) -> tuple[list[tuple[float, float]], list[RouteLeg]]:
        """車線を引けないときの経路（中心線をたどる）と、それが通る辺の列。"""
        legs: list[RouteLeg] = []
        arc = 0.0
        for a, b in zip(path[:-1], path[1:]):
            attrs = self.graph.get_edge_data(a, b)
            edge = None if attrs is None else self._edges_by_id.get(int(attrs["edge_id"]))
            if edge is None:
                arc += math.dist(self._node_point(a), self._node_point(b))
                continue
            legs.append(RouteLeg(int(edge.id), int(a), int(b), arc, arc + float(edge.length)))
            arc += float(edge.length)
        return self.route_polyline(path, resample_m), legs

    def road_lead(
        self, x: float, y: float, heading: float | None, min_length_m: float = 0.0
    ) -> RoadLead | None:
        """いま走っている道路を進行方向へたどり、最初に曲がれるノードまでの道のりを返す。"""
        start = self._current_edge(float(x), float(y), heading)
        if start is None:
            return None
        edge, points, exit_node, facing, lateral = start
        edge_ids = [int(edge.id)]
        polylines = [points]
        length = _polyline_length(points)
        seen = {int(edge.id)}
        # 速度が出ていると、すぐ先の交差点では曲がりきれない。まっすぐ抜けられる道を先へたどる
        while length < float(min_length_m) and len(edge_ids) <= LEAD_MAX_HOPS:
            nxt = self._straight_on(exit_node, facing, seen)
            if nxt is None:
                break
            edge, points, facing = nxt
            exit_node = int(edge.v) if int(edge.u) == exit_node else int(edge.u)
            seen.add(int(edge.id))
            edge_ids.append(int(edge.id))
            polylines.append(points)
            length += float(edge.length)
        return RoadLead(
            exit_node=int(exit_node),
            exit_heading=float(facing),
            edge_ids=edge_ids,
            polylines=polylines,
            length=float(length),
            entry_offset=float(lateral),
        )

    def _current_edge(
        self, x: float, y: float, heading: float | None
    ) -> tuple[MapEdge, list[tuple[float, float]], int, float, float] | None:
        """車の位置と向きに最も合う道路を選び、車の真横から先の中心線・出口・向き・横位置を返す。"""
        if self._edge_tree is None or not self._edge_lines:
            return None
        point = Point(x, y)
        idx = np.atleast_1d(
            np.asarray(
                self._edge_tree.query(point, predicate="dwithin", distance=LEAD_SEARCH_RADIUS_M)
            )
        ).ravel()
        if idx.size == 0:
            idx = np.atleast_1d(np.asarray(self._edge_tree.query_nearest(point))).ravel()

        best: tuple[float, MapEdge, LineString, float, bool] | None = None
        for i in idx:
            edge = self._edges_by_id.get(self._edge_line_ids[int(i)])
            if edge is None or edge.u == edge.v:
                continue
            line = self._edge_lines[int(i)]
            along = float(line.project(point))
            tangent = self._tangent_heading(line, along)
            gap = float(line.distance(point))
            # 道路の幅の内側にいれば、中心線からの距離はほとんど問わない（広い道路の外側の車線）
            outside = max(0.0, gap - float(edge.width) / 2.0) + LEAD_CENTER_WEIGHT * gap
            for forward in (True,) if edge.oneway else (True, False):
                facing = tangent if forward else tangent + math.pi
                turn = 0.0 if heading is None else abs(_wrap(float(heading) - facing))
                score = outside + LEAD_ANGLE_WEIGHT_M * turn
                if best is None or score < best[0]:
                    best = (score, edge, line, along, forward)
        if best is None:
            return None

        _score, edge, line, along, forward = best
        foot = line.interpolate(along)
        facing = self._tangent_heading(line, along) + (0.0 if forward else math.pi)
        lateral = -(x - foot.x) * math.sin(facing) + (y - foot.y) * math.cos(facing)
        half = float(edge.width) / 2.0
        lateral = min(half, max(-half, lateral))
        full = [(float(px), float(py)) for px, py in edge.polyline]
        if forward:
            points = _cut(line, along, float(line.length))
            return edge, points, int(edge.v), _direction(full, at_end=True), lateral
        points = _cut(line, 0.0, along)
        points.reverse()
        full.reverse()
        return edge, points, int(edge.u), _direction(full, at_end=True), lateral

    def _straight_on(
        self, node: int, facing: float, seen: set[int]
    ) -> tuple[MapEdge, list[tuple[float, float]], float] | None:
        """ノードをまっすぐ抜けた先のエッジ。無ければ None（そこで曲がるしかない）。"""
        best: tuple[float, MapEdge, list[tuple[float, float]]] | None = None
        for _n, succ, attrs in self.graph.out_edges(node, data=True):
            edge = self._edges_by_id.get(int(attrs["edge_id"]))
            if edge is None or int(edge.id) in seen or len(edge.polyline) < 2:
                continue
            if not (0 <= int(succ) < self._reachable_mask.size and self._reachable_mask[int(succ)]):
                continue
            pts = [(float(px), float(py)) for px, py in edge.polyline]
            if int(edge.u) != int(node):
                pts.reverse()
            turn = abs(_wrap(_direction(pts, at_end=False) - facing))
            if turn <= LEAD_STRAIGHT_RAD and (best is None or turn < best[0]):
                best = (turn, edge, pts)
        if best is None:
            return None
        return best[1], best[2], _direction(best[2], at_end=True)

    def signals_on_route(
        self, points: Sequence[tuple[float, float]], legs: Sequence[RouteLeg]
    ) -> list[tuple[float, int]]:
        """経路が通る辺の出口に立つ信号を (弧長, MapData.signals の添字) で返す。"""
        if len(points) < 2 or not legs or not self._signals_by_hop:
            return []

        pts = np.asarray(points, dtype=np.float64)
        cum = _cumulative_arc(pts)
        cand, _owner, arcs, route_h = _features_on_legs(
            pts, cum, legs, self._signals_by_hop, self._signal_xy
        )
        sh = self._signal_heading[cand]
        turn = np.abs(np.arctan2(np.sin(sh - route_h), np.cos(sh - route_h)))
        # 停止線が上流の交差点の中（曲がっている途中）に落ちた灯器は拾わない（#48）
        hits = (turn <= SIGNAL_HEADING_TOLERANCE_RAD) & (arcs >= 0.0) & (arcs <= float(cum[-1]))
        out = [(float(arcs[i]), int(cand[i])) for i in np.flatnonzero(hits)]
        out.sort(key=lambda item: item[0])

        merged: list[tuple[float, int]] = []
        for arc, idx in out:
            if merged and arc - merged[-1][0] < SIGNAL_MERGE_M:
                continue
            merged.append((arc, idx))
        return merged

    def speed_limits_on_route(
        self, points: Sequence[tuple[float, float]], legs: Sequence[RouteLeg]
    ) -> list[tuple[float, float]]:
        """経路が通る辺の規制速度を (弧長 [m], 規制速度 [m/s]) の区切りで返す。"""
        if len(points) < 2 or not legs:
            return []

        pts = np.asarray(points, dtype=np.float64)
        cum = _cumulative_arc(pts)

        # 値は経路が通る辺のもの。標識は「どこから変わるか」にだけ使う
        #   （同じ 2 ノードを結ぶ辺が 2 本あると、標識は通らない方の辺の値を持つことがある）
        _cand, owner, arcs, _route_h = _features_on_legs(
            pts, cum, legs, self._signs_by_hop, self._sign_xy
        )
        change_at = {int(k): float(a) for k, a in zip(owner, arcs) if a >= 0.0}
        breaks: list[tuple[float, float]] = []
        for k, leg in enumerate(legs):
            edge = self._edges_by_id.get(int(leg.edge_id))
            if edge is None:
                continue
            arc = 0.0 if k == 0 else change_at.get(k, float(leg.start_arc))
            breaks.append((arc, float(edge.speed_limit)))
        breaks.sort(key=lambda item: item[0])

        if not breaks:
            return []

        out: list[tuple[float, float]] = [breaks[0]]
        for arc, limit in breaks[1:]:
            if abs(limit - out[-1][1]) < 1e-6:
                continue
            out.append((arc, limit))
        return out

    def _by_hop(self, items: Sequence, at_exit: bool, kind: str | None = None) -> dict[tuple[int, int], list[int]]:
        """地物を、それが立つ辺の (入口のノード, 出口のノード) で引けるようにする。"""
        table: dict[tuple[int, int], list[int]] = {}
        for i, item in enumerate(items):
            if kind is not None and item.kind != kind:
                continue
            edge = self._edges_by_id.get(int(item.edge_id))
            node = int(item.node_id)
            if edge is None or node not in (int(edge.u), int(edge.v)):
                continue
            other = int(edge.u) if int(edge.v) == node else int(edge.v)
            # 信号は辺の出口（交差点の手前）、標識は辺の入口（交差点を出た先）に立つ
            key = (other, node) if at_exit else (node, other)
            table.setdefault(key, []).append(i)
        return table

    def _node_point(self, node_id: int) -> tuple[float, float]:
        if 0 <= node_id < self._node_xy.shape[0]:
            px, py = self._node_xy[node_id]
            return float(px), float(py)
        return 0.0, 0.0

    def random_node_pair(
        self,
        rng: np.random.Generator,
        min_distance_m: float = 150.0,
        max_distance_m: float = float("inf"),
    ) -> tuple[int, int]:
        """経路が存在し、指定の距離の範囲にある出発／目的ノードの組を返す。"""
        # 到達できるかは読み込み時の強連結成分で決める（ここで shortest_path を
        # 呼ぶと、再スポーン 1 回あたり dijkstra が 2 回になる）
        pool = self._reachable
        n = int(self._node_xy.shape[0])
        if pool.size >= 2:
            xy = self._node_xy[pool]
            lo2 = float(min_distance_m) ** 2
            hi = float(max_distance_m)
            src_i = int(rng.integers(0, pool.size))
            dx = xy[:, 0] - xy[src_i, 0]
            dy = xy[:, 1] - xy[src_i, 1]
            d2 = dx * dx + dy * dy
            ok = np.flatnonzero((d2 >= lo2) & (d2 <= hi * hi)) if hi < float("inf") else (
                np.flatnonzero(d2 >= lo2)
            )
            if ok.size == 0:
                # 上限の内側に候補が無い（狭いマップ・辺縁のノード）。まず下限だけで探す
                ok = np.flatnonzero(d2 >= lo2)
            if ok.size == 0:
                dst_i = int(rng.integers(0, pool.size))
                if dst_i == src_i:
                    dst_i = (src_i + 1) % int(pool.size)
            else:
                dst_i = int(ok[int(rng.integers(0, ok.size))])
            return int(pool[src_i]), int(pool[dst_i])

        if n == 0 or n == 1:
            return 0, 0
        return 0, min(1, n - 1)

    def raycast(
        self,
        origin_x: np.ndarray,
        origin_y: np.ndarray,
        angles: np.ndarray,
        max_distance: float,
        step: float = 1.0,
    ) -> np.ndarray:
        """占有グリッドの building レイヤを step 刻みでサンプリングして距離を測る。"""
        ox_arr = np.asarray(origin_x, dtype=np.float64).reshape(-1)
        oy_arr = np.asarray(origin_y, dtype=np.float64).reshape(-1)
        ang = np.asarray(angles, dtype=np.float64)
        if ang.ndim == 1:
            ang = ang.reshape(1, -1)

        n_rays = ang.shape[0]
        n_dirs = ang.shape[1]
        if n_rays == 0 or n_dirs == 0:
            return np.full(ang.shape, float(max_distance), dtype=np.float32)

        max_d = float(max_distance)
        step_m = max(float(step), 1e-3)

        samples = np.arange(step_m, max_d + step_m * 0.5, step_m, dtype=np.float64)
        if samples.size == 0:
            samples = np.array([max_d], dtype=np.float64)

        cos_a = np.cos(ang)[:, :, None]
        sin_a = np.sin(ang)[:, :, None]
        t = samples[None, None, :]

        xs = ox_arr[:, None, None] + cos_a * t
        ys = oy_arr[:, None, None] + sin_a * t

        hits = self.occupancy.sample_building(xs, ys)

        any_hit = hits.any(axis=2)
        first = np.argmax(hits, axis=2)
        distances = np.where(any_hit, samples[np.clip(first, 0, samples.size - 1)], max_d)
        return np.minimum(distances, max_d).astype(np.float32)

    def collides_with_buildings(
        self, corners: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """複数車両ぶんの外接矩形をまとめて判定する。shape (N,) bool。"""
        n = int(corners.shape[0])
        out = np.zeros(n, dtype=bool)
        if self._building_tree is None:
            return out
        idx = np.flatnonzero(np.asarray(mask, dtype=bool))
        if idx.size == 0:
            return out
        try:
            polys = shapely.polygons(np.asarray(corners[idx], dtype=np.float64))
            pairs = self._building_tree.query(polys, predicate="intersects")
            found = np.asarray(pairs)
            if found.size:
                out[idx[np.unique(found[0])]] = True
        except Exception:
            if not self._batch_collision_failed:
                self._batch_collision_failed = True
                logger.exception(
                    "建物とのバッチ衝突判定に失敗しました。衝突なしとして続行します"
                )
        return out


def build_map_index(data: MapData) -> MapIndexImpl:
    """`MapData` から実行時インデックスを組み立てる。"""
    return MapIndexImpl(data)


def _wrap(angle: float) -> float:
    """角度を (-pi, pi] に畳む。"""
    return math.atan2(math.sin(angle), math.cos(angle))


def _polyline_length(points: Sequence[tuple[float, float]]) -> float:
    return float(sum(math.dist(a, b) for a, b in zip(points[:-1], points[1:])))


def _cut(line: LineString, start: float, end: float) -> list[tuple[float, float]]:
    """ポリラインの弧長 start..end の部分を点列で返す。"""
    part = substring(line, start, end)
    if part.geom_type == "Point":
        return [(float(part.x), float(part.y))]
    return [(float(px), float(py)) for px, py in part.coords]


def _direction(points: Sequence[tuple[float, float]], at_end: bool) -> float:
    """点列の始点（または終点）での進行方位 [rad]。`LEAD_DIRECTION_SPAN_M` ぶん離れた点で測る。"""
    pts = list(reversed(points)) if at_end else list(points)
    origin = pts[0]
    far = pts[-1]
    walked = 0.0
    # 頂点までで止めると、頂点が疎な道では何十 m も先との弦になり、端の曲がりが消える
    for a, b in zip(pts[:-1], pts[1:]):
        step = math.dist(a, b)
        if walked + step >= LEAD_DIRECTION_SPAN_M:
            t = (LEAD_DIRECTION_SPAN_M - walked) / step
            far = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            break
        walked += step
    if at_end:
        return math.atan2(origin[1] - far[1], origin[0] - far[0])
    return math.atan2(far[1] - origin[1], far[0] - origin[0])


def _sq_dist(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return dx * dx + dy * dy


def _xy_of(items: Sequence) -> np.ndarray:
    """`x` / `y` を持つ地物の列から座標 (K, 2) を作る。"""
    if len(items) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array([[it.x, it.y] for it in items], dtype=np.float64)


def _cumulative_arc(pts: np.ndarray) -> np.ndarray:
    """経路点ごとの、始点からの弧長。"""
    seg = np.diff(pts, axis=0)
    return np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])


def _features_on_legs(
    pts: np.ndarray,
    cum: np.ndarray,
    legs: Sequence[RouteLeg],
    table: dict[tuple[int, int], list[int]],
    xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """経路が通る辺に立つ地物の添字と、それが立つ辺（`legs` の添字）・弧長・そこでの経路の向き。"""
    cand: list[int] = []
    owner: list[int] = []
    lo: list[float] = []
    hi: list[float] = []
    n = len(legs)
    for k, leg in enumerate(legs):
        found = table.get((int(leg.entry_node), int(leg.exit_node)))
        if not found:
            continue
        # その辺の車線と、前後の交差点のつなぎの中だけで探す
        a = float(legs[k - 1].end_arc) if k > 0 else -math.inf
        b = float(legs[k + 1].start_arc) if k + 1 < n else math.inf
        for i in found:
            cand.append(int(i))
            owner.append(k)
            lo.append(a)
            hi.append(b)
    if not cand:
        empty = np.zeros(0, dtype=np.float64)
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), empty, empty
    idx = np.asarray(cand, dtype=np.int64)
    arcs, heading = _project_in_windows(
        pts, cum, xy[idx], np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    )
    return idx, np.asarray(owner, dtype=np.int64), arcs, heading


def _project_in_windows(
    pts: np.ndarray, cum: np.ndarray, xy: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """地物を、弧長 lo..hi にある経路の線分へ射影した弧長とその線分の向き。始点より後ろは負、終点より先は全長より大きい。"""
    segs = pts.shape[0] - 1
    first = np.clip(np.searchsorted(cum, lo, side="right") - 1, 0, segs - 1)
    last = np.clip(np.searchsorted(cum, hi, side="right") - 1, 0, segs - 1)
    last = np.maximum(last, first)
    width = int((last - first).max()) + 1
    j = first[:, None] + np.arange(width)[None, :]
    valid = j <= last[:, None]
    j = np.minimum(j, segs - 1)

    a = pts[j]
    ab = pts[j + 1] - a
    len2 = np.maximum((ab * ab).sum(axis=-1), 1e-12)
    t = ((xy[:, None, :] - a) * ab).sum(axis=-1) / len2
    # 始点より後ろ・終点より先へは線分を延ばして測る（端の点に張り付かせない）
    t = np.clip(t, np.where(j == 0, -np.inf, 0.0), np.where(j == segs - 1, np.inf, 1.0))
    d2 = ((a + ab * t[..., None] - xy[:, None, :]) ** 2).sum(axis=-1)
    d2[~valid] = np.inf
    best = np.argmin(d2, axis=1)
    rows = np.arange(xy.shape[0])
    arcs = cum[j[rows, best]] + t[rows, best] * np.sqrt(len2[rows, best])
    heading = np.arctan2(ab[rows, best, 1], ab[rows, best, 0])

    total = float(cum[-1])
    arcs = np.where((arcs < 0.0) & (arcs > -ROUTE_END_EPS_M), 0.0, arcs)
    arcs = np.where((arcs > total) & (arcs < total + ROUTE_END_EPS_M), total, arcs)
    return arcs, heading


def _resample_polyline(
    points: Sequence[tuple[float, float]], spacing: float
) -> list[tuple[float, float]]:
    """ポリラインを弧長方向に等間隔リサンプルする。終点は必ず含める。"""
    arr = np.asarray(points, dtype=np.float64)
    if arr.shape[0] < 2:
        return [(float(p[0]), float(p[1])) for p in points]

    seg = np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))
    cumulative = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cumulative[-1])
    if total <= 1e-9:
        return [(float(arr[0, 0]), float(arr[0, 1]))]

    spacing = max(float(spacing), 1e-3)
    stations = np.arange(0.0, total, spacing, dtype=np.float64)
    if stations.size == 0 or total - stations[-1] > 1e-6:
        stations = np.concatenate((stations, [total]))

    xs = np.interp(stations, cumulative, arr[:, 0])
    ys = np.interp(stations, cumulative, arr[:, 1])

    out: list[tuple[float, float]] = []
    for px, py in zip(xs, ys):
        p = (float(px), float(py))
        if out and _sq_dist(out[-1], p) <= 1e-12:
            continue
        out.append(p)
    return out


def _rasterize_into(layer: np.ndarray, polygons: Sequence, grid: OccupancyGrid) -> None:
    """ポリゴン群をブール配列へ焼き込む。"""
    if not polygons:
        return

    cell = grid.cell_size
    ox_g = grid.origin_x
    oy_g = grid.origin_y

    for poly in polygons:
        if poly is None or poly.is_empty:
            continue
        min_x, min_y, max_x, max_y = poly.bounds

        c0 = int(math.floor((min_x - ox_g) / cell))
        c1 = int(math.ceil((max_x - ox_g) / cell))
        r0 = int(math.floor((min_y - oy_g) / cell))
        r1 = int(math.ceil((max_y - oy_g) / cell))

        c0 = max(c0, 0)
        r0 = max(r0, 0)
        c1 = min(c1, grid.width - 1)
        r1 = min(r1, grid.height - 1)
        if c1 < c0 or r1 < r0:
            continue

        xs = ox_g + np.arange(c0, c1 + 1, dtype=np.float64) * cell
        ys = oy_g + np.arange(r0, r1 + 1, dtype=np.float64) * cell
        gx, gy = np.meshgrid(xs, ys)

        shapely.prepare(poly)
        mask = shapely.contains_xy(poly, gx, gy)
        if mask.any():
            layer[r0 : r1 + 1, c0 : c1 + 1] |= mask
