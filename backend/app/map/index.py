"""`MapData` から実行時インデックス（`contracts.MapIndex`）を組み立てる。"""

from __future__ import annotations

import math
from typing import Sequence

import networkx as nx
import logging

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon

from app import config
from app.contracts import SIGNAL_MERGE_M, MapData, MapEdge, OccupancyGrid

__all__ = ["MapIndexImpl", "build_map_index"]

from app.map.lanes import RouteSegment, build_lane_route

logger = logging.getLogger("autoware_sim")

_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """同じ失敗を初回だけログに残す（code_review B-15 / E-06 / E-10）。"""
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.exception(message)


class MapIndexImpl:
    """`contracts.MapIndex` プロトコルの実装。"""

    def __init__(self, data: MapData) -> None:
        self.data = data
        self._batch_collision_failed = False

        nodes = sorted(data.nodes, key=lambda n: n.id)
        self._node_xy = np.array([[n.x, n.y] for n in nodes], dtype=np.float64)
        if self._node_xy.size == 0:
            self._node_xy = np.zeros((0, 2), dtype=np.float64)

        self._edges_by_id: dict[int, MapEdge] = {e.id: e for e in data.edges}

        self.graph = self._build_graph(data)

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

        self._signal_xy, self._signal_heading, self._signal_tree = _point_layer(
            data.signals
        )
        self._sign_xy, self._sign_heading, self._sign_tree = _point_layer(data.signs)
        self._sign_limit = np.array(
            [sn.speed_limit for sn in data.signs], dtype=np.float64
        )

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

    def _iter_edge_lines(self):
        for edge_id, line in zip(self._edge_line_ids, self._edge_lines):
            edge = self._edges_by_id.get(edge_id)
            if edge is not None:
                yield edge, line

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

    def shortest_path(self, src_node: int, dst_node: int) -> list[int] | None:
        """ノード ID 列で最短経路を返す。到達不能なら None。"""
        src = int(src_node)
        dst = int(dst_node)
        if src == dst:
            return [src] if self.graph.has_node(src) else None
        try:
            return list(nx.shortest_path(self.graph, src, dst, weight="length"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

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

    def lane_route_polyline(
        self, node_path: Sequence[int], resample_m: float = 2.0
    ) -> list[tuple[float, float]]:
        """左側通行の車線に沿った走行経路を返す（道交法 17 条 4 項 / 34 条）。"""
        path = [int(n) for n in node_path]
        if len(path) < 2:
            return self.route_polyline(path, resample_m)

        segments: list[RouteSegment] = []
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

        if not segments:
            return self.route_polyline(path, resample_m)

        route = build_lane_route(segments, float(resample_m))
        if len(route) < 2:
            return self.route_polyline(path, resample_m)
        return route

    def signals_on_route(
        self,
        points: Sequence[tuple[float, float]],
        max_lateral: float = 11.0,
        max_heading_diff: float = math.radians(35.0),
    ) -> list[tuple[float, int]]:
        """経路が通過する信号を (弧長, MapData.signals の添字) で返す。"""
        if self._signal_tree is None or len(points) < 2:
            return []

        pts = np.asarray(points, dtype=np.float64)
        cum, tang = _arc_and_tangent(pts)

        cand = self._near_route(self._signal_tree, pts, float(max_lateral))
        if cand.size == 0:
            return []

        nearest, dist = _nearest_on_route(pts, self._signal_xy[cand])
        sh = self._signal_heading[cand]
        route_h = tang[nearest]
        diff = np.abs(np.arctan2(np.sin(sh - route_h), np.cos(sh - route_h)))

        hits = (dist <= float(max_lateral)) & (diff <= float(max_heading_diff))
        out = [
            (float(cum[nearest[i]]), int(cand[i])) for i in np.flatnonzero(hits)
        ]
        out.sort(key=lambda item: item[0])

        # ★ 同じ交差点の灯器が二重に拾われることがある（近接した別ノードに灯器が
        #   立っている）。前方の信号すべてに停止線を引くので、まとめないと同じ
        #   交差点で二度止まる。現示は `MapSignal.phase_key` が揃えているので、
        #   どちらが残っても色は同じ。
        merged: list[tuple[float, int]] = []
        for arc, idx in out:
            if merged and arc - merged[-1][0] < SIGNAL_MERGE_M:
                continue
            merged.append((arc, idx))
        return merged

    def speed_limits_on_route(
        self,
        points: Sequence[tuple[float, float]],
        max_lateral: float = 8.0,
        max_heading_diff: float = math.radians(35.0),
    ) -> list[tuple[float, float]]:
        """経路に適用される規制速度を (弧長 [m], 規制速度 [m/s]) の区切りで返す。"""
        if len(points) < 2:
            return []

        pts = np.asarray(points, dtype=np.float64)
        cum, tang = _arc_and_tangent(pts)

        start_limit = self._edge_speed_limit_at(float(pts[0, 0]), float(pts[0, 1]))
        breaks: list[tuple[float, float]] = []
        if start_limit is not None:
            breaks.append((0.0, float(start_limit)))

        cand = self._near_route(self._sign_tree, pts, float(max_lateral))
        if cand.size:
            nearest, dist = _nearest_on_route(pts, self._sign_xy[cand])
            sh = self._sign_heading[cand]
            route_h = tang[nearest]
            diff = np.abs(np.arctan2(np.sin(sh - route_h), np.cos(sh - route_h)))

            hits = (dist <= float(max_lateral)) & (diff <= float(max_heading_diff))
            found = [
                (float(cum[nearest[i]]), float(self._sign_limit[cand[i]]))
                for i in np.flatnonzero(hits)
            ]
            found.sort(key=lambda item: item[0])
            breaks.extend(found)

        if not breaks:
            return []

        out: list[tuple[float, float]] = [breaks[0]]
        for arc, limit in breaks[1:]:
            if abs(limit - out[-1][1]) < 1e-6:
                continue
            out.append((arc, limit))
        return out

    @staticmethod
    def _near_route(
        tree: "shapely.STRtree | None", pts: np.ndarray, max_lateral: float
    ) -> np.ndarray:
        """経路の周り `max_lateral` [m] にある点の添字を返す。"""
        if tree is None or pts.shape[0] < 2:
            return np.zeros(0, dtype=np.int64)
        try:
            line = LineString(pts)
            found = tree.query(line, predicate="dwithin", distance=float(max_lateral))
        except Exception:
            logger.exception("経路の近傍検索に失敗しました。全件を候補にします")
            return np.arange(int(tree.geometries.size), dtype=np.int64)
        return np.sort(np.asarray(found, dtype=np.int64).reshape(-1))

    def _edge_speed_limit_at(self, x: float, y: float) -> float | None:
        """指定座標に最も近い道路の規制速度 [m/s]。道路が引けなければ None。"""
        try:
            _sx, _sy, edge_id, _heading = self.nearest_road_point(x, y)
        except Exception:
            _warn_once(
                "edge_speed_limit",
                "規制速度の引き当てに失敗しました。速度不明として扱うため、"
                "その区間の速度超過は計上されません（初回のみ記録）",
            )
            return None
        edge = self._edges_by_id.get(int(edge_id))
        return float(edge.speed_limit) if edge is not None else None

    def _node_point(self, node_id: int) -> tuple[float, float]:
        if 0 <= node_id < self._node_xy.shape[0]:
            px, py = self._node_xy[node_id]
            return float(px), float(py)
        return 0.0, 0.0

    def random_node_pair(
        self, rng: np.random.Generator, min_distance_m: float = 150.0
    ) -> tuple[int, int]:
        """経路が存在し、十分離れた出発／目的ノードの組を返す。"""
        n = self._node_xy.shape[0]
        if n == 0:
            return 0, 0
        if n == 1:
            return 0, 0

        threshold = float(min_distance_m)

        for _ in range(50):
            src = int(rng.integers(0, n))
            dx = self._node_xy[:, 0] - self._node_xy[src, 0]
            dy = self._node_xy[:, 1] - self._node_xy[src, 1]
            far = np.flatnonzero((dx * dx + dy * dy) >= threshold * threshold)
            if far.size == 0:
                continue
            dst = int(far[int(rng.integers(0, far.size))])
            if dst == src:
                continue
            if self.shortest_path(src, dst) is not None:
                return src, dst

        for _ in range(min(n, 20)):
            src = int(rng.integers(0, n))
            if not self.graph.has_node(src):
                continue
            lengths = nx.single_source_dijkstra_path_length(self.graph, src, weight="length")
            lengths.pop(src, None)
            if not lengths:
                continue
            dst = int(max(lengths, key=lengths.__getitem__))
            return src, dst

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

    def collides_with_building(self, corners: np.ndarray) -> bool:
        """車両の外接矩形（shape (4, 2)）が建物と重なるかを厳密に判定する。"""
        if self._building_tree is None:
            return False
        pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 3:
            return False
        try:
            rect = Polygon(pts)
        except Exception:
            _warn_once(
                "collision_polygon",
                "車両の外接矩形を作れませんでした。衝突なしとして扱います（初回のみ記録）",
            )
            return False
        if rect.is_empty:
            return False
        if not rect.is_valid:
            _warn_once(
                "collision_polygon_invalid",
                "車両の外接矩形が不正です（角に NaN / inf が入った可能性）。"
                "衝突判定の結果は保証されません（初回のみ記録）",
            )
            return False
        found = self._building_tree.query(rect, predicate="intersects")
        return int(np.asarray(found).size) > 0


def build_map_index(data: MapData) -> MapIndexImpl:
    """`MapData` から実行時インデックスを組み立てる。"""
    return MapIndexImpl(data)


def _sq_dist(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return dx * dx + dy * dy


def _point_layer(items: Sequence) -> tuple[np.ndarray, np.ndarray, "shapely.STRtree | None"]:
    """`x` / `y` / `heading` を持つ地物の列から、座標・方位・STRtree を作る。"""
    n = len(items)
    if n == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            None,
        )
    xy = np.array([[it.x, it.y] for it in items], dtype=np.float64)
    heading = np.array([it.heading for it in items], dtype=np.float64)
    return xy, heading, shapely.STRtree(shapely.points(xy))


_NEAREST_CHUNK_BYTES = 8 << 20


def _arc_and_tangent(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """経路点列から「始点からの弧長」と「各点での進行方位」を返す。"""
    seg = np.diff(pts, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    tang = np.empty(pts.shape[0], dtype=np.float64)
    tang[:-1] = np.arctan2(seg[:, 1], seg[:, 0])
    tang[-1] = tang[-2]
    return cum, tang


def _nearest_on_route(pts: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """各地物 `xy` (K, 2) に最も近い経路点の添字と、その距離を返す。"""
    k = int(xy.shape[0])
    nearest = np.empty(k, dtype=np.int64)
    dist = np.empty(k, dtype=np.float64)
    if k == 0:
        return nearest, dist

    p = int(pts.shape[0])
    chunk = max(1, _NEAREST_CHUNK_BYTES // (8 * max(p, 1)))
    route_x = pts[None, :, 0]
    route_y = pts[None, :, 1]
    for lo in range(0, k, chunk):
        hi = min(lo + chunk, k)
        d2 = (route_x - xy[lo:hi, 0:1]) ** 2 + (route_y - xy[lo:hi, 1:2]) ** 2
        idx = np.argmin(d2, axis=1)
        nearest[lo:hi] = idx
        dist[lo:hi] = np.sqrt(d2[np.arange(hi - lo), idx])
    return nearest, dist


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
