"""`MapData` から実行時インデックス（`contracts.MapIndex`）を組み立てる。

構築するもの:
    - 経路探索用の `networkx.DiGraph`（oneway を尊重した有向グラフ）
    - ノード座標の numpy 配列（最近傍ノード探索）
    - エッジ中心線の STRtree（道路へのスナップ）
    - 建物ポリゴンの STRtree（厳密な衝突判定）
    - 占有グリッド `OccupancyGrid`（建物レイヤのみ。レイキャスト用）

いずれも「構築は 1 回だけ・参照は毎ステップ」という使われ方なので、
参照側（`raycast` / `collides_with_building`）は完全ベクトル化して軽く保つ。
"""

from __future__ import annotations

import math
from typing import Sequence

import networkx as nx
import logging

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon

from app import config
from app.contracts import MapData, MapEdge, OccupancyGrid

__all__ = ["MapIndexImpl", "build_map_index"]

from app.map.lanes import RouteSegment, build_lane_route


logger = logging.getLogger("autoware_sim")


class MapIndexImpl:
    """`contracts.MapIndex` プロトコルの実装。

    `build_map_index()` から生成する。外から直接コンストラクタを呼ばないこと。
    """

    def __init__(self, data: MapData) -> None:
        self.data = data
        # バッチ衝突判定の失敗を 1 度だけログに出すためのフラグ（B-15）
        self._batch_collision_failed = False

        # --- ノード座標 -------------------------------------------------
        # ノード ID は loader で 0..N-1 の連番に振り直されているので、
        # 配列添字 == ノード ID になる。念のため ID 順に並べ替えて保証する。
        nodes = sorted(data.nodes, key=lambda n: n.id)
        self._node_xy = np.array([[n.x, n.y] for n in nodes], dtype=np.float64)
        if self._node_xy.size == 0:
            self._node_xy = np.zeros((0, 2), dtype=np.float64)

        # --- エッジの索引 -----------------------------------------------
        self._edges_by_id: dict[int, MapEdge] = {e.id: e for e in data.edges}

        # --- 経路探索グラフ ---------------------------------------------
        self.graph = self._build_graph(data)

        # --- エッジ中心線の STRtree -------------------------------------
        self._edge_lines: list[LineString] = []
        self._edge_line_ids: list[int] = []
        for e in data.edges:
            if len(e.polyline) < 2:
                continue
            self._edge_lines.append(LineString(e.polyline))
            self._edge_line_ids.append(e.id)
        self._edge_tree = shapely.STRtree(self._edge_lines) if self._edge_lines else None

        # --- 建物ポリゴンの STRtree -------------------------------------
        self._building_polys: list[Polygon] = []
        for b in data.buildings:
            if len(b.outline) < 3:
                continue
            poly = Polygon(b.outline)
            if not poly.is_valid:
                # 自己交差した輪郭は buffer(0) で救えることが多い
                poly = poly.buffer(0)
                if poly.is_empty or poly.geom_type not in ("Polygon", "MultiPolygon"):
                    continue
            self._building_polys.append(poly)
        self._building_tree = (
            shapely.STRtree(self._building_polys) if self._building_polys else None
        )

        # --- 占有グリッド -----------------------------------------------
        self.occupancy = self._build_occupancy(data)

    # ------------------------------------------------------------------
    # 構築
    # ------------------------------------------------------------------

    @staticmethod
    def _build_graph(data: MapData) -> nx.DiGraph:
        """oneway を尊重した有向グラフを作る。weight は "length"。"""
        graph = nx.DiGraph()
        for node in data.nodes:
            graph.add_node(node.id, x=node.x, y=node.y)

        for edge in data.edges:
            if edge.u == edge.v:
                # 自己ループ（ロータリー等）は経路に現れないので追加しない
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

        # 建物レイヤ: ポリゴン内部を True
        _rasterize_into(grid.building, self._building_polys, grid)

        # ★ 道路レイヤは作らない（code_review B-16）。
        #   `grid.road` / `sample_road()` を読む箇所はコードベースに 1 件も無い。
        #   レイキャストが `sample_building` だけを使う設計に落ち着いた時点で
        #   使われなくなったまま残っていたもので、マップ読み込みのたびに
        #   全エッジ（銀座 293 本）を buffer してラスタライズしていた。
        return grid

    def _iter_edge_lines(self):
        for edge_id, line in zip(self._edge_line_ids, self._edge_lines):
            edge = self._edges_by_id.get(edge_id)
            if edge is not None:
                yield edge, line

    # ------------------------------------------------------------------
    # 最近傍
    # ------------------------------------------------------------------

    def nearest_node(self, x: float, y: float) -> int:
        """指定座標に最も近い道路ノード ID を返す。"""
        if self._node_xy.shape[0] == 0:
            return 0
        dx = self._node_xy[:, 0] - float(x)
        dy = self._node_xy[:, 1] - float(y)
        return int(np.argmin(dx * dx + dy * dy))

    def nearest_road_point(self, x: float, y: float) -> tuple[float, float, int, float]:
        """指定座標を最寄りの道路中心線上へスナップする。

        Returns:
            (snapped_x, snapped_y, edge_id, heading)
            heading はスナップ地点における道路の進行方向 [rad]（+x 軸から反時計回り）。
        """
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
        """ポリライン上の距離 `distance` における接線方向 [rad]。

        前後 eps の 2 点を差分して求める。ポリラインの向きは u→v なので、
        得られる heading はそのエッジの正方向を指す。
        """
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

    # ------------------------------------------------------------------
    # 経路
    # ------------------------------------------------------------------

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
        """ノード列をエッジのポリラインへ展開し、等間隔にリサンプルして返す。

        グラフ上を v→u の向きに進む場合はエッジのポリラインを反転して連結する。
        """
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
                # 経路が壊れている場合はノード直結で埋める（描画の連続性を優先）
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
        """左側通行の車線に沿った走行経路を返す（道交法 17 条 4 項 / 34 条）。

        中心線をそのまま走ると中央線をまたぐことになるので、進行方向左側の車線へ
        寄せた経路を作る。右左折の手前では法令どおり寄せ、交差点は曲線でつなぐ。
        """
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
                pts.reverse()  # 進行方向に揃える
            segments.append(RouteSegment(edge=edge, points=pts))

        if not segments:
            # 車線を引けない経路（グラフが壊れている等）は中心線で代替する
            return self.route_polyline(path, resample_m)

        route = build_lane_route(segments, float(resample_m))
        if len(route) < 2:
            return self.route_polyline(path, resample_m)
        return route

    def signals_on_route(
        self,
        points: Sequence[tuple[float, float]],
        max_lateral: float = 11.0,
        # 60 度だと斜め交差点で「交差する方向の信号」まで自分の経路の信号として
        # 拾ってしまい、青で通過しているのに赤信号無視と数えられる。
        # 実測で 57 度ずれた信号を拾う例が出たので 35 度まで絞る。
        max_heading_diff: float = math.radians(35.0),
    ) -> list[tuple[float, int]]:
        """経路が通過する信号を (弧長, MapData.signals の添字) で返す。

        信号の座標は道路中心線上にあり、経路は車線へ寄っているので数メートル離れる。
        そこで「経路への最短距離が近い」かつ「進入方向が経路の向きと揃っている」
        ものだけを採る。向きを見ないと、同じ交差点の別方向の信号を拾ってしまう。
        """
        signals = self.data.signals
        if not signals or len(points) < 2:
            return []

        pts = np.asarray(points, dtype=np.float64)
        seg = np.diff(pts, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])

        # 各点での進行方位（最後の点は直前の区間を使う）
        tang = np.empty(pts.shape[0], dtype=np.float64)
        tang[:-1] = np.arctan2(seg[:, 1], seg[:, 0])
        tang[-1] = tang[-2] if pts.shape[0] >= 2 else 0.0

        sx = np.array([sg.x for sg in signals], dtype=np.float64)
        sy = np.array([sg.y for sg in signals], dtype=np.float64)
        sh = np.array([sg.heading for sg in signals], dtype=np.float64)

        # 各信号に最も近い経路上の点
        d2 = (pts[None, :, 0] - sx[:, None]) ** 2 + (pts[None, :, 1] - sy[:, None]) ** 2
        nearest = np.argmin(d2, axis=1)
        dist = np.sqrt(d2[np.arange(len(signals)), nearest])

        route_h = tang[nearest]
        diff = np.abs(np.arctan2(np.sin(sh - route_h), np.cos(sh - route_h)))

        hits = (dist <= float(max_lateral)) & (diff <= float(max_heading_diff))
        out = [(float(cum[nearest[i]]), int(i)) for i in np.flatnonzero(hits)]
        out.sort(key=lambda item: item[0])
        return out

    def speed_limits_on_route(
        self,
        points: Sequence[tuple[float, float]],
        # 標識は路端（中心線から width/2 + 0.8m）に立ち、経路は左端の車線を通るので
        # 実測でおよそ 2.4m 離れる。反対方向の標識は道路幅ぶん向こうにあるが、
        # 狭い道では 8m 以内に入りうるので、向きでも絞る。
        max_lateral: float = 8.0,
        max_heading_diff: float = math.radians(35.0),
    ) -> list[tuple[float, float]]:
        """経路に適用される規制速度を (弧長 [m], 規制速度 [m/s]) の区切りで返す。

        先頭は必ず `(0.0, 出発地点の規制速度)`。以降は標識を通過するたびに 1 件。
        同じ速度が続く区切りは畳む（弧長 a の規制速度は
        「a 以下で最後の区切り」の速度なので、重複しても結果は同じだが無駄）。

        標識の位置と向きの合わせ方は `signals_on_route()` と同じ約束にしてある。
        向きを見ないと、対向車線側や交差道路の標識まで拾ってしまう。
        """
        if len(points) < 2:
            return []

        pts = np.asarray(points, dtype=np.float64)
        seg = np.diff(pts, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])

        # 出発地点に適用されている速度。経路の途中からスポーンしても、
        # 最初の標識に出会うまで規制が分からない状態にはしない。
        start_limit = self._edge_speed_limit_at(float(pts[0, 0]), float(pts[0, 1]))
        breaks: list[tuple[float, float]] = []
        if start_limit is not None:
            breaks.append((0.0, float(start_limit)))

        signs = self.data.signs
        if signs:
            tang = np.empty(pts.shape[0], dtype=np.float64)
            tang[:-1] = np.arctan2(seg[:, 1], seg[:, 0])
            tang[-1] = tang[-2]

            sx = np.array([sn.x for sn in signs], dtype=np.float64)
            sy = np.array([sn.y for sn in signs], dtype=np.float64)
            sh = np.array([sn.heading for sn in signs], dtype=np.float64)

            d2 = (pts[None, :, 0] - sx[:, None]) ** 2 + (pts[None, :, 1] - sy[:, None]) ** 2
            nearest = np.argmin(d2, axis=1)
            dist = np.sqrt(d2[np.arange(len(signs)), nearest])

            route_h = tang[nearest]
            diff = np.abs(np.arctan2(np.sin(sh - route_h), np.cos(sh - route_h)))

            hits = (dist <= float(max_lateral)) & (diff <= float(max_heading_diff))
            found = [
                (float(cum[nearest[i]]), float(signs[i].speed_limit))
                for i in np.flatnonzero(hits)
            ]
            found.sort(key=lambda item: item[0])
            breaks.extend(found)

        if not breaks:
            return []

        # 同じ速度が続く区切りを畳む
        out: list[tuple[float, float]] = [breaks[0]]
        for arc, limit in breaks[1:]:
            if abs(limit - out[-1][1]) < 1e-6:
                continue
            out.append((arc, limit))
        return out

    def _edge_speed_limit_at(self, x: float, y: float) -> float | None:
        """指定座標に最も近い道路の規制速度 [m/s]。道路が引けなければ None。"""
        try:
            _sx, _sy, edge_id, _heading = self.nearest_road_point(x, y)
        except Exception:
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
        """経路が存在し、十分離れた出発／目的ノードの組を返す。

        50 回試して見つからなければ距離条件を捨て、到達可能な中で最も遠いノードを
        目的地に選ぶ。どんな場合でも例外は投げない（学習ループを止めないため）。
        """
        n = self._node_xy.shape[0]
        if n == 0:
            return 0, 0
        if n == 1:
            return 0, 0

        threshold = float(min_distance_m)

        # --- 通常経路: 出発点を引き、そこから threshold 以上離れた候補を試す ---
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

        # --- 緩和: 距離条件を捨てて到達可能な最遠点を選ぶ ---
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

        # --- 最終手段: 到達可能性を問わず適当な 2 点 ---
        return 0, min(1, n - 1)

    # ------------------------------------------------------------------
    # 知覚
    # ------------------------------------------------------------------

    def raycast(
        self,
        origin_x: np.ndarray,
        origin_y: np.ndarray,
        angles: np.ndarray,
        max_distance: float,
        step: float = 1.0,
    ) -> np.ndarray:
        """占有グリッドの building レイヤを step 刻みでサンプリングして距離を測る。

        完全ベクトル化。shape (N,), (N,), (N, R) -> (N, R)。
        何にも当たらなければ max_distance を返す。
        """
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

        # サンプル距離列（0 は自車位置なので step から始める）
        samples = np.arange(step_m, max_d + step_m * 0.5, step_m, dtype=np.float64)
        if samples.size == 0:
            samples = np.array([max_d], dtype=np.float64)

        cos_a = np.cos(ang)[:, :, None]           # (N, R, 1)
        sin_a = np.sin(ang)[:, :, None]           # (N, R, 1)
        t = samples[None, None, :]                # (1, 1, S)

        xs = ox_arr[:, None, None] + cos_a * t    # (N, R, S)
        ys = oy_arr[:, None, None] + sin_a * t

        hits = self.occupancy.sample_building(xs, ys)   # (N, R, S) bool

        any_hit = hits.any(axis=2)
        first = np.argmax(hits, axis=2)                 # ヒットが無い行は 0 になる
        distances = np.where(any_hit, samples[np.clip(first, 0, samples.size - 1)], max_d)
        return np.minimum(distances, max_d).astype(np.float32)

    def collides_with_buildings(
        self, corners: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """複数車両ぶんの外接矩形をまとめて判定する。shape (N,) bool。

        1 台ずつ `collides_with_building()` を呼ぶと、Shapely の `Polygon` 生成と
        木の検索が Python レベルで N 回走る。64 台では 1 ステップ 4.5ms かかり、
        高倍速では予算（8 倍で 6.25ms）をこれだけで使い切ってしまう。
        shapely 2.x のバッチ API を使うと同じ結果が約 20 分の 1 の時間で出る。

        Args:
            corners: shape (N, 4, 2) の外接矩形
            mask: shape (N,) bool。True のスロットだけ判定する
        """
        n = int(corners.shape[0])
        out = np.zeros(n, dtype=bool)
        if self._building_tree is None:
            return out
        idx = np.flatnonzero(np.asarray(mask, dtype=bool))
        if idx.size == 0:
            return out
        try:
            # (K, 4, 2) から K 個の四角形をまとめて作る（shapely が自動で閉じる）
            polys = shapely.polygons(np.asarray(corners[idx], dtype=np.float64))
            # 戻りは (2, M) の [入力添字, 木の添字] ペア
            pairs = self._building_tree.query(polys, predicate="intersects")
            found = np.asarray(pairs)
            if found.size:
                out[idx[np.unique(found[0])]] = True
        except Exception:
            # マップ側の想定外エラーで学習ループを止めない（衝突なし扱い）。
            # ★ ただし黙ってはいけない（code_review B-15）。ここが失敗し続けると
            #   「建物に当たらない世界」になり、症状が**成績が良くなる方向**に
            #   出るため外からは絶対に気づけない。初回だけ理由を残す。
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
            return False
        if rect.is_empty:
            return False
        found = self._building_tree.query(rect, predicate="intersects")
        return int(np.asarray(found).size) > 0


# ---------------------------------------------------------------------------
# 構築関数
# ---------------------------------------------------------------------------


def build_map_index(data: MapData) -> MapIndexImpl:
    """`MapData` から実行時インデックスを組み立てる。"""
    return MapIndexImpl(data)


# ---------------------------------------------------------------------------
# 内部ユーティリティ
# ---------------------------------------------------------------------------


def _sq_dist(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return dx * dx + dy * dy


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
    """ポリゴン群をブール配列へ焼き込む。

    rasterio 等の追加依存を入れず、shapely 2.x のベクトル化関数
    `shapely.contains_xy` を **ポリゴンごとの bbox に限定して** 呼ぶ方式。
    セル中心が内部にあるセルを True にする（world_to_cell と同じ丸め規約）。
    """
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

        # prepare しておくと点内外判定が大幅に速くなる
        shapely.prepare(poly)
        mask = shapely.contains_xy(poly, gx, gy)
        if mask.any():
            layer[r0 : r1 + 1, c0 : c1 + 1] |= mask
