"""運転状態を参照せず、静的な地図から走行経路を作る。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app import config
from app.contracts import MapIndex, RoadLead, RouteLeg
from app.warn import warn_once

NodeArrays = tuple[np.ndarray, np.ndarray, np.ndarray]
#: 立体の地図でノードへ寄せるとき、高さのずれ 1m を平面の距離に換算した重み [m]
SNAP_LEVEL_WEIGHT_M = 10.0

#: 経路が最初に曲がるまでに、少なくともまっすぐ進む距離 [m]
LEAD_MIN_M = 12.0
#: いまの速度から、曲がり始めるまでに要る距離を見込む減速度 [m/s²]
LEAD_DECEL_MPS2 = 2.5


@dataclass
class Route:
    """走行経路。点列 (K, 2) と、それが通る辺（信号と規制速度はここから引く）。"""

    points: np.ndarray
    legs: tuple[RouteLeg, ...]

    def with_points(self, points: np.ndarray) -> "Route":
        """点列だけを差し替えた経路（端を切り詰める・延ばすとき）。始点は動かさないこと。"""
        return Route(points=np.asarray(points, dtype=np.float32), legs=self.legs)



class RoutePlanner:
    """経路生成に必要な地図とノード座標だけを保持する。"""

    def __init__(self, map_index: MapIndex, *, nodes: NodeArrays | None = None) -> None:
        self.map_index = map_index
        self.elevated = bool(getattr(map_index, "elevated", False))
        if nodes is not None:
            self._node_xy, self._node_ids, self._node_z = nodes
            return
        map_data = getattr(map_index, "data", None)
        node_list = getattr(map_data, "nodes", None) if map_data is not None else None
        if node_list:
            self._node_xy = np.array([(nd.x, nd.y) for nd in node_list], dtype=np.float32)
            self._node_ids = np.array([nd.id for nd in node_list], dtype=np.int64)
            self._node_z = np.array([float(getattr(nd, "z", 0.0)) for nd in node_list], dtype=np.float32)
        else:
            self._node_xy = np.zeros((0, 2), dtype=np.float32)
            self._node_ids = np.zeros(0, dtype=np.int64)
            self._node_z = np.zeros(0, dtype=np.float32)

    def _lane_route(self, node_path: list[int], lead: RoadLead | None) -> Route | None:
        pts, legs = self.map_index.lane_route(node_path, config.ROUTE_RESAMPLE_M, lead=lead)
        if pts is None or len(pts) < 2:
            return None
        return Route(points=np.asarray(pts, dtype=np.float32), legs=tuple(legs))

    def _road_lead(
        self, x: float, y: float, heading: float | None, speed: float, z: float | None = None
    ) -> RoadLead | None:
        """いまいる道路を道なりに進み、曲がり始められるノードまでの道のり。立体の地図では `z` の路面を選ぶ。"""
        need = LEAD_MIN_M + max(0.0, float(speed)) ** 2 / (2.0 * LEAD_DECEL_MPS2)
        try:
            if self.elevated:
                if z is not None:
                    return self.map_index.road_lead(float(x), float(y), heading, need, z=float(z))
                # 高さの分からない地点（クリック・徒歩キャラ）は地面に近い路面を選び、無ければ（坑内など）平面で最寄りの道路
                lead = self.map_index.road_lead(float(x), float(y), heading, need, z=self._ground_hint(float(x), float(y)))
                return lead if lead is not None else self.map_index.road_lead(float(x), float(y), heading, need)
            return self.map_index.road_lead(float(x), float(y), heading, need)
        except Exception:
            pass
        warn_once(
            "sim.world.road_lead",
            f"いまいる道路を特定できませんでした: ({x:.1f}, {y:.1f})。経路を作らずに続けます（初回のみ記録）",
        )
        return None

    def _route_from_lead(self, lead: RoadLead, dst: int) -> Route | None:
        """道なりの出だし `lead` の先を、目的ノード `dst` まで最短経路でつなぐ。"""
        closed = getattr(self.map_index, "closed_edges", frozenset())
        if any(edge_id in closed for edge_id in lead.edge_ids):
            return None
        node_path = self.map_index.shortest_path(
            int(lead.exit_node), int(dst), arrive_heading=float(lead.exit_heading)
        )
        if not node_path:
            return None
        return self._lane_route(node_path, lead)

    def _ground_hint(self, x: float, y: float) -> float:
        """高さの分からない地点（画面のクリック・徒歩キャラ）の路面の高さ。地面に近い路面を選ぶ。"""
        surface = getattr(self.map_index, "surface_z", None)
        return 0.0 if surface is None else float(surface(float(x), float(y)))

    def _snap_node(self, x: float, y: float, z: float | None = None) -> int:
        """指定座標に最も近い道路ノードの添字。立体の地図では高さの離れたノード（上下の階）を遠く数える。"""
        dx = self._node_xy[:, 0] - np.float32(x)
        dy = self._node_xy[:, 1] - np.float32(y)
        dist2 = dx * dx + dy * dy
        if self.elevated:
            want = self._ground_hint(x, y) if z is None else float(z)
            dz = (self._node_z - np.float32(want)) * np.float32(SNAP_LEVEL_WEIGHT_M)
            dist2 = dist2 + dz * dz
        return int(np.argmin(dist2))

    @staticmethod
    def _trim_tail(route: Route, dst: tuple[float, float]) -> Route:
        """目的地にいちばん近い点より先を捨てる。"""
        pts = route.points
        if pts.shape[0] < 3:
            return route
        dx = pts[:, 0] - np.float32(dst[0])
        dy = pts[:, 1] - np.float32(dst[1])
        i = int(np.argmin(dx * dx + dy * dy))
        return route.with_points(pts[: max(2, i + 1)])

    @staticmethod
    def _with_destination(route: Route, dst: tuple[float, float]) -> Route:
        """経路の終端を、目的地そのものへ伸ばす。"""
        pts = route.points
        if float(np.hypot(pts[-1, 0] - dst[0], pts[-1, 1] - dst[1])) <= 0.5:
            return route
        return route.with_points(
            np.concatenate([pts, np.array([dst], dtype=np.float32)], axis=0)
        )

    def route_between(
        self,
        src: tuple[float, float],
        dst: tuple[float, float],
        *,
        heading: float | None = None,
        speed: float = 0.0,
        src_z: float | None = None,
        dst_z: float | None = None,
    ) -> Route | None:
        """出発地から道なりに出て、道路ノードへ寄せた目的地までの走行経路を作る（決定 7・12）。"""
        if self._node_xy.shape[0] == 0:
            return None
        node = self._snap_node(float(dst[0]), float(dst[1]), dst_z)
        dst_snap = (float(self._node_xy[node, 0]), float(self._node_xy[node, 1]))
        lead = self._road_lead(float(src[0]), float(src[1]), heading, speed, src_z)
        if lead is None:
            return None
        dst_node = int(self._node_ids[node]) if self.elevated else int(self.map_index.nearest_node(dst_snap[0], dst_snap[1]))
        route = self._route_from_lead(lead, dst_node)
        if route is None:
            return None
        # 別の道路から目的ノードへ入る経路もあるので、念のため折り返しを切り落とす
        return self._with_destination(self._trim_tail(route, dst_snap), dst_snap)

    @staticmethod
    def _cumulative_length(route: np.ndarray) -> np.ndarray:
        """経路点ごとの累積距離 (K,) を返す。"""
        seg = np.diff(route, axis=0)
        lengths = np.hypot(seg[:, 0], seg[:, 1]).astype(np.float32)
        cum = np.zeros(route.shape[0], dtype=np.float32)
        np.cumsum(lengths, out=cum[1:])
        return cum
