"""街を歩く NPC 歩行者の群れ。歩道を歩き、交差点で車道を横断する。"""

from __future__ import annotations

import math

import numpy as np

from app import config
from app.contracts import MapData, MapIndex, PedestrianSnapshot

__all__ = ["PedestrianCrowd", "SidewalkNetwork", "build_sidewalk_network"]

#: 歩道の候補から外す短すぎるエッジ [m]。端点だけで構成される路地に溜まるのを防ぐ
MIN_WALKABLE_M = 8.0

#: 横断にかける時間の下限 [秒]。幅の狭い道でも一瞬で渡り切らせない
CROSS_MIN_SEC = 1.2

#: 再配置のときに車両から離す距離 [m]。湧いた瞬間に轢かれるのを避ける
SPAWN_CLEARANCE_M = 6.0

#: これより遠い歩行者は誰の視界にも入らないので、車の近くへ回す [m]。
#: 擬似カメラの far は 120m なので、その倍は見えない
RECYCLE_FAR_M = 220.0
#: 回した先の距離 [m]。近すぎると目の前に湧いたように見える
RECYCLE_NEAR_M = 40.0
RECYCLE_REACH_M = 150.0


class SidewalkNetwork:
    """歩行者を歩かせるために前処理した道路網（ポリラインの連結表と隣接表）。"""

    def __init__(self, data: MapData) -> None:
        edges = [e for e in data.edges if len(e.polyline) >= 2 and e.length >= MIN_WALKABLE_M]
        if not edges:
            edges = [e for e in data.edges if len(e.polyline) >= 2]

        counts = np.array([len(e.polyline) for e in edges], dtype=np.int64)
        self.count = int(counts.size)
        self.start = np.zeros(self.count, dtype=np.int64)
        if self.count:
            self.start[1:] = np.cumsum(counts)[:-1]
        self.point_count = counts
        self.points = (
            np.concatenate([np.asarray(e.polyline, dtype=np.float64) for e in edges], axis=0)
            if self.count
            else np.zeros((0, 2), dtype=np.float64)
        )

        cum = np.zeros(self.points.shape[0], dtype=np.float64)
        length = np.zeros(self.count, dtype=np.float64)
        for i in range(self.count):
            lo = int(self.start[i])
            hi = lo + int(counts[i])
            seg = np.linalg.norm(np.diff(self.points[lo:hi], axis=0), axis=1)
            cum[lo + 1 : hi] = np.cumsum(seg)
            length[i] = float(cum[hi - 1])
        self.length = length

        self.offset = np.zeros(self.count, dtype=np.float64)
        if self.count:
            self.offset[1:] = np.cumsum(length)[:-1]
        self.cum = cum + np.repeat(self.offset, counts) if self.count else cum

        half = np.array([max(float(e.width), config.MIN_ROAD_WIDTH) * 0.5 for e in edges])
        self.walk_offset = half + float(config.PEDESTRIAN_SIDEWALK_MARGIN)

        self.node_u = np.array([e.u for e in edges], dtype=np.int64)
        self.node_v = np.array([e.v for e in edges], dtype=np.int64)

        ids = np.array([e.id for e in edges], dtype=np.int64)
        order = np.argsort(ids, kind="stable")
        self._id_sorted = ids[order]
        self._id_index = order.astype(np.int64)
        self._build_adjacency()

    def index_of(self, edge_id: int) -> int:
        """`MapEdge.id` から歩道網の添字を引く。無ければ -1。"""
        pos = int(np.searchsorted(self._id_sorted, int(edge_id)))
        if pos >= self._id_sorted.size or int(self._id_sorted[pos]) != int(edge_id):
            return -1
        return int(self._id_index[pos])

    def _build_adjacency(self) -> None:
        """ノードごとに「そこから出ていけるエッジ」を CSR 形式で持つ。"""
        if self.count == 0:
            self.adj_nodes = np.zeros(0, dtype=np.int64)
            self.adj_start = np.zeros(0, dtype=np.int64)
            self.adj_count = np.zeros(0, dtype=np.int64)
            self.adj_edge = np.zeros(0, dtype=np.int64)
            self.adj_dir = np.zeros(0, dtype=np.int8)
            return

        ends = np.concatenate([self.node_u, self.node_v])
        edge_ids = np.concatenate([np.arange(self.count), np.arange(self.count)])
        dirs = np.concatenate(
            [np.ones(self.count, dtype=np.int8), -np.ones(self.count, dtype=np.int8)]
        )
        order = np.argsort(ends, kind="stable")
        ends = ends[order]
        self.adj_edge = edge_ids[order]
        self.adj_dir = dirs[order]

        self.adj_nodes, first, counts = np.unique(
            ends, return_index=True, return_counts=True
        )
        self.adj_start = first.astype(np.int64)
        self.adj_count = counts.astype(np.int64)

    def segment_of(self, edge: np.ndarray, arc: np.ndarray) -> np.ndarray:
        """弧長位置が乗っているセグメントの先頭点の添字を返す。"""
        target = self.offset[edge] + arc
        idx = np.searchsorted(self.cum, target, side="right") - 1
        lo = self.start[edge]
        hi = lo + self.point_count[edge] - 2
        return np.clip(idx, lo, np.maximum(hi, lo))

    def sample(
        self, edge: np.ndarray, arc: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """道路中心線上の点 (x, y) と、そこでの接線 (cos, sin) を返す。"""
        seg = self.segment_of(edge, arc)
        p0 = self.points[seg]
        p1 = self.points[seg + 1]
        d = p1 - p0
        seg_len = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
        t = np.clip((self.offset[edge] + arc - self.cum[seg]) / seg_len, 0.0, 1.0)
        x = p0[:, 0] + d[:, 0] * t
        y = p0[:, 1] + d[:, 1] * t
        return x, y, d[:, 0] / seg_len, d[:, 1] / seg_len

    def arc_of(self, edge: int, x: float, y: float) -> float:
        """エッジ上（の近く）の点を、そのエッジの弧長位置へ落とす。

        ★ `MapIndex.nearest_road_point()` が返す 4 番目は**接線方位であって弧長ではない**。
        そのまま弧長として使うと、置いたつもりの場所とまったく違う所に立つ。
        """
        lo = int(self.start[edge])
        hi = lo + int(self.point_count[edge])
        pts = self.points[lo:hi]
        if pts.shape[0] < 2:
            return 0.0
        a = pts[:-1]
        d = pts[1:] - a
        seg2 = np.maximum((d * d).sum(axis=1), 1e-12)
        t = np.clip(((x - a[:, 0]) * d[:, 0] + (y - a[:, 1]) * d[:, 1]) / seg2, 0.0, 1.0)
        proj_x = a[:, 0] + d[:, 0] * t
        proj_y = a[:, 1] + d[:, 1] * t
        j = int(np.argmin((proj_x - x) ** 2 + (proj_y - y) ** 2))
        within = float(self.cum[lo + j] - self.offset[edge])
        return float(within + t[j] * math.sqrt(float(seg2[j])))

    def neighbours(self, node: int, rng: np.random.Generator) -> tuple[int, int] | None:
        """ノードから出ていけるエッジを 1 本選ぶ。無ければ None。"""
        pos = int(np.searchsorted(self.adj_nodes, node))
        if pos >= self.adj_nodes.size or int(self.adj_nodes[pos]) != int(node):
            return None
        lo = int(self.adj_start[pos])
        n = int(self.adj_count[pos])
        if n <= 0:
            return None
        k = lo + int(rng.integers(0, n))
        return int(self.adj_edge[k]), int(self.adj_dir[k])


def build_sidewalk_network(data: MapData) -> SidewalkNetwork:
    return SidewalkNetwork(data)


class PedestrianCrowd:
    """NPC 歩行者の集合。位置は全員まとめて numpy で進める。

    ★ 状態は「どのエッジの歩道を、どちら向きに、左右どちら側を歩いているか」で持つ。
    座標を直接積分すると建物へめり込むし、道路から離れていく。
    """

    def __init__(self, map_index: MapIndex, rng: np.random.Generator) -> None:
        self.map_index = map_index
        self.rng = rng
        self.net = build_sidewalk_network(map_index.data)

        n = config.MAX_PEDESTRIANS
        self.active = np.zeros(n, dtype=bool)
        self.edge = np.zeros(n, dtype=np.int64)
        self.dir = np.ones(n, dtype=np.int8)
        self.side = np.ones(n, dtype=np.int8)
        self.arc = np.zeros(n, dtype=np.float64)
        self.speed = np.zeros(n, dtype=np.float64)
        #: 横断の進捗 0.0〜1.0。0 なら歩道を歩いている
        self.cross = np.zeros(n, dtype=np.float64)
        self.crossing = np.zeros(n, dtype=bool)
        self.cross_to = np.ones(n, dtype=np.int8)
        self.stride = np.zeros(n, dtype=np.float64)

        self.x = np.zeros(n, dtype=np.float64)
        self.y = np.zeros(n, dtype=np.float64)
        self.heading = np.zeros(n, dtype=np.float64)

    @property
    def walkable(self) -> bool:
        """歩ける道路があるか。無いマップでは歩行者を出せない。"""
        return self.net.count > 0

    @property
    def count(self) -> int:
        return int(np.count_nonzero(self.active))

    def set_count(self, count: int, vehicle_xy: np.ndarray | None = None) -> None:
        """出す人数を変える。増えた分だけ道路上へ配置し、減った分は消す。"""
        want = int(np.clip(int(count), 0, config.MAX_PEDESTRIANS))
        if not self.walkable:
            self.active[:] = False
            return
        current = int(np.count_nonzero(self.active))
        if want == current:
            return
        if want < current:
            alive = np.flatnonzero(self.active)
            self.active[alive[want:]] = False
            return
        free = np.flatnonzero(~self.active)[: want - current]
        for slot in free:
            self._spawn(int(slot), vehicle_xy)
        self._refresh_pose()

    def _spawn(self, slot: int, vehicle_xy: np.ndarray | None = None) -> None:
        """1 人を道路上のどこかへ置く。車両の近くは避ける。"""
        rng = self.rng
        for _ in range(8):
            edge = int(rng.integers(0, self.net.count))
            arc = float(rng.uniform(0.0, float(self.net.length[edge])))
            if vehicle_xy is None or vehicle_xy.shape[0] == 0:
                break
            x, y, _, _ = self.net.sample(
                np.array([edge], dtype=np.int64), np.array([arc], dtype=np.float64)
            )
            dx = vehicle_xy[:, 0] - float(x[0])
            dy = vehicle_xy[:, 1] - float(y[0])
            if float(np.min(dx * dx + dy * dy)) >= SPAWN_CLEARANCE_M**2:
                break
        else:
            edge = int(rng.integers(0, self.net.count))
            arc = float(rng.uniform(0.0, float(self.net.length[edge])))

        self.active[slot] = True
        self.edge[slot] = edge
        self.arc[slot] = arc
        self.dir[slot] = 1 if rng.random() < 0.5 else -1
        self.side[slot] = 1 if rng.random() < 0.5 else -1
        self.cross_to[slot] = -self.side[slot]
        self.crossing[slot] = False
        self.cross[slot] = 0.0
        self.stride[slot] = float(rng.uniform(0.0, 2.0 * math.pi))
        self.speed[slot] = float(
            config.PEDESTRIAN_SPEED
            + rng.uniform(-1.0, 1.0) * config.PEDESTRIAN_SPEED_SPREAD
        )

    def step(self, dt: float) -> None:
        """全員を dt 秒ぶん進める。"""
        idx = np.flatnonzero(self.active)
        if idx.size == 0:
            return

        walking = idx[~self.crossing[idx]]
        if walking.size:
            step = self.speed[walking] * dt * self.dir[walking]
            self.arc[walking] += step
            self.stride[walking] += np.abs(step) * 2.0

        crossing = idx[self.crossing[idx]]
        if crossing.size:
            span = np.maximum(
                self.net.walk_offset[self.edge[crossing]] * 2.0, 1e-3
            )
            rate = self.speed[crossing] / np.maximum(span, 1e-3)
            self.cross[crossing] += np.minimum(rate, 1.0 / CROSS_MIN_SEC) * dt
            self.stride[crossing] += self.speed[crossing] * dt * 2.0
            done = crossing[self.cross[crossing] >= 1.0]
            if done.size:
                self.side[done] = self.cross_to[done]
                self.cross[done] = 0.0
                self.crossing[done] = False

        self._handle_ends(walking)
        self._refresh_pose()

    def _handle_ends(self, slots: np.ndarray) -> None:
        """エッジの端に着いた人を、横断させるか次のエッジへ移す。"""
        if slots.size == 0:
            return
        length = self.net.length[self.edge[slots]]
        over = slots[(self.arc[slots] < 0.0) | (self.arc[slots] > length)]
        rng = self.rng
        for raw in over:
            slot = int(raw)
            edge = int(self.edge[slot])
            forward = int(self.dir[slot]) > 0
            node = int(self.net.node_v[edge] if forward else self.net.node_u[edge])
            span = float(self.net.length[edge])

            if rng.random() < config.PEDESTRIAN_CROSS_PROB:
                setback = min(config.PEDESTRIAN_CROSS_SETBACK_M, span * 0.5)
                self.arc[slot] = span - setback if forward else setback
                self.crossing[slot] = True
                self.cross[slot] = 0.0
                self.cross_to[slot] = -self.side[slot]
                self.dir[slot] = -self.dir[slot]
                continue

            nxt = self.net.neighbours(node, rng)
            if nxt is None:
                self.arc[slot] = float(np.clip(self.arc[slot], 0.0, span))
                self.dir[slot] = -self.dir[slot]
                continue
            next_edge, next_dir = nxt
            self.edge[slot] = next_edge
            self.dir[slot] = np.int8(next_dir)
            self.arc[slot] = (
                0.0 if next_dir > 0 else float(self.net.length[next_edge])
            )
            if rng.random() < 0.5:
                self.side[slot] = -self.side[slot]

    def _refresh_pose(self) -> None:
        """エッジ上の状態から world 座標と向きを作り直す。"""
        idx = np.flatnonzero(self.active)
        if idx.size == 0:
            return
        edge = self.edge[idx]
        arc = np.clip(self.arc[idx], 0.0, self.net.length[edge])
        cx, cy, tx, ty = self.net.sample(edge, arc)

        side_now = np.where(
            self.crossing[idx],
            self.side[idx] + (self.cross_to[idx] - self.side[idx]) * self.cross[idx],
            self.side[idx],
        )
        offset = side_now * self.net.walk_offset[edge]
        self.x[idx] = cx - ty * offset
        self.y[idx] = cy + tx * offset

        walk_heading = np.arctan2(ty * self.dir[idx], tx * self.dir[idx])
        toward = np.sign(self.cross_to[idx] - self.side[idx])
        cross_heading = np.arctan2(tx * toward, -ty * toward)
        self.heading[idx] = np.where(self.crossing[idx], cross_heading, walk_heading)

    def recycle(self, vehicle_xy: np.ndarray) -> int:
        """どの車からも遠い歩行者を、車の近くの歩道へ回す。回した人数を返す。

        ★ これが無いと**広いマップでは 1 人も画に写らない**。金沢は 12.3km 四方
        なので、64 人を一様に撒くと 0.42 人/km² にしかならず、実測で真値 0 件だった。
        誰にも見えていない人だけを動かすので、画面上でワープして見えることはない。
        """
        idx = np.flatnonzero(self.active)
        if idx.size == 0 or vehicle_xy is None or vehicle_xy.shape[0] == 0:
            return 0
        dx = self.x[idx][:, None] - vehicle_xy[None, :, 0]
        dy = self.y[idx][:, None] - vehicle_xy[None, :, 1]
        far = idx[np.min(dx * dx + dy * dy, axis=1) > RECYCLE_FAR_M**2]
        if far.size == 0:
            return 0

        rng = self.rng
        pick = rng.integers(0, vehicle_xy.shape[0], size=far.size)
        angle = rng.uniform(-math.pi, math.pi, size=far.size)
        reach = rng.uniform(RECYCLE_NEAR_M, RECYCLE_REACH_M, size=far.size)
        moved = 0
        for k, raw in enumerate(far):
            slot = int(raw)
            i = int(pick[k])
            tx = float(vehicle_xy[i, 0]) + math.cos(float(angle[k])) * float(reach[k])
            ty = float(vehicle_xy[i, 1]) + math.sin(float(angle[k])) * float(reach[k])
            if self._place_at(slot, tx, ty):
                moved += 1
        if moved:
            self._refresh_pose()
        return moved

    def relocate(self, vehicle_xy: np.ndarray | None = None) -> None:
        """全員を置き直す（エピソードのリセット・教師データ収集の散らし直し）。"""
        for slot in np.flatnonzero(self.active):
            self._spawn(int(slot), vehicle_xy)
        self._refresh_pose()

    def gather_near(
        self,
        anchors_x: np.ndarray,
        anchors_y: np.ndarray,
        anchors_heading: np.ndarray,
        *,
        reach: float = 26.0,
        ahead_only: bool = True,
    ) -> None:
        """車両の周りへ寄せ集める（教師データを集めるとき）。

        道路網の上で位置を決めるので、寄せても歩道から外れない。
        `ahead_only` が True なら車両の前方の錐の中だけに置く（狙って集めるとき）。
        """
        idx = np.flatnonzero(self.active)
        if idx.size == 0 or anchors_x.size == 0 or not self.walkable:
            return
        rng = self.rng
        pick = rng.integers(0, anchors_x.size, size=idx.size)
        ahead = rng.uniform(6.0, reach, size=idx.size)
        spread = rng.uniform(-0.55, 0.55, size=idx.size) if ahead_only else rng.uniform(
            -math.pi, math.pi, size=idx.size
        )
        for k, raw in enumerate(idx):
            slot = int(raw)
            i = int(pick[k])
            heading = float(anchors_heading[i]) + float(spread[k])
            tx = float(anchors_x[i]) + math.cos(heading) * float(ahead[k])
            ty = float(anchors_y[i]) + math.sin(heading) * float(ahead[k])
            if not self._place_at(slot, tx, ty):
                self._spawn(slot)
        self._refresh_pose()

    def _place_at(self, slot: int, x: float, y: float) -> bool:
        """指定地点に最も近い歩道へ置く。道路が見つからなければ False。"""
        try:
            snap_x, snap_y, edge_id, _heading = self.map_index.nearest_road_point(x, y)
        except Exception:
            return False
        edge = self.net.index_of(int(edge_id))
        if edge < 0:
            return False
        arc = self.net.arc_of(edge, float(snap_x), float(snap_y))
        self.edge[slot] = edge
        self.arc[slot] = float(np.clip(arc, 0.0, float(self.net.length[edge])))
        self.dir[slot] = 1 if self.rng.random() < 0.5 else -1
        self.side[slot] = 1 if self.rng.random() < 0.5 else -1
        self.cross_to[slot] = -self.side[slot]
        self.crossing[slot] = False
        self.cross[slot] = 0.0
        return True

    def snapshot(self) -> list[PedestrianSnapshot]:
        """描画用のスナップショット。**アクティブな人だけ**返す。"""
        out: list[PedestrianSnapshot] = []
        for raw in np.flatnonzero(self.active):
            slot = int(raw)
            out.append(
                PedestrianSnapshot(
                    id=slot,
                    x=float(self.x[slot]),
                    y=float(self.y[slot]),
                    heading=float(self.heading[slot]),
                    stride=float(self.stride[slot] % (2.0 * math.pi)),
                    crossing=bool(self.crossing[slot]),
                )
            )
        return out

    @property
    def positions(self) -> np.ndarray:
        """アクティブな歩行者の座標 (K, 2) float64。"""
        idx = np.flatnonzero(self.active)
        if idx.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return np.column_stack((self.x[idx], self.y[idx]))
