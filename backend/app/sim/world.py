"""マップ・車両・障害物・衝突判定を束ねるシミュレーション世界。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import logging

import numpy as np

from app import config
from app.contracts import FrameSnapshot, MapIndex, ObstacleSnapshot, VehicleSnapshot
from app.sim.pedestrians import PedestrianCrowd
from app.sim.signals import (
    RED,
    YELLOW,
    SignalController,
    signal_speed_limit,
    stop_speed_limit,
)
from app.sim.vehicle import VehicleFleet

__all__ = ["ObstacleState", "SlotState", "World"]

ROUTE_MIN_DISTANCE_M = 120.0
DESTINATION_TRIALS = 24
PROJECT_WINDOW_BACK = 8
PROJECT_WINDOW_FWD = 48

VEHICLE_HIT_SEMI_LONG_M = config.VEHICLE_LENGTH * 0.9
VEHICLE_HIT_SEMI_LAT_M = config.VEHICLE_WIDTH * 1.1

#: 再配置するとき、他車からこれだけ離す [m²]。**車 1 台分では足りない**
#: （6.6m 先に湧くと、走ってきた車が次の瞬間に追突する）
SPAWN_CLEARANCE_M2 = (config.VEHICLE_LENGTH * 4.0) ** 2
SPAWN_ROUTE_TRIALS = 12
SPAWN_SIGNAL_SKIP_M = 3.0

SIGNAL_LOOKAHEAD_COUNT = 3

LANE_DEPARTURE_M = config.LANE_DEPARTURE_M
LANE_RETURN_M = config.LANE_RETURN_M

#: 誰にも見えていない歩行者を車の近くへ回す間隔 [秒]。毎ステップやる必要はない
PEDESTRIAN_RECYCLE_SEC = 2.0

#: 徒歩キャラが居ないときに返す空配列。毎回作らないよう 1 つだけ持つ
_NO_PLAYER = np.zeros((0, 2), dtype=np.float64)
_NO_PLAYER.flags.writeable = False

#: これより強い制動指令が出ている間はブレーキランプを点ける。
#: 0 にすると、方策が出す微小な負の値で点きっぱなしになる
BRAKE_COMMAND_THRESHOLD = -0.05

#: 方向指示器を出し始める距離 [m]（道交法施行令 21 条「30m 手前」）
TURN_LOOKAHEAD_M = 30.0
#: 先読み地点の接線を測る幅 [m]。1 点では接線が出ない
TURN_TANGENT_SPAN_M = 3.0
#: 出す／消すしきい値 [rad]。ヒステリシスを付けて、緩いカーブで点滅させない
TURN_ON_RAD = 0.35
TURN_OFF_RAD = 0.12

FORWARD_NODE_RADIUS_M = 150.0
FORWARD_NODE_MIN_M = 14.0
FORWARD_NODE_MAX_ANGLE_RAD = math.radians(60.0)


def _in_body_ellipse(
    dx: np.ndarray,
    dy: np.ndarray,
    cos_h: np.ndarray,
    sin_h: np.ndarray,
    semi_long: float | np.ndarray,
    semi_lat: float | np.ndarray,
) -> np.ndarray:
    """相対位置 (dx, dy) が、車体の向きに沿った楕円の内側かを返す。"""
    long_ = dx * cos_h + dy * sin_h
    lat = -dx * sin_h + dy * cos_h
    return (long_ / semi_long) ** 2 + (lat / semi_lat) ** 2 < 1.0


@dataclass
class ObstacleState:
    """ユーザーが置いたパイロン等。id は World 内で単調増加。"""

    id: int
    x: float
    y: float
    radius: float


@dataclass
class SlotState:
    """1 スロット分の経路・エピソード状態。"""

    route: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=np.float32))
    route_cum: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    route_progress: int = 0
    arc_position: float = 0.0
    goal: tuple[float, float] = (0.0, 0.0)
    steps: int = 0
    collided: bool = False
    reached_goal: bool = False
    route_dirty: bool = True
    route_speed_limit: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    signal_arcs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    signal_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    sign_arcs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    sign_limits: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    signals_passed: int = 0
    signals_floor: int = 0
    committed_signal: int = -1
    violations: int = 0
    lane_departures: int = 0
    outside_lane: bool = False
    speed_violations: int = 0
    over_speed: bool = False


logger = logging.getLogger("autoware_sim")


class World:
    """マップ上の車両群・障害物・衝突判定を保持する。"""

    def __init__(self, map_index: MapIndex, rng: np.random.Generator) -> None:
        self.map_index = map_index
        self.rng = rng
        self.fleet = VehicleFleet(config.MAX_VEHICLES)
        self.crowd = PedestrianCrowd(map_index, rng)
        #: 実用モードの徒歩キャラ (0, 2) か (1, 2)。**NPC 群衆とは別に持つ**
        self._player_xy = _NO_PLAYER
        self.obstacles: list[ObstacleState] = []
        self.slots: list[SlotState] = [SlotState() for _ in range(config.MAX_VEHICLES)]

        self._next_obstacle_id = 0
        self._recycled_at = 0.0
        self._signals_on_route_failed = False
        self._speed_limits_on_route_failed = False
        n = config.MAX_VEHICLES

        self.signals = SignalController(map_index.data.signals)
        self.sim_time = 0.0
        self.signal_phases: list[int] = self.signals.phases(0.0)

        self.arc = np.zeros(n, dtype=np.float32)
        self.route_total = np.zeros(n, dtype=np.float32)
        self._goal_x = np.zeros(n, dtype=np.float32)
        self._goal_y = np.zeros(n, dtype=np.float32)
        self.lateral = np.zeros(n, dtype=np.float32)
        self.tangent = np.zeros(n, dtype=np.float32)
        self.arc_delta = np.zeros(n, dtype=np.float32)

        self.collided_flags = np.zeros(n, dtype=bool)
        self.reached_flags = np.zeros(n, dtype=bool)

        self.braking = np.zeros(n, dtype=bool)
        #: 方向指示器。-1=左 / 0=消灯 / +1=右
        self.turn_signal = np.zeros(n, dtype=np.int8)
        #: 直近の `check_collisions()` で歩行者に当たったスロット
        self.pedestrian_hits = np.zeros(n, dtype=bool)

        self.stop_arc = np.full(n, np.inf, dtype=np.float64)

        self._obstacle_xy = np.zeros((0, 2), dtype=np.float32)
        self._obstacle_r = np.zeros(0, dtype=np.float32)

        map_data = getattr(map_index, "data", None)
        node_list = getattr(map_data, "nodes", None) if map_data is not None else None
        if node_list:
            self._node_xy = np.array([(nd.x, nd.y) for nd in node_list], dtype=np.float32)
            self._node_ids = np.array([nd.id for nd in node_list], dtype=np.int64)
        else:
            self._node_xy = np.zeros((0, 2), dtype=np.float32)
            self._node_ids = np.zeros(0, dtype=np.int64)

    def _route_from_nodes(self, src: int, dst: int) -> np.ndarray | None:
        """ノード ID の組から経路点列 (K, 2) を作る。失敗したら None。"""
        node_path = self.map_index.shortest_path(int(src), int(dst))
        if not node_path or len(node_path) < 2:
            return None
        pts = self.map_index.lane_route_polyline(node_path, config.ROUTE_RESAMPLE_M)
        if pts is None or len(pts) < 2:
            return None
        return np.asarray(pts, dtype=np.float32)

    def _start_clearance(self, route: np.ndarray, exclude: int) -> float:
        """経路の始点から、いちばん近い他車までの距離の二乗。他に誰もいなければ inf。"""
        if route is None or route.shape[0] < 1:
            return -1.0
        active = self.fleet.active.copy()
        active[exclude] = False
        idx = np.flatnonzero(active)
        if idx.size == 0:
            return float("inf")
        dx = self.fleet.x[idx] - np.float32(route[0, 0])
        dy = self.fleet.y[idx] - np.float32(route[0, 1])
        return float(np.min(dx * dx + dy * dy))

    def _start_is_clear(self, route: np.ndarray, exclude: int) -> bool:
        """経路の始点が、既に走っている車両と重なっていないか。"""
        return self._start_clearance(route, exclude) >= SPAWN_CLEARANCE_M2

    def _random_route(self, exclude: int = -1) -> np.ndarray | None:
        """ランダムな出発地・目的地の組から経路を作る。

        ★ 空いている始点が見つからないときは、**いちばん空いている候補**を返すこと。
        最初に作れた経路をそのまま返すと、他車の真上に湧いて出合い頭の事故になる
        （金沢のように到達可能な組が少ないマップで起きる）。
        """
        best: np.ndarray | None = None
        best_clearance = -1.0
        for _ in range(SPAWN_ROUTE_TRIALS):
            src, dst = self.map_index.random_node_pair(self.rng, ROUTE_MIN_DISTANCE_M)
            if int(src) == int(dst):
                continue
            route = self._route_from_nodes(src, dst)
            if route is None:
                continue
            if exclude < 0:
                return route
            clearance = self._start_clearance(route, exclude)
            if clearance >= SPAWN_CLEARANCE_M2:
                return route
            if clearance > best_clearance:
                best_clearance = clearance
                best = route
        return best

    def _route_from_point(
        self, x: float, y: float, *, heading: float | None = None
    ) -> np.ndarray | None:
        """指定座標を道路にスナップし、そこから到達可能な目的地への経路を作る。"""
        try:
            snap_x, snap_y, _edge_id, _heading = self.map_index.nearest_road_point(
                float(x), float(y)
            )
        except Exception:
            logger.debug("道路へのスナップに失敗しました: (%.1f, %.1f)", x, y, exc_info=True)
            return None
        src = (
            self._forward_node(float(snap_x), float(snap_y), heading)
            if heading is not None
            else None
        )
        if src is None:
            src = int(self.map_index.nearest_node(float(snap_x), float(snap_y)))

        route: np.ndarray | None = None
        if self._node_xy.shape[0] > 1:
            dx = self._node_xy[:, 0] - np.float32(snap_x)
            dy = self._node_xy[:, 1] - np.float32(snap_y)
            dist2 = dx * dx + dy * dy
            far = np.flatnonzero(dist2 >= np.float32(ROUTE_MIN_DISTANCE_M * ROUTE_MIN_DISTANCE_M))
            if far.size == 0:
                far = np.flatnonzero(dist2 > np.float32(1.0))
            trials = min(DESTINATION_TRIALS, max(int(far.size), 1))
            for _ in range(trials):
                if far.size == 0:
                    break
                pick = int(self.rng.integers(0, far.size))
                dst = int(self._node_ids[far[pick]])
                if dst == src:
                    continue
                route = self._route_from_nodes(src, dst)
                if route is not None:
                    break

        if route is None:
            return self._random_route()

        if float(np.hypot(route[0, 0] - snap_x, route[0, 1] - snap_y)) > 0.5:
            head = np.array([[snap_x, snap_y]], dtype=np.float32)
            route = np.concatenate([head, route], axis=0)
        return route

    def snap_to_road(self, x: float, y: float) -> tuple[float, float] | None:
        """指定座標を最寄りの道路中心線上へスナップする。"""
        try:
            snap_x, snap_y, _edge_id, _heading = self.map_index.nearest_road_point(
                float(x), float(y)
            )
        except Exception:
            logger.debug("道路へのスナップに失敗しました: (%.1f, %.1f)", x, y, exc_info=True)
            return None
        return float(snap_x), float(snap_y)

    def snap_to_road_node(self, x: float, y: float) -> tuple[float, float] | None:
        """指定座標を最寄りの道路ノードへ寄せる（決定 7）。

        **道路中心線上の任意の点ではなくノードにすること。** 経路はノード間で作るので、
        区間の途中を目的地にすると「そこを通り過ぎて次の交差点まで行き、折り返して戻る」
        経路になる。停車は経路の終点で掛けるため、車は目的地の前を減速せず通過する。
        """
        if self._node_xy.shape[0] == 0:
            return None
        dx = self._node_xy[:, 0] - np.float32(x)
        dy = self._node_xy[:, 1] - np.float32(y)
        i = int(np.argmin(dx * dx + dy * dy))
        return float(self._node_xy[i, 0]), float(self._node_xy[i, 1])

    def _forward_node(self, x: float, y: float, heading: float) -> int | None:
        """進行方向の前方にあるノード。

        **背後のノードを選ぶと逆走経路になり、真横や足元のノードを選ぶと
        経路がその場から直角に始まる。** 後者は車が曲がりきれずに膨らみ、
        対向車線や建物へ出て事故になる（旋回半径は最大舵角でも 4.4m ある）。
        """
        if self._node_xy.shape[0] == 0:
            return None
        dx = self._node_xy[:, 0] - np.float32(x)
        dy = self._node_xy[:, 1] - np.float32(y)
        dist2 = (dx * dx + dy * dy).astype(np.float64)
        forward = dx * math.cos(heading) + dy * math.sin(heading)
        ok = (
            (dist2 >= FORWARD_NODE_MIN_M**2)
            & (dist2 <= FORWARD_NODE_RADIUS_M**2)
            & (forward > 0.0)
            & (forward >= np.sqrt(dist2) * math.cos(FORWARD_NODE_MAX_ANGLE_RAD))
        )
        cand = np.flatnonzero(ok)
        if cand.size == 0:
            return None
        return int(self._node_ids[cand[int(np.argmin(dist2[cand]))]])

    @staticmethod
    def _trim_tail(route: np.ndarray, dst: tuple[float, float]) -> np.ndarray:
        """目的地にいちばん近い点より先を捨てる。

        ノード間の経路は目的地の先の交差点まで伸びているので、切らずに目的地を
        足すと**「目的地を通り過ぎてから折り返す」経路**になる。停車は経路の終点で
        掛けるため、車は目的地の前を減速せずに通過してしまう。
        """
        if route.shape[0] < 3:
            return route
        dx = route[:, 0] - np.float32(dst[0])
        dy = route[:, 1] - np.float32(dst[1])
        i = int(np.argmin(dx * dx + dy * dy))
        return route[: max(2, i + 1)]

    @staticmethod
    def _with_endpoints(
        route: np.ndarray, src: tuple[float, float], dst: tuple[float, float]
    ) -> np.ndarray:
        """経路の両端を、指定した出発地・目的地そのものへ伸ばす。"""
        parts: list[np.ndarray] = [route]
        if float(np.hypot(route[0, 0] - src[0], route[0, 1] - src[1])) > 0.5:
            parts.insert(0, np.array([src], dtype=np.float32))
        if float(np.hypot(route[-1, 0] - dst[0], route[-1, 1] - dst[1])) > 0.5:
            parts.append(np.array([dst], dtype=np.float32))
        return np.concatenate(parts, axis=0) if len(parts) > 1 else route

    def route_onward(self, x: float, y: float, heading: float) -> np.ndarray | None:
        """いまいる場所から、進行方向の先にある目的地への経路を作る（徴用の解除に使う）。"""
        return self._route_from_point(x, y, heading=heading)

    def route_between(
        self,
        src: tuple[float, float],
        dst: tuple[float, float],
        *,
        heading: float | None = None,
    ) -> np.ndarray | None:
        """出発地を道路へ、目的地を道路ノードへ寄せて走行経路を作る（決定 7・12）。

        **目的地だけノードに合わせる**のは、経路がノード間で作られるため。
        区間の途中に置くと、そこを通り過ぎてから折り返す経路になる。
        """
        src_snap = self.snap_to_road(*src)
        dst_snap = self.snap_to_road_node(*dst)
        if src_snap is None or dst_snap is None:
            return None

        src_node = (
            self._forward_node(src_snap[0], src_snap[1], heading)
            if heading is not None
            else None
        )
        if src_node is None:
            src_node = int(self.map_index.nearest_node(src_snap[0], src_snap[1]))
        dst_node = int(self.map_index.nearest_node(dst_snap[0], dst_snap[1]))
        if src_node == dst_node:
            return None

        route = self._route_from_nodes(src_node, dst_node)
        if route is None:
            return None
        # ★ 別の道路から目的ノードへ入る経路もあるので、念のため折り返しを切り落とす
        return self._with_endpoints(self._trim_tail(route, dst_snap), src_snap, dst_snap)

    def advance_time(self, dt: float) -> None:
        """シミュレーション内時刻を進め、信号の現示と歩行者を更新する。"""
        self.sim_time += float(dt)
        self.signal_phases = self.signals.phases(self.sim_time)
        # ★ 現示を作り直した**あと**に渡すこと。1 ステップ古い色で渡らせない
        self.crowd.step(float(dt), self.signal_phases)
        if self.sim_time - self._recycled_at >= PEDESTRIAN_RECYCLE_SEC:
            self._recycled_at = self.sim_time
            self.crowd.recycle(self._active_vehicle_xy())

    def set_pedestrian_count(self, count: int) -> None:
        """街を歩く NPC 歩行者の人数を変える。"""
        self.crowd.set_count(int(count), self._active_vehicle_xy())

    def relocate_pedestrians(self) -> None:
        """歩行者を街中へ置き直す（エピソードのリセット・教師データの散らし直し）。"""
        self.crowd.relocate(self._active_vehicle_xy())

    def _active_vehicle_xy(self) -> np.ndarray:
        """走っている車両の座標 (K, 2)。歩行者を湧かせる場所を避けるのに使う。"""
        idx = np.flatnonzero(self.fleet.active)
        if idx.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return np.column_stack(
            (self.fleet.x[idx].astype(np.float64), self.fleet.y[idx].astype(np.float64))
        )

    def set_player(self, at: tuple[float, float] | None) -> None:
        """実用モードの徒歩キャラの位置を差し替える（None で消す）。"""
        if at is None:
            self._player_xy = _NO_PLAYER
            return
        x, y = float(at[0]), float(at[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            self._player_xy = _NO_PLAYER
            return
        self._player_xy = np.array([[x, y]], dtype=np.float64)

    @property
    def has_player(self) -> bool:
        """徒歩キャラが街に立っているか。"""
        return bool(self._player_xy.shape[0] > 0)

    @property
    def pedestrian_xy(self) -> np.ndarray:
        """いる歩行者の座標 (K, 2) float64。擬似カメラ・正解ラベル・衝突判定が使う。

        NPC 群衆に**実用モードの徒歩キャラを連結して**返す。ここが唯一の出典なので、
        混ぜるだけで擬似カメラ・正解ラベル・観測・車間・衝突判定の 5 つが同時に
        利用者を見るようになる（別々に足すと、どれかを足し忘れて気づけない）。
        """
        people = self.crowd.positions
        if self._player_xy.shape[0] == 0:
            return people
        if people.shape[0] == 0:
            return self._player_xy.copy()
        return np.vstack((people, self._player_xy))

    def next_signal(self, slot: int) -> tuple[float, int]:
        """そのスロットの前方にある直近の信号を (停止線までの距離 [m], 灯色) で返す。"""
        state = self.slots[slot]
        if state.signal_arcs.size == 0:
            return (float("inf"), RED)
        arc = float(self.arc[slot])
        ahead = np.searchsorted(state.signal_arcs, arc, side="left")
        if ahead >= state.signal_arcs.size:
            return (float("inf"), RED)
        distance = float(state.signal_arcs[ahead] - arc)
        index = int(state.signal_ids[ahead])
        phase = self.signal_phases[index] if 0 <= index < len(self.signal_phases) else RED
        return (distance, int(phase))

    def nth_signals(self, offset: int) -> tuple[np.ndarray, np.ndarray]:
        """前方 offset 番目（0 が直近）の信号を (距離 [m], 灯色) の配列で返す。"""
        n = config.MAX_VEHICLES
        distance = np.full(n, np.inf, dtype=np.float64)
        phase = np.full(n, RED, dtype=np.int8)
        phases = self.signal_phases
        for slot in range(n):
            state = self.slots[slot]
            size = state.signal_arcs.size
            if size == 0:
                continue
            arc = float(self.arc[slot])
            ahead = int(np.searchsorted(state.signal_arcs, arc, side="left")) + int(offset)
            if ahead >= size:
                continue
            distance[slot] = float(state.signal_arcs[ahead] - arc)
            index = int(state.signal_ids[ahead])
            if 0 <= index < len(phases):
                phase[slot] = phases[index]
        return distance, phase

    def next_signals(self) -> tuple[np.ndarray, np.ndarray]:
        """全スロットの前方直近の信号を (停止線までの距離 [m], 灯色) の配列で返す。"""
        return self.nth_signals(0)

    @staticmethod
    def _build_speed_profile(route: np.ndarray, cum: np.ndarray) -> np.ndarray:
        """経路の曲率から、各点で出してよい速度 [m/s] を作る。"""
        n = route.shape[0]
        if n < 3:
            return np.full(max(n, 0), np.float32(1e6), dtype=np.float32)

        a_lat = float(config.MAX_LATERAL_ACCEL)
        seg = np.diff(route, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        heading = np.arctan2(seg[:, 1], seg[:, 0])
        dtheta = np.arctan2(np.sin(np.diff(heading)), np.cos(np.diff(heading)))
        ds = np.maximum((seg_len[:-1] + seg_len[1:]) * 0.5, 1e-3)
        curvature = np.abs(dtheta) / ds

        limit = np.full(n, np.float32(1e6), dtype=np.float32)
        safe = np.sqrt(a_lat / np.maximum(curvature, 1e-6))
        limit[1:-1] = np.minimum(limit[1:-1], safe.astype(np.float32))

        brake = a_lat * 0.9
        for i in range(n - 2, -1, -1):
            step = max(float(cum[i + 1] - cum[i]), 1e-3)
            reachable = math.sqrt(float(limit[i + 1]) ** 2 + 2.0 * brake * step)
            if reachable < limit[i]:
                limit[i] = np.float32(reachable)
        return limit

    def curve_speed_limits(self) -> np.ndarray:
        """いまいる位置で出してよい速度 [m/s]（カーブ手前の減速を含む）。"""
        out = np.full(config.MAX_VEHICLES, np.inf, dtype=np.float64)
        arc = self.arc
        slots = self.slots
        for slot in np.flatnonzero(self.fleet.active):
            state = slots[slot]
            limits = state.route_speed_limit
            last = limits.size - 1
            if last < 0:
                continue
            idx = int(np.searchsorted(state.route_cum, arc[slot], side="left"))
            if idx > last:
                idx = last
            out[slot] = limits[idx]
        return out

    def posted_speed_limits(self) -> np.ndarray:
        """いまいる位置に適用されている規制速度 [m/s]（道交法 22 条の最高速度）。"""
        out = np.full(config.MAX_VEHICLES, np.inf, dtype=np.float64)
        arc = self.arc
        slots = self.slots
        for slot in np.flatnonzero(self.fleet.active):
            state = slots[slot]
            limits = state.sign_limits
            if limits.size == 0:
                continue
            idx = int(np.searchsorted(state.sign_arcs, arc[slot], side="right")) - 1
            if idx < 0:
                idx = 0
            out[slot] = limits[idx]
        return out

    def speed_violations(self) -> np.ndarray:
        """このステップで規制速度を超え「始めた」スロットを True で返す。"""
        out = np.zeros(config.MAX_VEHICLES, dtype=bool)
        limit = self.posted_speed_limits()
        speed = self.fleet.speed
        enter = np.float32(config.OVERSPEED_TOLERANCE_MPS)
        leave = np.float32(config.OVERSPEED_RETURN_MPS)

        for slot in np.flatnonzero(self.fleet.active):
            cap = limit[slot]
            state = self.slots[slot]
            if not np.isfinite(cap):
                state.over_speed = False
                continue
            over = float(speed[slot]) - float(cap)
            if not state.over_speed and over > enter:
                state.over_speed = True
                state.speed_violations += 1
                out[slot] = True
            elif state.over_speed and over < leave:
                state.over_speed = False
        return out

    def update_lane_departures(self) -> None:
        """車線を外れた「回数」を数える。"""
        lateral = np.abs(self.lateral)
        for slot in range(config.MAX_VEHICLES):
            if not self.fleet.active[slot]:
                continue
            state = self.slots[slot]
            if not state.outside_lane and lateral[slot] > LANE_DEPARTURE_M:
                state.outside_lane = True
                state.lane_departures += 1
            elif state.outside_lane and lateral[slot] < LANE_RETURN_M:
                state.outside_lane = False

    def route_progress_ratio(self) -> np.ndarray:
        """経路の達成度 0.0〜1.0 を返す（進行距離 / 経路全長）。"""
        total = self.route_total
        valid = total > np.float32(1e-6)
        out = np.zeros(config.MAX_VEHICLES, dtype=np.float32)
        np.divide(self.arc, total, out=out, where=valid)
        return np.clip(out, 0.0, 1.0, out=out)

    def signal_speed_limits(self) -> np.ndarray:
        """信号に従うための速度上限を返す（道路交通法施行令 2 条）。"""
        n = config.MAX_VEHICLES
        speed = self.fleet.speed
        brake = abs(config.MAX_DECEL)
        limit = np.full(n, np.inf, dtype=np.float64)

        for k in range(int(SIGNAL_LOOKAHEAD_COUNT)):
            distance_k, phase_k = self.nth_signals(k)
            if not np.isfinite(distance_k).any():
                break
            limit = np.minimum(
                limit,
                signal_speed_limit(distance_k, phase_k, speed, brake, config.DT),
            )
        return limit

    def update_yellow_commitment(self) -> None:
        """黄色で「安全に停止できない」と判断して通した信号を記録する。"""
        distance, phase = self.nth_signals(0)
        speed = self.fleet.speed
        brake = abs(config.MAX_DECEL)
        near_limit = signal_speed_limit(distance, phase, speed, brake, config.DT)

        for slot in np.flatnonzero(self.fleet.active):
            if phase[slot] != YELLOW or np.isfinite(near_limit[slot]):
                continue
            state = self.slots[slot]
            if state.signal_arcs.size == 0:
                continue
            idx = int(
                np.searchsorted(
                    state.signal_arcs,
                    float(self.arc[slot]) - config.SIGNAL_STOP_TOLERANCE_M,
                    side="right",
                )
            )
            if idx < state.signal_arcs.size:
                state.committed_signal = idx

    def signal_violations(self) -> np.ndarray:
        """このステップで赤信号のまま停止線を越えたスロットを True で返す。"""
        out = np.zeros(config.MAX_VEHICLES, dtype=bool)
        speed = self.fleet.speed
        for slot in np.flatnonzero(self.fleet.active):
            state = self.slots[slot]
            if state.signal_arcs.size == 0:
                continue

            plausible = float(speed[slot]) * config.DT * 2.0 + 0.5
            jumped = abs(float(self.arc_delta[slot])) > plausible

            arc = float(self.arc[slot]) - config.SIGNAL_STOP_TOLERANCE_M
            passed = int(np.searchsorted(state.signal_arcs, arc, side="right"))
            if passed > state.signals_passed:
                if jumped:
                    state.signals_passed = passed
                    continue
                for k in range(state.signals_passed, passed):
                    if k == state.committed_signal:
                        continue
                    index = int(state.signal_ids[k])
                    phase = (
                        self.signal_phases[index]
                        if 0 <= index < len(self.signal_phases)
                        else RED
                    )
                    if phase == RED:
                        out[slot] = True
                        state.violations += 1
                state.signals_passed = passed
            elif passed < state.signals_passed:
                state.signals_passed = max(passed, state.signals_floor)
        return out

    @staticmethod
    def _cumulative_length(route: np.ndarray) -> np.ndarray:
        """経路点ごとの累積距離 (K,) を返す。"""
        seg = np.diff(route, axis=0)
        lengths = np.hypot(seg[:, 0], seg[:, 1]).astype(np.float32)
        cum = np.zeros(route.shape[0], dtype=np.float32)
        np.cumsum(lengths, out=cum[1:])
        return cum

    def _install_route(
        self, slot: int, route: np.ndarray, *, keep_pose: bool = False
    ) -> bool:
        """経路をスロットに設定し、車両を経路始点に配置する。

        `keep_pose` のときは車体を動かさず経路だけ差し替える（実用モードの迎車・乗車後）。
        """
        if route is None or route.shape[0] < 2:
            return False
        state = self.slots[slot]
        state.route = np.ascontiguousarray(route, dtype=np.float32)
        state.route_cum = self._cumulative_length(state.route)
        self.route_total[slot] = (
            state.route_cum[-1] if state.route_cum.size >= 2 else np.float32(0.0)
        )
        state.route_progress = 0
        state.route_speed_limit = self._build_speed_profile(state.route, state.route_cum)
        state.arc_position = 0.0
        state.goal = (float(route[-1, 0]), float(route[-1, 1]))
        self._goal_x[slot] = np.float32(state.goal[0])
        self._goal_y[slot] = np.float32(state.goal[1])
        state.steps = 0
        state.collided = False
        state.reached_goal = False
        state.route_dirty = True

        if keep_pose:
            arc, lateral, tangent, seg = self._project_slot(slot)
            state.arc_position = arc
            state.route_progress = seg
            self.arc[slot] = np.float32(arc)
            self.lateral[slot] = np.float32(lateral)
            self.tangent[slot] = np.float32(tangent)
        else:
            heading = math.atan2(
                float(route[1, 1] - route[0, 1]), float(route[1, 0] - route[0, 0])
            )
            self.fleet.reset_slot(slot, float(route[0, 0]), float(route[0, 1]), heading)
            self.arc[slot] = np.float32(0.0)
            self.lateral[slot] = np.float32(0.0)
            self.tangent[slot] = np.float32(heading)
        self.collided_flags[slot] = False
        self.reached_flags[slot] = False
        self.stop_arc[slot] = np.inf

        try:
            stops = self.map_index.signals_on_route(
                [(float(px), float(py)) for px, py in state.route]
            )
        except Exception:
            stops = []
            if not self._signals_on_route_failed:
                self._signals_on_route_failed = True
                logger.exception("経路上の信号の取得に失敗しました。信号なしとして続行します")
        if stops:
            state.signal_arcs = np.array([a for a, _ in stops], dtype=np.float32)
            state.signal_ids = np.array([i for _, i in stops], dtype=np.int32)
        else:
            state.signal_arcs = np.zeros(0, dtype=np.float32)
            state.signal_ids = np.zeros(0, dtype=np.int32)
        state.signals_floor = int(
            np.searchsorted(
                state.signal_arcs,
                float(self.arc[slot]) + SPAWN_SIGNAL_SKIP_M,
                side="right",
            )
        )
        state.signals_passed = state.signals_floor
        try:
            limits = self.map_index.speed_limits_on_route(
                [(float(px), float(py)) for px, py in state.route]
            )
        except Exception:
            limits = []
            if not self._speed_limits_on_route_failed:
                self._speed_limits_on_route_failed = True
                logger.exception(
                    "経路上の規制速度の取得に失敗しました。規制なしとして続行します"
                )
        if limits:
            state.sign_arcs = np.array([a for a, _ in limits], dtype=np.float32)
            state.sign_limits = np.array([v for _, v in limits], dtype=np.float32)
        else:
            state.sign_arcs = np.zeros(0, dtype=np.float32)
            state.sign_limits = np.zeros(0, dtype=np.float32)

        state.committed_signal = -1
        state.violations = 0
        state.lane_departures = 0
        state.outside_lane = False
        state.speed_violations = 0
        state.over_speed = False
        return True

    def install_route(
        self, slot: int, route: np.ndarray, *, keep_pose: bool = False
    ) -> bool:
        """経路を差し替える（実用モードの配車から呼ぶ公開版）。"""
        slot = int(slot)
        if not (0 <= slot < config.MAX_VEHICLES):
            return False
        return self._install_route(slot, route, keep_pose=keep_pose)

    def set_stop_target(self, slot: int, arc: float) -> None:
        """その弧長で止まらせる（乗降地点）。赤信号と同じ式で速度上限を掛ける。"""
        slot = int(slot)
        if 0 <= slot < config.MAX_VEHICLES:
            self.stop_arc[slot] = float(arc)

    def clear_stop_target(self, slot: int) -> None:
        slot = int(slot)
        if 0 <= slot < config.MAX_VEHICLES:
            self.stop_arc[slot] = np.inf

    def stop_speed_limits(self) -> np.ndarray:
        """停車指示のある車両が出してよい速度 [m/s]。指示が無ければ inf。"""
        return stop_speed_limit(
            self.stop_arc - self.arc.astype(np.float64),
            abs(config.MAX_DECEL),
            config.DT,
            margin_m=0.0,
        )

    def respawn(self, slot: int, *, at: tuple[float, float] | None = None) -> None:
        """スロットを再スポーンする。経路生成に失敗した場合は前の経路を保つ。"""
        self.try_respawn(slot, at=at)

    def try_respawn(self, slot: int, *, at: tuple[float, float] | None = None) -> bool:
        """respawn の成否を返す版（activate から使う）。"""
        slot = int(slot)
        if not (0 <= slot < config.MAX_VEHICLES):
            return False
        route = (
            self._route_from_point(at[0], at[1])
            if at is not None
            else self._random_route(exclude=slot)
        )
        if route is None:
            route = self._random_route(exclude=slot)
        if route is None:
            return False
        return self._install_route(slot, route)

    def activate(self, slot: int, *, at: tuple[float, float] | None = None) -> bool:
        """スロットを起動する。経路が作れなければ False を返して非アクティブのまま。"""
        slot = int(slot)
        if not (0 <= slot < config.MAX_VEHICLES):
            return False
        if not self.try_respawn(slot, at=at):
            self.fleet.active[slot] = False
            return False
        self.fleet.active[slot] = True
        return True

    def deactivate(self, slot: int) -> None:
        """スロットを非アクティブ化する（観測・行動はマスクされる）。"""
        slot = int(slot)
        if not (0 <= slot < config.MAX_VEHICLES):
            return
        self.fleet.active[slot] = False
        self.fleet.speed[slot] = np.float32(0.0)
        self.fleet.steer[slot] = np.float32(0.0)
        self.collided_flags[slot] = False
        self.reached_flags[slot] = False
        self.stop_arc[slot] = np.inf

    def set_active_count(self, n: int) -> None:
        """先頭 n スロットをアクティブにし、残りを非アクティブにする。"""
        n = int(np.clip(int(n), 0, config.MAX_VEHICLES))
        for slot in range(config.MAX_VEHICLES):
            if slot < n:
                if not self.fleet.active[slot]:
                    self.activate(slot)
            elif self.fleet.active[slot]:
                self.deactivate(slot)

    @property
    def active_count(self) -> int:
        return int(np.count_nonzero(self.fleet.active))

    def first_inactive_slot(self) -> int | None:
        """空きスロットの先頭添字。無ければ None。"""
        idx = np.flatnonzero(~self.fleet.active)
        return int(idx[0]) if idx.size else None

    def _refresh_obstacle_cache(self) -> None:
        if self.obstacles:
            self._obstacle_xy = np.array(
                [(o.x, o.y) for o in self.obstacles], dtype=np.float32
            )
            self._obstacle_r = np.array([o.radius for o in self.obstacles], dtype=np.float32)
        else:
            self._obstacle_xy = np.zeros((0, 2), dtype=np.float32)
            self._obstacle_r = np.zeros(0, dtype=np.float32)

    @property
    def obstacle_xy(self) -> np.ndarray:
        """障害物座標のキャッシュ (M, 2) float32。"""
        return self._obstacle_xy

    def add_obstacle(self, x: float, y: float, radius: float) -> int | None:
        """障害物を追加して ID を返す。上限に達していれば None。"""
        if len(self.obstacles) >= config.MAX_OBSTACLES:
            return None
        radius = float(np.clip(float(radius), 0.1, 5.0))
        obstacle_id = self._next_obstacle_id
        self._next_obstacle_id += 1
        self.obstacles.append(ObstacleState(obstacle_id, float(x), float(y), radius))
        self._refresh_obstacle_cache()
        return obstacle_id

    def remove_obstacle(self, obstacle_id: int) -> bool:
        obstacle_id = int(obstacle_id)
        for i, obstacle in enumerate(self.obstacles):
            if obstacle.id == obstacle_id:
                del self.obstacles[i]
                self._refresh_obstacle_cache()
                return True
        return False

    def clear_obstacles(self) -> None:
        self.obstacles.clear()
        self._refresh_obstacle_cache()

    def _project_slot(self, slot: int) -> tuple[float, float, float, int]:
        """1 スロットを経路ポリラインへ射影する。"""
        state = self.slots[slot]
        route = state.route
        k = route.shape[0]
        if k < 2:
            return 0.0, 0.0, float(self.fleet.heading[slot]), 0

        hint = int(state.route_progress)
        lo = max(0, hint - PROJECT_WINDOW_BACK)
        hi = min(k - 1, hint + PROJECT_WINDOW_FWD)
        if hi <= lo:
            lo = max(0, hi - 1)
            hi = min(k - 1, lo + 1)

        a = route[lo:hi]
        b = route[lo + 1:hi + 1]
        ab = b - a
        seg_len2 = np.maximum((ab * ab).sum(axis=1), np.float32(1e-9))
        px = self.fleet.x[slot]
        py = self.fleet.y[slot]
        apx = px - a[:, 0]
        apy = py - a[:, 1]
        t = np.clip((apx * ab[:, 0] + apy * ab[:, 1]) / seg_len2, 0.0, 1.0)
        proj_x = a[:, 0] + ab[:, 0] * t
        proj_y = a[:, 1] + ab[:, 1] * t
        dx = px - proj_x
        dy = py - proj_y
        dist2 = dx * dx + dy * dy
        j = int(np.argmin(dist2))
        seg = lo + j

        seg_len = math.sqrt(float(seg_len2[j]))
        arc = float(state.route_cum[seg]) + float(t[j]) * seg_len
        tangent = math.atan2(float(ab[j, 1]), float(ab[j, 0]))
        lateral = -math.sin(tangent) * float(dx[j]) + math.cos(tangent) * float(dy[j])
        return arc, lateral, tangent, seg

    def project_all(self) -> np.ndarray:
        """全アクティブスロットを経路へ射影し、弧長の増分 [m] を返す。"""
        delta = np.zeros(config.MAX_VEHICLES, dtype=np.float32)
        for slot in range(config.MAX_VEHICLES):
            if not self.fleet.active[slot]:
                continue
            arc, lateral, tangent, seg = self._project_slot(slot)
            state = self.slots[slot]
            delta[slot] = np.float32(arc - state.arc_position)
            state.arc_position = arc
            state.route_progress = seg
            self.arc[slot] = np.float32(arc)
            self.lateral[slot] = np.float32(lateral)
            self.tangent[slot] = np.float32(tangent)
        self.arc_delta[:] = delta
        return delta

    def lookahead_points(self, offsets: np.ndarray) -> np.ndarray:
        """各スロットの現在弧長位置から offsets [m] 先の経路点。shape (N, P, 2)。"""
        offsets = np.asarray(offsets, dtype=np.float32)
        n = config.MAX_VEHICLES
        out = np.zeros((n, offsets.shape[0], 2), dtype=np.float32)
        for slot in np.flatnonzero(self.fleet.active):
            state = self.slots[slot]
            route = state.route
            if route.shape[0] < 2:
                out[slot, :, 0] = self.fleet.x[slot]
                out[slot, :, 1] = self.fleet.y[slot]
                continue
            targets = np.float32(state.arc_position) + offsets
            cum = state.route_cum
            out[slot, :, 0] = np.interp(targets, cum, route[:, 0])
            out[slot, :, 1] = np.interp(targets, cum, route[:, 1])
        return out

    def goal_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """全スロットの目的地座標。**内部配列をそのまま返すので書き換えないこと。**"""
        return self._goal_x, self._goal_y

    def episode_steps(self) -> np.ndarray:
        """各スロットのエピソード内ステップ数。"""
        return np.array([s.steps for s in self.slots], dtype=np.int32)

    def increment_steps(self) -> None:
        """アクティブスロットのエピソード内ステップ数を進める。"""
        for slot in range(config.MAX_VEHICLES):
            if self.fleet.active[slot]:
                self.slots[slot].steps += 1

    def set_braking(self, accel_cmd: np.ndarray) -> None:
        """制動指令が出ているスロットを記録する（ブレーキランプ）。

        **加速度の実測ではなく指令を見る。** エンジンブレーキや空気抵抗で減速しても
        実車のブレーキランプは点かない。
        """
        cmd = np.asarray(accel_cmd, dtype=np.float32)
        self.braking = (cmd < np.float32(BRAKE_COMMAND_THRESHOLD)) & self.fleet.active

    def update_turn_signals(self) -> None:
        """経路の先を見て方向指示器を出す。**`project_all()` の後に呼ぶこと**。

        道交法施行令 21 条にならい、右左折の 30m 手前から出す。**ヒステリシスが要る**
        （出すしきい値だけだと、緩いカーブを曲がっている間じゅう点いたり消えたりする）。
        """
        offsets = np.array(
            [TURN_LOOKAHEAD_M - TURN_TANGENT_SPAN_M, TURN_LOOKAHEAD_M + TURN_TANGENT_SPAN_M],
            dtype=np.float32,
        )
        pts = self.lookahead_points(offsets)
        dx = (pts[:, 1, 0] - pts[:, 0, 0]).astype(np.float64)
        dy = (pts[:, 1, 1] - pts[:, 0, 1]).astype(np.float64)
        ahead = np.arctan2(dy, dx)
        here = self.tangent.astype(np.float64)
        diff = np.arctan2(np.sin(ahead - here), np.cos(ahead - here))

        # 経路の終端では先読み点が重なって方位が出ない。そのときは消灯のまま
        degenerate = np.hypot(dx, dy) < 1e-3
        # ENU は反時計回りが正なので、方位が増える側が左折
        want = np.where(diff > 0.0, np.int8(-1), np.int8(1))
        magnitude = np.abs(diff)

        signal = self.turn_signal.copy()
        signal = np.where(magnitude >= TURN_ON_RAD, want, signal)
        signal = np.where(magnitude < TURN_OFF_RAD, np.int8(0), signal)
        signal = np.where(degenerate | ~self.fleet.active, np.int8(0), signal)
        self.turn_signal = signal.astype(np.int8)

    def set_event_flags(self, collided: np.ndarray, reached: np.ndarray) -> None:
        """描画用の衝突／到達フラグを設定する（respawn 後も 1 フレームだけ残す）。"""
        self.collided_flags = np.asarray(collided, dtype=bool).copy()
        self.reached_flags = np.asarray(reached, dtype=bool).copy()

    def check_collisions(self) -> np.ndarray:
        """建物・車両同士・障害物・歩行者の 4 種類をまとめて判定する。shape (N,) bool。

        歩行者との接触だけは `pedestrian_hits` にも残す（学習タブで分けて出すため）。
        """
        n = config.MAX_VEHICLES
        hit = np.zeros(n, dtype=bool)
        self.pedestrian_hits = np.zeros(n, dtype=bool)
        active = self.fleet.active
        idx = np.flatnonzero(active)
        if idx.size == 0:
            return hit

        hit |= self.map_index.collides_with_buildings(self.fleet.corners(), active)

        if idx.size >= 2:
            px = self.fleet.x[idx].astype(np.float64)
            py = self.fleet.y[idx].astype(np.float64)
            cos_h = np.cos(self.fleet.heading[idx].astype(np.float64))
            sin_h = np.sin(self.fleet.heading[idx].astype(np.float64))
            dx = px[None, :] - px[:, None]
            dy = py[None, :] - py[:, None]
            inside = _in_body_ellipse(
                dx,
                dy,
                cos_h[:, None],
                sin_h[:, None],
                VEHICLE_HIT_SEMI_LONG_M,
                VEHICLE_HIT_SEMI_LAT_M,
            )
            close = np.triu(inside | inside.T, k=1)
            if close.any():
                rows, cols = np.nonzero(close)
                hit[idx[rows]] = True
                hit[idx[cols]] = True

        if self._obstacle_xy.shape[0] > 0:
            px = self.fleet.x[idx].astype(np.float64)[:, None]
            py = self.fleet.y[idx].astype(np.float64)[:, None]
            cos_h = np.cos(self.fleet.heading[idx].astype(np.float64))[:, None]
            sin_h = np.sin(self.fleet.heading[idx].astype(np.float64))[:, None]
            radius = self._obstacle_r[None, :].astype(np.float64)
            hit[idx] |= _in_body_ellipse(
                self._obstacle_xy[None, :, 0] - px,
                self._obstacle_xy[None, :, 1] - py,
                cos_h,
                sin_h,
                np.float64(config.VEHICLE_LENGTH * 0.5) + radius,
                np.float64(config.VEHICLE_WIDTH * 0.5) + radius,
            ).any(axis=1)

        people = self.pedestrian_xy
        if people.shape[0] > 0:
            px = self.fleet.x[idx].astype(np.float64)[:, None]
            py = self.fleet.y[idx].astype(np.float64)[:, None]
            cos_h = np.cos(self.fleet.heading[idx].astype(np.float64))[:, None]
            sin_h = np.sin(self.fleet.heading[idx].astype(np.float64))[:, None]
            radius = np.float64(config.PEDESTRIAN_RADIUS)
            struck = _in_body_ellipse(
                people[None, :, 0] - px,
                people[None, :, 1] - py,
                cos_h,
                sin_h,
                np.float64(config.VEHICLE_LENGTH * 0.5) + radius,
                np.float64(config.VEHICLE_WIDTH * 0.5) + radius,
            ).any(axis=1)
            self.pedestrian_hits[idx] = struck
            hit[idx] |= struck

        return hit

    def snapshot(self, tick: int, sim_time: float, include_routes: bool) -> FrameSnapshot:
        """docs/protocol.md 2.3 の frame に対応するスナップショットを作る。"""
        vehicles: list[VehicleSnapshot] = []
        progress = self.route_progress_ratio()
        posted = self.posted_speed_limits()
        for slot in range(config.MAX_VEHICLES):
            state = self.slots[slot]
            emit_route = include_routes or state.route_dirty
            route: list[tuple[float, float]] | None = None
            if emit_route and state.route.shape[0] >= 2:
                route = [(float(px), float(py)) for px, py in state.route]
            if emit_route:
                state.route_dirty = False
            vehicles.append(
                VehicleSnapshot(
                    id=slot,
                    active=bool(self.fleet.active[slot]),
                    x=float(self.fleet.x[slot]),
                    y=float(self.fleet.y[slot]),
                    heading=float(self.fleet.heading[slot]),
                    speed=float(self.fleet.speed[slot]),
                    steer=float(self.fleet.steer[slot]),
                    collided=bool(self.collided_flags[slot]),
                    reached_goal=bool(self.reached_flags[slot]),
                    goal=(float(state.goal[0]), float(state.goal[1])),
                    progress=float(progress[slot]),
                    signal_violations=int(state.violations),
                    lane_departures=int(state.lane_departures),
                    speed_limit=(
                        float(posted[slot]) if np.isfinite(posted[slot]) else 0.0
                    ),
                    speed_violations=int(state.speed_violations),
                    braking=bool(self.braking[slot]),
                    turn_signal=int(self.turn_signal[slot]),
                    route=route,
                )
            )
        obstacles = [
            ObstacleSnapshot(id=o.id, x=o.x, y=o.y, radius=o.radius) for o in self.obstacles
        ]
        return FrameSnapshot(
            tick=int(tick),
            sim_time=float(sim_time),
            vehicles=vehicles,
            obstacles=obstacles,
            pedestrians=self.crowd.snapshot(),
        )
