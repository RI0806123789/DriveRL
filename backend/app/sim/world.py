"""マップ・車両・障害物・衝突判定を束ねるシミュレーション世界。

MapIndex（app.map が提供する実行時インデックス）へは contracts.MapIndex
プロトコル越しにのみ触る。実体は呼び出し側から注入される。

memo 5章の反映:
    - 建物との衝突判定はバックエンド側で行う（MapIndex.collides_with_building）
    - 擬似固定エージェント数（配列の先頭次元は常に MAX_VEHICLES）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import logging

import numpy as np

from app import config
from app.contracts import FrameSnapshot, MapIndex, ObstacleSnapshot, VehicleSnapshot
from app.sim.signals import RED, YELLOW, SignalController, signal_speed_limit
from app.sim.vehicle import VehicleFleet

__all__ = ["ObstacleState", "SlotState", "World"]


# --- 経路 ---
#: 出発地と目的地に要求する最小直線距離 [m]。短すぎるとエピソードが一瞬で終わる
ROUTE_MIN_DISTANCE_M = 120.0
#: スポーン地点から到達可能な目的地を探すときの試行回数
DESTINATION_TRIALS = 24
#: 経路への射影を探す窓（セグメント数）。
#: 後方は取りこぼし防止、前方は瞬間移動対策。
PROJECT_WINDOW_BACK = 8
PROJECT_WINDOW_FWD = 48

# --- スポーン ---
#: スポーン時に既存車両と空けたい距離 [m] の 2 乗。
#: 車両同士の衝突しきい値は中心間 VEHICLE_LENGTH*0.9 なので、その 1.5 倍を取る。
#: これが無いと同じノードから同時に出た車が即座に衝突扱いになり、
#: 方策に避けようのない罰（-100）が入る。
SPAWN_CLEARANCE_M2 = (config.VEHICLE_LENGTH * 1.5) ** 2
#: 空いている始点を探して経路を引き直す回数
SPAWN_ROUTE_TRIALS = 12
#: 出発地点からこの距離までにある信号は、通過済みとして数えない [m]
SPAWN_SIGNAL_SKIP_M = 3.0

# --- 信号 ---
#: 信号を何基先まで見て速度を決めるか。
#: 全交差点に信号を置くと連続する信号が 8m まで詰まる区間があり、
#: 直近 1 基しか見ないと物理的に止まれない場面が出る（通過の 10.6% が違反になった）。
SIGNAL_LOOKAHEAD_COUNT = 3

# --- 車線逸脱 ---
#: 実体は config.py にある（docs/protocol.md 2.3 に数値が明記されているため。
#: code_review B-22）。ここは別名。
LANE_DEPARTURE_M = config.LANE_DEPARTURE_M
LANE_RETURN_M = config.LANE_RETURN_M


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
    route_progress: int = 0          # 射影が乗っているセグメント添字（探索ヒント）
    arc_position: float = 0.0        # 経路始点からの累積進行距離 [m]
    goal: tuple[float, float] = (0.0, 0.0)
    steps: int = 0
    collided: bool = False
    reached_goal: bool = False
    route_dirty: bool = True         # True のフレームだけ snapshot に route を載せる
    # 経路上の各点で出してよい速度 [m/s]。カーブの曲率と、その手前で減速できる
    # ことの両方を満たすように作る（route_cum と同じ並び）
    route_speed_limit: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    # 経路が通過する信号。arc 昇順の (弧長 [m], MapData.signals の添字)
    signal_arcs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    signal_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    # 経路に適用される規制速度の区切り（arc 昇順）。弧長 a の規制速度は
    # 「a 以下で最後の区切り」の値。先頭は必ず弧長 0（出発地点の規制速度）
    sign_arcs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    sign_limits: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    # 直前のステップで越えていた信号の本数（越えた瞬間を検出するため）
    signals_passed: int = 0
    # 黄色で「安全に停止できない」と判断して進入を許した信号の添字。
    # 赤に変わってから停止線を越えても違反に数えない（施行令 2 条ただし書き）
    committed_signal: int = -1
    # このエピソード中の赤信号無視の回数
    violations: int = 0
    # このエピソード中に車線を外れた回数と、いま外れているか
    lane_departures: int = 0
    outside_lane: bool = False
    # このエピソード中に規制速度を超えた回数と、いま超えているか。
    # 車線逸脱と同じく「超え始めた瞬間」を 1 回と数える（毎ステップ数えない）
    speed_violations: int = 0
    over_speed: bool = False


logger = logging.getLogger("autoware_sim")


class World:
    """マップ上の車両群・障害物・衝突判定を保持する。"""

    def __init__(self, map_index: MapIndex, rng: np.random.Generator) -> None:
        self.map_index = map_index
        self.rng = rng
        self.fleet = VehicleFleet(config.MAX_VEHICLES)
        self.obstacles: list[ObstacleState] = []
        self.slots: list[SlotState] = [SlotState() for _ in range(config.MAX_VEHICLES)]

        self._next_obstacle_id = 0
        # 経路上の信号取得の失敗を 1 度だけログに出すためのフラグ（B-15）
        self._signals_on_route_failed = False
        # 経路上の規制速度取得の失敗も同じく初回だけ残す。
        # ★ 黙らせると「なぜか誰も速度を守らない」ではなく
        #   **「規制が無いので守る必要がない」＝成績が上がる方向**に症状が出る。
        self._speed_limits_on_route_failed = False
        n = config.MAX_VEHICLES

        # 信号の現示。観測にも報酬にも使うので、環境の中に置く。
        self.signals = SignalController(map_index.data.signals)
        self.sim_time = 0.0
        self.signal_phases: list[int] = self.signals.phases(0.0)

        # 経路射影のキャッシュ（observation と報酬の両方から参照する）
        self.arc = np.zeros(n, dtype=np.float32)       # 経路始点からの弧長 [m]
        # 経路の全長 [m]。route_progress_ratio を毎ステップ 64 回のループで
        # 計算し直さずに済むよう、経路を張るときにここへ控える
        self.route_total = np.zeros(n, dtype=np.float32)
        # 目的地の座標。毎回リスト内包で組み直すと 1 ステップに 2 回積み上がるので、
        # 経路を張り替えたときだけ更新する（code_review B-27）。
        # ★ step() の前半と _compute_observations() の 2 回呼びは**正しい**。
        #   間に respawn が入るので、見ている目的地が違う。まとめてはいけない。
        self._goal_x = np.zeros(n, dtype=np.float32)
        self._goal_y = np.zeros(n, dtype=np.float32)
        self.lateral = np.zeros(n, dtype=np.float32)   # 符号付き横方向偏差 [m]（左が正）
        self.tangent = np.zeros(n, dtype=np.float32)   # 経路の進行方位 [rad]
        # 直前ステップの弧長の増分。射影の飛びを見分けるのに使う
        self.arc_delta = np.zeros(n, dtype=np.float32)

        # 描画用イベントフラグ（env が毎ステップ上書きする）
        self.collided_flags = np.zeros(n, dtype=bool)
        self.reached_flags = np.zeros(n, dtype=bool)

        # 障害物の座標キャッシュ（観測のベクトル化用）
        self._obstacle_xy = np.zeros((0, 2), dtype=np.float32)
        self._obstacle_r = np.zeros(0, dtype=np.float32)

        # ノード座標のキャッシュ（目的地候補の抽出用）
        map_data = getattr(map_index, "data", None)
        node_list = getattr(map_data, "nodes", None) if map_data is not None else None
        if node_list:
            self._node_xy = np.array([(nd.x, nd.y) for nd in node_list], dtype=np.float32)
            self._node_ids = np.array([nd.id for nd in node_list], dtype=np.int64)
        else:
            self._node_xy = np.zeros((0, 2), dtype=np.float32)
            self._node_ids = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------
    # 目的地・経路の生成
    # ------------------------------------------------------------------

    def _route_from_nodes(self, src: int, dst: int) -> np.ndarray | None:
        """ノード ID の組から経路点列 (K, 2) を作る。失敗したら None。"""
        node_path = self.map_index.shortest_path(int(src), int(dst))
        if not node_path or len(node_path) < 2:
            return None
        # 中心線ではなく左側通行の車線に沿った経路を使う（道交法 17 条 4 項 / 34 条）
        pts = self.map_index.lane_route_polyline(node_path, config.ROUTE_RESAMPLE_M)
        if pts is None or len(pts) < 2:
            return None
        return np.asarray(pts, dtype=np.float32)

    def _start_is_clear(self, route: np.ndarray, exclude: int) -> bool:
        """経路の始点が、既に走っている車両と重なっていないか。

        経路を張ると車両は始点へ瞬間移動する。そこに別の車両がいると
        **次のステップで即座に衝突扱いになる**。方策には避けようのない
        衝突の罰（既定 -100）が入るので、学習をそのぶん歪める。
        台数が増えるほど重なる確率が上がり、64 台では無視できない。
        """
        if route is None or route.shape[0] < 1:
            return False
        active = self.fleet.active.copy()
        active[exclude] = False
        idx = np.flatnonzero(active)
        if idx.size == 0:
            return True
        dx = self.fleet.x[idx] - np.float32(route[0, 0])
        dy = self.fleet.y[idx] - np.float32(route[0, 1])
        return bool(np.min(dx * dx + dy * dy) >= SPAWN_CLEARANCE_M2)

    def _random_route(self, exclude: int = -1) -> np.ndarray | None:
        """ランダムな出発地・目的地の組から経路を作る。

        始点が既存車両と重なる経路は捨てて引き直す。空いた場所が見つからない
        場合は最後に引いた経路を返す（起動できないよりはましなため）。
        """
        fallback: np.ndarray | None = None
        for _ in range(SPAWN_ROUTE_TRIALS):
            src, dst = self.map_index.random_node_pair(self.rng, ROUTE_MIN_DISTANCE_M)
            if int(src) == int(dst):
                continue
            route = self._route_from_nodes(src, dst)
            if route is None:
                continue
            if fallback is None:
                fallback = route
            if exclude < 0 or self._start_is_clear(route, exclude):
                return route
        return fallback

    def _route_from_point(self, x: float, y: float) -> np.ndarray | None:
        """指定座標を道路にスナップし、そこから到達可能な目的地への経路を作る。"""
        try:
            snap_x, snap_y, _edge_id, _heading = self.map_index.nearest_road_point(
                float(x), float(y)
            )
        except Exception:
            # クリック地点が道路から遠いなど、正常な失敗もありうる経路。
            # 呼び出し側が「経路が見つかりません」と利用者へ返すので、
            # ここは debug に留める（毎クリックで例外を吐かせない）
            logger.debug("道路へのスナップに失敗しました: (%.1f, %.1f)", x, y, exc_info=True)
            return None
        src = int(self.map_index.nearest_node(float(snap_x), float(snap_y)))

        route: np.ndarray | None = None
        if self._node_xy.shape[0] > 1:
            dx = self._node_xy[:, 0] - np.float32(snap_x)
            dy = self._node_xy[:, 1] - np.float32(snap_y)
            dist2 = dx * dx + dy * dy
            far = np.flatnonzero(dist2 >= np.float32(ROUTE_MIN_DISTANCE_M * ROUTE_MIN_DISTANCE_M))
            if far.size == 0:
                # 十分遠いノードが無いマップでは距離条件を捨てる
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
            # スナップ点から到達可能な目的地が無い場合はランダム経路にフォールバックする
            return self._random_route()

        # スナップ点が経路先頭から離れていれば、つなぎの点として前置する
        if float(np.hypot(route[0, 0] - snap_x, route[0, 1] - snap_y)) > 0.5:
            head = np.array([[snap_x, snap_y]], dtype=np.float32)
            route = np.concatenate([head, route], axis=0)
        return route

    def advance_time(self, dt: float) -> None:
        """シミュレーション内時刻を進め、信号の現示を更新する。"""
        self.sim_time += float(dt)
        self.signal_phases = self.signals.phases(self.sim_time)

    def next_signal(self, slot: int) -> tuple[float, int]:
        """そのスロットの前方にある直近の信号を (停止線までの距離 [m], 灯色) で返す。

        前方に信号が無ければ (無限大, 赤) ではなく (inf, RED) を返さず、
        呼び出し側が「信号なし」を区別できるよう距離 inf を返す。
        """
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
        """前方 offset 番目（0 が直近）の信号を (距離 [m], 灯色) の配列で返す。

        前方にその数だけ信号が無いスロットは距離 inf。
        """
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
        """全スロットの前方直近の信号を (停止線までの距離 [m], 灯色) の配列で返す。

        観測にも信号順守の制約にも使うので、1 ステップに 1 回まとめて求める。
        前方に信号が無いスロットの距離は inf。
        """
        n = config.MAX_VEHICLES
        distance = np.full(n, np.inf, dtype=np.float64)
        phase = np.full(n, RED, dtype=np.int8)
        for slot in range(n):
            d, p = self.next_signal(slot)
            distance[slot] = d
            phase[slot] = p
        return distance, phase

    @staticmethod
    def _build_speed_profile(route: np.ndarray, cum: np.ndarray) -> np.ndarray:
        """経路の曲率から、各点で出してよい速度 [m/s] を作る。

        車両は速度に応じて舵角が制限される（`config.MAX_LATERAL_ACCEL`）ため、
        速いままカーブに入ると曲がりきれずに道路の外へ出てしまう。
        そこで「曲率から決まる速度」を求め、さらに**手前から減速して間に合うように
        後ろ向きに伝播**させる。実車の速度計画と同じ作り方。

        伝播が無いと、カーブの直前まで全速で走ってから急に上限が下がることになり、
        物理的に間に合わない。
        """
        n = route.shape[0]
        if n < 3:
            return np.full(max(n, 0), np.float32(1e6), dtype=np.float32)

        a_lat = float(config.MAX_LATERAL_ACCEL)
        # ヘディングの変化率から曲率を出す
        seg = np.diff(route, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        heading = np.arctan2(seg[:, 1], seg[:, 0])
        dtheta = np.arctan2(np.sin(np.diff(heading)), np.cos(np.diff(heading)))
        ds = np.maximum((seg_len[:-1] + seg_len[1:]) * 0.5, 1e-3)
        curvature = np.abs(dtheta) / ds  # 点 1..n-2 に対応

        limit = np.full(n, np.float32(1e6), dtype=np.float32)
        safe = np.sqrt(a_lat / np.maximum(curvature, 1e-6))
        limit[1:-1] = np.minimum(limit[1:-1], safe.astype(np.float32))

        # 後ろ向きに伝播: v[i] <= sqrt(v[i+1]^2 + 2*a*ds)
        brake = a_lat * 0.9  # 減速に使う値。旋回と同時に使い切らないよう少し残す
        for i in range(n - 2, -1, -1):
            step = max(float(cum[i + 1] - cum[i]), 1e-3)
            reachable = math.sqrt(float(limit[i + 1]) ** 2 + 2.0 * brake * step)
            if reachable < limit[i]:
                limit[i] = np.float32(reachable)
        return limit

    def curve_speed_limits(self) -> np.ndarray:
        """いまいる位置で出してよい速度 [m/s]（カーブ手前の減速を含む）。"""
        out = np.full(config.MAX_VEHICLES, np.inf, dtype=np.float64)
        # 非アクティブなスロットは見ない。64 台・8 倍では 1 ステップの予算が
        # 6.25ms しかないので、ここの Python オーバーヘッドも無視できない
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
        """いまいる位置に適用されている規制速度 [m/s]（道交法 22 条の最高速度）。

        標識が引けなかった経路のスロットは `inf`（規制なし）。

        ★ `next_signals()` と同じく、**1 ステップに複数回呼ぶのが正しい。**
          `env.step()` の前半（加速度制約のため／射影前の位置）、
          `speed_violations()`（射影後・respawn 前の位置）、
          `_compute_observations()`（respawn 後の位置）で見ている状態が違う。
          中身は searchsorted 1 回なのでまとめる利得は無い。
        """
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
        """このステップで規制速度を超え「始めた」スロットを True で返す。

        道交法 22 条 1 項（最高速度）。毎ステップ数えると、1 度超えただけで
        戻るまでのステップ数ぶん罰が入って報酬が支配されてしまうので、
        車線逸脱（`update_lane_departures`）と同じく**超え始めた瞬間**を
        1 回と数え、戻り判定はヒステリシスを付ける。
        """
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
        """車線を外れた「回数」を数える。

        毎ステップの外れ具合（`lateral`）ではなく、**外れ始めた瞬間**を 1 回と数える。
        しきい値の出入りを繰り返して回数が跳ね上がらないよう、戻り判定は
        少し内側（ヒステリシス）にしてある。
        """
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
        """経路の達成度 0.0〜1.0 を返す（進行距離 / 経路全長）。

        全長は `_install_route()` が `self.route_total` に控えてあるので、
        スロットごとのループを回さずに一度に求められる。
        """
        total = self.route_total
        valid = total > np.float32(1e-6)
        out = np.zeros(config.MAX_VEHICLES, dtype=np.float32)
        np.divide(self.arc, total, out=out, where=valid)
        return np.clip(out, 0.0, 1.0, out=out)

    def signal_speed_limits(self) -> np.ndarray:
        """信号に従うための速度上限を返す（道路交通法施行令 2 条）。

        黄色で「停止位置に近接していて安全に停止できない」と判断して通した信号は、
        赤に変わってから停止線を越えても違反ではない。ここでその判断を記録しておく。

        **直近 1 基だけでなく前方 `SIGNAL_LOOKAHEAD_COUNT` 基を見る。**
        全交差点に信号を置くと連続する信号の間隔が中央 46m・下位 5% で 8m まで詰まり、
        最高速から止まるのに要る 22m を 22.5% の区間が下回る。直近 1 基しか見ないと、
        1 基目を通過した瞬間に 8m 先の赤が現れて物理的に止まれず、赤信号無視になる。
        実際の運転者も数本先の信号を見て速度を決めているので、この方が現実にも近い。
        """
        n = config.MAX_VEHICLES
        speed = self.fleet.speed
        brake = abs(config.MAX_DECEL)
        limit = np.full(n, np.inf, dtype=np.float64)

        # 先読みする基数ぶん、順に「そこで止まれる速度」を求めて最も厳しいものを採る
        for k in range(int(SIGNAL_LOOKAHEAD_COUNT)):
            distance_k, phase_k = self.nth_signals(k)
            if not np.isfinite(distance_k).any():
                break
            limit = np.minimum(
                limit,
                signal_speed_limit(distance_k, phase_k, speed, brake, config.DT),
            )

        # 黄色の「止まれないので進む」判定は直近 1 基だけで行う（施行令 2 条ただし書き）
        distance, phase = self.next_signals()

        for slot in range(config.MAX_VEHICLES):
            if phase[slot] != YELLOW or np.isfinite(limit[slot]):
                continue
            # 黄色なのに制限が付かない＝止まれないので進んでよい、と判断した
            state = self.slots[slot]
            if state.signal_arcs.size == 0:
                continue
            idx = int(np.searchsorted(state.signal_arcs, float(self.arc[slot]), side="left"))
            if idx < state.signal_arcs.size:
                state.committed_signal = idx
        return limit

    def signal_violations(self) -> np.ndarray:
        """このステップで赤信号のまま停止線を越えたスロットを True で返す。

        道路交通法施行令 2 条の「赤色の灯火」は停止位置を越えて進行してはならない。
        黄色は安全に停止できない場合の進行が認められているので違反に数えない。
        """
        out = np.zeros(config.MAX_VEHICLES, dtype=bool)
        speed = self.fleet.speed
        for slot in range(config.MAX_VEHICLES):
            state = self.slots[slot]
            if state.signal_arcs.size == 0:
                continue

            # 経路から外れると射影が先のセグメントへ飛び、弧長が一気に進むことがある。
            # そのステップを「停止線を越えた」と数えると、止まっている車まで
            # 信号無視にされてしまう。実際に進める距離を超えた増分は通過とみなさない。
            plausible = float(speed[slot]) * config.DT * 2.0 + 0.5
            jumped = abs(float(self.arc_delta[slot])) > plausible

            arc = float(self.arc[slot]) - config.SIGNAL_STOP_TOLERANCE_M
            passed = int(np.searchsorted(state.signal_arcs, arc, side="right"))
            if passed > state.signals_passed:
                if jumped:
                    # 飛んだぶんは「越えた」ことにだけして、違反には数えない
                    state.signals_passed = passed
                    continue
                # 新たに越えた信号のうち、赤だったものがあれば違反
                for k in range(state.signals_passed, passed):
                    if k == state.committed_signal:
                        continue  # 黄色で進入を許した信号（施行令 2 条ただし書き）
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
                # 後退した場合は数え直す
                state.signals_passed = passed
        return out

    @staticmethod
    def _cumulative_length(route: np.ndarray) -> np.ndarray:
        """経路点ごとの累積距離 (K,) を返す。"""
        seg = np.diff(route, axis=0)
        lengths = np.hypot(seg[:, 0], seg[:, 1]).astype(np.float32)
        cum = np.zeros(route.shape[0], dtype=np.float32)
        np.cumsum(lengths, out=cum[1:])
        return cum

    # ------------------------------------------------------------------
    # スロットの初期化
    # ------------------------------------------------------------------

    def _install_route(self, slot: int, route: np.ndarray) -> bool:
        """経路をスロットに設定し、車両を経路始点に配置する。"""
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

        # この経路が通過する信号を求めておく（毎ステップ探すと重いので一度だけ）
        try:
            stops = self.map_index.signals_on_route(
                [(float(px), float(py)) for px, py in state.route]
            )
        except Exception:
            # 経路上の信号が引けなくても走行自体は続けられる（信号なし扱い）。
            # ただし黙ると「なぜか誰も信号を守らない」としか見えないので
            # 初回だけ残す（code_review B-15）
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
        # 出発地点のすぐ上にある信号は、走る前から「越えた」ことになってしまうので
        # 最初から通過済みとして扱う（スポーン直後の誤検出を防ぐ）。
        state.signals_passed = int(
            np.searchsorted(state.signal_arcs, SPAWN_SIGNAL_SKIP_M, side="right")
        )
        # この経路に適用される規制速度の区切りを求めておく（毎ステップ探すと重い）
        try:
            limits = self.map_index.speed_limits_on_route(
                [(float(px), float(py)) for px, py in state.route]
            )
        except Exception:
            # 規制速度が引けなくても走行自体は続けられる（規制なし扱い）。
            # ただし黙ると「速度超過が 0 件で成績が良い」という**気づけない形**で
            # 症状が出るので初回だけ残す（code_review B-15 と同じ理由）
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

        # 初期方位は経路の第 1 セグメント方向（heading 誤差ゼロで開始させる）
        heading = math.atan2(
            float(route[1, 1] - route[0, 1]), float(route[1, 0] - route[0, 0])
        )
        self.fleet.reset_slot(slot, float(route[0, 0]), float(route[0, 1]), heading)
        self.arc[slot] = np.float32(0.0)
        self.lateral[slot] = np.float32(0.0)
        self.tangent[slot] = np.float32(heading)
        self.collided_flags[slot] = False
        self.reached_flags[slot] = False
        return True

    def respawn(self, slot: int, *, at: tuple[float, float] | None = None) -> None:
        """スロットを再スポーンする。経路生成に失敗した場合は前の経路を保つ。"""
        self.try_respawn(slot, at=at)

    def try_respawn(self, slot: int, *, at: tuple[float, float] | None = None) -> bool:
        """respawn の成否を返す版（activate から使う）。"""
        slot = int(slot)
        if not (0 <= slot < config.MAX_VEHICLES):
            return False
        # at 指定（クリックでのスポーン）は利用者が置いた場所を尊重する。
        # ランダム起動のときだけ、既存車両と重ならない始点を選び直す。
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

    # ------------------------------------------------------------------
    # 障害物
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 経路への射影
    # ------------------------------------------------------------------

    def _project_slot(self, slot: int) -> tuple[float, float, float, int]:
        """1 スロットを経路ポリラインへ射影する。

        Returns:
            (arc, lateral, tangent, segment_index)
        """
        state = self.slots[slot]
        route = state.route
        k = route.shape[0]
        if k < 2:
            return 0.0, 0.0, float(self.fleet.heading[slot]), 0

        # 直前の射影位置の周辺だけを探索する（自己交差する経路での飛びを防ぐ）
        hint = int(state.route_progress)
        lo = max(0, hint - PROJECT_WINDOW_BACK)
        hi = min(k - 1, hint + PROJECT_WINDOW_FWD)  # セグメント添字は [lo, hi-1]
        if hi <= lo:
            lo = max(0, hi - 1)
            hi = min(k - 1, lo + 1)

        a = route[lo:hi]            # (S, 2) セグメント始点
        b = route[lo + 1:hi + 1]    # (S, 2) セグメント終点
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
        # 符号付き横方向偏差（経路進行方向から見て左が正）
        lateral = -math.sin(tangent) * float(dx[j]) + math.cos(tangent) * float(dy[j])
        return arc, lateral, tangent, seg

    def project_all(self) -> np.ndarray:
        """全アクティブスロットを経路へ射影し、弧長の増分 [m] を返す。

        self.arc / self.lateral / self.tangent を更新し、各スロットの
        arc_position と route_progress をコミットする。
        戻り値 shape (MAX_VEHICLES,) float32。後退した場合は負になる。
        """
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
        for slot in range(n):
            state = self.slots[slot]
            route = state.route
            if route.shape[0] < 2:
                out[slot, :, 0] = self.fleet.x[slot]
                out[slot, :, 1] = self.fleet.y[slot]
                continue
            targets = np.float32(state.arc_position) + offsets
            cum = state.route_cum
            # np.interp は範囲外を端点にクランプするので経路末尾以降は目的地に張り付く
            out[slot, :, 0] = np.interp(targets, cum, route[:, 0])
            out[slot, :, 1] = np.interp(targets, cum, route[:, 1])
        return out

    def goal_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """全スロットの目的地座標。**内部配列をそのまま返すので書き換えないこと。**

        値は `_install_route()` が張り替え時に更新する。
        """
        return self._goal_x, self._goal_y

    def episode_steps(self) -> np.ndarray:
        """各スロットのエピソード内ステップ数。"""
        return np.array([s.steps for s in self.slots], dtype=np.int32)

    def increment_steps(self) -> None:
        """アクティブスロットのエピソード内ステップ数を進める。"""
        for slot in range(config.MAX_VEHICLES):
            if self.fleet.active[slot]:
                self.slots[slot].steps += 1

    def set_event_flags(self, collided: np.ndarray, reached: np.ndarray) -> None:
        """描画用の衝突／到達フラグを設定する（respawn 後も 1 フレームだけ残す）。"""
        self.collided_flags = np.asarray(collided, dtype=bool).copy()
        self.reached_flags = np.asarray(reached, dtype=bool).copy()

    # ------------------------------------------------------------------
    # 衝突判定
    # ------------------------------------------------------------------

    def check_collisions(self) -> np.ndarray:
        """建物・車両同士・障害物の 3 種類をまとめて判定する。shape (N,) bool。"""
        n = config.MAX_VEHICLES
        hit = np.zeros(n, dtype=bool)
        active = self.fleet.active
        idx = np.flatnonzero(active)
        if idx.size == 0:
            return hit

        # --- 1. 建物（バックエンド側で厳密判定。memo 5章） ---
        # 1 台ずつではなくまとめて判定する。64 台で 4.54ms -> 0.22ms（約 20 倍速）。
        # 高倍速では 1 ステップの予算が 6.25ms（8 倍）しかないので、ここは必ずバッチで。
        hit |= self.map_index.collides_with_buildings(self.fleet.corners(), active)

        # --- 2. 車両同士（中心間距離による簡易判定・上三角のみ） ---
        if idx.size >= 2:
            px = self.fleet.x[idx]
            py = self.fleet.y[idx]
            dx = px[None, :] - px[:, None]
            dy = py[None, :] - py[:, None]
            dist2 = dx * dx + dy * dy
            threshold = np.float32((config.VEHICLE_LENGTH * 0.9) ** 2)
            close = np.triu(dist2 < threshold, k=1)
            if close.any():
                rows, cols = np.nonzero(close)
                hit[idx[rows]] = True
                hit[idx[cols]] = True

        # --- 3. 障害物 ---
        if self._obstacle_xy.shape[0] > 0:
            px = self.fleet.x[idx][:, None]
            py = self.fleet.y[idx][:, None]
            ox = self._obstacle_xy[None, :, 0]
            oy = self._obstacle_xy[None, :, 1]
            radius = np.float32(config.VEHICLE_WIDTH * 0.5) + self._obstacle_r[None, :]
            dist2 = (px - ox) ** 2 + (py - oy) ** 2
            hit[idx] |= (dist2 < radius * radius).any(axis=1)

        return hit

    # ------------------------------------------------------------------
    # スナップショット
    # ------------------------------------------------------------------

    def snapshot(self, tick: int, sim_time: float, include_routes: bool) -> FrameSnapshot:
        """docs/protocol.md 2.3 の frame に対応するスナップショットを作る。"""
        vehicles: list[VehicleSnapshot] = []
        progress = self.route_progress_ratio()
        # 標識が無い区間は 0.0 で送る（protocol.md 2.3「speedLimit」）。
        # inf は JSON にできないうえ、フロント側で「規制なし」と区別したいため
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
        )
