"""MARL 環境本体（擬似固定エージェント数・parameter sharing 前提）。"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app import config
from app.contracts import (
    AssistDanger,
    DriveState,
    EpisodeResult,
    FrameSnapshot,
    InterventionEvent,
    MapIndex,
    SimParams,
    StepResult,
    TAXI_DRIVE_COMFORT,
    TAXI_DRIVE_HURRY,
    TAXI_DRIVE_NORMAL,
)
from app.percep.encoder import encode_observations
from app.percep.occlusion import RAY_ANGLES, RAY_HALF_WIDTH, CameraInput, evaluate_occlusion
from app.percep.types import (
    CAMERA_RIG,
    DEFAULT_CAMERA,
    REAR_CAMERA,
    SURROUND_CAMERAS,
    CameraSpec,
    DetClass,
    OcclusionResult,
    PerceptionResult,
)
from app.percep.weather import Weather, auto_weather
from app.sim.curriculum import (
    JAYWALK_AHEAD_M,
    JAYWALK_LATERAL_M,
    JAYWALK_PARALLEL_COS,
    KIND_JAYWALK,
    KIND_LEADER_BRAKE,
    CurriculumManager,
)
from app.sim.safety import FRESH_SEC as SAFETY_FRESH_SEC
from app.sim.safety import SafetyCommand, SafetySupervisor
from app.sim.signals import GREEN, RED, STOP_MARGIN_M, constrain_accel, stop_speed_limit
from app.sim.v2x import V2XMessageRouter
from app.sim.world import SPAWN_CLEARANCE_M2, World
from app.sim.scenario import Scenario, ScenarioTraffic, load_scenario

#: 1 ステップで再スポーンに使ってよい時間 [秒]。1 台ぶんは必ず処理するので、
#: これを超えたら残りは次のステップへ回す（金沢は 1 台 9.9ms、銀座は 1.6ms）
RESPAWN_BUDGET_SEC = 0.020

logger = logging.getLogger("autoware_sim")

__all__ = ["SimulationEnv"]

AUTOPILOT_LOOKAHEAD_MIN_M = 6.0
AUTOPILOT_LOOKAHEAD_SEC = 1.2
#: 経路から離れているときに詰める下限 [m]。遠くを見たままだと戻れずに膨らむ
AUTOPILOT_LOOKAHEAD_FLOOR_M = 3.0
#: これ以上曲がっている間は加速しない [rad]。
#: **ここでブレーキまで踏ませないこと**（銀座で交差点内に止まり、かえって事故が増えた）
AUTOPILOT_STRAIGHT_RAD = 0.22
#: この速度までは、切っていても加速する [m/s]（曲がりながら発進できるように）
AUTOPILOT_CREEP_MPS = 2.5
#: 目標速度（その地点の上限）との差 1 m/s あたりのアクセル。2 m/s 手前から絞り始める
AUTOPILOT_SPEED_GAIN = 0.5
#: アクセルを踏み込む／戻す速さの上限 [1/秒]。0 と 1 の間を 1 ステップで行き来させない
AUTOPILOT_PRESS_RATE = 2.0
AUTOPILOT_RELEASE_RATE = 5.0

AUTOPILOT_LANE_HALF_WIDTH_M = 2.4
AUTOPILOT_HEADWAY_M = 2.5
AUTOPILOT_LEAD_RANGE_M = 45.0
AUTOPILOT_SAME_WAY_COS = 0.5

#: 歩行者を見る範囲 [m] と、止まる相手と見なす車体中心からの横幅 [m]。
#: 歩道を歩いているだけの人で止まらないよう、車両より狭く取る
AUTOPILOT_PEDESTRIAN_RANGE_M = 30.0
AUTOPILOT_PEDESTRIAN_HALF_WIDTH_M = 2.0
#: 歩行者の手前で、バンパーから人の体までに空ける距離 [m]。前走車（2.5m）より広く取る
#: （人は急に向きを変える）
AUTOPILOT_PEDESTRIAN_CLEARANCE_M = 3.0
#: 上の距離を、車体の中心から歩行者の中心までに直したもの [m]（`_pedestrian_gap` は中心で測る）
AUTOPILOT_PEDESTRIAN_MARGIN_M = (
    config.VEHICLE_LENGTH / 2.0 + config.PEDESTRIAN_RADIUS + AUTOPILOT_PEDESTRIAN_CLEARANCE_M
)


@dataclass(frozen=True)
class DriveStyle:
    """配車中の車の経路追従の走り方。目標速度の割合・アクセルの利き・踏み込みの速さ・車間と歩行者の手前に足す余裕 [m]。"""

    speed_ratio: float
    speed_gain: float
    press_rate: float
    extra_headway_m: float
    extra_pedestrian_m: float


#: 走り方（`contracts.TAXI_DRIVE_MODES`）ごとの設定。どれも上限（constrain_accel）より速くは走らず、余裕は足すだけ
DRIVE_STYLES: dict[str, DriveStyle] = {
    TAXI_DRIVE_NORMAL: DriveStyle(1.0, AUTOPILOT_SPEED_GAIN, AUTOPILOT_PRESS_RATE, 0.0, 0.0),
    TAXI_DRIVE_HURRY: DriveStyle(1.0, 1.2, 3.0, 0.0, 0.0),
    TAXI_DRIVE_COMFORT: DriveStyle(0.8, 0.3, 1.0, 2.5, 1.0),
}

#: 速度上限がこれ以下まで抑えられていたら「交通に止められている」とみなす [m/s]
TRAFFIC_HOLD_MPS = 1.0

#: CNN で周囲カメラを回すとき、安全ギミックが要るカメラの待ち時間に掛ける重み。
#: 待ち時間は撮っていないとき inf になるので、比べるときはこの秒数で頭打ちにする
SURROUND_DEMAND_WEIGHT = 20.0
SURROUND_AGE_CAP_SEC = 2.0

#: 後退: これ以下の速さなら止まっているとみなしてギアを入れ替える [m/s]
REVERSE_SHIFT_MPS = 0.05
#: 後退の速さの差 1 m/s あたりのアクセルと、その上限（ゆっくり下がる）
REVERSE_GAIN = 0.6
REVERSE_MAX_THROTTLE = 0.5

# エキスパートが運転したステップの意図の教師（`option_labels`）。上から順に当てはまったものにする
OPTION_LABEL_RED_M = 30.0
OPTION_LABEL_PEDESTRIAN_M = 20.0
OPTION_LABEL_JUNCTION_M = 15.0
OPTION_LABEL_LEAD_M = 30.0


class SimulationEnv:
    """複数車両の物理更新・観測生成・報酬計算をまとめた環境。"""

    def __init__(
        self,
        map_index: MapIndex,
        params: SimParams,
        seed: int = 0,
        *,
        compute_observations: bool = True,
        scenario: Scenario | None = None,
        perception_mode: str | None = None,
        detector: Any | None = None,
        perception_transform: Callable[[PerceptionResult, CameraSpec], PerceptionResult] | None = None,
        deterministic_respawn: bool = False,
    ) -> None:
        if perception_mode not in (None, "oracle", "cnn"):
            raise ValueError("認識の評価モードは oracle / cnn のどちらかです")
        if detector is not None and perception_mode != "cnn":
            raise ValueError("評価用の認識器は cnn モードで指定してください")
        self._perception_mode = perception_mode
        self._evaluation_detector = detector
        self._perception_transform = perception_transform
        self._deterministic_respawn = bool(deterministic_respawn)
        self.map_index = map_index
        self.params = replace(params)
        self.rng = np.random.default_rng(seed)
        scenario_path = os.getenv("DRIVERL_SCENARIO", "").strip()
        if scenario_path and not Path(scenario_path).is_absolute():
            scenario_path = str(config.PROJECT_DIR / scenario_path)
        self.scenario = scenario if scenario is not None else (load_scenario(scenario_path) if scenario_path else None)
        self.world = World(
            map_index, self.rng,
            dynamics=self.scenario.dynamics if self.scenario else None,
            signal_plan=self.scenario.signals if self.scenario else None,
        )
        self.traffic: ScenarioTraffic | None = None

        n = config.MAX_VEHICLES
        self._obs = np.zeros((n, config.OBS_DIM), dtype=np.float32)
        # 介入・台数の変更で世界が変わったが、観測をまだ作り直していない
        self._obs_stale = False
        self._episode_lateral = np.zeros(n, dtype=np.float64)
        self._episode_reward = np.zeros(n, dtype=np.float32)
        # 下位方策の整形に使う、前のステップに実際に出した操作。エピソードの頭は「前」が無い
        self._prev_command = np.zeros((n, config.ACTION_DIM), dtype=np.float32)
        self._command_fresh = np.ones(n, dtype=bool)
        self._prev_accel = np.zeros(n, dtype=np.float64)
        #: 階層型の方策がいま選んでいる意図（`config.HRL_OPTIONS` の添字）。None なら frame に載せない
        self.current_options: np.ndarray | None = None

        # 実用モードで徴用している 1 台。**この間だけエピソードを閉じない**（決定 2）
        self.commandeered_slot: int = -1
        # 実用モードの間は全車を経路追従で走らせる。**学習中は必ず False**
        # （PPO から見た環境が変わってしまう）
        self.autopilot_all: bool = False
        # 配車中の車（`commandeered_slot`）の走り方。ほかの車には掛けない
        self.drive_style: str = TAXI_DRIVE_NORMAL
        # 直前のステップで経路追従が出した操作（上限で抑える前）。行動クローニングの教師に使う
        self.autopilot_actions = np.zeros((n, config.ACTION_DIM), dtype=np.float32)

        # 再スポーン待ちのスロット。経路生成が重いマップで 1 ステップに寄せない
        self._respawn_queue: list[int] = []

        self.latest_perception: dict[int, PerceptionResult] = {}
        self._latest_freespace: dict[int, np.ndarray] = {}
        # 周囲カメラ（後方・左・右）の認識結果と、それを撮った時刻（シミュレーション内の秒）。
        #   CNN で走るときは毎ステップ撮り直さないので、古さを添えて持つ
        self.latest_surround: dict[int, dict[str, PerceptionResult]] = {}
        #: (撮った時刻, そのときの経路の通し番号)。経路が変わったら古いものとして扱う
        self._surround_taken: dict[int, dict[str, tuple[float, int]]] = {}
        self._rear_free: dict[int, np.ndarray] = {}
        # 周囲カメラの走行可能距離（見通しと死角に使う）。認識結果と同じ時刻・通し番号で持つ
        self._surround_free: dict[int, dict[str, np.ndarray]] = {}
        # 4 台のカメラから作った見通しと死角（`percep/occlusion.py`）と、その観測の欄 (N, 8)
        self.latest_occlusion: dict[int, OcclusionResult] = {}
        self._occlusion_obs = np.zeros((n, config.OBS_OCCLUSION_DIM), dtype=np.float32)
        # 見通しと死角を frame に載せる車（`watch_occlusion`）
        self.watched_occlusion: frozenset[int] = frozenset()
        self.safety = SafetySupervisor()
        # ヒヤリハットのオートカリキュラム。環境の乱数とは別の乱数で回す（難易度 0 の間は 1 つも引かない）
        self.curriculum = CurriculumManager(np.random.default_rng([int(seed), 63]))
        # 車車間通信。受け取ったメッセージの平均は観測の末尾へ、受け取った相手は frame へ載せる
        self.v2x = V2XMessageRouter()
        self.v2x_links: dict[int, tuple[int, ...]] = {}
        # 周囲カメラの検出を frame に載せる車（`watch_surround`）。載せないと転送量が倍になる
        self.watched_surround: frozenset[int] = frozenset()
        self._camera_spec = DEFAULT_CAMERA
        self._camera: Any | None = None
        self._detector: Any | None = None
        self._ground_truth: Callable[..., list[PerceptionResult]] | None = None
        self._freespace_gt: Callable[..., np.ndarray] | None = None
        self._percep_ready = False
        self._observations_enabled = bool(compute_observations)
        self._detector_failed = False
        self._ground_truth_failed = False

        initial_count = int(self.params.vehicle_count)
        if self.scenario is not None:
            initial_count = self.scenario.learner_vehicles + self.scenario.background_vehicles
            self.params.vehicle_count = initial_count
        self.world.set_active_count(self.scenario.learner_vehicles if self.scenario else initial_count)
        self.world.set_pedestrian_count(int(self.params.pedestrian_count))
        if self.scenario is not None:
            self.traffic = ScenarioTraffic(self.scenario, seed, self)
            self.traffic.prepare(self, initial=True)
        self.world.project_all()
        self._obs = self._compute_observations()

    @property
    def active_mask(self) -> np.ndarray:
        """shape (MAX_VEHICLES,) bool。"""
        return self.world.fleet.active.copy()

    @property
    def observations(self) -> np.ndarray:
        """shape (MAX_VEHICLES, OBS_DIM) float32。介入の後なら、ここで 1 回だけ作り直す。"""
        self._refresh_observations()
        return self._obs.copy()

    def _refresh_observations(self) -> None:
        """介入の後で観測が古ければ作り直す（安全ギミックの判断も介入の後の世界で作り直す）。"""
        if self._obs_stale:
            self._obs = self._compute_observations()
            self._obs_stale = False

    @property
    def vehicle_count(self) -> int:
        """走っている台数。再スポーン待ちの車も数える（次のステップで起き上がるため）。"""
        return int(self.world.active_count) + len(self._respawn_queue)

    @property
    def fixed_vehicle_count(self) -> int | None:
        """シナリオ中の予約台数。通常の環境なら None。"""
        if self.scenario is None:
            return None
        return self.scenario.learner_vehicles + self.scenario.background_vehicles

    def _drop_from_respawn_queue(self, slot: int) -> bool:
        """再スポーン待ちから外す。積まれていたかを返す。"""
        if slot not in self._respawn_queue:
            return False
        self._respawn_queue = [s for s in self._respawn_queue if s != slot]
        return True

    def _free_slot(self) -> int | None:
        """新しく車を出せるスロット。再スポーン待ちの枠は次のステップで起き上がるので使わない。"""
        waiting = set(self._respawn_queue)
        for slot in np.flatnonzero(~self.world.fleet.active):
            if int(slot) not in waiting:
                return int(slot)
        return None

    def relocate_vehicle(self, slot: int, at: tuple[float, float] | None = None) -> bool:
        """スロットを指定地点（省略時はランダム）で起こし直す。成否を返す。"""
        slot = int(slot)
        if not (0 <= slot < config.MAX_VEHICLES):
            return False
        # 待ちに残すと、次のステップで待ちから起こし直されて別の場所へ飛ぶ
        self._drop_from_respawn_queue(slot)
        self.world.deactivate(slot)
        ok = self.world.activate(slot, at=at)
        self._reset_slot_stats(slot)
        self.params.vehicle_count = self.vehicle_count
        return ok

    def _autopilot(
        self, slot: int, target_speed: float, offset: float = 0.0
    ) -> tuple[float, float]:
        """経路の先を追う操作を返す（Pure Pursuit）。`offset` だけ経路を左へずらして追う（回避）。"""
        world = self.world
        state = world.slots[slot]
        route = state.route
        if route.shape[0] < 2:
            return 0.0, 0.0

        speed = float(world.fleet.speed[slot])
        off_route = abs(float(world.lateral[slot]) - float(offset))
        ahead = max(AUTOPILOT_LOOKAHEAD_MIN_M, speed * AUTOPILOT_LOOKAHEAD_SEC)
        # 経路から離れているほど近くを見る（遠くを見たままだと戻れない）
        ahead = max(AUTOPILOT_LOOKAHEAD_FLOOR_M, ahead - off_route * 2.0)

        target = np.float32(state.arc_position + ahead)
        tx = float(np.interp(target, state.route_cum, route[:, 0]))
        ty = float(np.interp(target, state.route_cum, route[:, 1]))
        if offset != 0.0:
            ahead_x = float(np.interp(target + 1.0, state.route_cum, route[:, 0]))
            ahead_y = float(np.interp(target + 1.0, state.route_cum, route[:, 1]))
            behind_x = float(np.interp(target - 1.0, state.route_cum, route[:, 0]))
            behind_y = float(np.interp(target - 1.0, state.route_cum, route[:, 1]))
            tangent = math.atan2(ahead_y - behind_y, ahead_x - behind_x)
            tx -= math.sin(tangent) * offset
            ty += math.cos(tangent) * offset

        heading = float(world.fleet.heading[slot])
        dx = tx - float(world.fleet.x[slot])
        dy = ty - float(world.fleet.y[slot])
        distance = math.hypot(dx, dy)
        want = math.atan2(dy, dx)
        alpha = math.atan2(math.sin(want - heading), math.cos(want - heading))

        curvature = 2.0 * math.sin(alpha) / max(distance, 1.0)
        steer = math.atan(config.WHEELBASE * curvature)
        # 曲がっている間は加速しない（突っ込むほど膨らむ）。
        # ただし止まっているときは必ず出すこと。切ったまま停まると、
        #   角度が変わらないので二度と発進できなくなる（銀座で 361 秒動かなくなった）
        straight = abs(alpha) < AUTOPILOT_STRAIGHT_RAD
        style = self._style(slot)
        want = 0.0
        if straight or speed < AUTOPILOT_CREEP_MPS:
            cruise = float(target_speed) * style.speed_ratio
            want = min(1.0, max(0.0, style.speed_gain * (cruise - speed)))
        # 踏み込みと戻しの速さを抑える。安全のための減速は constrain_accel が即座に掛ける
        prev = max(0.0, float(world.throttle[slot]))
        press = style.press_rate * config.DT
        release = AUTOPILOT_RELEASE_RATE * config.DT
        accel = prev + min(press, max(-release, want - prev))
        return accel, float(np.clip(steer / config.MAX_STEER, -1.0, 1.0))

    def _style(self, slot: int) -> DriveStyle:
        """そのスロットの経路追従の走り方。配車中の車だけが `drive_style` に従う。"""
        if int(slot) == int(self.commandeered_slot):
            return DRIVE_STYLES.get(self.drive_style, DRIVE_STYLES[TAXI_DRIVE_NORMAL])
        if self.traffic is not None and int(slot) in self.traffic.drivers:
            driver = self.traffic.drivers[int(slot)]
            return DriveStyle(driver.speed_ratio, AUTOPILOT_SPEED_GAIN, AUTOPILOT_PRESS_RATE, 0.0, 0.0)
        return DRIVE_STYLES[TAXI_DRIVE_NORMAL]

    def _headway_margin(self, slot: int) -> float:
        """前走車の手前で止める余裕 [m]（車体の中心から）。"""
        margin = config.VEHICLE_LENGTH + AUTOPILOT_HEADWAY_M + self._style(slot).extra_headway_m
        if self.traffic is not None and int(slot) in self.traffic.drivers:
            margin += abs(float(self.world.fleet.speed[slot])) * self.traffic.drivers[int(slot)].headway_sec
        return margin

    def _pedestrian_margin(self, slot: int) -> float:
        """歩行者の手前で止める余裕 [m]（車体の中心から歩行者の中心まで）。"""
        return AUTOPILOT_PEDESTRIAN_MARGIN_M + self._style(slot).extra_pedestrian_m

    def _autopilot_slots(self, active: np.ndarray) -> np.ndarray:
        """経路追従で走らせるスロット。実用モードでは全車、それ以外は徴用した 1 台だけ。"""
        if self.autopilot_all:
            return np.flatnonzero(active)
        background = np.flatnonzero(active & self.traffic.mask) if self.traffic is not None else np.zeros(0, dtype=np.int64)
        held = int(self.commandeered_slot)
        if 0 <= held < config.MAX_VEHICLES and active[held]:
            return np.union1d(background, [held]).astype(np.int64)
        return background

    def _assisted_slots(self, active: np.ndarray) -> np.ndarray:
        """安全ギミックを掛けるスロット。経路追従の車と、`safety_assist` のときは学習中の車も。"""
        if self.params.safety_assist:
            return np.flatnonzero(active)
        return self._autopilot_slots(active)

    def _lead_gap(self, slot: int) -> float:
        """前方の同じ進路上にいる他車までの車間 [m]。"""
        return self._lead_state(slot)[0]

    def _lead_state(self, slot: int) -> tuple[float, float, int]:
        """前方の同じ進路上にいる直近の他車の (車体の中心どうしの距離 [m], 自車の向きの速さ [m/s], スロット)。"""
        fleet = self.world.fleet
        idx = np.flatnonzero(fleet.active)
        if idx.size <= 1:
            return float("inf"), 0.0, -1
        heading = float(fleet.heading[slot])
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        dx = fleet.x[idx].astype(np.float64) - float(fleet.x[slot])
        dy = fleet.y[idx].astype(np.float64) - float(fleet.y[slot])
        lon = dx * cos_h + dy * sin_h
        lat = -dx * sin_h + dy * cos_h
        same_way = np.cos(fleet.heading[idx].astype(np.float64) - heading)
        ahead = (
            (idx != slot)
            & (lon > 0.0)
            & (lon <= AUTOPILOT_LEAD_RANGE_M)
            & (np.abs(lat) <= AUTOPILOT_LANE_HALF_WIDTH_M)
            & (same_way >= AUTOPILOT_SAME_WAY_COS)
        )
        if not ahead.any():
            return float("inf"), 0.0, -1
        nearest = int(np.flatnonzero(ahead)[np.argmin(lon[ahead])])
        lead_speed = float(fleet.speed[idx[nearest]]) * float(same_way[nearest])
        return float(lon[nearest]), lead_speed, int(idx[nearest])

    def assist_danger(self, slots: np.ndarray) -> AssistDanger:
        """オンライン模倣の危険の判定に使う真値（`rl/online_assist.py`）。`slots` 以外は危険なしで埋める。"""
        n = config.MAX_VEHICLES
        danger = AssistDanger.safe(n)
        world = self.world
        fleet = world.fleet
        for slot in np.flatnonzero(np.asarray(slots, dtype=bool).reshape(n)):
            s = int(slot)
            if not fleet.active[s]:
                continue
            speed = float(fleet.speed[s])
            danger.speed[s] = speed
            danger.lane_offset[s] = abs(float(world.lateral[s]))
            ttc = float("inf")
            gap, lead_speed, _leader = self._lead_state(s)
            closing = speed - lead_speed
            if math.isfinite(gap) and closing > 0.0:
                ttc = max(0.0, gap - config.VEHICLE_LENGTH) / closing
            person = self._pedestrian_gap(s)
            if math.isfinite(person) and speed > 0.0:
                reach = config.VEHICLE_LENGTH / 2.0 + config.PEDESTRIAN_RADIUS
                ttc = min(ttc, max(0.0, person - reach) / speed)
            danger.ttc[s] = ttc
            distance, phase = world.next_signal(s)
            if int(phase) == RED:
                danger.red_distance[s] = distance
            # 止められているかは、遅いときにだけ要る（固まりの判定）
            danger.held[s] = abs(speed) >= TRAFFIC_HOLD_MPS or self.traffic_hold(s)
        return danger

    def option_labels(self, slots: np.ndarray) -> np.ndarray:
        """いまの状況に合う意図（`config.HRL_OPTIONS` の添字）の真値。`slots` 以外は -1。エキスパートの教師に使う。"""
        n = config.MAX_VEHICLES
        labels = np.full(n, -1, dtype=np.int64)
        world = self.world
        for slot in np.flatnonzero(np.asarray(slots, dtype=bool).reshape(n)):
            s = int(slot)
            if not world.fleet.active[s]:
                continue
            distance, phase = world.next_signal(s)
            if self.traffic_hold(s) or (int(phase) == RED and distance <= OPTION_LABEL_RED_M):
                labels[s] = config.HRL_OPTIONS.index("STOP")
            elif (
                self._pedestrian_gap(s) <= OPTION_LABEL_PEDESTRIAN_M
                or world.next_junction(s) <= OPTION_LABEL_JUNCTION_M
            ):
                labels[s] = config.HRL_OPTIONS.index("YIELD")
            elif self._lead_gap(s) <= OPTION_LABEL_LEAD_M:
                labels[s] = config.HRL_OPTIONS.index("FOLLOW")
            else:
                labels[s] = config.HRL_OPTIONS.index("CRUISE")
        return labels

    def _drive_state(
        self, active: np.ndarray, command: np.ndarray, speed_before: np.ndarray
    ) -> DriveState:
        """1 ステップ後の走りの真値。前走車は方策が運転する車だけ引く（経路追従の車には要らない）。"""
        n = config.MAX_VEHICLES
        fleet = self.world.fleet
        lead_speed = np.full(n, np.nan, dtype=np.float64)
        for slot in np.flatnonzero(active):
            gap, speed, _leader = self._lead_state(int(slot))
            if math.isfinite(gap):
                lead_speed[int(slot)] = speed
        diff = command - self._prev_command
        fresh = self._command_fresh
        delta_sq = np.where(fresh, 0.0, np.sum(diff * diff, axis=1))
        accel = (fleet.speed.astype(np.float64) - speed_before) / config.DT
        jerk = np.where(fresh, 0.0, (accel - self._prev_accel) / config.DT)
        self._prev_command[:] = command
        self._prev_accel[:] = accel
        self._command_fresh[:] = ~np.asarray(fleet.active, dtype=bool)
        return DriveState(
            speed=fleet.speed.astype(np.float64),
            lateral=self.world.lateral.astype(np.float64),
            lead_speed=lead_speed,
            action_delta_sq=(delta_sq * active).astype(np.float64),
            jerk=(jerk * active).astype(np.float64),
            ground_speed=np.hypot(fleet.speed.astype(np.float64), fleet.lateral_speed.astype(np.float64)),
        )

    def _pedestrian_gap(self, slot: int) -> float:
        """前方の進路上にいる歩行者までの距離 [m]。"""
        people = self.world.pedestrian_xy
        if people.shape[0] == 0:
            return float("inf")
        fleet = self.world.fleet
        heading = float(fleet.heading[slot])
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        dx = people[:, 0] - float(fleet.x[slot])
        dy = people[:, 1] - float(fleet.y[slot])
        lon = dx * cos_h + dy * sin_h
        lat = -dx * sin_h + dy * cos_h
        ahead = (
            (lon > 0.0)
            & (lon <= AUTOPILOT_PEDESTRIAN_RANGE_M)
            & (np.abs(lat) <= AUTOPILOT_PEDESTRIAN_HALF_WIDTH_M)
        )
        if not ahead.any():
            return float("inf")
        return float(lon[ahead].min())

    def traffic_hold(self, slot: int) -> bool:
        """いま赤信号・前走車・歩行者に止められているか（`runtime/taxi.py` の停滞判定）。"""
        return self.hold_reason(slot) != ""

    def hold_reason(self, slot: int) -> str:
        """止められている理由（"safety" / "signal" / "lead_vehicle" / "pedestrian"）。止められていなければ空文字。"""
        slot = int(slot)
        if not (0 <= slot < config.MAX_VEHICLES):
            return ""
        decel = abs(config.MAX_DECEL)

        def held(gap: float, margin: float) -> bool:
            limit = stop_speed_limit(np.float64(gap), decel, config.DT, margin_m=margin)
            return float(limit) <= TRAFFIC_HOLD_MPS

        # 安全ギミックが止めている・切り返している間も、待たされているのと同じに扱う
        if self.safety.holding(slot):
            return "safety"
        distance, phase = self.world.next_signal(slot)
        if int(phase) != GREEN and held(distance, STOP_MARGIN_M):
            return "signal"
        if held(self._pedestrian_gap(slot), self._pedestrian_margin(slot)):
            return "pedestrian"
        if held(self._lead_gap(slot), self._headway_margin(slot)):
            return "lead_vehicle"
        return ""

    def set_player_pose(self, at: tuple[float, float] | None) -> None:
        """実用モードの徒歩キャラの位置を反映する（None で消す）。"""
        self.world.set_player(at)

    def commandeer_vehicle(self, slot: int) -> None:
        """実用モードの配車へ 1 台を徴用する。走行中のエピソードはここで打ち切る（決定 2）。"""
        slot = int(slot)
        if not (0 <= slot < config.MAX_VEHICLES):
            return
        self.commandeered_slot = slot
        self._reset_slot_stats(slot)

    def release_vehicle(self, slot: int) -> None:
        """徴用を解除し、その場から新しい目的地へ向かう新規エピソードに戻す（決定 3）。"""
        slot = int(slot)
        self.commandeered_slot = -1
        if not (0 <= slot < config.MAX_VEHICLES):
            return
        world = self.world
        world.clear_stop_target(slot)
        if not world.fleet.active[slot]:
            return
        route = world.route_onward(
            float(world.fleet.x[slot]),
            float(world.fleet.y[slot]),
            float(world.fleet.heading[slot]),
            float(world.fleet.speed[slot]),
        )
        if route is None or not world.install_route(slot, route, keep_pose=True):
            world.respawn(slot)
        self._reset_slot_stats(slot)

    def _reset_slot_stats(self, slot: int) -> None:
        """1 スロット分のエピソード統計を 0 に戻す。"""
        self._episode_reward[slot] = np.float32(0.0)
        self._episode_lateral[slot] = 0.0
        self._command_fresh[slot] = True

    def _reset_stats_for_changed(self, active_before: np.ndarray) -> None:
        """アクティブ状態が変わったスロットの統計を落とす。"""
        changed = np.flatnonzero(active_before != self.world.fleet.active)
        for slot in changed:
            self._reset_slot_stats(int(slot))

    def reset_all(self, *, relocate_walkers: bool = True) -> np.ndarray:
        """全アクティブスロットと再スポーン待ちのスロットを作り直し、観測を返す。"""
        waiting = set(self._respawn_queue)
        self._respawn_queue.clear()
        for slot in range(config.MAX_VEHICLES):
            if self.world.fleet.active[slot]:
                if self.traffic is not None and self.traffic.mask[slot]:
                    self.traffic.spawn(self, slot)
                else:
                    self.world.respawn(slot)
            elif slot in waiting:
                placed = self.traffic.spawn(self, slot) if self.traffic is not None and self.traffic.mask[slot] else self.world.activate(slot)
                if not placed:
                    logger.warning("スロット %d の再スポーンに失敗しました。非アクティブにします", slot)
        if self.scenario is None:
            self.params.vehicle_count = self.vehicle_count
        else:
            self.params.vehicle_count = self.scenario.learner_vehicles + self.scenario.background_vehicles
            self.traffic.prepare(self, initial=True)
        if relocate_walkers:
            self.world.relocate_pedestrians()
        self.curriculum.reset_incidents()
        self._episode_reward[:] = 0.0
        self._episode_lateral[:] = 0.0
        self._command_fresh[:] = True
        self.world.set_event_flags(
            np.zeros(config.MAX_VEHICLES, dtype=bool),
            np.zeros(config.MAX_VEHICLES, dtype=bool),
        )
        self._obs = self._compute_observations()
        self._obs_stale = False
        return self._obs.copy()

    @property
    def sim_time(self) -> float:
        """シミュレーション内の経過秒。信号の現示はこの時刻だけで決まる。"""
        return self.world.sim_time

    @property
    def weather(self) -> Weather:
        """いま効いている天候。`weather_auto` なら時刻から導き、パラメータ側は見ない。"""
        if self.params.weather_auto:
            return auto_weather(self.world.sim_time)
        return Weather(
            rain=float(self.params.weather_rain), fog=float(self.params.weather_fog)
        )

    @property
    def signal_phases(self) -> list[int]:
        """MapData.signals と同じ並びの灯色（0=青 / 1=黄 / 2=赤）。"""
        return self.world.signal_phases

    def _drain_respawn_queue(self) -> None:
        """再スポーン待ちを、時間の許すかぎり処理する（最低 1 台）。"""
        if not self._respawn_queue:
            return
        started = time.perf_counter()
        while self._respawn_queue:
            slot = self._respawn_queue.pop(0)
            if self.traffic is not None and self.traffic.mask[slot] and slot - self.scenario.learner_vehicles >= self.scenario.traffic_count(self.sim_time):
                continue
            placed = self.traffic.spawn(self, slot) if self.traffic is not None and self.traffic.mask[slot] else self.world.activate(slot)
            if not placed:
                if self.scenario is None:
                    self.params.vehicle_count = self.vehicle_count
                logger.warning(
                    "スロット %d の再スポーンに失敗しました（経路を作れず）。"
                    "このスロットを非アクティブにします",
                    slot,
                )
            if self.scenario is not None or (not self._deterministic_respawn and time.perf_counter() - started >= RESPAWN_BUDGET_SEC):
                break

    def step(self, actions: np.ndarray, expert: np.ndarray | None = None) -> StepResult:
        """1 ステップ進める。`expert` のスロットは方策の代わりに経路追従が運転する（オンライン模倣）。"""
        n = config.MAX_VEHICLES
        act = np.asarray(actions, dtype=np.float32).reshape(n, config.ACTION_DIM)
        act = np.clip(np.nan_to_num(act, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
        self._refresh_observations()

        active_before = self.world.fleet.active.copy()
        speed_before = self.world.fleet.speed.astype(np.float64)
        accel_cmd = np.where(active_before, act[:, 0], 0.0).astype(np.float32)
        steer_cmd = np.where(active_before, act[:, 1], 0.0).astype(np.float32)

        # 経路追従で走らせる車。**方策の実力に体験を左右させないため**で、
        # 実用モードでは街の車も止まったままにしない（詰まるとタクシーも来られない）
        piloted = self._autopilot_slots(active_before)
        # 安全ギミックを掛ける車と、操作を丸ごと引き受けている車（切り返し・回避）。
        #   指示は直前の `_compute_observations` の末尾で、表示中の検出枠から作ったもの
        assisted = self._assisted_slots(active_before)
        commands = self.safety.commands
        overridden = np.zeros(n, dtype=bool)
        for slot in assisted:
            overridden[int(slot)] = commands[int(slot)].override
        # オンライン模倣でエキスパートが運転する車。安全ギミックが引き受けている車は除く
        expert_mask = np.zeros(n, dtype=bool)
        if expert is not None:
            expert_mask = np.asarray(expert, dtype=bool).reshape(n) & active_before & ~overridden
            expert_mask[piloted] = False
        # エキスパートは経路追従そのものなので、前走車・歩行者の手前でも止める
        expert_options = self.option_labels(expert_mask) if expert_mask.any() else None
        driven = np.union1d(piloted, np.flatnonzero(expert_mask)).astype(np.int64)
        steered = np.union1d(driven, np.flatnonzero(overridden)).astype(np.int64)
        self._shift_out_of_reverse(active_before, assisted)

        params = self.params
        max_speed = max(float(params.max_speed), 1e-3)

        self.world.advance_time(config.DT)

        self.world.update_yellow_commitment()

        limit = self.world.curve_speed_limits()
        if params.obey_signals:
            limit = np.minimum(limit, self.world.signal_speed_limits())
        if params.obey_speed_signs:
            limit = np.minimum(limit, self.world.posted_speed_limits())
        limit = np.minimum(limit, self.world.stop_speed_limits())
        # 車間は経路追従の車にだけ掛ける。学習中の車に掛けると
        #   「追突しない世界」になり、PPO から見た環境が変わってしまう
        for slot in driven:
            limit[slot] = min(
                float(limit[slot]),
                float(
                    stop_speed_limit(
                        np.float64(self._lead_gap(int(slot))),
                        abs(config.MAX_DECEL),
                        config.DT,
                        margin_m=self._headway_margin(int(slot)),
                    )
                ),
                float(
                    stop_speed_limit(
                        np.float64(self._pedestrian_gap(int(slot))),
                        abs(config.MAX_DECEL),
                        config.DT,
                        margin_m=self._pedestrian_margin(int(slot)),
                    )
                ),
            )
        assist_mask = np.zeros(n, dtype=bool)
        assist_mask[assisted] = True
        for slot in assisted:
            s = int(slot)
            limit[s] = min(float(limit[s]), float(commands[s].speed_cap))
        # 経路追従は、下の constrain_accel と同じ上限を目標速度にして手前からアクセルを絞る
        self.autopilot_actions[:] = 0.0
        for slot in steered:
            s = int(slot)
            offset = float(commands[s].offset) if assist_mask[s] else 0.0
            accel_cmd[s], steer_cmd[s] = self._autopilot(
                s, min(float(limit[s]), max_speed), offset
            )
            if self.traffic is not None and s in self.traffic.drivers and not commands[s].override:
                accel_cmd[s], steer_cmd[s] = self.traffic.adjust_commands(s, float(accel_cmd[s]), float(steer_cmd[s]))
            if s in piloted:
                self.autopilot_actions[s, 0] = accel_cmd[s]
                self.autopilot_actions[s, 1] = steer_cmd[s]
        accel_cmd = constrain_accel(
            accel_cmd,
            self.world.fleet.speed,
            limit,
            config.DT,
            config.MAX_ACCEL,
            abs(config.MAX_DECEL),
        )
        # 後退とギアの切り替えは前進の上限（constrain_accel）の外で決める
        for slot in assisted:
            s = int(slot)
            if commands[s].reverse or int(self.world.fleet.gear[s]) < 0:
                accel_cmd[s], steer_cmd[s] = self._reverse_control(s, commands[s])

        # ヒヤリハットの急制動。前走車の操作を最大制動で上書きするので、その車のそのステップは学習に使わない
        forced_brake = self.curriculum.braking_mask() & active_before & ~overridden
        forced_brake[piloted] = False
        accel_cmd[forced_brake] = np.float32(-1.0)
        expert_mask &= ~forced_brake

        expert_actions = np.zeros((n, config.ACTION_DIM), dtype=np.float32)
        expert_mask &= self.world.fleet.gear >= 0
        expert_actions[expert_mask, 0] = accel_cmd[expert_mask]
        expert_actions[expert_mask, 1] = steer_cmd[expert_mask]
        if expert_options is not None:
            expert_options[~expert_mask] = -1

        self.world.set_braking(accel_cmd)
        self.world.fleet.step(accel_cmd, steer_cmd, config.DT, max_speed)

        delta = self.world.project_all()
        self.world.update_turn_signals()
        step_limit = np.float32(max_speed * config.DT * 2.0)
        delta = np.clip(delta, -step_limit, step_limit)

        collided = self.world.check_collisions() & active_before
        hit_pedestrian = self.world.pedestrian_hits & active_before
        lateral_abs = np.abs(self.world.lateral)
        offroad = (lateral_abs >= np.float32(config.OFFROAD_LIMIT)) & active_before

        goal_x, goal_y = self.world.goal_positions()
        goal_dist = np.hypot(
            self.world.fleet.x - goal_x, self.world.fleet.y - goal_y
        ).astype(np.float32)
        reached = (goal_dist <= np.float32(config.GOAL_RADIUS)) & active_before

        self.world.increment_steps()
        timeout = (self.world.episode_steps() >= config.MAX_EPISODE_STEPS) & active_before

        ran_red = self.world.signal_violations() & active_before
        over_speed = self.world.speed_violations() & active_before
        self.world.update_lane_departures()

        rewards = np.float32(params.reward_progress) * delta
        rewards += np.float32(params.reward_time)
        rewards += np.where(reached, np.float32(params.reward_goal), np.float32(0.0))
        rewards += np.where(collided, np.float32(params.reward_collision), np.float32(0.0))
        rewards += np.where(offroad, np.float32(params.reward_offroad), np.float32(0.0))
        rewards += np.where(ran_red, np.float32(params.reward_signal), np.float32(0.0))
        rewards += np.where(over_speed, np.float32(params.reward_overspeed), np.float32(0.0))
        self._episode_lateral += np.abs(self.world.lateral) * active_before
        rewards = (rewards * active_before).astype(np.float32)
        self._episode_reward += rewards

        policy_driven = active_before.copy()
        policy_driven[piloted] = False
        drive = self._drive_state(
            policy_driven,
            np.stack([accel_cmd, steer_cmd], axis=1).astype(np.float32),
            speed_before,
        )

        dones = (reached | collided | offroad | timeout) & active_before
        truncated = timeout & ~(reached | collided | offroad) & active_before
        held = int(self.commandeered_slot)
        if 0 <= held < n:
            # 徴用中の 1 台は乗降地点で止まるので、到達しても respawn させない
            # （させると乗客を置いて別の街区へ飛ぶ）。衝突・逸脱は配車側が拾う
            dones[held] = False
        truncated &= dones
        # 打ち切りの価値目標は「打ち切った時点の状態」から作る。再スポーンの後に
        # 評価すると、別のエピソードの初期状態の価値が混ざる（`rl/buffer.py`）
        final_obs = self._encode_last_perception() if bool(truncated.any()) else None

        episodes: list[EpisodeResult] = []
        for slot in np.flatnonzero(dones):
            slot = int(slot)
            if reached[slot]:
                reason = "goal"
            elif collided[slot]:
                reason = "collision"
            elif offroad[slot]:
                reason = "offroad"
            else:
                reason = "timeout"
            episodes.append(
                EpisodeResult(
                    slot=slot,
                    total_reward=float(self._episode_reward[slot]),
                    length=int(self.world.slots[slot].steps),
                    reason=reason,
                    signal_violations=int(self.world.slots[slot].violations),
                    speed_violations=int(self.world.slots[slot].speed_violations),
                    lane_departures=int(self.world.slots[slot].lane_departures),
                    lane_deviation=float(
                        self._episode_lateral[slot]
                        / max(1, int(self.world.slots[slot].steps))
                    ),
                    hit_pedestrian=bool(hit_pedestrian[slot]),
                )
            )
            self._reset_slot_stats(slot)
            self._respawn_queue.append(slot)
            self.world.deactivate(slot)

        self._drain_respawn_queue()
        self.world.set_event_flags(collided, reached)
        self._update_incidents(collided | offroad, dones, expert_mask, piloted)

        if self.traffic is not None:
            self.traffic.prepare(self)
            episodes = [episode for episode in episodes if not self.traffic.mask[episode.slot]]
        self._obs = self._compute_observations()
        self._obs_stale = False
        return StepResult(
            obs=self._obs.copy(),
            rewards=rewards,
            dones=dones,
            active=active_before,
            truncated=truncated,
            final_obs=final_obs,
            episodes=episodes,
            learn=active_before & ~overridden & ~forced_brake & ~(self.traffic.mask if self.traffic is not None else np.zeros(n, dtype=bool)),
            assisted=expert_mask,
            expert_actions=expert_actions,
            expert_options=expert_options,
            drive=drive,
        )

    def _incidents_enabled(self) -> bool:
        return bool(self.params.incident_curriculum) and not self.autopilot_all

    def _update_incidents(
        self,
        failed: np.ndarray,
        ended: np.ndarray,
        assisted: np.ndarray,
        piloted: np.ndarray,
    ) -> None:
        """起こしたヒヤリハットを見届け、条件のそろった車に新しく起こす（`sim/curriculum.py`）。"""
        curriculum = self.curriculum
        fleet = self.world.fleet
        curriculum.advance(failed, ended, assisted, fleet.active)
        if not self._incidents_enabled() or curriculum.incident_probability <= 0.0:
            return
        learners = fleet.active & ~np.asarray(ended, dtype=bool)
        learners[piloted] = False
        braking = curriculum.braking_mask()
        for slot in np.flatnonzero(learners):
            s = int(slot)
            speed = float(fleet.speed[s])
            center, _lead_speed, leader = self._lead_state(s)
            gap = center - config.VEHICLE_LENGTH
            following = leader >= 0 and curriculum.leader_geometry(gap, speed)
            rising = curriculum.update_following(s, following)
            if not curriculum.can_start(s):
                continue
            if (
                rising
                and learners[leader]
                and not braking[leader]
                and curriculum.should_trigger_leader_braking(gap, speed)
            ):
                curriculum.start(s, KIND_LEADER_BRAKE, leader)
                braking = curriculum.braking_mask()
                continue
            person = self._jaywalk_candidate(s, speed)
            if person >= 0:
                curriculum.mark_pedestrian(person)
                if curriculum.roll() and self.world.crowd.force_cross_street(
                    person, curriculum.dash_speed()
                ):
                    curriculum.start(s, KIND_JAYWALK)

    def _jaywalk_candidate(self, slot: int, speed: float) -> int:
        """飛び出させる歩行者を 1 人選ぶ（前方 8〜18m・横 2〜4m で、渡ると自車の進路を横切る人）。無ければ -1。"""
        if speed <= 0.0:
            return -1
        crowd = self.world.crowd
        # 動かす相手は NPC だけなので `pedestrian_xy`（利用者の徒歩キャラも並ぶ）ではなく群衆を直接見る
        idx = np.flatnonzero(crowd.active & ~crowd.crossing & ~crowd.waiting)
        if idx.size == 0:
            return -1
        fleet = self.world.fleet
        heading = float(fleet.heading[slot])
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        dx = crowd.x[idx] - float(fleet.x[slot])
        dy = crowd.y[idx] - float(fleet.y[slot])
        lon = dx * cos_h + dy * sin_h
        lat = -dx * sin_h + dy * cos_h
        near = (
            (lon >= JAYWALK_AHEAD_M[0])
            & (lon <= JAYWALK_AHEAD_M[1])
            & (np.abs(lat) >= JAYWALK_LATERAL_M[0])
            & (np.abs(lat) <= JAYWALK_LATERAL_M[1])
        )
        if not near.any():
            return -1
        pick = idx[near]
        lat = lat[near]
        lon = lon[near]
        edge = crowd.edge[pick]
        arc = np.clip(crowd.arc[pick], 0.0, crowd.net.length[edge])
        _cx, _cy, tx, ty = crowd.net.sample(edge, arc)
        # 道が自車と平行で、渡る向き（歩道の側から反対側へ）が自車の進路へ向かう人だけ
        parallel = np.abs(tx * cos_h + ty * sin_h) >= JAYWALK_PARALLEL_COS
        # 渡る向きは (-ty, tx) × 渡る先の側。自車の左向き (-sin, cos) との内積は、回転を打ち消すと t·h になる
        across = -crowd.side[pick].astype(np.float64)
        move_lat = across * (tx * cos_h + ty * sin_h)
        toward = np.sign(move_lat) == -np.sign(lat)
        ok = parallel & toward
        for k in np.flatnonzero(ok)[np.argsort(lon[ok])]:
            person = int(pick[k])
            if not self.curriculum.pedestrian_ready(person):
                continue
            if self.curriculum.jaywalk_geometry(float(lon[k]), float(lat[k]), speed):
                return person
        return -1

    def _reverse_control(self, slot: int, command: SafetyCommand) -> tuple[float, float]:
        """後退の操作（まっすぐ下がる）。前進から後退・後退から前進へは、止まってからギアを入れ替える。"""
        fleet = self.world.fleet
        gear = int(fleet.gear[slot])
        speed = float(fleet.speed[slot])
        want = -1 if command.reverse else 1
        if gear != want:
            if abs(speed) <= REVERSE_SHIFT_MPS:
                fleet.set_gear(slot, want)
                return 0.0, 0.0
            return -1.0, 0.0
        backing = max(0.0, -speed)
        target = max(0.0, float(command.reverse_cap))
        if target <= REVERSE_SHIFT_MPS or backing > target + REVERSE_SHIFT_MPS:
            # 0 へ向けた制動。上限まで落とすのに要る分だけ踏む（後退 AEB はここで止める）
            need = (backing - target) / config.DT / abs(config.MAX_DECEL)
            return -float(np.clip(need, 0.2, 1.0)), 0.0
        return float(np.clip(REVERSE_GAIN * (target - backing), 0.0, REVERSE_MAX_THROTTLE)), 0.0

    def _shift_out_of_reverse(self, active: np.ndarray, assisted: np.ndarray) -> None:
        """安全ギミックが外れた車（モードの切り替え・徴用の解除）が後退ギアのままなら、その場で止めて前進へ戻す。"""
        fleet = self.world.fleet
        backing = np.flatnonzero(active & (fleet.gear < 0))
        for slot in np.setdiff1d(backing, assisted):
            fleet.speed[int(slot)] = np.float32(0.0)
            fleet.set_gear(int(slot), 1)

    def apply_params(self, params: SimParams) -> None:
        """パラメータの実行時変更を反映する。学習は止めない。"""
        new_count = int(np.clip(int(params.vehicle_count), 0, config.MAX_VEHICLES))
        if self.scenario is not None:
            new_count = self.scenario.learner_vehicles + self.scenario.background_vehicles
        count_changed = new_count != (int(self.params.vehicle_count) if self.scenario is not None else self.vehicle_count)
        walkers = int(np.clip(int(params.pedestrian_count), 0, config.MAX_PEDESTRIANS))
        self.params = replace(params)
        self.params.vehicle_count = new_count
        self.params.pedestrian_count = walkers
        if walkers != int(self.world.crowd.count):
            self.world.set_pedestrian_count(walkers)
        if count_changed:
            # 待ちを残すと、減らしたはずのスロットが後から起き上がる
            self._respawn_queue.clear()
            active_before = self.world.fleet.active.copy()
            self.world.set_active_count(new_count)
            if self.traffic is not None:
                self.traffic.prepare(self)
            self._reset_stats_for_changed(active_before)
            self.world.project_all()
            self._obs_stale = True

    @staticmethod
    def _finite(payload: dict[str, Any], key: str, default: Any = None) -> float:
        """ペイロードから**有限な** float を取り出す。駄目なら ValueError。"""
        raw = payload[key] if default is None else payload.get(key, default)
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"{key} が有限な数値ではありません: {raw!r}")
        return value

    def apply_event(self, event: InterventionEvent) -> str | None:
        """ユーザー介入を適用する。失敗理由の文字列、成功なら None を返す。"""
        kind = str(event.kind)
        payload = event.payload or {}
        if self.scenario is not None and kind in ("spawn_vehicle", "despawn_vehicle"):
            return "シナリオ実行中の車両数は設定ファイルで指定してください"
        try:
            if kind == "spawn_vehicle":
                slot = self._free_slot()
                if slot is None:
                    return f"空きスロットがありません（最大 {config.MAX_VEHICLES} 台）"
                at = (self._finite(payload, "x"), self._finite(payload, "y"))
                # 利用者が選んだ地点は、再スポーンと同じ空きを求める（真後ろの車が追突する）
                why = self.world.activate_at(
                    slot, at, clearance_m=math.sqrt(SPAWN_CLEARANCE_M2)
                )
                if why is not None:
                    return why
                self._reset_slot_stats(slot)
                self.params.vehicle_count = self.vehicle_count

            elif kind == "despawn_vehicle":
                slot = int(self._finite(payload, "id"))
                if not (0 <= slot < config.MAX_VEHICLES):
                    return f"車両 ID が範囲外です: {slot}"
                waiting = self._drop_from_respawn_queue(slot)
                if not self.world.fleet.active[slot] and not waiting:
                    return f"車両 {slot} は既に非アクティブです"
                self.world.deactivate(slot)
                self._reset_slot_stats(slot)
                self.params.vehicle_count = self.vehicle_count

            elif kind == "add_obstacle":
                radius = self._finite(payload, "radius", config.OBSTACLE_RADIUS)
                obstacle_id = self.world.add_obstacle(
                    self._finite(payload, "x"), self._finite(payload, "y"), radius
                )
                if obstacle_id is None:
                    return f"障害物の数が上限に達しています（最大 {config.MAX_OBSTACLES} 個）"

            elif kind == "remove_obstacle":
                if not self.world.remove_obstacle(int(self._finite(payload, "id"))):
                    return f"障害物 {payload.get('id')} は存在しません"

            elif kind == "clear_obstacles":
                self.world.clear_obstacles()

            elif kind == "reset_episode":
                self.reset_all()
                return None

            else:
                return f"未知の介入イベントです: {kind}"

        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return f"介入イベントのペイロードが不正です: {exc}"

        # 作り直すのは読むとき・進めるときに 1 回だけ（届いた件数ぶん擬似カメラと推論を回さない）
        self._obs_stale = True
        return None

    def snapshot(self, tick: int, sim_time: float) -> FrameSnapshot:
        """描画用スナップショット。経路は変化があったスロットのみ載る。"""
        return self._decorate(self.world.snapshot(tick, sim_time, include_routes=False))

    def full_snapshot(self, tick: int, sim_time: float) -> FrameSnapshot:
        """新規接続クライアント向けに全スロットの経路を含めたスナップショット。"""
        return self._decorate(self.world.snapshot(tick, sim_time, include_routes=True))

    def _decorate(self, frame: FrameSnapshot) -> FrameSnapshot:
        """world の外にあるもの（認識結果・天候・安全ギミックの介入）を載せる。"""
        frame.detections = self._detections_wire()
        frame.surround = self._surround_wire()
        frame.occlusion = {
            slot: self.latest_occlusion[slot].to_wire(RAY_ANGLES, RAY_HALF_WIDTH)
            for slot in sorted(self.watched_occlusion)
            if slot in self.latest_occlusion
        }
        frame.weather = self.weather.to_wire(float(self._camera_spec.far))
        for vehicle in frame.vehicles:
            if vehicle.active:
                vehicle.assist = self.safety.assist(vehicle.id)
                vehicle.v2x_links = list(self.v2x_links.get(vehicle.id, ()))
                if self.current_options is not None and not self.autopilot_all and not (self.traffic is not None and self.traffic.mask[vehicle.id]):
                    option = int(self.current_options[vehicle.id])
                    if 0 <= option < len(config.HRL_OPTIONS):
                        vehicle.current_option = config.HRL_OPTIONS[option]
        return frame

    def _detections_wire(self) -> dict[int, list[dict[str, Any]]]:
        """直近の認識結果をワイヤ形式にする。"""
        return {slot: result.to_wire() for slot, result in self.latest_perception.items()}

    def _surround_wire(self) -> dict[int, dict[str, list[dict[str, Any]]]]:
        """購読されている車の周囲カメラの認識結果をワイヤ形式にする。"""
        out: dict[int, dict[str, list[dict[str, Any]]]] = {}
        for slot in self.watched_surround:
            cams = self.latest_surround.get(int(slot))
            if cams:
                out[int(slot)] = {key: result.to_wire() for key, result in cams.items()}
        return out

    @property
    def detector_active(self) -> bool:
        """観測が CNN 由来か（False なら真値フォールバック）。"""
        return self._detector is not None

    def reload_detector(self) -> bool:
        """`config.DETECTOR_PATH` を読み直して認識器を差し替える。成否を返す。"""
        self._percep_ready = False
        self._detector = None
        self._camera = None
        self._detector_failed = False
        self._ensure_percep()
        return self._detector is not None

    def _ensure_percep(self) -> None:
        """擬似カメラと認識器を用意する（1 度だけ走る）。"""
        if self._percep_ready:
            return
        self._percep_ready = True

        from app.percep.groundtruth import (
            detect_ground_truth_views,
            freespace_ground_truth_views,
        )

        self._ground_truth = detect_ground_truth_views
        self._freespace_gt = freespace_ground_truth_views

        if self._perception_mode == "oracle":
            return
        if self._perception_mode == "cnn":
            from app.percep.camera import PseudoCamera
            from app.percep.detector import Detector

            self._camera = PseudoCamera(self.map_index, self._camera_spec)
            self._detector = self._evaluation_detector or Detector.load(config.DETECTOR_PATH, self._camera_spec)
            if self._detector is None:
                raise ValueError("CNN の評価には読み込める学習済み認識器が必要です")
            return

        if not config.DETECTOR_PATH.exists():
            logger.info(
                "学習済みの認識器がありません（%s）。world の真値から作った"
                "理想の検出結果で代用します（train_detector.py で学習できます）",
                config.DETECTOR_PATH.name,
            )
            return

        try:
            from app.percep.camera import PseudoCamera

            self._camera = PseudoCamera(self.map_index, self._camera_spec)
        except Exception:
            logger.exception("擬似カメラの構築に失敗しました。真値で代用します")
            self._camera = None
            return

        try:
            from app.percep.detector import Detector

            self._detector = Detector.load(config.DETECTOR_PATH, self._camera_spec)
        except Exception:
            logger.exception("認識器の読み込みに失敗しました。真値で代用します")
            self._detector = None
        if self._detector is None:
            logger.warning(
                "認識器を読み込めませんでした（%s）。真値で代用します",
                config.DETECTOR_PATH.name,
            )
        else:
            logger.info("認識器を読み込みました: %s", config.DETECTOR_PATH.name)

    def _compute_observations(self) -> np.ndarray:
        """前方と周囲のカメラを描いて検出し、安全ギミックを評価してから観測ベクトルへ落とす。"""
        if not self._observations_enabled:
            self._clear_perception()
            return np.zeros((config.MAX_VEHICLES, config.OBS_DIM), dtype=np.float32)

        active = self.world.fleet.active
        idx = np.flatnonzero(active)
        if idx.size == 0:
            self._clear_perception()
            return np.zeros((config.MAX_VEHICLES, config.OBS_DIM), dtype=np.float32)

        self._ensure_percep()
        spec = self._camera_spec
        weather = self.weather
        freespace: dict[int, np.ndarray] = {}
        results: list[PerceptionResult] | None = None
        # このステップで撮り直した周囲カメラ（CNN は予算の分だけ、真値は全部）と、その走行可能距離
        fresh: dict[int, dict[str, PerceptionResult]] = {}
        fresh_free: dict[int, dict[str, np.ndarray]] = {}
        rear_free: dict[int, np.ndarray] = {}

        if self._detector is not None and self._camera is not None:
            try:
                results, freespace, fresh, fresh_free = self._detect_cnn(idx, weather)
                for slot, frees in fresh_free.items():
                    if REAR_CAMERA.key in frees:
                        rear_free[slot] = frees[REAR_CAMERA.key]
            except Exception:
                if self._perception_mode == "cnn":
                    raise
                self._detector = None
                self._camera = None
                if not self._detector_failed:
                    self._detector_failed = True
                    logger.exception(
                        "認識器の推論に失敗しました。以後は真値で代用します"
                        "（「モデル作成」タブで学習し直すと元に戻ります）"
                    )
                results = None
                freespace.clear()
                fresh.clear()
                fresh_free.clear()
                rear_free.clear()

        replace_all = False
        if results is None and (self._perception_mode == "oracle" or config.PERCEP_FALLBACK_GROUND_TRUTH):
            if self._ground_truth is not None and self._freespace_gt is not None:
                # 認識器を使う経路では CNN の出力をそのまま使う（画から判断させる）。
                # 視程で頭打ちにするのは真値で代用するこちら側だけ。
                reach = min(
                    float(config.OBS_FREESPACE_MAX_DISTANCE),
                    weather.visibility_m(float(spec.far)),
                )
                try:
                    results = []
                    rig = (spec, *SURROUND_CAMERAS)
                    # 4 台ぶんの走行可能距離を全車まとめて 1 回で作る（1 台 1 カメラずつと値は同じ）
                    frees = self._freespace_gt(self.world, [int(s) for s in idx], rig, reach)
                    for i, slot in enumerate(idx):
                        s = int(slot)
                        views = self._ground_truth(self.world, s, CAMERA_RIG, weather)
                        results.append(views[0])
                        fresh[s] = {
                            cam.key: view for cam, view in zip(CAMERA_RIG[1:], views[1:])
                        }
                        freespace[s] = frees[i, 0]
                        fresh_free[s] = {cam.key: frees[i, j + 1] for j, cam in enumerate(SURROUND_CAMERAS)}
                        # 安全ギミックに渡す後方の建物までの距離は、下がる前後の車の分だけ
                        if REAR_CAMERA.key in self.safety.demand(s):
                            rear_free[s] = fresh_free[s][REAR_CAMERA.key]
                    replace_all = True
                    if self._perception_mode == "oracle":
                        cam, slots = self._surround_schedule(idx)
                        chosen = set(slots)
                        fresh = {s: {cam.key: views[cam.key]} for s, views in fresh.items() if s in chosen}
                        fresh_free = {s: {cam.key: views[cam.key]} for s, views in fresh_free.items() if s in chosen}
                        rear_free = {s: views[REAR_CAMERA.key] for s, views in fresh_free.items() if REAR_CAMERA.key in views}
                        replace_all = False
                except Exception:
                    if self._perception_mode == "oracle":
                        raise
                    if not self._ground_truth_failed:
                        self._ground_truth_failed = True
                        logger.exception(
                            "真値からの検出生成に失敗しました。観測のカメラ欄は空になります"
                        )
                    results = None
                    freespace.clear()
                    fresh.clear()
                    fresh_free.clear()
                    rear_free.clear()

        perceptions: dict[int, PerceptionResult] = {}
        if results is not None:
            for slot, result in zip(idx, results):
                perceptions[int(slot)] = result

        if self._perception_transform is not None:
            perceptions = {slot: self._perception_transform(result, spec) for slot, result in perceptions.items()}
            cameras = {cam.key: cam for cam in SURROUND_CAMERAS}
            fresh = {slot: {key: self._perception_transform(result, cameras[key]) for key, result in views.items()} for slot, views in fresh.items()}

        self.latest_perception = perceptions
        self._latest_freespace = freespace
        self._merge_surround(idx, fresh, rear_free, replace_all, fresh_free)
        self._update_occlusion(idx)
        self._run_safety()
        return encode_observations(
            self.world,
            self.params,
            perceptions,
            freespace=freespace,
            spec=spec,
            surround=self.latest_surround,
            v2x=self._exchange_v2x(perceptions, spec),
            occlusion=self._occlusion_obs,
        )

    def _clear_perception(self) -> None:
        """認識していない（観測を作らない・走っている車がいない）ときに、前の認識結果を残さない。"""
        self.latest_perception = {}
        self.latest_surround = {}
        self._surround_free = {}
        self.latest_occlusion = {}
        self._occlusion_obs[:] = 0.0

    def _update_occlusion(self, idx: np.ndarray) -> None:
        """前方と周囲のカメラの検出・走行可能距離から、車ごとの見通しと死角を作る（`percep/occlusion.py`）。"""
        self.latest_occlusion = {}
        self._occlusion_obs[:] = 0.0
        if not self.latest_perception:
            return
        spec = self._camera_spec
        for slot in idx:
            s = int(slot)
            front = self.latest_perception.get(s)
            cams = self.latest_surround.get(s) or {}
            frees = self._surround_free.get(s) or {}
            views = [CameraInput(spec, front, self._latest_freespace.get(s) if front is not None else None)]
            views.extend(
                CameraInput(cam, cams.get(cam.key), frees.get(cam.key) if cam.key in cams else None)
                for cam in SURROUND_CAMERAS
            )
            result = evaluate_occlusion(views)
            self.latest_occlusion[s] = result
            self._occlusion_obs[s] = result.features

    def _exchange_v2x(self, perceptions: dict[int, PerceptionResult], spec: CameraSpec) -> np.ndarray:
        """V2X のメッセージを作って近くの車へ配り、受け取った平均 (N, 4) を返す（`sim/v2x.py`）。切っていれば 0。"""
        n = config.MAX_VEHICLES
        if not self.params.v2x_comm:
            self.v2x_links = {}
            return np.zeros((n, config.OBS_V2X_DIM), dtype=np.float32)
        fleet = self.world.fleet
        messages = self.v2x.compute_messages(
            self.world, float(self.params.max_speed), perceptions, self.latest_surround, spec
        )
        xy = np.column_stack((fleet.x, fleet.y))
        inbox, self.v2x_links = self.v2x.route_and_aggregate(xy, fleet.active, messages)
        return inbox

    def _detect_cnn(
        self, idx: np.ndarray, weather: Weather
    ) -> tuple[
        list[PerceptionResult],
        dict[int, np.ndarray],
        dict[int, dict[str, PerceptionResult]],
        dict[int, dict[str, np.ndarray]],
    ]:
        """前方は全車、周囲は予算の分だけ描いて、1 回の推論にまとめて通す。周囲は検出と走行可能距離を返す。"""
        assert self._camera is not None and self._detector is not None
        frame_index = int(self.world.sim_time * config.SIM_HZ)
        images = [self._camera.render(self.world, idx, weather, frame_index)]
        order: list[tuple[int, str]] = []
        cam, slots = self._surround_schedule(idx)
        if slots:
            images.append(
                self._camera.render(
                    self.world, np.asarray(slots, dtype=np.int64), weather, frame_index, spec=cam
                )
            )
            order.extend((s, cam.key) for s in slots)
        batch = images[0] if len(images) == 1 else np.concatenate(images, axis=0)
        slots_all = [int(s) for s in idx] + [s for s, _ in order]
        found, free_arr = self._detector.detect_with_freespace(batch, slots_all)

        n = int(idx.size)
        freespace = {int(slot): free_arr[i] for i, slot in enumerate(idx)}
        fresh: dict[int, dict[str, PerceptionResult]] = {}
        fresh_free: dict[int, dict[str, np.ndarray]] = {}
        for k, (slot, key) in enumerate(order):
            result = found[n + k]
            # 車線は前方カメラの意味（経路の先）しか持たないので、周囲の画からは捨てる
            result.detections = [d for d in result.detections if d.cls != DetClass.LANE]
            fresh.setdefault(slot, {})[key] = result
            fresh_free.setdefault(slot, {})[key] = free_arr[n + k]
        return found[:n], freespace, fresh, fresh_free

    def _surround_schedule(self, idx: np.ndarray) -> tuple[CameraSpec, list[int]]:
        """CNN に通す周囲カメラを 1 種類だけ選び（描画はカメラ 1 種類ごとに固定費が掛かる）、その車を古い順に予算まで返す。"""
        now = float(self.world.sim_time)
        budget = int(config.SURROUND_CNN_IMAGES_PER_STEP)
        assisted = set(int(s) for s in self._assisted_slots(self.world.fleet.active))
        best: tuple[float, CameraSpec, list[int]] | None = None
        for cam in SURROUND_CAMERAS:
            ranked: list[tuple[float, int]] = []
            for slot in idx:
                s = int(slot)
                age = self._surround_age(s, cam.key, now)
                wanted = s in assisted and cam.key in self.safety.demand(s)
                weight = SURROUND_DEMAND_WEIGHT if wanted else 1.0
                ranked.append((min(age, SURROUND_AGE_CAP_SEC) * weight, s))
            ranked.sort(key=lambda item: -item[0])
            chosen = ranked[: max(1, budget)]
            score = sum(p for p, _s in chosen)
            if best is None or score > best[0]:
                best = (score, cam, [s for _p, s in chosen])
        if best is None:
            return SURROUND_CAMERAS[0], []
        return best[1], best[2]

    def _surround_age(self, slot: int, key: str, now: float) -> float:
        """その周囲カメラを撮ってからの時間 [秒]。撮っていない・経路が変わった後なら inf。"""
        taken = self._surround_taken.get(slot, {}).get(key)
        if taken is None:
            return math.inf
        at, serial = taken
        if serial != int(self.world.route_serial[slot]):
            return math.inf
        return now - at

    def _merge_surround(
        self,
        idx: np.ndarray,
        fresh: dict[int, dict[str, PerceptionResult]],
        rear_free: dict[int, np.ndarray],
        replace_all: bool,
        fresh_free: dict[int, dict[str, np.ndarray]] | None = None,
    ) -> None:
        """撮り直した周囲カメラを取り込む。撮り直していないものは古さを添えたまま残す。"""
        now = float(self.world.sim_time)
        alive = {int(s) for s in idx}
        if replace_all:
            self.latest_surround = {}
            self._surround_taken = {}
            self._rear_free = {}
            self._surround_free = {}
        for slot in list(self.latest_surround):
            if slot not in alive:
                self.latest_surround.pop(slot, None)
                self._surround_taken.pop(slot, None)
                self._rear_free.pop(slot, None)
                self._surround_free.pop(slot, None)
                continue
            # 経路が変わった（再スポーンで別の場所へ移った）車の結果は、前の場所で写したもの
            serial = int(self.world.route_serial[slot])
            kept = self.latest_surround[slot]
            stamps = self._surround_taken.setdefault(slot, {})
            frees = self._surround_free.get(slot, {})
            for key in [k for k in kept if stamps.get(k, (0.0, -1))[1] != serial]:
                kept.pop(key, None)
                stamps.pop(key, None)
                frees.pop(key, None)
                if key == REAR_CAMERA.key:
                    self._rear_free.pop(slot, None)
            if not kept:
                self.latest_surround.pop(slot, None)
                self._surround_free.pop(slot, None)
        for slot, cams in fresh.items():
            serial = int(self.world.route_serial[slot])
            kept = self.latest_surround.setdefault(slot, {})
            stamps = self._surround_taken.setdefault(slot, {})
            for key, result in cams.items():
                kept[key] = result
                stamps[key] = (now, serial)
        for slot, frees in (fresh_free or {}).items():
            if slot in self.latest_surround:
                self._surround_free.setdefault(slot, {}).update(frees)
        for slot, free in rear_free.items():
            self._rear_free[slot] = free

    def _run_safety(self) -> None:
        """表示する検出枠から、次のステップの安全ギミックの指示を作る（枠に危険度も付ける）。"""
        now = float(self.world.sim_time)
        assisted = self._assisted_slots(self.world.fleet.active)
        ages = {
            slot: {key: self._surround_age(slot, key, now) for key in cams}
            for slot, cams in self.latest_surround.items()
        }
        rear_free = {
            slot: free
            for slot, free in self._rear_free.items()
            if self._surround_age(slot, REAR_CAMERA.key, now) <= SAFETY_FRESH_SEC
        }
        self.safety.evaluate(
            self.world,
            assisted,
            now,
            self.latest_perception,
            self.latest_surround,
            ages,
            rear_free,
            self.latest_occlusion,
        )

    def _encode_last_perception(self) -> np.ndarray:
        """直近の検出のまま、いまの世界の状態で観測だけ作り直す。"""
        if not self._observations_enabled:
            return np.zeros((config.MAX_VEHICLES, config.OBS_DIM), dtype=np.float32)
        return encode_observations(
            self.world,
            self.params,
            self.latest_perception,
            freespace=self._latest_freespace,
            spec=self._camera_spec,
            surround=self.latest_surround,
            v2x=self._exchange_v2x(self.latest_perception, self._camera_spec),
            occlusion=self._occlusion_obs,
        )
