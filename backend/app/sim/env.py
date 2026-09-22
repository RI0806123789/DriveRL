"""MARL 環境本体（擬似固定エージェント数・parameter sharing 前提）。"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import replace
from typing import Any, Callable

import numpy as np

from app import config
from app.contracts import (
    EpisodeResult,
    FrameSnapshot,
    InterventionEvent,
    MapIndex,
    SimParams,
    StepResult,
)
from app.percep.encoder import encode_observations
from app.percep.types import DEFAULT_CAMERA, PerceptionResult
from app.percep.weather import Weather, auto_weather
from app.sim.signals import GREEN, STOP_MARGIN_M, constrain_accel, stop_speed_limit
from app.sim.world import World

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

AUTOPILOT_LANE_HALF_WIDTH_M = 2.4
AUTOPILOT_HEADWAY_M = 2.5
AUTOPILOT_LEAD_RANGE_M = 45.0
AUTOPILOT_SAME_WAY_COS = 0.5

#: 歩行者を見る範囲 [m] と、止まる相手と見なす車体中心からの横幅 [m]。
#: 歩道を歩いているだけの人で止まらないよう、車両より狭く取る
AUTOPILOT_PEDESTRIAN_RANGE_M = 30.0
AUTOPILOT_PEDESTRIAN_HALF_WIDTH_M = 2.0
#: 歩行者の手前で空ける距離 [m]。前走車より広く取る（人は急に向きを変える）
AUTOPILOT_PEDESTRIAN_MARGIN_M = 4.5

#: 速度上限がこれ以下まで抑えられていたら「交通に止められている」とみなす [m/s]
TRAFFIC_HOLD_MPS = 1.0


class SimulationEnv:
    """複数車両の物理更新・観測生成・報酬計算をまとめた環境。"""

    def __init__(
        self,
        map_index: MapIndex,
        params: SimParams,
        seed: int = 0,
        *,
        compute_observations: bool = True,
    ) -> None:
        self.map_index = map_index
        self.params = replace(params)
        self.rng = np.random.default_rng(seed)
        self.world = World(map_index, self.rng)

        n = config.MAX_VEHICLES
        self._obs = np.zeros((n, config.OBS_DIM), dtype=np.float32)
        self._episode_lateral = np.zeros(n, dtype=np.float64)
        self._episode_reward = np.zeros(n, dtype=np.float32)

        # 実用モードで徴用している 1 台。**この間だけエピソードを閉じない**（決定 2）
        self.commandeered_slot: int = -1
        # 実用モードの間は全車を経路追従で走らせる。**学習中は必ず False**
        # （PPO から見た環境が変わってしまう）
        self.autopilot_all: bool = False

        # 再スポーン待ちのスロット。経路生成が重いマップで 1 ステップに寄せない
        self._respawn_queue: list[int] = []

        self.latest_perception: dict[int, PerceptionResult] = {}
        self._camera_spec = DEFAULT_CAMERA
        self._camera: Any | None = None
        self._detector: Any | None = None
        self._ground_truth: Callable[..., list[PerceptionResult]] | None = None
        self._freespace_gt: Callable[..., np.ndarray] | None = None
        self._percep_ready = False
        self._observations_enabled = bool(compute_observations)
        self._detector_failed = False
        self._ground_truth_failed = False

        self.world.set_active_count(int(self.params.vehicle_count))
        self.world.set_pedestrian_count(int(self.params.pedestrian_count))
        self.world.project_all()
        self._obs = self._compute_observations()

    @property
    def active_mask(self) -> np.ndarray:
        """shape (MAX_VEHICLES,) bool。"""
        return self.world.fleet.active.copy()

    @property
    def observations(self) -> np.ndarray:
        """shape (MAX_VEHICLES, OBS_DIM) float32。"""
        return self._obs.copy()

    def relocate_vehicle(self, slot: int, at: tuple[float, float] | None = None) -> bool:
        """スロットを指定地点（省略時はランダム）で起こし直す。成否を返す。"""
        slot = int(slot)
        if not (0 <= slot < config.MAX_VEHICLES):
            return False
        self.world.deactivate(slot)
        ok = self.world.activate(slot, at=at)
        self._reset_slot_stats(slot)
        self.params.vehicle_count = self.world.active_count
        return ok

    def _autopilot(self, slot: int) -> tuple[float, float]:
        """経路の先を追う操作を返す（Pure Pursuit）。"""
        world = self.world
        state = world.slots[slot]
        route = state.route
        if route.shape[0] < 2:
            return 0.0, 0.0

        speed = float(world.fleet.speed[slot])
        off_route = abs(float(world.lateral[slot]))
        ahead = max(AUTOPILOT_LOOKAHEAD_MIN_M, speed * AUTOPILOT_LOOKAHEAD_SEC)
        # 経路から離れているほど近くを見る（遠くを見たままだと戻れない）
        ahead = max(AUTOPILOT_LOOKAHEAD_FLOOR_M, ahead - off_route * 2.0)

        target = np.float32(state.arc_position + ahead)
        tx = float(np.interp(target, state.route_cum, route[:, 0]))
        ty = float(np.interp(target, state.route_cum, route[:, 1]))

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
        accel = 1.0 if (straight or speed < AUTOPILOT_CREEP_MPS) else 0.0
        return accel, float(np.clip(steer / config.MAX_STEER, -1.0, 1.0))

    def _autopilot_slots(self, active: np.ndarray) -> np.ndarray:
        """経路追従で走らせるスロット。実用モードでは全車、それ以外は徴用した 1 台だけ。"""
        if self.autopilot_all:
            return np.flatnonzero(active)
        held = int(self.commandeered_slot)
        if 0 <= held < config.MAX_VEHICLES and active[held]:
            return np.array([held], dtype=np.int64)
        return np.zeros(0, dtype=np.int64)

    def _lead_gap(self, slot: int) -> float:
        """前方の同じ進路上にいる他車までの車間 [m]。"""
        fleet = self.world.fleet
        idx = np.flatnonzero(fleet.active)
        if idx.size <= 1:
            return float("inf")
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
            return float("inf")
        return float(lon[ahead].min())

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
        slot = int(slot)
        if not (0 <= slot < config.MAX_VEHICLES):
            return False
        decel = abs(config.MAX_DECEL)

        def held(gap: float, margin: float) -> bool:
            limit = stop_speed_limit(np.float64(gap), decel, config.DT, margin_m=margin)
            return float(limit) <= TRAFFIC_HOLD_MPS

        distance, phase = self.world.next_signal(slot)
        if int(phase) != GREEN and held(distance, STOP_MARGIN_M):
            return True
        if held(self._lead_gap(slot), config.VEHICLE_LENGTH + AUTOPILOT_HEADWAY_M):
            return True
        return held(self._pedestrian_gap(slot), AUTOPILOT_PEDESTRIAN_MARGIN_M)

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
        )
        if route is None or not world.install_route(slot, route, keep_pose=True):
            world.respawn(slot)
        self._reset_slot_stats(slot)

    def _reset_slot_stats(self, slot: int) -> None:
        """1 スロット分のエピソード統計を 0 に戻す。"""
        self._episode_reward[slot] = np.float32(0.0)
        self._episode_lateral[slot] = 0.0

    def _reset_stats_for_changed(self, active_before: np.ndarray) -> None:
        """アクティブ状態が変わったスロットの統計を落とす。"""
        changed = np.flatnonzero(active_before != self.world.fleet.active)
        for slot in changed:
            self._reset_slot_stats(int(slot))

    def reset_all(self, *, relocate_walkers: bool = True) -> np.ndarray:
        """全アクティブスロットを再スポーンし、観測を返す。"""
        self._respawn_queue.clear()
        for slot in range(config.MAX_VEHICLES):
            if self.world.fleet.active[slot]:
                self.world.respawn(slot)
        if relocate_walkers:
            self.world.relocate_pedestrians()
        self._episode_reward[:] = 0.0
        self._episode_lateral[:] = 0.0
        self.world.set_event_flags(
            np.zeros(config.MAX_VEHICLES, dtype=bool),
            np.zeros(config.MAX_VEHICLES, dtype=bool),
        )
        self._obs = self._compute_observations()
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
            if not self.world.activate(slot):
                self.params.vehicle_count = self.world.active_count
                logger.warning(
                    "スロット %d の再スポーンに失敗しました（経路を作れず）。"
                    "このスロットを非アクティブにします",
                    slot,
                )
            if time.perf_counter() - started >= RESPAWN_BUDGET_SEC:
                break

    def step(self, actions: np.ndarray) -> StepResult:
        """1 ステップ進める。終了したスロットは再スポーン待ちへ積む（`_drain_respawn_queue`）。"""
        n = config.MAX_VEHICLES
        act = np.asarray(actions, dtype=np.float32).reshape(n, config.ACTION_DIM)
        act = np.clip(np.nan_to_num(act, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)

        active_before = self.world.fleet.active.copy()
        accel_cmd = np.where(active_before, act[:, 0], 0.0).astype(np.float32)
        steer_cmd = np.where(active_before, act[:, 1], 0.0).astype(np.float32)

        # 経路追従で走らせる車。**方策の実力に体験を左右させないため**で、
        # 実用モードでは街の車も止まったままにしない（詰まるとタクシーも来られない）
        piloted = self._autopilot_slots(active_before)
        for slot in piloted:
            accel_cmd[slot], steer_cmd[slot] = self._autopilot(int(slot))

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
        for slot in piloted:
            limit[slot] = min(
                float(limit[slot]),
                float(
                    stop_speed_limit(
                        np.float64(self._lead_gap(int(slot))),
                        abs(config.MAX_DECEL),
                        config.DT,
                        margin_m=config.VEHICLE_LENGTH + AUTOPILOT_HEADWAY_M,
                    )
                ),
                float(
                    stop_speed_limit(
                        np.float64(self._pedestrian_gap(int(slot))),
                        abs(config.MAX_DECEL),
                        config.DT,
                        margin_m=AUTOPILOT_PEDESTRIAN_MARGIN_M,
                    )
                ),
            )
        accel_cmd = constrain_accel(
            accel_cmd,
            self.world.fleet.speed,
            limit,
            config.DT,
            config.MAX_ACCEL,
            abs(config.MAX_DECEL),
        )

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

        dones = (reached | collided | offroad | timeout) & active_before
        held = int(self.commandeered_slot)
        if 0 <= held < n:
            # 徴用中の 1 台は乗降地点で止まるので、到達しても respawn させない
            # （させると乗客を置いて別の街区へ飛ぶ）。衝突・逸脱は配車側が拾う
            dones[held] = False
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

        self._obs = self._compute_observations()
        return StepResult(
            obs=self._obs.copy(),
            rewards=rewards,
            dones=dones,
            active=active_before,
            truncated=timeout & ~(reached | collided | offroad),
            episodes=episodes,
        )

    def apply_params(self, params: SimParams) -> None:
        """パラメータの実行時変更を反映する。学習は止めない。"""
        new_count = int(np.clip(int(params.vehicle_count), 0, config.MAX_VEHICLES))
        count_changed = new_count != int(self.world.active_count)
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
            self._reset_stats_for_changed(active_before)
            self.world.project_all()
            self._obs = self._compute_observations()

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
        try:
            if kind == "spawn_vehicle":
                slot = self.world.first_inactive_slot()
                if slot is None:
                    return f"空きスロットがありません（最大 {config.MAX_VEHICLES} 台）"
                at = (self._finite(payload, "x"), self._finite(payload, "y"))
                if not self.world.activate(slot, at=at):
                    return "指定地点から到達可能な経路が見つかりませんでした"
                self._reset_slot_stats(slot)
                self.params.vehicle_count = self.world.active_count

            elif kind == "despawn_vehicle":
                slot = int(self._finite(payload, "id"))
                if not (0 <= slot < config.MAX_VEHICLES):
                    return f"車両 ID が範囲外です: {slot}"
                if not self.world.fleet.active[slot]:
                    return f"車両 {slot} は既に非アクティブです"
                self.world.deactivate(slot)
                self._reset_slot_stats(slot)
                self.params.vehicle_count = self.world.active_count

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

        self._obs = self._compute_observations()
        return None

    def snapshot(self, tick: int, sim_time: float) -> FrameSnapshot:
        """描画用スナップショット。経路は変化があったスロットのみ載る。"""
        frame = self.world.snapshot(tick, sim_time, include_routes=False)
        frame.detections = self._detections_wire()
        frame.weather = self.weather.to_wire(float(self._camera_spec.far))
        return frame

    def full_snapshot(self, tick: int, sim_time: float) -> FrameSnapshot:
        """新規接続クライアント向けに全スロットの経路を含めたスナップショット。"""
        frame = self.world.snapshot(tick, sim_time, include_routes=True)
        frame.detections = self._detections_wire()
        frame.weather = self.weather.to_wire(float(self._camera_spec.far))
        return frame

    def _detections_wire(self) -> dict[int, list[dict[str, Any]]]:
        """直近の認識結果をワイヤ形式にする。"""
        return {slot: result.to_wire() for slot, result in self.latest_perception.items()}

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
            detect_ground_truth_batch,
            freespace_ground_truth,
        )

        self._ground_truth = detect_ground_truth_batch
        self._freespace_gt = freespace_ground_truth

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
        """擬似カメラで描き、CNN で検出し、観測ベクトルへ落とす。"""
        if not self._observations_enabled:
            self.latest_perception = {}
            return np.zeros((config.MAX_VEHICLES, config.OBS_DIM), dtype=np.float32)

        active = self.world.fleet.active
        idx = np.flatnonzero(active)
        if idx.size == 0:
            self.latest_perception = {}
            return np.zeros((config.MAX_VEHICLES, config.OBS_DIM), dtype=np.float32)

        self._ensure_percep()
        spec = self._camera_spec
        weather = self.weather
        freespace: dict[int, np.ndarray] = {}
        results: list[PerceptionResult] | None = None

        if self._detector is not None and self._camera is not None:
            try:
                images = self._camera.render(
                    self.world, idx, weather, int(self.world.sim_time * config.SIM_HZ)
                )
                results, free_arr = self._detector.detect_with_freespace(images, idx)
                for i, slot in enumerate(idx):
                    freespace[int(slot)] = free_arr[i]
            except Exception:
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

        if results is None and config.PERCEP_FALLBACK_GROUND_TRUTH:
            if self._ground_truth is not None and self._freespace_gt is not None:
                # 認識器を使う経路では CNN の出力をそのまま使う（画から判断させる）。
                # 視程で頭打ちにするのは真値で代用するこちら側だけ。
                reach = min(
                    float(config.OBS_FREESPACE_MAX_DISTANCE),
                    weather.visibility_m(float(spec.far)),
                )
                try:
                    results = self._ground_truth(self.world, idx, spec, weather)
                    for slot in idx:
                        freespace[int(slot)] = self._freespace_gt(
                            self.world, int(slot), spec, reach
                        )
                except Exception:
                    if not self._ground_truth_failed:
                        self._ground_truth_failed = True
                        logger.exception(
                            "真値からの検出生成に失敗しました。観測のカメラ欄は空になります"
                        )
                    results = None
                    freespace.clear()

        perceptions: dict[int, PerceptionResult] = {}
        if results is not None:
            for slot, result in zip(idx, results):
                perceptions[int(slot)] = result

        self.latest_perception = perceptions
        return encode_observations(
            self.world, self.params, perceptions, freespace=freespace, spec=spec
        )
