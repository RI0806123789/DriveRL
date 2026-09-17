"""MARL 環境本体（擬似固定エージェント数・parameter sharing 前提）。"""

from __future__ import annotations

import logging
import math
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
from app.sim.signals import constrain_accel
from app.sim.world import World

logger = logging.getLogger("autoware_sim")

__all__ = ["SimulationEnv"]


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

    def _reset_slot_stats(self, slot: int) -> None:
        """1 スロット分のエピソード統計を 0 に戻す。"""
        self._episode_reward[slot] = np.float32(0.0)
        self._episode_lateral[slot] = 0.0

    def _reset_stats_for_changed(self, active_before: np.ndarray) -> None:
        """アクティブ状態が変わったスロットの統計を落とす。"""
        changed = np.flatnonzero(active_before != self.world.fleet.active)
        for slot in changed:
            self._reset_slot_stats(int(slot))

    def reset_all(self) -> np.ndarray:
        """全アクティブスロットを再スポーンし、観測を返す。"""
        for slot in range(config.MAX_VEHICLES):
            if self.world.fleet.active[slot]:
                self.world.respawn(slot)
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

    def step(self, actions: np.ndarray) -> StepResult:
        """1 ステップ進める。終了したスロットは同じ step の中で即座に respawn する。"""
        n = config.MAX_VEHICLES
        act = np.asarray(actions, dtype=np.float32).reshape(n, config.ACTION_DIM)
        act = np.clip(np.nan_to_num(act, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)

        active_before = self.world.fleet.active.copy()
        accel_cmd = np.where(active_before, act[:, 0], 0.0).astype(np.float32)
        steer_cmd = np.where(active_before, act[:, 1], 0.0).astype(np.float32)

        params = self.params
        max_speed = max(float(params.max_speed), 1e-3)

        self.world.advance_time(config.DT)

        self.world.update_yellow_commitment()

        limit = self.world.curve_speed_limits()
        if params.obey_signals:
            limit = np.minimum(limit, self.world.signal_speed_limits())
        if params.obey_speed_signs:
            limit = np.minimum(limit, self.world.posted_speed_limits())
        accel_cmd = constrain_accel(
            accel_cmd,
            self.world.fleet.speed,
            limit,
            config.DT,
            config.MAX_ACCEL,
            abs(config.MAX_DECEL),
        )

        self.world.fleet.step(accel_cmd, steer_cmd, config.DT, max_speed)

        delta = self.world.project_all()
        step_limit = np.float32(max_speed * config.DT * 2.0)
        delta = np.clip(delta, -step_limit, step_limit)

        collided = self.world.check_collisions() & active_before
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
                )
            )
            self._reset_slot_stats(slot)
            if not self.world.try_respawn(slot):
                self.world.deactivate(slot)
                self.params.vehicle_count = self.world.active_count
                logger.warning(
                    "スロット %d の再スポーンに失敗しました（経路を作れず）。"
                    "このスロットを非アクティブにします",
                    slot,
                )

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
        self.params = replace(params)
        self.params.vehicle_count = new_count
        if count_changed:
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
        reach = min(
            float(config.OBS_FREESPACE_MAX_DISTANCE), weather.visibility_m(float(spec.far))
        )
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
                if not self._detector_failed:
                    self._detector_failed = True
                    logger.exception("認識器の推論に失敗しました。真値で代用します")
                results = None
                freespace.clear()

        if results is None and config.PERCEP_FALLBACK_GROUND_TRUTH:
            if self._ground_truth is not None and self._freespace_gt is not None:
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
