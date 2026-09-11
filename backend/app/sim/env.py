"""MARL 環境本体（擬似固定エージェント数・parameter sharing 前提）。

配列の先頭次元は常に config.MAX_VEHICLES で、未使用スロットは active=False で
マスクされる（memo 5章）。

★ **観測は画像認識ベース。** 1 ステップごとに擬似カメラで運転席視点を描き、
CNN 認識器にかけ、その検出結果を `percep.encoder` が観測ベクトルへ変換する。
**認識の誤りも見落としもそのまま学習に入る**のが狙いなので、カメラで見える
はずのものへ world の真値を混ぜてはいけない。

観測の並びは config.OBS_* 定数に厳密に従う（合計 57 次元）:

    [0:2]    自車       速度比 / 舵角比                      （速度計・舵角センサー）
    [2:5]    目的地     自車座標系相対位置 (dx, dy) と距離   （ナビ）
    [5:15]   経路案内   6m 間隔 5 点の自車座標系相対位置     （ナビ）
    [15:19]  車線       横方向偏差 / ずれ sin, cos / 信頼度  （カメラ）
    [19:24]  信号       停止線までの距離 / 青, 黄, 赤 / 信頼度（カメラ）
    [24:27]  速度標識   規制速度比 / 超過量 / 信頼度         （カメラ）
    [27:39]  前方車両   3 台の (dx, dy, 距離, 信頼度)        （カメラ）
    [39:48]  障害物     3 個の (dx, dy, 信頼度)              （カメラ）
    [48:57]  走行可能   前方 ±90 度 9 分割の進める距離       （カメラ）

自車速度・舵角・目的地・経路案内だけが真値なのは実車と同じ切り分けで、
**目的地までカメラに探させると「見えないものは検出できない」ため学習が
原理的に成立しない**から。

自車座標系は「前方 +x / 左 +y」。ENU からの変換は
    fx =  dx*cos(h) + dy*sin(h)
    fy = -dx*sin(h) + dy*cos(h)
"""

from __future__ import annotations

import logging
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
from app.sim.signals import constrain_accel
from app.sim.world import World

logger = logging.getLogger("autoware_sim")

__all__ = ["SimulationEnv"]


class SimulationEnv:
    """複数車両の物理更新・観測生成・報酬計算をまとめた環境。

    runtime 側はこのクラスのシグネチャに依存するので、公開メソッドを変えないこと。
    """

    def __init__(self, map_index: MapIndex, params: SimParams, seed: int = 0) -> None:
        self.map_index = map_index
        self.params = replace(params)  # 呼び出し側の dataclass を共有しない
        self.rng = np.random.default_rng(seed)
        self.world = World(map_index, self.rng)

        n = config.MAX_VEHICLES
        self._obs = np.zeros((n, config.OBS_DIM), dtype=np.float32)
        # エピソード中の |横方向偏差| の累積。ステップ数で割ると平均逸脱量になる
        self._episode_lateral = np.zeros(n, dtype=np.float64)
        self._episode_reward = np.zeros(n, dtype=np.float32)

        # --- 画像認識パイプライン ---
        # ★ 直近の認識結果。**PPO へ渡した観測の元になったもの**をそのまま持つ。
        #   可視化（frame.detections）はここから作るので、別経路で作り直さないこと。
        self.latest_perception: dict[int, PerceptionResult] = {}
        self._camera_spec = DEFAULT_CAMERA
        self._camera: Any | None = None            # PseudoCamera（遅延生成）
        self._detector: Any | None = None          # Detector（遅延読込）
        self._ground_truth: Callable[..., list[PerceptionResult]] | None = None
        # 走行可能距離の真値。**建物と路端はこの経路にしか無い**
        # （検出クラスに建物が無いので、検出結果からは車両と障害物しか拾えない）
        self._freespace_gt: Callable[..., np.ndarray] | None = None
        self._percep_ready = False
        # 失敗を 1 度だけログに出すためのフラグ（B-15）。
        # ★ 認識が失敗し続けると観測のカメラ欄が「常に何も見えない」で固定される。
        #   画面上は車が走っているので、黙らせると外から絶対に気づけない。
        self._detector_failed = False
        self._ground_truth_failed = False

        self.world.set_active_count(int(self.params.vehicle_count))
        self.world.project_all()
        self._obs = self._compute_observations()

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    @property
    def active_mask(self) -> np.ndarray:
        """shape (MAX_VEHICLES,) bool。"""
        return self.world.fleet.active.copy()

    @property
    def observations(self) -> np.ndarray:
        """shape (MAX_VEHICLES, OBS_DIM) float32。"""
        return self._obs.copy()

    def _reset_slot_stats(self, slot: int) -> None:
        """1 スロット分のエピソード統計を 0 に戻す。

        ★ スロットが走り始める／走り終わる経路は**必ずここを通すこと**
        （code_review B-10）。以前は `step()` の done 分岐と `reset_all()` でしか
        クリアしておらず、`set_active_count()` -> `activate()` -> `_install_route()`
        の経路を通らなかった。その結果、台数を下げてから上げると、再起動した
        スロットの**次の**エピソードの `lane_deviation` と `total_reward` に
        前のエピソードの累積がそのまま上乗せされていた
        （`metrics.laneDeviation` は直近 50 エピソードの平均なので、
        しばらく嘘の値が出続ける）。
        """
        self._episode_reward[slot] = np.float32(0.0)
        self._episode_lateral[slot] = 0.0

    def _reset_stats_for_changed(self, active_before: np.ndarray) -> None:
        """アクティブ状態が変わったスロットの統計を落とす。

        起きたスロット（新しいエピソードが始まる）も、寝たスロット
        （次に起きたとき前の残骸を持ち込ませない）も対象にする。
        """
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

        # --- 1. 時刻と信号の更新、物理更新 ---
        # 信号は観測にも報酬にも使うので、行動を反映する前に現示を進める
        self.world.advance_time(config.DT)

        # 信号に従わせる（道路交通法施行令 2 条）。
        # エージェントの指令を書き換えるのではなく「赤信号に近づくとアクセルが
        # 効かなくなる環境」として扱う。PPO から見れば環境の性質なので学習は成立する。
        # カーブ手前の減速。舵角は速度に応じて制限されるので、
        # 速いままカーブへ入ると曲がりきれず道路外へ出てしまう。
        limit = self.world.curve_speed_limits()
        if params.obey_signals:
            # 信号とカーブ、厳しいほうを採る
            limit = np.minimum(limit, self.world.signal_speed_limits())
        if params.obey_speed_signs:
            # 最高速度標識（道交法 22 条）。信号と同じく環境側の制約として扱う。
            # ★ ここは fleet.step() の**前**＝まだ動いていない位置での規制速度。
            #   後段の speed_violations() は射影後の位置で見るので値が違う。
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

        # --- 2. 経路への射影（進捗・横方向偏差） ---
        delta = self.world.project_all()
        step_limit = np.float32(max_speed * config.DT * 2.0)  # 暴走した射影値のクリップ
        delta = np.clip(delta, -step_limit, step_limit)

        # --- 3. 終了条件の評価 ---
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

        # 赤信号のまま停止線を越えたか（道路交通法施行令 2 条）
        ran_red = self.world.signal_violations() & active_before
        # 規制速度を超え始めたか（道交法 22 条）。超えている間ずっとではなく
        # 「超え始めた瞬間」だけ True になる
        over_speed = self.world.speed_violations() & active_before
        # 車線を外れた回数（外れ始めた瞬間を 1 回）
        self.world.update_lane_departures()

        # --- 4. 報酬 ---
        rewards = np.float32(params.reward_progress) * delta
        rewards += np.float32(params.reward_time)
        rewards += np.where(reached, np.float32(params.reward_goal), np.float32(0.0))
        rewards += np.where(collided, np.float32(params.reward_collision), np.float32(0.0))
        rewards += np.where(offroad, np.float32(params.reward_offroad), np.float32(0.0))
        rewards += np.where(ran_red, np.float32(params.reward_signal), np.float32(0.0))
        rewards += np.where(over_speed, np.float32(params.reward_overspeed), np.float32(0.0))
        # 車線中心（＝経路）からのずれを溜める。指標「車線逸脱」に使う
        self._episode_lateral += np.abs(self.world.lateral) * active_before
        rewards = (rewards * active_before).astype(np.float32)
        self._episode_reward += rewards

        # --- 5. 終了記録と即時 respawn ---
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
            self.world.respawn(slot)

        # respawn でクリアされる描画用フラグを、このステップの事実で上書きする
        self.world.set_event_flags(collided, reached)

        # --- 6. respawn 後の観測を返す（学習側は dones でブートストラップを切る） ---
        self._obs = self._compute_observations()
        return StepResult(
            obs=self._obs.copy(),
            rewards=rewards,
            dones=dones,
            active=active_before,
            episodes=episodes,
        )

    def apply_params(self, params: SimParams) -> None:
        """パラメータの実行時変更を反映する。学習は止めない。"""
        new_count = int(np.clip(int(params.vehicle_count), 0, config.MAX_VEHICLES))
        # ★ 比べる相手は `self.params.vehicle_count` ではなく**実際に走っている台数**
        #   （code_review B-03）。3D 画面のクリックで手動スポーンすると
        #   `world.active_count` だけが増える。そこで前者と比べていたため、
        #   次に来た `set_params` を「利用者が台数を減らした」と誤認して
        #   スポーンした車両を消していた。
        count_changed = new_count != int(self.world.active_count)
        self.params = replace(params)
        self.params.vehicle_count = new_count
        if count_changed:
            active_before = self.world.fleet.active.copy()
            self.world.set_active_count(new_count)
            # 起きた／寝たスロットの統計を落とす（B-10）
            self._reset_stats_for_changed(active_before)
            self.world.project_all()
            self._obs = self._compute_observations()

    def apply_event(self, event: InterventionEvent) -> str | None:
        """ユーザー介入を適用する。失敗理由の文字列、成功なら None を返す。

        memo 5章「介入も現実の交通現象の一部」。適用しても観測 shape は変わらず、
        学習も止めない。
        """
        kind = str(event.kind)
        payload = event.payload or {}
        try:
            if kind == "spawn_vehicle":
                slot = self.world.first_inactive_slot()
                if slot is None:
                    return f"空きスロットがありません（最大 {config.MAX_VEHICLES} 台）"
                at = (float(payload["x"]), float(payload["y"]))
                if not self.world.activate(slot, at=at):
                    return "指定地点から到達可能な経路が見つかりませんでした"
                self._reset_slot_stats(slot)
                self.params.vehicle_count = self.world.active_count

            elif kind == "despawn_vehicle":
                slot = int(payload["id"])
                if not (0 <= slot < config.MAX_VEHICLES):
                    return f"車両 ID が範囲外です: {slot}"
                if not self.world.fleet.active[slot]:
                    return f"車両 {slot} は既に非アクティブです"
                self.world.deactivate(slot)
                self._reset_slot_stats(slot)
                self.params.vehicle_count = self.world.active_count

            elif kind == "add_obstacle":
                radius = float(payload.get("radius", config.OBSTACLE_RADIUS))
                obstacle_id = self.world.add_obstacle(
                    float(payload["x"]), float(payload["y"]), radius
                )
                if obstacle_id is None:
                    return f"障害物の数が上限に達しています（最大 {config.MAX_OBSTACLES} 個）"

            elif kind == "remove_obstacle":
                if not self.world.remove_obstacle(int(payload["id"])):
                    return f"障害物 {payload.get('id')} は存在しません"

            elif kind == "clear_obstacles":
                self.world.clear_obstacles()

            elif kind == "reset_episode":
                self.reset_all()
                return None

            else:
                return f"未知の介入イベントです: {kind}"

        except (KeyError, TypeError, ValueError) as exc:
            return f"介入イベントのペイロードが不正です: {exc}"

        # 介入で世界が変わったので観測を作り直す（shape は不変）
        self._obs = self._compute_observations()
        return None

    def snapshot(self, tick: int, sim_time: float) -> FrameSnapshot:
        """描画用スナップショット。経路は変化があったスロットのみ載る。"""
        frame = self.world.snapshot(tick, sim_time, include_routes=False)
        frame.detections = self._detections_wire()
        return frame

    def full_snapshot(self, tick: int, sim_time: float) -> FrameSnapshot:
        """新規接続クライアント向けに全スロットの経路を含めたスナップショット。"""
        frame = self.world.snapshot(tick, sim_time, include_routes=True)
        frame.detections = self._detections_wire()
        return frame

    def _detections_wire(self) -> dict[int, list[dict[str, Any]]]:
        """直近の認識結果をワイヤ形式にする。

        ★ **学習が入力として受け取ったのと同じ検出結果**を送る。運転席カメラの
        バウンディングボックスはこれを描くので、ここで作り直したり間引いたりすると
        「画面では信号を認識できているのに学習は別の値を見ている」という、
        外から絶対に気づけない食い違いが生まれる。
        """
        return {slot: result.to_wire() for slot, result in self.latest_perception.items()}

    # ------------------------------------------------------------------
    # 観測の生成（擬似カメラ -> CNN 認識 -> 観測ベクトル）
    # ------------------------------------------------------------------

    def _ensure_percep(self) -> None:
        """擬似カメラと認識器を用意する（1 度だけ走る）。

        ★ **`import keras` は数秒かかる。** 学習済みの認識器がまだ無いうちは
          Keras に触れないようにして、起動直後から真値フォールバックで走り出せる
          ようにしてある。ここで無条件に読むと、マップを取り込むたびに
          「地図は出たのに車が数秒動かない」状態になる（`engine._ensure_trainer()`
          が torch の遅延初期化を先に済ませているのと同じ理由）。
        """
        if self._percep_ready:
            return
        self._percep_ready = True

        from app.percep.camera import PseudoCamera
        from app.percep.groundtruth import (
            detect_ground_truth_batch,
            freespace_ground_truth,
        )

        self._camera = PseudoCamera(self.map_index, self._camera_spec)
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
        """擬似カメラで描き、CNN で検出し、観測ベクトルへ落とす。

        ★ 重い処理は**アクティブなスロットだけ**に限る（B-14）。shape を
          MAX_VEHICLES に固定する必要があるのは学習側へ渡す obs だけで、
          描画と推論までそこへ合わせる理由は無い。
        """
        active = self.world.fleet.active
        idx = np.flatnonzero(active)
        if idx.size == 0:
            self.latest_perception = {}
            return np.zeros((config.MAX_VEHICLES, config.OBS_DIM), dtype=np.float32)

        self._ensure_percep()
        spec = self._camera_spec
        freespace: dict[int, np.ndarray] = {}
        results: list[PerceptionResult] | None = None

        # --- 1. 学習済みの認識器があればそれを使う ---
        if self._detector is not None and self._camera is not None:
            try:
                images = self._camera.render(self.world, idx)
                results, free_arr = self._detector.detect_with_freespace(images, idx)
                for i, slot in enumerate(idx):
                    freespace[int(slot)] = free_arr[i]
            except Exception:
                # ★ 認識が落ちても学習ループは止めない。ただし黙らせない（B-15）。
                #   握り潰すと観測のカメラ欄が「常に何も見えない」で固定されるが、
                #   画面では車が走り続けるので**外から絶対に気づけない**。
                if not self._detector_failed:
                    self._detector_failed = True
                    logger.exception("認識器の推論に失敗しました。真値で代用します")
                results = None
                freespace.clear()

        # --- 2. 認識器が無い／落ちたときは真値から「理想の検出結果」を作る ---
        # これは互換のための逃げ道ではなく、認識器の教師データを作る経路そのもの。
        if results is None and config.PERCEP_FALLBACK_GROUND_TRUTH:
            if self._ground_truth is not None and self._freespace_gt is not None:
                try:
                    results = self._ground_truth(self.world, idx, spec)
                    for slot in idx:
                        freespace[int(slot)] = self._freespace_gt(
                            self.world,
                            int(slot),
                            spec,
                            float(config.OBS_FREESPACE_MAX_DISTANCE),
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

        # 可視化（frame.detections）はここから作る。
        # **PPO へ渡すのと同じ検出結果**でなければ意味を成さない。
        self.latest_perception = perceptions
        return encode_observations(
            self.world, self.params, perceptions, freespace=freespace, spec=spec
        )
