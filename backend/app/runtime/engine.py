"""シミュレーション実行スレッド。"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np

from app import config
from app.contracts import (
    EpisodeResult,
    FrameSnapshot,
    InterventionEvent,
    MapIndex,
    MetricsSnapshot,
    ParamPatchResult,
    SimParams,
    validate_hidden_sizes,
)
from app.runtime.detector_job import DetectorTrainingJob
from app.runtime.taxi import TaxiService

if TYPE_CHECKING:
    from pathlib import Path

    from app.rl.export import ExportResult
    from app.rl.importer import CheckpointInfo
    from app.rl.ppo import PPOTrainer
    from app.sim.env import SimulationEnv


logger = logging.getLogger(__name__)

_EPISODE_WINDOW = 50


@dataclass
class ExportTicket:
    """モデル書き出しの依頼票。"""

    kind: str
    done: threading.Event = field(default_factory=threading.Event)
    result: "ExportResult | None" = None
    error: str | None = None


@dataclass
class ImportTicket:
    """モデル読み込みの依頼票。"""

    path: "Path"
    done: threading.Event = field(default_factory=threading.Event)
    info: "CheckpointInfo | None" = None
    backup: "ExportResult | None" = None
    error: str | None = None


class SimulationEngine:
    """物理 + オンライン学習のループを所有するオブジェクト。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inbox: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._notices: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._state: str = "idle"
        self._message: str = "マップが読み込まれていません"
        self._preset_id: str | None = None
        self._preset_name: str | None = None
        self._render_paused: bool = False
        self._sim_suspended: bool = False
        self._suspend_reason: str = ""
        self._detector_active: bool = False
        self._latest_frame: FrameSnapshot | None = None
        self._frame_seq: int = 0
        self._want_full_frame: bool = True
        self._last_frame_at: float = 0.0
        self._map_pending: bool = False
        self._shared_map_index: MapIndex | None = None
        self._metrics = MetricsSnapshot()
        self._network: dict[str, Any] = {}
        self._params = SimParams()
        self._params_dirty: bool = False

        # 実用モード（自動運転タクシー）。**この間は全車の学習を止めて推論だけで走る**（決定 5）
        self._practical_mode: bool = False
        self._taxi = TaxiService()
        self._taxi_wire: dict[str, Any] | None = None
        self._taxi_seq: int = 0
        self._taxi_sent_revision: int = -1
        self._taxi_last_phase: str = ""
        self._taxi_vehicle_id: int = -1

        self.detector_job = DetectorTrainingJob(self)

        self._map_index: MapIndex | None = None
        self._env: "SimulationEnv | None" = None
        self._trainer: "PPOTrainer | None" = None
        self._tick = 0
        self._sim_time = 0.0
        self._started_at = time.perf_counter()
        self._episode_log: deque[EpisodeResult] = deque(maxlen=_EPISODE_WINDOW)
        self._total_episodes = 0
        self._last_update_stats: dict[str, float] = {}
        self._step_marks: deque[float] = deque(maxlen=100)
        self._last_metrics_at: float = 0.0
        self._network_snapshot_failed = False
        self._autosave_every = 20
        self._last_autosave_updates = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="sim-engine", daemon=True)
        self._thread.start()
        logger.info("シミュレーションスレッドを起動しました")

    def stop(self, timeout: float = 5.0) -> None:
        self.detector_job.stop(timeout=timeout)
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None
        logger.info("シミュレーションスレッドを停止しました")

    def set_map(self, map_index: MapIndex, preset_id: str, preset_name: str) -> None:
        """読み込み済みのマップを差し込む。"""
        self.pause_frames()
        self._inbox.put(("set_map", (map_index, preset_id, preset_name)))

    def pause_frames(self) -> None:
        """フレーム配信を止める。マップを差し替える前に呼ぶこと（R-02）。"""
        with self._lock:
            self._map_pending = True

    def resume_frames(self) -> None:
        """`pause_frames()` を取り消す。マップ差し替えを中断したときに呼ぶ。"""
        with self._lock:
            self._map_pending = False

    def set_loading(self, preset_id: str, preset_name: str) -> None:
        with self._lock:
            self._state = "loading_map"
            self._preset_id = preset_id
            self._preset_name = preset_name
            self._message = f"{preset_name} の地図データを取得しています"

    def set_error(self, message: str) -> None:
        with self._lock:
            self._state = "error"
            self._message = message

    def submit_event(self, event: InterventionEvent) -> None:
        """ユーザー介入。学習は止めず、次のステップ境界で適用される。"""
        self._inbox.put(("event", event))

    def update_params(self, patch: dict[str, Any]) -> tuple[SimParams, ParamPatchResult]:
        """camelCase の部分更新を検証して適用し、(更新後のパラメータ, 検証結果) を返す。"""
        with self._lock:
            result = self._params.apply_wire(patch, max_vehicles=config.MAX_VEHICLES)
            snapshot = SimParams(**vars(self._params))
        self._inbox.put(("params", snapshot))
        return snapshot, result

    def set_render_paused(self, paused: bool) -> None:
        """描画（フレーム配信）だけを止める。学習は継続する（memo 5章）。"""
        with self._lock:
            self._render_paused = paused

    def request_full_frame(self) -> None:
        """次のフレームに全スロットの経路を載せるよう要求する。"""
        with self._lock:
            self._want_full_frame = True

    def set_practical_mode(self, enabled: bool) -> None:
        """実用モードに入る／出る（ステップ境界で適用する）。"""
        self._inbox.put(("app_mode", bool(enabled)))

    def taxi_command(self, action: str, payload: dict[str, Any] | None = None) -> None:
        """配車の操作を積む（request / board / alight / cancel / halt）。"""
        self._inbox.put(("taxi", (str(action), payload or {})))

    def take_taxi(self, last_seq: int) -> tuple[int, dict[str, Any] | None]:
        """前回配信した番号より新しい配車状態があれば返す（frame と同じ作法）。"""
        with self._lock:
            if self._taxi_wire is None or self._taxi_seq == last_seq:
                return last_seq, None
            return self._taxi_seq, dict(self._taxi_wire)

    def request_full_taxi(self) -> None:
        """経路を載せた配車状態を 1 通作らせる（新規接続向け）。"""
        self._inbox.put(("publish_taxi", None))

    def _publish_taxi(self) -> None:
        """配車の状態を asyncio 側へ渡す。**経路は版が変わったときだけ載せる。**"""
        status = self._taxi.status
        with self._lock:
            include_route = status.route_revision != self._taxi_sent_revision
            self._taxi_sent_revision = status.route_revision
            self._taxi_wire = status.to_wire(include_route=include_route)
            self._taxi_seq += 1
            self._taxi_vehicle_id = int(status.vehicle_id)
            self._taxi_last_phase = status.phase

    def command(self, name: str) -> None:
        """"save_checkpoint" / "load_checkpoint" / "reset_policy" を送る。"""
        self._inbox.put((name, None))

    def set_hidden_sizes(self, sizes: list[int]) -> None:
        """隠れ層の構成を変える。**重みは引き継げないので学習は 0 からになる。**"""
        self._inbox.put(("set_network", list(sizes)))

    def current_map(self) -> tuple[MapIndex | None, str | None, str | None]:
        """いま取り込んでいる (マップ, プリセット ID, 表示名)。"""
        with self._lock:
            return self._shared_map_index, self._preset_id, self._preset_name

    def suspend_sim(self, reason: str) -> None:
        """物理と PPO を止める（配信・コマンド処理・書き出しは動いたまま）。"""
        with self._lock:
            self._sim_suspended = True
            self._suspend_reason = reason
        logger.info("シミュレーションを一時停止します: %s", reason)

    def resume_sim(self) -> None:
        """止めていた物理と PPO を再開する。"""
        self._inbox.put(("resume_sim", None))

    def reload_detector(self) -> None:
        """学習し直した認識器を実行中の環境へ載せ替える（ステップ境界で行う）。"""
        self._inbox.put(("reload_detector", None))

    def notify(self, message: str) -> None:
        """`status.message` として 1 行流す（ジョブスレッドからも呼ばれる）。"""
        self._notify(message)

    def detector_in_use(self) -> bool:
        """いま観測が CNN 由来か（False なら真値フォールバック）。"""
        with self._lock:
            return self._detector_active

    def request_export(self, kind: str) -> ExportTicket:
        """モデルの書き出しを依頼する。呼び出し側は `ticket.done` を待つこと。"""
        ticket = ExportTicket(kind=kind)
        self._inbox.put(("export", ticket))
        return ticket

    def request_import(self, path: "Path") -> ImportTicket:
        """書き出したモデルを読み込んで学習を再開する。呼び出し側は `ticket.done` を待つこと。"""
        ticket = ImportTicket(path=path)
        self._inbox.put(("import", ticket))
        return ticket

    def snapshot_params(self) -> SimParams:
        with self._lock:
            return SimParams(**vars(self._params))

    def take_params_update(self) -> SimParams | None:
        """エンジン側でパラメータが変わっていれば 1 度だけ返す。無ければ None。"""
        with self._lock:
            if not self._params_dirty:
                return None
            self._params_dirty = False
            return SimParams(**vars(self._params))

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "mapLoaded": self._latest_frame is not None or self._state == "running",
                "presetId": self._preset_id,
                "renderPaused": self._render_paused,
                "learning": (
                    self._state == "running"
                    and not self._sim_suspended
                    and not self._practical_mode
                ),
                "simSuspended": self._sim_suspended,
                "suspendReason": self._suspend_reason,
                "practicalMode": self._practical_mode,
                "taxiVehicleId": self._taxi_vehicle_id,
                "message": self._message,
            }

    def take_frame(self, last_seq: int) -> tuple[int, FrameSnapshot | None]:
        """前回配信した番号より新しいフレームがあれば返す。"""
        with self._lock:
            if self._render_paused or self._map_pending or self._frame_seq == last_seq:
                return last_seq, None
            return self._frame_seq, self._latest_frame

    def metrics_payload(self) -> dict[str, Any]:
        with self._lock:
            return self._metrics.to_wire()

    def network_payload(self) -> dict[str, Any] | None:
        """層ごとの重み・勾配・変化量。まだ作られていなければ None。"""
        with self._lock:
            return dict(self._network) if self._network else None

    def drain_notices(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        while True:
            try:
                out.append(self._notices.get_nowait())
            except queue.Empty:
                return out

    def _notify(self, message: str) -> None:
        self._notices.put({"message": message})

    def _run(self) -> None:
        """エンジンスレッド本体。"""
        try:
            self._run_loop()
        except Exception:
            logger.exception("シミュレーションスレッドが異常終了しました")
            with self._lock:
                self._state = "error"
                self._message = (
                    "シミュレーションスレッドが異常終了しました。"
                    "サーバーを再起動してください"
                )
                self._map_pending = False
                self._sim_suspended = False
                self._suspend_reason = ""
            self._notify(
                "シミュレーションスレッドが異常終了しました。サーバーを再起動してください"
            )

    def _run_loop(self) -> None:
        import torch

        torch.set_num_threads(config.TORCH_NUM_THREADS)
        logger.info("torch スレッド数を %d に設定しました", config.TORCH_NUM_THREADS)

        self._ensure_trainer()

        next_deadline = time.perf_counter()

        while not self._stop_event.is_set():
            self._drain_inbox()

            if self._env is None or self._trainer is None:
                time.sleep(0.05)
                next_deadline = time.perf_counter()
                continue

            with self._lock:
                suspended = self._sim_suspended
            if suspended:
                time.sleep(0.05)
                next_deadline = time.perf_counter()
                continue

            step_started = time.perf_counter()
            try:
                self._step_once()
            except Exception:
                logger.exception("シミュレーションステップで例外が発生しました")
                with self._lock:
                    self._state = "error"
                    self._message = "シミュレーション中に内部エラーが発生しました"
                self._env = None
                continue
            self._step_marks.append(step_started)

            with self._lock:
                sim_speed = max(0.05, float(self._params.sim_speed))
            next_deadline += config.DT / sim_speed
            sleep_for = next_deadline - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            elif sleep_for < -1.0:
                next_deadline = time.perf_counter()

    def _drain_inbox(self) -> None:
        while True:
            try:
                kind, payload = self._inbox.get_nowait()
            except queue.Empty:
                return

            try:
                self._handle_inbox_item(kind, payload)
            except Exception:
                logger.exception("コマンド %s の処理で例外が発生しました", kind)
                self._notify(f"コマンド {kind} の処理に失敗しました")
                if kind == "set_map":
                    with self._lock:
                        self._map_pending = False

    def _handle_inbox_item(self, kind: str, payload: Any) -> None:
        if kind == "set_map":
            map_index, preset_id, preset_name = payload
            self._install_map(map_index, preset_id, preset_name)

        elif kind == "event":
            if self._env is None:
                self._notify("マップが読み込まれていないため介入を無視しました")
                return
            reason = self._env.apply_event(payload)
            if reason:
                self._notify(reason)
            elif getattr(payload, "kind", None) == "reset_episode" and self._trainer is not None:
                self._trainer.reset_rollout()
            self._sync_vehicle_count()

        elif kind == "params":
            params: SimParams = payload
            if self._env is not None:
                self._env.apply_params(params)
            if self._trainer is not None:
                self._trainer.apply_params(params)

        elif kind == "save_checkpoint":
            if self._trainer is None:
                self._notify("学習器が未初期化のため保存できません")
                return
            self._trainer.save(config.CHECKPOINT_PATH)
            self._notify("学習済みモデルを保存しました")

        elif kind == "load_checkpoint":
            if self._trainer is None:
                self._notify("学習器が未初期化のため読み込めません")
                return
            ok = self._trainer.load(config.CHECKPOINT_PATH)
            self._notify(
                "学習済みモデルを読み込みました" if ok else "読み込めるチェックポイントがありません"
            )

        elif kind == "resume_sim":
            with self._lock:
                if not self._sim_suspended:
                    return
                self._sim_suspended = False
                self._suspend_reason = ""
            logger.info("シミュレーションを再開します")

        elif kind == "reload_detector":
            if self._env is None:
                self._notify(
                    "マップが読み込まれていないため、認識器の載せ替えは次回の読込時に行われます"
                )
                return
            ok = self._env.reload_detector()
            with self._lock:
                self._detector_active = ok
            self._notify(
                "学習した認識器に切り替えました。観測が CNN の出力になります"
                if ok
                else "認識器を読み込めなかったため、真値の検出結果で走り続けます"
            )

        elif kind == "app_mode":
            self._apply_app_mode(bool(payload))

        elif kind == "taxi":
            action, args = payload
            self._handle_taxi(str(action), args or {})

        elif kind == "publish_taxi":
            with self._lock:
                self._taxi_sent_revision = -1
            self._publish_taxi()

        elif kind == "export":
            self._handle_export(payload)

        elif kind == "import":
            self._handle_import(payload)

        elif kind == "reset_policy":
            if self._trainer is None:
                return
            self._trainer.reset_policy()
            self._episode_log.clear()
            self._total_episodes = 0
            self._last_update_stats = {}
            self._notify("ポリシーを初期化しました。学習を最初からやり直します")

        elif kind == "set_network":
            if self._trainer is None:
                return
            sizes = list(payload or ())
            before = tuple(self._trainer.hidden_sizes)
            changed = self._trainer.set_hidden_sizes(sizes)
            if not changed:
                self._notify(
                    "隠れ層の構成は変わっていません"
                    f"（{' × '.join(str(h) for h in before)}）"
                )
                return
            self._episode_log.clear()
            self._total_episodes = 0
            self._last_update_stats = {}
            with self._lock:
                self._network = {}
            shape = " → ".join(str(h) for h in self._trainer.hidden_sizes)
            self._notify(
                f"隠れ層を {shape} に変えました。"
                "形が変わったので重みは引き継げず、学習は最初からやり直します"
            )

        else:
            logger.warning("未知のコマンド: %s", kind)

    def _apply_app_mode(self, practical: bool) -> None:
        """開発モードと実用モードを切り替える。**配車は必ずここで畳む。**"""
        with self._lock:
            if self._practical_mode == practical:
                return
            self._practical_mode = practical

        if self._env is not None:
            # 実用モードの間だけ街の車も経路追従にする（止まったままだと道が詰まる）
            self._env.autopilot_all = practical

        self._taxi.cancel(
            self._env,
            "モードを切り替えたため配車を終了しました",
            halt=practical,
        )
        if self._trainer is not None:
            # 走行の連続性が切れるので、溜めかけのロールアウトは捨てる
            self._trainer.reset_rollout()
        self._publish_taxi()
        self._notify(
            "実用モードに入りました。学習は止まり、いまの重みのまま走ります"
            if practical
            else "開発モードに戻りました。学習を再開します"
        )

    def _handle_taxi(self, action: str, args: dict[str, Any]) -> None:
        """配車の操作を適用する。断った理由は `status.message` で返す。"""
        env = self._env
        if env is None:
            self._notify("マップが読み込まれていないため配車できません")
            return
        with self._lock:
            practical = self._practical_mode
        if not practical:
            self._notify("実用モードでないため配車の操作を無視しました")
            return

        problem: str | None = None
        if action == "request":
            problem = self._taxi.request(
                env,
                (float(args["pickupX"]), float(args["pickupY"])),
                (float(args["dropoffX"]), float(args["dropoffY"])),
            )
        elif action == "board":
            problem = self._taxi.board(env)
        elif action == "alight":
            problem = self._taxi.alight(env)
        elif action == "cancel":
            self._taxi.cancel(env, "配車を取り消しました")
        elif action == "halt":
            self._taxi.cancel(env, "緊急停止しました。自動運転を終了します", halt=True)
        else:
            logger.warning("未知の配車操作: %s", action)
            return

        if problem:
            self._notify(problem)
        self._publish_taxi()

    def _sync_vehicle_count(self) -> None:
        """world の実台数を asyncio 側のパラメータへ映す。"""
        if self._env is None:
            return
        count = int(self._env.world.active_count)
        with self._lock:
            if int(self._params.vehicle_count) == count:
                return
            self._params.vehicle_count = count
            self._params_dirty = True

    def _handle_export(self, ticket: ExportTicket) -> None:
        """モデルを書き出す。何があっても必ず `done` を立てる（依頼側が待ち続けないように）。"""
        from app.rl.export import ExportError, export_model

        started = time.perf_counter()
        try:
            if self._trainer is None:
                ticket.error = "学習器がまだ初期化されていません。数秒おいて再試行してください"
                return

            with self._lock:
                preset_id = self._preset_id
                preset_name = self._preset_name
                metrics = self._metrics.to_wire()

            ticket.result = export_model(
                self._trainer,
                ticket.kind,
                preset_id=preset_id,
                preset_name=preset_name,
                metrics=metrics,
                params=self.snapshot_params(),
            )
            logger.info(
                "モデルを書き出しました: %s（%.0f KB, %.0f ms）",
                ticket.result.filename,
                ticket.result.size_bytes / 1024.0,
                (time.perf_counter() - started) * 1000.0,
            )
            self._notify(f"モデルを書き出しました: {ticket.result.filename}")

        except ExportError as exc:
            ticket.error = str(exc)
        except Exception as exc:  # noqa: BLE001 - 理由を画面に返したい
            logger.exception("モデルの書き出しで例外が発生しました")
            ticket.error = f"モデルの書き出しに失敗しました: {exc}"
        finally:
            ticket.done.set()

    def _handle_import(self, ticket: ImportTicket) -> None:
        """書き出したモデルを読み込んで学習を再開する。必ず `done` を立てる。"""
        from app.rl.export import ExportError, export_model
        from app.rl.importer import CheckpointImportError, inspect_checkpoint

        started = time.perf_counter()
        try:
            if self._trainer is None:
                ticket.error = "学習器がまだ初期化されていません。数秒おいて再試行してください"
                return

            trainer = self._trainer

            info = inspect_checkpoint(
                ticket.path,
                expected_obs_dim=trainer.obs_dim,
                expected_action_dim=trainer.action_dim,
                expected_hidden_sizes=trainer.policy.hidden_sizes,
            )

            with self._lock:
                preset_id = self._preset_id
                preset_name = self._preset_name
                metrics = self._metrics.to_wire()
            try:
                ticket.backup = export_model(
                    trainer,
                    "checkpoint",
                    preset_id=preset_id,
                    preset_name=preset_name,
                    metrics=metrics,
                    label="before-import",
                    params=self.snapshot_params(),
                )
            except ExportError:
                logger.exception("読み込み前のバックアップに失敗しました（読み込みは続行します）")

            if not trainer.load(ticket.path):
                ticket.error = (
                    "重みの読み込みに失敗しました。ファイルが壊れている可能性があります"
                )
                return

            try:
                trainer.save(config.CHECKPOINT_PATH)
            except Exception:
                logger.exception("読み込んだモデルの保存に失敗しました（学習は継続します）")

            self._episode_log.clear()
            self._total_episodes = 0
            self._last_update_stats = {}
            self._last_autosave_updates = trainer.updates

            ticket.info = info
            logger.info(
                "モデルを読み込みました（更新回数 %d から再開、%.0f ms）",
                trainer.updates,
                (time.perf_counter() - started) * 1000.0,
            )
            self._notify(
                f"モデルを読み込みました。学習回数 {trainer.updates} 回の状態から再開します"
            )

        except CheckpointImportError as exc:
            ticket.error = str(exc)
        except Exception as exc:  # noqa: BLE001 - 理由を画面に返したい
            logger.exception("モデルの読み込みで例外が発生しました")
            ticket.error = f"モデルの読み込みに失敗しました: {exc}"
        finally:
            # ★ アップロードを消すのは**読み終えたこちら側**（code_review E-05）。
            #   HTTP 側の finally で消すと、504 の後にエンジンが読みにいって失敗する
            try:
                ticket.path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "アップロードファイルを削除できませんでした: %s（%s）",
                    ticket.path.name,
                    exc,
                )
            ticket.done.set()

    def _ensure_trainer(self) -> None:
        """共有ポリシーを用意する。マップに依存しないので起動直後に呼べる。"""
        if self._trainer is not None:
            return

        started = time.perf_counter()
        from app.rl.ppo import PPOTrainer, peek_hidden_sizes

        params = self.snapshot_params()

        restored_hidden_sizes: tuple[int, ...] | None = None
        if config.CHECKPOINT_PATH.exists():
            peeked = peek_hidden_sizes(config.CHECKPOINT_PATH)
            if peeked is not None:
                validated, reason = validate_hidden_sizes(list(peeked))
                if validated is not None:
                    restored_hidden_sizes = tuple(validated)
                else:
                    logger.warning(
                        "チェックポイントの隠れ層構成が不正なため既定を使います: %s（%s）",
                        peeked,
                        reason,
                    )

        trainer = PPOTrainer(
            obs_dim=config.OBS_DIM,
            action_dim=config.ACTION_DIM,
            params=params,
            num_agents=config.MAX_VEHICLES,
            seed=0,
            hidden_sizes=restored_hidden_sizes,
        )

        restored = trainer.load(config.CHECKPOINT_PATH)
        stale = not restored and config.CHECKPOINT_PATH.exists()

        warm_obs = np.zeros((config.MAX_VEHICLES, config.OBS_DIM), dtype=np.float32)
        warm_active = np.ones(config.MAX_VEHICLES, dtype=bool)
        trainer.act(warm_obs, warm_active)

        self._trainer = trainer
        logger.info(
            "学習器を初期化しました（%.0f ms、チェックポイント復元: %s）",
            (time.perf_counter() - started) * 1000.0,
            "あり" if restored else "なし",
        )
        if restored:
            self._notify("前回の学習済みモデルを復元しました")
        elif stale:
            logger.warning(
                "既存のチェックポイントは現在のモデル定義と一致しないため読み込めませんでした"
                "（観測 %d 次元）。学習は最初からやり直しになります",
                config.OBS_DIM,
            )
            self._notify(
                "以前のモデルは観測の構成が変わったため読み込めませんでした。"
                "学習は最初からやり直しになります"
            )

    def _install_map(self, map_index: MapIndex, preset_id: str, preset_name: str) -> None:
        started = time.perf_counter()
        from app.sim.env import SimulationEnv

        # 配車は古い env の車両を指しているので、地図ごと入れ替える前に畳む
        self._taxi.cancel(self._env, "エリアを切り替えたため配車を終了しました")
        self._publish_taxi()

        params = self.snapshot_params()
        try:
            env = SimulationEnv(map_index, params, seed=0)
            env.reset_all()
        except Exception:
            logger.exception("環境の構築に失敗しました: %s", preset_id)
            self._env = None
            with self._lock:
                self._state = "error"
                self._message = f"{preset_name} の環境構築に失敗しました"
                self._latest_frame = None
            raise
        with self._lock:
            env.autopilot_all = self._practical_mode
        self._env = env
        self._map_index = map_index

        logger.info("%s", self._env.world.signals.describe())

        self._ensure_trainer()
        assert self._trainer is not None
        self._trainer.apply_params(params)
        self._trainer.reset_rollout()

        logger.info(
            "マップ %s を取り込みました（環境構築 %.0f ms）",
            preset_id,
            (time.perf_counter() - started) * 1000.0,
        )

        self._tick = 0
        self._sim_time = 0.0
        self._episode_log.clear()
        self._total_episodes = 0
        self._last_update_stats = {}
        self._step_marks.clear()
        self._last_metrics_at = 0.0

        initial_frame = self._env.snapshot(self._tick, self._sim_time)
        initial_frame.signals = self._env.signal_phases

        with self._lock:
            self._state = "running"
            self._map_pending = False
            self._preset_id = preset_id
            self._preset_name = preset_name
            self._shared_map_index = map_index
            self._message = f"{preset_name} を読み込みました"
            self._latest_frame = initial_frame
            self._frame_seq += 1
            self._detector_active = self._env.detector_active

        self._notify(f"{preset_name} を読み込みました。学習を開始します")

    def _step_once(self) -> None:
        assert self._env is not None and self._trainer is not None
        env = self._env
        trainer = self._trainer

        with self._lock:
            practical = self._practical_mode

        obs = env.observations
        active = env.active_mask

        actions, log_probs, values = trainer.act(obs, active)
        result = env.step(actions)

        # 実用モードでは推論だけ回す。重みは触らない（決定 5）
        if not practical:
            trainer.store(
                obs=obs,
                actions=actions,
                log_probs=log_probs,
                values=values,
                rewards=result.rewards,
                dones=result.dones,
                active=result.active,
                truncated=result.truncated,
            )

            stats = trainer.maybe_update(result.obs, result.active)
            if stats is not None:
                self._last_update_stats = stats
                if trainer.updates - self._last_autosave_updates >= self._autosave_every:
                    self._last_autosave_updates = trainer.updates
                    try:
                        trainer.save(config.CHECKPOINT_PATH)
                    except Exception:
                        logger.exception("チェックポイントの自動保存に失敗しました")

            for episode in result.episodes:
                self._episode_log.append(episode)
                self._total_episodes += 1

        self._tick += 1
        self._sim_time = env.sim_time

        now = time.perf_counter()
        metrics_interval = 1.0 / max(0.1, float(config.METRICS_HZ))

        if practical:
            # ETA は毎ステップ引き直し、配信は段階が変わったときと 1Hz（決定 14）
            self._taxi.update(env)
            if (
                self._taxi.status.phase != self._taxi_last_phase
                or (now - self._last_metrics_at) >= metrics_interval
            ):
                self._publish_taxi()

        metrics = None
        network = None
        detector_active = None
        if (now - self._last_metrics_at) >= metrics_interval:
            self._last_metrics_at = now
            metrics = self._build_metrics(trainer.updates)
            # 推論が落ちて真値へ落ちたことがあるので、載せ替えのときだけでなく
            # ここでも取り直す（`model.inUse` は「いま実際に使っているか」）
            detector_active = env.detector_active
            try:
                network = trainer.network_snapshot()
            except Exception:  # noqa: BLE001 - 可視化のために学習を止めない
                if not self._network_snapshot_failed:
                    self._network_snapshot_failed = True
                    logger.exception("ネットワークの要約に失敗しました（学習は継続します）")

        with self._lock:
            if metrics is not None:
                self._metrics = metrics
            if network is not None:
                self._network = network
            if detector_active is not None:
                self._detector_active = detector_active
            render_paused = self._render_paused

        interval = 1.0 / max(1.0, float(config.FRAME_HZ))
        if render_paused or (now - self._last_frame_at) < interval:
            return
        self._last_frame_at = now

        with self._lock:
            want_full = self._want_full_frame
            self._want_full_frame = False

        frame = (
            env.full_snapshot(self._tick, self._sim_time)
            if want_full
            else env.snapshot(self._tick, self._sim_time)
        )
        frame.signals = env.signal_phases

        with self._lock:
            self._latest_frame = frame
            self._frame_seq += 1

    def _build_metrics(self, updates: int) -> MetricsSnapshot:
        episodes = list(self._episode_log)
        if episodes:
            rewards = np.fromiter((e.total_reward for e in episodes), dtype=np.float64)
            lengths = np.fromiter((e.length for e in episodes), dtype=np.float64)
            mean_reward = float(rewards.mean())
            mean_length = float(lengths.mean())
            n = len(episodes)
            goal_rate = sum(1 for e in episodes if e.reason == "goal") / n
            collision_rate = sum(1 for e in episodes if e.reason == "collision") / n
            violations = sum(e.signal_violations for e in episodes) / n
            speeding = sum(e.speed_violations for e in episodes) / n
            lane_deviation = sum(e.lane_deviation for e in episodes) / n
        else:
            mean_reward = mean_length = goal_rate = collision_rate = 0.0
            violations = speeding = lane_deviation = 0.0

        if len(self._step_marks) >= 2:
            span = self._step_marks[-1] - self._step_marks[0]
            steps_per_sec = (len(self._step_marks) - 1) / span if span > 0 else 0.0
        else:
            steps_per_sec = 0.0

        stats = self._last_update_stats
        return MetricsSnapshot(
            tick=self._tick,
            wall_time=time.perf_counter() - self._started_at,
            updates=updates,
            episodes=self._total_episodes,
            mean_episode_reward=mean_reward,
            mean_episode_length=mean_length,
            policy_loss=float(stats.get("policy_loss", 0.0)),
            value_loss=float(stats.get("value_loss", 0.0)),
            entropy=float(stats.get("entropy", 0.0)),
            approx_kl=float(stats.get("approx_kl", 0.0)),
            collision_rate=collision_rate,
            goal_rate=goal_rate,
            steps_per_sec=steps_per_sec,
            signal_violations=violations,
            speed_violations=speeding,
            lane_deviation=lane_deviation,
        )
