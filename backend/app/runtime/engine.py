"""シミュレーション実行スレッド。

物理更新と PPO の重み更新を **専用の OS スレッド** で回し、asyncio 側（WebSocket）とは
次の 2 つだけでやり取りする：

- `queue.Queue` に積まれたコマンド／介入イベント（asyncio -> エンジン）
- ロックで守られた最新スナップショット（エンジン -> asyncio）

こうしておくと、memo 5章の「ユーザー介入があっても学習を止めない」「一時停止は描画だけ」
という要求が自然に満たせる。配信を止めてもこのスレッドは回り続けるだけだからである。
"""

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

if TYPE_CHECKING:  # 型チェック時のみ。実行時は下の遅延インポートを使う
    from pathlib import Path

    from app.rl.export import ExportResult
    from app.rl.importer import CheckpointInfo
    from app.rl.ppo import PPOTrainer
    from app.sim.env import SimulationEnv

logger = logging.getLogger(__name__)

# 学習指標の移動平均をとるエピソード数
_EPISODE_WINDOW = 50


@dataclass
class ExportTicket:
    """モデル書き出しの依頼票。

    書き出しはエンジンスレッドの**ステップ境界**で行う。asyncio 側から直接
    `state_dict()` を取ると、ちょうど `optimizer.step()` の最中の中途半端な重みを
    掴む可能性があるためである。依頼側は `done` を待って結果を受け取る。
    """

    kind: str
    done: threading.Event = field(default_factory=threading.Event)
    result: "ExportResult | None" = None
    error: str | None = None


@dataclass
class ImportTicket:
    """モデル読み込みの依頼票。

    読み込みは **いま学習中の重みを不可逆に置き換える** 操作なので、
    書き出しと同じくステップ境界で行い、直前の状態を自動でバックアップしてから
    差し替える（誤って古いモデルを読み込んでも取り戻せるようにするため）。
    """

    path: "Path"
    done: threading.Event = field(default_factory=threading.Event)
    info: "CheckpointInfo | None" = None
    backup: "ExportResult | None" = None
    error: str | None = None


class SimulationEngine:
    """物理 + オンライン学習のループを所有するオブジェクト。

    公開メソッドはすべて **asyncio スレッドから呼ばれる前提** でスレッドセーフに書く。
    """

    def __init__(self) -> None:
        # --- スレッド間の受け渡し ---
        self._lock = threading.Lock()
        self._inbox: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._notices: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # --- ロックで守る共有状態 ---
        self._state: str = "idle"          # "idle" | "loading_map" | "running" | "error"
        self._message: str = "マップが読み込まれていません"
        self._preset_id: str | None = None
        self._preset_name: str | None = None
        self._render_paused: bool = False
        self._latest_frame: FrameSnapshot | None = None
        self._frame_seq: int = 0           # 配信側が「新しいフレームか」を判定するための番号
        self._want_full_frame: bool = True  # 次のフレームに全スロットの経路を載せるか
        # フレームを最後に作った実時刻。物理の刻みとは切り離して FRAME_HZ で作る
        self._last_frame_at: float = 0.0
        self._map_pending: bool = False     # マップ差し替えの依頼を出してから取り込むまで
        self._metrics = MetricsSnapshot()
        # ネットワークの可視化用スナップショット（層ごとの重み・勾配・変化量）。
        # **必ずエンジンスレッドのステップ境界で作る。** asyncio 側から
        # trainer に触ると optimizer.step() の途中の重みを掴む可能性がある。
        self._network: dict[str, Any] = {}
        self._params = SimParams()
        # エンジン側の都合でパラメータが変わったか（手動スポーン等で台数が動いたとき）。
        # 立てたままにすると配信のたびに送ってしまうので take_params_update() で降ろす
        self._params_dirty: bool = False

        # --- エンジンスレッドだけが触る状態 ---
        self._map_index: MapIndex | None = None
        self._env: "SimulationEnv | None" = None
        self._trainer: "PPOTrainer | None" = None
        self._tick = 0
        self._sim_time = 0.0
        self._started_at = time.perf_counter()
        self._episode_log: deque[EpisodeResult] = deque(maxlen=_EPISODE_WINDOW)
        self._total_episodes = 0
        self._last_update_stats: dict[str, float] = {}
        self._step_marks: deque[float] = deque(maxlen=100)   # 各ステップの実時刻 [s]
        # 指標を最後に作った実時刻。配信は METRICS_HZ なので毎ステップは作らない
        self._last_metrics_at: float = 0.0
        # ネットワーク要約の失敗は初回だけログに出す（ホットループを黙らせない）
        self._network_snapshot_failed = False
        self._autosave_every = 20          # PPO 更新 20 回ごとに自動保存
        self._last_autosave_updates = 0

    # ------------------------------------------------------------------
    # ライフサイクル
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="sim-engine", daemon=True)
        self._thread.start()
        logger.info("シミュレーションスレッドを起動しました")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None
        logger.info("シミュレーションスレッドを停止しました")

    # ------------------------------------------------------------------
    # asyncio 側から呼ぶ API
    # ------------------------------------------------------------------

    def set_map(self, map_index: MapIndex, preset_id: str, preset_name: str) -> None:
        """読み込み済みのマップを差し込む。

        OSM の取得は数十秒かかるので、呼び出し側（main.py）が `asyncio.to_thread` で
        実行してからこのメソッドに渡す。エンジンスレッドはブロックしない。
        """
        # 依頼を出してから実際に取り込むまでの間、フレーム配信を止める。
        # ここで止めないと、クライアントが新しい map を受け取った直後に
        # **古いマップ上の座標のフレーム**が数フレーム届き、一瞬とんでもない
        # 位置に車が描かれる（tick も巻き戻る）。
        with self._lock:
            self._map_pending = True
        self._inbox.put(("set_map", (map_index, preset_id, preset_name)))

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
        """camelCase の部分更新を検証して適用し、(更新後のパラメータ, 検証結果) を返す。

        値域外は端に丸め、非有限値は反映しない（`contracts.SimParams.apply_wire`）。
        呼び出し側は検証結果を見て `INVALID_MESSAGE` を返すこと。
        """
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
        """次のフレームに全スロットの経路を載せるよう要求する。

        経路は通常「変化があったスロットだけ」送るため、途中から接続してきた
        クライアントには経路線が一本も届かない。新規接続のたびにこれを呼ぶ。
        """
        with self._lock:
            self._want_full_frame = True

    def command(self, name: str) -> None:
        """"save_checkpoint" / "load_checkpoint" / "reset_policy" を送る。"""
        self._inbox.put((name, None))

    def set_hidden_sizes(self, sizes: list[int]) -> None:
        """隠れ層の構成を変える。**重みは引き継げないので学習は 0 からになる。**

        エンジンスレッドのステップ境界で適用される（学習器を作り直すため、
        asyncio 側から触ると更新中の重みを壊す）。
        """
        self._inbox.put(("set_network", list(sizes)))

    def request_export(self, kind: str) -> ExportTicket:
        """モデルの書き出しを依頼する。呼び出し側は `ticket.done` を待つこと。

        書き出し自体はエンジンスレッドがステップ境界で行うので、学習は止まらない
        （1 ステップ分だけ余分に時間がかかる）。
        """
        ticket = ExportTicket(kind=kind)
        self._inbox.put(("export", ticket))
        return ticket

    def request_import(self, path: "Path") -> ImportTicket:
        """書き出したモデルを読み込んで学習を再開する。呼び出し側は `ticket.done` を待つこと。"""
        ticket = ImportTicket(path=path)
        self._inbox.put(("import", ticket))
        return ticket

    # ------------------------------------------------------------------
    # asyncio 側から読む API
    # ------------------------------------------------------------------

    def snapshot_params(self) -> SimParams:
        with self._lock:
            return SimParams(**vars(self._params))

    def take_params_update(self) -> SimParams | None:
        """エンジン側でパラメータが変わっていれば 1 度だけ返す。無ければ None。

        手動スポーン／デスポーンは `world.active_count` を動かすが、これを
        asyncio 側の `_params` に映さないと、次の `set_params` で
        「利用者が台数を戻した」と誤認されて手で足した車両が消える。
        `drain_notices()` と同じく配信ループが毎周期拾い、`params` メッセージにして送る。
        """
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
                "learning": self._state == "running",
                "message": self._message,
            }

    def take_frame(self, last_seq: int) -> tuple[int, FrameSnapshot | None]:
        """前回配信した番号より新しいフレームがあれば返す。

        `render_paused` のときは None を返す（＝配信しない）が、
        シミュレーション自体は裏で進み続けている。
        """
        with self._lock:
            if self._render_paused or self._map_pending or self._frame_seq == last_seq:
                # 未配信のまま番号だけ進めない。進めると、その間に変わった経路が
                # 「配信済み」と誤認されてフロントへ届かなくなる
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

    # ------------------------------------------------------------------
    # エンジンスレッド本体
    # ------------------------------------------------------------------

    def _run(self) -> None:
        # torch のインポートは数秒かかる。asyncio の起動を待たせないよう、
        # このスレッドの中で初めて読み込む。
        import torch

        torch.set_num_threads(config.TORCH_NUM_THREADS)
        logger.info("torch スレッド数を %d に設定しました", config.TORCH_NUM_THREADS)

        # 学習器はマップに依存しないので、マップ読込を待たずにここで作っておく。
        # torch の遅延初期化で 2 秒以上かかることがあり、これを _install_map の中で
        # やると「マップは表示されたのに車が数秒間動かない」状態になってしまう。
        self._ensure_trainer()

        next_deadline = time.perf_counter()

        while not self._stop_event.is_set():
            self._drain_inbox()

            if self._env is None or self._trainer is None:
                # マップ未読込。コマンドだけ拾って待つ。
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
            # 実効ステップ数は「開始時刻の並び」から出す。1 ステップの計算時間を
            # 別に貯めても誰も読まないので持たない（stepsPerSec は _step_marks から）
            self._step_marks.append(step_started)

            # --- 実時間ペーシング（累積deadline方式なのでドリフトしない） ---
            with self._lock:
                sim_speed = max(0.05, float(self._params.sim_speed))
            next_deadline += config.DT / sim_speed
            sleep_for = next_deadline - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            elif sleep_for < -1.0:
                # 1 秒以上遅れたら追いつくのを諦めて基準を引き直す
                next_deadline = time.perf_counter()

    # ------------------------------------------------------------------

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
                    # 取り込みに失敗したまま配信を止め続けると画面が固まる
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
                # 全車がテレポートするので、直前のステップとはつながっていない。
                # そのまま GAE を計算すると価値関数がワープをブートストラップする
                self._trainer.reset_rollout()
            # 成否によらず実台数を映す。spawn / despawn は env.params 側だけを
            # 更新するので、ここで拾わないと次の set_params で車両が消える
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

    # ------------------------------------------------------------------

    def _sync_vehicle_count(self) -> None:
        """world の実台数を asyncio 側のパラメータへ映す。

        3D 画面のクリックで車両を足すと `world.active_count` は増えるが、
        `engine._params.vehicle_count` は `set_params` でしか更新されない。
        放っておくと、次にスライダーを 1 つ動かしただけで
        `env.apply_params()` が「利用者が台数を戻した」と誤認し、
        **手で足した車両が黙って消える**（フロントの mockServer は
        spawn/despawn のたびに params を送り返しており、そちらが正しい契約）。
        """
        if self._env is None:
            return
        count = int(self._env.world.active_count)
        with self._lock:
            if int(self._params.vehicle_count) == count:
                return
            self._params.vehicle_count = count
            self._params_dirty = True

    # ------------------------------------------------------------------

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

            # 1) まず安全モードで中身を検証する（形が合わないまま load すると
            #    「失敗した」としか分からず、原因が伝わらない）
            info = inspect_checkpoint(
                ticket.path,
                expected_obs_dim=trainer.obs_dim,
                expected_action_dim=trainer.action_dim,
                expected_hidden_sizes=trainer.policy.hidden_sizes,
            )

            # 2) 上書き前に現状を退避する。読み込みは取り消せないので、
            #    誤って古いモデルを入れても戻せるようにしておく。
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
                )
            except ExportError:
                logger.exception("読み込み前のバックアップに失敗しました（読み込みは続行します）")

            # 3) 実際に載せ替える
            if not trainer.load(ticket.path):
                ticket.error = (
                    "重みの読み込みに失敗しました。ファイルが壊れている可能性があります"
                )
                return

            # 4) 再起動しても残るよう、標準のチェックポイントにも書いておく
            try:
                trainer.save(config.CHECKPOINT_PATH)
            except Exception:
                logger.exception("読み込んだモデルの保存に失敗しました（学習は継続します）")

            # 5) 統計は前のポリシーのものなので捨てる。混ぜると読み違える。
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
            ticket.done.set()

    def _ensure_trainer(self) -> None:
        """共有ポリシーを用意する。マップに依存しないので起動直後に呼べる。

        memo 5章「全車両が同一の共有ポリシーを使う（parameter sharing）」の通り、
        学習器はマップやエージェント数と独立に 1 つだけ持つ。
        """
        if self._trainer is not None:
            return

        started = time.perf_counter()
        from app.rl.ppo import PPOTrainer, peek_hidden_sizes

        params = self.snapshot_params()

        # 前回 `set_network` で隠れ層を変えて保存していれば、その構成を
        # PPOTrainer を作る**前**に読んでおく。既定（config.PPO_HIDDEN_SIZES）
        # のまま作ってから load() すると、保存されている hidden_sizes と
        # 食い違って読み込みが拒否され、「次回起動時も同じ層の数・幅で
        # 学習を再開する」が満たせない（構成も重みも黙って既定へ戻ってしまう）。
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

        # memo 5章「学習状態の永続化」：前回の重みがあれば復元する
        restored = trainer.load(config.CHECKPOINT_PATH)
        stale = not restored and config.CHECKPOINT_PATH.exists()

        # 最初の推論は torch の遅延初期化で数百 ms かかることがある。
        # ここで 1 回空打ちしておき、走り出しでフレームが飛ぶのを防ぐ。
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
            # 観測ベクトルの構成を変えると重みの形が合わなくなる。
            # 黙って初期状態から始めると「学習が進まない」と誤解されるので明示する。
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

        params = self.snapshot_params()
        self._map_index = map_index
        self._env = SimulationEnv(map_index, params, seed=0)
        self._env.reset_all()

        # 信号は観測にも報酬にも使うので環境が持つ。ここではログに出すだけ。
        logger.info("%s", self._env.world.signals.describe())

        # 起動時に作れていなければここで作る（保険）
        self._ensure_trainer()
        assert self._trainer is not None
        # マップを切り替えても共有ポリシーは引き継ぐ（parameter sharing の利点）。
        # ただし**収集中のロールアウトは捨てる**。切り替えの瞬間の 1 ステップは
        # done=False / active=True のまま座標だけ別マップへ飛ぶので、
        # 引き継ぐと 1 回ぶんの更新に新旧マップをまたぐ遷移が混ざる
        self._trainer.apply_params(params)
        self._trainer.reset_rollout()

        logger.info(
            "マップ %s を取り込みました（環境構築 %.0f ms）",
            preset_id,
            (time.perf_counter() - started) * 1000.0,
        )

        self._tick = 0
        self._sim_time = 0.0
        # 統計は前のマップのものなので全部捨てる。`_total_episodes` を残すと
        # マップを切り替えても metrics.episodes だけが前のマップから連続してしまう
        self._episode_log.clear()
        self._total_episodes = 0
        self._last_update_stats = {}
        self._step_marks.clear()
        self._last_metrics_at = 0.0   # 次のステップで作り直させる

        # 最初の 1 枚にも信号の現示を載せる（_step_once と同じ扱い）。
        # 入れないと、信号のあるマップでも読込直後の 1 フレームだけ signals が落ちる
        initial_frame = self._env.snapshot(self._tick, self._sim_time)
        initial_frame.signals = self._env.signal_phases

        with self._lock:
            self._state = "running"
            self._map_pending = False
            self._preset_id = preset_id
            self._preset_name = preset_name
            self._message = f"{preset_name} を読み込みました"
            self._latest_frame = initial_frame
            self._frame_seq += 1

        # 環境の構築には数秒かかることがある。完了した「時点」を配信側へ知らせないと
        # クライアントが loading_map のまま取り残されるので、必ず通知を積む。
        self._notify(f"{preset_name} を読み込みました。学習を開始します")

    # ------------------------------------------------------------------

    def _step_once(self) -> None:
        assert self._env is not None and self._trainer is not None
        env = self._env
        trainer = self._trainer

        obs = env.observations
        active = env.active_mask

        actions, log_probs, values = trainer.act(obs, active)
        result = env.step(actions)

        trainer.store(
            obs=obs,
            actions=actions,
            log_probs=log_probs,
            values=values,
            rewards=result.rewards,
            dones=result.dones,
            active=result.active,
        )

        stats = trainer.maybe_update(result.obs, result.active)
        if stats is not None:
            self._last_update_stats = stats
            # memo 5章「学習状態の永続化」：一定間隔で自動保存しておく
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
        # 時刻は環境が持つ（信号の現示がこの時刻だけで決まるため、二重管理しない）
        self._sim_time = env.sim_time

        # --- 学習指標 ---
        # 配信は METRICS_HZ（1Hz）なので、20Hz で作ると 19 回ぶんは捨てられる。
        # フレームの間引きと同じく、前回作ってからの経過で判断する。
        now = time.perf_counter()
        metrics_interval = 1.0 / max(0.1, float(config.METRICS_HZ))
        metrics = None
        network = None
        if (now - self._last_metrics_at) >= metrics_interval:
            self._last_metrics_at = now
            metrics = self._build_metrics(trainer.updates)
            # 重みの複製は約 190KB。1Hz なので無視できる。
            # ここ（ステップ境界）で作るのが重要で、asyncio 側から取ってはいけない。
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
            render_paused = self._render_paused

        # --- 描画用フレーム ---
        # 物理は 20Hz * simSpeed（8 倍なら 160Hz）で進むが、フレームは FRAME_HZ で作る。
        # 作ったフレームが配信されずに捨てられると、そこに載っていた経路が
        # フロントへ二度と届かない（route は変化時のみ送る仕様のため）。
        # 一時停止中も作らない。再開時に溜まった route_dirty がまとめて載る。
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

    # ------------------------------------------------------------------

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

        # 実際に何ステップ／秒 進んでいるかを実時刻から測る。
        # 「1 ステップの計算時間の逆数」ではないことに注意：後者は sim_speed による
        # ペーシング待ちを含まないため、20Hz で走っていても 400 などと出てしまい、
        # 画面に嘘の数字を出すことになる。
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
