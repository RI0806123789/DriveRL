# -*- coding: utf-8 -*-
"""認識器（CNN）の学習を Web アプリから回すためのジョブ。"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING

import numpy as np

from app import config
from app.percep.trainer import (
    DATASET_FILE,
    DatasetSummary,
    TrainingCancelled,
    collect_dataset,
    fit_detector,
    load_dataset,
    summarize_dataset,
)

if TYPE_CHECKING:
    from app.contracts import MapIndex


logger = logging.getLogger(__name__)

MODES = ("full", "collect", "train")

SAMPLES_MIN, SAMPLES_MAX = 200, 4800
EPOCHS_MIN, EPOCHS_MAX = 1, 60
BATCH_MIN, BATCH_MAX = 8, 128
WIDTH_MIN, WIDTH_MAX = 0.25, 2.0
SEED_MIN, SEED_MAX = 0, 999_999


@dataclass(frozen=True)
class DetectorTrainRequest:
    """学習の依頼。値域は `parse()` が検証済み。"""

    mode: str = "full"
    preset_id: str | None = None
    samples: int = 2400
    epochs: int = 12
    batch_size: int = 32
    width: float = 1.0
    seed: int = 0

    @property
    def collects(self) -> bool:
        return self.mode in ("full", "collect")

    @property
    def trains(self) -> bool:
        return self.mode in ("full", "train")

    def to_wire(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "presetId": self.preset_id,
            "samples": int(self.samples),
            "epochs": int(self.epochs),
            "batchSize": int(self.batch_size),
            "width": float(self.width),
            "seed": int(self.seed),
        }


def _bounded_int(value: Any, lo: int, hi: int, name: str) -> tuple[int, str]:
    """整数として読み、値域外なら理由を返す。**丸めずに弾く。**"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0, f"{name} は数値で指定してください"
    number = int(value)
    if not (lo <= number <= hi):
        return 0, f"{name} は {lo}〜{hi} の範囲で指定してください（受け取った値: {number}）"
    return number, ""


def parse_request(wire: Any) -> tuple[DetectorTrainRequest | None, str]:
    """`start_detector_training` のペイロードを検証する。"""
    if not isinstance(wire, dict):
        return None, "オブジェクトを送ってください"

    mode = wire.get("mode", "full")
    if mode not in MODES:
        return None, f"mode は {' / '.join(MODES)} のいずれかです（受け取った値: {mode!r}）"

    samples, why = _bounded_int(
        wire.get("samples", 2400), SAMPLES_MIN, SAMPLES_MAX, "samples"
    )
    if why:
        return None, why
    epochs, why = _bounded_int(wire.get("epochs", 12), EPOCHS_MIN, EPOCHS_MAX, "epochs")
    if why:
        return None, why
    batch_size, why = _bounded_int(
        wire.get("batchSize", 32), BATCH_MIN, BATCH_MAX, "batchSize"
    )
    if why:
        return None, why

    raw_width = wire.get("width", 1.0)
    if isinstance(raw_width, bool) or not isinstance(raw_width, (int, float)):
        return None, "width は数値で指定してください"
    width = float(raw_width)
    if not (WIDTH_MIN <= width <= WIDTH_MAX):
        return None, f"width は {WIDTH_MIN}〜{WIDTH_MAX} の範囲で指定してください"

    seed, why = _bounded_int(wire.get("seed", 0), SEED_MIN, SEED_MAX, "seed")
    if why:
        return None, why

    preset_id = wire.get("presetId")
    if preset_id is not None and not isinstance(preset_id, str):
        return None, "presetId は文字列で指定してください"

    return (
        DetectorTrainRequest(
            mode=str(mode),
            preset_id=preset_id,
            samples=samples,
            epochs=epochs,
            batch_size=batch_size,
            width=width,
            seed=seed,
        ),
        "",
    )


class EngineHooks(Protocol):
    """ジョブがエンジンへ頼む操作。"""

    def current_map(self) -> tuple["MapIndex | None", str | None, str | None]:
        """(いま読み込んでいるマップ, プリセット ID, 表示名)。"""

    def suspend_sim(self, reason: str) -> None:
        """物理と PPO を止める（配信とコマンド処理は動いたまま）。"""

    def resume_sim(self) -> None: ...

    def reload_detector(self) -> None:
        """学習し直した `detector.keras` を実行中の環境へ載せ替える。"""

    def notify(self, message: str) -> None:
        """`status.message` として画面へ 1 行流す。"""

    def detector_in_use(self) -> bool:
        """いま観測が CNN 由来か（False なら真値フォールバック）。"""


@dataclass
class _Progress:
    """ロックで守る進捗。`snapshot()` がそのままワイヤ形式にする。"""

    state: str = "idle"
    message: str = "認識器の学習はまだ実行していません"
    request: DetectorTrainRequest | None = None
    preset_name: str | None = None
    progress: float = 0.0
    collected: int = 0
    epoch: int = 0
    batch: int = 0
    batches: int = 0
    history: list[dict[str, float]] = field(default_factory=list)
    dataset: DatasetSummary | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    warning: str = ""
    param_count: int = 0


class DetectorTrainingJob:
    """認識器の学習を 1 本だけ走らせるジョブ。"""

    def __init__(self, hooks: EngineHooks) -> None:
        self._hooks = hooks
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._progress = _Progress()
        self._own_index = False

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self, request: DetectorTrainRequest) -> str:
        """学習を始める。**始められなければ理由を返す**（空文字なら成功）。"""
        if self.running:
            return "すでに学習を実行中です。完了を待つか中止してください"

        if request.trains and not request.collects:
            if not self._dataset_path().exists():
                return (
                    "保存された教師データがありません。"
                    "先に「収集して学習」か「収集だけ」を実行してください"
                )

        self._cancel.clear()
        with self._lock:
            self._progress = _Progress(
                state="preparing",
                message="学習の準備をしています",
                request=request,
                started_at=time.time(),
            )

        thread = threading.Thread(
            target=self._run, args=(request,), name="detector-train", daemon=True
        )
        self._thread = thread
        thread.start()
        return ""

    def cancel(self) -> str:
        """中断を要求する。**すぐには止まらない**（バッチ／ステップの境界で抜ける）。"""
        if not self.running:
            return "実行中の学習はありません"
        self._cancel.set()
        self._update(message="中断しています…")
        return ""

    def stop(self, timeout: float = 10.0) -> None:
        """サーバー停止時に呼ぶ。中断を頼んで少しだけ待つ。"""
        if not self.running:
            return
        self._cancel.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def snapshot(self) -> dict[str, Any]:
        """`detector` メッセージの中身（docs/protocol.md 2.9）。"""
        with self._lock:
            p = self._progress
            running = self.running
            elapsed = (
                (p.finished_at or time.time()) - p.started_at if p.started_at else 0.0
            )
            payload: dict[str, Any] = {
                "state": p.state,
                "running": running,
                "message": p.message,
                "progress": float(max(0.0, min(1.0, p.progress))),
                "collected": int(p.collected),
                "samples": int(p.request.samples) if p.request else 0,
                "epoch": int(p.epoch),
                "epochs": int(p.request.epochs) if p.request else 0,
                "batch": int(p.batch),
                "batches": int(p.batches),
                "history": [dict(h) for h in p.history],
                "elapsedSec": float(max(0.0, elapsed)),
                "warning": p.warning,
                "paramCount": int(p.param_count),
                "presetName": p.preset_name,
                "request": p.request.to_wire() if p.request else None,
                "dataset": p.dataset.to_wire() if p.dataset is not None else None,
            }
        payload["model"] = self.model_info()
        payload["datasetFile"] = self._dataset_file_info()
        payload["limits"] = {
            "samplesMin": SAMPLES_MIN,
            "samplesMax": SAMPLES_MAX,
            "epochsMin": EPOCHS_MIN,
            "epochsMax": EPOCHS_MAX,
            "batchMin": BATCH_MIN,
            "batchMax": BATCH_MAX,
            "widthMin": WIDTH_MIN,
            "widthMax": WIDTH_MAX,
            "seedMin": SEED_MIN,
            "seedMax": SEED_MAX,
        }
        return payload

    def model_info(self) -> dict[str, Any]:
        """`data/detector/detector.keras` の在りかと、いま使われているか。"""
        path = config.DETECTOR_PATH
        info: dict[str, Any] = {
            "exists": False,
            "filename": path.name,
            "sizeBytes": 0,
            "modifiedAt": None,
            "inUse": False,
        }
        try:
            stat = path.stat()
            info["exists"] = True
            info["sizeBytes"] = int(stat.st_size)
            info["modifiedAt"] = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)
            )
        except OSError:
            pass
        try:
            info["inUse"] = bool(self._hooks.detector_in_use())
        except Exception:  # pragma: no cover - 情報表示なので落とさない
            logger.exception("認識器の使用状況を取得できませんでした")
        return info

    def _dataset_path(self) -> Path:
        return config.DETECTOR_DATASET_DIR / DATASET_FILE

    def _dataset_file_info(self) -> dict[str, Any]:
        path = self._dataset_path()
        info: dict[str, Any] = {"exists": False, "sizeBytes": 0, "modifiedAt": None}
        try:
            stat = path.stat()
            info["exists"] = True
            info["sizeBytes"] = int(stat.st_size)
            info["modifiedAt"] = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)
            )
        except OSError:
            pass
        return info

    def _update(self, **changes: Any) -> None:
        """進捗を更新する。配信側は `snapshot()` の中身が変わったら送る。"""
        with self._lock:
            for key, value in changes.items():
                setattr(self._progress, key, value)

    def _should_cancel(self) -> bool:
        return self._cancel.is_set()

    def _run(self, request: DetectorTrainRequest) -> None:
        """ジョブスレッド本体。**何があっても最後に resume_sim() を通す。**"""
        suspended = False
        self._own_index = False
        try:
            self._hooks.suspend_sim("認識器の学習中はシミュレーションを止めています")
            suspended = True
            self._hooks.notify(
                "認識器の学習を始めました。完了するまでシミュレーションは止まります"
            )

            data = self._prepare_dataset(request)
            if request.trains:
                self._train(request, data)
            else:
                self._update(
                    state="done",
                    message="教師データの収集が終わりました",
                    progress=1.0,
                    finished_at=time.time(),
                )
                self._hooks.notify("教師データの収集が終わりました")

        except TrainingCancelled as exc:
            logger.info("認識器の学習を中断しました: %s", exc)
            self._update(
                state="cancelled",
                message=str(exc),
                progress=0.0,
                finished_at=time.time(),
            )
            self._hooks.notify("認識器の学習を中断しました")
        except MemoryError:
            logger.exception("認識器の学習でメモリが足りなくなりました")
            self._update(
                state="error",
                message=(
                    "メモリが足りませんでした。収集枚数を減らすか、"
                    "モデルの大きさ（チャンネル倍率）を下げてください"
                ),
                finished_at=time.time(),
            )
            self._hooks.notify("認識器の学習に失敗しました（メモリ不足）")
        except Exception as exc:  # noqa: BLE001 - 理由を画面へ返したい
            logger.exception("認識器の学習に失敗しました")
            self._update(
                state="error",
                message=f"認識器の学習に失敗しました: {exc}",
                finished_at=time.time(),
            )
            self._hooks.notify(f"認識器の学習に失敗しました: {exc}")
        finally:
            if self._own_index:
                from app.percep.groundtruth import clear_static_cache

                clear_static_cache()
            if suspended:
                self._hooks.resume_sim()

    def _prepare_dataset(self, request: DetectorTrainRequest) -> dict[str, np.ndarray]:
        """教師データを用意する（収集するか、保存済みを読む）。"""
        dataset_path = self._dataset_path()

        if not request.collects:
            self._update(state="preparing", message="保存済みの教師データを読み込んでいます")
            data = load_dataset(dataset_path)
            summary = summarize_dataset(data)
            self._update(
                dataset=summary,
                collected=summary.samples,
                message=f"教師データを読み込みました（{summary.samples} 枚）",
            )
            return data

        index, preset_name = self._resolve_map(request)
        self._update(
            state="collecting",
            preset_name=preset_name,
            message=f"{preset_name} を走らせて教師データを集めています",
            progress=0.0,
            collected=0,
        )

        started = time.perf_counter()
        data = collect_dataset(
            index,
            int(request.samples),
            seed=int(request.seed),
            on_progress=self._on_collect,
            should_cancel=self._should_cancel,
        )
        summary = summarize_dataset(data)
        logger.info(
            "教師データを %d 枚集めました（%.0f 秒、クラス内訳 %s）",
            summary.samples,
            time.perf_counter() - started,
            summary.class_counts,
        )

        self._update(
            state="saving",
            message="教師データを保存しています",
            dataset=summary,
            collected=summary.samples,
            progress=1.0,
        )
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dataset_path, **data)

        empty = [name for name, count in summary.class_counts.items() if count == 0]
        if empty:
            self._update(
                warning=(
                    f"教師データに写っていないクラスがあります: {', '.join(empty)}。"
                    "このクラスは学習しても検出できるようになりません"
                )
            )
        return data

    def _resolve_map(self, request: DetectorTrainRequest) -> tuple["MapIndex", str]:
        """走らせるマップを決める。"""
        current_index, current_id, current_name = self._hooks.current_map()
        wanted = request.preset_id or current_id

        if current_index is not None and (wanted is None or wanted == current_id):
            return current_index, current_name or (current_id or "現在のマップ")

        if wanted is None:
            raise RuntimeError(
                "マップが読み込まれていません。「マップ」タブでエリアを選んでください"
            )

        from app.map import build_map_index, get_preset, load_map

        preset = get_preset(wanted)
        if preset is None:
            raise RuntimeError(f"未知のプリセットです: {wanted}")

        self._update(
            state="preparing",
            preset_name=preset.name,
            message=f"{preset.name} の地図データを読み込んでいます",
        )
        index = build_map_index(load_map(preset))
        self._own_index = True
        if self._should_cancel():
            raise TrainingCancelled("地図の読み込み後に中断しました")
        return index, preset.name

    def _on_collect(self, collected: int, samples: int, elapsed: float) -> None:
        self._update(
            collected=int(collected),
            progress=float(collected) / float(max(1, samples)),
            message=f"教師データを集めています（{collected} / {samples} 枚）",
        )

    def _train(self, request: DetectorTrainRequest, data: dict[str, np.ndarray]) -> None:
        self._update(
            state="training",
            message="認識器を学習しています（初回は Keras の読み込みに数秒かかります）",
            progress=0.0,
            epoch=0,
            batch=0,
            history=[],
        )

        result = fit_detector(
            data,
            epochs=int(request.epochs),
            batch_size=int(request.batch_size),
            out_path=config.DETECTOR_PATH,
            width=float(request.width),
            on_batch=self._on_batch,
            on_epoch=self._on_epoch,
            should_cancel=self._should_cancel,
            log=lambda message: logger.info("[認識器の学習] %s", message),
            verbose=2,
        )

        self._update(
            state="saving",
            message="学習した認識器を実行中の環境へ載せ替えています",
            progress=1.0,
            param_count=result.param_count,
        )

        self._hooks.reload_detector()

        with self._lock:
            existing_warning = self._progress.warning
        warning = result.warning or existing_warning

        counts = ", ".join(str(c) for c in result.verify_counts) or "—"
        self._update(
            state="done",
            message=(
                f"学習が完了しました（{result.epochs_run} エポック / "
                f"パラメータ {result.param_count:,} / 検証時の検出数 {counts}）"
            ),
            warning=warning,
            finished_at=time.time(),
        )
        self._hooks.notify(
            "認識器の学習が完了しました。観測が新しい CNN の出力へ切り替わります"
        )

    def _on_batch(self, epoch: int, batch: int, batches: int) -> None:
        total_epochs = 1
        with self._lock:
            if self._progress.request is not None:
                total_epochs = max(1, int(self._progress.request.epochs))
        done = (float(epoch - 1) + float(batch) / float(max(1, batches))) / total_epochs
        self._update(
            epoch=int(epoch),
            batch=int(batch),
            batches=int(batches),
            progress=done,
            message=f"学習中（{epoch} / {total_epochs} エポック）",
        )

    def _on_epoch(self, epoch: int, epochs: int, entry: dict[str, float]) -> None:
        with self._lock:
            history = list(self._progress.history)
            history.append(dict(entry))
            self._progress.history = history
            self._progress.epoch = int(epoch)
            self._progress.message = (
                f"学習中（{epoch} / {epochs} エポック、損失 {entry['loss']:.4f}）"
            )
