"""ハイパーパラメータの自動探索（Optuna）。CLI（tune_hyperparams.py）と学習タブの「学習の自動化」が同じ実装を使う。"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import logging
import os
import queue
import shutil
import sqlite3
import threading
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from app import config
from app.contracts import (
    AutotuneBest,
    AutotuneSnapshot,
    AutotuneTrial,
    EpisodeResult,
    SimParams,
    StepResult,
)
from app.warn import warn_once

if TYPE_CHECKING:
    import torch

    from app.rl.ppo import PPOTrainer

__all__ = [
    "SEARCH_SPACE",
    "TUNED_WIRE_KEYS",
    "OPTUNA_MISSING_MESSAGE",
    "TrialScore",
    "TrialMonitor",
    "LiveAutoTune",
    "TunePaths",
    "score_trial",
    "is_policy_healthy",
    "atomic_write_json",
    "open_study",
    "optuna_available",
]

logger = logging.getLogger(__name__)

OPTUNA_MISSING_MESSAGE = (
    "Optuna が入っていないため自動探索を始められません。"
    "backend/.venv の Python で pip install -r requirements.txt を実行してから、サーバーを起動し直してください"
)

DB_PATH = config.TUNING_DIR / "driverl_optuna.db"
BEST_PARAMS_PATH = config.TUNING_DIR / "best_params.json"
TRIAL_SUMMARY_PATH = config.TUNING_DIR / "trial_summary.csv"
#: 自動探索の結果を本番の重みへ書く前に、元の重みを控える名前（`shared_policy.pt.before-autotune`）
BACKUP_SUFFIX = ".before-autotune"

#: SQLite のロックが外れるのを待つ時間 [秒]。0 だと別の接続が書いている瞬間に `database is locked` で落ちる
SQLITE_TIMEOUT_SEC = 30.0

#: 学習タブから回す 1 試行のステップ数（20Hz で 76.8 秒ぶん）
LIVE_TRIAL_STEPS = 1536
#: CLI の既定の 1 試行のステップ数
CLI_TRIAL_STEPS = 3072
#: 途中経過を Optuna へ報告する間隔 [ステップ]。ロールアウト長に依らない刻みにする（枝刈りは同じ刻みどうしで比べる）
REPORT_EVERY_STEPS = 256
#: 最後のこの割合のステップだけでスコアを出す（試行の始めは探索前の重みの走りがそのまま出るため）
EVAL_FRACTION = 0.5
#: 枝刈り（`MedianPruner`）は、試行がこれだけ終わるまで掛けない
PRUNER_STARTUP_TRIALS = 5
#: 1 試行の中で、これだけ報告するまでは枝刈りしない
PRUNER_WARMUP_REPORTS = 2
#: 画面に出す直近の試行の数
HISTORY_LEN = 20

#: スコアの重み。**報酬の重みは探索の対象なので、スコアに報酬（`total_reward`）を使わないこと**
SCORE_GOAL = 1.0
SCORE_COLLISION = -1.0
SCORE_OFFROAD = -0.5
SCORE_TIMEOUT = -0.25
SCORE_SIGNAL = -0.1
#: 1 エピソードあたりの信号無視はここで頭打ちにする（ほかの項より大きく効かせない）
SIGNAL_CAP = 3.0
#: 走っている割合（速さ / maxSpeed の平均）。止まったまま固まる方策を下げる
SCORE_MOVING = 0.3
#: 重みが NaN / Inf になった試行のスコア。取りうる最小（-1.3）より下に置き、TPE にその辺りを避けさせる
SCORE_DIVERGED = -2.0


@dataclass(frozen=True)
class SearchDim:
    """探索する 1 つのパラメータ。値域と刻みは `contracts._PARAM_SPECS` と学習タブのスライダーに合わせる。"""

    snake: str
    wire: str
    low: float
    high: float
    log: bool = False
    #: 刻み（スライダーと同じ）。最良の値をスライダーでそのまま再現できるようにする。対数で探すものは None
    step: float | None = None

    @property
    def digits(self) -> int | None:
        """刻みに合わせて丸める小数の桁数（0.1 なら 1）。刻みが無ければ None。"""
        if self.step is None:
            return None
        text = f"{self.step:.10f}".rstrip("0").rstrip(".")
        return len(text.split(".")[1]) if "." in text else 0


SEARCH_SPACE: tuple[SearchDim, ...] = (
    SearchDim("learning_rate", "learningRate", 1e-5, 1e-3, log=True),
    SearchDim("gamma", "gamma", 0.95, 0.999, step=0.001),
    SearchDim("clip_range", "clipRange", 0.05, 0.4, step=0.01),
    SearchDim("entropy_coef", "entropyCoef", 0.0, 0.01, step=0.001),
    SearchDim("reward_goal", "rewardGoal", 20.0, 300.0, step=5.0),
    SearchDim("reward_collision", "rewardCollision", -300.0, -20.0, step=5.0),
    SearchDim("reward_progress", "rewardProgress", 0.2, 3.0, step=0.1),
    SearchDim("reward_offroad", "rewardOffroad", -5.0, -0.1, step=0.1),
    SearchDim("reward_signal", "rewardSignal", -200.0, -10.0, step=5.0),
    SearchDim("reward_overspeed", "rewardOverspeed", -50.0, -1.0, step=1.0),
    SearchDim("reward_time", "rewardTime", -0.5, 0.0, step=0.01),
)
TUNED_WIRE_KEYS: tuple[str, ...] = tuple(d.wire for d in SEARCH_SPACE)


class AutotuneUnavailable(RuntimeError):
    """Optuna が入っていない・履歴の DB を開けないなど、探索を始められない。"""


def optuna_available() -> bool:
    """Optuna が import できるか（import はせずに探すだけ。起動を重くしない）。"""
    return importlib.util.find_spec("optuna") is not None


def distributions() -> dict[str, Any]:
    """Optuna の `ask()` に渡す探索空間。名前は params のワイヤ名（camelCase）にする。"""
    from optuna.distributions import FloatDistribution

    return {d.wire: FloatDistribution(d.low, d.high, log=d.log, step=d.step) for d in SEARCH_SPACE}


def tuned_patch(params: SimParams) -> dict[str, float]:
    """探索の対象の値だけを抜き出す（ワイヤ名）。"""
    wire = params.to_wire()
    return {key: float(wire[key]) for key in TUNED_WIRE_KEYS}


def validated_patch(raw: dict[str, Any], base: SimParams) -> dict[str, float]:
    """探索の対象のキーだけを `apply_wire` に通して検証する。範囲外は丸め、読めない値は元のまま、刻みのあるものは刻みの桁に丸める。"""
    params = SimParams(**vars(base))
    params.apply_wire({k: v for k, v in raw.items() if k in TUNED_WIRE_KEYS})
    patch = tuned_patch(params)
    for d in SEARCH_SPACE:
        digits = d.digits
        if digits is not None:
            # Optuna の刻みは low + k * step で作るので、0.9000000000000001 のような値になる
            patch[d.wire] = round(patch[d.wire], digits)
    return patch


@dataclass(frozen=True)
class TrialScore:
    """1 試行の成績。値（`value`）が Optuna の目的関数の値で、大きいほど良い。"""

    value: float
    episodes: int
    goal_rate: float
    collision_rate: float
    offroad_rate: float
    timeout_rate: float
    signal_per_episode: float
    moving: float

    def attrs(self) -> dict[str, float]:
        return {
            "episodes": float(self.episodes),
            "goal_rate": round(self.goal_rate, 4),
            "collision_rate": round(self.collision_rate, 4),
            "offroad_rate": round(self.offroad_rate, 4),
            "timeout_rate": round(self.timeout_rate, 4),
            "signal_per_episode": round(self.signal_per_episode, 4),
            "moving": round(self.moving, 4),
        }


def score_trial(episodes: Sequence[EpisodeResult], moving: float) -> TrialScore:
    """終わったエピソードの結果と走っていた割合からスコアを出す。報酬の重みには依らない。"""
    n = len(episodes)
    if n:
        goal = sum(1 for e in episodes if e.reason == "goal") / n
        collision = sum(1 for e in episodes if e.reason == "collision") / n
        offroad = sum(1 for e in episodes if e.reason == "offroad") / n
        timeout = sum(1 for e in episodes if e.reason == "timeout") / n
        signal = min(sum(e.signal_violations for e in episodes) / n, SIGNAL_CAP)
    else:
        goal = collision = offroad = timeout = signal = 0.0
    moving = float(np.clip(moving, 0.0, 1.0))
    value = (
        SCORE_GOAL * goal
        + SCORE_COLLISION * collision
        + SCORE_OFFROAD * offroad
        + SCORE_TIMEOUT * timeout
        + SCORE_SIGNAL * signal
        + SCORE_MOVING * moving
    )
    return TrialScore(float(value), n, goal, collision, offroad, timeout, float(signal), moving)


@dataclass
class _Tally:
    episodes: list[EpisodeResult] = field(default_factory=list)
    moving_sum: float = 0.0
    moving_count: int = 0

    def add(self, episodes: Iterable[EpisodeResult], moving: np.ndarray) -> None:
        self.episodes.extend(episodes)
        self.moving_sum += float(moving.sum())
        self.moving_count += int(moving.size)

    def score(self) -> TrialScore:
        mean = self.moving_sum / self.moving_count if self.moving_count else 0.0
        return score_trial(self.episodes, mean)


class TrialMonitor:
    """1 試行ぶんの走りを見届けてスコアを出す。CLI もエンジンも毎ステップ `observe` する。"""

    def __init__(
        self,
        budget_steps: int,
        max_speed: float,
        report_every: int = REPORT_EVERY_STEPS,
    ) -> None:
        self.budget = max(1, int(budget_steps))
        self.report_every = max(1, int(report_every))
        self.eval_from = self.budget - max(1, int(round(self.budget * EVAL_FRACTION)))
        self.max_speed = max(1e-6, float(max_speed))
        self.steps = 0
        self._all = _Tally()
        self._eval = _Tally()

    def observe(self, result: StepResult, speeds: np.ndarray) -> None:
        self.steps += 1
        active = np.asarray(result.active, dtype=bool)
        moving = np.clip(np.abs(np.asarray(speeds, dtype=np.float64)[active]) / self.max_speed, 0.0, 1.0)
        self._all.add(result.episodes, moving)
        if self.steps > self.eval_from:
            self._eval.add(result.episodes, moving)

    @property
    def done(self) -> bool:
        return self.steps >= self.budget

    @property
    def progress(self) -> float:
        return min(1.0, self.steps / self.budget)

    def checkpoint(self) -> int | None:
        """途中経過を報告する刻みに来たら、その番号（1 から）。最後のステップでは報告しない（最終の値を出すため）。"""
        if self.done or self.steps % self.report_every != 0:
            return None
        return self.steps // self.report_every

    def interim(self) -> TrialScore:
        """ここまでの全ステップのスコア（枝刈りの判断に使う）。"""
        return self._all.score()

    def final(self) -> TrialScore:
        """評価の窓（最後の `EVAL_FRACTION`）のスコア。"""
        return self._eval.score()


def is_policy_healthy(policy: "torch.nn.Module") -> bool:
    """重みに NaN や Inf が混じっていないか。混じった重みは保存も適用もしない。"""
    import torch

    with torch.no_grad():
        return all(bool(torch.isfinite(p).all()) for p in policy.parameters())


def atomic_write_text(path: Path, text: str) -> None:
    """一時ファイル（`.tmp`）へ書き切ってから `os.replace` で差し替える。途中で落ちても元のファイルは残る。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, data: Any) -> None:
    """JSON をアトミックに書き出す。"""
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_best_params(path: Path, patch: dict[str, float]) -> None:
    """最良のパラメータを書き出す。キーは `SimParams.to_wire()` と同じ（探索の対象の部分集合）なので、そのまま `set_params` に渡せる。"""
    atomic_write_json(path, {key: patch[key] for key in TUNED_WIRE_KEYS if key in patch})


def save_policy_checked(trainer: "PPOTrainer", path: Path) -> bool:
    """重みが健全なときだけアトミックに保存する。NaN / Inf が混じっていれば保存せず False。"""
    if not is_policy_healthy(trainer.policy):
        return False
    trainer.save(path)
    return True


def backup_checkpoint(path: Path) -> Path | None:
    """本番の重みを `<名前>.before-autotune` へ控える。元が無ければ None。失敗は OSError のまま投げる。"""
    path = Path(path)
    if not path.exists():
        return None
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    tmp = backup.with_name(backup.name + ".tmp")
    shutil.copy2(path, tmp)
    os.replace(tmp, backup)
    return backup


_SUMMARY_SCORE_KEYS = (
    "episodes",
    "goal_rate",
    "collision_rate",
    "offroad_rate",
    "timeout_rate",
    "signal_per_episode",
    "moving",
)


def _shown_param(dim: SearchDim, value: Any) -> Any:
    if value is None:
        return ""
    digits = dim.digits
    return round(float(value), digits) if digits is not None else value


def write_trial_summary(study: Any, path: Path) -> None:
    """study の全試行を CSV に書き出す（毎回まるごと作り直してアトミックに差し替える）。"""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(
        ["trial", "state", "score", "started", "finished", "seconds", *TUNED_WIRE_KEYS, *_SUMMARY_SCORE_KEYS, "outcome"]
    )
    for t in study.trials:
        seconds = ""
        if t.datetime_start is not None and t.datetime_complete is not None:
            seconds = f"{(t.datetime_complete - t.datetime_start).total_seconds():.1f}"
        writer.writerow(
            [
                t.number,
                t.state.name,
                "" if t.value is None else f"{t.value:.6f}",
                "" if t.datetime_start is None else t.datetime_start.isoformat(timespec="seconds"),
                "" if t.datetime_complete is None else t.datetime_complete.isoformat(timespec="seconds"),
                seconds,
                *(_shown_param(d, t.params.get(d.wire)) for d in SEARCH_SPACE),
                *(t.user_attrs.get(key, "") for key in _SUMMARY_SCORE_KEYS),
                t.user_attrs.get("outcome", ""),
            ]
        )
    atomic_write_text(path, buf.getvalue())


def _enable_wal(db_path: Path) -> None:
    """DB ファイルを WAL にしておく（ファイルに残るので一度で効く）。Optuna が最初に開く接続より先に行う。"""
    con = sqlite3.connect(str(db_path), timeout=SQLITE_TIMEOUT_SEC)
    try:
        mode = con.execute("PRAGMA journal_mode=WAL").fetchone()
        if not mode or str(mode[0]).lower() != "wal":
            warn_once("runtime.autotune.wal", f"探索の履歴の DB を WAL にできませんでした（{mode}）")
    finally:
        con.close()


def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
    cur = dbapi_connection.cursor()
    try:
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute(f"PRAGMA busy_timeout={int(SQLITE_TIMEOUT_SEC * 1000)}")
    finally:
        cur.close()


def open_study(db_path: Path, study_name: str, *, seed: int | None = None, prune: bool = True) -> tuple[Any, Any]:
    """履歴の DB（WAL）を開き、同じ名前の study があれば続きから、無ければ作って返す。(study, storage)。"""
    try:
        import optuna
        from sqlalchemy import event
    except ImportError as exc:
        raise AutotuneUnavailable(OPTUNA_MISSING_MESSAGE) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _enable_wal(db_path)
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path.resolve().as_posix()}",
        engine_kwargs={"connect_args": {"timeout": SQLITE_TIMEOUT_SEC}},
    )
    engine = getattr(storage, "engine", None)
    if engine is not None:
        event.listen(engine, "connect", _sqlite_pragmas)
        # 登録より前に作られた接続（テーブルを作ったもの）には効かないので、捨てて張り直させる
        engine.dispose()
    else:
        warn_once("runtime.autotune.no_engine", "この Optuna の RDBStorage に engine が無いため、接続ごとの PRAGMA を掛けられません")

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner: Any = (
        optuna.pruners.MedianPruner(
            n_startup_trials=PRUNER_STARTUP_TRIALS, n_warmup_steps=PRUNER_WARMUP_REPORTS
        )
        if prune
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    return study, storage


def close_storage(storage: Any) -> None:
    """接続を閉じる（Windows では開いたままだと DB ファイルを消せない）。"""
    engine = getattr(storage, "engine", None)
    if engine is not None:
        try:
            engine.dispose()
        except Exception:
            warn_once("runtime.autotune.dispose", "探索の履歴の DB の接続を閉じられませんでした")


def completed_trials(study: Any) -> int:
    from optuna.trial import TrialState

    return sum(1 for t in study.trials if t.state == TrialState.COMPLETE)


_OUTCOME_STATES = {"complete": "COMPLETE", "diverged": "COMPLETE", "pruned": "PRUNED", "interrupted": "FAIL"}


class OptunaWorker:
    """Optuna の study を専用スレッドで持つ。SQLite と TPE をエンジンスレッドに乗せないため。"""

    def __init__(self, db_path: Path, study_name: str, summary_path: Path, *, seed: int | None = None) -> None:
        self.db_path = Path(db_path)
        self.study_name = study_name
        self.summary_path = Path(summary_path)
        self.seed = seed
        self._requests: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._replies: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._closing = threading.Event()
        self._thread = threading.Thread(target=self._run, name="autotune-optuna", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def ask(self) -> None:
        self._requests.put(("ask", None))

    def report(self, number: int, step: int, value: float) -> None:
        self._requests.put(("report", (int(number), int(step), float(value))))

    def tell(self, number: int, value: float | None, outcome: str, attrs: dict[str, Any]) -> None:
        self._requests.put(("tell", (int(number), value, outcome, dict(attrs))))

    def submit(self, fn: Callable[[], None], what: str) -> None:
        """ファイルの書き出しを頼む（エンジンスレッドで fsync を待たない）。頼んだ順に、close より前に済む。"""
        self._requests.put(("io", (fn, what)))

    def close(self) -> None:
        self._closing.set()
        self._requests.put(("close", None))

    def join(self, timeout: float) -> None:
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def poll(self) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        while True:
            try:
                out.append(self._replies.get_nowait())
            except queue.Empty:
                return out

    def _run(self) -> None:
        try:
            study, storage = open_study(self.db_path, self.study_name, seed=self.seed)
        except AutotuneUnavailable as exc:
            self._replies.put(("error", str(exc)))
            return
        except Exception:
            logger.exception("探索の履歴の DB を開けませんでした: %s", self.db_path)
            self._replies.put(("error", "探索の履歴の DB を開けませんでした。サーバーのログを見てください"))
            return

        from optuna.trial import TrialState

        open_trials: dict[int, Any] = {}
        try:
            self._replies.put(("ready", completed_trials(study)))
            space = distributions()
            while True:
                kind, payload = self._requests.get()
                if kind == "close":
                    break
                if kind == "io":
                    fn, what = payload
                    try:
                        fn()
                    except Exception:
                        logger.exception("%sを書き出せませんでした（探索は続けます）", what)
                elif kind == "ask":
                    if self._closing.is_set():
                        # 走らせない試行を作ると、中断（FAIL）の行だけが履歴に残る
                        continue
                    trial = study.ask(space)
                    open_trials[trial.number] = trial
                    self._replies.put(("asked", (trial.number, {k: float(v) for k, v in trial.params.items()})))
                elif kind == "report":
                    number, step, value = payload
                    trial = open_trials.get(number)
                    if trial is not None:
                        trial.report(value, step)
                        if trial.should_prune():
                            self._replies.put(("pruned", number))
                elif kind == "tell":
                    number, value, outcome, attrs = payload
                    trial = open_trials.pop(number, None)
                    if trial is None:
                        continue
                    for key, item in attrs.items():
                        trial.set_user_attr(key, item)
                    trial.set_user_attr("outcome", outcome)
                    state = TrialState[_OUTCOME_STATES.get(outcome, "FAIL")]
                    study.tell(trial, value if state == TrialState.COMPLETE else None, state=state)
                    try:
                        write_trial_summary(study, self.summary_path)
                    except Exception:
                        logger.exception("試行の一覧（CSV）を書き出せませんでした（探索は続けます）")
                    self._replies.put(("told", number))
        except Exception:
            logger.exception("自動探索の Optuna の処理で例外が発生しました")
            self._replies.put(("error", "自動探索の履歴の記録に失敗しました。サーバーのログを見てください"))
        finally:
            for trial in open_trials.values():
                try:
                    trial.set_user_attr("outcome", "interrupted")
                    study.tell(trial, state=TrialState.FAIL)
                except Exception:
                    warn_once("runtime.autotune.close_trial", "中断した試行を閉じられませんでした")
            close_storage(storage)
            self._replies.put(("closed", None))


class TuneHost(Protocol):
    """探索のセッションがエンジンへ頼む操作（エンジンスレッドから呼ぶ）。"""

    def begin_trial(self, patch: dict[str, float], state: dict[str, Any]) -> None:
        """重みを `state` へ巻き戻し、`patch` のパラメータにして、街を作り直す。"""

    def restore(self, patch: dict[str, float], state: dict[str, Any]) -> None:
        """重みを `state` へ巻き戻し、`patch` のパラメータにする（街はそのまま）。"""


@dataclass(frozen=True)
class TunePaths:
    """探索が書くファイル。テストでは一時ディレクトリへ差し替える。"""

    db: Path = DB_PATH
    best_params: Path = BEST_PARAMS_PATH
    summary: Path = TRIAL_SUMMARY_PATH
    best_policy: Path = config.BEST_TUNED_POLICY_PATH
    checkpoint: Path = config.CHECKPOINT_PATH


@dataclass
class _Best:
    number: int
    score: TrialScore
    patch: dict[str, float]
    state: dict[str, Any]


class LiveAutoTune:
    """学習タブの「学習の自動化」。エンジンスレッドから毎ステップ `tick` され、試行を切り替える。"""

    def __init__(
        self,
        *,
        paths: TunePaths | None = None,
        trial_steps: int = LIVE_TRIAL_STEPS,
        report_every: int = REPORT_EVERY_STEPS,
        seed: int | None = None,
    ) -> None:
        self.paths = paths or TunePaths()
        self.trial_steps = int(trial_steps)
        self.report_every = int(report_every)
        self.seed = seed
        self.available = optuna_available()
        self.revision = 0
        self._retired: list[OptunaWorker] = []
        self._reset_session()
        self.message = "自動探索はまだ実行していません"

    def _reset_session(self) -> None:
        self.phase = "idle"
        self.study_name: str | None = None
        self._worker: OptunaWorker | None = None
        self._baseline_state: dict[str, Any] | None = None
        self._baseline_patch: dict[str, float] = {}
        self._max_speed = config.MAX_SPEED
        self._current: int | None = None
        self._current_patch: dict[str, float] = {}
        self._monitor: TrialMonitor | None = None
        self.finished = 0
        self.prior = 0
        self.best: _Best | None = None
        self.history: deque[AutotuneTrial] = deque(maxlen=HISTORY_LEN)
        self.revision += 1

    @property
    def active(self) -> bool:
        return self.phase != "idle"

    def start(self, trainer: "PPOTrainer", params: SimParams, preset_id: str | None) -> str | None:
        """探索を始める。始められなければ理由を返す。"""
        if self.active:
            return "すでに自動探索を実行中です"
        if not self.available:
            return OPTUNA_MISSING_MESSAGE
        if not is_policy_healthy(trainer.policy):
            return "いまの重みに NaN / Inf が混じっているため、自動探索を始められません。ポリシーを初期化するか、読み込み直してください"
        self._reset_session()
        self._baseline_state = trainer.snapshot_state()
        self._baseline_patch = tuned_patch(params)
        self._max_speed = float(params.max_speed)
        try:
            # 退避（OFF にしたとき）がちょうど探索前の重みになるように、いまの重みを本番へ書いておく
            trainer.save(self.paths.checkpoint)
        except Exception:
            logger.exception("自動探索の前の重みを保存できませんでした（探索は始めます）")
        self.study_name = f"live-{preset_id or 'unknown'}"
        worker = OptunaWorker(self.paths.db, self.study_name, self.paths.summary, seed=self.seed)
        worker.start()
        worker.ask()
        self._worker = worker
        self.phase = "preparing"
        self.message = "自動探索の準備をしています（探索の履歴を開いています）"
        self.revision += 1
        return None

    def tick(
        self,
        host: TuneHost,
        trainer: "PPOTrainer",
        result: StepResult | None,
        speeds: np.ndarray | None,
    ) -> list[str]:
        """1 ステップぶん見届け、Optuna の返事を処理する。画面へ出す知らせを返す。"""
        if not self.active or self._worker is None:
            return []
        notices: list[str] = []
        monitor = self._monitor
        if self.phase == "running" and monitor is not None and result is not None and speeds is not None:
            monitor.observe(result, speeds)
            if not is_policy_healthy(trainer.policy):
                self._finish(trainer, "diverged")
            else:
                k = monitor.checkpoint()
                if k is not None and self._current is not None:
                    self._worker.report(self._current, k, monitor.interim().value)
                if monitor.done:
                    self._finish(trainer, "complete")

        for kind, payload in self._worker.poll():
            if kind == "ready":
                self.prior = int(payload)
                self.revision += 1
            elif kind == "asked":
                number, raw = payload
                self._begin(host, int(number), raw)
            elif kind == "pruned":
                if payload == self._current:
                    self._finish(trainer, "pruned")
            elif kind == "error":
                notices.append(f"{payload}。自動探索を中止し、探索の前のパラメータと重みに戻しました")
                self.abort(host, "")
                return notices
        return notices

    def _begin(self, host: TuneHost, number: int, raw: dict[str, Any]) -> None:
        assert self._baseline_state is not None
        patch = validated_patch(raw, SimParams())
        self._current = number
        self._current_patch = patch
        host.begin_trial(patch, self._baseline_state)
        self._monitor = TrialMonitor(self.trial_steps, self._max_speed, self.report_every)
        self.phase = "running"
        self.message = f"試行 #{number} を走らせています"
        self.revision += 1

    def _finish(self, trainer: "PPOTrainer", outcome: str) -> None:
        monitor = self._monitor
        number = self._current
        if monitor is None or number is None or self._worker is None:
            return
        attrs: dict[str, Any] = {"steps": monitor.steps}
        value: float | None
        if outcome == "complete":
            score = monitor.final()
            value = score.value
            attrs.update(score.attrs())
        elif outcome == "pruned":
            score = monitor.interim()
            value = None
            attrs.update(score.attrs())
        else:
            score = None
            value = SCORE_DIVERGED
        self._worker.tell(number, value, outcome, attrs)
        shown = value if value is not None else (score.value if score is not None else None)
        self.history.append(AutotuneTrial(trial=number, score=shown, outcome=outcome))
        self.finished += 1

        better = score is not None and (self.best is None or score.value > self.best.score.value)
        if outcome == "complete" and score is not None and better and is_policy_healthy(trainer.policy):
            state = trainer.snapshot_state()
            patch = dict(self._current_patch)
            self.best = _Best(number, score, patch, state)
            paths = self.paths

            def write() -> None:
                trainer.save_state(state, paths.best_policy)
                write_best_params(paths.best_params, patch)

            # 複製を書くので、エンジンスレッドは学習を続けてよい（保存と fsync で 1 ステップが 50ms を超えていた）
            self._worker.submit(write, "最良の試行の重み・パラメータ")

        self._current = None
        self._monitor = None
        self.phase = "waiting"
        self.message = "次の試行のパラメータを選んでいます"
        self._worker.ask()
        self.revision += 1

    def stop(self, host: TuneHost, trainer: "PPOTrainer") -> str:
        """探索をやめ、最良の試行のパラメータと重みを適用して本番の重みへ保存する。画面へ出す文を返す。"""
        if not self.active:
            return "自動探索は実行していません"
        self._close_worker()
        best = self.best
        baseline = self._baseline_state
        assert baseline is not None
        if best is None:
            host.restore(self._baseline_patch, baseline)
            message = "完了した試行が無かったため、自動探索の前のパラメータと重みに戻しました"
        else:
            try:
                backup = backup_checkpoint(self.paths.checkpoint)
            except OSError:
                logger.exception("自動探索の結果を適用する前の退避に失敗しました")
                host.restore(self._baseline_patch, baseline)
                message = (
                    "いまの重みを退避できなかったため、最良の試行は適用せず、自動探索の前のパラメータと重みに戻しました。"
                    f"最良の試行の重みは {self.paths.best_policy.name} に残っています"
                )
            else:
                host.restore(best.patch, best.state)
                if save_policy_checked(trainer, self.paths.checkpoint):
                    kept = f"元の重みは {backup.name} に退避しました" if backup is not None else "元の重みはありませんでした"
                    message = (
                        f"試行 #{best.number}（スコア {best.score.value:+.3f}）のパラメータと重みを適用し、"
                        f"{self.paths.checkpoint.name} に保存しました（{kept}）。この設定のまま学習を続けます"
                    )
                else:
                    host.restore(self._baseline_patch, baseline)
                    message = "最良の試行の重みに NaN / Inf が混じっていたため適用せず、自動探索の前のパラメータと重みに戻しました"
        self._reset_session()
        self.message = message
        return message

    def abort(self, host: TuneHost | None, reason: str) -> str:
        """探索を捨てて、探索の前のパラメータと重みに戻す（地図の差し替え・失敗など）。"""
        if not self.active:
            return ""
        self._close_worker()
        if host is not None and self._baseline_state is not None:
            host.restore(self._baseline_patch, self._baseline_state)
        self._reset_session()
        self.message = reason or "自動探索を中止しました"
        return self.message

    def shutdown(self, timeout: float = 2.0) -> None:
        """サーバーを止めるとき。走っている試行を閉じ、頼んだ書き出しを済ませて DB の接続を手放す（重みには触らない）。"""
        self._close_worker()
        retired, self._retired = self._retired, []
        for worker in retired:
            worker.join(timeout)

    def _close_worker(self) -> None:
        worker = self._worker
        if worker is None:
            return
        if self._current is not None:
            attrs = {"steps": self._monitor.steps if self._monitor is not None else 0}
            worker.tell(self._current, None, "interrupted", attrs)
        worker.close()
        self._worker = None
        # 閉じた後も、頼んだ書き出しと履歴の記録が終わるまでスレッドは動く
        self._retired = [w for w in self._retired if w.alive] + [worker]

    def snapshot(self) -> AutotuneSnapshot:
        """画面へ配る形（`autotune` メッセージ）。"""
        best = self.best
        monitor = self._monitor
        return AutotuneSnapshot(
            available=self.available,
            running=self.active,
            phase=self.phase,
            study_name=self.study_name,
            trial=self._current,
            trial_progress=monitor.progress if monitor is not None else 0.0,
            trial_steps=self.trial_steps,
            finished_trials=self.finished,
            prior_trials=self.prior,
            tuned_keys=list(TUNED_WIRE_KEYS),
            current=dict(self._current_patch) if self._current is not None else None,
            best=(
                AutotuneBest(trial=best.number, score=best.score.value, params=dict(best.patch))
                if best is not None
                else None
            ),
            history=list(self.history),
            message=self.message,
        )
