"""ハイパーパラメータの自動探索（runtime/autotune.py・tune_hyperparams.py・エンジンの学習の自動化）の検査。地図のキャッシュは読まない。"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from app import config
from app.contracts import _PARAM_SPECS, EpisodeResult, SimParams, StepResult
from app.rl.ppo import PPOTrainer
from app.runtime import autotune

optuna = pytest.importorskip("optuna")

N = config.MAX_VEHICLES
FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "src"


def _episode(reason: str, reward: float = 0.0, violations: int = 0) -> EpisodeResult:
    return EpisodeResult(slot=0, total_reward=reward, length=100, reason=reason, signal_violations=violations)


def _step(active: int, episodes: list[EpisodeResult] | None = None) -> StepResult:
    mask = np.zeros(N, dtype=bool)
    mask[:active] = True
    zeros = np.zeros(N, dtype=np.float32)
    return StepResult(obs=np.zeros((N, config.OBS_DIM), np.float32), rewards=zeros, dones=mask & False, active=mask, episodes=episodes or [])


def _trainer(rollout: int = 16) -> PPOTrainer:
    return PPOTrainer(config.OBS_DIM, config.ACTION_DIM, SimParams(rollout_length=rollout), N, seed=0)


def _state_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


class TestSearchSpace:
    def test_inside_param_specs_and_not_clamped(self) -> None:
        wire = SimParams().to_wire()
        for d in autotune.SEARCH_SPACE:
            spec = _PARAM_SPECS[d.snake]
            assert spec.wire == d.wire
            assert d.wire in wire
            assert spec.minimum <= d.low < d.high <= spec.maximum, d
            for edge in (d.low, d.high):
                result = SimParams().apply_wire({d.wire: edge})
                assert not result.has_problem, (d.wire, edge)

    def test_ranges_and_steps_fit_the_learning_tab_sliders(self) -> None:
        source = (FRONTEND / "panel" / "LearningTab.tsx").read_text(encoding="utf-8")
        for d in autotune.SEARCH_SPACE:
            head = source.index(f"value={{params.{d.wire}}}")
            block = source[head : head + 300]
            low = float(block.split("min={", 1)[1].split("}", 1)[0])
            high = float(block.split("max={", 1)[1].split("}", 1)[0])
            step = float(block.split("step={", 1)[1].split("}", 1)[0])
            assert low <= d.low and d.high <= high, f"{d.wire} の探索範囲がスライダー（{low}〜{high}）からはみ出す"
            assert f"auto={{auto('{d.wire}')}}" in block, f"{d.wire} のスライダーが自動調整中に動かせてしまう"
            if d.step is not None:
                assert d.step == step, f"{d.wire} の刻み {d.step} がスライダー（{step}）と違う"

    def test_mock_search_space_matches(self) -> None:
        source = (FRONTEND / "store" / "mock" / "autotune.ts").read_text(encoding="utf-8")
        rows = re.findall(
            r"\{ key: '(\w+)', low: ([-\d.e]+), high: ([-\d.e]+)(, log: true)?(?:, step: ([-\d.e]+))? \}", source
        )
        assert [(k, float(lo), float(hi), bool(lg), float(st) if st else None) for k, lo, hi, lg, st in rows] == [
            (d.wire, d.low, d.high, d.log, d.step) for d in autotune.SEARCH_SPACE
        ]

    def test_digits_follow_the_step(self) -> None:
        digits = {d.wire: d.digits for d in autotune.SEARCH_SPACE}
        assert digits["learningRate"] is None
        assert digits["gamma"] == 3 and digits["rewardProgress"] == 1 and digits["rewardGoal"] == 0
        patch = autotune.validated_patch({"rewardProgress": 0.9000000000000001, "gamma": 0.9810000000000001}, SimParams())
        assert patch["rewardProgress"] == 0.9 and patch["gamma"] == 0.981

    def test_distributions_use_wire_names(self) -> None:
        dists = autotune.distributions()
        assert tuple(dists) == autotune.TUNED_WIRE_KEYS
        assert dists["learningRate"].log is True

    def test_validated_patch_clamps_and_keeps_only_tuned_keys(self) -> None:
        patch = autotune.validated_patch({"learningRate": 5.0, "vehicleCount": 99, "gamma": "x"}, SimParams())
        assert set(patch) == set(autotune.TUNED_WIRE_KEYS)
        assert patch["learningRate"] == _PARAM_SPECS["learning_rate"].maximum
        assert patch["gamma"] == SimParams().gamma


class TestScore:
    def test_does_not_depend_on_reward(self) -> None:
        a = autotune.score_trial([_episode("goal", 500.0), _episode("collision", -900.0)], 0.5)
        b = autotune.score_trial([_episode("goal", 1.0), _episode("collision", -1.0)], 0.5)
        assert a.value == b.value

    def test_goal_beats_collision_and_moving_beats_stuck(self) -> None:
        goal = autotune.score_trial([_episode("goal")] * 4, 0.6).value
        crash = autotune.score_trial([_episode("collision")] * 4, 0.6).value
        assert goal > crash
        assert autotune.score_trial([], 0.8).value > autotune.score_trial([], 0.0).value

    def test_signal_violations_are_capped(self) -> None:
        many = autotune.score_trial([_episode("timeout", violations=50)], 0.0)
        assert many.signal_per_episode == autotune.SIGNAL_CAP
        assert many.value > autotune.SCORE_DIVERGED

    def test_monitor_scores_only_the_evaluation_window(self) -> None:
        monitor = autotune.TrialMonitor(budget_steps=8, max_speed=10.0, report_every=2)
        checkpoints = []
        for i in range(8):
            episodes = [_episode("collision")] if i < 4 else [_episode("goal")]
            monitor.observe(_step(2, episodes), np.full(N, 5.0))
            checkpoints.append(monitor.checkpoint())
        assert monitor.done
        assert checkpoints == [None, 1, None, 2, None, 3, None, None]
        final = monitor.final()
        assert final.goal_rate == 1.0 and final.collision_rate == 0.0 and final.episodes == 4
        assert monitor.interim().collision_rate == 0.5
        assert final.moving == pytest.approx(0.5)


class TestFiles:
    def test_atomic_json_keeps_old_file_when_replace_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "best_params.json"
        autotune.atomic_write_json(path, {"learningRate": 1e-4})

        def boom(*_args: object) -> None:
            raise OSError("電源断のつもり")

        monkeypatch.setattr(autotune.os, "replace", boom)
        with pytest.raises(OSError):
            autotune.atomic_write_json(path, {"learningRate": 9.0})
        assert json.loads(path.read_text(encoding="utf-8")) == {"learningRate": 1e-4}
        assert not (tmp_path / "best_params.json.tmp").exists()

    def test_best_params_can_be_applied_as_set_params(self, tmp_path: Path) -> None:
        path = tmp_path / "best_params.json"
        patch = autotune.tuned_patch(SimParams(learning_rate=2e-4, reward_goal=150.0))
        autotune.write_best_params(path, patch)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        params = SimParams()
        assert not params.apply_wire(loaded).has_problem
        assert params.learning_rate == 2e-4 and params.reward_goal == 150.0

    def test_backup_checkpoint(self, tmp_path: Path) -> None:
        path = tmp_path / "shared_policy.pt"
        assert autotune.backup_checkpoint(path) is None
        path.write_bytes(b"weights")
        backup = autotune.backup_checkpoint(path)
        assert backup is not None and backup.name == "shared_policy.pt.before-autotune"
        assert backup.read_bytes() == b"weights"

    def test_open_study_uses_wal_and_busy_timeout(self, tmp_path: Path) -> None:
        db = tmp_path / "driverl_optuna.db"
        study, storage = autotune.open_study(db, "wal-check", seed=0)
        try:
            trial = study.ask(autotune.distributions())
            study.tell(trial, 0.1)
            with storage.engine.connect() as con:
                raw = con.connection.dbapi_connection
                assert raw.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
                assert raw.execute("PRAGMA synchronous").fetchone()[0] == 1
                assert raw.execute("PRAGMA busy_timeout").fetchone()[0] == int(autotune.SQLITE_TIMEOUT_SEC * 1000)
        finally:
            autotune.close_storage(storage)
        con = sqlite3.connect(db)
        try:
            assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        finally:
            con.close()


class TestTrainerState:
    def test_health_check_finds_nan(self, tmp_path: Path) -> None:
        trainer = _trainer()
        assert autotune.is_policy_healthy(trainer.policy)
        with torch.no_grad():
            next(trainer.policy.parameters()).view(-1)[0] = float("nan")
        assert not autotune.is_policy_healthy(trainer.policy)
        target = tmp_path / "never-written.pt"
        assert not autotune.save_policy_checked(trainer, target)
        assert not target.exists()

    def test_restore_brings_back_weights_and_is_not_changed_by_training(self) -> None:
        trainer = _trainer(rollout=16)
        snapshot = trainer.snapshot_state()
        frozen = {k: v.clone() for k, v in snapshot["policy"].items()}
        rng = np.random.default_rng(0)
        active = np.ones(N, dtype=bool)
        for _ in range(64):
            obs = rng.normal(size=(N, config.OBS_DIM)).astype(np.float32)
            actions, log_probs, values = trainer.act(obs, active)
            trainer.store(obs, actions, log_probs, values, rng.normal(size=N).astype(np.float32), np.zeros(N, bool), active)
            trainer.maybe_update(obs, active)
        assert trainer.updates > 0
        assert not _state_equal(trainer.policy.state_dict(), frozen)
        assert _state_equal(snapshot["policy"], frozen), "学習で手元の複製まで書き換わった"
        trainer.restore_state(snapshot)
        assert trainer.updates == 0
        assert _state_equal(trainer.policy.state_dict(), frozen)
        # 巻き戻した後に学習しても、複製（Adam の統計を含む）は書き換わらない
        exp_avg = [s["exp_avg"].clone() for s in snapshot["optimizer"]["state"].values() if "exp_avg" in s]
        for _ in range(32):
            obs = rng.normal(size=(N, config.OBS_DIM)).astype(np.float32)
            actions, log_probs, values = trainer.act(obs, active)
            trainer.store(obs, actions, log_probs, values, np.ones(N, np.float32), np.zeros(N, bool), active)
            trainer.maybe_update(obs, active)
        after = [s["exp_avg"] for s in snapshot["optimizer"]["state"].values() if "exp_avg" in s]
        assert all(torch.equal(x, y) for x, y in zip(exp_avg, after))


def _run_cli(tmp_path: Path, *extra: str) -> int:
    import tune_hyperparams

    return tune_hyperparams.run(
        [
            "--preset", "grid",
            "--trial-steps", "48",
            "--report-every", "16",
            "--vehicles", "4",
            "--pedestrians", "0",
            "--init", "fresh",
            "--tuning-dir", str(tmp_path / "tuning"),
            "--best-policy", str(tmp_path / "best_tuned_policy.pt"),
            *extra,
        ]
    )


def test_cli_writes_results_and_resumes(tmp_path: Path) -> None:
    assert _run_cli(tmp_path, "--trials", "2") == 0
    tuning = tmp_path / "tuning"
    params = json.loads((tuning / "best_params.json").read_text(encoding="utf-8"))
    assert tuple(params) == autotune.TUNED_WIRE_KEYS
    assert not SimParams().apply_wire(params).has_problem

    trainer = _trainer()
    assert trainer.load(tmp_path / "best_tuned_policy.pt")
    assert autotune.is_policy_healthy(trainer.policy)

    with open(tuning / "trial_summary.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["state"] for r in rows] == ["COMPLETE", "COMPLETE"]
    assert all(r["outcome"] == "complete" and r["score"] for r in rows)

    # 同じ study 名で回すと続きから（履歴が積み増される）
    assert _run_cli(tmp_path, "--trials", "1") == 0
    with open(tuning / "trial_summary.csv", encoding="utf-8") as f:
        assert [int(r["trial"]) for r in csv.DictReader(f)] == [0, 1, 2]
    assert not (tuning / "best_params.json.tmp").exists()


# ---------------------------------------------------------------------------
# エンジン（学習タブの「学習の自動化」）


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """本番のファイルに触れないエンジン。合成の碁盤の目を読み込み、試行は短くしてある。"""
    from app.map.index import build_map_index
    from app.runtime.engine import SimulationEngine
    import tune_hyperparams

    monkeypatch.setattr(config, "CHECKPOINT_PATH", tmp_path / "shared_policy.pt")
    eng = SimulationEngine()
    eng._autotune = autotune.LiveAutoTune(
        paths=autotune.TunePaths(
            db=tmp_path / "driverl_optuna.db",
            best_params=tmp_path / "best_params.json",
            summary=tmp_path / "trial_summary.csv",
            best_policy=tmp_path / "best_tuned_policy.pt",
            checkpoint=tmp_path / "shared_policy.pt",
        ),
        trial_steps=64,
        report_every=32,
        seed=0,
    )
    eng._params.rollout_length = 32
    eng._params.vehicle_count = 4
    eng._params.pedestrian_count = 0
    eng._ensure_trainer()
    eng._install_map(build_map_index(tune_hyperparams.synthetic_grid()), "grid", "碁盤の目")
    yield eng
    eng._autotune.shutdown()


def _drain(eng) -> list[str]:
    eng._drain_inbox()
    return [n["message"] for n in eng.drain_notices()]


def _run_until(eng, finished: int, limit: int = 2000) -> None:
    for _ in range(limit):
        if eng._autotune.finished >= finished:
            return
        eng._step_once()
        _drain(eng)
    raise AssertionError(f"{limit} ステップで試行が {finished} 件終わらなかった")


def _wait_for_ready(eng, limit_sec: float = 30.0) -> None:
    deadline = time.perf_counter() + limit_sec
    while eng._autotune.phase == "preparing":
        assert time.perf_counter() < deadline, "履歴の DB を開けないまま"
        eng._step_once()
        _drain(eng)


def test_engine_autotune_applies_best_and_backs_up(engine, tmp_path: Path) -> None:
    eng = engine
    trainer = eng._trainer
    baseline = {k: v.clone() for k, v in trainer.policy.state_dict().items()}
    baseline_lr = eng._params.learning_rate

    eng.start_autotune()
    assert any("始めました" in m for m in _drain(eng))
    assert eng.autotune_running()
    assert eng.autotune_payload()["running"] is True
    # 探索を始めた瞬間の重みが本番に書かれる（OFF にしたときの退避がちょうど探索前の重みになる）
    saved = torch.load(tmp_path / "shared_policy.pt", weights_only=True)["policy"]
    assert _state_equal(saved, baseline)

    eng._autosave_every = 1
    _wait_for_ready(eng)
    first = eng.take_params_update()
    _run_until(eng, 3)
    # 探索中は本番の重みを自動保存しない
    assert _state_equal(torch.load(tmp_path / "shared_policy.pt", weights_only=True)["policy"], baseline)

    # 探索している値は外から変えられない（それ以外は通る）
    params, _ = eng.update_params({"learningRate": 9e-4, "simSpeed": 2.0})
    assert params.learning_rate != 9e-4 and params.sim_speed == 2.0
    for kind in ("save_checkpoint", "load_checkpoint", "reset_policy"):
        eng.command(kind)
        assert any("学習の自動化の間" in m for m in _drain(eng)), kind

    wire = eng.autotune_payload()
    assert wire["finishedTrials"] >= 3 and wire["best"] is not None
    # 試行を始めるたびに、その試行の値を params で配る（学習タブのスライダーがこれで動く）
    assert first is not None and first.learning_rate != baseline_lr

    best = eng._autotune.best
    assert best is not None
    eng.stop_autotune()
    notices = _drain(eng)
    assert any(f"試行 #{best.number}" in m and "before-autotune" in m for m in notices), notices
    assert not eng.autotune_running()

    backup = torch.load(tmp_path / "shared_policy.pt.before-autotune", weights_only=True)["policy"]
    assert _state_equal(backup, baseline), "退避は探索を始める前の重み"
    applied = torch.load(tmp_path / "shared_policy.pt", weights_only=True)["policy"]
    assert _state_equal(applied, best.state["policy"])
    assert _state_equal(trainer.policy.state_dict(), best.state["policy"])
    for key, value in best.patch.items():
        assert eng._params.to_wire()[key] == pytest.approx(value)
    # 最良の書き出しは Optuna の専用スレッドが行うので、済むのを待ってから読む
    eng._autotune.shutdown()
    on_disk = json.loads((tmp_path / "best_params.json").read_text(encoding="utf-8"))
    assert on_disk == pytest.approx(best.patch)
    written = torch.load(tmp_path / "best_tuned_policy.pt", weights_only=True)
    assert _state_equal(written["policy"], best.state["policy"])
    assert written["updates"] == best.state["updates"]

    # 止めた後は通常の学習に戻る（保存も通る）
    eng.command("save_checkpoint")
    assert any("保存しました" in m for m in _drain(eng))


def test_engine_stop_without_trials_restores_baseline(engine, tmp_path: Path) -> None:
    eng = engine
    trainer = eng._trainer
    baseline = {k: v.clone() for k, v in trainer.policy.state_dict().items()}
    before = replace(eng._params)

    eng.start_autotune()
    _drain(eng)
    _wait_for_ready(eng)
    for _ in range(10):
        eng._step_once()
        _drain(eng)
    eng.stop_autotune()
    assert any("戻しました" in m for m in _drain(eng))
    assert _state_equal(trainer.policy.state_dict(), baseline)
    assert autotune.tuned_patch(eng._params) == autotune.tuned_patch(before)
    assert not (tmp_path / "shared_policy.pt.before-autotune").exists()


def test_engine_marks_diverged_trial_and_restores_next(engine) -> None:
    eng = engine
    trainer = eng._trainer
    eng.start_autotune()
    _drain(eng)
    _wait_for_ready(eng)
    while eng._autotune.phase != "running":
        eng._step_once()
        _drain(eng)
    with torch.no_grad():
        next(trainer.policy.parameters()).view(-1)[0] = float("inf")
    eng._step_once()
    _drain(eng)
    assert eng._autotune.history[-1].outcome == "diverged"
    assert eng._autotune.history[-1].score == autotune.SCORE_DIVERGED
    _run_until(eng, 2)
    assert autotune.is_policy_healthy(trainer.policy), "次の試行は探索前の重みから始まる"
    eng.stop_autotune()
    _drain(eng)


def test_engine_refuses_conflicting_actions(engine) -> None:
    eng = engine
    eng.start_autotune()
    _drain(eng)
    eng.set_practical_mode(True)
    assert any("実用モードに切り替えられません" in m for m in _drain(eng))
    assert not eng.practical_mode()
    assert "学習の自動化の間" in eng.detector_job.start(_detector_request())
    eng.stop_autotune()
    _drain(eng)

    eng.set_practical_mode(True)
    _drain(eng)
    eng.start_autotune()
    assert any("実用モードの間" in m for m in _drain(eng))
    assert not eng.autotune_running()


def _detector_request():
    from app.runtime.detector_job import parse_request

    request, why = parse_request({"mode": "full"})
    assert request is not None, why
    return request


def test_engine_reports_missing_optuna(engine) -> None:
    engine._autotune.available = False
    engine.start_autotune()
    assert any("Optuna が入っていない" in m for m in _drain(engine))
    assert engine.autotune_payload()["available"] is False
    assert not engine.autotune_running()
