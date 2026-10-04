"""学習タブの設定値の控え（`runtime/param_store.py`）と、エンジンでの保存・復元の契約テスト。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.contracts import SimParams
from app.runtime.autotune import TUNED_WIRE_KEYS
from app.runtime.engine import SimulationEngine
from app.runtime.param_store import PERSISTED_WIRE_KEYS, ParamStore


def test_persisted_keys_are_valid_params() -> None:
    wire = SimParams().to_wire()
    assert set(PERSISTED_WIRE_KEYS) <= set(wire)
    # 探索の対象はすべて学習タブの値なので、探索の結果を控えられる
    assert set(TUNED_WIRE_KEYS) <= set(PERSISTED_WIRE_KEYS)
    # 台数・天候・倍速は学習タブの値ではないので控えない
    for key in ("vehicleCount", "pedestrianCount", "simSpeed", "weatherRain", "weatherFog", "weatherAuto"):
        assert key not in PERSISTED_WIRE_KEYS


def test_save_merges_and_ignores_other_keys(tmp_path: Path) -> None:
    store = ParamStore(tmp_path / "p.json")
    assert store.load() == {}
    assert store.save({"learningRate": 1e-4, "vehicleCount": 8}) is True
    assert store.save({"gamma": 0.95}) is True
    assert store.load() == {"learningRate": 1e-4, "gamma": 0.95}
    assert store.save({"simSpeed": 2.0}) is True
    assert json.loads((tmp_path / "p.json").read_text(encoding="utf-8")) == {"learningRate": 1e-4, "gamma": 0.95}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("text", ["", "{", "[1, 2]", "null"])
def test_broken_file_falls_back_to_defaults(tmp_path: Path, text: str) -> None:
    path = tmp_path / "p.json"
    path.write_text(text, encoding="utf-8")
    store = ParamStore(path)
    assert store.load() == {}
    params = SimParams()
    assert store.restore_into(params) == []
    assert params == SimParams()
    # 壊れていても次の保存で直る
    assert store.save({"gamma": 0.9}) is True
    assert store.load() == {"gamma": 0.9}


def test_restore_clamps_and_skips_garbage(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(
        json.dumps({"learningRate": 5.0, "gamma": "abc", "onlineAssist": False, "unknown": 1, "simSpeed": 4.0}),
        encoding="utf-8",
    )
    params = SimParams()
    ParamStore(path).restore_into(params)
    assert params.learning_rate == pytest.approx(1e-2)
    assert params.gamma == SimParams().gamma
    assert params.online_assist is False
    assert params.sim_speed == SimParams().sim_speed


def test_save_failure_is_reported_not_raised(tmp_path: Path) -> None:
    # 親がファイルだと書けない
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    assert ParamStore(blocker / "p.json").save({"gamma": 0.9}) is False


def test_engine_restores_and_saves_changes(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    ParamStore(path).save({"learningRate": 1e-4, "rewardGoal": 150.0, "v2xComm": False})

    eng = SimulationEngine(param_store=ParamStore(path))
    params = eng.snapshot_params()
    assert params.learning_rate == pytest.approx(1e-4)
    assert params.reward_goal == 150.0
    assert params.v2x_comm is False

    eng.update_params({"gamma": 0.95, "simSpeed": 2.0, "learningRate": 5.0})
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["gamma"] == 0.95
    # 丸められた値が控えに残る（範囲外の 5.0 ではない）
    assert saved["learningRate"] == pytest.approx(1e-2)
    assert saved["rewardGoal"] == 150.0
    assert "simSpeed" not in saved

    # 次の起動で同じ値から始まる
    again = SimulationEngine(param_store=ParamStore(path)).snapshot_params()
    assert again.gamma == 0.95 and again.reward_goal == 150.0


def test_engine_without_store_does_not_touch_disk(tmp_path: Path) -> None:
    eng = SimulationEngine()
    eng.update_params({"gamma": 0.95})
    assert eng.snapshot_params().gamma == 0.95
    assert not list(tmp_path.iterdir())


def test_trial_values_are_not_persisted_until_autotune_finishes(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    store = ParamStore(path)
    store.save({"learningRate": 3e-4, "onlineAssist": True})
    eng = SimulationEngine(param_store=store)

    # 探索中: 探索の対象は断られて控えに入らず、それ以外（スイッチ）は通って控えに入る
    eng._autotune_running = True
    eng.update_params({"learningRate": 9e-4, "onlineAssist": False})
    assert store.load() == {"learningRate": 3e-4, "onlineAssist": False}

    # 試行が値を書き換えても（`_tune_restore` 相当）、終わるまでは控えない
    eng._params.learning_rate = 7e-4
    eng._params.reward_goal = 200.0
    assert store.load()["learningRate"] == 3e-4

    # 終了（最良の試行の値を適用した後）で、探索の対象も含めて確定した値が控えに入る
    eng._finish_autotune()
    saved = store.load()
    assert saved["learningRate"] == pytest.approx(7e-4)
    assert saved["rewardGoal"] == 200.0
    assert saved["onlineAssist"] is False
    assert set(saved) == set(PERSISTED_WIRE_KEYS)
