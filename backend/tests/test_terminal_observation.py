"""打ち切り（タイムアウト）の最終観測 final_obs が、打ち切った瞬間の前方カメラから作られているか（#127）。"""
from __future__ import annotations

import numpy as np
import pytest

from app import config
from app.contracts import MapSignal, SimParams
from app.map.index import build_map_index
from app.percep.encoder import OBS_OFFSETS
from app.sim.env import SimulationEnv
from tune_hyperparams import synthetic_grid

N = config.MAX_VEHICLES
SIZES = dict(config.OBS_LAYOUT)
#: 前方カメラだけから作る欄（周囲カメラは意図して非同期に撮り直すので、終端でも直近の結果のまま）
FRONT_SECTIONS = ("lane", "signal", "sign", "vehicles", "obstacles", "pedestrians", "freespace", "traffic_signs")
#: 死角のうち前方カメラだけで決まる欄（前方の遮蔽率）
FRONT_OCCLUSION = OBS_OFFSETS["occlusion"] + 2
SIG = OBS_OFFSETS["signal"]
#: 場面を探すステップ数（seed 3 では色の変化が 93、停止線の通過が 242 ステップ目）
SEARCH_STEPS = 320


def signal_grid():
    """碁盤の目の、3 本以上の道が交わる交差点のすべての進入路に信号を立てる（ローダーと同じく停止線の手前・進入の向き）。"""
    data = synthetic_grid()
    nodes = {n.id: n for n in data.nodes}
    degree: dict[int, int] = {}
    for e in data.edges:
        degree[e.v] = degree.get(e.v, 0) + 1
    signals = []
    for e in data.edges:
        if degree[e.v] < 3:
            continue
        u, v = nodes[e.u], nodes[e.v]
        dx, dy = v.x - u.x, v.y - u.y
        length = float(np.hypot(dx, dy))
        setback = 10.0
        signals.append(MapSignal(
            id=len(signals), node_id=e.v, x=v.x - dx / length * setback, y=v.y - dy / length * setback,
            heading=float(np.arctan2(dy, dx)), group=0 if abs(dx) > abs(dy) else 1, road_width=e.width,
            phase_key=e.v, edge_id=e.id, source="synthetic",
        ))
    data.signals = signals
    return build_map_index(data)


@pytest.fixture(scope="module")
def index():
    return signal_grid()


def make_env(index) -> SimulationEnv:
    env = SimulationEnv(index, SimParams(vehicle_count=4, pedestrian_count=0), seed=3, perception_mode="oracle")
    env.autopilot_all = True
    env.reset_all()
    return env


def run(index, steps: int):
    """打ち切らない環境を回し、各ステップの観測を残す（同じ乱数種なら打ち切る環境と同じ状態列になる）。"""
    env = make_env(index)
    zeros = np.zeros((N, config.ACTION_DIM), dtype=np.float32)
    history = [env.observations.copy()]
    for _ in range(steps):
        history.append(env.step(zeros).obs.copy())
    return history


def final_obs_at(index, steps: int, monkeypatch):
    """`steps` ステップ目で全車を打ち切り、そのステップの final_obs と打ち切った車を返す。"""
    monkeypatch.setattr(config, "MAX_EPISODE_STEPS", steps)
    env = make_env(index)
    zeros = np.zeros((N, config.ACTION_DIM), dtype=np.float32)
    for _ in range(steps - 1):
        assert not env.step(zeros).truncated.any()
    result = env.step(zeros)
    return result.final_obs, np.flatnonzero(result.truncated)


def front_columns(row: np.ndarray) -> np.ndarray:
    parts = [row[OBS_OFFSETS[name] : OBS_OFFSETS[name] + SIZES[name]] for name in FRONT_SECTIONS]
    return np.concatenate([*parts, row[FRONT_OCCLUSION : FRONT_OCCLUSION + 1]])


def find_moments(history):
    """信号の色が変わったステップと、停止線を越えて信号が見えなくなったステップを 1 つずつ探す（(t, slot)）。"""
    color = passed = None
    for t in range(2, len(history)):
        before, now = history[t - 1], history[t]
        for s in range(N):
            seen_before, seen_now = before[s, SIG + 4] > 0, now[s, SIG + 4] > 0
            if color is None and seen_before and seen_now and np.argmax(before[s, SIG + 1 : SIG + 4]) != np.argmax(now[s, SIG + 1 : SIG + 4]):
                color = (t, s)
            if passed is None and seen_before and not seen_now and before[s, SIG] < 0.3:
                passed = (t, s)
    return color, passed


@pytest.fixture(scope="module")
def history(index):
    return run(index, SEARCH_STEPS)


@pytest.mark.parametrize("moment", ["color", "passed"])
def test_final_obs_matches_the_terminal_front_camera(index, history, monkeypatch, moment) -> None:
    color, passed = find_moments(history)
    found = {"color": color, "passed": passed}[moment]
    assert found is not None, f"{SEARCH_STEPS} ステップで{moment}の場面が見つからない（合成の地図か運転を見直す）"
    t, slot = found
    final_obs, truncated = final_obs_at(index, t, monkeypatch)
    assert slot in truncated
    expected = history[t]  # 打ち切らなかった環境の同じステップの観測（終端の状態で撮ったもの）
    for s in truncated:
        np.testing.assert_allclose(front_columns(final_obs[s]), front_columns(expected[s]), atol=1e-6, err_msg=f"slot {s}")
    # 打ち切った瞬間に起きたこと（信号の色の変化・停止線の通過）が final_obs に入っている
    assert not np.allclose(front_columns(history[t - 1][slot]), front_columns(final_obs[slot]))


def test_previous_behaviour_mixed_the_stale_front_camera(index, history, monkeypatch) -> None:
    """以前の作り方（直近の検出のまま観測を作り直す）だと、打ち切った瞬間の変化が入らない（この検査が効いていることの確認）。"""
    color, _passed = find_moments(history)
    assert color is not None
    t, slot = color
    monkeypatch.setattr(SimulationEnv, "_encode_terminal_observations", lambda self, slots: self._encode_last_perception())
    final_obs, _ = final_obs_at(index, t, monkeypatch)
    stale = final_obs[slot, SIG : SIG + 5]
    assert not np.allclose(stale, history[t][slot, SIG : SIG + 5])
    np.testing.assert_allclose(stale, history[t - 1][slot, SIG : SIG + 5], atol=1e-6)


def test_no_extra_perception_without_truncation(index, monkeypatch) -> None:
    """打ち切りの無いステップでは前方カメラを撮り直さない（毎ステップの推論を増やさない）。"""
    calls = []
    original = SimulationEnv._terminal_front
    monkeypatch.setattr(SimulationEnv, "_terminal_front", lambda self, slots: calls.append(list(slots)) or original(self, slots))
    monkeypatch.setattr(config, "MAX_EPISODE_STEPS", 30)
    env = make_env(index)
    zeros = np.zeros((N, config.ACTION_DIM), dtype=np.float32)
    truncated_steps = 0
    for _ in range(45):
        r = env.step(zeros)
        truncated_steps += int(r.truncated.any())
    assert truncated_steps >= 1
    assert len(calls) == truncated_steps
    assert all(len(c) >= 1 for c in calls)
