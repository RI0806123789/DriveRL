"""信号プランの既定互換性・時間帯・需要配分と現示の安全性。"""

from __future__ import annotations

import numpy as np
import pytest

from app.contracts import MapSignal
from app.sim.signal_plan import SignalPeriod, SignalPlan
from app.sim.signals import GREEN, RED, YELLOW, SignalController, _scatter01


def _signals(groups: int = 3, copies: int = 2, key: int = 7) -> list[MapSignal]:
    return [
        MapSignal(i, i, 0.0, 0.0, float(group), group, 6.0, key, i)
        for i, group in enumerate(group for group in range(groups) for _ in range(copies))
    ]


def test_default_matches_previous_formula_exactly() -> None:
    signals = _signals(4) + _signals(2, key=11) + _signals(1, key=99)
    controller = SignalController(signals)
    counts = np.array([max(2, max(s.group for s in signals if s.phase_key == head.phase_key) + 1) for head in signals])
    green = np.maximum(6.0, 60.0 / counts - 5.0)
    cycle = (green + 5.0) * counts
    offsets = _scatter01(np.array([head.phase_key for head in signals])) * cycle
    shift = (green + 5.0) * np.array([head.group for head in signals])
    for time in [-100.0, 0.0, 0.05, 30.0, 60.0, 999999.0]:
        local = (time + offsets + shift) % cycle
        expected = np.where(local < green, GREEN, np.where(local < green + 3.0, YELLOW, RED))
        assert controller.phases(time) == expected.tolist()


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 7])
def test_adaptive_clearances_and_minimum_green(groups: int) -> None:
    controller = SignalController(_signals(groups), plan=SignalPlan(mode="adaptive"))
    demand = np.zeros(groups * 2)
    demand[0] = 100.0
    controller.phases(0.0, demand)
    cycle = controller._key_cycles[0]
    offset = controller._key_offsets[0]
    controller.phases(cycle - offset, demand)
    green_seen = np.zeros(groups)
    previous = None
    last_nonred = None
    red_duration = 0.0
    for position in np.arange(0.0, cycle, 0.05):
        phases = np.array(controller.phases(cycle - offset + position, demand))
        assert np.array_equal(phases[::2], phases[1::2])
        active = np.flatnonzero(phases[::2] != RED)
        assert active.size <= 1
        green_seen += (phases[::2] == GREEN) * 0.05
        if active.size:
            group = int(active[0])
            if last_nonred is not None and last_nonred != group:
                assert red_duration >= 1.95
            if previous is not None and previous[group] == GREEN:
                assert phases[group * 2] in (GREEN, YELLOW)
            last_nonred = group
            red_duration = 0.0
        else:
            red_duration += 0.05
        previous = phases[::2]
    assert np.all(green_seen >= 5.95)
    assert np.all(controller._slot_green >= 6.0)
    assert green_seen[0] >= green_seen[-1]


def test_demand_changes_only_at_cycle_boundary() -> None:
    controller = SignalController(_signals(2), plan=SignalPlan(mode="adaptive"))
    first = np.array([10.0, 0.0, 0.0, 0.0])
    second = np.array([0.0, 0.0, 10.0, 0.0])
    controller.phases(0.0, first)
    original = controller._slot_green.copy()
    controller.phases(0.1, second)
    assert np.array_equal(original, controller._slot_green)
    boundary = controller._key_cycles[0] - controller._key_offsets[0]
    controller.phases(boundary + 0.01, second)
    assert controller._slot_green[1] > controller._slot_green[0]


def test_exact_yellow_and_all_red_clearances() -> None:
    controller = SignalController(_signals(3, copies=1), plan=SignalPlan(group_weights=(4.0, 1.0, 2.0)))
    controller.phases(0.0)
    boundary = controller._key_cycles[0] - controller._key_offsets[0]
    controller.phases(boundary + 0.01)
    for group in range(3):
        start = controller._slot_start[group]
        green = controller._slot_green[group]
        assert controller.phases(boundary + start + green - 0.001)[group] == GREEN
        assert controller.phases(boundary + start + green + 0.001)[group] == YELLOW
        assert controller.phases(boundary + start + green + 2.999)[group] == YELLOW
        assert set(controller.phases(boundary + start + green + 3.001)) == {RED}
        assert set(controller.phases(boundary + start + green + 4.999)) == {RED}


def test_time_of_day_uses_cycle_start_and_wraps_midnight() -> None:
    plan = SignalPlan(start_hour=8.0, time_of_day=(SignalPeriod(8.0, (4.0, 1.0)), SignalPeriod(18.0, (1.0, 4.0))))
    controller = SignalController(_signals(2), plan=plan)
    controller.phases(0.0)
    assert controller._slot_green[0] < controller._slot_green[1]
    boundary = controller._key_cycles[0] - controller._key_offsets[0]
    controller.phases(boundary + 0.01)
    assert controller._slot_green[0] > controller._slot_green[1]
    controller.phases(16.0 * 3600.0)
    assert controller._slot_green[0] < controller._slot_green[1]


def test_forward_jump_and_reset_match_fresh_fixed_controller() -> None:
    signals = _signals(3) + _signals(4, key=19)
    plan = SignalPlan(time_of_day=(SignalPeriod(0.0, (1.0, 2.0)), SignalPeriod(7.0, (4.0, 1.0))))
    controller = SignalController(signals, plan=plan)
    controller.phases(0.0)
    for time in [0.5, 3600.0 * 8, 1e9, 0.0, 5.0]:
        fresh = SignalController(signals, plan=plan)
        assert controller.phases(time) == fresh.phases(time)
        assert np.allclose(controller._slot_green, fresh._slot_green)


def test_zero_weights_fall_back_to_uniform_and_missing_groups_use_one() -> None:
    controller = SignalController(_signals(3), plan=SignalPlan(group_weights=(0.0, 0.0, 0.0)))
    controller.phases(0.0)
    assert np.allclose(controller._slot_green, [15.0, 15.0, 15.0])
    controller = SignalController(_signals(3), plan=SignalPlan(group_weights=(4.0,)))
    controller.phases(0.0)
    assert controller._slot_green[0] > controller._slot_green[1]
    assert controller._slot_green[1] == controller._slot_green[2]


@pytest.mark.parametrize("kwargs", [
    {"mode": "unknown"}, {"green_sec": float("inf")}, {"yellow_sec": 0.0},
    {"all_red_sec": -1.0}, {"green_min_sec": 26.0}, {"start_hour": 24.0},
    {"demand_gain": -1.0}, {"demand_cap": float("nan")}, {"group_weights": (-1.0,)},
    {"time_of_day": (SignalPeriod(8.0), SignalPeriod(8.0))},
    {"time_of_day": (SignalPeriod(18.0), SignalPeriod(8.0))},
])
def test_invalid_plan_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SignalPlan(**kwargs)


@pytest.mark.parametrize("demand", [np.zeros(1), np.array([float("nan")] * 6), np.full(6, -1.0)])
def test_invalid_demand_is_rejected(demand: np.ndarray) -> None:
    controller = SignalController(_signals(), plan=SignalPlan(mode="adaptive"))
    with pytest.raises(ValueError):
        controller.phases(0.0, demand)


def test_empty_map() -> None:
    assert SignalController([], plan=SignalPlan(mode="adaptive")).phases(1e9) == []


def test_large_finite_weights_and_demands_preserve_green_budget() -> None:
    plan = SignalPlan(mode="adaptive", group_weights=(1e308, 1e307), demand_cap=1e308, demand_gain=0.5)
    controller = SignalController(_signals(2), plan=plan)
    controller.phases(0.0, np.full(4, 1e308))
    assert np.isfinite(controller._slot_green).all()
    assert np.all(controller._slot_green >= 6.0)
    assert controller._slot_green.sum() == pytest.approx(50.0)
