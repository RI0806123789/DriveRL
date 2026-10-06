"""経路上の信号を前後に数える `World.nth_signals` / `World.next_signal` の契約テスト（#88）。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app import config
from app.sim.signals import GREEN, RED, YELLOW
from app.sim.world import SlotState, World

ARCS = (10.0, 40.0, 90.0)


def _world(arcs: dict[int, float]) -> SimpleNamespace:
    """指定したスロットだけ 3 基の信号を持つ経路に置いた、nth_signals が読む欄だけの World。"""
    slots = [SlotState() for _ in range(config.MAX_VEHICLES)]
    arc = np.zeros(config.MAX_VEHICLES, dtype=np.float32)
    for slot, position in arcs.items():
        slots[slot].signal_arcs = np.array(ARCS, dtype=np.float32)
        slots[slot].signal_ids = np.array([0, 1, 2], dtype=np.int32)
        arc[slot] = position
    # 経路の末尾の信号だけ青にして、末尾を引いたら灯色でも分かるようにする
    phases = np.array([RED, YELLOW, GREEN], dtype=np.int8)
    return SimpleNamespace(slots=slots, arc=arc, signal_phases=phases)


@pytest.mark.parametrize("offset", [-1, -2, -3, -100])
def test_negative_offset_before_first_signal_is_none(offset: int) -> None:
    world = _world({0: 5.0})
    distance, phase = World.nth_signals(world, offset)
    assert distance[0] == np.inf
    assert phase[0] == RED


def test_negative_offset_counts_passed_signals() -> None:
    world = _world({1: 50.0})
    expected = {
        1: (np.inf, RED),
        0: (40.0, GREEN),
        -1: (-10.0, YELLOW),
        -2: (-40.0, RED),
        -3: (np.inf, RED),
    }
    for offset, (want_distance, want_phase) in expected.items():
        distance, phase = World.nth_signals(world, offset)
        assert distance[1] == pytest.approx(want_distance), offset
        assert phase[1] == want_phase, offset


@pytest.mark.parametrize("offset", [-1, 0, 1, 3])
def test_route_without_signals(offset: int) -> None:
    world = _world({})
    distance, phase = World.nth_signals(world, offset)
    assert np.all(distance == np.inf)
    assert np.all(phase == RED)


def test_offset_zero_matches_next_signal() -> None:
    world = _world({0: 5.0, 1: 50.0, 2: 90.0, 3: 120.0})
    distance, phase = World.nth_signals(world, 0)
    for slot in range(config.MAX_VEHICLES):
        want_distance, want_phase = World.next_signal(world, slot)
        assert distance[slot] == pytest.approx(want_distance), slot
        assert phase[slot] == want_phase, slot
