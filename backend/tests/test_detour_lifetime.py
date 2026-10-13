"""迂回探索の静的入力と、取消・完了・失敗時の世界および占有グリッドの解放を検査する。"""

from __future__ import annotations

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from app.runtime.engine import SimulationEngine
from app import warn as warn_module
from app.sim import road_closure
from app.sim.road_closure import RoadClosureSupervisor
from .test_road_closure import ControlledSearch, close_ahead, world_on_route


def test_search_captures_closed_edges_pose_goal_and_readonly_nodes(monkeypatch) -> None:
    executor = ControlledSearch()
    monkeypatch.setattr(road_closure, "_SEARCH", executor)
    world = world_on_route()
    close_ahead(world)
    supervisor = RoadClosureSupervisor()
    supervisor.evaluate(world, frozenset((0,)))
    inputs = supervisor._pending[0].inputs
    assert inputs.map_index is not world.map_index
    assert inputs.map_index.occupancy is world.map_index.occupancy
    assert inputs.closed == frozenset((1,))
    assert inputs.nodes is not None
    for frozen, source in zip(inputs.nodes, (world._node_xy, world._node_ids, world._node_z)):
        assert np.shares_memory(frozen, source)
        assert not frozen.flags.writeable

    monkeypatch.setattr(world, "route_between", lambda *args, **kwargs: pytest.fail("探索が World を参照しました"))
    world.fleet.x[0] = 22.0
    world.slots[0].goal = (100.0, 100.0)
    world.map_index.set_closed_edges(frozenset((1, 2)))
    route = executor.finish()
    assert route is not None
    assert route.points[0, 0] == pytest.approx(20.0)
    assert route.points[-1, 0] == pytest.approx(200.0)
    assert any(leg.edge_id == 2 for leg in route.legs)
    assert inputs.map_index is None and inputs.nodes is None
    serial = int(world.route_serial[0])
    supervisor.evaluate(world, frozenset((0,)))
    assert int(world.route_serial[0]) == serial
    supervisor.close()


@pytest.mark.parametrize("reason", ["close", "map", "collected"])
def test_queued_cancellation_releases_old_world_and_grid_before_worker_is_available(reason, monkeypatch) -> None:
    started, release = Event(), Event()

    def block() -> None:
        started.set()
        assert release.wait(10)

    with ThreadPoolExecutor(max_workers=1) as executor:
        blocker = executor.submit(block)
        assert started.wait(5)
        monkeypatch.setattr(road_closure, "_SEARCH", executor)
        try:
            world = world_on_route()
            close_ahead(world)
            world_ref = weakref.ref(world)
            index_ref = weakref.ref(world.map_index)
            grid_ref = weakref.ref(world.map_index.occupancy)
            array_ref = weakref.ref(world.map_index.occupancy.building)
            supervisor = RoadClosureSupervisor()
            supervisor_ref = weakref.ref(supervisor)
            supervisor.evaluate(world, frozenset((0,)))
            future = supervisor._pending[0].future
            inputs = supervisor._pending[0].inputs
            if reason == "close":
                supervisor.close()
            elif reason == "map":
                new_world = world_on_route()
                supervisor.evaluate(new_world, frozenset((0,)))
            else:
                del supervisor
            del world
            gc.collect()
            assert future.cancelled()
            assert inputs.map_index is None and inputs.nodes is None
            assert world_ref() is None and index_ref() is None
            assert grid_ref() is None and array_ref() is None
            assert not blocker.done()
            if reason == "collected":
                assert supervisor_ref() is None
        finally:
            release.set()
        blocker.result(timeout=5)


@pytest.mark.parametrize("failure", ["none", "search", "lead"])
def test_running_search_keeps_only_static_grid_until_success_or_failure(failure, monkeypatch, caplog) -> None:
    started, release = Event(), Event()
    detour = road_closure._detour
    monkeypatch.setattr(warn_module, "_WARNED", set())

    def delayed(map_index, *args, **kwargs):
        started.set()
        assert release.wait(10)
        if failure == "search":
            raise RuntimeError("検索の失敗を再現")
        return detour(map_index, *args, **kwargs)

    monkeypatch.setattr(road_closure, "_detour", delayed)
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(road_closure, "_SEARCH", executor)
        try:
            world = world_on_route()
            close_ahead(world)
            if failure == "lead":
                def fail_lead(self, *args, **kwargs):
                    raise RuntimeError("出発道路の検出失敗を再現")

                monkeypatch.setattr(type(world.map_index), "road_lead", fail_lead)
            world_ref = weakref.ref(world)
            index_ref = weakref.ref(world.map_index)
            grid_ref = weakref.ref(world.map_index.occupancy)
            array_ref = weakref.ref(world.map_index.occupancy.building)
            supervisor = RoadClosureSupervisor()
            supervisor.evaluate(world, frozenset((0,)))
            future = supervisor._pending[0].future
            assert started.wait(5)
            supervisor.close()
            del world
            gc.collect()
            assert world_ref() is None and index_ref() is None
            assert grid_ref() is not None and array_ref() is not None
            assert not future.cancelled()
        finally:
            release.set()
        route = future.result(timeout=5)
        executor.submit(lambda: None).result(timeout=5)
        gc.collect()
        assert future.exception() is None
        assert (route is None) == (failure != "none")
        assert grid_ref() is None and array_ref() is None
        records = [record for record in caplog.records if record.name == "autoware_sim"]
        assert len(records) == (0 if failure == "none" else 1)
        assert all(not record.exc_info for record in records)


@pytest.mark.parametrize("fail", [False, True])
def test_engine_exit_closes_detour_requests(fail, monkeypatch) -> None:
    engine = SimulationEngine()
    supervisor = RoadClosureSupervisor()
    world = world_on_route()
    close_ahead(world)
    executor = ControlledSearch()
    monkeypatch.setattr(road_closure, "_SEARCH", executor)
    supervisor.evaluate(world, frozenset((0,)))
    inputs = supervisor._pending[0].inputs
    engine._env = SimpleNamespace(road_closures=supervisor)

    def run_loop() -> None:
        if fail:
            raise RuntimeError("エンジン停止を再現")

    monkeypatch.setattr(engine, "_run_loop", run_loop)
    engine._run()
    assert not supervisor._pending
    assert supervisor._map_index is None
    assert inputs.map_index is None and inputs.nodes is None


def test_map_install_clears_detour_input_before_constructing_new_environment(monkeypatch) -> None:
    from app.sim import env as env_module

    engine = SimulationEngine()
    world = world_on_route()
    close_ahead(world)
    supervisor = RoadClosureSupervisor()
    executor = ControlledSearch()
    monkeypatch.setattr(road_closure, "_SEARCH", executor)
    supervisor.evaluate(world, frozenset((0,)))
    inputs = supervisor._pending[0].inputs
    engine._env = SimpleNamespace(road_closures=supervisor)
    monkeypatch.setattr(engine, "_abort_autotune", lambda *args: None)
    monkeypatch.setattr(engine._taxi, "cancel", lambda *args: None)
    monkeypatch.setattr(engine, "_publish_taxi", lambda: None)

    def construct(*args, **kwargs):
        assert not supervisor._pending and supervisor._map_index is None
        assert inputs.map_index is None and inputs.nodes is None
        raise RuntimeError("環境構築の前までを検査")

    monkeypatch.setattr(env_module, "SimulationEnv", construct)
    with pytest.raises(RuntimeError, match="環境構築の前までを検査"):
        engine._install_map(world.map_index, "detour", "合成道路")
