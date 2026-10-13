"""既知の全幅閉鎖に対する経路探索を、物理ステップの外で進める。"""

from __future__ import annotations

import math
import weakref
from copy import copy
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

from app import config
from app.contracts import LaneChangeDecision, MapIndex
from app.sim.routes import NodeArrays, Route, RoutePlanner
from app.sim.signals import stop_speed_limit
from app.warn import warn_once

if TYPE_CHECKING:
    from app.sim.world import World

_SEARCH = ThreadPoolExecutor(max_workers=1, thread_name_prefix="driverl-detour")


@dataclass
class _Input:
    map_index: MapIndex | None
    start: tuple[float, float]
    goal: tuple[float, float]
    heading: float
    closed: frozenset[int]
    start_z: float | None
    goal_z: float | None
    nodes: NodeArrays | None
    _lock: Lock = field(default_factory=Lock, repr=False)

    def take(self) -> tuple[MapIndex | None, NodeArrays | None]:
        with self._lock:
            map_index, self.map_index = self.map_index, None
            nodes, self.nodes = self.nodes, None
            return map_index, nodes

    def clear(self) -> None:
        with self._lock:
            self.map_index = None
            self.nodes = None


@dataclass
class _Search:
    serial: int
    generation: int
    future: Future
    from_xy: tuple[float, float]
    map_index: object
    inputs: _Input

    def cancel(self) -> None:
        self.inputs.clear()
        self.future.cancel()


def _cancel_pending(pending: dict[int, _Search]) -> None:
    for job in pending.values():
        job.cancel()
    pending.clear()


def _run_search(inputs: _Input) -> Route | None:
    map_index, nodes = inputs.take()
    if map_index is None:
        return None
    try:
        return _detour(map_index, inputs.start, inputs.goal, inputs.heading, inputs.closed, inputs.start_z, inputs.goal_z, nodes)
    except Exception:
        pass
    warn_once("sim.road_closure.search", "閉鎖道路の迂回探索に失敗しました。閉鎖の手前で待機します")
    return None


def _detour(
    map_index: MapIndex, start: tuple[float, float], goal: tuple[float, float], heading: float, closed: frozenset[int],
    start_z: float | None = None, goal_z: float | None = None,
    nodes: NodeArrays | None = None,
) -> Route | None:
    """閉鎖辺と車体幅での建物重なりを含む候補を検索側で拒否する。立体の地図では出発・目的地の階を保つ。"""
    route = RoutePlanner(map_index, nodes=nodes).route_between(start, goal, heading=heading, speed=0.0, src_z=start_z, dst_z=goal_z)
    if route is None or len(route.points) < 2 or any(leg.edge_id in closed for leg in route.legs):
        return None
    cum = RoutePlanner._cumulative_length(route.points)
    length = float(cum[-1])
    if not math.isfinite(length) or length <= 1e-9:
        return None
    marks = np.arange(0.0, length + 0.25, 0.5)
    if len(marks) < 2:
        marks = np.array([0.0, length])
    points = np.column_stack((np.interp(marks, cum, route.points[:, 0]), np.interp(marks, cum, route.points[:, 1])))
    tangent = np.gradient(points, axis=0)
    norm = np.maximum(np.linalg.norm(tangent, axis=1), 1e-9)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) / norm[:, None]
    for side in (-config.VEHICLE_WIDTH / 2.0 - 0.3, 0.0, config.VEHICLE_WIDTH / 2.0 + 0.3):
        probe = points + normal * side
        if np.any(map_index.occupancy.sample_building(probe[:, 0], probe[:, 1])):
            return None
    return route


class RoadClosureSupervisor:
    """結果が古くなった再探索は捨て、閉鎖解除時に待機を残さない。"""

    def __init__(self) -> None:
        self.decisions: dict[int, LaneChangeDecision] = {}
        self.caps: dict[int, float] = {}
        self._pending: dict[int, _Search] = {}
        self._failed: set[tuple[int, int, int]] = set()
        self._map_index: object | None = None
        self._finalizer = weakref.finalize(self, _cancel_pending, self._pending)

    def close(self) -> None:
        """待機中の入力と旧地図への参照を解放し、実行中の結果を取り込まない。"""
        _cancel_pending(self._pending)
        self._map_index = None
        self._failed.clear()
        self.decisions.clear()
        self.caps.clear()

    def evaluate(self, world: World, enabled: frozenset[int]) -> tuple[int, ...]:
        """完了した検索だけ取り込み、1ステップで新しく依頼する検索は1台までにする。"""
        self.decisions, self.caps = {}, {}
        installed = []
        if self._map_index is not world.map_index:
            _cancel_pending(self._pending)
            self._failed.clear()
            self._map_index = world.map_index
        closed = frozenset(getattr(world.map_index, "closed_edges", frozenset()))
        generation = int(getattr(world.map_index, "closure_generation", 0))
        if not closed:
            _cancel_pending(self._pending)
            self._failed.clear()
            return ()
        submitted = False
        self._failed = {key for key in self._failed if key[2] == generation and world.fleet.active[key[0]] and int(world.route_serial[key[0]]) == key[1]}
        for slot in list(self._pending):
            job = self._pending[slot]
            if slot not in enabled or not world.fleet.active[slot] or job.map_index is not world.map_index or job.generation != generation or job.serial != int(world.route_serial[slot]):
                job.cancel()
                self._pending.pop(slot)
        for slot in sorted(enabled):
            if not world.fleet.active[slot]:
                continue
            arc = float(world.arc[slot])
            leg = next((leg for leg in world.slots[slot].route_legs if leg.edge_id in closed and leg.end_arc > arc - config.VEHICLE_LENGTH / 2.0), None)
            if leg is None:
                continue
            key = (slot, int(world.route_serial[slot]), generation)
            job = self._pending.get(slot)
            if job is not None and job.future.done():
                self._pending.pop(slot)
                try:
                    route = job.future.result()
                except Exception:
                    warn_once("sim.road_closure.search", "閉鎖道路の迂回探索に失敗しました。閉鎖の手前で待機します")
                    route = None
                drift = math.hypot(float(world.fleet.x[slot]) - job.from_xy[0], float(world.fleet.y[slot]) - job.from_xy[1])
                if route is not None and drift <= 1.0 and not any(l.edge_id in closed for l in route.legs) and world.revise_route(slot, route):
                    installed.append(slot)
                    self.decisions[slot] = LaneChangeDecision(state="rerouting", source="road_event", imposed=True)
                    continue
                if drift > 1.0:
                    job = None
                else:
                    self._failed.add(key)
            self.caps[slot] = float(stop_speed_limit(np.float64(leg.start_arc - arc), abs(config.MAX_DECEL), config.DT, margin_m=config.VEHICLE_LENGTH / 2.0 + 2.0))
            if slot in self._pending:
                self.caps[slot] = 0.0
            reason = "no_detour" if key in self._failed else "blocked"
            self.decisions[slot] = LaneChangeDecision(state="blocked" if reason == "no_detour" else "rerouting", reason=reason, source="road_event")
            if key not in self._failed and slot not in self._pending:
                self.caps[slot] = 0.0
                if not submitted and abs(float(world.fleet.speed[slot])) < 0.1:
                    start = (float(world.fleet.x[slot]), float(world.fleet.y[slot]))
                    goal = tuple(float(value) for value in world.slots[slot].goal)
                    heading = float(world.fleet.heading[slot])
                    route_z = world.slots[slot].route_z
                    start_z = float(world.z[slot]) if world.elevated else None
                    goal_z = float(route_z[-1]) if world.elevated and route_z.size else None
                    view = copy(world.map_index)
                    view.closed_edges = closed
                    nodes = tuple(array.view() for array in (world._node_xy, world._node_ids, world._node_z))
                    for array in nodes:
                        array.flags.writeable = False
                    inputs = _Input(view, start, goal, heading, closed, start_z, goal_z, nodes)
                    future = _SEARCH.submit(_run_search, inputs)
                    self._pending[slot] = _Search(key[1], generation, future, start, world.map_index, inputs)
                    submitted = True
        return tuple(installed)
