"""既知の全幅閉鎖に対する経路探索を、物理ステップの外で進める。"""

from __future__ import annotations

import math
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from app import config
from app.contracts import LaneChangeDecision
from app.sim.signals import stop_speed_limit
from app.warn import warn_once

if TYPE_CHECKING:
    from app.sim.world import Route, World

_SEARCH = ThreadPoolExecutor(max_workers=1, thread_name_prefix="driverl-detour")


@dataclass
class _Search:
    serial: int
    generation: int
    future: Future
    from_xy: tuple[float, float]
    map_index: object


def _detour(
    world: World, start: tuple[float, float], goal: tuple[float, float], heading: float, closed: frozenset[int],
    start_z: float | None = None, goal_z: float | None = None,
) -> Route | None:
    """閉鎖辺と車体幅での建物重なりを含む候補を検索側で拒否する。立体の地図では出発・目的地の階を保つ。"""
    route = world.route_between(start, goal, heading=heading, speed=0.0, src_z=start_z, dst_z=goal_z)
    if route is None or any(leg.edge_id in closed for leg in route.legs):
        return None
    cum = world._cumulative_length(route.points)
    marks = np.arange(0.0, float(cum[-1]) + 0.25, 0.5)
    points = np.column_stack((np.interp(marks, cum, route.points[:, 0]), np.interp(marks, cum, route.points[:, 1])))
    tangent = np.gradient(points, axis=0)
    norm = np.maximum(np.linalg.norm(tangent, axis=1), 1e-9)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) / norm[:, None]
    for side in (-config.VEHICLE_WIDTH / 2.0 - 0.3, 0.0, config.VEHICLE_WIDTH / 2.0 + 0.3):
        probe = points + normal * side
        if np.any(world.map_index.occupancy.sample_building(probe[:, 0], probe[:, 1])):
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

    def evaluate(self, world: World, enabled: frozenset[int]) -> tuple[int, ...]:
        """完了した検索だけ取り込み、1ステップで新しく依頼する検索は1台までにする。"""
        self.decisions, self.caps = {}, {}
        installed = []
        if self._map_index is not world.map_index:
            for job in self._pending.values():
                job.future.cancel()
            self._pending.clear()
            self._failed.clear()
            self._map_index = world.map_index
        closed = getattr(world.map_index, "closed_edges", frozenset())
        generation = int(getattr(world.map_index, "closure_generation", 0))
        if not closed:
            for job in self._pending.values():
                job.future.cancel()
            self._pending.clear()
            self._failed.clear()
            return ()
        submitted = False
        self._failed = {key for key in self._failed if key[2] == generation and world.fleet.active[key[0]] and int(world.route_serial[key[0]]) == key[1]}
        for slot in list(self._pending):
            job = self._pending[slot]
            if slot not in enabled or not world.fleet.active[slot] or job.map_index is not world.map_index or job.generation != generation or job.serial != int(world.route_serial[slot]):
                job.future.cancel()
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
                    goal = world.slots[slot].goal
                    heading = float(world.fleet.heading[slot])
                    route_z = world.slots[slot].route_z
                    start_z = float(world.z[slot]) if world.elevated else None
                    goal_z = float(route_z[-1]) if world.elevated and route_z.size else None
                    future = _SEARCH.submit(_detour, world, start, goal, heading, closed, start_z, goal_z)
                    self._pending[slot] = _Search(key[1], generation, future, start, world.map_index)
                    submitted = True
        return tuple(installed)
