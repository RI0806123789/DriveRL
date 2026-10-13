"""既知の同方向車線で、認識した固定障害物の回避と復帰を判断する。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from app import config
from app.contracts import LaneChangeDecision, LaneCorridor, RouteLeg
from app.map.elevation import LEVEL_GAP_M
from app.percep.occlusion import RAY_ANGLES
from app.percep.types import CAMERA_RIG, DetClass, OcclusionResult, PerceptionResult
from app.sim.intersection import FRESH_SEC, IntersectionSupervisor, _Actor, _State, _project
from app.sim.signals import stop_speed_limit
from app.warn import warn_once

if TYPE_CHECKING:
    from app.sim.world import World

CHANGE_M = 24.0
CHANGE_SPEED = 5.0
LOOKAHEAD_M = 65.0
GAP_M = 5.0
BUFFER_M = 1.0
HALF_LENGTH = config.VEHICLE_LENGTH / 2.0
HALF_WIDTH = config.VEHICLE_WIDTH / 2.0


@dataclass
class LaneCommand:
    """実際に制御する車だけに適用する速度上限と経路案。"""

    speed_cap: float = math.inf
    plan: tuple[np.ndarray, tuple[RouteLeg, ...]] | None = None
    driving: bool = False


@dataclass
class _LaneState:
    serial: int
    tracking: _State = field(default_factory=_State)
    corridor: LaneCorridor | None = None
    points: np.ndarray | None = None
    cum: np.ndarray | None = None
    legs: tuple[RouteLeg, ...] = ()
    start: float = math.inf
    changed_at: float = math.inf
    passed_at: float = math.inf
    return_at: float = math.inf
    end: float = math.inf
    obstacle_at: float = math.inf
    obstacle_radius: float = 0.0
    active: bool = False
    returning: bool = False
    completed: int = 0
    last_lateral_accel: float = 0.0
    last_at: float = -math.inf
    last_heading: float = 0.0
    demand: tuple[str, ...] = ()


def _smooth(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def avoidance_path(points: np.ndarray, cum: np.ndarray, legs: tuple[RouteLeg, ...], start: float, change_end: float, return_start: float, end: float, offset: float) -> tuple[np.ndarray, tuple[RouteLeg, ...]]:
    """位置・接線・曲率を端でつなぐ車線変更軌道と、写し直した辺の弧長を返す。"""
    marks = np.unique(np.concatenate((cum, np.arange(start, min(end, float(cum[-1])), 0.5), [start, change_end, return_start, end])))
    marks = marks[(marks >= 0.0) & (marks <= cum[-1])]
    base = np.column_stack((np.interp(marks, cum, points[:, 0]), np.interp(marks, cum, points[:, 1])))
    tangent = np.gradient(base, marks, axis=0)
    norm = np.maximum(np.linalg.norm(tangent, axis=1), 1e-9)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) / norm[:, None]
    outward = _smooth((marks - start) / max(change_end - start, 0.1))
    inward = _smooth((marks - return_start) / max(end - return_start, 0.1))
    path = (base + normal * (offset * (outward - inward))[:, None]).astype(np.float32)
    new_cum = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
    mapped = tuple(replace(leg, start_arc=float(np.interp(leg.start_arc, marks, new_cum)), end_arc=float(np.interp(leg.end_arc, marks, new_cum))) for leg in legs)
    return path, mapped


class LaneChangeSupervisor:
    """方策の車には真値を渡さず、背景交通だけ別の入力を使う。"""

    def __init__(self) -> None:
        self.decisions: dict[int, LaneChangeDecision] = {}
        self.commands: dict[int, LaneCommand] = {}
        self._states: dict[int, _LaneState] = {}
        self._tracker = IntersectionSupervisor()
        self._map_index: object | None = None

    def demand(self, slot: int) -> tuple[str, ...]:
        state = self._states.get(slot)
        return state.demand if state else ()

    def active(self, slot: int) -> bool:
        state = self._states.get(slot)
        return state is not None and state.active

    def installed(self, world: World, slot: int) -> None:
        """自分の経路案の採用だけは、認識履歴を保ちながら世代を進める。"""
        self._states[slot].serial = int(world.route_serial[slot])

    @staticmethod
    def _background(world: World, slot: int, now: float) -> list[_Actor]:
        fleet = world.fleet
        x, y = float(fleet.x[slot]), float(fleet.y[slot])
        actors = []
        level = world.same_level_vehicles(slot)
        people_level = world.same_level_people(slot)
        own_z = float(world.z[slot])
        for other in np.flatnonzero(fleet.active):
            if other == slot or not level[other] or math.hypot(float(fleet.x[other]) - x, float(fleet.y[other]) - y) > 80.0:
                continue
            speed, heading = float(fleet.speed[other]), float(fleet.heading[other])
            actors.append(_Actor(DetClass.VEHICLE, float(fleet.x[other]), float(fleet.y[other]), now, speed * math.cos(heading), speed * math.sin(heading), True))
        actors.extend(_Actor(DetClass.OBSTACLE, o.x, o.y, now, measured=True, radius=o.radius) for o in world.obstacles if math.hypot(o.x - x, o.y - y) <= 80.0 and (not world.elevated or abs(o.z - own_z) < LEVEL_GAP_M))
        actors.extend(_Actor(DetClass.PEDESTRIAN, float(px), float(py), now, measured=True, radius=config.PEDESTRIAN_RADIUS) for (px, py), same in zip(world.pedestrian_xy, people_level) if same and math.hypot(float(px) - x, float(py) - y) <= 60.0)
        return actors

    def evaluate(self, world: World, now: float, front: dict[int, PerceptionResult], surround: dict[int, dict[str, PerceptionResult]], ages: dict[int, dict[str, float]], poses: dict[int, dict[str, tuple[float, float, float]]], occlusion: dict[int, OcclusionResult], background: frozenset[int] = frozenset()) -> None:
        """経路を変えずに判断する。経路の採用はenvが制御対象だけに行う。"""
        self.decisions = {}
        self.commands = {}
        if self._map_index is not world.map_index:
            self._states.clear()
            self._tracker = IntersectionSupervisor()
            self._map_index = world.map_index
        alive = set(int(s) for s in np.flatnonzero(world.fleet.active))
        for slot in set(self._states) - alive:
            self._states.pop(slot)
        for slot in sorted(alive):
            state = self._states.get(slot)
            serial = int(world.route_serial[slot])
            if state is None or state.serial != serial:
                completed = state.completed if state is not None and world.slots[slot].steps > 0 else 0
                state = self._states[slot] = _LaneState(serial, completed=completed)
            state.demand = ()
            is_background = slot in background
            if not state.active:
                route = world.slots[slot]
                build = getattr(world.map_index, "lane_corridor", None)
                corridor = build(route.route, route.route_legs, float(world.arc[slot])) if build is not None else None
                if corridor is None:
                    continue
                state.corridor = corridor
                state.points, state.cum, state.legs = route.route, route.route_cum, route.route_legs
            if is_background:
                actors = self._background(world, slot, now)
            else:
                self._tracker._track(world, slot, state.tracking, now, front.get(slot), surround.get(slot, {}), ages.get(slot, {}), poses.get(slot, {}), (DetClass.VEHICLE, DetClass.PEDESTRIAN, DetClass.OBSTACLE))
                actors = state.tracking.actors
            self._evaluate_slot(world, slot, state, actors, now, is_background, front.get(slot), surround.get(slot, {}), ages.get(slot, {}), occlusion.get(slot))

    @staticmethod
    def _frame(state: _LaneState, x: float, y: float) -> tuple[float, float, float]:
        corridor = state.corridor
        lo = max(0, int(np.searchsorted(state.cum, corridor.begin_arc)) - 1)
        hi = min(len(state.cum), int(np.searchsorted(state.cum, corridor.end_arc)) + 2)
        if hi - lo < 2:
            lo = max(0, min(lo, len(state.points) - 2))
            hi = min(len(state.points), lo + 2)
        points, cum = state.points[lo:hi], state.cum[lo:hi]
        if len(points) < 2:
            return float(cum[0]) if len(cum) else 0.0, math.inf, 0.0
        arc, _, heading = _project(points, cum, x, y)
        px, py = np.interp(arc, cum, points[:, 0]), np.interp(arc, cum, points[:, 1])
        return arc, -math.sin(heading) * (x - px) + math.cos(heading) * (y - py), heading

    @staticmethod
    def _visible(state: _LaneState, delta: float, heading_error: float, rear_distance: float, front: PerceptionResult | None, surround: dict[str, PerceptionResult], ages: dict[str, float], occlusion: OcclusionResult | None) -> bool:
        side = "left" if delta > 0.0 else "right"
        if front is None or any(key not in surround or ages.get(key, math.inf) > FRESH_SEC for key in ("rear", side)) or occlusion is None:
            return False
        probes = [("front", math.atan2(delta, 25.0), 24.0), ("rear", math.atan2(delta, -25.0), rear_distance), (side, math.copysign(math.pi / 2, delta), abs(delta) + HALF_WIDTH)]
        if rear_distance < 24.0:
            probes.append(("rear", math.pi + heading_error, 24.0))
        for key, angle, distance in probes:
            view = occlusion.cameras.get(key)
            if view is None or not view.known:
                return False
            camera = next(c for c in CAMERA_RIG if c.key == key)
            relative = math.atan2(math.sin(angle - heading_error - camera.yaw), math.cos(angle - heading_error - camera.yaw))
            if abs(relative) > math.radians(camera.fov_deg) / 2.0:
                return False
            ray = int(np.argmin(np.abs(RAY_ANGLES - relative)))
            if float(view.depths[ray]) < distance:
                return False
        return True

    def _gap(self, state: _LaneState, actors: list[_Actor], arc: float, speed: float, offset: float, now: float) -> tuple[bool, float, float]:
        minimum, required = math.inf, 0.0
        clear = True
        for actor in actors:
            position = actor.position(now)
            where, lateral, heading = self._frame(state, *position)
            radius = actor.radius if actor.cls == DetClass.OBSTACLE else (config.PEDESTRIAN_RADIUS if actor.cls == DetClass.PEDESTRIAN else HALF_WIDTH)
            if abs(lateral - offset) > HALF_WIDTH + radius + BUFFER_M:
                continue
            length = actor.radius if actor.cls != DetClass.VEHICLE else HALF_LENGTH
            signed = where - arc
            gap = abs(signed) - HALF_LENGTH - length
            minimum = min(minimum, gap)
            if actor.cls != DetClass.VEHICLE and arc - HALF_LENGTH <= where <= state.end + HALF_LENGTH:
                clear = False
            velocity = actor.vx * math.cos(heading) + actor.vy * math.sin(heading) if actor.measured else (25.0 if signed < 0.0 else 0.0)
            planned = min(speed, CHANGE_SPEED) if signed < 0.0 else max(speed, CHANGE_SPEED)
            horizon = max(5.0, CHANGE_M / max(2.5, min(speed, CHANGE_SPEED)))
            closing = (velocity - planned) if signed < 0.0 else (planned - velocity)
            closing = max(0.0, closing)
            braking = closing * closing / (2.0 * max(gap - GAP_M, 0.1))
            if signed < 0.0:
                required = max(required, braking)
            if gap < GAP_M + max(planned if signed > 0.0 else velocity, 0.0) * 1.2 or gap - closing * horizon < GAP_M or braking > 2.0:
                clear = False
        return clear, minimum, required

    def _evaluate_slot(self, world: World, slot: int, state: _LaneState, actors: list[_Actor], now: float, background: bool, front: PerceptionResult | None, surround: dict[str, PerceptionResult], ages: dict[str, float], occlusion: OcclusionResult | None) -> None:
        fleet = world.fleet
        arc, lateral, road_heading = self._frame(state, float(fleet.x[slot]), float(fleet.y[slot]))
        if not math.isfinite(lateral):
            warn_once("lane_change.short_route", "車線変更の経路に線分がないため、安全確認待ちにします。")
            self.commands[slot] = LaneCommand(speed_cap=0.0)
            self.decisions[slot] = LaneChangeDecision(reason="visibility", source="background" if background else "camera", completed=state.completed)
            return
        speed = max(0.0, float(fleet.speed[slot]))
        corridor = state.corridor
        if not state.active:
            blockers = []
            traffic = False
            for actor in actors:
                where, lat, _ = self._frame(state, *actor.position(now))
                radius = actor.radius if actor.cls == DetClass.OBSTACLE else HALF_WIDTH
                if 0.0 < where - arc <= LOOKAHEAD_M and abs(lat) <= HALF_WIDTH + radius + 0.4:
                    if actor.cls == DetClass.OBSTACLE:
                        blockers.append((where, radius))
                    elif actor.cls == DetClass.VEHICLE:
                        traffic = True
            if not blockers:
                if traffic:
                    self.decisions[slot] = LaneChangeDecision(state="waiting", reason="traffic", source="background" if background else "camera", completed=state.completed)
                return
            state.obstacle_at, state.obstacle_radius = min(blockers)
            state.passed_at = max(a + r for a, r in blockers) + HALF_LENGTH + GAP_M + 2.0
            state.start = arc
            state.changed_at = arc + CHANGE_M
            state.return_at = max(state.passed_at, state.changed_at + 6.0)
            state.end = state.return_at + CHANGE_M
        command = self.commands[slot] = LaneCommand()
        decision = self.decisions[slot] = LaneChangeDecision(source="background" if background else "camera", completed=state.completed)
        target = 0.0 if state.returning or (state.active and arc >= state.return_at - 12.0 and arc >= state.passed_at) else corridor.neighbor_offset
        state.demand = ("rear", "left" if target - lateral > 0.0 else "right")
        clear, decision.min_gap, decision.rear_decel = self._gap(state, actors, arc, speed, target, now)
        decision.direction = (1 if corridor.neighbor_offset > 0.0 else -1) * (-1 if target == 0.0 else 1)
        rear_distance = 24.0
        if state.active and target == 0.0:
            for actor in actors:
                where, lat, _ = self._frame(state, *actor.position(now))
                if actor.cls == DetClass.OBSTACLE and now - actor.at <= FRESH_SEC and abs(lat) < 0.3 and actor.radius + config.VEHICLE_WIDTH >= corridor.lane_width / 2.0 and where + actor.radius + HALF_LENGTH + GAP_M < arc:
                    rear_distance = min(rear_distance, max(GAP_M, arc - where - actor.radius - HALF_LENGTH - 1.0))
        visible = background or self._visible(state, target - lateral, float(fleet.heading[slot]) - road_heading, rear_distance, front, surround, ages, occlusion)
        if not state.active:
            command.speed_cap = float(stop_speed_limit(np.float64(state.obstacle_at - arc - state.obstacle_radius - HALF_LENGTH), abs(config.MAX_DECEL), config.DT, margin_m=2.0))
            decision.state, decision.reason = "waiting", "obstacle"
            if not corridor.allowed:
                decision.reason = corridor.reason
            elif state.end + HALF_LENGTH + 4.0 >= corridor.end_arc or state.start - HALF_LENGTH < corridor.begin_arc + 4.0 or state.changed_at + HALF_LENGTH + state.obstacle_radius + 2.0 >= state.obstacle_at:
                decision.reason = "regulation"
            elif traffic:
                decision.reason = "traffic"
            elif int(fleet.gear[slot]) < 0:
                decision.reason = "traffic"
            elif not clear:
                decision.reason = "gap"
            elif not visible:
                decision.reason = "visibility"
            else:
                command.plan = avoidance_path(state.points, state.cum, state.legs, state.start, state.changed_at, state.return_at, state.end, corridor.neighbor_offset)
                command.speed_cap = CHANGE_SPEED
                command.driving = True
                decision.state, decision.reason = "changing", ""
            return
        if arc >= state.end + HALF_LENGTH and abs(lateral) < 0.35:
            state.active = False
            state.completed += 1
            decision.state, decision.reason, decision.direction, decision.completed = "complete", "", 0, state.completed
            return
        state.returning = state.returning or arc >= state.return_at
        decision.state = "returning" if state.returning else ("passing" if arc >= state.changed_at else "changing")
        command.driving = True
        command.speed_cap = CHANGE_SPEED
        if not visible or not clear:
            decision.reason = "visibility" if not visible else "gap"
            if arc >= state.changed_at and not state.returning:
                command.speed_cap = float(stop_speed_limit(np.float64(state.return_at - arc), abs(config.MAX_DECEL), config.DT, margin_m=2.5))
            else:
                command.speed_cap = 0.0
        for actor in actors:
            current_arc, distance, _ = _project(world.slots[slot].route, world.slots[slot].route_cum, *actor.position(now))
            if distance <= HALF_WIDTH + actor.radius + 0.5 and current_arc >= float(world.arc[slot]) - HALF_LENGTH:
                cap = float(stop_speed_limit(np.float64(current_arc - float(world.arc[slot]) - HALF_LENGTH - (HALF_LENGTH if actor.cls == DetClass.VEHICLE else actor.radius)), abs(config.MAX_DECEL), config.DT, margin_m=3.0))
                command.speed_cap = min(command.speed_cap, cap)
        heading = float(fleet.heading[slot])
        elapsed = now - state.last_at
        if elapsed > 1e-6:
            turn = math.atan2(math.sin(heading - state.last_heading), math.cos(heading - state.last_heading))
            acceleration = float(fleet.lateral_accel[slot]) if fleet.dynamics is not None else (speed * turn / elapsed if math.isfinite(state.last_at) else 0.0)
            decision.lateral_accel = acceleration
            decision.lateral_jerk = (acceleration - state.last_lateral_accel) / elapsed if math.isfinite(state.last_at) else 0.0
            state.last_lateral_accel, state.last_at, state.last_heading = acceleration, now, heading
        else:
            decision.lateral_accel = state.last_lateral_accel

    def activate(self, slot: int) -> None:
        """環境が安全な経路案を採用したら、変更中の状態へ進む。"""
        self._states[slot].active = True
