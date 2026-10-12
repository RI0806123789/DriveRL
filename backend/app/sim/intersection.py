"""静的な交錯区間とカメラ追跡から、交差点の進入上限を決める。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from app import config
from app.contracts import IntersectionDecision, IntersectionMovement, RouteIntersection
from app.percep.encoder import local_xy
from app.percep.types import CAMERA_RIG, DetClass, OcclusionResult, PerceptionResult
from app.sim.signals import stop_speed_limit

if TYPE_CHECKING:
    from app.sim.world import World

FRESH_SEC = 0.25
MEMORY_SEC = 1.0
LOOKAHEAD_M = 35.0
TIME_MARGIN_SEC = 1.2
CROSSING_SPEED_MPS = 3.0
UNKNOWN_SPEED_MPS = 8.0
STOP_SEC = 1.0
HALF_LENGTH = config.VEHICLE_LENGTH / 2.0
HALF_WIDTH = config.VEHICLE_WIDTH / 2.0


@dataclass
class _Actor:
    """撮影時の世界座標と、検出の差分で求めた速度。"""

    cls: DetClass
    x: float
    y: float
    at: float
    vx: float = 0.0
    vy: float = 0.0
    measured: bool = False
    movement: tuple[int, int, int] | None = None
    history: tuple[tuple[float, float, float], ...] = ()
    heading: float | None = None
    radius: float = 0.9

    def position(self, now: float) -> tuple[float, float]:
        """短い遮蔽の間だけ位置を外挿する。"""
        dt = max(0.0, min(MEMORY_SEC, now - self.at))
        return self.x + self.vx * dt, self.y + self.vy * dt


@dataclass
class _State:
    """経路の世代ごとに捨てる追跡・停止履歴。"""

    serial: int = -1
    key: tuple[int, float] | None = None
    actors: list[_Actor] = field(default_factory=list)
    stamps: dict[str, tuple[float, PerceptionResult]] = field(default_factory=dict)
    waiting_since: float | None = None
    stop_seen: bool = False
    stopped_since: float | None = None
    stop_done: bool = False
    demand: tuple[str, ...] = ()


def _project(points: np.ndarray, cum: np.ndarray, x: float, y: float) -> tuple[float, float, float]:
    """点列上の最近点の弧長・距離・接線方位を返す。"""
    seg = np.diff(points, axis=0)
    sq = np.sum(seg * seg, axis=1)
    delta = np.array([x, y]) - points[:-1]
    t = np.clip(np.sum(delta * seg, axis=1) / np.maximum(sq, 1e-9), 0.0, 1.0)
    distance_sq = np.sum((delta - seg * t[:, None]) ** 2, axis=1)
    i = int(np.argmin(distance_sq))
    return (
        float(cum[i] + t[i] * (cum[i + 1] - cum[i])),
        math.sqrt(float(distance_sq[i])),
        math.atan2(float(seg[i, 1]), float(seg[i, 0])),
    )


def _stop_cap(target: float, arc: float, margin: float = 0.6) -> float:
    """バンパーと交錯区間の間を空ける速度上限。"""
    return float(stop_speed_limit(np.float64(target - arc), abs(config.MAX_DECEL), config.DT, margin_m=HALF_LENGTH + margin))


class IntersectionSupervisor:
    """学習車は認識のみ、背景車は専用の真値入力で進入判断する。"""

    def __init__(self) -> None:
        self.decisions: dict[int, IntersectionDecision] = {}
        self._states: dict[int, _State] = {}
        self._geometry: dict[int, tuple[IntersectionMovement, np.ndarray, np.ndarray]] = {}
        self._stop_edges: frozenset[tuple[int, int]] | None = None

    def demand(self, slot: int) -> tuple[str, ...]:
        """次回撮影を優先する周囲カメラ。"""
        state = self._states.get(slot)
        return state.demand if state else ()

    def evaluate(
        self,
        world: World,
        now: float,
        front: dict[int, PerceptionResult],
        surround: dict[int, dict[str, PerceptionResult]],
        ages: dict[int, dict[str, float]],
        poses: dict[int, dict[str, tuple[float, float, float]]],
        occlusion: dict[int, OcclusionResult],
        background: frozenset[int] = frozenset(),
    ) -> None:
        """交差点の近くの車だけ評価し、支援を適用するかは環境に任せる。"""
        self.decisions = {}
        alive = set(int(s) for s in np.flatnonzero(world.fleet.active))
        for slot in set(self._states) - alive:
            self._states.pop(slot)
        for slot in alive:
            serial = int(world.route_serial[slot])
            state = self._states.get(slot)
            if state is None or state.serial != serial:
                state = self._states[slot] = _State(serial=serial)
            arc = float(world.arc[slot])
            passage = next((p for p in world.slots[slot].intersections if p.exit_arc + HALF_LENGTH >= arc), None)
            state.demand = ()
            if passage is None or passage.entry_arc - arc > LOOKAHEAD_M:
                state.actors.clear()
                state.stamps.clear()
                continue
            key = (passage.node_id, passage.entry_arc)
            if state.key != key:
                state.key = key
                state.waiting_since = None
                state.stop_seen = state.stop_done = False
                state.stopped_since = None
            is_background = slot in background
            if is_background:
                actors = self._background_actors(world, slot, now, passage.node_id)
                state.stop_seen = self._background_stop(world, slot, passage)
            else:
                self._track(world, slot, state, now, front.get(slot), surround.get(slot, {}), ages.get(slot, {}), poses.get(slot, {}))
                actors = state.actors
            decision = self._decide(world, slot, passage, state, actors, now, is_background, front.get(slot), ages.get(slot, {}), occlusion.get(slot))
            self.decisions[slot] = decision

    @staticmethod
    def _background_actors(world: World, slot: int, now: float, node_id: int) -> list[_Actor]:
        """背景交通だけが参照する他車・徒歩キャラを含む歩行者。"""
        fleet = world.fleet
        x, y = float(fleet.x[slot]), float(fleet.y[slot])
        actors: list[_Actor] = []
        level = world.same_level_vehicles(slot)
        people_level = world.same_level_people(slot)
        for other in np.flatnonzero(fleet.active):
            ox, oy = float(fleet.x[other]), float(fleet.y[other])
            if int(other) == slot or not level[other] or math.hypot(ox - x, oy - y) > 75.0:
                continue
            speed, heading = float(fleet.speed[other]), float(fleet.heading[other])
            intent = None
            other_arc = float(world.arc[other])
            other_state = world.slots[int(other)]
            passage = next((p for p in other_state.intersections if p.node_id == node_id and p.exit_arc + HALF_LENGTH >= other_arc), None)
            if passage is not None:
                for incoming, outgoing in zip(other_state.route_legs, other_state.route_legs[1:]):
                    if incoming.exit_node == passage.node_id and abs(incoming.end_arc - passage.entry_arc) < 0.5:
                        intent = (incoming.exit_node, incoming.entry_node, outgoing.exit_node)
                        break
            actors.append(_Actor(DetClass.VEHICLE, ox, oy, now, speed * math.cos(heading), speed * math.sin(heading), True, intent))
        actors.extend(_Actor(DetClass.PEDESTRIAN, float(px), float(py), now, measured=True) for (px, py), same in zip(world.pedestrian_xy, people_level) if same and math.hypot(float(px) - x, float(py) - y) <= 50.0)
        return actors

    def _background_stop(self, world: World, slot: int, passage: RouteIntersection) -> bool:
        """背景車の進入辺に属する一時停止標識。"""
        approaches: set[tuple[int, int]] = set()
        for leg in getattr(world.slots[slot], "route_legs", ()):
            if leg.exit_node == passage.node_id and abs(leg.end_arc - passage.entry_arc) < 0.5:
                approaches.add((leg.edge_id, leg.exit_node))
        if self._stop_edges is None:
            self._stop_edges = frozenset((s.edge_id, s.node_id) for s in world.map_index.data.signs if s.kind == "stop")
        return not approaches.isdisjoint(self._stop_edges)

    def _track(
        self, world: World, slot: int, state: _State, now: float,
        front: PerceptionResult | None, surround: dict[str, PerceptionResult],
        ages: dict[str, float], poses: dict[str, tuple[float, float, float]],
        classes: tuple[DetClass, ...] = (DetClass.VEHICLE, DetClass.PEDESTRIAN, DetClass.STOP_SIGN),
    ) -> None:
        """同じ古い画像を速度測定に使い直さず、撮影時の自車姿勢で追跡する。"""
        fleet = world.fleet
        current = (float(fleet.x[slot]), float(fleet.y[slot]), float(fleet.heading[slot]))
        samples: list[tuple[DetClass, float, float, float, frozenset[str]]] = []
        radii: list[tuple[float, float, float]] = []
        for camera in CAMERA_RIG:
            result = front if camera.key == "front" else surround.get(camera.key)
            age = 0.0 if camera.key == "front" else ages.get(camera.key, math.inf)
            stamp = now - age
            previous_result = state.stamps.get(camera.key)
            if result is None or age > FRESH_SEC or (previous_result is not None and stamp == previous_result[0] and result is previous_result[1]):
                continue
            state.stamps[camera.key] = (stamp, result)
            px, py, heading = current if camera.key == "front" else poses.get(camera.key, current)
            ch, sh = math.cos(heading), math.sin(heading)
            for detection in result.detections:
                if detection.confidence < 0.35 or detection.cls not in classes:
                    continue
                fx, fy, distance = local_xy(detection, camera)
                if not all(math.isfinite(v) for v in (fx, fy, distance)) or distance > 75.0:
                    continue
                x, y = px + ch * fx - sh * fy, py + sh * fx + ch * fy
                if detection.cls == DetClass.OBSTACLE:
                    radius = max(0.3, min(6.0, (detection.x1 - detection.x0) * camera.width * distance / (2.0 * camera.focal_px)))
                    radii.append((x, y, radius))
                if detection.cls == DetClass.STOP_SIGN:
                    if camera.key == "front" and fx > 0.0 and abs(fy) <= 8.0:
                        route = world.slots[slot]
                        sign_arc, _, _ = _project(route.route, route.route_cum, x, y)
                        if state.key is not None and abs(sign_arc - state.key[1]) <= 12.0:
                            state.stop_seen = True
                    continue
                merge = next((i for i, (cls, sx, sy, at, cameras) in enumerate(samples) if camera.key not in cameras and cls == detection.cls and abs(stamp - at) < 0.06 and math.hypot(x - sx, y - sy) < (3.25 if cls == DetClass.VEHICLE else 1.0)), None)
                if merge is None:
                    samples.append((detection.cls, x, y, stamp, frozenset({camera.key})))
                else:
                    cls, sx, sy, at, cameras = samples[merge]
                    samples[merge] = (cls, (x + sx) / 2.0, (y + sy) / 2.0, max(at, stamp), cameras | {camera.key})
        previous = [a for a in state.actors if now - a.at <= MEMORY_SEC]
        used: set[int] = set()
        updated: list[_Actor] = []
        for cls, x, y, stamp, _ in sorted(samples, key=lambda sample: sample[3]):
            candidates = [(math.hypot(x - a.position(stamp)[0], y - a.position(stamp)[1]), i) for i, a in enumerate(previous) if i not in used and a.cls == cls and stamp >= a.at]
            distance, index = min(candidates, default=(math.inf, -1))
            actor = _Actor(cls, x, y, stamp)
            if cls == DetClass.OBSTACLE:
                actor.radius = min(radii, key=lambda r: math.hypot(x - r[0], y - r[1]))[2]
            actor.history = ((stamp, x, y),)
            if index >= 0 and distance < 4.0 + 15.0 * max(0.0, stamp - previous[index].at):
                old = previous[index]
                used.add(index)
                actor.heading = old.heading
                history = tuple(p for p in (old.history or ((old.at, old.x, old.y),)) if stamp - p[0] <= 0.5)
                actor.history = (*history, (stamp, x, y)) if stamp > old.at else (*history[:-1], (stamp, x, y))
                dt = stamp - old.at
                if dt > 1e-4:
                    at, hx, hy = history[0] if history else (old.at, old.x, old.y)
                    span = max(stamp - at, 1e-4)
                    vx, vy = (x - hx) / span, (y - hy) / span
                    if math.hypot(vx, vy) <= 35.0:
                        if old.measured:
                            vx, vy = 0.65 * old.vx + 0.35 * vx, 0.65 * old.vy + 0.35 * vy
                        actor.vx, actor.vy, actor.measured = vx, vy, True
                        if math.hypot(vx, vy) >= 1.5:
                            actor.heading = math.atan2(vy, vx)
                else:
                    actor.vx, actor.vy, actor.measured = old.vx, old.vy, old.measured
            updated.append(actor)
        updated.extend(a for i, a in enumerate(previous) if i not in used)
        state.actors = updated

    def _movement_arrays(self, movement: IntersectionMovement) -> tuple[np.ndarray, np.ndarray]:
        """候補進路のnumpy変換を、使っている地図の期間だけ再利用する。"""
        key = id(movement)
        cached = self._geometry.get(key)
        if cached is None:
            cached = (movement, np.asarray(movement.points), np.asarray(movement.cum))
            self._geometry[key] = cached
        return cached[1], cached[2]

    def _decide(
        self, world: World, slot: int, passage: RouteIntersection, state: _State,
        actors: list[_Actor], now: float, background: bool, front: PerceptionResult | None,
        ages: dict[str, float], occlusion: OcclusionResult | None,
    ) -> IntersectionDecision:
        """停止・出口・横断者・優先流の順に、安全な通過時間を確かめる。"""
        arc = float(world.arc[slot])
        speed = max(0.0, float(world.fleet.speed[slot]))
        committed = arc + HALF_LENGTH > passage.entry_arc + 0.5
        decision = IntersectionDecision(passage.node_id, state="committed" if committed else "clear", source="background" if background else "camera")
        target, reason, gap = math.inf, "", math.inf
        if state.stop_seen and not state.stop_done and not committed:
            if passage.entry_arc - arc <= HALF_LENGTH + 1.2 and speed < 0.15:
                if state.stopped_since is None:
                    state.stopped_since = now
                state.stop_done = now - state.stopped_since >= STOP_SEC
            else:
                state.stopped_since = None
            if not state.stop_done:
                target, reason = passage.entry_arc, "stop"
        route = world.slots[slot]
        lo = max(0, int(np.searchsorted(route.route_cum, max(arc, passage.entry_arc - 5.0))) - 1)
        hi = min(len(route.route), int(np.searchsorted(route.route_cum, passage.exit_arc + config.VEHICLE_LENGTH + 5.0)) + 2)
        points, cum = route.route[lo:hi], route.route_cum[lo:hi]
        ego_edge = next((leg.edge_id for leg in route.route_legs if leg.exit_node == passage.node_id and abs(leg.end_arc - passage.entry_arc) < 0.5), -1)
        for actor in actors:
            x, y = actor.position(now)
            actor_arc, distance, tangent = _project(points, cum, x, y)
            if actor.cls == DetClass.PEDESTRIAN:
                future = [(_project(points, cum, x + actor.vx * dt, y + actor.vy * dt), dt) for dt in (0.0, 1.0, 2.0, 3.0)]
                risks = [(a, dt) for (a, d, _), dt in future if d <= HALF_WIDTH + config.PEDESTRIAN_RADIUS + 0.6 and max(arc, passage.entry_arc - 3.0) <= a <= passage.exit_arc + 5.0]
                if risks:
                    crossing, _ = min(risks)
                    target = min(target, crossing - config.PEDESTRIAN_RADIUS - 1.0, passage.entry_arc if not committed else math.inf)
                    reason = "pedestrian"
                continue
            if not committed and distance <= config.VEHICLE_WIDTH + 0.6 and passage.exit_arc - HALF_LENGTH <= actor_arc <= passage.exit_arc + config.VEHICLE_LENGTH + HALF_LENGTH + 3.0:
                target, reason = min(target, passage.entry_arc), "exit_blocked"
            plausible: set[int] | None = None
            if actor.movement is None and passage.movements:
                scores = []
                for candidate in passage.movements:
                    mpoints, mcum = self._movement_arrays(candidate)
                    _, offset, heading = _project(mpoints, mcum, x, y)
                    angle_cost = 2.0 * (1.0 - math.cos(heading - actor.heading)) if actor.heading is not None else 0.0
                    scores.append((offset + angle_cost, id(candidate)))
                best = min(score for score, _ in scores)
                plausible = {key for score, key in scores if score <= best + 0.5}
            for conflict in passage.conflicts:
                movement = conflict.movement
                if plausible is not None and id(movement) not in plausible:
                    continue
                if actor.movement is not None and actor.movement != (movement.node_id, movement.entry_node, movement.exit_node):
                    continue
                mpoints, mcum = self._movement_arrays(movement)
                other_arc, offset, heading = _project(mpoints, mcum, x, y)
                if offset > HALF_WIDTH + 1.0 or other_arc - HALF_LENGTH > conflict.other_end_arc:
                    continue
                projected_speed = actor.vx * math.cos(heading) + actor.vy * math.sin(heading)
                if actor.measured and (projected_speed < -0.5 or (math.hypot(actor.vx, actor.vy) > 1.0 and projected_speed < math.hypot(actor.vx, actor.vy) * 0.6)):
                    continue
                occupied = distance <= HALF_WIDTH + 1.0 and other_arc + HALF_LENGTH > movement.entry_arc and other_arc + HALF_LENGTH >= conflict.other_start_arc and other_arc - HALF_LENGTH <= conflict.other_end_arc
                if actor.measured and projected_speed < 0.5 and not occupied:
                    if not committed and other_arc <= movement.entry_arc and movement.entry_arc - other_arc <= 12.0 and ego_edge > movement.incoming_edge_id:
                        target = min(target, conflict.ego_start_arc, passage.entry_arc)
                        if not reason:
                            reason = "priority"
                    continue
                v = max(0.5, projected_speed) if actor.measured else UNKNOWN_SPEED_MPS
                other_start = max(0.0, (conflict.other_start_arc - other_arc - HALF_LENGTH) / v)
                other_end = max(0.0, (conflict.other_end_arc - other_arc + HALF_LENGTH) / v)
                ego_v = max(CROSSING_SPEED_MPS, speed)
                ego_start = max(0.0, (conflict.ego_start_arc - arc - HALF_LENGTH) / ego_v)
                ego_end = max(0.0, (conflict.ego_end_arc - arc + HALF_LENGTH) / ego_v)
                if arc - HALF_LENGTH > conflict.ego_end_arc or other_start > ego_end + TIME_MARGIN_SEC or ego_start > other_end + TIME_MARGIN_SEC:
                    continue
                delta = movement.entry_heading - passage.entry_heading
                oncoming = math.cos(delta) < -0.7 and passage.turn == "right" and movement.turn in ("straight", "left")
                opposing_turn = math.cos(delta) < -0.7 and passage.turn == movement.turn == "right" and ego_edge > movement.incoming_edge_id
                signals = route.signal_arcs
                signalized = bool(np.any((signals >= passage.entry_arc - 20.0) & (signals <= passage.entry_arc + 5.0)))
                wider = movement.road_width >= passage.road_width + 2.0
                equal_left = abs(movement.road_width - passage.road_width) < 2.0 and math.sin(delta) < -0.5
                priority = not signalized and (wider or equal_left)
                if occupied or oncoming or priority or opposing_turn:
                    target = min(target, conflict.ego_start_arc, passage.entry_arc if not committed else math.inf)
                    if reason not in ("pedestrian", "exit_blocked", "stop"):
                        reason = "oncoming" if oncoming or reason == "oncoming" else "priority"
                    gap = min(gap, other_start - ego_end)
        needed: set[str] = set()
        for conflict in passage.conflicts:
            delta = conflict.movement.entry_heading - passage.entry_heading
            if math.cos(delta) < -0.7:
                needed.add("front")
            elif math.sin(delta) < -0.5:
                needed.add("left")
            elif math.sin(delta) > 0.5:
                needed.add("right")
        if passage.turn != "straight":
            needed.add(passage.turn)
        state.demand = tuple(sorted(needed - {"front"})) if not background else ()
        if not background:
            missing = front is None or any(ages.get(key, math.inf) > FRESH_SEC for key in needed - {"front"})
            blind = occlusion is not None and any((occlusion.los_left if key == "left" else occlusion.los_right) is None or float(occlusion.los_left if key == "left" else occlusion.los_right) < 18.0 for key in needed & {"left", "right"})
            if (missing and not committed) or blind:
                visibility_cap = _stop_cap(passage.entry_arc, arc) if missing else math.sqrt(1.0 + 3.0 * max(passage.entry_arc - arc - HALF_LENGTH - 1.5, 0.0))
                if visibility_cap < _stop_cap(target, arc) or not reason:
                    decision.speed_cap = visibility_cap
                    if not reason:
                        reason = "visibility"
        decision.speed_cap = min(decision.speed_cap, _stop_cap(target, arc))
        decision.reason = reason
        decision.gap_seconds = gap if math.isfinite(gap) else None
        if reason:
            if not committed:
                decision.state = "waiting" if speed < 0.3 and decision.speed_cap <= 1.0 else "approaching"
            if state.waiting_since is None:
                state.waiting_since = now
            decision.waiting_seconds = max(0.0, now - state.waiting_since)
        else:
            state.waiting_since = None
        return decision
