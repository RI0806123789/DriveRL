"""固定条件の合成交差点で、待機・解放・再発進と衝突を閉ループ検証する。"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from shapely.geometry import LineString, Point

from app import config
from app.contracts import Bounds, MapData, MapEdge, MapNode, MapSign, MapSignal, SimParams
from app.map.index import build_map_index
from app.percep.types import OcclusionResult, PerceptionResult
from app.sim.intersection import HALF_LENGTH, STOP_SEC, IntersectionSupervisor
from app.sim.env import SimulationEnv
from app.sim.signals import GREEN, RED, SignalController
from app.sim.world import World

ARM_M = 80.0
NODES = {"C": 0, "W": 1, "E": 2, "N": 3, "S": 4}
EDGES = {"W": 0, "E": 1, "N": 2, "S": 3}
FAILURES: list[str] = []


@dataclass(frozen=True)
class Actor:
    """相手車の再現可能な進路、入口までの距離、一定速度。"""

    src: str
    dst: str
    gap: float = 15.0
    speed: float = 8.0
    blocker: bool = False


@dataclass
class Result:
    """1 ケースの通過・待機・解放を記録する。"""

    rows: list[dict] = field(default_factory=list)
    collided: bool = False
    passed: bool = False
    cleared_at: float | None = None
    start_after_clear: float | None = None
    entered_at: float | None = None


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK' if ok else 'NG'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def junction_map(*, signalized: bool = False, priority_axis: bool = False, stop: bool = False) -> MapData:
    """原点の十字路に信号・道路幅・一時停止だけを設定する。"""
    nodes = [MapNode(i, x, y) for i, (x, y) in enumerate([
        (0.0, 0.0), (-ARM_M, 0.0), (ARM_M, 0.0), (0.0, ARM_M), (0.0, -ARM_M),
    ])]
    edges = [MapEdge(i, 0, node.id, 4 if priority_axis and i < 2 else 2,
                     14.0 if priority_axis and i < 2 else 7.0, False, 11.1,
                     [(0.0, 0.0), (node.x, node.y)], ARM_M)
             for i, node in enumerate(nodes[1:])]
    signals = []
    if signalized:
        for edge in edges:
            node = nodes[edge.v]
            heading = math.atan2(-node.y, -node.x)
            signals.append(MapSignal(edge.id, 0, node.x / ARM_M * 12.0, node.y / ARM_M * 12.0,
                                     heading, 0 if edge.id < 2 else 1, edge.width, 0, edge.id))
    signs = [MapSign(0, 0, EDGES["W"], -9.0, 4.5, 0.0, kind="stop")] if stop else []
    return MapData("priority-check", "交差点の合成検証", 0.0, 0.0, ARM_M,
                   Bounds(-90.0, -90.0, 90.0, 90.0), nodes, edges, [], signals=signals, signs=signs)


def place(world: World, slot: int, arc: float, speed: float) -> None:
    """経路に沿って車の姿勢・弧長を設定する。"""
    state = world.slots[slot]
    arc = max(0.0, min(float(state.route_cum[-1]), arc))
    segment = max(0, min(len(state.route) - 2, int(np.searchsorted(state.route_cum, arc, side="right")) - 1))
    direction = state.route[segment + 1] - state.route[segment]
    world.fleet.x[slot] = np.interp(arc, state.route_cum, state.route[:, 0])
    world.fleet.y[slot] = np.interp(arc, state.route_cum, state.route[:, 1])
    world.fleet.heading[slot] = math.atan2(float(direction[1]), float(direction[0]))
    world.fleet.speed[slot] = speed
    world.fleet.active[slot] = True
    state.route_progress = segment
    state.arc_position = arc
    world.arc[slot] = arc


def route(world: World, slot: int, src: str, dst: str) -> None:
    plan = world._route_from_nodes(NODES[src], NODES[dst])
    assert plan is not None and world.install_route(slot, plan)
    state = world.slots[slot]
    state.intersections = tuple(world.map_index.intersections_on_route(state.route, plan.legs))
    assert len(world.slots[slot].intersections) == 1


def closed_loop(
    *, ego: tuple[str, str] = ("W", "S"), actors: tuple[Actor, ...] = (),
    signalized: bool = False, priority_axis: bool = False, stop: bool = False,
    pedestrian: bool = False, release_at: float = 3.0, seconds: float = 18.0,
    camera_missing: bool = False,
) -> Result:
    """判断の速度上限で自車を加減速し、相手は固定軌道で進める。"""
    world = World(build_map_index(junction_map(signalized=signalized, priority_axis=priority_axis, stop=stop)),
                  np.random.default_rng(134))
    route(world, 0, *ego)
    passage = world.slots[0].intersections[0]
    place(world, 0, passage.entry_arc - (12.0 if stop else 5.0), 3.0 if stop else 0.0)
    for slot, actor in enumerate(actors, 1):
        route(world, slot, actor.src, actor.dst)
        arc = passage.exit_arc + 4.0 if actor.blocker else world.slots[slot].intersections[0].entry_arc - actor.gap
        place(world, slot, arc, actor.speed)
    person = None
    if pedestrian:
        state = world.slots[0]
        at = passage.exit_arc + 2.0
        person = (float(np.interp(at, state.route_cum, state.route[:, 0])),
                  float(np.interp(at, state.route_cum, state.route[:, 1])))
    supervisor = IntersectionSupervisor()
    result = Result()
    clear_occlusion = OcclusionResult({}, [], 30.0, 30.0, 0.0, None,
                                      (1.0, 1.0, 1.0, 1.0), np.zeros(config.OBS_OCCLUSION_DIM))
    for step in range(int(seconds / config.DT)):
        now = step * config.DT
        world.sim_time = now
        world.signal_phases = [GREEN if light.group == 0 else RED for light in world.map_index.data.signals]
        world.set_player(person if pedestrian and now < release_at else None)
        for slot, actor in enumerate(actors, 1):
            if actor.blocker:
                world.fleet.active[slot] = now < release_at
            elif world.fleet.active[slot]:
                arc = float(world.arc[slot]) + actor.speed * config.DT
                if arc >= world.route_total[slot] - HALF_LENGTH:
                    world.fleet.active[slot] = False
                else:
                    place(world, slot, arc, actor.speed)
        camera_ready = camera_missing and now >= release_at
        front = {0: PerceptionResult(0)} if camera_ready else {}
        surround = {0: {key: PerceptionResult(0) for key in ("left", "right", "rear")}} if camera_ready else {}
        ages = {0: {"left": 0.0, "right": 0.0, "rear": 0.0}} if camera_ready else {}
        background = frozenset(range(len(actors) + 1)) if not camera_missing else frozenset(range(1, len(actors) + 1))
        supervisor.evaluate(world, now, front, surround, ages, {}, {0: clear_occlusion}, background=background)
        decision = supervisor.decisions.get(0)
        cap = decision.speed_cap if decision is not None else math.inf
        speed = float(world.fleet.speed[0])
        speed += np.clip(min(4.0, cap) - speed, -abs(config.MAX_DECEL) * config.DT, 2.5 * config.DT)
        arc = float(world.arc[0]) + max(0.0, speed) * config.DT
        place(world, 0, arc, max(0.0, speed))
        if result.entered_at is None and arc + HALF_LENGTH > passage.entry_arc:
            result.entered_at = now
        result.collided |= bool(world.check_collisions()[0])
        result.rows.append({
            "time": now, "arc": arc, "speed": speed,
            "state": decision.state if decision else "clear",
            "reason": decision.reason if decision else "",
            "cap": cap,
        })
        cleared = True
        for slot, actor in enumerate(actors, 1):
            if not world.fleet.active[slot]:
                continue
            if actor.blocker:
                cleared = False
                continue
            at = Point(float(world.fleet.x[slot]), float(world.fleet.y[slot]))
            conflicts = [conflict for conflict in passage.conflicts
                         if conflict.movement.entry_node == NODES[actor.src] and conflict.movement.exit_node == NODES[actor.dst]]
            if any(LineString(conflict.movement.points).project(at) - HALF_LENGTH <= conflict.other_end_arc for conflict in conflicts):
                cleared = False
        if (actors and cleared) or ((pedestrian or camera_missing) and now >= release_at):
            if result.cleared_at is None:
                result.cleared_at = now
            if result.start_after_clear is None and speed >= 0.5:
                result.start_after_clear = now - result.cleared_at
        if arc > passage.exit_arc + HALF_LENGTH + 3.0:
            result.passed = True
            break
    return result


def check_run(label: str, result: Result, reason: str) -> None:
    waiting = [row for row in result.rows if row["reason"] == reason]
    stopped = [row for row in waiting if row["speed"] < 0.15]
    restart = result.start_after_clear
    detail = f"待機 {len(stopped) * config.DT:.2f}秒 / 解放後の発進 {restart:.2f}秒" if restart is not None else "発進なし"
    check(f"{label}: 指定理由で待つ", bool(waiting) and bool(stopped), detail)
    check(f"{label}: 接触せず通過する", not result.collided and result.passed)
    check(f"{label}: 安全になってから3秒以内に発進する", restart is not None and restart <= 3.0, detail)


def integrated_physics() -> None:
    """実際の物理・認識・環境の操作上書きでも右折の譲りを確認する。"""
    env = SimulationEnv(build_map_index(junction_map(signalized=True)),
                        SimParams(vehicle_count=0, pedestrian_count=0, max_speed=8.0),
                        seed=134, perception_mode="oracle")
    env.autopilot_all = True
    env.world.signals._offsets[:] = 0.0
    route(env.world, 0, "W", "S")
    route(env.world, 1, "E", "W")
    for slot, gap, speed in ((0, 5.0, 0.0), (1, 15.0, 8.0)):
        place(env.world, slot, env.world.slots[slot].intersections[0].entry_arc - gap, speed)
    env.world.project_all()
    env._obs_stale = True
    passage = env.world.slots[0].intersections[0]
    actions = np.zeros((config.MAX_VEHICLES, config.ACTION_DIM), dtype=np.float32)
    passed, collided, imposed, stopped = False, False, 0, 0
    seconds = 0.0
    for step in range(int(15.0 / config.DT)):
        env.step(actions)
        seconds = (step + 1) * config.DT
        collided |= bool(env.world.collided_flags[0])
        decision = env.intersections.decisions.get(0)
        if decision is not None and decision.reason == "oncoming":
            imposed += int(decision.imposed)
            stopped += int(float(env.world.fleet.speed[0]) < 0.15)
        if float(env.world.arc[0]) > passage.exit_arc + HALF_LENGTH + 3.0:
            passed = True
            break
    check("実物理・カメラ認識: 対向車への譲りが操作へ適用される", imposed > 0 and stopped > 0,
          f"適用 {imposed}ステップ / 停止 {stopped * config.DT:.2f}秒")
    check("実物理・カメラ認識: 衝突せず右折を完了する", passed and not collided, f"通過 {seconds:.2f}秒")


def coordinated_passages(
    paths: tuple[tuple[str, str], ...], gap: float, *, camera: bool, signalized: bool,
) -> dict:
    """停止から発進する複数車の判断を同時に適用し、膠着と衝突を記録する。"""
    count = len(paths)
    index = build_map_index(junction_map(signalized=signalized))
    env = None
    if camera:
        env = SimulationEnv(index, SimParams(vehicle_count=0, pedestrian_count=0, max_speed=4.0),
                            seed=134, perception_mode="oracle")
        env.autopilot_all = True
        env.world.signals = SignalController(index.data.signals, green_sec=60.0)
        env.world.signals._offsets[:] = 0.0
        world = env.world
    else:
        world = World(index, np.random.default_rng(134))
    for slot, path in enumerate(paths):
        route(world, slot, *path)
    passages = [world.slots[slot].intersections[0] for slot in range(count)]
    for slot, passage in enumerate(passages):
        place(world, slot, passage.entry_arc - gap, 0.0)
    if env is not None:
        world.project_all()
        env._obs_stale = True
    supervisor = IntersectionSupervisor()
    actions = np.zeros((config.MAX_VEHICLES, config.ACTION_DIM), dtype=np.float32)
    collided, passed = False, np.zeros(count, dtype=bool)
    starts = np.zeros(count, dtype=np.int32)
    moving = np.zeros(count, dtype=bool)
    start_events: list[list[dict]] = [[] for _ in range(count)]
    low_speed_steps = np.zeros(count, dtype=np.int32)
    full_stop_steps = np.zeros(count, dtype=np.int32)
    minimum_speed = np.full(count, np.inf)
    last_reason = [""] * count
    trace = []
    for step in range(int(30.0 / config.DT)):
        if env is not None:
            env.step(actions)
            collided |= bool(np.any(world.collided_flags[:count]))
            decisions = env.intersections.decisions
        else:
            now = step * config.DT
            world.sim_time = now
            world.signal_phases = [GREEN if light.group == 0 else RED for light in index.data.signals]
            supervisor.evaluate(world, now, {}, {}, {}, {}, {}, background=frozenset(range(count)))
            decisions = supervisor.decisions
            updates = []
            for slot in range(count):
                decision = supervisor.decisions.get(slot)
                cap = decision.speed_cap if decision is not None else math.inf
                speed = float(world.fleet.speed[slot])
                speed += np.clip(min(4.0, cap) - speed, -abs(config.MAX_DECEL) * config.DT, 2.5 * config.DT)
                updates.append((float(world.arc[slot]) + max(0.0, speed) * config.DT, max(0.0, speed)))
            for slot, (arc, speed) in enumerate(updates):
                place(world, slot, arc, speed)
            collided |= bool(np.any(world.check_collisions()[:count]))
        if step % int(config.SIM_HZ) == 0:
            trace.append({
                "time": round((step + 1) * config.DT, 2),
                "arc": world.arc[:count].round(2).tolist(),
                "speed": world.fleet.speed[:count].round(2).tolist(),
                "reason": [decisions[slot].reason if slot in decisions else "" for slot in range(count)],
            })
        for slot, passage in enumerate(passages):
            speed = float(world.fleet.speed[slot])
            now_moving = speed >= 0.5
            minimum_speed[slot] = min(minimum_speed[slot], speed)
            low_speed_steps[slot] += int(speed < 0.5)
            full_stop_steps[slot] += int(speed < 0.15)
            reason = decisions[slot].reason if slot in decisions else ""
            if reason:
                last_reason[slot] = reason
            if now_moving and not moving[slot]:
                previous = start_events[slot][-1] if start_events[slot] else None
                start_events[slot].append({
                    "time": round((step + 1) * config.DT, 3),
                    "arc": round(float(world.arc[slot]), 3),
                    "initial": previous is None,
                    "reason": reason,
                    "last_hold_reason": last_reason[slot],
                    "safety": env.safety.assist(slot) if env is not None else "",
                    "interval_seconds": round((step + 1) * config.DT - previous["time"], 3) if previous else None,
                    "progress_m": round(float(world.arc[slot]) - previous["arc"], 3) if previous else None,
                    "below_half_mps_seconds": round(int(low_speed_steps[slot]) * config.DT, 3),
                    "fully_stopped_seconds": round(int(full_stop_steps[slot]) * config.DT, 3),
                    "min_speed_mps": round(float(minimum_speed[slot]), 3),
                })
                starts[slot] += 1
                minimum_speed[slot] = np.inf
                low_speed_steps[slot] = full_stop_steps[slot] = 0
            moving[slot] = now_moving
            passed[slot] |= float(world.arc[slot]) > passage.exit_arc + HALF_LENGTH + 3.0
        if passed.all():
            break
    return {"passed": bool(passed.all()), "collided": collided, "starts": starts,
            "elapsed": (step + 1) * config.DT, "arc": world.arc[:count].round(2).tolist(), "trace": trace,
            "start_events": start_events}


def simultaneous_right_turns(gap: float, *, camera: bool = False) -> None:
    """両方が停止状態から右折するとき、相互待機や発進停止の振動を起こさない。"""
    result = coordinated_passages((("W", "S"), ("E", "N")), gap, camera=camera, signalized=True)
    source = "カメラ・実物理" if camera else "背景車"
    detail = f"{result['elapsed']:.2f}秒 / 発進回数 {result['starts'].tolist()} / 弧長 {result['arc']}"
    check(f"{source}: 入口{gap:.0f}m手前の双方右折が膠着せず通過する", result["passed"] and not result["collided"], detail)
    check(f"{source}: 双方右折で発進停止を繰り返さない", bool(np.max(result["starts"]) <= 3), detail)


def four_way_straight(*, camera: bool = False) -> None:
    """同格の無信号十字路で、四方向の直進車が停止状態から順番に通過する。"""
    result = coordinated_passages((("W", "E"), ("E", "W"), ("N", "S"), ("S", "N")),
                                  5.0, camera=camera, signalized=False)
    source = "カメラ・実物理" if camera else "背景車"
    detail = f"{result['elapsed']:.2f}秒 / 発進回数 {result['starts'].tolist()} / 弧長 {result['arc']}"
    check(f"{source}: 四方向の直進車が停止から膠着せず通過する", result["passed"] and not result["collided"], detail)
    restarts = [event for events in result["start_events"] for event in events if not event["initial"]]
    stalled = [event for event in restarts if event["interval_seconds"] < 1.0 or event["progress_m"] < 1.0]
    minimum_interval = min((event["interval_seconds"] for event in restarts), default=math.inf)
    minimum_progress = min((event["progress_m"] for event in restarts), default=math.inf)
    check(f"{source}: 四方向の直進で同じ位置の発進停止を繰り返さない",
          bool(np.max(result["starts"]) <= 5) and not stalled,
          f"再発進の最短間隔 {minimum_interval:.2f}秒 / 最小進捗 {minimum_progress:.2f}m / {detail}")
    if not result["passed"] or result["collided"]:
        print(f"  経過: {result['trace'][:3] + result['trace'][-3:]}")
    if stalled:
        print(f"  小刻みな再発進: {stalled[:6]}")


def main() -> int:
    check_run("青信号の右折と対向直進", closed_loop(actors=(Actor("E", "W"),), signalized=True), "oncoming")
    check_run("右折と対向左折", closed_loop(actors=(Actor("E", "S"),), signalized=True), "oncoming")
    check_run("同格道路の左方優先", closed_loop(ego=("W", "E"), actors=(Actor("N", "S"),)), "priority")
    check_run("明らかに広い道路への譲り", closed_loop(ego=("S", "N"), actors=(Actor("E", "W"),), priority_axis=True), "priority")
    check_run("退出側の横断者", closed_loop(pedestrian=True), "pedestrian")
    check_run("出口の停車列", closed_loop(actors=(Actor("W", "S", speed=0.0, blocker=True),)), "exit_blocked")
    stopped = closed_loop(ego=("W", "E"), stop=True)
    full_stop = [row for row in stopped.rows if row["reason"] == "stop" and row["speed"] < 0.15]
    check("一時停止を1秒保ってから発進する", len(full_stop) * config.DT >= STOP_SEC - config.DT and stopped.passed and not stopped.collided,
          f"停止 {len(full_stop) * config.DT:.2f}秒")
    continuous = closed_loop(actors=(Actor("E", "W", gap=15.0), Actor("E", "W", gap=34.0), Actor("E", "W", gap=53.0)), signalized=True)
    check_run("連続対向車の解放", continuous, "oncoming")
    check("連続対向車が交錯区間を抜けるまで交差点へ入らない",
          continuous.entered_at is not None and continuous.cleared_at is not None and continuous.entered_at >= continuous.cleared_at)
    check_run("必要なカメラの観測不足と更新", closed_loop(camera_missing=True), "visibility")
    integrated_physics()
    for gap in (5.0, 10.0, 15.0):
        simultaneous_right_turns(gap)
        simultaneous_right_turns(gap, camera=True)
    four_way_straight()
    four_way_straight(camera=True)
    print(f"\n{'NG' if FAILURES else 'OK'}: {len(FAILURES)} 件の失敗")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
