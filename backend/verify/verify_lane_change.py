"""実カメラと車体物理で、回避・復帰と実介入の契約を検証する。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app import config
from app.contracts import Bounds, MapData, MapEdge, MapNode, SimParams
from app.map import build_map_index
from app.sim.dynamics import VehicleDynamics
from app.sim.env import SimulationEnv
from app.sim.scenario import RoadEvent, Scenario
from app.sim.world import Route


def run(*, dynamic: bool, safety: bool, expert: bool) -> dict:
    data = MapData("lane_change", "合成2車線", 0, 0, 300, Bounds(-20, -20, 320, 20), [MapNode(0, 0, 0), MapNode(1, 300, 0)], [MapEdge(0, 0, 1, 2, 7, True, 13.9, [(0, 0), (300, 0)], 300, True)], [])
    index = build_map_index(data)
    scenario = Scenario(dynamics=VehicleDynamics(), learner_vehicles=1, background_vehicles=0) if dynamic else None
    env = SimulationEnv(index, SimParams(vehicle_count=1, pedestrian_count=0, safety_assist=safety, incident_curriculum=False), seed=135, scenario=scenario, perception_mode="oracle")
    points, legs = index.lane_route([0, 1])
    env.world.install_route(0, Route(np.asarray(points, dtype=np.float32), tuple(legs)))
    env.world.fleet.active[0] = True
    env.world.fleet.x[0], env.world.fleet.y[0], env.world.fleet.speed[0] = 20, 1.75, 4
    env.world.project_all()
    env.world.add_obstacle(70, 1.75, 0.9)
    if not safety and not expert:
        env.commandeered_slot = 0
        env.autopilot_all = True
    env._obs_stale = True
    zero = np.zeros((config.MAX_VEHICLES, config.ACTION_DIM), dtype=np.float32)
    teacher = np.zeros(config.MAX_VEHICLES, dtype=bool)
    teacher[0] = expert
    states = set()
    clearance = float("inf")
    maximum_accel = maximum_jerk = maximum_steer_rate = 0.0
    previous_steer = 0.0
    interventions = 0
    for k in range(900):
        result = env.step(zero, expert=teacher if expert else None)
        assert not env.world.collided_flags[0], "接触しました"
        corners = env.world.fleet.corners()[0]
        assert np.all(np.abs(corners[:, 1]) <= 3.5 + 1e-4), "車体が道路外へ出ました"
        decision = env.lane_changes.decisions.get(0)
        if decision is not None:
            states.add(decision.state)
            maximum_accel = max(maximum_accel, abs(decision.lateral_accel))
            maximum_jerk = max(maximum_jerk, abs(decision.lateral_jerk))
            if decision.imposed:
                interventions += 1
                if safety:
                    assert not result.learn[0] and not result.assisted[0], "実介入を方策の成績に混ぜました"
        if expert:
            assert result.assisted[0] and result.learn[0], "お手本の学習マスクを壊しました"
        steer = float(env.world.fleet.steer[0])
        maximum_steer_rate = max(maximum_steer_rate, abs(steer - previous_steer) / config.DT)
        previous_steer = steer
        distance = np.linalg.norm(corners - [70, 1.75], axis=1).min() - 0.9
        clearance = min(clearance, float(distance))
        if decision is not None and decision.completed:
            assert {"changing", "passing", "returning", "complete"} <= states
            assert abs(float(env.world.fleet.y[0]) - 1.75) < 0.3
            assert maximum_accel < 2.5 and maximum_steer_rate <= config.STEER_RATE + 1e-4
            if safety:
                assert interventions > 0
            return {"dynamic": dynamic, "safety": safety, "expert": expert, "seconds": round(env.sim_time, 2), "states": sorted(states), "cornerClearanceM": round(clearance, 3), "maxLateralAccel": round(maximum_accel, 3), "maxLateralJerk": round(maximum_jerk, 3), "maxSteerRate": round(maximum_steer_rate, 3), "interventions": interventions, "completed": decision.completed}
    raise AssertionError("止まったままで回避・復帰が完了していません")


def closure_run(*, reopen: bool) -> dict:
    coordinates = [(0, 0), (100, 0), (100, 100), (200, 0)]
    edges = [MapEdge(i, a, b, 2, 6.5, False, 13.9, [coordinates[a], coordinates[b]], float(np.linalg.norm(np.subtract(coordinates[a], coordinates[b])))) for i, (a, b) in enumerate(((0, 1), (1, 3), (1, 2), (2, 3)))]
    index = build_map_index(MapData("closure", "合成迂回", 0, 0, 220, Bounds(-15, -15, 215, 115), [MapNode(i, *xy) for i, xy in enumerate(coordinates)], edges, []))
    events = (RoadEvent(kind="closure", edge_id=1, start_sec=1, duration_sec=3 if reopen else 120),)
    if reopen:
        events += (RoadEvent(kind="closure", edge_id=2, start_sec=1, duration_sec=3),)
    env = SimulationEnv(index, SimParams(vehicle_count=1, pedestrian_count=0, incident_curriculum=False), seed=135, scenario=Scenario(learner_vehicles=1, background_vehicles=0, events=events), perception_mode="oracle")
    points, legs = index.lane_route([0, 1, 3])
    env.world.install_route(0, Route(np.asarray(points, dtype=np.float32), tuple(legs)))
    env.world.fleet.active[0] = True
    env.world.fleet.x[0], env.world.fleet.y[0] = 20, 1.625
    env.world.project_all()
    env.commandeered_slot = 0
    env.world.set_stop_target(0, float(env.world.route_total[0]))
    env._obs_stale = True
    seen = set()
    route_changed = False
    for _ in range(1800):
        env.step(np.zeros((config.MAX_VEHICLES, config.ACTION_DIM), dtype=np.float32))
        assert not env.world.collided_flags[0], "閉鎖の迂回中に接触しました"
        decision = env.road_closures.decisions.get(0)
        if decision is not None:
            seen.add(decision.reason or decision.state)
        route_changed |= any(leg.edge_id == 2 for leg in env.world.slots[0].route_legs)
        if env.world.arc[0] >= env.world.route_total[0] - 2.0 and env.world.fleet.speed[0] < 0.2:
            assert "blocked" in seen
            if reopen:
                assert "no_detour" in seen and not index.closed_edges
            else:
                assert route_changed and all(leg.edge_id != 1 for leg in env.world.slots[0].route_legs)
            return {"closureReopen": reopen, "seconds": round(env.sim_time, 2), "reasons": sorted(seen), "detour": route_changed}
    raise AssertionError("閉鎖後に目的地へ進めませんでした")


if __name__ == "__main__":
    results = [run(dynamic=False, safety=False, expert=False), run(dynamic=True, safety=False, expert=False), run(dynamic=False, safety=True, expert=False), run(dynamic=False, safety=False, expert=True)]
    results += [closure_run(reopen=False), closure_run(reopen=True)]
    print(json.dumps(results, ensure_ascii=False, indent=2))
