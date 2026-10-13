# -*- coding: utf-8 -*-
"""高低差・立体道路（#136）を合成の立体コースで閉ループに検証する。マップのキャッシュは読まない。"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for st in (sys.stdout, sys.stderr):
    try:
        st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np

from app import config
from app.contracts import SimParams
from app.map.elevation import GRAVITY, LEVEL_GAP_M
from app.map.index import build_map_index
from app.map.presets import get_preset
from app.map.synthetic import DECK_Z, OVERPASS_X, build_synthetic_map
from app.percep.camera import LBL_VEHICLE, LBL_VEHICLE_DARK, PseudoCamera
from app.percep.groundtruth import clear_static_cache, detect_ground_truth
from app.percep.types import DEFAULT_CAMERA, DetClass
from app.sim.dynamics import VehicleDynamics
from app.sim.env import SimulationEnv
from app.sim.scenario import standard_scenario
from app.sim.vehicle import VehicleFleet

FAILURES: list[str] = []
N = config.MAX_VEHICLES
ACT = np.zeros((N, config.ACTION_DIM), dtype=np.float32)


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if ok else 'NG  '}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def stop_distance(dynamics, grade: float, v0: float = 10.0) -> float:
    fleet = VehicleFleet(1, dynamics=dynamics)
    fleet.reset_slot(0, 0.0, 0.0, 0.0)
    fleet.active[0] = True
    fleet.speed[0] = v0
    fleet.slope_x[0] = grade
    for _ in range(2000):
        if fleet.speed[0] <= 0.01:
            break
        fleet.step(np.array([-1.0]), np.array([0.0]), config.DT, 30.0)
    return float(fleet.x[0])


def empty_env(index) -> SimulationEnv:
    params = SimParams()
    params.pedestrian_count = 0
    env = SimulationEnv(index, params, seed=136, compute_observations=False)
    env.world.set_pedestrian_count(0)
    for slot in range(N):
        env.world.deactivate(slot)
    return env


def main() -> int:
    preset = get_preset("terrain_lab")
    t0 = time.perf_counter()
    data = build_synthetic_map(preset)
    build_ms = (time.perf_counter() - t0) * 1000.0
    index = build_map_index(data)
    t0 = time.perf_counter()
    grids = index.solid_grids()
    grid_ms = (time.perf_counter() - t0) * 1000.0

    section("1. 合成の立体コース")
    kinds = sorted({e.structure for e in data.edges})
    check("道路構造（地上・橋桁・トンネル）がそろう", kinds == ["bridge", "ground", "tunnel"], f"{kinds}")
    check("地形の格子と固体判定グリッドがある", data.terrain is not None and grids is not None,
          f"構築 {build_ms:.0f}ms / グリッド {grid_ms:.0f}ms")
    check("橋桁の中は固体・桁下は空き", bool(grids.solid(OVERPASS_X, 0.0, DECK_Z - 0.5)) and not bool(grids.solid(OVERPASS_X, 0.0, 2.0)))
    check("トンネルの空洞は空き・上の地山は固体", not bool(grids.solid(-100.0, -40.0, 2.0)) and bool(grids.solid(-100.0, -40.0, 7.5)))

    section("2. 勾配の物理（10m/s から全制動の停止距離）")
    for name, dynamics in (("運動学", None), ("詳細物理", VehicleDynamics())):
        flat, up, down = stop_distance(dynamics, 0.0), stop_distance(dynamics, 0.08), stop_distance(dynamics, -0.08)
        theory = 6.0 / (6.0 + GRAVITY * 0.08 / math.sqrt(1.0064))
        check(f"{name}: 上り 8% < 平地 < 下り 8%", up < flat < down, f"{up:.2f} / {flat:.2f} / {down:.2f} m")
        check(f"{name}: 上りの比が理論に 3% 以内", abs(up / flat - theory) / theory < 0.03, f"{up / flat:.3f}（理論 {theory:.3f}）")

    section("3. 立体交差の上と下（車・障害物）")
    env = empty_env(index)
    world = env.world
    world.activate_at(0, (OVERPASS_X + 1.6, -5.0), clearance_m=1.0, heading=math.pi / 2, z=DECK_Z)
    world.activate_at(1, (OVERPASS_X, 1.6), clearance_m=1.0, heading=0.0, z=0.0)
    world.fleet.x[1] = world.fleet.x[0]
    world.fleet.y[1] = world.fleet.y[0] + 0.5
    world.project_all()
    check("橋桁の上と桁下で平面が重なっても衝突しない", not world.check_collisions()[:2].any(),
          f"z = {float(world.z[0]):.2f} / {float(world.z[1]):.2f}")
    world.add_obstacle(float(world.fleet.x[1]) + 0.5, float(world.fleet.y[1]), 0.5, z=0.0)
    check("同じ路面のパイロンには当たる", bool(world.check_collisions()[1]))

    section("4. 全車経路追従で走らせる（8 台・歩行者 16 人・100 秒）")
    params = SimParams()
    params.pedestrian_count = 16
    env = SimulationEnv(index, params, seed=136, scenario=standard_scenario(), compute_observations=False)
    env.autopilot_all = True
    world = env.world
    times: list[float] = []
    collisions = 0
    over_under = 0
    on_deck = 0
    in_tunnel = 0
    climbing = 0
    for _ in range(2000):
        t = time.perf_counter()
        result = env.step(ACT)
        times.append((time.perf_counter() - t) * 1000.0)
        collisions += sum(1 for e in result.episodes if e.reason == "collision")
        active = np.flatnonzero(world.fleet.active)
        on_deck += int(np.count_nonzero(world.z[active] > DECK_Z - 0.5))
        climbing += int(np.count_nonzero(world.pitch[active] > math.radians(3.0)))
        xs, ys, zs = world.fleet.x[active], world.fleet.y[active], world.z[active]
        in_tunnel += int(np.count_nonzero((np.abs(ys + 40.0) < 4.0) & (xs > -200.0) & (xs < 0.0)))
        for i in range(active.size):
            for j in range(i + 1, active.size):
                if math.hypot(float(xs[i] - xs[j]), float(ys[i] - ys[j])) < 4.0 and abs(float(zs[i] - zs[j])) > LEVEL_GAP_M:
                    over_under += 1
    arr = np.array(times[20:])
    check("車が橋桁の上・上り坂を走る", on_deck > 0 and climbing > 0,
          f"橋桁 {on_deck} / 坑内 {in_tunnel} / 上り {climbing}（台×ステップ）")
    check("高さと姿勢が有限", bool(np.all(np.isfinite(world.z))) and bool(np.all(np.isfinite(world.pitch))))
    print(f"  上下に重なった組 {over_under}（台の組×ステップ）・衝突で終わったエピソード {collisions}")
    print(f"  1 ステップ（認識なし）: 中央値 {np.median(arr):.2f}ms / p95 {np.percentile(arr, 95):.2f}ms / p99 {np.percentile(arr, 99):.2f}ms")

    section("4b. 坑内から東の坑口を抜ける（経路追従・1 台）")
    env = empty_env(index)
    world = env.world
    env.autopilot_all = True
    world.activate_at(0, (-180.0, -38.4), clearance_m=1.0, heading=0.0)
    heights = []
    exited = False
    for _ in range(600):
        result = env.step(ACT)
        if not world.fleet.active[0] or any(e.slot == 0 for e in result.episodes):
            break
        x, y = float(world.fleet.x[0]), float(world.fleet.y[0])
        if abs(y + 40.0) < 4.0 and x < 0.0:
            heights.append(float(world.z[0]))
        if x > 5.0:
            exited = True
            break
    check("坑内の路面（高さ 0）を走って坑口を抜ける", exited and bool(heights) and max(abs(h) for h in heights) < 0.3,
          f"坑内で {len(heights)} ステップ・最大の高さ {max((abs(h) for h in heights), default=0.0):.2f}m")

    section("5. 擬似カメラと正解ラベルが同じものを見る")
    scenes = [
        ("頂上の手前は見える", ((-70.0, 1.6), 0.0, None), ((-20.0, 1.6), 0.0, None), True),
        ("頂上の向こうは見えない", ((-70.0, 1.6), 0.0, None), ((20.0, 1.6), 0.0, None), False),
        ("桁下から橋の向こうが見える", ((225.0, 1.6), 0.0, 0.0), ((300.0, 1.6), 0.0, 0.0), True),
        ("坑内から山の上は見えない", ((-160.0, -38.4), 0.0, None), ((-100.0, 1.6), 0.0, None), False),
        ("坑内の前の車は見える", ((-160.0, -38.4), 0.0, None), ((-120.0, -38.4), 0.0, None), True),
    ]
    camera = PseudoCamera(index)
    for name, ego, other, visible in scenes:
        env = empty_env(index)
        clear_static_cache()
        env.world.activate_at(0, ego[0], clearance_m=1.0, heading=ego[1], z=ego[2])
        env.world.activate_at(1, other[0], clearance_m=1.0, heading=other[1], z=other[2])
        env.world.project_all()
        camera.render(env.world, np.array([0]))
        labels = camera.last_labels[0]
        pixels = int(((labels == LBL_VEHICLE) | (labels == LBL_VEHICLE_DARK)).sum())
        boxes = [d for d in detect_ground_truth(env.world, 0, DEFAULT_CAMERA).detections if d.cls == DetClass.VEHICLE]
        check(name, (len(boxes) == 1) is visible and (pixels > 0) is visible, f"正解ラベル {len(boxes)} 件・描いた画素 {pixels}")

    section("6. 認識つきの 1 ステップ（真値の検出・8 台・歩行者 16 人）")
    params = SimParams()
    params.pedestrian_count = 16
    env = SimulationEnv(index, params, seed=136, scenario=standard_scenario(), perception_mode="oracle")
    env.autopilot_all = True
    times = []
    for _ in range(220):
        t = time.perf_counter()
        env.step(ACT)
        times.append((time.perf_counter() - t) * 1000.0)
    arr = np.array(times[20:])
    print(f"  中央値 {np.median(arr):.2f}ms / p95 {np.percentile(arr, 95):.2f}ms / p99 {np.percentile(arr, 99):.2f}ms / 50ms 超過 {(arr > 50).mean() * 100:.1f}%")
    slots = np.flatnonzero(env.world.fleet.active)
    camera.render(env.world, slots)
    t = time.perf_counter()
    for _ in range(3):
        camera.render(env.world, slots)
    per_image = (time.perf_counter() - t) * 1000.0 / 3 / max(1, slots.size)
    print(f"  擬似カメラ（立体）1 枚あたり {per_image:.2f}ms（{slots.size} 台ぶんをまとめて描いた平均）")

    print()
    if FAILURES:
        print(f"NG: {len(FAILURES)} 項目 — " + " / ".join(FAILURES))
        return 1
    print("OK: すべて合格")
    return 0


if __name__ == "__main__":
    sys.exit(main())
