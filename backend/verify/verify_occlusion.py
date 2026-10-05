# -*- coding: utf-8 -*-
"""カメラだけから作る見通しと死角と、見通しの悪い交差点の顔出し（クリープ）を検証する。マップのキャッシュは読まない（建物つきの合成の交差点で走らせる）。"""
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
from app.contracts import BRANCH_LEFT, BRANCH_RIGHT, Bounds, MapBuilding, MapData, MapEdge, MapNode, SimParams
from app.map.index import build_map_index
from app.percep.encoder import OBS_OFFSETS
from app.sim.env import SimulationEnv
from app.sim.safety import (
    ASSIST_CREEP,
    CREEP_PAST_M,
    CREEP_SPEED_MPS,
    LOS_OPEN_M,
)

FAILURES: list[str] = []
N = config.MAX_VEHICLES
ACT = np.zeros((N, config.ACTION_DIM), dtype=np.float32)
HALF = config.VEHICLE_LENGTH * 0.5
ARM_M = 200.0
#: 交差点の四隅の建物の内側の角（道路の中心線から）[m]。道幅 7m なので路端から 1m
CORNER_M = 4.5
OCC = OBS_OFFSETS["occlusion"]
LEAVES = {"S": (0.0, -ARM_M), "N": (0.0, ARM_M), "E": (ARM_M, 0.0), "W": (-ARM_M, 0.0)}


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if ok else 'NG  '}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def junction_map(arms: str, quadrants: str) -> MapData:
    """原点の交差点から `arms`（S/N/E/W）へ 200m の道が伸びる地図。`quadrants`（SW/SE/NW/NE を空白区切り）の角に建物を置く。"""
    nodes = [MapNode(0, 0.0, 0.0)]
    ids: dict[str, int] = {}
    for arm in arms:
        ids[arm] = len(nodes)
        nodes.append(MapNode(ids[arm], *LEAVES[arm]))
    edges: list[MapEdge] = []
    for arm in arms:
        a, b = nodes[0], nodes[ids[arm]]
        for u, v in ((a, b), (b, a)):
            edges.append(MapEdge(len(edges), u.id, v.id, 1, 7.0, False, 11.1, [(u.x, u.y), (v.x, v.y)], ARM_M))
    buildings = []
    far = ARM_M * 0.8
    for quad in quadrants.split():
        sx = -1.0 if "W" in quad else 1.0
        sy = -1.0 if "S" in quad else 1.0
        xs = sorted((sx * CORNER_M, sx * far))
        ys = sorted((sy * CORNER_M, sy * far))
        buildings.append(
            MapBuilding(len(buildings), 9.0, [(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])])
        )
    edge = ARM_M + 20.0
    return MapData("junction", "junction", 0.0, 0.0, ARM_M, Bounds(-edge, -edge, edge, edge), nodes, edges, buildings)


def drive_through(arms: str, quadrants: str, src: str = "S", dst: str = "N", seconds: float = 60.0) -> dict:
    """1 台を src → dst へ経路追従で走らせ、交差点の前後の様子を記録する。"""
    data = junction_map(arms, quadrants)
    index = build_map_index(data)
    params = SimParams()
    params.vehicle_count = 1
    params.pedestrian_count = 0
    env = SimulationEnv(index, params, seed=0)
    env._ensure_percep()
    env._detector = None
    env._camera = None
    env.autopilot_all = True
    ids = {arm: i + 1 for i, arm in enumerate(arms)}
    route = env.world._route_from_nodes(ids[src], ids[dst])
    assert route is not None, "合成の地図で経路が作れない"
    env.world.install_route(0, route)
    env.watched_occlusion = frozenset({0})
    rows = []
    collided = False
    passed = False
    junction_arc = math.nan
    for _ in range(int(seconds * config.SIM_HZ)):
        result = env.step(ACT)
        collided |= bool(env.world.collided_flags[0]) if hasattr(env.world, "collided_flags") else False
        gap, sides = env.world.junction_ahead(0, CREEP_PAST_M)
        arc = float(env.world.arc[0])
        if math.isfinite(gap):
            junction_arc = arc + gap
        occ = env.latest_occlusion.get(0)
        rows.append(
            {
                "arc": arc,
                "gap": gap - HALF if math.isfinite(gap) else math.inf,
                "sides": sides,
                "speed": abs(float(env.world.fleet.speed[0])),
                "assist": env.safety.assist(0),
                "los": None if occ is None else (occ.los_left, occ.los_right),
                "obs": result.obs[0, OCC : OCC + config.OBS_OCCLUSION_DIM].copy(),
            }
        )
        if math.isfinite(junction_arc) and arc > junction_arc + 15.0:
            passed = True
            break
    return {"rows": rows, "collided": collided, "passed": passed, "env": env}


# ---------------------------------------------------------------------------
section("1. 四隅を建物で囲んだ十字路（見通しが悪い）")
blind = drive_through("SNEW", "SW SE NW NE")
rows = blind["rows"]
creep = [r for r in rows if r["assist"] == ASSIST_CREEP]
check("交差点を通り抜け、ぶつからない", blind["passed"] and not blind["collided"], f"{len(rows) * config.DT:.1f} 秒")
check("交差点の手前で顔出し（creep）になる", len(creep) > 0, f"{len(creep) * config.DT:.1f} 秒")
at_entry = [r["speed"] for r in rows if -1.0 <= r["gap"] <= 0.5]
check(
    f"入口では クリープ速度（{CREEP_SPEED_MPS}m/s）付近まで落ちる",
    bool(at_entry) and max(at_entry) <= CREEP_SPEED_MPS + 1.0 and min(at_entry) >= CREEP_SPEED_MPS - 0.6,
    f"入口 ±1m の速さ {min(at_entry, default=math.nan):.2f}〜{max(at_entry, default=math.nan):.2f} m/s",
)
before = [r for r in rows if r["assist"] == ASSIST_CREEP and r["los"] is not None]
check(
    "顔出しの間は、左右どちらかの見通しが開けていない（20m 未満）",
    all(min(r["los"]) < LOS_OPEN_M for r in before[:5]),
    f"最初の見通し {before[0]['los'] if before else None}",
)
release = next((i for i in range(1, len(rows)) if rows[i - 1]["assist"] == ASSIST_CREEP and rows[i]["assist"] != ASSIST_CREEP), None)
if release is None:
    check("見通しが開けると顔出しを解く", False)
else:
    r = rows[release]
    later = rows[release : release + int(3.0 * config.SIM_HZ)]
    check(
        "見通しが開けると顔出しを解く（入口を 2m 越える前に）",
        r["los"] is not None and min(r["los"]) >= LOS_OPEN_M and r["gap"] > -(CREEP_PAST_M + HALF),
        f"解いたときの見通し {r['los']} / バンパーから入口 {r['gap']:.1f}m",
    )
    check(
        "解いてから 3 秒で再加速する",
        max(x["speed"] for x in later) >= CREEP_SPEED_MPS + 1.5,
        f"{r['speed']:.2f} → 最大 {max(x['speed'] for x in later):.2f} m/s",
    )
blind_obs = [r["obs"] for r in rows if r["gap"] > 3.0 and math.isfinite(r["gap"])]
check(
    "建物の脇では、観測の左の見通しと左の見えている割合が小さい",
    bool(blind_obs) and float(np.median([o[0] for o in blind_obs])) < 0.2 and float(np.median([o[6] for o in blind_obs])) < 0.1,
    f"左の見通し {np.median([o[0] for o in blind_obs]):.2f} / 左の割合 {np.median([o[6] for o in blind_obs]):.3f}",
)
open_obs = rows[-1]["obs"]
check("交差点を抜けた先の観測は 0..1 に収まる", bool(np.all((open_obs >= 0.0) & (open_obs <= 1.0))))
frame = blind["env"].snapshot(0, 0.0).to_wire()
check(
    "watch_occlusion で頼んだ車の見通しと死角が frame に載る",
    "occlusion" in frame and "0" in frame["occlusion"] and frame["occlusion"]["0"]["cameras"],
)
blind["env"].watched_occlusion = frozenset()
check("頼まれていなければ載せない", "occlusion" not in blind["env"].snapshot(0, 0.0).to_wire())

# ---------------------------------------------------------------------------
section("2. 建物の無い十字路（見通しが良い）")
clear = drive_through("SNEW", "")
rows = clear["rows"]
check("交差点を通り抜け、ぶつからない", clear["passed"] and not clear["collided"])
check("顔出しにならない", not any(r["assist"] == ASSIST_CREEP for r in rows))
at_entry = [r["speed"] for r in rows if -1.0 <= r["gap"] <= 0.5]
check("入口は徐行（4m/s）のまま通る", bool(at_entry) and min(at_entry) >= 3.0, f"最小 {min(at_entry, default=math.nan):.2f} m/s")

# ---------------------------------------------------------------------------
section("3. 丁字路: 道が無い左側だけ建物が迫っている（右から来る道は見通せる）")
tee = drive_through("SNE", "SW NW")
rows = tee["rows"]
sides = {r["sides"] for r in rows if math.isfinite(r["gap"])}
check("交差点の左右から来る道は右だけ", sides == {BRANCH_RIGHT}, f"{sides}")
check("左の見通しは短いが、道の無い側なので顔出しにならない", not any(r["assist"] == ASSIST_CREEP for r in rows))
left_los = [r["los"][0] for r in rows if r["los"] is not None and 0.0 < r["gap"] < 8.0]
check("（左の見通しは実際に短い）", bool(left_los) and max(left_los) < LOS_OPEN_M, f"最大 {max(left_los, default=math.nan):.1f}m")

section("4. 丁字路: 右から来る道の角に建物がある")
tee_blind = drive_through("SNE", "SE NE")
rows = tee_blind["rows"]
check("交差点の左右から来る道は右だけ", {r["sides"] for r in rows if math.isfinite(r["gap"])} == {BRANCH_RIGHT})
check("右の角が見えないので顔出しになる", any(r["assist"] == ASSIST_CREEP for r in rows))
check("交差点を通り抜け、ぶつからない", tee_blind["passed"] and not tee_blind["collided"])
check("（左右とも道がある十字路の左右の値）", BRANCH_LEFT | BRANCH_RIGHT == 3)

# ---------------------------------------------------------------------------
section("5. 所要時間（8 台・死角の計算だけ）")
env = blind["env"]
env.params.vehicle_count = N
env.world.set_active_count(N)
env.reset_all()
idx = np.flatnonzero(env.world.fleet.active)
times = []
for _ in range(200):
    started = time.perf_counter()
    env._update_occlusion(idx)
    times.append(time.perf_counter() - started)
per_car = float(np.median(times)) * 1000.0 / max(int(idx.size), 1)
check("死角の計算は 1 台あたり 1ms 未満（issue の予算）", per_car < 1.0, f"中央値 {per_car:.3f} ms/台（{idx.size} 台）")

print()
if FAILURES:
    print(f"NG: {len(FAILURES)} 件")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("OK: すべての検査に通りました")
