# -*- coding: utf-8 -*-
"""経路に載せる信号・標識が通る辺のものだけで、始点より後ろ・終点より先を含まず、出だしで後ろの赤信号に止められないかを検証する。"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for st in (sys.stdout, sys.stderr):
    try:
        st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


import numpy as np
from shapely.geometry import LineString, Point

from app import config
from app.contracts import SimParams
from app.map import build_map_index, get_preset, load_map
from app.map.index import ROUTE_END_EPS_M
from app.map.lanes import lane_offset_left
from app.map.loader import MapLoadError
from app.map.presets import list_presets
from app.sim.env import SimulationEnv
from app.sim.signals import GREEN, RED
from app.sim.world import Route

FAILURES: list[str] = []
#: 金沢は 1 本が重いので本数を減らす
TRIALS = {"kanazawa": 100}
TRIALS_DEFAULT = 200
CASES = 8
SPAWN_TRIES = 400
POSES = 40
#: 停止線からずらす距離 [m]。越えた直後（配車の作り直し）と、赤で止まっている位置
PAST_LINE_M = 3.0
BEFORE_LINE_M = 1.5
#: 規制速度を比べるとき、交差点からこれだけ離れた経路点だけを見る [m]（標識は交差点を出た先に立つ）
LIMIT_CHECK_AWAY_M = 25.0


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if ok else 'NG  '}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def along(pts: np.ndarray, xy: tuple[float, float], at_end: bool) -> float:
    """始点（終点）から、最初（最後）の区間の向きに測った地物の位置 [m]。"""
    a, b = (pts[-2], pts[-1]) if at_end else (pts[0], pts[1])
    d = (b - a) / max(float(np.hypot(*(b - a))), 1e-9)
    o = b if at_end else a
    return float((xy[0] - o[0]) * d[0] + (xy[1] - o[1]) * d[1])


def outside_ends(pts: np.ndarray, found: list[tuple[float, tuple[float, float]]]) -> int:
    """弧長が端にある地物のうち、始点より後ろ・終点より先にあるものの数。"""
    total = float(np.hypot(*np.diff(pts, axis=0).T).sum())
    bad = 0
    for arc, xy in found:
        if arc < 0.5 and along(pts, xy, False) < -ROUTE_END_EPS_M:
            bad += 1
        elif arc > total - 0.5 and along(pts, xy, True) > ROUTE_END_EPS_M:
            bad += 1
    return bad


def behind_start(index, pts: np.ndarray) -> int:
    """始点より後ろにあり、経路の向きを向いた信号（旧実装なら弧長 0 に載る）。無ければ -1。"""
    xy = index._signal_xy
    if xy.shape[0] == 0:
        return -1
    u = (pts[1] - pts[0]) / max(float(np.hypot(*(pts[1] - pts[0]))), 1e-9)
    h = math.atan2(u[1], u[0])
    d = np.hypot(xy[:, 0] - pts[0, 0], xy[:, 1] - pts[0, 1])
    a = (xy[:, 0] - pts[0, 0]) * u[0] + (xy[:, 1] - pts[0, 1]) * u[1]
    sh = index._signal_heading
    diff = np.abs(np.arctan2(np.sin(sh - h), np.cos(sh - h)))
    ok = np.flatnonzero((d <= 11.0) & (a <= -0.5) & (diff <= math.radians(35.0)))
    return int(ok[np.argmin(d[ok])]) if ok.size else -1


def limit_mismatches(index, route: Route) -> tuple[int, int]:
    """交差点から離れた経路点で、表示される規制速度と走っている道の規制が食い違う数と、見た点の数。"""
    pts = np.asarray(route.points, dtype=np.float64)
    limits = index.speed_limits_on_route([(float(x), float(y)) for x, y in pts], route.legs)
    if not limits:
        return 0, 0
    arcs = np.array([a for a, _v in limits])
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    nodes = index._node_xy[
        sorted({lg.entry_node for lg in route.legs} | {lg.exit_node for lg in route.legs})
    ]
    near = ((pts[:, None, :] - nodes[None, :, :]) ** 2).sum(axis=-1).min(axis=1)
    bad = seen = 0
    for k, leg in enumerate(route.legs):
        edge = index._edges_by_id[leg.edge_id]
        lo = leg.start_arc
        hi = route.legs[k + 1].start_arc if k + 1 < len(route.legs) else float(cum[-1])
        for i in np.flatnonzero((cum >= lo) & (cum < hi) & (near > LIMIT_CHECK_AWAY_M**2)):
            shown = limits[int(np.searchsorted(arcs, cum[i], side="right")) - 1][1]
            seen += 1
            bad += int(abs(shown - edge.speed_limit) > 1e-6)
    return bad, seen


def off_route_signals(index, route: Route, found: list[tuple[float, int]]) -> int:
    """経路が通らない進入路の灯器の数（通る辺の出口に立つ灯器だけを載せる）。"""
    hops = {(lg.entry_node, lg.exit_node) for lg in route.legs}
    bad = 0
    for _arc, i in found:
        s = index.data.signals[i]
        e = index._edges_by_id[int(s.edge_id)]
        upstream = int(e.u) if int(e.v) == int(s.node_id) else int(e.v)
        bad += int((upstream, int(s.node_id)) not in hops)
    return bad


def force_phases(world, red: set[int]) -> None:
    """指定した信号だけを赤、ほかを青に固定する（地図の癖に左右されずに判定するため）。"""
    forced = [RED if i in red else GREEN for i in range(len(world.map_index.data.signals))]
    world.signals.phases = lambda _t, *, demand=None: forced
    world.signal_phases = forced


def same_phase(data, s) -> set[int]:
    """その灯器と同時に赤になる灯器（同じ交差点・同じ群）。"""
    return {
        k for k, t in enumerate(data.signals) if t.phase_key == s.phase_key and t.group == s.group
    }


def starts_on_approach(index, route: Route, s) -> bool:
    """経路がその信号の進入路から始まるか。近くを並んで走る別の道から始まる経路は、その信号に従わない。"""
    if not route.legs:
        return False
    e = index._edges_by_id[int(s.edge_id)]
    upstream = int(e.u) if int(e.v) == int(s.node_id) else int(e.v)
    first = route.legs[0]
    return (first.entry_node, first.exit_node) == (upstream, int(s.node_id))


def release_phases(world) -> None:
    world.signals.__dict__.pop("phases", None)
    world.signal_phases = world.signals.phases(world.sim_time)


def drive(env, slot: int, steps: int) -> float:
    """アクセル 1.0 で走らせ、出た最高速度 [m/s] を返す。"""
    act = np.zeros((config.MAX_VEHICLES, config.ACTION_DIM), dtype=np.float32)
    act[slot, 0] = 1.0
    top = 0.0
    for _ in range(steps):
        env.step(act)
        top = max(top, float(env.world.fleet.speed[slot]))
    return top


def consistent(world, slot: int) -> bool:
    """据え付けた直後、速度の上限が見る「前方の信号」と信号無視の判定が見る「まだ越えていない信号」が同じか。"""
    state = world.slots[slot]
    arc = float(world.arc[slot])
    ahead = int(np.searchsorted(state.signal_arcs, arc, side="left"))
    passed = int(
        np.searchsorted(state.signal_arcs, arc - config.SIGNAL_STOP_TOLERANCE_M, side="right")
    )
    return state.signals_passed == passed and ahead >= state.signals_passed


def approach_pose(index, s, shift: float) -> tuple[float, float, float] | None:
    """信号の停止線から進行方向へ `shift` [m] ずらした、左端の車線の上の姿勢。"""
    e = index._edges_by_id.get(int(s.edge_id))
    if e is None or len(e.polyline) < 2:
        return None
    pts = [tuple(p) for p in e.polyline]
    if int(e.v) != int(s.node_id):
        pts.reverse()
    line = LineString(pts)
    d = line.project(Point(s.x, s.y)) + shift
    if not (0.5 <= d <= line.length - 0.5):
        return None
    a = line.interpolate(max(0.0, d - 0.5))
    b = line.interpolate(min(line.length, d + 0.5))
    h = math.atan2(b.y - a.y, b.x - a.x)
    c = line.interpolate(d)
    off = lane_offset_left(e, 0)
    return c.x - math.sin(h) * off, c.y + math.cos(h) * off, h


def spawn_cases(world, index, rng, routes: list[Route]):
    """再スポーンの経路。足りなければ信号のある交差点を始点にして作る（金沢は始点に信号が少ない）。"""
    yield from routes
    nodes = [int(s.node_id) for s in index.data.signals if index._reachable_mask[int(s.node_id)]]
    for _ in range(SPAWN_TRIES if nodes else 0):
        src = nodes[int(rng.integers(0, len(nodes)))]
        x, y = (float(v) for v in index._node_xy[src])
        dst = int(index.nearest_node(*far_node(index, rng, x, y)))
        r = world._route_from_nodes(src, dst)
        if r is not None:
            yield r


def far_node(index, rng, x: float, y: float) -> tuple[float, float]:
    reach, nodes = index._reachable, index._node_xy
    dst = int(reach[0])
    for _ in range(30):
        dst = int(reach[int(rng.integers(0, reach.size))])
        if 150.0 <= math.hypot(nodes[dst, 0] - x, nodes[dst, 1] - y) <= 1500.0:
            break
    return float(nodes[dst, 0]), float(nodes[dst, 1])


targets = sys.argv[1:] or [p.id for p in list_presets()]

for preset_id in targets:
    preset = get_preset(preset_id)
    if preset is None:
        print(f"未知のプリセット: {preset_id}")
        sys.exit(2)
    print("=" * 72)
    print(f"{preset.id}（{preset.name}）")
    print("=" * 72)
    try:
        data = load_map(preset)
    except MapLoadError as exc:
        print(f"  マップを読めません（{exc}）。prefetch してから実行すること")
        sys.exit(2)

    index = build_map_index(data)
    env = SimulationEnv(
        index, SimParams(vehicle_count=1, pedestrian_count=0), seed=3, compute_observations=False
    )
    world = env.world
    slot = int(np.flatnonzero(world.fleet.active)[0])
    rng = np.random.default_rng(5)
    trials = TRIALS.get(preset.id, TRIALS_DEFAULT)
    edges = [e for e in data.edges if len(e.polyline) >= 2 and e.length > 5.0]

    routes: list[Route] = []
    for _ in range(trials):
        r = world._random_route()
        if r is not None:
            routes.append(r)
    spawned = len(routes)
    weights = np.array([e.length for e in edges], dtype=np.float64)
    for _ in range(trials):
        e = edges[int(rng.choice(len(edges), p=weights / weights.sum()))]
        pts = [tuple(p) for p in e.polyline]
        if not e.oneway and rng.random() < 0.5:
            pts.reverse()
        line = LineString(pts)
        d = float(rng.uniform(0.1, 0.9)) * line.length
        a, b = line.interpolate(max(0.0, d - 0.5)), line.interpolate(min(line.length, d + 0.5))
        h = math.atan2(b.y - a.y, b.x - a.x)
        c = line.interpolate(d)
        off = lane_offset_left(e, 0)
        x, y = c.x - math.sin(h) * off, c.y + math.cos(h) * off
        speed = float(rng.uniform(0.0, 13.9))
        r = world.route_between((x, y), far_node(index, rng, x, y), heading=h, speed=speed)
        if r is not None:
            routes.append(r)

    bad_sig = had_behind = off_route = start_bad = limit_bad = limit_seen = 0
    for r in routes:
        pts = np.asarray(r.points, dtype=np.float64)
        route = [(float(px), float(py)) for px, py in pts]
        sig = index.signals_on_route(route, r.legs)
        bad_sig += outside_ends(pts, [(a, (data.signals[i].x, data.signals[i].y)) for a, i in sig])
        had_behind += int(behind_start(index, pts) >= 0)
        off_route += off_route_signals(index, r, sig)

        limits = index.speed_limits_on_route(route, r.legs)
        first = index._edges_by_id[r.legs[0].edge_id].speed_limit if r.legs else None
        start_bad += int(first is None or not limits or abs(limits[0][1] - first) > 1e-6)
        bad, seen = limit_mismatches(index, r)
        limit_bad += bad
        limit_seen += seen
    check(
        "経路に載る信号に、始点より後ろ・終点より先のものが無い",
        bad_sig == 0,
        f"再スポーン {spawned} 本・配車 {len(routes) - spawned} 本のうち {bad_sig} 基"
        f"（始点の後ろに経路の向きの信号がある経路は {had_behind} 本）",
    )
    check(
        "経路に載る信号は、経路が通る進入路の灯器だけ",
        off_route == 0,
        f"通らない進入路の灯器 {off_route} 基",
    )
    check(
        "始点の規制速度は始点の道路のもので、交差する道や後ろの標識に引きずられない",
        start_bad == 0,
        f"{len(routes)} 本のうち {start_bad} 本",
    )
    check(
        f"交差点から {LIMIT_CHECK_AWAY_M:.0f}m 以上離れた所で、規制速度が走っている道のものと一致する",
        limit_bad == 0 and limit_seen > 0,
        f"{limit_seen} 点のうち {limit_bad} 点が食い違う",
    )

    # 置いた位置で当たってもエピソードを閉じさせない（再スポーンした別の車を測らないため）
    env.commandeered_slot = slot
    cases = 0
    held = 0
    inconsistent = 0
    for r in spawn_cases(world, index, rng, routes[:spawned]):
        if cases >= CASES:
            break
        sig = behind_start(index, np.asarray(r.points, dtype=np.float64))
        if sig < 0:
            continue
        cases += 1
        force_phases(world, {sig})
        world.install_route(slot, r)
        inconsistent += int(not consistent(world, slot))
        if drive(env, slot, 20) <= 0.5:
            held += 1
        release_phases(world)
    check(
        "再スポーンの直後、後ろの停止線の赤信号に止められない（1 秒で 0.5m/s を超える）",
        cases > 0 and held == 0,
        f"{cases} 件のうち止められた {held} 件（後ろの信号だけ赤・ほかは青に固定）",
    )

    past_cases = past_listed = past_held = 0
    before_cases = before_crossed = before_skipped = 0
    order = rng.permutation(len(data.signals))
    for i in order:
        if past_cases >= POSES and before_cases >= POSES:
            break
        s = data.signals[int(i)]
        if not index._reachable_mask[int(s.node_id)]:
            continue
        for shift, kind in ((PAST_LINE_M, "past"), (-BEFORE_LINE_M, "before")):
            if (past_cases if kind == "past" else before_cases) >= POSES:
                continue
            pose = approach_pose(index, s, shift)
            if pose is None:
                continue
            x, y, h = pose
            route = world.route_between((x, y), far_node(index, rng, x, y), heading=h, speed=0.0)
            if route is None:
                continue
            if kind == "before" and not starts_on_approach(index, route, s):
                before_skipped += 1
                continue
            world.fleet.reset_slot(slot, x, y, h)
            if kind == "past":
                force_phases(world, {int(i)})
            else:
                force_phases(world, same_phase(data, s))
            world.install_route(slot, route, keep_pose=True)
            inconsistent += int(not consistent(world, slot))
            if kind == "past":
                past_cases += 1
                past_listed += int(int(i) in set(int(k) for k in world.slots[slot].signal_ids))
                past_held += int(drive(env, slot, 20) <= 0.5)
            else:
                before_cases += 1
                drive(env, slot, 100)
                crossed = (
                    (float(world.fleet.x[slot]) - s.x) * math.cos(s.heading)
                    + (float(world.fleet.y[slot]) - s.y) * math.sin(s.heading)
                ) > config.SIGNAL_STOP_TOLERANCE_M
                before_crossed += int(crossed)
            release_phases(world)
    env.commandeered_slot = -1
    check(
        f"停止線を {PAST_LINE_M:.0f}m 越えた所から経路を作り直しても、越えた信号を載せず止められない",
        past_cases > 0 and past_listed == 0 and past_held == 0,
        f"{past_cases} 件のうち載った {past_listed} 件・止められた {past_held} 件",
    )
    check(
        f"停止線の {BEFORE_LINE_M:.1f}m 手前から経路を作り直しても、赤なら停止線を越えない",
        before_cases > 0 and before_crossed == 0,
        f"{before_cases} 件のうち越えた {before_crossed} 件（同じ交差点・同じ群の灯器を赤に固定。"
        f"並んで走る別の道から経路が始まった {before_skipped} 件は対象外）",
    )
    check(
        "経路を据え付けた直後、速度の上限と信号無視の判定が同じ信号を見る",
        inconsistent == 0,
        f"{inconsistent} 件が食い違う",
    )

print("=" * 72)
if FAILURES:
    print(f"結果: {len(FAILURES)} 件の不合格")
    sys.exit(1)
print("結果: すべて合格")
