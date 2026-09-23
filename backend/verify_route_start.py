# -*- coding: utf-8 -*-
"""配車の経路が車の位置から道なりに出て、建物を突き抜けないかを、プリセットのマップで検証する。"""
from __future__ import annotations

import gc
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
for st in (sys.stdout, sys.stderr):
    try:
        st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


import networkx as nx
import numpy as np
from shapely.geometry import LineString

from app import config
from app.map import build_map_index, get_preset, load_map
from app.map.lanes import lane_offset_left
from app.map.loader import MapLoadError
from app.map.presets import list_presets
from app.sim.world import LEAD_DECEL_MPS2, LEAD_MIN_M, World

FAILURES: list[str] = []
TRIALS = 400
#: これ以上建物の中を通っていたら「突き抜けた」とみなす [m]。角をかすめるだけのものは除く
PIERCE_M = 1.0
TOP_SPEED = 13.9


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if ok else 'NG  '}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def inside_buildings(index, pts: np.ndarray) -> float:
    """点列が建物の中を通る長さ [m]。"""
    if pts.shape[0] < 2 or index._building_tree is None:
        return 0.0
    line = LineString(pts)
    total = 0.0
    for h in np.atleast_1d(index._building_tree.query(line, predicate="intersects")):
        total += line.intersection(index._building_polys[int(h)]).length
    return total


def pose_on_road(rng, edges) -> tuple[float, float, float]:
    """道路上の車線の中に、車線に沿った向きで置く（走っている車と同じ状態）。"""
    weights = np.array([e.length for e in edges], dtype=np.float64)
    e = edges[int(rng.choice(len(edges), p=weights / weights.sum()))]
    pts = [tuple(p) for p in e.polyline]
    if not e.oneway and rng.random() < 0.5:
        pts.reverse()
    line = LineString(pts)
    d = float(rng.uniform(0.1, 0.9)) * line.length
    a = line.interpolate(max(0.0, d - 0.5))
    b = line.interpolate(min(line.length, d + 0.5))
    h = math.atan2(b.y - a.y, b.x - a.x)
    c = line.interpolate(d)
    off = lane_offset_left(e, 0)
    return c.x - math.sin(h) * off, c.y + math.cos(h) * off, h


def route_edges(index, lead, path) -> list[tuple[int, bool | None]]:
    """経路が通るエッジと向き（u→v なら True。道のりの区間は向きを問わず None）。"""
    out: list[tuple[int, bool | None]] = []
    if lead is not None:
        out += [(int(e), None) for e in lead.edge_ids]
    for a, b in zip(path[:-1], path[1:]):
        eid = int(index.graph[a][b]["edge_id"])
        out.append((eid, int(index._edges_by_id[eid].u) == int(a)))
    return out


def uses_overlap(index, lead, path, bad: set[int]) -> bool:
    return any(e in bad for e, _fwd in route_edges(index, lead, path))


_NARROW: dict[tuple[int, bool], bool] = {}


def lane_hits_building(index, edge_id: int, forward: bool) -> bool:
    """その向きの左端の車線の中心が、建物の中を 1m 以上通るか（狭い道に広い道幅を見込んでいる）。"""
    key = (edge_id, forward)
    if key not in _NARROW:
        e = index._edges_by_id[edge_id]
        pts = [tuple(p) for p in e.polyline]
        if not forward:
            pts.reverse()
        lane = LineString(pts).offset_curve(lane_offset_left(e, 0))
        _NARROW[key] = (not lane.is_empty) and inside_buildings(
            index, np.asarray(lane.coords, dtype=np.float64)
        ) >= PIERCE_M
    return _NARROW[key]


def uses_narrow(index, lead, path) -> bool:
    for e, fwd in route_edges(index, lead, path):
        dirs = (True, False) if fwd is None else (fwd,)
        if any(lane_hits_building(index, e, d) for d in dirs):
            return True
    return False


def avoidable(index, path, bad: set[int]) -> bool:
    """建物と重なるエッジを通らずに同じ 2 点を結べるか。"""
    drop = [
        (u, v) for u, v, d in index.graph.edges(data=True) if int(d["edge_id"]) in bad
    ]
    view = nx.restricted_view(index.graph, [], drop)
    return nx.has_path(view, path[0], path[-1])


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
    t0 = time.perf_counter()
    overlap = index._edge_building_overlap()
    overlap_ms = (time.perf_counter() - t0) * 1000
    bad = set(overlap)
    check(
        "建物の中を通る道路の判定が 1 秒未満（build_map_index の中で 1 回）",
        overlap_ms < 1000.0,
        f"{overlap_ms:.0f}ms・該当 {len(bad)} 本（中の長さ 合計 {sum(overlap.values()):.0f}m）",
    )

    world = World(index, np.random.default_rng(11))
    rng = np.random.default_rng(11)
    mask = index._reachable_mask
    reach = index._reachable
    nodes = index._node_xy
    edges = [
        e for e in data.edges
        if len(e.polyline) >= 2 and e.length > 5.0 and e.id not in bad and mask[e.u] and mask[e.v]
    ]

    made = failed = far_start = bent = backward = long_step = 0
    pierced = by_overlap = by_narrow = unexplained = 0
    short_lead = uturn = 0
    times: list[float] = []
    for _ in range(TRIALS):
        x, y, h = pose_on_road(rng, edges)
        speed = float(rng.uniform(0.0, TOP_SPEED))
        for _k in range(30):
            dst = int(reach[int(rng.integers(0, reach.size))])
            if 150.0 <= math.hypot(nodes[dst, 0] - x, nodes[dst, 1] - y) <= 1500.0:
                break
        target = (float(nodes[dst, 0]), float(nodes[dst, 1]))
        gc.disable()
        t0 = time.perf_counter()
        route = world.route_between((x, y), target, heading=h, speed=speed)
        times.append((time.perf_counter() - t0) * 1000)
        gc.enable()
        if route is None:
            failed += 1
            continue
        made += 1
        pts = np.asarray(route, dtype=np.float64)

        if math.hypot(pts[0, 0] - x, pts[0, 1] - y) > 4.0:
            far_start += 1
        seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        k = int(np.searchsorted(cum, 2.0))
        k = min(max(k, 1), pts.shape[0] - 1)
        rh = math.atan2(pts[k, 1] - pts[0, 1], pts[k, 0] - pts[0, 0])
        mismatch = abs(math.degrees(math.atan2(math.sin(rh - h), math.cos(rh - h))))
        bent += int(mismatch > 20.0)
        backward += int(mismatch > 90.0)
        head = seg[: int(np.searchsorted(cum, 60.0))]
        if head.size and float(head.max()) > config.ROUTE_RESAMPLE_M * 1.5:
            long_step += 1

        need = LEAD_MIN_M + speed * speed / (2.0 * LEAD_DECEL_MPS2)
        lead = index.road_lead(x, y, h, need)
        dst_node = int(index.nearest_node(*target))
        path = index.shortest_path(lead.exit_node, dst_node, arrive_heading=lead.exit_heading)
        if inside_buildings(index, np.vstack([[x, y], pts])) >= PIERCE_M:
            pierced += 1
            if uses_overlap(index, lead, path, bad):
                by_overlap += 1
            elif uses_narrow(index, lead, path):
                by_narrow += 1
            else:
                unexplained += 1
        if lead.length < need and len(lead.edge_ids) <= 8:
            seen = set(int(e) for e in lead.edge_ids)
            facing = math.atan2(
                lead.polylines[-1][-1][1] - lead.polylines[-1][-2][1],
                lead.polylines[-1][-1][0] - lead.polylines[-1][-2][0],
            ) if len(lead.polylines[-1]) >= 2 else h
            if index._straight_on(lead.exit_node, facing, seen) is not None:
                short_lead += 1
        out = lead.exit_node
        if len(path) >= 2 and index._turns_back(out, index.graph[out][path[1]], lead.exit_heading):
            sharp = [
                (out, v) for v in index.graph.successors(out)
                if index._turns_back(out, index.graph[out][v], lead.exit_heading)
            ]
            if nx.has_path(nx.restricted_view(index.graph, [], sharp), out, dst_node):
                uturn += 1

    check("配車の経路が作れる（道路上を走っている車から）", failed <= TRIALS * 0.01, f"{made} 本・失敗 {failed}")
    check("経路は車の位置から始まる（4m 以内）", far_start == 0, f"{far_start} 本が離れている")
    check(
        "経路の出だし 2m が車の向きから 20° を超えてずれるのは 1% 未満（曲がるしかない交差点の直前を除けない）",
        bent < made * 0.01,
        f"{bent} 本",
    )
    check("経路が後ろ向き（車の向きから 90° 超）に始まらない", backward == 0, f"{backward} 本")
    check(
        f"出だし 60m に道路を外れた直線が無い（点の間隔が {config.ROUTE_RESAMPLE_M * 1.5:.0f}m 以下）",
        long_step == 0,
        f"{long_step} 本",
    )
    check(
        f"建物を {PIERCE_M:.0f}m 以上突き抜けるのは、下の 2 つで説明できるものだけ",
        unexplained == 0,
        f"突き抜ける {pierced} 本（建物と重なる道路 {by_overlap}・狭い道の車線 {by_narrow}・"
        f"説明できない {unexplained}）",
    )
    check(
        "速度が出ているときは、まっすぐ抜けられる限り曲がり始めまでの距離を取る",
        short_lead == 0,
        f"{short_lead} 本が手前で曲がる",
    )
    check("出口で、ほかに道があるのに 120° を超えて折り返さない", uturn == 0, f"{uturn} 本")
    ms = np.array(times)
    check(
        "配車の経路 1 本の作成が 1 ステップの予算（50ms）に収まり、95% は 35ms 未満（GC を止めて測る）",
        float(ms.max()) < 50.0 and float(np.percentile(ms, 95)) < 35.0,
        f"中央値 {np.median(ms):.1f}ms・95% {np.percentile(ms, 95):.1f}ms・最大 {ms.max():.1f}ms",
    )

    detour = checked = 0
    for _ in range(TRIALS):
        src, dst = index.random_node_pair(rng, 150.0, 1500.0)
        path = index.shortest_path(int(src), int(dst))
        if not path or len(path) < 2:
            continue
        checked += 1
        if uses_overlap(index, None, path, bad) and avoidable(index, path, bad):
            detour += 1
    check(
        "再スポーンの経路は、建物と重なる道路を避けられるなら避ける",
        detour == 0,
        f"{checked} 本のうち避けられるのに通った {detour} 本",
    )

print("=" * 72)
if FAILURES:
    print(f"結果: {len(FAILURES)} 件の不合格")
    sys.exit(1)
print("結果: すべて合格")
