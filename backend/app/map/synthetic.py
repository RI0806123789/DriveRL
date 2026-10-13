"""高さと勾配を明示した合成の立体コース（坂・バンク・高架とランプ・トンネル）。通信せずに作る。"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from app import config
from app.contracts import (
    ELEVATION_SYNTHETIC,
    STRUCTURE_BRIDGE,
    STRUCTURE_GROUND,
    STRUCTURE_TUNNEL,
    Bounds,
    MapBuilding,
    MapData,
    MapEdge,
    MapNode,
    MapPreset,
    TerrainGrid,
)

__all__ = ["SYNTHETIC_PRESET_IDS", "TERRAIN_LAB_ID", "build_synthetic_map", "is_synthetic"]

TERRAIN_LAB_ID = "terrain_lab"
SYNTHETIC_PRESET_IDS = frozenset({TERRAIN_LAB_ID})

#: 外周の道路の位置 [m]
OUTER_X = 360.0
OUTER_Y = 220.0
#: 山の中心・高さ・平らな頂上の半幅・裾までの長さ [m]
HILL_X = -100.0
HILL_HEIGHT_M = 10.0
HILL_FLAT_X = 40.0
HILL_FALL_X = 150.0
HILL_FLAT_Y = 70.0
HILL_FALL_Y = 120.0
#: 山を抜けるトンネルの道路の y と、坑口の x [m]
TUNNEL_Y = -40.0
TUNNEL_PORTALS = (-200.0, 0.0)
#: 高架の道路の x、橋桁の範囲（y）と高さ、ランプの裾（y）[m]
OVERPASS_X = 250.0
DECK_Y = (-60.0, 20.0)
DECK_Z = 6.5
RAMP_FOOT_Y = (-200.0, 160.0)
#: 高架の北のランプから大通りへ降りる接続ランプの分岐点（y）と合流点（x）
LINK_FROM_Y = 100.0
LINK_TO_X = 320.0
#: 北東の角のカーブ（中心・半径）と片勾配
CURVE_CENTER = (OUTER_X - 100.0, OUTER_Y - 100.0)
CURVE_RADIUS = 100.0
CURVE_CROSS_SLOPE = 0.06
#: ポリラインの頂点の間隔 [m]（高さの変化を滑らかに持たせる）
VERTEX_SPACING_M = 5.0
#: 地形の格子のセル [m]
TERRAIN_CELL_M = 2.0
TERRAIN_MARGIN_M = 40.0
ROAD_LANES = 2
ROAD_WIDTH = ROAD_LANES * config.DEFAULT_LANE_WIDTH
SPEED_40 = 40.0 / 3.6
SPEED_50 = 50.0 / 3.6


def is_synthetic(preset: MapPreset) -> bool:
    """通信せずにコードから作るプリセットか。"""
    return bool(preset.synthetic)


def _smooth_fall(r: np.ndarray, flat: float, fall: float) -> np.ndarray:
    """|r| が flat までは 1、そこから fall かけて余弦で 0 へ落ちる。"""
    a = np.clip((np.abs(r) - flat) / fall, 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(math.pi * a))


def hill_height(x, y) -> np.ndarray:
    """道路を削る前の地形の高さ [m]。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return HILL_HEIGHT_M * _smooth_fall(x - HILL_X, HILL_FLAT_X, HILL_FALL_X) * _smooth_fall(y, HILL_FLAT_Y, HILL_FALL_Y)


def _ramp(t: float, a: float, b: float, za: float, zb: float) -> float:
    if b == a:
        return zb
    u = min(max((t - a) / (b - a), 0.0), 1.0)
    return za + (zb - za) * u


def overpass_height(y: float) -> float:
    """高架の道路の路面の高さ [m]（裾 0m → 橋桁 `DECK_Z`）。"""
    lo, hi = DECK_Y
    south, north = RAMP_FOOT_Y
    if y <= lo:
        return _ramp(y, south, lo, 0.0, DECK_Z)
    if y >= hi:
        return _ramp(y, hi, north, DECK_Z, 0.0)
    return DECK_Z


class _Builder:
    def __init__(self) -> None:
        self.nodes: list[MapNode] = []
        self.edges: list[MapEdge] = []
        self._by_xy: dict[tuple[int, int], int] = {}

    def node(self, x: float, y: float, z: float) -> int:
        key = (int(round(x * 10)), int(round(y * 10)))
        found = self._by_xy.get(key)
        if found is not None:
            return found
        nid = len(self.nodes)
        self.nodes.append(MapNode(id=nid, x=float(x), y=float(y), z=float(z)))
        self._by_xy[key] = nid
        return nid

    def edge(
        self,
        points: list[tuple[float, float]],
        height: Callable[[float, float], float],
        *,
        structure: str = STRUCTURE_GROUND,
        speed: float = SPEED_40,
        cross: float = 0.0,
    ) -> None:
        pts = _densify(points)
        z = [float(height(px, py)) for px, py in pts]
        u = self.node(pts[0][0], pts[0][1], z[0])
        v = self.node(pts[-1][0], pts[-1][1], z[-1])
        pts[0] = (self.nodes[u].x, self.nodes[u].y)
        pts[-1] = (self.nodes[v].x, self.nodes[v].y)
        length = float(sum(math.dist(a, b) for a, b in zip(pts[:-1], pts[1:])))
        self.edges.append(
            MapEdge(
                id=len(self.edges),
                u=u,
                v=v,
                lanes=ROAD_LANES,
                width=float(ROAD_WIDTH),
                oneway=False,
                speed_limit=float(speed),
                polyline=pts,
                length=length,
                elevation=z,
                cross_slope=float(cross),
                structure=structure,
                layer=1 if structure == STRUCTURE_BRIDGE else (-1 if structure == STRUCTURE_TUNNEL else 0),
                elevation_source=ELEVATION_SYNTHETIC,
            )
        )

    def chain(self, stops: list[tuple[float, float]], height: Callable[[float, float], float], **kw) -> None:
        for a, b in zip(stops[:-1], stops[1:]):
            self.edge([a, b], height, **kw)


def _densify(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        length = math.dist(a, b)
        n = max(1, int(math.ceil(length / VERTEX_SPACING_M)))
        for i in range(1, n + 1):
            t = i / n
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


def _arc(center: tuple[float, float], radius: float, a0: float, a1: float) -> list[tuple[float, float]]:
    n = max(2, int(math.ceil(abs(a1 - a0) * radius / VERTEX_SPACING_M)) + 1)
    return [
        (center[0] + radius * math.cos(a0 + (a1 - a0) * i / (n - 1)), center[1] + radius * math.sin(a0 + (a1 - a0) * i / (n - 1)))
        for i in range(n)
    ]


def _flat(_x: float, _y: float) -> float:
    return 0.0


def _terrain(x: float, y: float) -> float:
    return float(hill_height(x, y))


def _build_roads() -> _Builder:
    b = _Builder()
    cx, cy = CURVE_CENTER
    # 外周（北東の角だけ片勾配のついたカーブ）
    b.chain([(-OUTER_X, -OUTER_Y), (OVERPASS_X, -OUTER_Y), (OUTER_X, -OUTER_Y)], _flat)
    b.chain([(OUTER_X, -OUTER_Y), (OUTER_X, TUNNEL_Y), (OUTER_X, 0.0), (OUTER_X, cy)], _flat)
    curve = _arc(CURVE_CENTER, CURVE_RADIUS, 0.0, math.pi / 2.0)
    b.edge(curve, _flat, cross=-CURVE_CROSS_SLOPE)
    b.chain([(cx, OUTER_Y), (OVERPASS_X, OUTER_Y), (-OUTER_X, OUTER_Y)], _flat)
    b.chain([(-OUTER_X, OUTER_Y), (-OUTER_X, 0.0), (-OUTER_X, TUNNEL_Y), (-OUTER_X, -OUTER_Y)], _flat)

    # 山を越える大通り（地形どおりの坂。頂上の先は見通せない）
    avenue = [-OUTER_X, -280.0, -200.0, -120.0, -40.0, 40.0, 120.0, 200.0, LINK_TO_X, OUTER_X]
    b.chain([(x, 0.0) for x in avenue], _terrain)

    # 山を抜けるトンネル（坑口の外は切通し）
    west, east = TUNNEL_PORTALS
    b.chain([(-OUTER_X, TUNNEL_Y), (-280.0, TUNNEL_Y), (west, TUNNEL_Y)], _flat, speed=SPEED_50)
    b.chain([(west, TUNNEL_Y), (-100.0, TUNNEL_Y), (east, TUNNEL_Y)], _flat, structure=STRUCTURE_TUNNEL, speed=SPEED_50)
    b.chain([(east, TUNNEL_Y), (120.0, TUNNEL_Y), (OUTER_X, TUNNEL_Y)], _flat, speed=SPEED_50)

    # 大通りとトンネルの道路をまたぐ高架（ランプは盛土、交差部は橋桁）
    def deck(_x: float, y: float) -> float:
        return overpass_height(y)

    south, north = RAMP_FOOT_Y
    lo, hi = DECK_Y
    b.chain([(OVERPASS_X, -OUTER_Y), (OVERPASS_X, south), (OVERPASS_X, lo)], deck)
    b.edge([(OVERPASS_X, lo), (OVERPASS_X, hi)], deck, structure=STRUCTURE_BRIDGE)
    b.chain([(OVERPASS_X, hi), (OVERPASS_X, LINK_FROM_Y), (OVERPASS_X, north), (OVERPASS_X, OUTER_Y)], deck)

    # 高架の北のランプから大通りへ降りる接続ランプ
    z_from = overpass_height(LINK_FROM_Y)
    start = (OVERPASS_X, LINK_FROM_Y)
    end = (LINK_TO_X, 0.0)
    total = math.dist(start, end)

    def link(x: float, y: float) -> float:
        return z_from * (1.0 - min(math.dist((x, y), start) / total, 1.0))

    b.edge([start, end], link)
    return b


def _carve_terrain(edges: list[MapEdge], cell: float, bounds: Bounds) -> TerrainGrid:
    """山の地形を作り、地上の道路の幅（と路肩）を路面の高さへ削る・盛る。"""
    from app.map.elevation import CARVE_MARGIN_M, EdgeSurface, cumulative_arc, stamp_edge

    ox = bounds.min_x - TERRAIN_MARGIN_M
    oy = bounds.min_y - TERRAIN_MARGIN_M
    w = int(math.ceil((bounds.max_x + TERRAIN_MARGIN_M - ox) / cell)) + 1
    h = int(math.ceil((bounds.max_y + TERRAIN_MARGIN_M - oy) / cell)) + 1
    gx = ox + np.arange(w, dtype=np.float64) * cell
    gy = oy + np.arange(h, dtype=np.float64) * cell
    px, py = np.meshgrid(gx, gy)
    z = hill_height(px, py)
    best = np.full((h, w), np.inf)
    carved = np.zeros((h, w))
    mask = np.zeros((h, w), dtype=bool)
    for e in edges:
        if e.structure != STRUCTURE_GROUND or e.elevation is None:
            continue
        pts = np.asarray(e.polyline, dtype=np.float64)
        surf = EdgeSurface(
            edge_id=e.id, pts=pts, cum=cumulative_arc(pts), z=np.asarray(e.elevation, dtype=np.float64),
            cross=float(e.cross_slope), half_width=float(e.width) * 0.5, structure=e.structure, layer=e.layer,
        )
        mask |= stamp_edge((h, w), (ox, oy), cell, surf, CARVE_MARGIN_M + cell, carved, best)
    z = np.where(mask, carved, z)
    return TerrainGrid(
        origin_x=float(ox), origin_y=float(oy), cell_size=float(cell), width=int(w), height=int(h),
        z=z.astype(np.float32), source=ELEVATION_SYNTHETIC,
    )


def _buildings(terrain: TerrainGrid) -> list[MapBuilding]:
    boxes = [
        (-330.0, 40.0, -290.0, 110.0, 12.0),
        (-330.0, -150.0, -280.0, -90.0, 18.0),
        (120.0, -170.0, 190.0, -110.0, 15.0),
        (40.0, 60.0, 90.0, 120.0, 9.0),
        (290.0, -170.0, 330.0, -100.0, 21.0),
    ]
    out: list[MapBuilding] = []
    for x0, y0, x1, y1, height in boxes:
        base = float(np.min(terrain.sample(np.array([x0, x1, x0, x1]), np.array([y0, y0, y1, y1]))))
        out.append(MapBuilding(id=len(out), height=height, outline=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)], base_z=base))
    return out


def build_synthetic_map(preset: MapPreset) -> MapData:
    """合成プリセットの `MapData` を作る。"""
    from app.map.loader import _build_signals, _build_speed_signs, _compute_bounds

    if preset.id != TERRAIN_LAB_ID:
        raise ValueError(f"合成の地図ではありません: {preset.id}")
    builder = _build_roads()
    nodes, edges = builder.nodes, builder.edges
    bounds = _compute_bounds(nodes, edges, [])
    terrain = _carve_terrain(edges, TERRAIN_CELL_M, bounds)
    buildings = _buildings(terrain)
    bounds = _compute_bounds(nodes, edges, buildings)
    node_z = {n.id: n.z for n in nodes}
    remap = {n.id: n.id for n in nodes}
    signals = _build_signals(set(), remap, edges, at_all_intersections=True)
    for sg in signals:
        sg.z = float(node_z.get(int(sg.node_id), 0.0))
    signs = _build_speed_signs(edges)
    edge_by_id = {e.id: e for e in edges}
    from app.map.elevation import cumulative_arc, project_polyline

    for sign in signs:
        e = edge_by_id[int(sign.edge_id)]
        pts = np.asarray(e.polyline, dtype=np.float64)
        cum = cumulative_arc(pts)
        s, _lat, _d = project_polyline(pts, cum, np.array([[sign.x, sign.y]]))
        sign.z = float(np.interp(s[0], cum, np.asarray(e.elevation, dtype=np.float64)))
    return MapData(
        preset_id=preset.id,
        name=preset.name,
        center_lat=preset.center_lat,
        center_lon=preset.center_lon,
        radius_m=preset.radius_m,
        bounds=bounds,
        nodes=nodes,
        edges=edges,
        buildings=buildings,
        signals=signals,
        signs=signs,
        terrain=terrain,
    )
