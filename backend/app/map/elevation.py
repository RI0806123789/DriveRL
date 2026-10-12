"""道路の高さ・勾配・横断勾配と、立体の地図の固体判定グリッド（路面モデルの唯一の実装）。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app import config
from app.contracts import (
    MAX_CROSS_SLOPE,
    STRUCTURE_BRIDGE,
    STRUCTURE_TUNNEL,
    MapData,
    MapEdge,
    RouteLeg,
)
from app.warn import warn_once

__all__ = [
    "DECK_THICKNESS_M",
    "GRADE_SPAN_M",
    "LEVEL_GAP_M",
    "TUNNEL_CLEARANCE_M",
    "EdgeSurface",
    "RoadSurface",
    "SolidGrids",
    "attitude_from_slopes",
    "gravity_components",
    "vertical_overlap",
]

GRAVITY = 9.81
#: 橋桁の厚さ [m]。実地図の橋桁の寸法は分からないので、合成の地図と描画の見込みにだけ使う
DECK_THICKNESS_M = 1.2
#: トンネルの内空の高さ [m]（路面から天井の下面まで）。合成の地図の値で、実地図の寸法としては使わない
TUNNEL_CLEARANCE_M = 5.0
#: 天井の上にこれだけ土かぶりが無ければ、トンネルではなく切通し（空が見える）とみなす [m]
TUNNEL_MIN_COVER_M = 0.5
#: 縦断勾配を測る幅 [m]（ホイールベースより少し長い。経路点の 2m 刻みの段差を均す）
GRADE_SPAN_M = 4.0
#: 高さがこれ以上離れた物体は別の路面にいるとみなす [m]（立体交差の桁下は 4.5m 以上ある）
LEVEL_GAP_M = 3.0
#: 固体判定グリッドのセル [m]
SOLID_CELL_M = 1.0
#: 道路の幅に足して、地形を路面の高さへ削る・盛る幅 [m]（路肩）
CARVE_MARGIN_M = 1.5
#: 橋桁・トンネルの幅に足す余裕 [m]
STRUCTURE_MARGIN_M = 0.5
#: トンネルの壁とみなす、坑内の外側の幅 [セル]
WALL_CELLS = 2
#: 固体判定グリッドを作る上限のセル数（これより広い立体の地図は描画と見通しを平面のまま扱う）
SOLID_MAX_CELLS = 16_000_000


def vertical_overlap(a_lo, a_hi, b_lo, b_hi):
    """高さの範囲 [a_lo, a_hi] と [b_lo, b_hi] が重なるか。スカラーでも配列でもよい。"""
    return (a_lo < b_hi) & (b_lo < a_hi)


def attitude_from_slopes(s_x, s_y):
    """車体の前後・左右の路面勾配の比から (ピッチ, ロール) [rad] を返す。回転は Rz(方位)·Ry(-ピッチ)·Rx(ロール)。"""
    s_x = np.asarray(s_x, dtype=np.float64)
    s_y = np.asarray(s_y, dtype=np.float64)
    norm = np.sqrt(1.0 + s_x * s_x + s_y * s_y)
    return np.arctan(s_x), np.arcsin(np.clip(s_y / norm, -1.0, 1.0))


def gravity_components(s_x, s_y):
    """路面の勾配の比から、車体の前後・左右・法線方向の重力の成分 [m/s²] を返す（法線は押し付けの大きさで正）。"""
    s_x = np.asarray(s_x, dtype=np.float64)
    s_y = np.asarray(s_y, dtype=np.float64)
    norm = np.sqrt(1.0 + s_x * s_x + s_y * s_y)
    root_x = np.sqrt(1.0 + s_x * s_x)
    return -GRAVITY * s_x / root_x, -GRAVITY * s_y / (norm * root_x), GRAVITY / norm


def cumulative_arc(pts: np.ndarray) -> np.ndarray:
    seg = np.diff(pts, axis=0)
    out = np.zeros(pts.shape[0], dtype=np.float64)
    np.cumsum(np.hypot(seg[:, 0], seg[:, 1]), out=out[1:])
    return out


def project_polyline(pts: np.ndarray, cum: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """点 q (P, 2) をポリライン pts (S, 2) へ射影し、(弧長, 左を正とする横位置, 距離) を返す。"""
    q = np.asarray(q, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2 or q.shape[0] == 0:
        zero = np.zeros(q.shape[0], dtype=np.float64)
        return zero, zero.copy(), np.full(q.shape[0], np.inf)
    a = pts[:-1]
    ab = pts[1:] - a
    len2 = np.maximum((ab * ab).sum(axis=1), 1e-12)
    rel = q[:, None, :] - a[None, :, :]
    t = np.clip((rel * ab[None]).sum(axis=2) / len2[None], 0.0, 1.0)
    foot = a[None] + ab[None] * t[..., None]
    d2 = ((q[:, None, :] - foot) ** 2).sum(axis=2)
    best = np.argmin(d2, axis=1)
    rows = np.arange(q.shape[0])
    seg_len = np.sqrt(len2[best])
    s = cum[best] + t[rows, best] * seg_len
    ux = ab[best, 0] / seg_len
    uy = ab[best, 1] / seg_len
    rx = q[:, 0] - foot[rows, best, 0]
    ry = q[:, 1] - foot[rows, best, 1]
    lateral = -uy * rx + ux * ry
    return s, lateral, np.sqrt(d2[rows, best])


@dataclass(frozen=True)
class EdgeSurface:
    """1 本の辺の中心線の高さと横断勾配。"""

    edge_id: int
    pts: np.ndarray
    cum: np.ndarray
    z: np.ndarray
    cross: float
    half_width: float
    structure: str
    layer: int

    def z_at(self, s: np.ndarray, lateral: np.ndarray | float = 0.0) -> np.ndarray:
        """弧長 s・横位置 lateral（u→v の左が正）の路面の高さ。"""
        return np.interp(np.asarray(s, dtype=np.float64), self.cum, self.z) + np.asarray(lateral) * self.cross


def edge_heights(edge: MapEdge, node_z: dict[int, float]) -> np.ndarray | None:
    """polyline の頂点ごとの高さ。平面（端点もどちらも 0）なら None。"""
    n = len(edge.polyline)
    if edge.elevation is not None:
        z = np.asarray(edge.elevation, dtype=np.float64)
        if z.shape != (n,) or not np.all(np.isfinite(z)):
            warn_once(
                "map.elevation.bad_profile",
                "辺の高さの点数がポリラインと合わないか、有限でない値を含みます。その辺は端点の高さを結んで扱います",
            )
        else:
            return z
    zu = float(node_z.get(int(edge.u), 0.0))
    zv = float(node_z.get(int(edge.v), 0.0))
    if zu == 0.0 and zv == 0.0:
        return None
    pts = np.asarray(edge.polyline, dtype=np.float64)
    cum = cumulative_arc(pts)
    total = float(cum[-1]) if cum.size else 0.0
    if total <= 1e-9:
        return np.full(n, zu, dtype=np.float64)
    return zu + (zv - zu) * cum / total


class RoadSurface:
    """道路の路面の高さを引く。平面の地図では `elevated` が偽で、どの高さも 0 を返す。"""

    def __init__(self, data: MapData) -> None:
        self.data = data
        self.terrain = data.terrain
        self.edges: dict[int, EdgeSurface] = {}
        node_z = {int(n.id): float(n.z) for n in data.nodes}
        self.node_z = node_z
        self.elevated = bool(data.elevated)
        self._grids: SolidGrids | None = None
        self._grids_failed = False
        if not self.elevated:
            return
        for edge in data.edges:
            if len(edge.polyline) < 2:
                continue
            pts = np.asarray(edge.polyline, dtype=np.float64)
            z = edge_heights(edge, node_z)
            if z is None:
                z = np.zeros(pts.shape[0], dtype=np.float64)
            cross = float(np.clip(edge.cross_slope, -MAX_CROSS_SLOPE, MAX_CROSS_SLOPE))
            if cross != edge.cross_slope:
                warn_once("map.elevation.cross_slope", "横断勾配が上限を超えています。上限で切って扱います")
            self.edges[int(edge.id)] = EdgeSurface(
                edge_id=int(edge.id),
                pts=pts,
                cum=cumulative_arc(pts),
                z=z,
                cross=cross,
                half_width=float(edge.width) * 0.5,
                structure=edge.structure,
                layer=int(edge.layer),
            )

    def ground_z(self, x, y) -> np.ndarray:
        """道路以外の地面の高さ。地形が無ければ 0。"""
        if self.terrain is None:
            return np.zeros(np.shape(np.asarray(x)), dtype=np.float64)
        return self.terrain.sample(x, y)

    def edge_surface_z(self, edge_id: int, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """点 q (P, 2) をその辺の路面へ射影した高さと、中心線からの距離。"""
        surf = self.edges.get(int(edge_id))
        q = np.asarray(q, dtype=np.float64).reshape(-1, 2)
        if surf is None:
            return np.zeros(q.shape[0], dtype=np.float64), np.full(q.shape[0], np.inf)
        s, lateral, dist = project_polyline(surf.pts, surf.cum, q)
        return surf.z_at(s, lateral), dist

    def route_profile(
        self, points: np.ndarray, cum: np.ndarray, legs: Sequence[RouteLeg]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """経路点ごとの (路面の高さ, 進行方向の左が上がる横断勾配, 縦断勾配) を返す。平面の地図ではすべて 0。"""
        k = int(points.shape[0])
        z = np.zeros(k, dtype=np.float64)
        cross = np.zeros(k, dtype=np.float64)
        if not self.elevated or k == 0:
            return z.astype(np.float32), cross.astype(np.float32), z.astype(np.float32)
        pts = np.asarray(points, dtype=np.float64)
        arc = np.asarray(cum, dtype=np.float64)
        covered = np.zeros(k, dtype=bool)
        for leg in legs:
            surf = self.edges.get(int(leg.edge_id))
            if surf is None:
                continue
            idx = np.flatnonzero((arc >= float(leg.start_arc) - 1e-6) & (arc <= float(leg.end_arc) + 1e-6))
            if idx.size == 0:
                continue
            s, lateral, _ = project_polyline(surf.pts, surf.cum, pts[idx])
            z[idx] = surf.z_at(s, lateral)
            edge = self._edge(int(leg.edge_id))
            forward = edge is None or int(leg.entry_node) == int(edge.u)
            cross[idx] = surf.cross if forward else -surf.cross
            covered[idx] = True
        if not covered.any():
            z = self.ground_z(pts[:, 0], pts[:, 1]).astype(np.float64)
        elif not covered.all():
            # 交差点のつなぎは、前後の車線の高さを弧長で結ぶ
            gap = ~covered
            z[gap] = np.interp(arc[gap], arc[covered], z[covered])
            cross[gap] = np.interp(arc[gap], arc[covered], cross[covered])
        grade = route_grade(arc, z)
        return z.astype(np.float32), cross.astype(np.float32), grade.astype(np.float32)

    def _edge(self, edge_id: int) -> MapEdge | None:
        table = getattr(self, "_edge_table", None)
        if table is None:
            table = {int(e.id): e for e in self.data.edges}
            self._edge_table = table
        return table.get(edge_id)

    def grids(self) -> SolidGrids | None:
        """描画と見通しが使う固体判定グリッド。平面の地図・広すぎる地図では None。"""
        if not self.elevated or self._grids_failed:
            return None
        if self._grids is None:
            try:
                self._grids = build_solid_grids(self)
            except Exception:
                self._grids_failed = True
                warn_once(
                    "map.elevation.solid_grids",
                    "立体の地図の固体判定グリッドを作れませんでした。擬似カメラと見通しは平面として扱います",
                )
                return None
        return self._grids


def route_grade(arc: np.ndarray, z: np.ndarray) -> np.ndarray:
    """経路の弧長と高さから、`GRADE_SPAN_M` の幅で測った縦断勾配の比。"""
    if arc.size < 2 or float(arc[-1]) <= 1e-9:
        return np.zeros_like(z)
    half = GRADE_SPAN_M * 0.5
    total = float(arc[-1])
    lo = np.clip(arc - half, 0.0, total)
    hi = np.clip(arc + half, 0.0, total)
    span = np.maximum(hi - lo, 1e-6)
    return (np.interp(hi, arc, z) - np.interp(lo, arc, z)) / span


MARK_CAVITY = 1
MARK_SLAB = 2
MARK_WALL = 4


@dataclass
class SolidGrids:
    """立体の地図の固体の範囲。セル (行, 列) の中心は origin + 添字 × cell（`OccupancyGrid` と同じ取り方）。"""

    origin_x: float
    origin_y: float
    cell: float
    width: int
    height: int
    #: 地面の上端（地形を道路の路面へ削り・盛りした高さ。トンネルの上は山の表面）
    floor: np.ndarray
    #: 橋桁の下端・上端（無いセルは +inf / -inf）
    slab_lo: np.ndarray
    slab_hi: np.ndarray
    #: トンネルの空洞の下端（坑内の路面）・上端（天井の下面）。無いセルは +inf / -inf
    cavity_lo: np.ndarray
    cavity_hi: np.ndarray
    #: トンネルの壁（坑内のすぐ外の地山）と、壁として塗る高さの上限（となりの天井の高さ。無いセルは -inf）
    wall: np.ndarray
    wall_top: np.ndarray
    #: セルの印（1 = トンネルの空洞・2 = 橋桁・4 = トンネルの壁）。描画で構造物の無い所を飛ばすのに使う
    marks: np.ndarray | None = None
    #: 建物の占有（`OccupancyGrid.building` と同じ格子のときだけ持つ。見通し判定で添字を共有する）
    building: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.marks is None:
            marks = np.zeros((self.height, self.width), dtype=np.uint8)
            marks[np.isfinite(self.cavity_lo)] |= MARK_CAVITY
            marks[np.isfinite(self.slab_lo)] |= MARK_SLAB
            marks[self.wall] |= MARK_WALL
            self.marks = marks

    def cell_index(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """ENU 座標を (行, 列) の添字にする。範囲外は端に寄せる。"""
        col = np.rint((np.asarray(x, dtype=np.float64) - self.origin_x) / self.cell).astype(np.int64)
        row = np.rint((np.asarray(y, dtype=np.float64) - self.origin_y) / self.cell).astype(np.int64)
        np.clip(col, 0, self.width - 1, out=col)
        np.clip(row, 0, self.height - 1, out=row)
        return row, col

    def floor_at(self, x, y) -> np.ndarray:
        """地面の上端の高さ（双一次補間）。"""
        fx = np.clip((np.asarray(x, dtype=np.float64) - self.origin_x) / self.cell, 0.0, self.width - 1.000001)
        fy = np.clip((np.asarray(y, dtype=np.float64) - self.origin_y) / self.cell, 0.0, self.height - 1.000001)
        ix = fx.astype(np.int64)
        iy = fy.astype(np.int64)
        tx = fx - ix
        ty = fy - iy
        f = self.floor
        top = f[iy, ix] * (1.0 - tx) + f[iy, ix + 1] * tx
        bottom = f[iy + 1, ix] * (1.0 - tx) + f[iy + 1, ix + 1] * tx
        return top * (1.0 - ty) + bottom * ty

    def solid(self, x, y, z, building_height: float | None = None) -> np.ndarray:
        """点 (x, y, z) が固体（地面の下・橋桁の中・トンネルの空洞の外の地山。`building_height` があれば建物）の中か。"""
        z = np.asarray(z, dtype=np.float64)
        fx = np.clip((np.asarray(x, dtype=np.float64) - self.origin_x) / self.cell, 0.0, self.width - 1.000001)
        fy = np.clip((np.asarray(y, dtype=np.float64) - self.origin_y) / self.cell, 0.0, self.height - 1.000001)
        ix = fx.astype(np.int64)
        iy = fy.astype(np.int64)
        tx = fx - ix
        ty = fy - iy
        base = iy * self.width + ix
        f = self.floor.reshape(-1)
        floor = (f[base] * (1.0 - tx) + f[base + 1] * tx) * (1.0 - ty) + (
            f[base + self.width] * (1.0 - tx) + f[base + self.width + 1] * tx
        ) * ty
        near = base + (tx >= 0.5) + (ty >= 0.5) * self.width
        marks = self.marks.reshape(-1)[near]
        out = z < floor - 0.05
        if building_height is not None and self.building is not None:
            out |= self.building.reshape(-1)[near] & (z < floor + building_height)
        structured = marks != 0
        if structured.any():
            sel = np.flatnonzero(structured)
            cells = near.reshape(-1)[sel]
            zs = z.reshape(-1)[sel] if z.shape == near.shape else np.broadcast_to(z, near.shape).reshape(-1)[sel]
            in_cavity = (zs >= self.cavity_lo.reshape(-1)[cells]) & (zs <= self.cavity_hi.reshape(-1)[cells])
            in_slab = (zs >= self.slab_lo.reshape(-1)[cells]) & (zs <= self.slab_hi.reshape(-1)[cells])
            flat = out.reshape(-1)
            flat[sel] = (flat[sel] & ~in_cavity) | in_slab
        return out


def stamp_edge(
    grids_shape: tuple[int, int],
    origin: tuple[float, float],
    cell: float,
    surf: EdgeSurface,
    margin: float,
    value: np.ndarray,
    best: np.ndarray,
) -> np.ndarray:
    """辺の路面の高さを、中心線に近いセルから順に `value` へ書く。書いたセルの印を返す。"""
    h, w = grids_shape
    ox, oy = origin
    reach = surf.half_width + margin
    stamped = np.zeros((h, w), dtype=bool)
    pts = surf.pts
    for j in range(pts.shape[0] - 1):
        ax, ay = pts[j]
        bx, by = pts[j + 1]
        c0 = max(int(math.floor((min(ax, bx) - reach - ox) / cell)), 0)
        c1 = min(int(math.ceil((max(ax, bx) + reach - ox) / cell)), w - 1)
        r0 = max(int(math.floor((min(ay, by) - reach - oy) / cell)), 0)
        r1 = min(int(math.ceil((max(ay, by) + reach - oy) / cell)), h - 1)
        if c1 < c0 or r1 < r0:
            continue
        gx = ox + np.arange(c0, c1 + 1, dtype=np.float64) * cell
        gy = oy + np.arange(r0, r1 + 1, dtype=np.float64) * cell
        px, py = np.meshgrid(gx, gy)
        dx, dy = bx - ax, by - ay
        length2 = max(dx * dx + dy * dy, 1e-12)
        length = math.sqrt(length2)
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / length2, 0.0, 1.0)
        fx = ax + dx * t
        fy = ay + dy * t
        lateral = (-(px - fx) * dy + (py - fy) * dx) / length
        dist = np.hypot(px - fx, py - fy)
        z = surf.z[j] + (surf.z[j + 1] - surf.z[j]) * t + lateral * surf.cross
        window = best[r0 : r1 + 1, c0 : c1 + 1]
        take = (dist <= reach) & (dist < window)
        if not take.any():
            continue
        window[take] = dist[take]
        value[r0 : r1 + 1, c0 : c1 + 1][take] = z[take]
        stamped[r0 : r1 + 1, c0 : c1 + 1] |= take
    return stamped


def _dilate(mask: np.ndarray, values: np.ndarray, cells: int) -> tuple[np.ndarray, np.ndarray]:
    """印を 4 近傍へ `cells` 回広げ、広げた先へ値の最大を運ぶ。"""
    out = mask.copy()
    vals = np.where(mask, values, -np.inf)
    for _ in range(int(cells)):
        grown = out.copy()
        grown_vals = vals.copy()
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            moved = np.roll(out, shift, axis=axis)
            moved_vals = np.roll(vals, shift, axis=axis)
            if axis == 0:
                if shift > 0:
                    moved[0, :] = False
                else:
                    moved[-1, :] = False
            else:
                if shift > 0:
                    moved[:, 0] = False
                else:
                    moved[:, -1] = False
            grown |= moved
            grown_vals = np.maximum(grown_vals, np.where(moved, moved_vals, -np.inf))
        out, vals = grown, grown_vals
    return out, vals


def build_solid_grids(surface: RoadSurface) -> SolidGrids | None:
    """地形と道路から固体判定グリッドを作る。広すぎる地図では None。"""
    data = surface.data
    margin = float(config.GRID_MARGIN)
    cell = SOLID_CELL_M
    ox = data.bounds.min_x - margin
    oy = data.bounds.min_y - margin
    w = int(math.ceil((data.bounds.max_x + margin - ox) / cell)) + 1
    h = int(math.ceil((data.bounds.max_y + margin - oy) / cell)) + 1
    if w * h > SOLID_MAX_CELLS:
        warn_once(
            "map.elevation.solid_too_large",
            "立体の地図が広すぎて固体判定グリッドを作れません。擬似カメラと見通しは平面として扱います",
        )
        return None
    gx = ox + np.arange(w, dtype=np.float64) * cell
    gy = oy + np.arange(h, dtype=np.float64) * cell
    if surface.terrain is not None:
        px, py = np.meshgrid(gx, gy)
        floor = surface.terrain.sample(px, py).astype(np.float64)
    else:
        floor = np.zeros((h, w), dtype=np.float64)
    shape = (h, w)
    inf = np.full(shape, np.inf)
    ninf = np.full(shape, -np.inf)

    ground_best = np.full(shape, np.inf)
    ground_z = np.zeros(shape)
    deck_best = np.full(shape, np.inf)
    deck_z = np.zeros(shape)
    tunnel_best = np.full(shape, np.inf)
    tunnel_z = np.zeros(shape)
    ground_mask = np.zeros(shape, dtype=bool)
    deck_mask = np.zeros(shape, dtype=bool)
    tunnel_mask = np.zeros(shape, dtype=bool)
    for surf in surface.edges.values():
        if surf.structure == STRUCTURE_BRIDGE:
            deck_mask |= stamp_edge(shape, (ox, oy), cell, surf, STRUCTURE_MARGIN_M, deck_z, deck_best)
        elif surf.structure == STRUCTURE_TUNNEL:
            tunnel_mask |= stamp_edge(shape, (ox, oy), cell, surf, STRUCTURE_MARGIN_M, tunnel_z, tunnel_best)
        else:
            ground_mask |= stamp_edge(shape, (ox, oy), cell, surf, CARVE_MARGIN_M, ground_z, ground_best)

    # 地上の道路は地形を路面まで削り・盛る（橋桁の下とトンネルの上は地形のまま）
    carve = ground_mask & ~deck_mask & ~tunnel_mask
    floor = np.where(carve, ground_z, floor)

    slab_hi = np.where(deck_mask, deck_z, ninf)
    slab_lo = np.where(deck_mask, deck_z - DECK_THICKNESS_M, inf)

    # 土かぶりが足りない所は切通し（空洞ではなく、地形を路面まで削る）
    covered = tunnel_mask & (floor >= tunnel_z + TUNNEL_CLEARANCE_M + TUNNEL_MIN_COVER_M)
    open_cut = tunnel_mask & ~covered
    floor = np.where(open_cut, tunnel_z, floor)
    cavity_lo = np.where(covered, tunnel_z, inf)
    cavity_hi = np.where(covered, tunnel_z + TUNNEL_CLEARANCE_M, ninf)

    grown, grown_top = _dilate(covered, cavity_hi, WALL_CELLS)
    wall = grown & ~covered & (floor > grown_top)
    wall_top = np.where(wall, grown_top, -np.inf)

    return SolidGrids(
        origin_x=float(ox),
        origin_y=float(oy),
        cell=float(cell),
        width=int(w),
        height=int(h),
        floor=floor.astype(np.float32),
        slab_lo=slab_lo.astype(np.float32),
        slab_hi=slab_hi.astype(np.float32),
        cavity_lo=cavity_lo.astype(np.float32),
        cavity_hi=cavity_hi.astype(np.float32),
        wall=wall,
        wall_top=wall_top.astype(np.float32),
    )


def structure_level(structure: str) -> int:
    """擬似カメラの路面ラスタを分ける番号（0=地上・1=橋桁・2=トンネル）。"""
    if structure == STRUCTURE_BRIDGE:
        return 1
    if structure == STRUCTURE_TUNNEL:
        return 2
    return 0
