"""立体の地図の擬似カメラの姿勢・地面の描画と、描画と正解ラベルが共有する立体の見通し判定。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app import config
from app.map.elevation import SolidGrids
from app.percep.types import CameraSpec, DetClass, PerceptionResult, estimate_distance

__all__ = [
    "OFF_LEVEL_M",
    "drop_other_levels",
    "level_offset",
    "vehicle_box_corners",
    "CameraBatch",
    "GroundView",
    "camera_batch",
    "freespace_3d",
    "render_ground",
    "rotate_box_corners",
    "roll_image",
    "visible_3d",
]

#: 地面を見る列ごとの奥行きの標本数と、最初の標本の距離 [m]（等比で `far` まで）
SAMPLE_COUNT = 128
NEAR_SAMPLE_M = 0.3
#: 地面を奥へなめる列を何画素ごとに共有するか（となりの画素とは方位が 0.4 度しか違わない）
COLUMN_STEP = 2
#: 列の水平方向を測る画素の高さ（tan。地面に当たる行の代表）
COLUMN_PROBE_TAN = -0.15
#: 標本の間で床がこれ以上せり上がる所は、壁（擁壁・盛土の法面・トンネルの壁）として塗る [m]
FACE_RISE_M = 0.6
#: 形（車・人・標識）を地面の手前と見なす奥行きの余裕 [m]（車体の半分の長さ + α）
SHAPE_DEPTH_TOLERANCE_M = 3.0
#: 見通し判定の標本の間隔と上限
LOS_STEP_M = 1.0
LOS_MAX_SAMPLES = 128
#: 走行可能距離で、隣の標本との段差がこれを超えたら走れない（路肩の落差・壁）[m]
FREESPACE_STEP_M = 0.35
#: 走行可能距離で「見えている」とみなす路面からの高さ [m]（縁石ほど）
FREESPACE_EYE_MARGIN_M = 0.3

#: 検出枠から推定した、自車の路面からの高さがこれを超える車・障害物・人は別の階とみなす [m]
#: （桁下の建築限界 4.5m。同じ道路の坂の変わり目では 40m 先でも 2m ほどしかずれない）
OFF_LEVEL_M = 4.5
_LEVELED_CLASSES = (DetClass.VEHICLE, DetClass.OBSTACLE, DetClass.PEDESTRIAN)

#: 地面の描画が返す種別（ラベルは描く側が決める）
HIT_NONE = 0
#: 路面（`level` のラスタで路面・標示・地面を塗る）
HIT_ROAD = 1
#: 構造物（橋桁の下面と側面・トンネルの天井と壁・坑口・擁壁と盛土の法面）
HIT_STRUCTURE = 2
#: 地山の表面（坑外から見たトンネルの上の山）
HIT_TERRAIN = 3

#: 床が当たった路面のラスタの番号（`map.elevation.structure_level` と同じ）
LEVEL_GROUND = 0
LEVEL_BRIDGE = 1
LEVEL_TUNNEL = 2


@dataclass(frozen=True)
class CameraBatch:
    """複数台ぶんのカメラの目の位置と向き。向きは Rz(yaw)·Ry(-pitch)·Rx(roll) の分解。"""

    eye_x: np.ndarray
    eye_y: np.ndarray
    eye_z: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray
    roll: np.ndarray

    @property
    def cos_yaw(self) -> np.ndarray:
        return np.cos(self.yaw)

    @property
    def sin_yaw(self) -> np.ndarray:
        return np.sin(self.yaw)

    @property
    def cos_pitch(self) -> np.ndarray:
        return np.cos(self.pitch)

    @property
    def sin_pitch(self) -> np.ndarray:
        return np.sin(self.pitch)


def _rot_z(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    out = np.zeros(a.shape + (3, 3))
    out[..., 0, 0], out[..., 0, 1] = c, -s
    out[..., 1, 0], out[..., 1, 1] = s, c
    out[..., 2, 2] = 1.0
    return out


def _rot_y_neg(a: np.ndarray) -> np.ndarray:
    """Ry(-a)：a が正なら前（x）を上（z）へ起こす。"""
    c, s = np.cos(a), np.sin(a)
    out = np.zeros(a.shape + (3, 3))
    out[..., 0, 0], out[..., 0, 2] = c, -s
    out[..., 1, 1] = 1.0
    out[..., 2, 0], out[..., 2, 2] = s, c
    return out


def _rot_x(a: np.ndarray) -> np.ndarray:
    """Rx(a)：a が正なら左（y）を上（z）へ起こす。"""
    c, s = np.cos(a), np.sin(a)
    out = np.zeros(a.shape + (3, 3))
    out[..., 0, 0] = 1.0
    out[..., 1, 1], out[..., 1, 2] = c, -s
    out[..., 2, 1], out[..., 2, 2] = s, c
    return out


def body_rotation(heading, pitch, roll) -> np.ndarray:
    """車体の回転行列 (..., 3, 3)。列は車体の前・左・上を世界座標で表したもの。"""
    heading = np.asarray(heading, dtype=np.float64)
    return _rot_z(heading) @ _rot_y_neg(np.asarray(pitch, dtype=np.float64)) @ _rot_x(np.asarray(roll, dtype=np.float64))


def camera_batch(x, y, z, heading, pitch, roll, spec: CameraSpec) -> CameraBatch:
    """車体の位置・方位・路面の姿勢から、そのカメラの目の位置と向きを作る。"""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))
    body = body_rotation(np.atleast_1d(heading), np.atleast_1d(pitch), np.atleast_1d(roll))
    offset = np.array([spec.forward, -spec.right, spec.eye_height], dtype=np.float64)
    eye = np.stack([x, y, z], axis=-1) + body @ offset
    mount = _rot_z(np.array(spec.yaw)) @ _rot_y_neg(np.array(spec.pitch))
    cam = body @ mount
    yaw = np.arctan2(cam[..., 1, 0], cam[..., 0, 0])
    cam_pitch = np.arcsin(np.clip(cam[..., 2, 0], -1.0, 1.0))
    cam_roll = np.arctan2(cam[..., 2, 1], cam[..., 2, 2])
    return CameraBatch(eye[..., 0], eye[..., 1], eye[..., 2], yaw, cam_pitch, cam_roll)


def rotate_box_corners(u: np.ndarray, v: np.ndarray, roll: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    """ロールの無い画の座標を、ロールのある画の座標へ回す（主点まわり）。"""
    if roll == 0.0:
        return u, v
    c, s = math.cos(roll), math.sin(roll)
    du = u - cx
    dv = v - cy
    return cx + c * du + s * dv, cy - s * du + c * dv


def roll_image(arrays: list[np.ndarray], roll: np.ndarray, cx: float, cy: float) -> None:
    """ロールの無い画（カメラごと）を、主点まわりに回した画へその場で置き換える（最近傍）。"""
    if not np.any(roll != 0.0):
        return
    n, h, w = arrays[0].shape[:3]
    rows = np.arange(h, dtype=np.float64) + 0.5 - cy
    cols = np.arange(w, dtype=np.float64) + 0.5 - cx
    du, dv = np.meshgrid(cols, rows)
    for i in range(n):
        if roll[i] == 0.0:
            continue
        c, s = math.cos(float(roll[i])), math.sin(float(roll[i]))
        src_u = np.clip(np.floor(cx + c * du - s * dv), 0, w - 1).astype(np.int64)
        src_v = np.clip(np.floor(cy + s * du + c * dv), 0, h - 1).astype(np.int64)
        for arr in arrays:
            arr[i] = arr[i][src_v, src_u]


@dataclass
class GroundView:
    """地面の描画の結果（ロール前の画）。形はすべて (カメラ, 行, 列)。"""

    kind: np.ndarray
    level: np.ndarray
    depth: np.ndarray
    hit_x: np.ndarray
    hit_y: np.ndarray
    #: トンネルの中の画素（暗く塗る）
    shade: np.ndarray
    #: 視線が下を向いている画素（何にも当たらなければ遠くの地面、上なら空）
    looking_down: np.ndarray


def _pixel_rays(batch: CameraBatch, spec: CameraSpec):
    """画素ごとの視線の (水平方向 x, y, 水平の長さ, 傾き = 上向きの比)。視線は奥行き 1 の長さ。"""
    w, h = int(spec.width), int(spec.height)
    focal = float(spec.focal_px)
    a = ((np.arange(w, dtype=np.float64) + 0.5) - w * 0.5) / focal
    b = (h * 0.5 - (np.arange(h, dtype=np.float64) + 0.5)) / focal
    cp = np.cos(batch.pitch)[:, None, None]
    sp = np.sin(batch.pitch)[:, None, None]
    cy = np.cos(batch.yaw)[:, None, None]
    sy = np.sin(batch.yaw)[:, None, None]
    fwd = cp - b[None, :, None] * sp
    dz = sp + b[None, :, None] * cp
    aa = a[None, None, :]
    dx = cy * fwd + sy * aa
    dy = sy * fwd - cy * aa
    hl = np.sqrt(fwd * fwd + aa * aa)
    return dx / hl, dy / hl, hl, dz / hl


def _column_dirs(batch: CameraBatch, spec: CameraSpec):
    w = int(spec.width)
    focal = float(spec.focal_px)
    a = ((np.arange(w // COLUMN_STEP, dtype=np.float64) * COLUMN_STEP + COLUMN_STEP * 0.5) - w * 0.5) / focal
    cp = np.cos(batch.pitch)[:, None]
    sp = np.sin(batch.pitch)[:, None]
    cy = np.cos(batch.yaw)[:, None]
    sy = np.sin(batch.yaw)[:, None]
    fwd = cp - COLUMN_PROBE_TAN * sp
    dx = cy * fwd + sy * a[None, :]
    dy = sy * fwd - cy * a[None, :]
    norm = np.sqrt(dx * dx + dy * dy)
    return dx / norm, dy / norm


def _first_at_least(values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """列ごとの非減少の値 (C, K) で、問い合わせ (C, Q) 以上になる最初の添字（無ければ K）。"""
    c, k = values.shape
    finite = values[np.isfinite(values)]
    span = float(np.max(np.abs(finite), initial=1.0)) + float(np.max(np.abs(query), initial=1.0)) + 1.0
    offset = (np.arange(c, dtype=np.float64) * span * 4.0)[:, None]
    # 列をずらして 1 本の昇順の並びにする（無限大は列の幅の中へ畳む。-inf は「まだ当たらない」）
    keys = np.nan_to_num(values, nan=-2.0 * span, posinf=2.0 * span, neginf=-2.0 * span) + offset
    found = np.searchsorted(keys.reshape(-1), (query + offset).reshape(-1), side="left").reshape(query.shape)
    return np.clip(found - (np.arange(c) * k)[:, None], 0, k)


_DIST_CACHE: dict[float, np.ndarray] = {}


def _sample_distances(far: float) -> np.ndarray:
    cached = _DIST_CACHE.get(float(far))
    if cached is None:
        cached = np.geomspace(NEAR_SAMPLE_M, float(far), SAMPLE_COUNT).astype(np.float32)
        _DIST_CACHE[float(far)] = cached
    return cached


def render_ground(grids: SolidGrids, batch: CameraBatch, spec: CameraSpec) -> GroundView:
    """固体判定グリッドを列ごとに奥へなめ、各画素で最初に当たる路面・構造物・地山を決める。"""
    n = batch.eye_x.size
    h, width = int(spec.height), int(spec.width)
    step = COLUMN_STEP
    w = width // step
    dist = _sample_distances(float(spec.far))
    k = dist.size
    col_x, col_y = _column_dirs(batch, spec)
    sx = (batch.eye_x[:, None, None] + col_x[:, :, None] * dist[None, None, :]).astype(np.float32)
    sy = (batch.eye_y[:, None, None] + col_y[:, :, None] * dist[None, None, :]).astype(np.float32)
    gw = grids.width
    fx = np.clip((sx - np.float32(grids.origin_x)) / np.float32(grids.cell), 0.0, gw - 1.001)
    fy = np.clip((sy - np.float32(grids.origin_y)) / np.float32(grids.cell), 0.0, grids.height - 1.001)
    ix = fx.astype(np.int64)
    iy = fy.astype(np.int64)
    tx = fx - ix
    ty = fy - iy
    base = iy * gw + ix
    flat_floor = grids.floor.reshape(-1)
    floor = (flat_floor[base] * (1.0 - tx) + flat_floor[base + 1] * tx) * (1.0 - ty) + (
        flat_floor[base + gw] * (1.0 - tx) + flat_floor[base + gw + 1] * tx
    ) * ty
    near = base + (tx >= 0.5) + (ty >= 0.5) * gw
    marks = grids.marks.reshape(-1)[near]
    eye_z = batch.eye_z.astype(np.float32)
    erow, ecol = grids.cell_index(batch.eye_x, batch.eye_y)
    inside = (batch.eye_z >= grids.cavity_lo[erow, ecol]) & (batch.eye_z <= grids.cavity_hi[erow, ecol])

    eff = floor.astype(np.float32)
    level = np.zeros(floor.shape, dtype=np.int8)
    shape = floor.shape
    sel = np.flatnonzero(marks)
    structured = sel.size > 0
    if structured:
        # 構造物のある標本だけを疎に扱う（視界のごく一部）
        cam_s = sel // (w * k)
        cells = near.reshape(-1)[sel]
        mk = marks.reshape(-1)[sel]
        f_s = eff.reshape(-1)[sel]
        ez_s = eye_z[cam_s]
        ins = inside[cam_s]
        cav_lo = grids.cavity_lo.reshape(-1)[cells]
        cav_hi = grids.cavity_hi.reshape(-1)[cells]
        slab_lo = grids.slab_lo.reshape(-1)[cells]
        slab_hi = grids.slab_hi.reshape(-1)[cells]
        has_cav = (mk & 1) != 0
        has_slab = (mk & 2) != 0
        # トンネル：坑内では床は坑内の路面・上は天井。坑外からは「天井の下面〜山の表面」の固体が浮かぶ
        above_hill = has_cav & (f_s < ez_s) & ~ins
        tunnel_floor = has_cav & ~above_hill
        e_s = np.where(tunnel_floor, cav_lo, f_s)
        lvl = np.where(tunnel_floor, LEVEL_TUNNEL, LEVEL_GROUND).astype(np.int8)
        ceil_s = np.where(has_cav & ins, cav_hi, np.inf)
        roof = has_cav & ~ins & ~above_hill
        lo_s = np.where(roof, cav_hi, np.inf)
        hi_s = np.where(roof, f_s, -np.inf)
        hill_s = roof.copy()
        # 橋桁：上面が目より下なら床（橋の上の路面）、上にあれば浮いた固体（桁下から見上げる）
        deck_floor = has_slab & (slab_hi < ez_s)
        deck_top = deck_floor & (slab_hi >= e_s)
        e_s = np.where(deck_top, slab_hi, e_s)
        lvl = np.where(deck_top, LEVEL_BRIDGE, lvl).astype(np.int8)
        deck_over = has_slab & ~deck_floor
        ceil_s = np.where(deck_over & ins, np.minimum(ceil_s, slab_lo), ceil_s)
        use_deck = deck_over & ~ins & ~roof
        lo_s = np.where(use_deck, slab_lo, lo_s)
        hi_s = np.where(use_deck, slab_hi, hi_s)
        hill_s &= ~use_deck
        eff.reshape(-1)[sel] = e_s
        level.reshape(-1)[sel] = lvl
        cav_lo_full = np.full(shape, np.inf, dtype=np.float32)
        cav_hi_full = np.full(shape, -np.inf, dtype=np.float32)
        wall_full = np.full(shape, -np.inf, dtype=np.float32)
        cav_lo_full.reshape(-1)[sel] = cav_lo
        cav_hi_full.reshape(-1)[sel] = cav_hi
        wall_full.reshape(-1)[sel] = grids.wall_top.reshape(-1)[cells]
        lo_full = np.full(shape, np.inf, dtype=np.float32)
        hi_full = np.full(shape, -np.inf, dtype=np.float32)
        hill_full = np.zeros(shape, dtype=bool)
        ceil_full = np.full(shape, np.inf, dtype=np.float32)
        lo_full.reshape(-1)[sel] = lo_s
        hi_full.reshape(-1)[sel] = hi_s
        hill_full.reshape(-1)[sel] = hill_s
        ceil_full.reshape(-1)[sel] = ceil_s

    ray_x, ray_y, ray_hl, slope = _pixel_rays(batch, spec)
    slope = slope.astype(np.float32)
    # 画素 (n, h, width) を、同じ列の標本を共有する組 (n * w, h * step) に並べ替えて、列ごとに探す
    cols_slope = np.ascontiguousarray(slope.reshape(n, h, w, step).transpose(0, 2, 1, 3)).reshape(n * w, h * step)
    run_floor = np.maximum.accumulate((eff - eye_z[:, None, None]) / dist[None, None, :], axis=2)
    best = _first_at_least(run_floor.reshape(n * w, k), cols_slope)
    what = np.where(best < k, 1, 0).astype(np.int8)

    if structured and bool(np.isfinite(ceil_s).any()):
        cams = np.flatnonzero(inside)
        rel = (ceil_full[cams] - eye_z[cams][:, None, None]) / dist[None, None, :]
        run_ceil = np.minimum.accumulate(rel, axis=2).reshape(cams.size * w, k)
        cols = (cams[:, None] * w + np.arange(w)[None, :]).reshape(-1)
        k_ceil = _first_at_least(-run_ceil, -cols_slope[cols])
        nearer = k_ceil < best[cols]
        best[cols] = np.where(nearer, k_ceil, best[cols])
        what[cols] = np.where(nearer, 2, what[cols])

    if structured:
        interval = np.isfinite(lo_s)
        if interval.any():
            _interval_hits(
                sel[interval], lo_s[interval], hi_s[interval], eye_z, dist, cols_slope, best, what, w, k,
                grids, batch, col_x, col_y,
            )

    best3 = best.reshape(n, w, h, step).transpose(0, 2, 1, 3).reshape(n, h, width)
    what3 = what.reshape(n, w, h, step).transpose(0, 2, 1, 3).reshape(n, h, width)
    kind = np.zeros((n, h, width), dtype=np.int8)
    depth = np.full((n, h, width), np.inf, dtype=np.float32)
    hit_x = np.zeros((n, h, width), dtype=np.float64)
    hit_y = np.zeros((n, h, width), dtype=np.float64)
    shade = np.zeros((n, h, width), dtype=bool)
    out_level = np.zeros((n, h, width), dtype=np.int8)
    pix = np.flatnonzero(best3 < k)
    if pix.size:
        cam = pix // (h * width)
        colp = (pix % width) // step
        kk = best3.reshape(-1)[pix].astype(np.int64)
        prevk = np.maximum(kk - 1, 0)
        sample = (cam * w + colp) * k
        si = sample + kk
        d1 = dist[kk]
        d0 = np.where(kk > 0, dist[prevk], np.float32(0.0)).astype(np.float32)
        m = slope.reshape(-1)[pix]
        ez = eye_z[cam]
        wh = what3.reshape(-1)[pix]
        eff_f = eff.reshape(-1)
        # 床：直前の標本（距離 0 は目の真下の路面）との間で、視線と床の交点を線形に求める
        f1 = eff_f[si]
        f0 = np.where(kk > 0, eff_f[sample + prevk], ez - np.float32(spec.eye_height))
        g0 = ez + m * d0 - f0
        g1 = ez + m * d1 - f1
        t = np.clip(g0 / np.maximum(g0 - g1, 1e-9), 0.0, 1.0)
        d_hit = d0 + t * (d1 - d0)
        floor_hit = wh == 1
        k_out = np.where(floor_hit, HIT_ROAD, HIT_STRUCTURE).astype(np.int8)
        lvl_out = np.where(floor_hit, level.reshape(-1)[si], LEVEL_GROUND).astype(np.int8)
        hz = ez + m * d_hit
        rise = floor_hit & (f1 - f0 > FACE_RISE_M) & (np.abs(hz - f1) > 0.3)
        sh = np.zeros(pix.size, dtype=bool)
        if structured:
            block = ~floor_hit
            if block.any():
                bi = si[block]
                lo_b = lo_full.reshape(-1)[bi]
                interval_hit = np.isfinite(lo_b)
                under = np.where(interval_hit, lo_b, ceil_full.reshape(-1)[bi])
                over = np.where(interval_hit, hi_full.reshape(-1)[bi], np.inf)
                mb = m[block]
                ezb = ez[block]
                z0 = ezb + mb * d0[block]
                front = (z0 >= under) & (z0 <= over)
                safe_m = np.where(np.abs(mb) < 1e-9, np.float32(1e-9), mb)
                d_under = np.clip((under - ezb) / safe_m, d0[block], d1[block])
                d_over = np.clip((over - ezb) / safe_m, d0[block], d1[block])
                d_front = d0[block]
                if front.any():
                    # 前面は標本の間にあるので、構造物の印が立つ境目を二分探索で詰める（橋桁の縁のギザつきを消す）
                    fi = np.flatnonzero(front)
                    a = d0[block][fi].copy()
                    b = d1[block][fi].copy()
                    cb = cam[block][fi]
                    dxr = ray_x.reshape(-1)[pix][block][fi]
                    dyr = ray_y.reshape(-1)[pix][block][fi]
                    for _ in range(4):
                        mid = (a + b) * 0.5
                        _r, _c = grids.cell_index(batch.eye_x[cb] + dxr * mid, batch.eye_y[cb] + dyr * mid)
                        marked = grids.marks[_r, _c] != 0
                        b = np.where(marked, mid, b)
                        a = np.where(marked, a, mid)
                    d_front = d_front.copy()
                    d_front[fi] = b
                d_hit[block] = np.where(front, d_front, np.where(mb > 0, d_under, d_over))
                hz_b = ezb + mb * d_hit[block]
                hz[block] = hz_b
                on_hill = hill_full.reshape(-1)[bi] & (hz_b >= over - 0.5)
                k_out[block] = np.where(on_hill, HIT_TERRAIN, HIT_STRUCTURE)
            walled = floor_hit & (hz <= wall_full.reshape(-1)[si] + 0.5)
            k_out[rise | walled] = HIT_STRUCTURE
            cl = cav_lo_full.reshape(-1)[si]
            in_cavity = (hz >= cl - 0.5) & (hz <= cav_hi_full.reshape(-1)[si] + 0.5)
            sh = (in_cavity & (k_out != HIT_TERRAIN)) | walled
        else:
            k_out[rise] = HIT_STRUCTURE
        kind.reshape(-1)[pix] = k_out
        out_level.reshape(-1)[pix] = lvl_out
        depth.reshape(-1)[pix] = d_hit / ray_hl.reshape(-1)[pix]
        hit_x.reshape(-1)[pix] = batch.eye_x[cam] + ray_x.reshape(-1)[pix] * d_hit
        hit_y.reshape(-1)[pix] = batch.eye_y[cam] + ray_y.reshape(-1)[pix] * d_hit
        shade.reshape(-1)[pix] = sh
    return GroundView(
        kind=kind,
        level=out_level,
        depth=depth,
        hit_x=hit_x,
        hit_y=hit_y,
        shade=shade,
        looking_down=slope < 0.0,
    )


def _interval_hits(flat_idx, lo, hi, eye_z, dist, cols_slope, best, what, w, k, grids, batch, col_x, col_y) -> None:
    """浮いた固体に最初に当たる標本を、固体のある標本だけ総当たりで探す（その場で best・what を書き換える）。"""
    column = flat_idx // k
    sample = flat_idx % k
    cols, start, counts = np.unique(column, return_index=True, return_counts=True)
    depth = int(counts.max())
    pos = np.arange(flat_idx.size) - np.repeat(start, counts)
    row_of = np.repeat(np.arange(cols.size), counts)
    ez = eye_z[column // w]
    d1 = dist[sample]
    d0 = np.where(sample > 0, dist[np.maximum(sample - 1, 0)], np.float32(1e-6)).astype(np.float32)
    # 固体の並びの最初の標本は、手前の境目（前面）までの距離を二分探索で詰めてから窓を作る。
    #   ひとつ前の標本の距離のままだと、前面の上端が標本の間隔で段々になる
    first = np.ones(flat_idx.size, dtype=bool)
    first[1:] = ~((column[1:] == column[:-1]) & (sample[1:] == sample[:-1] + 1))
    fi = np.flatnonzero(first & (sample > 0))
    if fi.size:
        cam = column[fi] // w
        cc = column[fi] % w
        dx = col_x[cam, cc]
        dy = col_y[cam, cc]
        a = d0[fi].astype(np.float64)
        b = d1[fi].astype(np.float64)
        for _ in range(5):
            mid = (a + b) * 0.5
            r, c = grids.cell_index(batch.eye_x[cam] + dx * mid, batch.eye_y[cam] + dy * mid)
            marked = grids.marks[r, c] != 0
            b = np.where(marked, mid, b)
            a = np.where(marked, a, mid)
        d0[fi] = b.astype(np.float32)
    rel_lo = lo - ez
    rel_hi = hi - ez
    # 視線が標本の間 [d0, d1] で固体の高さの範囲に入る傾きの窓 [g, gh]
    g = np.where(rel_lo >= 0.0, rel_lo / d1, rel_lo / d0).astype(np.float32)
    gh = np.where(rel_hi >= 0.0, rel_hi / d0, rel_hi / d1).astype(np.float32)
    lo_w = np.full((cols.size, depth), np.inf, dtype=np.float32)
    hi_w = np.full((cols.size, depth), -np.inf, dtype=np.float32)
    k_w = np.full((cols.size, depth), k, dtype=np.int64)
    lo_w[row_of, pos] = g
    hi_w[row_of, pos] = gh
    k_w[row_of, pos] = sample
    rows = cols_slope[cols]
    h = rows.shape[1]
    budget = max(1, 4_000_000 // max(1, h * depth))
    for begin in range(0, cols.size, budget):
        part = slice(begin, begin + budget)
        sl = rows[part][:, :, None]
        hit = (sl >= lo_w[part][:, None, :]) & (sl <= hi_w[part][:, None, :])
        any_hit = hit.any(axis=2)
        first = np.take_along_axis(k_w[part], np.argmax(hit, axis=2), axis=1)
        idx = cols[part]
        current = best[idx]
        nearer = any_hit & (first < current)
        best[idx] = np.where(nearer, first, current)
        what[idx] = np.where(nearer, 2, what[idx])


def level_offset(det, spec: CameraSpec) -> float:
    """検出枠の下端と推定距離から、物体の足元が自車の路面より何 m 上にあるか（カメラの画だけから推定する）。"""
    distance = float(estimate_distance(det, spec))
    v_bottom = float(det.y1) * float(spec.height)
    b = (float(spec.height) * 0.5 - v_bottom) / float(spec.focal_px)
    pitch = float(spec.pitch)
    cp, sp = math.cos(pitch), math.sin(pitch)
    forward = cp - b * sp
    if forward <= 1e-6:
        return 0.0
    # 車体の座標で、足元へ向かう視線の傾き。カメラは車体（＝路面）に固定なので地図を見ない
    return float(spec.eye_height) + distance * (sp + b * cp) / forward


def drop_other_levels(result: PerceptionResult, spec: CameraSpec) -> PerceptionResult:
    """立体交差の上下の階に見えた車・障害物・人を、同じ路面の相手として扱わないよう外す。"""
    keep = [
        det for det in result.detections
        if det.cls not in _LEVELED_CLASSES or abs(level_offset(det, spec)) <= OFF_LEVEL_M
    ]
    if len(keep) == len(result.detections):
        return result
    return PerceptionResult(slot=result.slot, detections=keep)


def vehicle_box_corners(x, y, z, heading, pitch, roll) -> np.ndarray:
    """車体の外接直方体の 8 頂点 (K, 8, 3)。路面の姿勢で傾ける（描画と正解ラベルが同じ形を使う）。"""
    half_l = float(config.VEHICLE_LENGTH) * 0.5
    half_w = float(config.VEHICLE_WIDTH) * 0.5
    height = float(config.VEHICLE_HEIGHT)
    local = np.array(
        [[sx * half_l, sy * half_w, hz] for hz in (0.0, height) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))],
        dtype=np.float64,
    )
    rot = body_rotation(np.atleast_1d(heading), np.atleast_1d(pitch), np.atleast_1d(roll))
    origin = np.stack([np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)], axis=-1).astype(np.float64)
    return origin[:, None, :] + np.einsum("kij,pj->kpi", rot, local)


def visible_3d(
    grids: SolidGrids,
    occupancy,
    eye: tuple[float, float, float],
    targets: np.ndarray,
    building_height: float,
) -> np.ndarray:
    """目から各目標点 (M, 3) までの線分が、固体（地面・橋桁・山）と建物を貫かないものを True で返す。"""
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
    m = targets.shape[0]
    if m == 0:
        return np.zeros(0, dtype=bool)
    d = targets - np.asarray(eye, dtype=np.float64)[None, :]
    reach = float(np.max(np.hypot(d[:, 0], d[:, 1])))
    steps = int(min(LOS_MAX_SAMPLES, max(2, math.ceil(reach / LOS_STEP_M))))
    t = _los_params(steps)
    px = eye[0] + d[:, 0:1] * t[None, :]
    py = eye[1] + d[:, 1:2] * t[None, :]
    pz = eye[2] + d[:, 2:3] * t[None, :]
    if grids.building is not None or occupancy is None:
        blocked = grids.solid(px, py, pz, building_height)
    else:
        blocked = grids.solid(px, py, pz)
        blocked |= occupancy.sample_building(px, py) & (pz < grids.floor_at(px, py) + building_height)
    return ~blocked.any(axis=1)


_LOS_PARAMS: dict[int, np.ndarray] = {}


def _los_params(steps: int) -> np.ndarray:
    cached = _LOS_PARAMS.get(steps)
    if cached is None:
        cached = np.linspace(0.0, 1.0, steps + 2, dtype=np.float64)[1:-1]
        _LOS_PARAMS[steps] = cached
    return cached


def freespace_3d(
    grids: SolidGrids,
    occupancy,
    origin: np.ndarray,
    eye_z: np.ndarray,
    road_z: np.ndarray,
    angles: np.ndarray,
    max_distance: float,
    step: float = 1.0,
) -> np.ndarray:
    """立体の地図の走行可能距離 (P, D)。建物・壁・路肩の落差・低い天井に加え、目から見えない路面（坂の頂上の先）で止める。"""
    origin = np.asarray(origin, dtype=np.float64).reshape(-1, 2)
    p = origin.shape[0]
    angles = np.asarray(angles, dtype=np.float64).reshape(p, -1)
    samples = np.arange(step, float(max_distance) + step * 0.5, step, dtype=np.float64)
    xs = origin[:, 0, None, None] + np.cos(angles)[:, :, None] * samples[None, None, :]
    ys = origin[:, 1, None, None] + np.sin(angles)[:, :, None] * samples[None, None, :]
    row, col = grids.cell_index(xs, ys)
    floor = grids.floor_at(xs, ys).astype(np.float64)
    cav_lo = grids.cavity_lo[row, col].astype(np.float64)
    cav_hi = grids.cavity_hi[row, col].astype(np.float64)
    slab_hi = grids.slab_hi[row, col].astype(np.float64)
    slab_lo = grids.slab_lo[row, col].astype(np.float64)
    base = np.asarray(road_z, dtype=np.float64).reshape(p)[:, None, None]
    # 標本ごとに、車のいる路面に近い候補（地面・坑内の路面・橋桁の上）を選んで高さの列を作る
    cand = np.stack(
        [floor, np.where(np.isfinite(cav_lo), cav_lo, np.nan), np.where(np.isfinite(slab_hi), slab_hi, np.nan)],
        axis=-1,
    )
    surface = np.empty(xs.shape, dtype=np.float64)
    prev = base[:, :, 0]
    for j in range(samples.size):
        options = cand[:, :, j, :]
        gap = np.abs(options - prev[..., None])
        gap = np.where(np.isnan(gap), np.inf, gap)
        pick = np.argmin(gap, axis=-1)
        chosen = np.take_along_axis(options, pick[..., None], axis=-1)[..., 0]
        surface[:, :, j] = chosen
        prev = chosen
    prev_all = np.concatenate([np.broadcast_to(base[:, :, :1], surface[:, :, :1].shape), surface[:, :, :-1]], axis=2)
    stepped = np.abs(surface - prev_all) > FREESPACE_STEP_M * step
    # 天井（坑内の天井・橋桁の下面）が車の高さより低い所は走れない
    roof = np.minimum(np.where(np.isfinite(cav_hi) & (np.abs(surface - cav_lo) < 0.5), cav_hi, np.inf),
                      np.where(np.isfinite(slab_lo) & (slab_lo > surface), slab_lo, np.inf))
    low_roof = roof < surface + config.VEHICLE_HEIGHT
    # 目から見えない路面（坂の頂上の先）は空いているか分からないので、そこで止める
    eye = np.asarray(eye_z, dtype=np.float64).reshape(p)[:, None, None]
    angle_to = (surface + FREESPACE_EYE_MARGIN_M - eye) / samples[None, None, :]
    horizon = np.maximum.accumulate((surface - eye) / samples[None, None, :], axis=2)
    horizon = np.concatenate([np.full(horizon[:, :, :1].shape, -np.inf), horizon[:, :, :-1]], axis=2)
    hidden = angle_to < horizon
    blocked = stepped | low_roof | hidden
    if occupancy is not None:
        blocked |= occupancy.sample_building(xs, ys)
    any_hit = blocked.any(axis=2)
    first = np.argmax(blocked, axis=2)
    out = np.where(any_hit, samples[first], float(max_distance))
    return np.minimum(out, float(max_distance)).astype(np.float32)
