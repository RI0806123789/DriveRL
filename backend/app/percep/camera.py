"""擬似カメラ描画（運転席視点のラスタライズ）。

学習ループはバックエンドのエンジンスレッドが回すので、認識器の入力になる画像も
Python 側で作る必要がある。3D 描画はフロントエンド（Three.js）にしか無いため、
ここで `world` と `MapIndex` から**同じ運転席視点**を numpy だけで描く。

カメラの内部パラメータは `percep.types.CameraSpec` に従う。これは
`frontend/src/scene/cameraMath.ts` の運転席カメラと同じ値なので、ここで描いた
画像から得た検出結果は Three.js の映像にそのまま重ねられる。

**この描画は「認識に必要な情報が写っていること」だけを目的にしている。**
陰影・質感・アンチエイリアスは持たない。塗るのは次の 8 種類:

    空 / 地面 / 路面 / 車線標示 / 信号機（灯色つき）/ 最高速度標識 /
    他車両 / 障害物 / 建物

方式は 3 つの層に分かれる。**どれも「1 台ずつの Python ループ」を持たない**
（8 台 20Hz で回すので、ピクセル単位どころか物体単位のループでも予算を使い切る）。

1. 空と地面
   ロールが無いので地平線は水平な直線になる。行スライス 2 回で塗る。

2. 路面と車線標示（地面レイヤ）
   地平線より下の各画素から地面 z=0 への逆透視変換で世界座標を求め、
   起動時に焼いた**道路ラスタ**を 1 回引くだけで路面・白線が決まる。
   多角形の塗り分けではなく参照 1 回なので、道路の本数に依らず一定時間。
   逆透視変換は行方向と列方向に分離できる（`_ground_g` / `_ground_q`）ので、
   カメラごとに要るのは掛け算 2 回だけ。

3. 立体物（建物・車両・信号機・標識・障害物）
   すべて**画像座標の矩形／楕円のスパン**に落として、全カメラ・全物体を
   1 本の配列にまとめ、奥行きの降順に並べてから一括で書き込む
   （ペインターズアルゴリズム）。書き込みは ragged range 展開による
   ファンシー代入 1 回で済む。

   建物だけは面ではなく**画面の列ごとのレイキャスト**で描く。占有グリッドは
   建物しか持たないので、列ごとに水平方向の距離を測って壁の上端・下端を
   投影するほうが、全建物の壁面を投影するより桁違いに安い。

描画は 1 バイトの**ラベル画像**へ行い、最後に一度だけパレットを引いて RGB にする。
色を直接書くと画素あたりの転送量が 3 倍になり、この規模ではそれが支配項になる。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from app import config
from app.contracts import MapIndex
from app.percep.types import DEFAULT_CAMERA, CameraSpec

if TYPE_CHECKING:  # 実行時に import すると sim -> percep -> sim の循環になりうる
    from app.sim.world import World

__all__ = ["PseudoCamera", "LABELS", "PALETTE"]


# ---------------------------------------------------------------------------
# ラベルとパレット
# ---------------------------------------------------------------------------

# ★ 地面を 0 にしてあるのは、道路ラスタの範囲外を clip で端へ丸めたときに
#   そのまま「地面」になるようにするため。マスクを 1 枚省ける。
LBL_GROUND = 0
LBL_ROAD = 1
LBL_MARKING = 2
LBL_SKY = 3
LBL_BUILDING = 4
LBL_VEHICLE = 5
LBL_VEHICLE_DARK = 6
LBL_OBSTACLE = 7
LBL_SIGNAL_BODY = 8
LBL_LAMP_OFF = 9
LBL_LAMP_GREEN = 10
LBL_LAMP_YELLOW = 11
LBL_LAMP_RED = 12
LBL_SIGN_RIM = 13
LBL_SIGN_FACE = 14
LBL_SIGN_DIGIT = 15
LBL_POLE = 16

LABELS = {
    "ground": LBL_GROUND,
    "road": LBL_ROAD,
    "marking": LBL_MARKING,
    "sky": LBL_SKY,
    "building": LBL_BUILDING,
    "vehicle": LBL_VEHICLE,
    "vehicle_dark": LBL_VEHICLE_DARK,
    "obstacle": LBL_OBSTACLE,
    "signal_body": LBL_SIGNAL_BODY,
    "lamp_off": LBL_LAMP_OFF,
    "lamp_green": LBL_LAMP_GREEN,
    "lamp_yellow": LBL_LAMP_YELLOW,
    "lamp_red": LBL_LAMP_RED,
    "sign_rim": LBL_SIGN_RIM,
    "sign_face": LBL_SIGN_FACE,
    "sign_digit": LBL_SIGN_DIGIT,
    "pole": LBL_POLE,
}

#: ラベル -> RGB。灯火の 3 色は互いに離して置く（認識器が灯色を分けられることが要件）。
PALETTE = np.array(
    [
        (86, 92, 78),      # 地面
        (58, 60, 64),      # 路面
        (232, 234, 230),   # 車線標示
        (150, 178, 205),   # 空
        (128, 122, 112),   # 建物
        (52, 96, 168),     # 他車両
        (26, 32, 44),      # 他車両（窓・影の帯）
        (232, 108, 24),    # 障害物（パイロン）
        (38, 42, 40),      # 灯器の筐体
        (58, 60, 58),      # 消灯
        (0, 190, 130),     # 青信号（日本の青は青緑）
        (250, 200, 20),    # 黄信号
        (235, 40, 35),     # 赤信号
        (205, 35, 35),     # 標識の赤縁
        (243, 243, 239),   # 標識の白地
        (18, 18, 18),      # 標識の数字
        (150, 150, 152),   # 支柱
    ],
    dtype=np.uint8,
)

#: 灯色（0=青 / 1=黄 / 2=赤）-> ラベル。`world.signal_phases` の並びに合わせる
_PHASE_LABEL = np.array([LBL_LAMP_GREEN, LBL_LAMP_YELLOW, LBL_LAMP_RED], dtype=np.uint8)


# ---------------------------------------------------------------------------
# 実物の寸法（フロントエンドの定義と一致させること）
# ---------------------------------------------------------------------------

# `frontend/src/scene/signalGeometry.ts` より。灯器の向き・高さ・灯火の並びは
# 日本の車両用信号機（横型 3 灯・運転者から見て左から青黄赤）に合わせてある。
_SIGNAL_MOUNT_HEIGHT = 5.0
_SIGNAL_HOUSING_W = 1.16
_SIGNAL_HOUSING_H = 0.44
_SIGNAL_LAMP_PITCH = 0.35
_SIGNAL_LAMP_RADIUS = 0.15
_SIGNAL_HEAD_Z = _SIGNAL_MOUNT_HEIGHT + _SIGNAL_HOUSING_H * 0.5

# `frontend/src/scene/signGeometry.ts` / `config.SPEED_SIGN_*` より
_SIGN_RADIUS = config.SPEED_SIGN_DIAMETER * 0.5
_SIGN_CENTER_Z = config.SPEED_SIGN_BOTTOM_HEIGHT + _SIGN_RADIUS

#: 建物の高さ。占有グリッドは高さを持たないので既定値で描く。
#: 銀座では 1,299 棟中 1,189 棟がこの既定値なので、実測との差は小さい。
#: 走行不能領域としての手がかりに要るのは「壁の足元がどこか」であって
#: 上端の高さではないため、ここで高さを持つ意味は薄い。
_BUILDING_HEIGHT = config.DEFAULT_BUILDING_HEIGHT

#: 7 セグメントの点灯表（数字 0〜9 × セグメント a,b,c,d,e,f,g）
_SEVEN_SEG = np.array(
    [
        [1, 1, 1, 1, 1, 1, 0],  # 0
        [0, 1, 1, 0, 0, 0, 0],  # 1
        [1, 1, 0, 1, 1, 0, 1],  # 2
        [1, 1, 1, 1, 0, 0, 1],  # 3
        [0, 1, 1, 0, 0, 1, 1],  # 4
        [1, 0, 1, 1, 0, 1, 1],  # 5
        [1, 0, 1, 1, 1, 1, 1],  # 6
        [1, 1, 1, 0, 0, 0, 0],  # 7
        [1, 1, 1, 1, 1, 1, 1],  # 8
        [1, 1, 1, 1, 0, 1, 1],  # 9
    ],
    dtype=bool,
)

#: 各セグメントの正規化矩形 (cx, cy, hw, hh)。数字枠を [0,1]x[0,1]（下向き +y）とする
_SEG_T = 0.20
_SEG_RECT = np.array(
    [
        (0.50, _SEG_T * 0.5, 0.50, _SEG_T * 0.5),                    # a 上
        (1.0 - _SEG_T * 0.5, 0.25, _SEG_T * 0.5, 0.25),              # b 右上
        (1.0 - _SEG_T * 0.5, 0.75, _SEG_T * 0.5, 0.25),              # c 右下
        (0.50, 1.0 - _SEG_T * 0.5, 0.50, _SEG_T * 0.5),              # d 下
        (_SEG_T * 0.5, 0.75, _SEG_T * 0.5, 0.25),                    # e 左下
        (_SEG_T * 0.5, 0.25, _SEG_T * 0.5, 0.25),                    # f 左上
        (0.50, 0.50, 0.50, _SEG_T * 0.5),                            # g 中
    ],
    dtype=np.float32,
)

#: 数字を描く最小の大きさ [px]。これ未満は潰れて読めないので描かない。
#: 「小さいのに読める」嘘の画像を作らないための下限（types.py の解像度の議論と同じ）
_DIGIT_MIN_HW = 0.45
_DIGIT_MIN_HH = 1.2


# ---------------------------------------------------------------------------
# 道路ラスタ
# ---------------------------------------------------------------------------

#: ラスタの目標セル寸法 [m]。0.15m の区画線が 1 セルに乗る細かさ
_RASTER_CELL_M = 0.25
#: ラスタの一辺の最大セル数。広域プリセット（金沢は一辺 12.3km）で
#: メモリが破裂しないよう、大きいマップではセルを粗くして総量を抑える
_RASTER_MAX_SIDE = 6000
#: ラスタ焼き込みの 1 チャンクあたり点数。中間配列のメモリを一定に保つ
_RASTER_CHUNK_POINTS = 400_000

#: 区画線の寸法（`frontend/src/scene/RoadMarkings.tsx` と同じ）
_MARK_HALF_WIDTH = 0.075
_MARK_EDGE_INSET = 0.35
_MARK_DASH_ON = 5.0
_MARK_DASH_PERIOD = 10.0

#: 近傍検索のグリッドを使わずに全点を総当たりする上限。
#: 銀座の信号 261 基程度なら総当たりのほうが速い
_SMALL_POINT_SET = 1500


def _ragged_arange(counts: np.ndarray) -> np.ndarray:
    """`[arange(c) for c in counts]` を連結したものを返す（0 も許す）。"""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total <= 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.zeros(counts.size, dtype=np.int64)
    np.cumsum(counts[:-1], out=offsets[1:])
    out = np.arange(total, dtype=np.int64)
    out -= np.repeat(offsets, counts)
    return out


def _ragged_range(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """`[arange(s, s+c) for s, c in zip(starts, counts)]` の連結。"""
    return np.repeat(np.asarray(starts, dtype=np.int64), counts) + _ragged_arange(counts)


class _NeighborIndex:
    """点群を一様グリッドに入れて、半径内の候補を返す索引。

    信号機・標識は広域プリセットでは数万個ある（金沢は標識 27,583 基）。
    カメラ 8 台ぶん全点に投影計算を掛けると、それだけで 1 ステップの予算を
    超えるので、視程 `radius` の範囲だけに絞る。

    セル一辺を `2 * radius` にしてあるので、問い合わせ 1 回が触るセルは
    最大 2x2 の 4 つ。結果はセル単位でキャッシュするので、車がセルを跨がない
    限り 2 回目以降は辞書引き 1 回で済む。
    """

    def __init__(self, xs: np.ndarray, ys: np.ndarray, radius: float) -> None:
        self._count = int(xs.size)
        self._all = np.arange(self._count, dtype=np.int64)
        self._small = self._count <= _SMALL_POINT_SET
        self._cache: dict[tuple[int, int], np.ndarray] = {}
        if self._small or self._count == 0:
            return

        self._cell = max(float(radius) * 2.0, 1.0)
        ci = np.floor(xs / self._cell).astype(np.int64)
        ri = np.floor(ys / self._cell).astype(np.int64)
        self._buckets: dict[tuple[int, int], np.ndarray] = {}
        order = np.lexsort((ci, ri))
        keys = list(zip(ri[order].tolist(), ci[order].tolist()))
        start = 0
        for i in range(1, len(keys) + 1):
            if i == len(keys) or keys[i] != keys[start]:
                self._buckets[keys[start]] = order[start:i]
                start = i

    def query(self, x: float, y: float) -> np.ndarray:
        if self._small or self._count == 0:
            return self._all
        cr = int(math.floor(y / self._cell))
        cc = int(math.floor(x / self._cell))
        hit = self._cache.get((cr, cc))
        if hit is None:
            parts = [
                self._buckets[(cr + dr, cc + dc)]
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (cr + dr, cc + dc) in self._buckets
            ]
            hit = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
            self._cache[(cr, cc)] = hit
        return hit


class PseudoCamera:
    """運転席視点の擬似カメラ。

    `render()` は指定スロットぶんの画像を (N, H, W, 3) uint8 で返す。
    構築は 1 回・呼び出しは毎ステップという使われ方なので、カメラに依存しない
    量（逆透視変換の係数・道路ラスタ・近傍索引）はここで全部作っておく。
    """

    def __init__(self, map_index: MapIndex, spec: CameraSpec = DEFAULT_CAMERA) -> None:
        self.map_index = map_index
        self.spec = spec

        self._w = int(spec.width)
        self._h = int(spec.height)
        self._focal = float(spec.focal_px)
        self._cx = self._w * 0.5
        self._cy = self._h * 0.5
        self._near = float(spec.near)
        self._far = float(spec.far)
        self._eye_h = float(spec.eye_height)

        pitch = float(spec.pitch)
        self._cp = math.cos(pitch)
        self._sp = math.sin(pitch)

        self._prepare_ground()
        self._prepare_columns()
        self._build_road_raster()
        self._prepare_signals()
        self._prepare_signs()

    # ------------------------------------------------------------------
    # 事前計算（カメラの姿勢に依存しない部分）
    # ------------------------------------------------------------------

    def _prepare_ground(self) -> None:
        """地平線と、地面画素の逆透視変換の係数を作る。

        画素 (u, v) の視線は  d = f + a(u)*right + b(v)*up  で、
        `right` は水平なので **d の鉛直成分は行 v だけで決まる**。
        よって地面 z=0 までの距離 t も行だけの関数になり、地面上の点は

            P = eye + g(v) * (cos h, sin h) + q(v, u) * (sin h, -cos h)

        と書ける（g, q はカメラの向きに依らない）。カメラごとに要るのは
        この 2 つを heading で回して足すだけになる。
        """
        w, h = self._w, self._h
        focal = self._focal

        # 地平線: d の鉛直成分が 0 になる行。俯角ぶん画面中央より上に来る
        horizon = self._cy + focal * math.tan(math.atan2(self._sp, self._cp))
        self._horizon_row = int(np.clip(math.floor(horizon) + 1, 0, h))

        rows = np.arange(self._horizon_row, h, dtype=np.float64)
        if rows.size == 0:
            self._ground_g = np.zeros((0, 1), dtype=np.float32)
            self._ground_q = np.zeros((0, w), dtype=np.float32)
            self._ground_ok = np.zeros((0, w), dtype=bool)
            return

        b = (self._cy - (rows + 0.5)) / focal
        dz = self._sp + b * self._cp          # 地平線より下なので必ず負
        dz = np.minimum(dz, -1e-6)
        t = -self._eye_h / dz
        g = t * (self._cp - b * self._sp)

        a = ((np.arange(w, dtype=np.float64) + 0.5) - self._cx) / focal
        q = t[:, None] * a[None, :]

        # 視程の外（spec.far より遠い）は地面のまま残す。
        # (cos h, sin h) と (sin h, -cos h) は直交系なので距離は g, q だけで出る
        dist2 = g[:, None] ** 2 + q ** 2
        self._ground_ok = dist2 <= (self._far * self._far)
        self._ground_g = g.astype(np.float32)[:, None]
        self._ground_q = q.astype(np.float32)

    def _prepare_columns(self) -> None:
        """建物レイキャストに使う、画面の列ごとの水平方向。

        列の水平角は本来その画素の行にも僅かに依存するが、俯角が 2.12 度しか
        無いので画面端でも 0.5 度未満しかずれない。**画面の高さ方向を無視して
        列だけで決める**ことで、レイの本数を 1/144 に減らしている。
        """
        step = self._column_step = 2
        centers = np.arange(0, self._w, step, dtype=np.float64) + step * 0.5
        a = (centers - self._cx) / self._focal
        length = np.sqrt(self._cp * self._cp + a * a)
        self._col_centers = centers.astype(np.float32)
        self._col_a = (a / length).astype(np.float64)      # 進行方向右手の成分
        self._col_f = (self._cp / length).astype(np.float64)  # 進行方向の成分
        # レイの刻みと打ち切り。刻みは占有グリッドのセル（1m）に合わせる
        ray_max = min(self._far, 90.0)
        self._ray_samples = np.arange(1.0, ray_max + 1e-6, 1.0, dtype=np.float64)

    # ------------------------------------------------------------------
    # 道路ラスタ
    # ------------------------------------------------------------------

    def _build_road_raster(self) -> None:
        """路面と車線標示を ENU 平面のラベルラスタへ焼く。

        画面側では逆透視変換で世界座標を出して**参照 1 回**にしたいので、
        道路の形はここで全部つぶしておく。多角形のまま持つと、画素ごとに
        道路本数ぶんの内外判定が要り 20Hz には到底間に合わない。

        セル寸法はマップの大きさで決める。広域プリセット（金沢は一辺 12.3km）を
        0.25m で焼くと 24 億セルになるため、一辺のセル数に上限を設けて粗くする。
        """
        data = self.map_index.data
        bounds = data.bounds
        margin = float(config.GRID_MARGIN)
        min_x = bounds.min_x - margin
        min_y = bounds.min_y - margin
        span_x = (bounds.max_x + margin) - min_x
        span_y = (bounds.max_y + margin) - min_y
        span = max(span_x, span_y, 1.0)

        cell = max(_RASTER_CELL_M, span / _RASTER_MAX_SIDE)
        self._rcell = cell
        self._rinv = 1.0 / cell
        self._rox = min_x
        self._roy = min_y
        self._rw = max(int(math.ceil(span_x / cell)) + 1, 1)
        self._rh = max(int(math.ceil(span_y / cell)) + 1, 1)
        self._raster = np.zeros((self._rh, self._rw), dtype=np.uint8)  # 0 = 地面

        segments = self._collect_segments()
        if segments is None:
            self._raster_flat = self._raster.reshape(-1)
            return
        ax, ay, ux, uy, seg_len, half_w, oneway, arc0 = segments

        step = cell * 0.6  # セルより細かく打って塗り残しを防ぐ
        # --- 路面 ---
        self._stamp_band(ax, ay, ux, uy, seg_len, -half_w, half_w, LBL_ROAD, step)

        # --- 車道外側線（両端の実線）---
        edge_off = np.maximum(half_w - _MARK_EDGE_INSET, half_w * 0.5)
        mark_hw = max(_MARK_HALF_WIDTH, cell * 0.6)
        for sign in (1.0, -1.0):
            off = edge_off * sign
            self._stamp_band(
                ax, ay, ux, uy, seg_len, off - mark_hw, off + mark_hw, LBL_MARKING, step
            )

        # --- 車道中央線（対面通行のみ・破線）---
        # 追越し禁止の黄色線は描かない（OSM から判別できないため。README 参照）
        two_way = ~oneway
        if bool(two_way.any()):
            self._stamp_band(
                ax[two_way],
                ay[two_way],
                ux[two_way],
                uy[two_way],
                seg_len[two_way],
                np.full(int(two_way.sum()), -mark_hw, dtype=np.float64),
                np.full(int(two_way.sum()), mark_hw, dtype=np.float64),
                LBL_MARKING,
                step,
                arc0=arc0[two_way],
            )
        self._raster_flat = self._raster.reshape(-1)

    def _collect_segments(self):
        """全エッジのポリラインを 1 本の線分配列へ展開する。

        線分ごとに (始点, 単位方向, 長さ, 道路半幅, 一方通行か, エッジ内弧長) を持つ。
        弧長は破線の位相を決めるのに要る（線分ごとに 0 から数え直すと、
        折れ点のたびに破線が途切れる）。
        """
        ax: list[float] = []
        ay: list[float] = []
        bx: list[float] = []
        by: list[float] = []
        hw: list[float] = []
        ow: list[bool] = []
        arc: list[float] = []
        for edge in self.map_index.data.edges:
            poly = edge.polyline
            if len(poly) < 2:
                continue
            half = float(edge.width) * 0.5
            one = bool(edge.oneway)
            run = 0.0
            for i in range(len(poly) - 1):
                x0, y0 = float(poly[i][0]), float(poly[i][1])
                x1, y1 = float(poly[i + 1][0]), float(poly[i + 1][1])
                length = math.hypot(x1 - x0, y1 - y0)
                if length < 1e-6:
                    continue
                ax.append(x0)
                ay.append(y0)
                bx.append(x1)
                by.append(y1)
                hw.append(half)
                ow.append(one)
                arc.append(run)
                run += length
        if not ax:
            return None

        ax_a = np.asarray(ax, dtype=np.float64)
        ay_a = np.asarray(ay, dtype=np.float64)
        bx_a = np.asarray(bx, dtype=np.float64)
        by_a = np.asarray(by, dtype=np.float64)
        seg_len = np.hypot(bx_a - ax_a, by_a - ay_a)
        ux = (bx_a - ax_a) / seg_len
        uy = (by_a - ay_a) / seg_len
        return (
            ax_a,
            ay_a,
            ux,
            uy,
            seg_len,
            np.asarray(hw, dtype=np.float64),
            np.asarray(ow, dtype=bool),
            np.asarray(arc, dtype=np.float64),
        )

    def _stamp_band(
        self,
        ax: np.ndarray,
        ay: np.ndarray,
        ux: np.ndarray,
        uy: np.ndarray,
        seg_len: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        value: int,
        step: float,
        arc0: np.ndarray | None = None,
    ) -> None:
        """線分に沿った帯（横方向 lo〜hi）をラスタへ塗る。

        線分ごとのループは持たず、「線分内の通し番号」を ragged 展開して
        全線分ぶんを一度に世界座標へ直す。中間配列が膨らまないよう
        点数でチャンクに切る（金沢は道路総延長 4,345km ある）。

        `arc0` を渡すと破線（5m 実線 / 5m 空白）にする。
        """
        n_along = np.ceil(seg_len / step).astype(np.int64) + 1
        n_across = np.ceil((hi - lo) / step).astype(np.int64) + 1
        counts = n_along * n_across
        if counts.size == 0:
            return

        edges = np.concatenate([[0], np.cumsum(counts)])
        total = int(edges[-1])
        if total == 0:
            return

        # チャンク境界（線分の区切りで切る）
        chunk_starts = [0]
        pos = 0
        while pos < counts.size:
            end = int(np.searchsorted(edges, edges[pos] + _RASTER_CHUNK_POINTS))
            end = max(min(end, counts.size), pos + 1)
            chunk_starts.append(end)
            pos = end

        rw, rh = self._rw, self._rh
        raster = self._raster
        for s, e in zip(chunk_starts[:-1], chunk_starts[1:]):
            cnt = counts[s:e]
            k = _ragged_arange(cnt)
            seg = np.repeat(np.arange(s, e, dtype=np.int64), cnt)
            nc = n_across[seg]
            ia = k // nc
            ic = k - ia * nc

            ta = np.minimum(ia * step, seg_len[seg])
            tc = np.minimum(ic * step, hi[seg] - lo[seg]) + lo[seg]

            x = ax[seg] + ux[seg] * ta - uy[seg] * tc
            y = ay[seg] + uy[seg] * ta + ux[seg] * tc

            if arc0 is not None:
                keep = ((arc0[seg] + ta) % _MARK_DASH_PERIOD) < _MARK_DASH_ON
                x = x[keep]
                y = y[keep]
                if x.size == 0:
                    continue

            ci = ((x - self._rox) * self._rinv).astype(np.int32)
            ri = ((y - self._roy) * self._rinv).astype(np.int32)
            inside = (ci >= 0) & (ci < rw) & (ri >= 0) & (ri < rh)
            if not inside.all():
                ci = ci[inside]
                ri = ri[inside]
            raster[ri, ci] = value

    # ------------------------------------------------------------------
    # 信号機・標識の事前計算
    # ------------------------------------------------------------------

    def _prepare_signals(self) -> None:
        """灯器（3 灯のバー）の中心と向きを求めておく。

        位置の決め方は `frontend/src/scene/signalGeometry.ts` と同じ。
        交差点の対面側へ `roadWidth/2 + 2m` 進み、進行方向左へ `roadWidth/4`
        寄せた位置に灯器が来る。ここを合わせておかないと、バックエンドの
        検出結果を Three.js の映像へ重ねたときに灯器と箱がずれる。
        """
        signals = self.map_index.data.signals
        n = len(signals)
        self._sig_count = n
        if n == 0:
            self._sig_grid = _NeighborIndex(
                np.zeros(0), np.zeros(0), self._far
            )
            return

        node_xy = {nd.id: (nd.x, nd.y) for nd in self.map_index.data.nodes}
        heading = np.array([s.heading for s in signals], dtype=np.float64)
        road_w = np.array([s.road_width for s in signals], dtype=np.float64)
        cx = np.array(
            [node_xy.get(s.node_id, (s.x, s.y))[0] for s in signals], dtype=np.float64
        )
        cy = np.array(
            [node_xy.get(s.node_id, (s.x, s.y))[1] for s in signals], dtype=np.float64
        )

        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        left_x = -sin_h
        left_y = cos_h
        beyond = road_w * 0.5 + 2.0
        self._sig_x = cx + cos_h * beyond + left_x * (road_w * 0.25)
        self._sig_y = cy + sin_h * beyond + left_y * (road_w * 0.25)
        # 灯器の並びは「運転者から見て左」= 進行方向左
        self._sig_ax = left_x
        self._sig_ay = left_y
        # 灯器は進入車両に正対する = 面の法線は進行方向の逆
        self._sig_nx = -cos_h
        self._sig_ny = -sin_h
        self._sig_grid = _NeighborIndex(self._sig_x, self._sig_y, self._far)

    def _prepare_signs(self) -> None:
        """最高速度標識の板の中心・向き・表示する数字を求めておく。"""
        signs = self.map_index.data.signs
        n = len(signs)
        self._sign_count = n
        if n == 0:
            self._sign_grid = _NeighborIndex(np.zeros(0), np.zeros(0), self._far)
            return

        heading = np.array([s.heading for s in signs], dtype=np.float64)
        self._sign_x = np.array([s.x for s in signs], dtype=np.float64)
        self._sign_y = np.array([s.y for s in signs], dtype=np.float64)
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        self._sign_ax = -sin_h   # 板の横方向（水平・法線に直交）
        self._sign_ay = cos_h
        self._sign_nx = -cos_h   # 板は heading + pi を向く
        self._sign_ny = -sin_h

        kph = np.rint(
            np.array([s.speed_limit for s in signs], dtype=np.float64) * 3.6
        ).astype(np.int32)
        kph = np.clip(kph, 0, 999)
        # 右詰め 3 桁。先頭の 0 は「消灯」を意味する -1 にする
        digits = np.stack([kph // 100, (kph // 10) % 10, kph % 10], axis=1)
        digits[:, 0] = np.where(kph >= 100, digits[:, 0], -1)
        digits[:, 1] = np.where(kph >= 10, digits[:, 1], -1)
        self._sign_digits = digits.astype(np.int8)
        self._sign_grid = _NeighborIndex(self._sign_x, self._sign_y, self._far)

    # ------------------------------------------------------------------
    # 描画
    # ------------------------------------------------------------------

    def render(self, world: "World", slots: np.ndarray) -> np.ndarray:
        """指定スロットの運転席視点を描く。戻り値 (len(slots), H, W, 3) uint8。"""
        slots = np.asarray(slots, dtype=np.int64).reshape(-1)
        n = int(slots.size)
        h, w = self._h, self._w
        if n == 0:
            return np.zeros((0, h, w, 3), dtype=np.uint8)

        fleet = world.fleet
        heading = fleet.heading[slots].astype(np.float64)
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        # 運転席は車体中心から前へ forward、右へ right（日本車の右ハンドル）
        eye_x = fleet.x[slots].astype(np.float64) + cos_h * self.spec.forward + sin_h * self.spec.right
        eye_y = fleet.y[slots].astype(np.float64) + sin_h * self.spec.forward - cos_h * self.spec.right

        label = np.empty((n, h, w), dtype=np.uint8)
        label[:, : self._horizon_row, :] = LBL_SKY
        label[:, self._horizon_row :, :] = LBL_GROUND

        self._draw_ground(label, eye_x, eye_y, cos_h, sin_h)

        specs = []
        walls = self._collect_buildings(eye_x, eye_y, cos_h, sin_h)
        if walls is not None:
            specs.append(walls)
        for part in (
            self._collect_vehicles(world, slots, eye_x, eye_y, cos_h, sin_h),
            self._collect_obstacles(world, eye_x, eye_y, cos_h, sin_h),
            self._collect_signals(world, eye_x, eye_y, cos_h, sin_h),
            self._collect_signs(eye_x, eye_y, cos_h, sin_h),
        ):
            if part is not None:
                specs.append(part)
        self._draw_shapes(label, specs)

        return PALETTE[label]

    def _draw_ground(
        self,
        label: np.ndarray,
        eye_x: np.ndarray,
        eye_y: np.ndarray,
        cos_h: np.ndarray,
        sin_h: np.ndarray,
    ) -> None:
        """地平線より下を、道路ラスタの参照 1 回で路面／車線標示／地面に塗る。"""
        g = self._ground_g
        q = self._ground_q
        if q.shape[0] == 0:
            return

        cosb = cos_h[:, None, None]
        sinb = sin_h[:, None, None]
        px = eye_x[:, None, None] + g[None] * cosb + q[None] * sinb
        py = eye_y[:, None, None] + g[None] * sinb - q[None] * cosb

        ci = ((px - self._rox) * self._rinv).astype(np.int32)
        ri = ((py - self._roy) * self._rinv).astype(np.int32)
        # 範囲外はマスクを作らずに端へ丸める。ラスタの外周は余白（= 地面 0）なので
        # 丸めた先も地面になり、結果は同じで配列を 1 枚減らせる
        np.clip(ci, 0, self._rw - 1, out=ci)
        np.clip(ri, 0, self._rh - 1, out=ri)
        ri *= self._rw
        ri += ci
        values = self._raster_flat[ri]
        np.copyto(
            label[:, self._horizon_row :, :],
            values,
            where=self._ground_ok[None],
        )

    def _collect_buildings(
        self,
        eye_x: np.ndarray,
        eye_y: np.ndarray,
        cos_h: np.ndarray,
        sin_h: np.ndarray,
    ):
        """画面の列ごとに建物までの距離を測り、壁の上端・下端を投影する。

        占有グリッドは建物の有無しか持たないので、壁面を多角形として投影する
        代わりに列ごとのレイキャストで距離を出す。列数は画面幅の半分に間引く
        （建物の輪郭が 2px 単位になるが、走行不能領域の手がかりには十分）。
        """
        occ = self.map_index.occupancy
        grid = occ.building
        if grid.size == 0:
            return None

        n = eye_x.size
        wx = self._col_f[None, :] * cos_h[:, None] + self._col_a[None, :] * sin_h[:, None]
        wy = self._col_f[None, :] * sin_h[:, None] - self._col_a[None, :] * cos_h[:, None]

        samples = self._ray_samples
        xs = eye_x[:, None, None] + wx[:, :, None] * samples[None, None, :]
        ys = eye_y[:, None, None] + wy[:, :, None] * samples[None, None, :]

        gw = grid.shape[1]
        gh = grid.shape[0]
        inv = 1.0 / occ.cell_size
        ci = np.rint((xs - occ.origin_x) * inv).astype(np.int32)
        ri = np.rint((ys - occ.origin_y) * inv).astype(np.int32)
        np.clip(ci, 0, gw - 1, out=ci)
        np.clip(ri, 0, gh - 1, out=ri)
        ri *= gw
        ri += ci
        hits = grid.reshape(-1)[ri]

        any_hit = hits.any(axis=2)
        if not any_hit.any():
            return None
        first = hits.argmax(axis=2)
        dist = samples[first]

        # 壁の足元（z=0）と上端（z=建物高さ）を投影する
        fwd = dist * self._col_f[None, :]
        z_bot = -self._eye_h
        z_top = _BUILDING_HEIGHT - self._eye_h
        zc_bot = self._cp * fwd + self._sp * z_bot
        zc_top = self._cp * fwd + self._sp * z_top
        ok = any_hit & (zc_bot > self._near) & (zc_top > self._near)
        if not ok.any():
            return None

        v_bot = self._cy - self._focal * (-self._sp * fwd + self._cp * z_bot) / zc_bot
        v_top = self._cy - self._focal * (-self._sp * fwd + self._cp * z_top) / zc_top

        cam = np.repeat(np.arange(n, dtype=np.int32), self._col_centers.size).reshape(
            n, -1
        )[ok]
        cxf = np.broadcast_to(self._col_centers[None, :], ok.shape)[ok]
        vb = v_bot[ok]
        vt = v_top[ok]
        return (
            cam,
            cxf.astype(np.float32),
            ((vt + vb) * 0.5).astype(np.float32),
            np.full(cam.size, self._column_step * 0.5, dtype=np.float32),
            ((vb - vt) * 0.5).astype(np.float32),
            np.zeros(cam.size, dtype=bool),
            np.full(cam.size, LBL_BUILDING, dtype=np.uint8),
            zc_bot[ok].astype(np.float32),
            np.zeros(cam.size, dtype=np.int16),
        )

    def _project(
        self,
        px: np.ndarray,
        py: np.ndarray,
        pz: np.ndarray,
        eye_x: np.ndarray,
        eye_y: np.ndarray,
        cos_h: np.ndarray,
        sin_h: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """世界座標を画像座標へ落とす。戻り値は (u, v, 奥行き Zc)。

        `px` などは呼び出し側でカメラ添字に合わせて展開済みであること。
        Zc <= near の点は呼び出し側で捨てる（**カメラの後ろの点を割ると
        座標が反転して、画面に存在しない物体が現れる**）。
        """
        rel_x = px - eye_x
        rel_y = py - eye_y
        fwd = rel_x * cos_h + rel_y * sin_h
        lat = rel_x * sin_h - rel_y * cos_h
        rel_z = pz - self._eye_h
        zc = self._cp * fwd + self._sp * rel_z
        yc = -self._sp * fwd + self._cp * rel_z
        safe = np.maximum(zc, 1e-3)
        u = self._cx + self._focal * lat / safe
        v = self._cy - self._focal * yc / safe
        return u, v, zc

    def _collect_vehicles(
        self,
        world: "World",
        slots: np.ndarray,
        eye_x: np.ndarray,
        eye_y: np.ndarray,
        cos_h: np.ndarray,
        sin_h: np.ndarray,
    ):
        """他車両を、車体の外接直方体の投影から作った矩形で描く。

        寸法を実車どおりにしてあるので、画面上の幅から既知の車幅を使った
        単眼測距（`config.OBS_VEHICLE_*`）が成り立つ。
        """
        active = np.flatnonzero(world.fleet.active)
        if active.size == 0:
            return None
        n = eye_x.size
        corners = world.fleet.corners()[active].astype(np.float64)  # (K, 4, 2)
        k = active.size

        cxw = corners[None, :, :, 0]
        cyw = corners[None, :, :, 1]
        rel_x = cxw - eye_x[:, None, None]
        rel_y = cyw - eye_y[:, None, None]
        fwd = rel_x * cos_h[:, None, None] + rel_y * sin_h[:, None, None]
        lat = rel_x * sin_h[:, None, None] - rel_y * cos_h[:, None, None]

        heights = np.array([0.0, config.VEHICLE_HEIGHT], dtype=np.float64) - self._eye_h
        zc = self._cp * fwd[..., None] + self._sp * heights          # (n, k, 4, 2)
        yc = -self._sp * fwd[..., None] + self._cp * heights
        xc = np.broadcast_to(lat[..., None], zc.shape)

        flat_zc = zc.reshape(n, k, -1)
        visible = (flat_zc.min(axis=2) > self._near) & (flat_zc.min(axis=2) < self._far)
        # 自車は写さない（カメラは自分の運転席にある）
        visible &= active[None, :] != slots[:, None]
        if not visible.any():
            return None

        safe = np.maximum(flat_zc, 1e-3)
        u = self._cx + self._focal * xc.reshape(n, k, -1) / safe
        v = self._cy - self._focal * yc.reshape(n, k, -1) / safe
        u0 = u.min(axis=2)
        u1 = u.max(axis=2)
        v0 = v.min(axis=2)
        v1 = v.max(axis=2)
        depth = flat_zc.mean(axis=2)

        cam = np.broadcast_to(np.arange(n, dtype=np.int32)[:, None], visible.shape)[visible]
        cxf = ((u0 + u1) * 0.5)[visible].astype(np.float32)
        cyf = ((v0 + v1) * 0.5)[visible].astype(np.float32)
        hw = ((u1 - u0) * 0.5)[visible].astype(np.float32)
        hh = ((v1 - v0) * 0.5)[visible].astype(np.float32)
        dep = depth[visible].astype(np.float32)

        # 車体（明色）＋上半分の窓の帯（暗色）。上下 2 色にしておくと、
        # 影も質感も無い画像でも「路面ではなく立体物」の手がかりになる
        m = cam.size
        return (
            np.concatenate([cam, cam]),
            np.concatenate([cxf, cxf]),
            np.concatenate([cyf, cyf - hh * 0.5]),
            np.concatenate([hw, hw * 0.8]),
            np.concatenate([hh, hh * 0.35]),
            np.zeros(m * 2, dtype=bool),
            np.concatenate(
                [
                    np.full(m, LBL_VEHICLE, dtype=np.uint8),
                    np.full(m, LBL_VEHICLE_DARK, dtype=np.uint8),
                ]
            ),
            np.concatenate([dep, dep]),
            np.concatenate([np.zeros(m, dtype=np.int16), np.ones(m, dtype=np.int16)]),
        )

    def _collect_obstacles(
        self,
        world: "World",
        eye_x: np.ndarray,
        eye_y: np.ndarray,
        cos_h: np.ndarray,
        sin_h: np.ndarray,
    ):
        """パイロンを矩形で描く。"""
        xy = world.obstacle_xy
        if xy.shape[0] == 0:
            return None
        n = eye_x.size
        radius = np.fromiter(
            (o.radius for o in world.obstacles), dtype=np.float64, count=len(world.obstacles)
        )
        if radius.size != xy.shape[0]:  # 直前に増減した場合の保険
            radius = np.full(xy.shape[0], config.OBSTACLE_RADIUS, dtype=np.float64)

        ox = np.broadcast_to(xy[None, :, 0].astype(np.float64), (n, xy.shape[0]))
        oy = np.broadcast_to(xy[None, :, 1].astype(np.float64), (n, xy.shape[0]))
        u_top, v_top, zc = self._project(
            ox,
            oy,
            np.full_like(ox, config.OBSTACLE_HEIGHT),
            eye_x[:, None],
            eye_y[:, None],
            cos_h[:, None],
            sin_h[:, None],
        )
        _u_bot, v_bot, _z = self._project(
            ox,
            oy,
            np.zeros_like(ox),
            eye_x[:, None],
            eye_y[:, None],
            cos_h[:, None],
            sin_h[:, None],
        )
        visible = (zc > self._near) & (zc < self._far)
        if not visible.any():
            return None
        hw = (self._focal * radius[None, :] / np.maximum(zc, 1e-3)).astype(np.float32)

        cam = np.broadcast_to(np.arange(n, dtype=np.int32)[:, None], visible.shape)[visible]
        return (
            cam,
            u_top[visible].astype(np.float32),
            ((v_top + v_bot) * 0.5)[visible].astype(np.float32),
            hw[visible],
            ((v_bot - v_top) * 0.5)[visible].astype(np.float32),
            np.zeros(cam.size, dtype=bool),
            np.full(cam.size, LBL_OBSTACLE, dtype=np.uint8),
            zc[visible].astype(np.float32),
            np.zeros(cam.size, dtype=np.int16),
        )

    def _pairs(self, grid: _NeighborIndex, eye_x: np.ndarray, eye_y: np.ndarray):
        """カメラごとに視程内の候補を引き、(カメラ添字, 物体添字) の組を返す。"""
        cams: list[np.ndarray] = []
        objs: list[np.ndarray] = []
        for i in range(eye_x.size):
            cand = grid.query(float(eye_x[i]), float(eye_y[i]))
            if cand.size:
                cams.append(np.full(cand.size, i, dtype=np.int32))
                objs.append(cand)
        if not cams:
            return None
        return np.concatenate(cams), np.concatenate(objs)

    def _collect_signals(
        self,
        world: "World",
        eye_x: np.ndarray,
        eye_y: np.ndarray,
        cos_h: np.ndarray,
        sin_h: np.ndarray,
    ):
        """信号機を「灯器の筐体 + 3 灯」で描く。

        **灯色が読めることが最優先**なので、点灯している灯だけを現示の色で塗り、
        残り 2 灯は消灯色にする。灯火は 30cm しかなく遠方では 1px を割るが、
        矩形は最低 1 画素を占めるようにしてあるので、遠くても色は残る
        （実物の灯火も自発光なので、幾何的な大きさより見えるのが自然）。
        """
        if self._sig_count == 0:
            return None
        pair = self._pairs(self._sig_grid, eye_x, eye_y)
        if pair is None:
            return None
        cam, sig = pair

        hx = self._sig_x[sig]
        hy = self._sig_y[sig]
        ex = eye_x[cam]
        ey = eye_y[cam]
        # 灯器の正面から見ているものだけ（背面は灯色が読めない）
        facing = (ex - hx) * self._sig_nx[sig] + (ey - hy) * self._sig_ny[sig] > 0.0
        if not facing.any():
            return None
        cam = cam[facing]
        sig = sig[facing]
        hx = hx[facing]
        hy = hy[facing]
        ex = ex[facing]
        ey = ey[facing]

        ch = cos_h[cam]
        sh = sin_h[cam]
        ax = self._sig_ax[sig]
        ay = self._sig_ay[sig]
        z = np.full(hx.shape, _SIGNAL_HEAD_Z)

        # 筐体の左右端（灯器は横型なので、横方向の見かけの幅は姿勢で変わる）
        half = _SIGNAL_HOUSING_W * 0.5
        u_l, _v_l, zc = self._project(hx + ax * half, hy + ay * half, z, ex, ey, ch, sh)
        u_r, v_c, _zc = self._project(hx - ax * half, hy - ay * half, z, ex, ey, ch, sh)
        visible = (zc > self._near) & (zc < self._far)
        if not visible.any():
            return None

        cam = cam[visible]
        sig = sig[visible]
        u_l = u_l[visible]
        u_r = u_r[visible]
        v_c = v_c[visible]
        zc = zc[visible].astype(np.float32)
        hx = hx[visible]
        hy = hy[visible]
        ex = ex[visible]
        ey = ey[visible]
        ax = ax[visible]
        ay = ay[visible]
        ch = ch[visible]
        sh = sh[visible]
        z = z[visible]

        scale = (self._focal / np.maximum(zc, 1e-3)).astype(np.float64)
        body_cx = (u_l + u_r) * 0.5
        body_hw = np.abs(u_r - u_l) * 0.5
        body_hh = _SIGNAL_HOUSING_H * 0.5 * scale

        phases = np.asarray(world.signal_phases, dtype=np.int32)
        phase = phases[sig] if phases.size else np.full(sig.shape, 2, dtype=np.int32)

        # 3 灯。運転者から見て左（= 進行方向左）から 青・黄・赤
        lamp_cam = []
        lamp_cx = []
        lamp_cy = []
        lamp_hw = []
        lamp_hh = []
        lamp_lbl = []
        lamp_dep = []
        for role in range(3):
            offset = (1 - role) * _SIGNAL_LAMP_PITCH
            u_k, v_k, _z = self._project(
                hx + ax * offset, hy + ay * offset, z, ex, ey, ch, sh
            )
            lamp_cam.append(cam)
            lamp_cx.append(u_k)
            lamp_cy.append(v_k)
            lamp_hw.append(_SIGNAL_LAMP_RADIUS * scale)
            lamp_hh.append(_SIGNAL_LAMP_RADIUS * scale)
            lamp_lbl.append(
                np.where(phase == role, _PHASE_LABEL[role], LBL_LAMP_OFF).astype(np.uint8)
            )
            lamp_dep.append(zc)

        m = cam.size
        return (
            np.concatenate([cam] + lamp_cam),
            np.concatenate([body_cx] + lamp_cx).astype(np.float32),
            np.concatenate([v_c] + lamp_cy).astype(np.float32),
            np.concatenate([body_hw] + lamp_hw).astype(np.float32),
            np.concatenate([body_hh] + lamp_hh).astype(np.float32),
            np.concatenate(
                [np.zeros(m, dtype=bool), np.ones(m * 3, dtype=bool)]
            ),  # 灯火は円板
            np.concatenate(
                [np.full(m, LBL_SIGNAL_BODY, dtype=np.uint8)] + lamp_lbl
            ),
            np.concatenate([zc] + lamp_dep),
            np.concatenate(
                [np.zeros(m, dtype=np.int16), np.ones(m * 3, dtype=np.int16)]
            ),
        )

    def _collect_signs(
        self,
        eye_x: np.ndarray,
        eye_y: np.ndarray,
        cos_h: np.ndarray,
        sin_h: np.ndarray,
    ):
        """最高速度標識を「支柱 + 赤縁の円 + 白地 + 7 セグの数字」で描く。

        規制速度ごとに見分けがつく必要があるので、板の中に数字を描く。
        ただし 1 画素を割る大きさになったら**描かない**。読めない距離で
        読めるように描くと、認識器に「入力に無い情報」を学習させることになる。
        """
        if self._sign_count == 0:
            return None
        pair = self._pairs(self._sign_grid, eye_x, eye_y)
        if pair is None:
            return None
        cam, sgn = pair

        sx = self._sign_x[sgn]
        sy = self._sign_y[sgn]
        ex = eye_x[cam]
        ey = eye_y[cam]
        facing = (ex - sx) * self._sign_nx[sgn] + (ey - sy) * self._sign_ny[sgn] > 0.0
        if not facing.any():
            return None
        cam, sgn, sx, sy, ex, ey = (
            cam[facing],
            sgn[facing],
            sx[facing],
            sy[facing],
            ex[facing],
            ey[facing],
        )
        ch = cos_h[cam]
        sh = sin_h[cam]
        ax = self._sign_ax[sgn]
        ay = self._sign_ay[sgn]
        zc_center = np.full(sx.shape, _SIGN_CENTER_Z)

        u_l, v_c, zc = self._project(
            sx + ax * _SIGN_RADIUS, sy + ay * _SIGN_RADIUS, zc_center, ex, ey, ch, sh
        )
        u_r, _v, _z = self._project(
            sx - ax * _SIGN_RADIUS, sy - ay * _SIGN_RADIUS, zc_center, ex, ey, ch, sh
        )
        visible = (zc > self._near) & (zc < self._far)
        if not visible.any():
            return None
        cam, sgn = cam[visible], sgn[visible]
        u_l, u_r, v_c = u_l[visible], u_r[visible], v_c[visible]
        zc = zc[visible].astype(np.float32)
        sx, sy, ex, ey, ch, sh = (
            sx[visible],
            sy[visible],
            ex[visible],
            ey[visible],
            ch[visible],
            sh[visible],
        )

        scale = self._focal / np.maximum(zc.astype(np.float64), 1e-3)
        board_cx = (u_l + u_r) * 0.5
        board_hw = np.maximum(np.abs(u_r - u_l) * 0.5, 0.1)
        board_hh = _SIGN_RADIUS * scale

        # 支柱（路面から板の下端まで）
        _up, v_ground, _z2 = self._project(
            sx, sy, np.zeros_like(sx), ex, ey, ch, sh
        )
        v_bottom = v_c + board_hh
        pole_hw = np.maximum(config.SPEED_SIGN_POLE_RADIUS * scale, 0.05)

        # 白地（赤縁は板の直径のおよそ 1 割）
        face_hw = board_hw * 0.78
        face_hh = board_hh * 0.78

        m = cam.size
        cams = [cam, cam, cam]
        cxs = [board_cx, board_cx, board_cx]
        cys = [(v_ground + v_bottom) * 0.5, v_c, v_c]
        hws = [pole_hw, board_hw, face_hw]
        hhs = [np.abs(v_ground - v_bottom) * 0.5, board_hh, face_hh]
        ell = [np.zeros(m, dtype=bool), np.ones(m, dtype=bool), np.ones(m, dtype=bool)]
        lbl = [
            np.full(m, LBL_POLE, dtype=np.uint8),
            np.full(m, LBL_SIGN_RIM, dtype=np.uint8),
            np.full(m, LBL_SIGN_FACE, dtype=np.uint8),
        ]
        dep = [zc, zc, zc]
        order = [
            np.zeros(m, dtype=np.int16),
            np.ones(m, dtype=np.int16),
            np.full(m, 2, dtype=np.int16),
        ]

        digits = self._sign_digit_shapes(cam, sgn, board_cx, v_c, face_hw, face_hh, zc)
        if digits is not None:
            cams.append(digits[0])
            cxs.append(digits[1])
            cys.append(digits[2])
            hws.append(digits[3])
            hhs.append(digits[4])
            ell.append(digits[5])
            lbl.append(digits[6])
            dep.append(digits[7])
            order.append(digits[8])

        return (
            np.concatenate(cams),
            np.concatenate(cxs).astype(np.float32),
            np.concatenate(cys).astype(np.float32),
            np.concatenate(hws).astype(np.float32),
            np.concatenate(hhs).astype(np.float32),
            np.concatenate(ell),
            np.concatenate(lbl),
            np.concatenate(dep).astype(np.float32),
            np.concatenate(order),
        )

    def _sign_digit_shapes(
        self,
        cam: np.ndarray,
        sgn: np.ndarray,
        board_cx: np.ndarray,
        board_cy: np.ndarray,
        face_hw: np.ndarray,
        face_hh: np.ndarray,
        depth: np.ndarray,
    ):
        """規制速度の数字を 7 セグメントの矩形群として作る。

        矩形にしておくと、標識・車両・建物とまったく同じ塗りの経路に乗るので
        文字専用の描画を持たずに済む。
        """
        digits = self._sign_digits[sgn]                    # (m, 3)
        used = digits >= 0
        n_used = used.sum(axis=1)
        m = cam.size
        if m == 0:
            return None

        # 数字 1 文字ぶんの枠。白地の内側に横並びで詰める
        field_hw = face_hw * 0.80
        field_hh = face_hh * 0.72
        cell_hw = field_hw / np.maximum(n_used, 1)
        # 右詰め: 使う桁だけを中央に寄せる
        slot = np.cumsum(used, axis=1) - 1                 # 使う桁の通し番号
        centre = (
            board_cx[:, None]
            - field_hw[:, None]
            + cell_hw[:, None] * (2 * slot + 1)
        )

        seg_cx = centre[:, :, None] + (_SEG_RECT[None, None, :, 0] - 0.5) * (
            cell_hw[:, None, None] * 2.0 * 0.9
        )
        seg_cy = board_cy[:, None, None] + (_SEG_RECT[None, None, :, 1] - 0.5) * (
            field_hh[:, None, None] * 2.0
        )
        seg_hw = _SEG_RECT[None, None, :, 2] * (cell_hw[:, None, None] * 2.0 * 0.9)
        seg_hh = _SEG_RECT[None, None, :, 3] * (field_hh[:, None, None] * 2.0)

        lit = np.zeros((m, 3, 7), dtype=bool)
        valid = used
        if valid.any():
            lit[valid] = _SEVEN_SEG[digits[valid]]
        # 潰れて読めない大きさの数字は描かない
        lit &= (seg_hw >= _DIGIT_MIN_HW * 0.5) & (seg_hh >= _DIGIT_MIN_HH * 0.5)
        lit &= (field_hh >= _DIGIT_MIN_HH)[:, None, None]
        if not lit.any():
            return None

        cam3 = np.broadcast_to(cam[:, None, None], lit.shape)[lit]
        dep3 = np.broadcast_to(depth[:, None, None], lit.shape)[lit]
        return (
            cam3.astype(np.int32),
            np.broadcast_to(seg_cx, lit.shape)[lit].astype(np.float32),
            np.broadcast_to(seg_cy, lit.shape)[lit].astype(np.float32),
            np.broadcast_to(seg_hw, lit.shape)[lit].astype(np.float32),
            np.broadcast_to(seg_hh, lit.shape)[lit].astype(np.float32),
            np.zeros(cam3.size, dtype=bool),
            np.full(cam3.size, LBL_SIGN_DIGIT, dtype=np.uint8),
            dep3.astype(np.float32),
            np.full(cam3.size, 3, dtype=np.int16),
        )

    # ------------------------------------------------------------------
    # スパン塗り（ペインターズアルゴリズム）
    # ------------------------------------------------------------------

    def _draw_shapes(self, label: np.ndarray, specs: list) -> None:
        """矩形／楕円のスパンを、奥から手前へ一括で塗る。

        物体ごとのループは持たない。全カメラ・全物体を 1 本の配列にまとめ、
        奥行きの降順（同じ奥行きなら部品の重ね順）に並べてから、
        行の展開 → 画素添字の展開 → ファンシー代入 1 回で書き込む。

        **同じ画素に複数回書くのは意図どおり**（手前の物体が後から上書きする）。
        numpy のファンシー代入は添字配列の順に書くので、並べ替えた順序が
        そのまま前後関係になる。
        """
        if not specs:
            return
        cam = np.concatenate([s[0] for s in specs]).astype(np.int32)
        if cam.size == 0:
            return
        cxf = np.concatenate([s[1] for s in specs])
        cyf = np.concatenate([s[2] for s in specs])
        hw = np.concatenate([s[3] for s in specs])
        hh = np.concatenate([s[4] for s in specs])
        ell = np.concatenate([s[5] for s in specs])
        lbl = np.concatenate([s[6] for s in specs])
        depth = np.concatenate([s[7] for s in specs])
        order = np.concatenate([s[8] for s in specs])

        h, w = self._h, self._w
        y0 = np.floor(cyf - hh).astype(np.int32)
        y1 = np.floor(cyf + hh).astype(np.int32)
        x0 = np.floor(cxf - hw).astype(np.int32)
        x1 = np.floor(cxf + hw).astype(np.int32)
        keep = (y1 >= 0) & (y0 < h) & (x1 >= 0) & (x0 < w) & (hw >= 0.0) & (hh >= 0.0)
        if not keep.any():
            return

        idx = np.flatnonzero(keep)
        # 奥から手前へ。lexsort は最後のキーが優先されるので、
        # 主キー = -depth（奥ほど先）、副キー = order（同じ物体の重ね順）
        idx = idx[np.lexsort((order[idx], -depth[idx]))]

        cam = cam[idx]
        cxf = cxf[idx]
        cyf = cyf[idx]
        hw = hw[idx]
        hh = hh[idx]
        ell = ell[idx]
        lbl = lbl[idx]
        y0 = np.clip(y0[idx], 0, h - 1)
        y1 = np.clip(y1[idx], 0, h - 1)

        n_rows = (y1 - y0 + 1).astype(np.int64)
        rows = _ragged_range(y0, n_rows)
        obj = np.repeat(np.arange(idx.size, dtype=np.int64), n_rows)
        if rows.size == 0:
            return

        # 楕円は行ごとに横幅が変わる（信号の灯火・標識の円板がこれ）
        row_hw = hw[obj].astype(np.float64)
        is_ell = ell[obj]
        if is_ell.any():
            hh_o = np.maximum(hh[obj].astype(np.float64), 1e-6)
            dy = (rows + 0.5 - cyf[obj]) / hh_o
            shrink = np.sqrt(np.maximum(1.0 - dy * dy, 0.0))
            row_hw = np.where(is_ell, row_hw * shrink, row_hw)

        cx_o = cxf[obj].astype(np.float64)
        sx0 = np.clip(np.floor(cx_o - row_hw).astype(np.int32), 0, w - 1)
        sx1 = np.clip(np.floor(cx_o + row_hw).astype(np.int32), 0, w - 1)
        lens = (sx1 - sx0 + 1).astype(np.int64)

        start = cam[obj].astype(np.int64) * (h * w) + rows * w + sx0
        pixels = _ragged_range(start, lens)
        values = np.repeat(lbl[obj], lens)
        label.reshape(-1)[pixels] = values
