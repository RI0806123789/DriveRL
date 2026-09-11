"""車線に沿った経路の生成（左側通行）。

道路中心線をそのまま走らせると中央線をまたぐことになるので、
**進行方向の左側の車線**へ寄せた経路を作る。交差点では前後の車線中心線が
つながらないため、両端を切り詰めてベジエ曲線で滑らかに接続する。

日本の道路交通法に合わせている点:

- **左側通行**（17条4項）。車線は進行方向の左半分だけを使う。
- **基本は一番左の車線**を走る（20条1項: 車両通行帯があるときは最も左側の通行帯）。
- **左折時**（34条1項）: あらかじめできる限り道路の左側端に寄る
  → 手前の区間で左端車線（lane 0）に入る。
- **右折時**（34条2項）: あらかじめできる限り道路の中央に寄る
  → 手前の区間で自分の進行方向の一番右の車線に移る。

車線変更は「経路の横方向オフセットを手前で滑らかに変える」ことで表現する。
別途の車線変更判断を持たせず経路に織り込むので、走行側は経路を追うだけでよい。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from app.contracts import MapEdge

__all__ = ["RouteSegment", "build_lane_route", "lane_count_for_direction", "lane_offset_left"]

Point = tuple[float, float]

#: これ以上曲がっていれば右左折とみなす [rad]（30 度）
TURN_ANGLE_THRESHOLD = math.radians(30.0)

#: 右左折の手前で車線を寄せ始める距離 [m]
LANE_CHANGE_LEAD_M = 35.0

#: 交差点でポリラインを切り詰める最小距離 [m]
MIN_TRIM_M = 3.0

#: 区間の何割までなら切り詰めてよいか
MAX_TRIM_RATIO = 0.4

#: ベジエの制御点を端点からどれだけ離すか（両端間距離に対する比）
BEZIER_HANDLE_RATIO = 0.45


@dataclass
class RouteSegment:
    """経路上の 1 区間。`points` は **進行方向に並べた** 中心線。"""

    edge: MapEdge
    points: list[Point]


# ---------------------------------------------------------------------------
# 車線の幾何
# ---------------------------------------------------------------------------


def lane_count_for_direction(edge: MapEdge) -> int:
    """この道路で自分の進行方向が使える車線数。"""
    lanes = max(1, int(edge.lanes))
    if edge.oneway:
        return lanes
    # 対面通行なら車線は両方向で分け合う
    return max(1, lanes // 2)


def lane_width_for_direction(edge: MapEdge) -> float:
    """自分の進行方向が使える幅を車線数で割った、1 車線あたりの幅 [m]。

    対面通行では**必ず中心線の左半分だけ**を使う。`lanes` を素直に割ると、
    車線数が奇数（データ上 lanes=1 の対面通行路など）のときに車線幅が広くなりすぎ、
    中心線の上を走ることになってしまう。
    """
    n = lane_count_for_direction(edge)
    usable = edge.width if edge.oneway else edge.width / 2.0
    return usable / n


def lane_offset_left(edge: MapEdge, lane_index: int) -> float:
    """車線中心が道路中心線からどれだけ**左**へ離れているか [m]。

    lane_index=0 が最も左（路肩側）。左側通行なので、対面通行路では
    自分の車線は常に中心線の左半分にある。
    """
    n = lane_count_for_direction(edge)
    lane_width = lane_width_for_direction(edge)
    idx = min(max(int(lane_index), 0), n - 1)
    return edge.width / 2.0 - lane_width * (idx + 0.5)


# ---------------------------------------------------------------------------
# ポリラインの小道具
# ---------------------------------------------------------------------------


def _cumulative(points: Sequence[Point]) -> list[float]:
    cum = [0.0]
    for i in range(1, len(points)):
        cum.append(cum[-1] + math.dist(points[i - 1], points[i]))
    return cum


def _resample(points: Sequence[Point], step: float) -> list[Point]:
    """等間隔にリサンプルする。端点は必ず残す。"""
    if len(points) < 2:
        return [tuple(p) for p in points]
    cum = _cumulative(points)
    total = cum[-1]
    if total <= 1e-9:
        return [tuple(points[0])]

    step = max(float(step), 0.1)
    count = max(1, int(round(total / step)))
    out: list[Point] = []
    seg = 0
    for k in range(count + 1):
        d = min(total, total * k / count)
        while seg < len(cum) - 2 and cum[seg + 1] < d:
            seg += 1
        span = cum[seg + 1] - cum[seg]
        t = 0.0 if span <= 1e-12 else (d - cum[seg]) / span
        x = points[seg][0] + (points[seg + 1][0] - points[seg][0]) * t
        y = points[seg][1] + (points[seg + 1][1] - points[seg][1]) * t
        out.append((x, y))
    return out


def _trim(points: Sequence[Point], from_start: float, from_end: float) -> list[Point]:
    """両端を指定距離だけ切り詰める。切り詰めすぎないよう長さで頭打ちにする。"""
    pts = [tuple(p) for p in points]
    if len(pts) < 2:
        return pts
    cum = _cumulative(pts)
    total = cum[-1]
    if total <= 1e-6:
        return pts

    limit = total * MAX_TRIM_RATIO
    a = min(max(0.0, from_start), limit)
    b = min(max(0.0, from_end), limit)
    if a + b >= total * 0.9:
        # 短すぎる区間。中央付近だけ残す
        a = b = total * 0.45

    start_d = a
    end_d = total - b
    if end_d - start_d < 0.5:
        mid = total / 2.0
        start_d, end_d = mid - 0.25, mid + 0.25

    return _slice(pts, cum, start_d, end_d)


def _slice(points: Sequence[Point], cum: Sequence[float], start_d: float, end_d: float) -> list[Point]:
    def at(d: float) -> Point:
        seg = 0
        while seg < len(cum) - 2 and cum[seg + 1] < d:
            seg += 1
        span = cum[seg + 1] - cum[seg]
        t = 0.0 if span <= 1e-12 else (d - cum[seg]) / span
        return (
            points[seg][0] + (points[seg + 1][0] - points[seg][0]) * t,
            points[seg][1] + (points[seg + 1][1] - points[seg][1]) * t,
        )

    out = [at(start_d)]
    for i, d in enumerate(cum):
        if start_d < d < end_d:
            out.append(tuple(points[i]))
    out.append(at(end_d))

    # 重複点を落とす
    cleaned = [out[0]]
    for p in out[1:]:
        if math.dist(cleaned[-1], p) > 1e-6:
            cleaned.append(p)
    return cleaned


def _tangents(points: Sequence[Point]) -> list[Point]:
    """各点での進行方向（前後の点の平均）。"""
    n = len(points)
    out: list[Point] = []
    for i in range(n):
        a = points[max(0, i - 1)]
        b = points[min(n - 1, i + 1)]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            out.append(out[-1] if out else (1.0, 0.0))
        else:
            out.append((dx / length, dy / length))
    return out


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _bezier(p0: Point, p1: Point, p2: Point, p3: Point, samples: int) -> list[Point]:
    out: list[Point] = []
    for k in range(1, samples):
        t = k / samples
        mt = 1.0 - t
        w0 = mt * mt * mt
        w1 = 3 * mt * mt * t
        w2 = 3 * mt * t * t
        w3 = t * t * t
        out.append(
            (
                p0[0] * w0 + p1[0] * w1 + p2[0] * w2 + p3[0] * w3,
                p0[1] * w0 + p1[1] * w1 + p2[1] * w2 + p3[1] * w3,
            )
        )
    return out


def _heading(points: Sequence[Point], at_end: bool) -> float:
    """区間の入口／出口での進行方位。"""
    if len(points) < 2:
        return 0.0
    if at_end:
        a, b = points[-2], points[-1]
    else:
        a, b = points[0], points[1]
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _angle_diff(a: float, b: float) -> float:
    """a - b を (-pi, pi] に畳む。左折が正、右折が負。"""
    return math.atan2(math.sin(a - b), math.cos(a - b))


# ---------------------------------------------------------------------------
# 車線経路の組み立て
# ---------------------------------------------------------------------------


def _target_lane_before_turn(edge: MapEdge, turn: str) -> int:
    """この区間を抜けるときに居るべき車線。

    右折なら道路の中央寄り（自分の方向で一番右）、それ以外は左端。
    道交法 34 条の「あらかじめ寄る」に対応する。
    """
    if turn == "right":
        return lane_count_for_direction(edge) - 1
    return 0


def _offset_points(
    points: Sequence[Point],
    entry_offset: float,
    exit_offset: float,
    lead: float,
) -> list[Point]:
    """中心線を左方向へオフセットする。終端手前で `exit_offset` へ滑らかに移る。"""
    if len(points) < 2:
        return [tuple(p) for p in points]

    tangents = _tangents(points)
    cum = _cumulative(points)
    total = cum[-1]

    out: list[Point] = []
    for i, (px, py) in enumerate(points):
        remaining = total - cum[i]
        if abs(exit_offset - entry_offset) < 1e-9 or lead <= 1e-6:
            offset = entry_offset
        elif remaining >= lead:
            offset = entry_offset
        else:
            # 残り距離が短くなるほど exit_offset に近づく
            t = _smoothstep(1.0 - remaining / lead)
            offset = entry_offset + (exit_offset - entry_offset) * t
        dx, dy = tangents[i]
        # 進行方向の左手
        out.append((px - dy * offset, py + dx * offset))
    return out


def build_lane_route(
    segments: Sequence[RouteSegment], resample_m: float = 2.0
) -> list[Point]:
    """区間列から、車線に沿った走行経路を作る。

    各区間は「入口では左端車線、出口では次の曲がり方に応じた車線」に置き、
    交差点は前後の車線中心線をベジエ曲線でつなぐ。
    """
    usable = [s for s in segments if len(s.points) >= 2]
    if not usable:
        return []

    # --- 1. 交差点ごとの曲がり方を先に決める ---
    turns: list[str] = []
    for i in range(len(usable) - 1):
        out_h = _heading(usable[i].points, at_end=True)
        in_h = _heading(usable[i + 1].points, at_end=False)
        delta = _angle_diff(in_h, out_h)
        if delta > TURN_ANGLE_THRESHOLD:
            turns.append("left")
        elif delta < -TURN_ANGLE_THRESHOLD:
            turns.append("right")
        else:
            turns.append("straight")
    turns.append("straight")  # 最終区間の先は曲がらない（目的地）

    # --- 2. 区間ごとに車線へ寄せ、交差点手前を切り詰める ---
    lane_segments: list[list[Point]] = []
    for i, seg in enumerate(usable):
        edge = seg.edge
        entry_offset = lane_offset_left(edge, 0)  # 入口は常に左端車線
        exit_lane = _target_lane_before_turn(edge, turns[i])
        exit_offset = lane_offset_left(edge, exit_lane)

        offset_pts = _offset_points(seg.points, entry_offset, exit_offset, LANE_CHANGE_LEAD_M)

        # 交差点の手前・直後を切り詰めて、曲線を差し込む余地を作る
        prev_width = usable[i - 1].edge.width if i > 0 else 0.0
        next_width = usable[i + 1].edge.width if i + 1 < len(usable) else 0.0
        trim_start = max(MIN_TRIM_M, prev_width / 2.0 + 1.0) if i > 0 else 0.0
        trim_end = max(MIN_TRIM_M, next_width / 2.0 + 1.0) if i + 1 < len(usable) else 0.0

        lane_segments.append(_trim(offset_pts, trim_start, trim_end))

    # --- 3. 区間どうしをベジエでつなぐ ---
    points: list[Point] = []
    for i, seg_pts in enumerate(lane_segments):
        if not points:
            points.extend(seg_pts)
            continue

        p0 = points[-1]
        p3 = seg_pts[0]
        gap = math.dist(p0, p3)
        if gap < 0.2:
            points.extend(seg_pts[1:])
            continue

        d0x, d0y = _tangents(points[-3:] if len(points) >= 3 else points)[-1]
        d1x, d1y = _tangents(seg_pts[:3] if len(seg_pts) >= 3 else seg_pts)[0]
        handle = gap * BEZIER_HANDLE_RATIO
        p1 = (p0[0] + d0x * handle, p0[1] + d0y * handle)
        p2 = (p3[0] - d1x * handle, p3[1] - d1y * handle)

        samples = max(4, int(gap / max(resample_m, 0.5)) + 2)
        points.extend(_bezier(p0, p1, p2, p3, samples))
        points.extend(seg_pts)

    # --- 4. 重複を落として等間隔に均す ---
    cleaned: list[Point] = []
    for p in points:
        if cleaned and math.dist(cleaned[-1], p) <= 1e-6:
            continue
        cleaned.append(p)
    if len(cleaned) < 2:
        return cleaned
    return _resample(cleaned, resample_m)
