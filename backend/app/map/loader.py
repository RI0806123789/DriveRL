"""OpenStreetMap から道路網・建物を取得し、`MapData` へ正規化する。

処理の流れ:
    1. JSON キャッシュ（`config.MAP_CACHE_DIR/<preset_id>.json`）があれば読む
    2. 無ければ OSMnx で取得 → UTM に投影 → プリセット中心を原点とする ENU 平面へ移す
    3. ノード ID を 0 始まりの連番に振り直し、エッジ・建物を正規化
    4. キャッシュへ書き出す

座標系は docs/protocol.md 1章に従う（単位メートル・x=東・y=北）。
投影は OSMnx が選んだ UTM 帯をそのまま使い、そこからプリセット中心の
UTM 座標を引くだけ。400m 圏ではスケール歪みが 0.1% 未満なので ENU とみなす。
"""

from __future__ import annotations

import json
import math
import logging
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from app import config
from app.contracts import (
    Bounds,
    MapBuilding,
    MapData,
    MapEdge,
    MapNode,
    MapPreset,
    MapSign,
    MapSignal,
)

# ---------------------------------------------------------------------------
# 例外
# ---------------------------------------------------------------------------


logger = logging.getLogger("autoware_sim")


class MapLoadError(RuntimeError):
    """OSM の取得・正規化に失敗したときに投げる。

    docs/protocol.md 2.7 の `MAP_LOAD_FAILED` に対応する。
    """


# ---------------------------------------------------------------------------
# キャッシュ
# ---------------------------------------------------------------------------

#: JSON キャッシュのスキーマ版。正規化ロジックを変えたら必ず上げること。
#: 版が違うキャッシュは読まずに再取得する。
#: 5: 最高速度標識（signs）を追加
#: 6: 標識の生成を隣ノード単位にした（相互エッジ対による重複の解消。code_review M-02）
CACHE_VERSION = 6


# ---------------------------------------------------------------------------
# OSM タグの既定値
# ---------------------------------------------------------------------------

# highway 種別ごとの「片方向あたりの」既定車線数。
# OSM の lanes タグが欠けている（銀座 400m 圏でも大半が欠けている）ときの補完値。
_DEFAULT_LANES: dict[str, int] = {
    "motorway": 3,
    "motorway_link": 1,
    "trunk": 3,
    "trunk_link": 1,
    "primary": 2,
    "primary_link": 1,
    "secondary": 2,
    "secondary_link": 1,
    "tertiary": 2,
    "tertiary_link": 1,
    "unclassified": 1,
    "residential": 1,
    "living_street": 1,
    "service": 1,
}
_FALLBACK_LANES = 1

# highway 種別ごとの既定制限速度 [km/h]。maxspeed タグが欠けているときの補完値。
_DEFAULT_MAXSPEED_KPH: dict[str, float] = {
    "motorway": 80.0,
    "motorway_link": 50.0,
    "trunk": 60.0,
    "trunk_link": 40.0,
    "primary": 50.0,
    "primary_link": 40.0,
    "secondary": 50.0,
    "secondary_link": 40.0,
    "tertiary": 40.0,
    "tertiary_link": 30.0,
    "unclassified": 30.0,
    "residential": 30.0,
    "living_street": 20.0,
    "service": 20.0,
}
_FALLBACK_MAXSPEED_KPH = 40.0

_KPH_TO_MPS = 1.0 / 3.6
_MPH_TO_MPS = 0.44704

# 正規化後に許容する車線数の範囲（異常タグ対策）
_MIN_LANES = 1
_MAX_LANES = 10


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------


def load_map(preset: MapPreset, *, force_refresh: bool = False) -> MapData:
    """プリセット 1 件分のマップを読み込む。

    Args:
        preset: 対象プリセット。
        force_refresh: True なら既存の JSON キャッシュを無視して OSM から取り直す。

    Raises:
        MapLoadError: 取得に失敗した、または結果が使い物にならない規模だった場合。
    """
    cache_path = _cache_path(preset)

    if not force_refresh:
        cached = _read_cache(cache_path, preset)
        if cached is not None:
            return cached

    data = _fetch_and_normalize(preset)
    _write_cache(cache_path, data)
    return data


def cache_path_for(preset: MapPreset) -> Path:
    """プリセットに対応する JSON キャッシュのパス（prefetch などの表示用）。"""
    return _cache_path(preset)


# ---------------------------------------------------------------------------
# OSM 取得と正規化
# ---------------------------------------------------------------------------


def _fetch_and_normalize(preset: MapPreset) -> MapData:
    # osmnx / geopandas は import が重い（数秒）ので、キャッシュヒット時に
    # コストを払わないよう関数内 import にしている。
    try:
        import osmnx as ox
        from pyproj import Transformer
    except Exception as exc:  # pragma: no cover - 環境不備
        raise MapLoadError(f"地図処理ライブラリの読み込みに失敗しました: {exc}") from exc

    # OSMnx の生レスポンスキャッシュ（Overpass への再問い合わせを避ける）
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(config.OSMNX_CACHE_DIR)

    center = (preset.center_lat, preset.center_lon)
    dist = float(preset.radius_m)

    # --- 道路網 ---------------------------------------------------------
    try:
        graph = ox.graph_from_point(
            center,
            dist=dist,
            network_type="drive",
            simplify=True,
            # truncate_by_edge=False にする理由（実測して決めた）:
            #   True にすると境界をまたぐエッジの外側端点まで残るため、銀座では
            #   bounds が ±400m から x[-704, 529] まで広がる。ところが ±400m 圏内の
            #   ノード数は 167 → 165 とほぼ変わらず、増えるのは圏外へ伸びる枝だけ。
            #   建物は ±400m 圏しか取得しないので、その枝の周りは何も無い荒野になる。
            #   占有グリッドも 40% ほど無駄に大きくなるため、False で切り詰める。
            truncate_by_edge=False,
            retain_all=False,        # 最大連結成分のみ（経路探索が成立しない孤島を捨てる）
        )
    except Exception as exc:
        raise MapLoadError(
            f"道路網の取得に失敗しました（preset={preset.id}）: {exc}"
        ) from exc

    try:
        projected = ox.project_graph(graph)
        crs = projected.graph["crs"]
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(projected)
    except Exception as exc:
        raise MapLoadError(
            f"道路網の投影・変換に失敗しました（preset={preset.id}）: {exc}"
        ) from exc

    # --- 原点（プリセット中心の UTM 座標） -------------------------------
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    origin_x, origin_y = transformer.transform(preset.center_lon, preset.center_lat)

    # --- 建物 -----------------------------------------------------------
    # 建物が 1 件も無くてもシミュレーション自体は成立するので、
    # ここでの失敗は致命的エラーにせず空リストで続行する。
    #
    # ★ ただし**黙って続行してはいけない**（code_review B-15 と同じ理由）。
    #   Overpass は混雑すると単発で失敗する。ここで無言だと建物 0 件のマップが
    #   そのままキャッシュに焼き付き、以後は「衝突する物が何も無い世界」になる。
    #   症状は衝突率が下がる＝**成績が良くなる方向**に出るので絶対に気づけない。
    #   実際に金沢を追加したとき、Overpass 側には 925 件あるのに 0 件で
    #   キャッシュされた（道路網は正常に取れていたため、失敗の兆候が何も無かった）。
    buildings_gdf = None
    try:
        buildings_gdf = ox.features_from_point(center, tags={"building": True}, dist=dist)
        if len(buildings_gdf) > 0:
            buildings_gdf = buildings_gdf.to_crs(crs)
        else:
            buildings_gdf = None
            logger.warning(
                "建物が 1 件も取得できませんでした（preset=%s）。"
                "建物との衝突が起きないマップになります。"
                "Overpass の一時的な失敗であれば force_refresh で取り直してください",
                preset.id,
            )
    except Exception:
        buildings_gdf = None
        logger.exception(
            "建物の取得に失敗しました（preset=%s）。建物なしで続行しますが、"
            "建物との衝突が起きないマップになります。"
            "force_refresh で取り直してください",
            preset.id,
        )

    # --- 正規化 ---------------------------------------------------------
    raw_node_xy = _collect_node_xy(nodes_gdf, origin_x, origin_y)
    raw_edges = _collect_edges(edges_gdf, raw_node_xy, origin_x, origin_y)

    if not raw_edges:
        raise MapLoadError(f"走行可能なエッジが 1 本も得られませんでした（preset={preset.id}）")

    nodes, edges, remap = _renumber(raw_node_xy, raw_edges)

    if len(nodes) < 5:
        raise MapLoadError(
            f"ノード数が少なすぎます（preset={preset.id}, nodes={len(nodes)}）"
        )
    if not edges:
        raise MapLoadError(f"走行可能なエッジが 1 本も得られませんでした（preset={preset.id}）")

    buildings = _collect_buildings(buildings_gdf, origin_x, origin_y)
    signals = _build_signals(
        _collect_signal_osmids(nodes_gdf),
        remap,
        edges,
        at_all_intersections=preset.signals_at_all_intersections,
    )
    signs = _build_speed_signs(edges)

    bounds = _compute_bounds(nodes, edges, buildings)

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
    )


def _collect_node_xy(
    nodes_gdf: Any, origin_x: float, origin_y: float
) -> dict[int, tuple[float, float]]:
    """osmid -> ENU 座標 の辞書を作る。"""
    out: dict[int, tuple[float, float]] = {}
    xs = nodes_gdf["x"].to_numpy(dtype=float)
    ys = nodes_gdf["y"].to_numpy(dtype=float)
    for osmid, px, py in zip(nodes_gdf.index, xs, ys):
        out[int(osmid)] = (float(px) - origin_x, float(py) - origin_y)
    return out


def _collect_edges(
    edges_gdf: Any,
    node_xy: dict[int, tuple[float, float]],
    origin_x: float,
    origin_y: float,
) -> dict[tuple[int, int], dict[str, Any]]:
    """(u, v) -> エッジ属性 の辞書を作る。

    MultiDiGraph 由来の重複エッジ（key 違い）は **長さが最短のものだけ** 残す。
    フロントの描画でもシミュレーションでも「同じ交差点間に複数の道」は扱わないため。
    """
    best: dict[tuple[int, int], dict[str, Any]] = {}

    for (u, v, _key), row in edges_gdf.iterrows():
        u = int(u)
        v = int(v)
        if u not in node_xy or v not in node_xy:
            continue

        polyline = _edge_polyline(row, node_xy[u], node_xy[v], origin_x, origin_y)
        if polyline is None or len(polyline) < 2:
            continue

        length = _polyline_length(polyline)
        if length <= 0.0:
            continue

        key = (u, v)
        previous = best.get(key)
        if previous is not None and previous["length"] <= length:
            continue

        highway = _first_tag(row.get("highway"))
        oneway = _parse_oneway(row.get("oneway"))
        lanes_value, lanes_from_tag = _parse_lanes(row.get("lanes"), highway)

        # 車線数の扱い（MapEdge.lanes は「道路全体＝両方向合計」の車線数を持つ）:
        #   - OSM の lanes タグが付いている場合、その値は既に両方向の合計なので
        #     そのまま使う（双方向路で 2 倍すると銀座の外堀通りが 10 車線 32.5m に
        #     なってしまい、実測で明らかに過大だった）。
        #   - タグが欠けていて highway 種別の既定値で埋めた場合、その既定値は
        #     「片方向あたり」の想定なので、双方向路では 2 倍して両方向分を確保する。
        #   いずれの場合も width / lanes == DEFAULT_LANE_WIDTH になるので、
        #   フロント側は width をそのまま道路メッシュの全幅として使える。
        total_lanes = lanes_value if (lanes_from_tag or oneway) else lanes_value * 2
        total_lanes = int(min(max(total_lanes, _MIN_LANES), _MAX_LANES))
        width = max(total_lanes * config.DEFAULT_LANE_WIDTH, config.MIN_ROAD_WIDTH)

        speed_limit = _parse_maxspeed(row.get("maxspeed"), highway)

        best[key] = {
            "u": u,
            "v": v,
            "lanes": total_lanes,
            "width": float(width),
            "oneway": bool(oneway),
            "speed_limit": float(speed_limit),
            "polyline": polyline,
            "length": float(length),
        }

    return best


def _edge_polyline(
    row: Any,
    u_xy: tuple[float, float],
    v_xy: tuple[float, float],
    origin_x: float,
    origin_y: float,
) -> list[tuple[float, float]] | None:
    """エッジのポリラインを ENU 座標で取り出し、始点が u・終点が v になるよう揃える。"""
    geom = row.get("geometry")
    coords: list[tuple[float, float]]

    if geom is not None and getattr(geom, "geom_type", None) == "LineString":
        coords = [(float(px) - origin_x, float(py) - origin_y) for px, py in geom.coords]
    else:
        # geometry が無い（simplify されていない素の 2 点エッジ）場合は端点を直結する
        coords = [u_xy, v_xy]

    coords = _dedupe_consecutive(coords)
    if len(coords) < 2:
        return None

    # 向きの判定は `reversed` 列に頼らず、実際の端点距離で決める方が確実。
    # （reversed は list を持つことがあり、simplify 後は当てにならないケースがある）
    head_to_u = _sq_dist(coords[0], u_xy)
    head_to_v = _sq_dist(coords[0], v_xy)
    if head_to_v < head_to_u:
        coords.reverse()

    # 端点をノード座標にスナップして、経路連結時の隙間を無くす
    coords[0] = u_xy
    coords[-1] = v_xy
    coords = _dedupe_consecutive(coords)
    return coords if len(coords) >= 2 else None


def _renumber(
    node_xy: dict[int, tuple[float, float]],
    raw_edges: dict[tuple[int, int], dict[str, Any]],
) -> tuple[list[MapNode], list[MapEdge], dict[int, int]]:
    """osmid を 0 始まりの連番に振り直す。

    戻り値の 3 番目は osmid -> 新 ID の対応表。信号機ノードを引き当てるのに使う。

    巨大な osmid をワイヤ／キャッシュに載せないための処理。
    エッジから参照されないノードは捨てる。
    """
    used: set[int] = set()
    for u, v in raw_edges:
        used.add(u)
        used.add(v)

    # osmid 昇順で採番して、同じ入力からは常に同じ ID が出るようにする
    ordered = sorted(used)
    remap = {osmid: i for i, osmid in enumerate(ordered)}

    nodes = [
        MapNode(id=remap[osmid], x=node_xy[osmid][0], y=node_xy[osmid][1])
        for osmid in ordered
    ]

    edges: list[MapEdge] = []
    for (u, v), attrs in sorted(raw_edges.items()):
        edges.append(
            MapEdge(
                id=len(edges),
                u=remap[u],
                v=remap[v],
                lanes=attrs["lanes"],
                width=attrs["width"],
                oneway=attrs["oneway"],
                speed_limit=attrs["speed_limit"],
                polyline=attrs["polyline"],
                length=attrs["length"],
            )
        )
    return nodes, edges, remap


# ---------------------------------------------------------------------------
# 交通信号機（日本の設置基準に合わせた抽出）
# ---------------------------------------------------------------------------

#: 既定で、OSM で `highway=traffic_signals` が付いたノードだけでなく
#: **すべての交差点**（接続する道路が SIGNAL_MIN_STREETS 本以上のノード）に信号機を置く。
#: 現実の銀座では 138 交差点のうち 27 か所にしか信号が無いが、
#: 交差点ごとの停止判断を学習させたい場合はすべてに置いたほうが題材になる。
#:
#: ★ これは**既定値**で、プリセットごとに `MapPreset.signals_at_all_intersections`
#:   で上書きできる。広域プリセットでは必ず False にすること（信号の数は面積に比例し、
#:   `frame.signals` が毎フレーム同じ長さの配列を送るため配信量の支配項になる）。
#: 変えたら CACHE_VERSION を上げてキャッシュを作り直すこと。
SIGNALS_AT_ALL_INTERSECTIONS = True

SIGNAL_MIN_STREETS = 3

#: 横断歩道の幅（道路横断方向の長さ）[m]。
#: 「道路標識、区画線及び道路標示に関する命令」の横断歩道は 3m 以上、実務では 4m 程度。
SIGNAL_CROSSWALK_M = 4.0

#: 停止線を横断歩道の手前にどれだけ離すか [m]。実務では 1〜5m。
SIGNAL_STOPLINE_MARGIN_M = 1.0


def _collect_signal_osmids(nodes_gdf: Any) -> set[int]:
    """`highway=traffic_signals` が付いたノードの osmid を集める。"""
    if nodes_gdf is None:
        return set()
    columns = getattr(nodes_gdf, "columns", [])
    if "highway" not in columns:
        return set()

    out: set[int] = set()
    for osmid, value in zip(nodes_gdf.index, nodes_gdf["highway"]):
        for tag in _iter_tag_values(value):
            if str(tag).strip() == "traffic_signals":
                out.add(int(osmid))
                break
    return out


def _point_before_end(
    polyline: Sequence[tuple[float, float]], setback: float
) -> tuple[tuple[float, float], float]:
    """終点から `setback` だけ手前に戻った点と、そこでの進行方向を返す。

    進行方向は「始点 -> 終点」の向き。交差点へ進入してくる車の向きに一致する。
    """
    remaining = float(setback)
    for i in range(len(polyline) - 1, 0, -1):
        x1, y1 = polyline[i]
        x0, y0 = polyline[i - 1]
        dx, dy = x1 - x0, y1 - y0
        seg = math.hypot(dx, dy)
        if seg <= 1e-9:
            continue
        heading = math.atan2(dy, dx)
        if remaining <= seg:
            t = remaining / seg
            return (x1 - dx * t, y1 - dy * t), heading
        remaining -= seg

    # 手前に戻りきれないほど短いエッジ。始点で妥協する
    x0, y0 = polyline[0]
    x1, y1 = polyline[-1]
    return (x0, y0), math.atan2(y1 - y0, x1 - x0)


def _axis_angle(a: float, b: float) -> float:
    """2 つの方位を「軸」として比べた角度差 [0, pi/2]。向きの正負は無視する。"""
    d = abs(math.atan2(math.sin(a - b), math.cos(a - b)))
    return min(d, math.pi - d)


def _build_signals(
    signal_osmids: set[int],
    remap: dict[int, int],
    edges: Sequence[MapEdge],
    at_all_intersections: bool | None = None,
) -> list[MapSignal]:
    """交差点ごとに、進入路 1 本につき 1 基の車両用信号機を作る。

    どの交差点に置くかは `at_all_intersections`（省略時は既定の
    `SIGNALS_AT_ALL_INTERSECTIONS`）で決まる。
    True ならすべての交差点、False なら OSM で `highway=traffic_signals` が
    付いたノードだけ。

    日本の信号機は進入する車両に正対して設置されるので、灯器の姿勢は
    「その進入路をどちら向きに走ってくるか」で決まる。したがって交差点 1 か所に
    つき進入路の数だけ灯器を置く。

    同じ交差点の中では、進入方向の**軸**（向きの正負を無視した方向）が近いものを
    同じグループにまとめる。こうすると直交する流れが必ず別グループになり、
    交差する車線が同時に青になることがない。
    """
    incident: dict[int, list[MapEdge]] = {}
    for e in edges:
        if e.u == e.v:
            continue  # 自己ループ（ロータリー等）は交差点として扱わない
        incident.setdefault(e.u, []).append(e)
        incident.setdefault(e.v, []).append(e)

    # 信号を置く候補ノードを決める
    if at_all_intersections is None:
        at_all_intersections = SIGNALS_AT_ALL_INTERSECTIONS
    if at_all_intersections:
        # 交差点であればすべて置く（OSM のタグは見ない）
        candidates = sorted(incident)
    else:
        # OSM で traffic_signals が付いたノードだけ
        if not signal_osmids:
            return []
        candidates = sorted(
            {remap[o] for o in signal_osmids if o in remap}
        )

    signals: list[MapSignal] = []
    for node_id in candidates:
        around = incident.get(node_id, [])
        if not around:
            continue

        # 双方向路は (u,v) と (v,u) の 2 本に分かれて入っているので、
        # 「隣のノード」で数えないと接続本数を二重に数えてしまう。
        neighbours = {(e.u if e.v == node_id else e.v) for e in around}
        if len(neighbours) < SIGNAL_MIN_STREETS:
            continue

        # 停止線の位置は交差する道路の広さで決まる
        half_width = max(e.width for e in around) / 2.0
        setback = half_width + SIGNAL_CROSSWALK_M + SIGNAL_STOPLINE_MARGIN_M

        # 隣ノードごとに 1 つの進入路を作る（往復の重複を潰す）
        approaches: dict[int, tuple[float, tuple[float, float], float]] = {}
        for e in around:
            if e.v == node_id:
                neighbour = e.u
                point, heading = _point_before_end(e.polyline, setback)
            elif e.u == node_id and not e.oneway:
                neighbour = e.v
                point, heading = _point_before_end(list(reversed(e.polyline)), setback)
            else:
                continue  # 一方通行の出口側。ここから進入してくる車はいない
            approaches.setdefault(neighbour, (heading, point, e.width))

        if not approaches:
            continue

        ordered = [approaches[k] for k in sorted(approaches)]
        ref = ordered[0][0]
        for heading, (px, py), width in ordered:
            signals.append(
                MapSignal(
                    id=len(signals),
                    node_id=node_id,
                    x=px,
                    y=py,
                    heading=heading,
                    group=0 if _axis_angle(heading, ref) < math.pi / 4 else 1,
                    road_width=width,
                )
            )

    return signals


# ---------------------------------------------------------------------------
# 最高速度標識（規制標識「最高速度」）
# ---------------------------------------------------------------------------

#: True なら**規制速度が変わる進入口だけ**に標識を置く（既定）。
#: False にするとすべての有向エッジの始点に置く（交差点を出るたびに再掲される運用）。
#: 銀座では前者 136 基 / 後者 293 基。変えたら CACHE_VERSION を上げること。
SIGNS_ONLY_WHERE_LIMIT_CHANGES = True

#: 規制速度が「変わった」とみなす差 [m/s]。OSM の maxspeed は 5km/h 刻みなので、
#: 1km/h 未満の差は同じ規制として扱う（丸め誤差で標識が乱立するのを防ぐ）。
SIGN_LIMIT_EPSILON_MPS = 1.0 / 3.6


def _point_after_start(
    polyline: Sequence[tuple[float, float]], setback: float
) -> tuple[tuple[float, float], float]:
    """始点から `setback` だけ進んだ点と、そこでの進行方向を返す。

    `_point_before_end` の対で、進行方向は「始点 -> 終点」の向き。
    エッジが短くて進みきれない場合は終点で妥協する。
    """
    remaining = float(setback)
    for i in range(len(polyline) - 1):
        x0, y0 = polyline[i]
        x1, y1 = polyline[i + 1]
        dx, dy = x1 - x0, y1 - y0
        seg = math.hypot(dx, dy)
        if seg <= 1e-9:
            continue
        heading = math.atan2(dy, dx)
        if remaining <= seg:
            t = remaining / seg
            return (x0 + dx * t, y0 + dy * t), heading
        remaining -= seg

    x0, y0 = polyline[0]
    x1, y1 = polyline[-1]
    return (x1, y1), math.atan2(y1 - y0, x1 - x0)


def _build_speed_signs(edges: Sequence[MapEdge]) -> list[MapSign]:
    """規制速度が変わる進入口に、最高速度標識を 1 基ずつ立てる。

    OSM の `traffic_sign` ノードは日本ではほとんど付いていないので、
    道路（way）の `maxspeed` から生成する。`MapEdge.speed_limit` は
    `_parse_maxspeed()` がタグまたは highway 種別の既定値で埋めてある。

    どの進入口に置くかは `SIGNS_ONLY_WHERE_LIMIT_CHANGES` で決まる。
    True のときは「その交差点へ入ってくる**別の**道路の規制速度と違う」場合だけ置く。
    同じ速度が続く直線に同じ数字の標識を並べても情報が増えないためで、
    規制が変わる地点に標識を設置するという実際の運用にも合う。
    行き止まりから出る場合（比較対象が無い）は必ず置く。

    支柱は**進行方向の左側**の路端に立てる（左側通行）。`heading` は
    `MapSignal` と同じ約束で「その標識が規制する側の進行方向」を持ち、
    標示板は運転者に正対するよう `heading + pi` を向く。
    """
    # ★ どちらも**隣ノード単位**で持つ（エッジ単位にしないこと。code_review M-02）。
    #   OSMnx の `graph_from_point()` は対面通行路を (u,v) と (v,u) の
    #   **相互エッジ対**として返すので、エッジ単位で登録すると同じ道路が
    #   2 つの別物として数えられる。`_build_signals()` が `approaches` を
    #   隣ノードで持って往復を潰しているのと同じ理由・同じ作法。
    #   エッジ単位だと次の 2 つが同時に壊れていた:
    #     1. 同じ場所・同じ向きの標識が 2 基立つ（金沢で 27,583 基中 13,250 基）
    #     2. 下の「U ターンを比較対象にしない」がエッジ id 基準なので、
    #        相互エッジ対では片割れが残り、行き止まりで必ず「手前と同じ速度」に
    #        なって標識が立たなくなる（金沢の行き止まり 1,393 か所中 5 か所しか
    #        立っていなかった。docstring の約束と逆）
    # ノードへ入ってくる道路（到着ノード -> {隣ノード: 規制速度}）
    arriving: dict[int, dict[int, float]] = {}
    # ノードから出ていく道路（出発ノード -> {隣ノード: (エッジ, 進行方向の点列)}）
    leaving: dict[int, dict[int, tuple[MapEdge, list[tuple[float, float]]]]] = {}

    for e in edges:
        if e.u == e.v or len(e.polyline) < 2:
            continue  # 自己ループは進入口を持たない
        forward = [(float(px), float(py)) for px, py in e.polyline]
        arriving.setdefault(e.v, {}).setdefault(e.u, e.speed_limit)
        leaving.setdefault(e.u, {}).setdefault(e.v, (e, forward))
        if not e.oneway:
            arriving.setdefault(e.u, {}).setdefault(e.v, e.speed_limit)
            leaving.setdefault(e.v, {}).setdefault(e.u, (e, list(reversed(forward))))

    signs: list[MapSign] = []
    for node_id in sorted(leaving):
        for neighbour in sorted(leaving[node_id]):
            edge, points = leaving[node_id][neighbour]
            if SIGNS_ONLY_WHERE_LIMIT_CHANGES:
                # 同じ道路の逆走（U ターン）は比較対象にしない。
                # これを含めると、行き止まりから出るときに必ず「同じ速度」に
                # なってしまい、経路の出発点に標識が 1 基も立たなくなる。
                # ★ 除外は**隣ノード**で行う。相互エッジ対では同じ道路が
                #   2 つの id を持つので、id で除くと必ず片割れが残る。
                incoming = [
                    limit
                    for other, limit in arriving.get(node_id, {}).items()
                    if other != neighbour
                ]
                if incoming and all(
                    abs(limit - edge.speed_limit) < SIGN_LIMIT_EPSILON_MPS
                    for limit in incoming
                ):
                    continue  # 手前と同じ規制なので標識は要らない

            (px, py), heading = _point_after_start(points, config.SPEED_SIGN_SETBACK_M)
            # 進行方向の左側へ寄せる（左 = heading + pi/2）
            offset = edge.width / 2.0 + config.SPEED_SIGN_SIDE_MARGIN
            px += -math.sin(heading) * offset
            py += math.cos(heading) * offset

            signs.append(
                MapSign(
                    id=len(signs),
                    node_id=node_id,
                    edge_id=edge.id,
                    x=px,
                    y=py,
                    heading=heading,
                    speed_limit=float(edge.speed_limit),
                )
            )

    return signs


def _collect_buildings(
    buildings_gdf: Any, origin_x: float, origin_y: float
) -> list[MapBuilding]:
    """建物フットプリントを正規化する。

    - MultiPolygon は最大面積のポリゴンのみ採用
    - Point / LineString など面を持たないものは捨てる
    - 穴（interiors）は無視し、外周のみを使う
    - `simplify` で頂点を間引き、極小面積のものは捨てる
    """
    if buildings_gdf is None or len(buildings_gdf) == 0:
        return []

    heights = _column_or_none(buildings_gdf, "height")
    levels = _column_or_none(buildings_gdf, "building:levels")

    out: list[MapBuilding] = []
    for i, geom in enumerate(buildings_gdf.geometry):
        polygon = _largest_polygon(geom)
        if polygon is None:
            continue

        try:
            polygon = polygon.simplify(config.BUILDING_SIMPLIFY_TOLERANCE, preserve_topology=True)
        except Exception:
            continue
        if polygon is None or polygon.is_empty:
            continue
        # simplify で GeometryCollection 等になり得るので再度ポリゴンを取り出す
        polygon = _largest_polygon(polygon)
        if polygon is None or polygon.area < config.BUILDING_MIN_AREA:
            continue

        outline = [
            (float(px) - origin_x, float(py) - origin_y) for px, py in polygon.exterior.coords
        ]
        # shapely の exterior は閉じている（末尾＝先頭）。protocol.md 2.2 は
        # 「閉じない」outline を要求するので末尾を落とす。
        if len(outline) >= 2 and _sq_dist(outline[0], outline[-1]) < 1e-12:
            outline.pop()
        outline = _dedupe_consecutive(outline)
        if len(outline) < 3:
            continue

        height = _resolve_height(
            heights[i] if heights is not None else None,
            levels[i] if levels is not None else None,
        )

        out.append(MapBuilding(id=len(out), height=height, outline=outline))

    return out


def _resolve_height(height_tag: Any, levels_tag: Any) -> float:
    """建物高さを決める（memo 5章「情報がなければ一律の固定高さ」）。"""
    height = _parse_float_tag(height_tag)
    if height is not None and 1.0 <= height <= 700.0:
        return float(height)

    levels = _parse_float_tag(levels_tag)
    if levels is not None and 1.0 <= levels <= 200.0:
        return float(levels) * config.BUILDING_LEVEL_HEIGHT

    return float(config.DEFAULT_BUILDING_HEIGHT)


def _compute_bounds(
    nodes: Sequence[MapNode],
    edges: Sequence[MapEdge],
    buildings: Sequence[MapBuilding],
) -> Bounds:
    """全ノード・全エッジ頂点・全建物頂点から外接矩形を求める。"""
    xs: list[float] = [n.x for n in nodes]
    ys: list[float] = [n.y for n in nodes]
    for e in edges:
        for px, py in e.polyline:
            xs.append(px)
            ys.append(py)
    for b in buildings:
        for px, py in b.outline:
            xs.append(px)
            ys.append(py)

    if not xs:
        raise MapLoadError("bounds を計算できる座標がありません")

    return Bounds(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys))


# ---------------------------------------------------------------------------
# タグのパース（OSM のタグは NaN / 文字列 / リストが混在する）
# ---------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _iter_tag_values(value: Any) -> Iterable[Any]:
    """タグ値を平坦化して 1 個ずつ返す（`['2', '3']` のようなリスト対策）。"""
    if _is_missing(value):
        return
    if isinstance(value, (list, tuple, set, np.ndarray)):
        for item in value:
            if not _is_missing(item):
                yield item
        return
    yield value


def _first_tag(value: Any) -> str:
    """highway のようなカテゴリタグから代表値 1 個を取り出す。"""
    for item in _iter_tag_values(value):
        return str(item).strip().lower()
    return ""


def _parse_float_tag(value: Any) -> float | None:
    """`'12'` / `'12 m'` / `'12;15'` / `['12', '9']` などから最初の数値を取り出す。"""
    for item in _iter_tag_values(value):
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            if math.isnan(float(item)):
                continue
            return float(item)
        text = str(item).strip().replace(",", ".")
        number = ""
        seen_dot = False
        for ch in text:
            if ch.isdigit():
                number += ch
            elif ch == "." and number and not seen_dot:
                number += ch
                seen_dot = True
            elif number:
                break
            elif ch in "+-" and not number:
                continue
            else:
                continue
        if number:
            try:
                return float(number)
            except ValueError:
                continue
    return None


def _parse_lanes(value: Any, highway: str) -> tuple[int, bool]:
    """車線数を頑健にパースする。

    Returns:
        (車線数, タグ由来か)。タグ由来なら OSM の慣習どおり「両方向の合計」、
        そうでなければ highway 種別の既定値（＝片方向あたり）を意味する。
        `['2', '3']` のようなリストは最大値を採る。
    """
    candidates: list[int] = []
    for item in _iter_tag_values(value):
        parsed = _parse_float_tag(item)
        if parsed is not None and parsed >= 1.0:
            candidates.append(int(parsed))
    if candidates:
        return int(min(max(max(candidates), _MIN_LANES), _MAX_LANES)), True
    return _DEFAULT_LANES.get(highway, _FALLBACK_LANES), False


def _parse_maxspeed(value: Any, highway: str) -> float:
    """制限速度を m/s で返す。`'50'` / `'50 km/h'` / `'30 mph'` / リストに対応。"""
    for item in _iter_tag_values(value):
        text = str(item).strip().lower()
        number = _parse_float_tag(text)
        if number is None or number <= 0.0:
            continue
        if "mph" in text:
            return float(number) * _MPH_TO_MPS
        if "knot" in text:
            return float(number) * 0.514444
        return float(number) * _KPH_TO_MPS

    kph = _DEFAULT_MAXSPEED_KPH.get(highway, _FALLBACK_MAXSPEED_KPH)
    return float(kph) * _KPH_TO_MPS


def _parse_oneway(value: Any) -> bool:
    """oneway を bool に正規化する。`True` / `'yes'` / `'-1'` / リストに対応。"""
    for item in _iter_tag_values(value):
        if isinstance(item, (bool, np.bool_)):
            return bool(item)
        text = str(item).strip().lower()
        if text in ("yes", "true", "1", "-1", "reversible"):
            return True
        if text in ("no", "false", "0"):
            return False
    return False


# ---------------------------------------------------------------------------
# ジオメトリのユーティリティ
# ---------------------------------------------------------------------------


def _sq_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _dedupe_consecutive(
    points: Sequence[tuple[float, float]], eps: float = 1e-9
) -> list[tuple[float, float]]:
    """連続する重複点を取り除く。"""
    out: list[tuple[float, float]] = []
    for p in points:
        if out and _sq_dist(out[-1], p) <= eps:
            continue
        out.append((float(p[0]), float(p[1])))
    return out


def _polyline_length(points: Sequence[tuple[float, float]]) -> float:
    """ポリラインの実長 [m]。"""
    total = 0.0
    for i in range(1, len(points)):
        total += math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
    return total


def _largest_polygon(geom: Any) -> Any | None:
    """Polygon をそのまま、MultiPolygon なら最大面積の 1 枚を返す。面が無ければ None。"""
    if geom is None or getattr(geom, "is_empty", True):
        return None
    geom_type = getattr(geom, "geom_type", None)
    if geom_type == "Polygon":
        return geom
    if geom_type in ("MultiPolygon", "GeometryCollection"):
        best = None
        best_area = 0.0
        for part in geom.geoms:
            if getattr(part, "geom_type", None) != "Polygon" or part.is_empty:
                continue
            if part.area > best_area:
                best = part
                best_area = part.area
        return best
    return None


def _column_or_none(gdf: Any, name: str) -> list[Any] | None:
    """GeoDataFrame の列を Python のリストとして取り出す。無い列は None。"""
    if name not in gdf.columns:
        return None
    return list(gdf[name])


# ---------------------------------------------------------------------------
# JSON キャッシュの入出力
# ---------------------------------------------------------------------------


def _cache_path(preset: MapPreset) -> Path:
    return config.MAP_CACHE_DIR / f"{preset.id}.json"


def _read_cache(path: Path, preset: MapPreset) -> MapData | None:
    """キャッシュを読む。壊れている／版が古い／プリセット定義が変わっていれば None。"""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("version") != CACHE_VERSION:
        return None
    if payload.get("presetId") != preset.id:
        return None

    # プリセット定義（中心・半径）が変わっていたらキャッシュは無効
    if not _close(payload.get("centerLat"), preset.center_lat, 1e-6):
        return None
    if not _close(payload.get("centerLon"), preset.center_lon, 1e-6):
        return None
    if not _close(payload.get("radiusM"), preset.radius_m, 1e-3):
        return None

    try:
        return _from_cache_dict(payload, preset)
    except (KeyError, TypeError, ValueError):
        return None


def _write_cache(path: Path, data: MapData) -> None:
    """キャッシュを書き出す。書き込みに失敗しても処理は続行する（読み込みは成功済み）。"""
    tmp = path.with_suffix(".json.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as fp:
            json.dump(_to_cache_dict(data), fp, ensure_ascii=False, separators=(",", ":"))
        tmp.replace(path)
    except OSError:
        return


def _to_cache_dict(data: MapData) -> dict[str, Any]:
    """MapData を JSON へ落とす。to_wire() と違い length など内部値も保存する。"""
    return {
        "version": CACHE_VERSION,
        "generatedAt": time.time(),
        "presetId": data.preset_id,
        "name": data.name,
        "centerLat": data.center_lat,
        "centerLon": data.center_lon,
        "radiusM": data.radius_m,
        "bounds": {
            "minX": data.bounds.min_x,
            "minY": data.bounds.min_y,
            "maxX": data.bounds.max_x,
            "maxY": data.bounds.max_y,
        },
        # 冗長なキー名を避けて配列で持つ（キャッシュサイズ削減）
        "nodes": [[n.id, round(n.x, 3), round(n.y, 3)] for n in data.nodes],
        "edges": [
            {
                "id": e.id,
                "u": e.u,
                "v": e.v,
                "lanes": e.lanes,
                "width": round(e.width, 3),
                "oneway": e.oneway,
                "speedLimit": round(e.speed_limit, 3),
                "length": round(e.length, 3),
                "polyline": [[round(px, 3), round(py, 3)] for px, py in e.polyline],
            }
            for e in data.edges
        ],
        "buildings": [
            {
                "id": b.id,
                "height": round(b.height, 2),
                "outline": [[round(px, 3), round(py, 3)] for px, py in b.outline],
            }
            for b in data.buildings
        ],
        # 信号機も配列で持つ: [id, nodeId, x, y, heading, group, roadWidth]
        "signals": [
            [
                sg.id,
                sg.node_id,
                round(sg.x, 3),
                round(sg.y, 3),
                round(sg.heading, 4),
                sg.group,
                round(sg.road_width, 2),
            ]
            for sg in data.signals
        ],
        # 標識も配列で持つ: [id, nodeId, edgeId, x, y, heading, speedLimit]
        "signs": [
            [
                sn.id,
                sn.node_id,
                sn.edge_id,
                round(sn.x, 3),
                round(sn.y, 3),
                round(sn.heading, 4),
                round(sn.speed_limit, 3),
            ]
            for sn in data.signs
        ],
    }


def _from_cache_dict(payload: dict[str, Any], preset: MapPreset) -> MapData:
    bounds_raw = payload["bounds"]
    bounds = Bounds(
        min_x=float(bounds_raw["minX"]),
        min_y=float(bounds_raw["minY"]),
        max_x=float(bounds_raw["maxX"]),
        max_y=float(bounds_raw["maxY"]),
    )

    nodes = [MapNode(id=int(n[0]), x=float(n[1]), y=float(n[2])) for n in payload["nodes"]]

    edges: list[MapEdge] = []
    for e in payload["edges"]:
        polyline = [(float(px), float(py)) for px, py in e["polyline"]]
        edges.append(
            MapEdge(
                id=int(e["id"]),
                u=int(e["u"]),
                v=int(e["v"]),
                lanes=int(e["lanes"]),
                width=float(e["width"]),
                oneway=bool(e["oneway"]),
                speed_limit=float(e["speedLimit"]),
                polyline=polyline,
                length=float(e["length"]),
            )
        )

    buildings = [
        MapBuilding(
            id=int(b["id"]),
            height=float(b["height"]),
            outline=[(float(px), float(py)) for px, py in b["outline"]],
        )
        for b in payload["buildings"]
    ]

    # 信号機は後から追加した項目なので、無ければ空で扱う
    signals = [
        MapSignal(
            id=int(sg[0]),
            node_id=int(sg[1]),
            x=float(sg[2]),
            y=float(sg[3]),
            heading=float(sg[4]),
            group=int(sg[5]),
            road_width=float(sg[6]),
        )
        for sg in payload.get("signals", [])
    ]

    # 標識も後から追加した項目なので、無ければ空で扱う
    signs = [
        MapSign(
            id=int(sn[0]),
            node_id=int(sn[1]),
            edge_id=int(sn[2]),
            x=float(sn[3]),
            y=float(sn[4]),
            heading=float(sn[5]),
            speed_limit=float(sn[6]),
        )
        for sn in payload.get("signs", [])
    ]

    if len(nodes) < 5 or not edges:
        raise ValueError("キャッシュの内容が最小要件を満たしていません")

    return MapData(
        preset_id=str(payload["presetId"]),
        name=str(payload.get("name", preset.name)),
        center_lat=float(payload["centerLat"]),
        center_lon=float(payload["centerLon"]),
        radius_m=float(payload["radiusM"]),
        bounds=bounds,
        nodes=nodes,
        edges=edges,
        buildings=buildings,
        signals=signals,
        signs=signs,
    )


def _close(value: Any, expected: float, tol: float) -> bool:
    try:
        return abs(float(value) - float(expected)) <= tol
    except (TypeError, ValueError):
        return False
