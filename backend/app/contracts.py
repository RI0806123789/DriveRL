"""バックエンド内部の共有契約（データ構造とインターフェース）。

このモジュールは **依存を持たない中立地帯** であり、`map` / `sim` / `rl` / `runtime`
の各パッケージはここで定義された型を通じてのみ相互にやり取りする。
実装を追加するときにこのファイルのシグネチャを変えると他パッケージが壊れるので、
変更が必要な場合は必ず全利用箇所を同時に直すこと。

座標系は docs/protocol.md 1章に従う（ENU 平面・メートル・x=東 / y=北）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from app import config

# ---------------------------------------------------------------------------
# マッププリセット
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MapPreset:
    """事前定義された読み込み対象エリア（memo 5章「複数の固定エリアを事前プリセット」）。"""

    id: str
    name: str
    description: str
    center_lat: float
    center_lon: float
    radius_m: float
    #: すべての交差点に信号を置くか。False なら OSM で `highway=traffic_signals` が
    #: 付いたノードだけに置く。
    #: ★ 広域プリセットでは必ず False にすること。信号の数は面積に比例して増え、
    #:   `frame.signals` は**毎フレーム**同じ長さの配列を送るため、ここが配信量の
    #:   支配項になる（半径 6km では全交差点だと 55,714 基 = 123KB/frame = 2.4MB/s）。
    #: ワイヤには載せない（フロントは信号の一覧を map.signals で受け取るので不要）。
    signals_at_all_intersections: bool = True

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "centerLat": self.center_lat,
            "centerLon": self.center_lon,
            "radiusM": self.radius_m,
        }


# ---------------------------------------------------------------------------
# マップの静的データ（シリアライズ可能・JSON キャッシュ対象）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def to_wire(self) -> dict[str, float]:
        return {"minX": self.min_x, "minY": self.min_y, "maxX": self.max_x, "maxY": self.max_y}


@dataclass
class MapNode:
    """道路ネットワークの交差点／端点。"""

    id: int
    x: float
    y: float


@dataclass
class MapEdge:
    """道路セグメント。polyline は始点が u、終点が v に対応する。"""

    id: int
    u: int
    v: int
    lanes: int
    width: float          # メートル。道路メッシュの全幅（片側ではない）
    oneway: bool
    speed_limit: float    # m/s
    polyline: list[tuple[float, float]]
    length: float         # ポリラインに沿った実長 [m]


@dataclass
class MapBuilding:
    """建物のフットプリント。outline は閉じない（末尾と先頭は重複させない）。"""

    id: int
    height: float
    outline: list[tuple[float, float]]


@dataclass
class MapSignal:
    """交通信号機（車両用）。OSM の `highway=traffic_signals` ノードから作る。

    1 つの交差点に対し、**進入路ごとに 1 基**を作る。日本の信号機は進入車両に
    正対して設置されるため、灯器の向きは進入方向で決まるからである。

    日本の道路交通法・信号機の設置基準に合わせるための情報:
      - `heading` は進入車両の進行方向。灯器はこの逆（`heading + pi`）を向く
      - 灯器は横型 3 灯で、**運転者から見て左から 青・黄・赤**（赤が右端）
      - 左側通行なので支柱は進行方向左側に立ち、アームで車道上へ張り出す
      - `group` が同じ信号は同時に青になる（交差する流れは必ず赤）
    """

    id: int
    node_id: int          # 交差点ノード（MapNode.id）への参照
    x: float              # 停止線の位置（交差点手前）の ENU 座標
    y: float
    heading: float        # 進入車両の進行方向 [rad]
    group: int            # 0 か 1。同じ値どうしが同時に青
    road_width: float     # 停止線・横断歩道を描くための進入路の幅 [m]


@dataclass
class MapSign:
    """最高速度標識（規制標識「最高速度」）。

    OSM の `traffic_sign` タグは日本ではほとんど付いていないので、
    **道路（way）の `maxspeed` から生成する**。`MapEdge.speed_limit` が
    手前の道路と変わる進入口にだけ置く（規制が変わる地点に設置するという
    道路標識の運用に合わせる）。

    日本の設置基準に合わせるための情報:
      - `heading` は**その標識が規制する進行方向**。`MapSignal.heading` と同じ約束で、
        標示板はこの逆（`heading + pi`）を向いて運転者に正対する
      - 左側通行なので支柱は進行方向左側の路端に立つ
      - 標示板は白地の円に赤縁・黒数字。直径 60cm、下端は路面から 1.8m 以上
    """

    id: int
    node_id: int          # 標識が立つ交差点ノード（MapNode.id）
    edge_id: int          # この標識が規制する道路（MapEdge.id）
    x: float              # 支柱の位置の ENU 座標
    y: float
    heading: float        # 規制する側の進行方向 [rad]
    speed_limit: float    # 規制速度 [m/s]


@dataclass
class MapData:
    """1 プリセット分の静的マップデータ。JSON にそのまま落とせる範囲だけを持つ。"""

    preset_id: str
    name: str
    center_lat: float
    center_lon: float
    radius_m: float
    bounds: Bounds
    nodes: list[MapNode]
    edges: list[MapEdge]
    buildings: list[MapBuilding]
    # 交通信号機。古いキャッシュには入っていないので既定値を持たせる
    signals: list[MapSignal] = field(default_factory=list)
    # 最高速度標識。信号と同じく、古いキャッシュには入っていない
    signs: list[MapSign] = field(default_factory=list)

    def to_wire(self) -> dict[str, Any]:
        """docs/protocol.md 2.2 の map メッセージ本体を返す（type は呼び出し側で付与）。"""
        return {
            "presetId": self.preset_id,
            "name": self.name,
            "bounds": self.bounds.to_wire(),
            "nodes": [{"id": n.id, "x": round(n.x, 3), "y": round(n.y, 3)} for n in self.nodes],
            "edges": [
                {
                    "id": e.id,
                    "u": e.u,
                    "v": e.v,
                    "lanes": e.lanes,
                    "width": round(e.width, 2),
                    "oneway": e.oneway,
                    "speedLimit": round(e.speed_limit, 2),
                    "polyline": [[round(px, 3), round(py, 3)] for px, py in e.polyline],
                }
                for e in self.edges
            ],
            "buildings": [
                {
                    "id": b.id,
                    "height": round(b.height, 2),
                    "outline": [[round(px, 3), round(py, 3)] for px, py in b.outline],
                }
                for b in self.buildings
            ],
            "signals": [
                {
                    "id": s.id,
                    "nodeId": s.node_id,
                    "x": round(s.x, 3),
                    "y": round(s.y, 3),
                    "heading": round(s.heading, 4),
                    "group": s.group,
                    "roadWidth": round(s.road_width, 2),
                }
                for s in self.signals
            ],
            "signs": [
                {
                    "id": sg.id,
                    "nodeId": sg.node_id,
                    "edgeId": sg.edge_id,
                    "x": round(sg.x, 3),
                    "y": round(sg.y, 3),
                    "heading": round(sg.heading, 4),
                    "speedLimit": round(sg.speed_limit, 3),
                }
                for sg in self.signs
            ],
        }


# ---------------------------------------------------------------------------
# ラスタ（高速な建物／道路判定とレイキャストのための占有グリッド）
# ---------------------------------------------------------------------------


@dataclass
class OccupancyGrid:
    """ENU 平面を等間隔セルに区切った占有マップ。

    配列は [row, col] 順で、col が x（東）方向、row が y（北）方向に対応する。
    origin_x / origin_y はセル (0, 0) の **中心** の ENU 座標。
    """

    origin_x: float
    origin_y: float
    cell_size: float
    width: int            # x 方向セル数
    height: int           # y 方向セル数
    building: np.ndarray  # shape (height, width), dtype=bool。建物内部が True
    # ★ 走行可能領域（road）レイヤは持たない（code_review B-16）。
    #   レイキャストも衝突判定も building だけを見る設計に落ち着いており、
    #   読む箇所が 1 件も無いまま、マップ読み込みのたびに全エッジを
    #   buffer してラスタライズしていた。

    def world_to_cell(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ENU 座標をセル添字 (col, row) に変換する。範囲外の値も返るので呼び出し側でクリップする。"""
        col = np.rint((np.asarray(x) - self.origin_x) / self.cell_size).astype(np.int32)
        row = np.rint((np.asarray(y) - self.origin_y) / self.cell_size).astype(np.int32)
        return col, row

    def sample(self, layer: np.ndarray, x: np.ndarray, y: np.ndarray, outside: bool) -> np.ndarray:
        """指定レイヤを ENU 座標で参照する。グリッド外は outside の値を返す。"""
        col, row = self.world_to_cell(x, y)
        inside = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        out = np.full(np.shape(col), outside, dtype=bool)
        if inside.any():
            out[inside] = layer[row[inside], col[inside]]
        return out

    def sample_building(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """建物内部なら True。グリッド外は False（何もない空間扱い）。"""
        return self.sample(self.building, x, y, outside=False)


# ---------------------------------------------------------------------------
# マップの実行時インデックス（キャッシュから復元後に構築する派生データ）
# ---------------------------------------------------------------------------


class MapIndex(Protocol):
    """経路探索・最近傍探索・レイキャストを提供する実行時インデックス。

    実装は app.map.index.build_map_index(map_data) が返す。
    """

    data: MapData
    occupancy: OccupancyGrid

    def nearest_node(self, x: float, y: float) -> int:
        """指定座標に最も近い道路ノード ID を返す。"""
        ...

    def nearest_road_point(self, x: float, y: float) -> tuple[float, float, int, float]:
        """指定座標を最寄りの道路中心線上にスナップする。

        Returns:
            (snapped_x, snapped_y, edge_id, heading)
            heading はその地点での道路進行方向 [rad]。
        """
        ...

    def shortest_path(self, src_node: int, dst_node: int) -> list[int] | None:
        """ノード ID 列で最短経路を返す。到達不能なら None。"""
        ...

    def route_polyline(
        self, node_path: Sequence[int], resample_m: float = 2.0
    ) -> list[tuple[float, float]]:
        """ノード列をエッジのポリラインに展開し、等間隔にリサンプルした点列を返す。"""
        ...

    def lane_route_polyline(
        self, node_path: Sequence[int], resample_m: float = 2.0
    ) -> list[tuple[float, float]]:
        """左側通行の車線に沿った走行経路を返す。

        道路中心線ではなく進行方向左側の車線を通り、右左折の手前では
        道交法 34 条に従って寄せる（左折・直進は左端、右折は中央寄り）。
        交差点は前後の車線中心線をベジエ曲線でつなぐ。
        """
        ...

    def signals_on_route(
        self, points: Sequence[tuple[float, float]]
    ) -> list[tuple[float, int]]:
        """経路が通過する信号を (経路始点からの弧長 [m], MapData.signals の添字) で返す。

        弧長の昇順。進入方向が経路の進行方向と大きく違う信号は含めない。
        """
        ...

    def speed_limits_on_route(
        self, points: Sequence[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """経路に適用される規制速度を (弧長 [m], 規制速度 [m/s]) の区切りで返す。

        弧長の昇順で、先頭は必ず 0.0（出発地点に適用される速度）。
        弧長 a における規制速度は「a 以下で最後の区切り」の速度。
        """
        ...

    def random_node_pair(
        self, rng: np.random.Generator, min_distance_m: float = 150.0
    ) -> tuple[int, int]:
        """経路が存在し、かつ十分離れた出発ノードと目的ノードの組を返す。"""
        ...

    def raycast(
        self,
        origin_x: np.ndarray,
        origin_y: np.ndarray,
        angles: np.ndarray,
        max_distance: float,
        step: float = 1.0,
    ) -> np.ndarray:
        """占有グリッド上で建物までの距離を測る。

        Args:
            origin_x, origin_y: shape (N,) の始点。
            angles: shape (N, R) の各レイの絶対方位角 [rad]（ENU）。
            max_distance: 打ち切り距離 [m]。
        Returns:
            shape (N, R) の距離配列。何にも当たらなければ max_distance。
        """
        ...

    def collides_with_building(self, corners: np.ndarray) -> bool:
        """車両の外接矩形（shape (4, 2) の頂点列）が建物と重なるかを厳密に判定する。"""
        ...

    def collides_with_buildings(
        self, corners: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """複数台ぶんをまとめて判定する。shape (N,) bool。

        `corners` は (N, 4, 2)、`mask` は (N,) bool で True のスロットだけ見る。
        1 台ずつ呼ぶと高倍速で予算を使い切るので、ホットパスはこちらを使うこと。
        """
        ...


# ---------------------------------------------------------------------------
# シミュレーションのパラメータ（フロントから set_params で部分更新される）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ParamSpec:
    """1 パラメータのワイヤ名・型・値域。

    値域を持たせているのは飾りではない。`set_params` は WebSocket から誰でも
    投げられるので、`{"gamma": "nan"}` のような値が素通りすると GAE から損失、
    `optimizer.step()` を経て**重み全体が NaN になる**。しかも `env.step()` の
    `np.nan_to_num` が行動を 0 に潰すため画面は正常に見えたまま学習だけが死に、
    20 更新ごとの自動保存でその重みがディスクに残る。ここで止めるしかない。
    """

    wire: str
    kind: type              # int / float / bool
    minimum: float | None = None
    maximum: float | None = None


# snake_case（Python 側） <-> camelCase（ワイヤ形式）の対応表と値域。
# 範囲は docs/protocol.md 2.5 と UI のスライダー（frontend/src/panel/）に合わせてある。
_PARAM_SPECS: dict[str, _ParamSpec] = {
    # vehicle_count の上限は config.MAX_VEHICLES。ここで config を import すると
    # contracts が「依存を持たない中立地帯」でなくなるので、上限は apply_wire で解決する。
    # ★ 下限は 0（code_review R-12）。3D 画面から全車をデスポーンすると
    #   `world.active_count` が 0 になり、engine がその値を `params` で配信する。
    #   下限を 1 にしていたとき、**サーバーが自分の値域検証を通らない値を送る**
    #   状態になっていた。全部消せることは実際に仕様なので、下限を実態へ合わせる
    #   （docs/protocol.md 2.5 の `(0..maxVehicles)` と対）
    "vehicle_count": _ParamSpec("vehicleCount", int, 0, None),
    "sim_speed": _ParamSpec("simSpeed", float, 0.25, 8.0),
    "learning_rate": _ParamSpec("learningRate", float, 1e-6, 1e-2),
    "gamma": _ParamSpec("gamma", float, 0.5, 0.9999),
    "clip_range": _ParamSpec("clipRange", float, 0.01, 0.9),
    "entropy_coef": _ParamSpec("entropyCoef", float, 0.0, 0.5),
    # 上限は確保するバッファの大きさに直結する。
    # 2048 ステップ × MAX_VEHICLES(8) スロット × OBS_DIM(57) 次元 × 4B ≒ 3.7MB
    # （64 台 / 56 次元だった頃の見積もりは約 29MB で、8 倍過大だった。
    #   code_review Q-08。いまの実サイズなら上限を上げる余地がある）
    "rollout_length": _ParamSpec("rolloutLength", int, 16, 2048),
    "max_speed": _ParamSpec("maxSpeed", float, 1.0, 40.0),
    "reward_goal": _ParamSpec("rewardGoal", float, 0.0, 1000.0),
    "reward_collision": _ParamSpec("rewardCollision", float, -1000.0, 0.0),
    "reward_progress": _ParamSpec("rewardProgress", float, 0.0, 50.0),
    "reward_offroad": _ParamSpec("rewardOffroad", float, -100.0, 0.0),
    "reward_time": _ParamSpec("rewardTime", float, -10.0, 0.0),
    "reward_signal": _ParamSpec("rewardSignal", float, -1000.0, 0.0),
    "reward_overspeed": _ParamSpec("rewardOverspeed", float, -1000.0, 0.0),
    "obey_signals": _ParamSpec("obeySignals", bool),
    "obey_speed_signs": _ParamSpec("obeySpeedSigns", bool),
}

# 真偽値として受け付ける文字列。`bool("false")` は True になってしまうので、
# 型変換に任せず明示的に対応表を引く
_TRUE_WORDS = {"true", "1", "yes", "on"}
_FALSE_WORDS = {"false", "0", "no", "off"}


@dataclass
class ParamPatchResult:
    """`SimParams.apply_wire()` の結果。

    - `changed`: 実際に値が変わったフィールド（snake_case）
    - `rejected`: 型が違う・非有限などで**反映しなかった**ワイヤキー
    - `clamped`: 値域外だったので端に丸めて反映したワイヤキー

    `rejected` と `clamped` が空でなければ、呼び出し側は `docs/protocol.md` 2.7 の
    `INVALID_MESSAGE` を返す。黙って捨てると、利用者は反映されない理由が分からない。
    """

    changed: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    clamped: list[str] = field(default_factory=list)

    @property
    def has_problem(self) -> bool:
        return bool(self.rejected or self.clamped)


def coerce_bool(value: Any) -> bool | None:
    """真偽値へ変換する。解釈できなければ None。

    `bool("false") == True` なので、文字列は必ず語で判定する。
    ★ WebSocket から来た真偽値は**必ずこれを通すこと**（code_review R-09）。
      `bool(message.get(...))` だと `{"paused": "false"}` が一時停止になる。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return None if not math.isfinite(float(value)) else bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    return None


@dataclass
class SimParams:
    """docs/protocol.md 2.5 の params に対応。

    フィールド名は snake_case で持ち、送信時に camelCase へ変換する。
    """

    vehicle_count: int = 4
    sim_speed: float = 1.0
    learning_rate: float = 3e-4
    # 20Hz 制御では 1/(1-gamma) が「何ステップ先まで見えるか」になり、0.99 では
    # 100 ステップ = 5 秒しかない。150m 先の目的地（約 375 ステップ）の報酬は
    # 0.018 倍に潰れるため、0.997 / 0.999 のほうが良いと考えて実測した。
    # 結果は 200 更新の時点で 0.99=到達52.0% / 0.997=50.3% / 0.999=41.3% となり、
    # **0.99 と 0.997 は誤差の範囲、0.999 は明確に悪化**だった。
    # 経路進捗（1m あたり +1.0）が毎ステップ入る密な報酬なので、遠い目的地報酬が
    # 見えなくても学習が進むためと考えられる。確認できなかったので既定は変えない。
    gamma: float = 0.99
    clip_range: float = 0.2
    # 行動は 2 次元・範囲 [-1, 1] しかないので、探索ボーナスは小さくてよい。
    # 0.01 では報酬の信号が弱いときにエントロピー項が支配し、log_std を
    # 上限まで押し上げて方策がランダムに潰れた（実測）。
    entropy_coef: float = 0.001
    rollout_length: int = 256
    max_speed: float = 13.9
    reward_goal: float = 100.0
    reward_collision: float = -100.0
    reward_progress: float = 1.0
    reward_offroad: float = -1.0
    reward_time: float = -0.05
    # 赤信号で停止線を越えたときの罰（道交法施行令 2 条）
    reward_signal: float = -60.0
    # 最高速度標識を超えたときの罰（道交法 22 条）。
    # 超え「始めた」ステップに 1 回だけ入る（毎ステップではない）。
    # obey_speed_signs=True なら環境側が速度を抑えるので滅多に発火しないが、
    # 下り勾配相当の慣性やカーブ出口で瞬間的に超えることがある。
    # 信号無視（-60）より軽くしてあるのは、速度超過が「事故に直結する度合い」で
    # 劣るためで、赤信号無視と同格に扱うと停止挙動の学習を邪魔する。
    reward_overspeed: float = -5.0
    # 信号に従わせるか。True なら赤・黄（止まれる場合）で停止線の手前に止める制約が働く。
    # False にすると罰だけになり、守るかどうかは学習しだいになる
    obey_signals: bool = True
    # 最高速度標識に従わせるか。True なら規制速度を超えないよう加速指令が抑えられる。
    # False にすると罰だけになり、守るかどうかは学習しだいになる（信号と同じ約束）
    obey_speed_signs: bool = True

    def to_wire(self) -> dict[str, Any]:
        return {spec.wire: getattr(self, snake) for snake, spec in _PARAM_SPECS.items()}

    def apply_wire(self, patch: dict[str, Any], *, max_vehicles: int = 64) -> ParamPatchResult:
        """camelCase の部分更新を検証してから適用する。

        値域外は端に丸め、非有限値や解釈できない型は**反映しない**。
        知らないキーは黙って無視する（プロトコルの前方互換のため）。

        Args:
            max_vehicles: `vehicleCount` の上限（`config.MAX_VEHICLES`）。
                contracts は config に依存しない約束なので呼び出し側から渡す。
        """
        result = ParamPatchResult()
        reverse = {spec.wire: snake for snake, spec in _PARAM_SPECS.items()}

        for wire_key, value in patch.items():
            snake = reverse.get(wire_key)
            if snake is None:
                continue
            spec = _PARAM_SPECS[snake]
            current = getattr(self, snake)

            if spec.kind is bool:
                coerced: Any = coerce_bool(value)
                if coerced is None:
                    result.rejected.append(wire_key)
                    continue
            else:
                # bool は int の派生なので、数値として扱う前にここで潰しておく
                if isinstance(value, bool):
                    value = int(value)
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    result.rejected.append(wire_key)
                    continue
                if not math.isfinite(number):
                    # NaN / Inf。ここを通すと重みが NaN になり、自動保存で永続化される
                    result.rejected.append(wire_key)
                    continue

                low = spec.minimum
                high = spec.maximum
                if snake == "vehicle_count":
                    high = float(max(1, int(max_vehicles)))
                clipped = number
                if low is not None:
                    clipped = max(clipped, float(low))
                if high is not None:
                    clipped = min(clipped, float(high))
                if clipped != number:
                    result.clamped.append(wire_key)
                coerced = int(round(clipped)) if spec.kind is int else float(clipped)

            if coerced != current:
                setattr(self, snake, coerced)
                result.changed.append(snake)

        return result


# ---------------------------------------------------------------------------
# フレームスナップショット（sim -> runtime -> WebSocket）
# ---------------------------------------------------------------------------


@dataclass
class VehicleSnapshot:
    id: int
    active: bool
    x: float
    y: float
    heading: float
    speed: float
    steer: float
    collided: bool
    reached_goal: bool
    goal: tuple[float, float]
    # 経路の達成度 0.0〜1.0（経路始点からの進行距離 / 経路全長）
    progress: float = 0.0
    # このエピソード中に赤信号で停止線を越えた回数
    signal_violations: int = 0
    # このエピソード中に車線を外れた回数（外れ始めた瞬間を 1 回と数える）
    lane_departures: int = 0
    # いま適用されている最高速度標識の規制速度 [m/s]。標識が無い区間は 0.0
    speed_limit: float = 0.0
    # このエピソード中に規制速度を超えた回数（超え始めた瞬間を 1 回と数える）
    speed_violations: int = 0
    route: list[tuple[float, float]] | None = None  # 変化があったフレームのみ非 None

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "active": self.active,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "heading": round(self.heading, 4),
            "speed": round(self.speed, 3),
            "steer": round(self.steer, 4),
            "collided": self.collided,
            "reachedGoal": self.reached_goal,
            "goal": [round(self.goal[0], 3), round(self.goal[1], 3)],
            "progress": round(self.progress, 4),
            "signalViolations": self.signal_violations,
            "laneDepartures": self.lane_departures,
            "speedLimit": round(self.speed_limit, 3),
            "speedViolations": self.speed_violations,
        }
        if self.route is not None:
            out["route"] = [[round(px, 2), round(py, 2)] for px, py in self.route]
        return out


@dataclass
class ObstacleSnapshot:
    id: int
    x: float
    y: float
    radius: float

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "radius": self.radius,
        }


@dataclass
class FrameSnapshot:
    """docs/protocol.md 2.3 の frame に対応。vehicles は常に全スロット分含む。"""

    tick: int
    sim_time: float
    vehicles: list[VehicleSnapshot]
    obstacles: list[ObstacleSnapshot]
    # 信号の現示。MapData.signals と同じ並びで 0=青 / 1=黄 / 2=赤
    signals: list[int] = field(default_factory=list)
    # 各車両の認識結果（スロット番号 -> `percep.Detection.to_wire()` のリスト）。
    # ★ **PPO が入力として受け取っているのと同じ検出結果**を送る。
    #   運転席カメラのバウンディングボックスはこれを描くので、
    #   別経路で作り直してはいけない（画面と学習が食い違うと誰も気づけない）。
    detections: dict[int, list[dict[str, Any]]] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tick": self.tick,
            "simTime": round(self.sim_time, 3),
            "vehicles": [v.to_wire() for v in self.vehicles],
            "obstacles": [o.to_wire() for o in self.obstacles],
        }
        if self.signals:
            payload["signals"] = self.signals
        if self.detections:
            # JSON のキーは文字列でなければならない
            payload["detections"] = {
                str(slot): dets for slot, dets in self.detections.items()
            }
        return payload


def validate_hidden_sizes(value: Any) -> tuple[list[int] | None, str]:
    """`set_network` の hiddenSizes を検証する。

    通れば `(サイズの一覧, "")`、駄目なら `(None, 理由)`。
    **`set_params` と同じく、値を信用せずここで弾く。** 形が壊れたまま
    通すと `nn.Linear` の生成で落ち、学習スレッドごと死ぬ。
    """
    if not isinstance(value, (list, tuple)):
        return None, "hiddenSizes は数値の配列で指定してください"
    if not (config.PPO_HIDDEN_MIN_LAYERS <= len(value) <= config.PPO_HIDDEN_MAX_LAYERS):
        return None, (
            f"隠れ層の数は {config.PPO_HIDDEN_MIN_LAYERS}〜"
            f"{config.PPO_HIDDEN_MAX_LAYERS} 層にしてください（指定は {len(value)} 層）"
        )
    out: list[int] = []
    for i, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None, f"{i + 1} 層目が数値ではありません"
        if not math.isfinite(float(raw)):
            return None, f"{i + 1} 層目が有限の数値ではありません"
        width = int(raw)
        if not (config.PPO_HIDDEN_MIN_WIDTH <= width <= config.PPO_HIDDEN_MAX_WIDTH):
            return None, (
                f"{i + 1} 層目の幅は {config.PPO_HIDDEN_MIN_WIDTH}〜"
                f"{config.PPO_HIDDEN_MAX_WIDTH} にしてください（指定は {width}）"
            )
        out.append(width)
    return out, ""


@dataclass
class MetricsSnapshot:
    """docs/protocol.md 2.6 の metrics に対応。"""

    tick: int = 0
    wall_time: float = 0.0
    updates: int = 0
    episodes: int = 0
    mean_episode_reward: float = 0.0
    mean_episode_length: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    collision_rate: float = 0.0
    goal_rate: float = 0.0
    steps_per_sec: float = 0.0
    # 1 エピソードあたりの信号無視回数（道交法施行令 2 条の赤色の灯火）
    signal_violations: float = 0.0
    # 1 エピソードあたりの速度超過回数（道交法 22 条の最高速度）
    speed_violations: float = 0.0
    # 車線中心からの横方向のずれの平均 [m]。経路が車線中心線なので、そのまま
    # 「車線からどれだけはみ出しているか」を表す
    lane_deviation: float = 0.0

    def to_wire(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "wallTime": round(self.wall_time, 2),
            "updates": self.updates,
            "episodes": self.episodes,
            "meanEpisodeReward": round(self.mean_episode_reward, 3),
            "meanEpisodeLength": round(self.mean_episode_length, 1),
            "policyLoss": round(self.policy_loss, 5),
            "valueLoss": round(self.value_loss, 5),
            "entropy": round(self.entropy, 4),
            "approxKl": round(self.approx_kl, 5),
            "collisionRate": round(self.collision_rate, 3),
            "goalRate": round(self.goal_rate, 3),
            "stepsPerSec": round(self.steps_per_sec, 2),
            "signalViolations": round(self.signal_violations, 3),
            "speedViolations": round(self.speed_violations, 3),
            "laneDeviation": round(self.lane_deviation, 3),
        }


# ---------------------------------------------------------------------------
# 環境の 1 ステップ結果（sim -> runtime -> rl）
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """1 台分のエピソードが終了したときの記録。"""

    slot: int
    total_reward: float
    length: int
    reason: str  # "goal" | "collision" | "offroad" | "timeout"
    signal_violations: int = 0  # このエピソード中に赤信号で停止線を越えた回数
    speed_violations: int = 0    # 規制速度を超えた回数（超え始めた瞬間を 1 回）
    lane_deviation: float = 0.0  # 車線中心からの横方向のずれの平均 [m]
    lane_departures: int = 0     # 車線を外れた回数（外れ始めた瞬間を 1 回）


@dataclass
class StepResult:
    """SimulationEnv.step() の戻り値。

    配列はすべて先頭次元が MAX_VEHICLES のスロット添字で、非アクティブなスロットも
    含む（memo 5章「擬似固定エージェント数方式」）。学習側は `active` でマスクする。
    """

    obs: np.ndarray          # shape (MAX_VEHICLES, OBS_DIM), float32。ステップ後の観測
    rewards: np.ndarray      # shape (MAX_VEHICLES,), float32
    dones: np.ndarray        # shape (MAX_VEHICLES,), bool。エピソード終了フラグ
    active: np.ndarray       # shape (MAX_VEHICLES,), bool。ステップ時点で有効だったか

    #: shape (MAX_VEHICLES,), bool。`dones` のうち**打ち切り**（時間切れ）だったもの。
    #: ★ 到達・衝突・道路外は本物の終端だが、`MAX_EPISODE_STEPS` による時間切れは
    #:   truncation であって世界の終わりではない。ここを `dones` と混ぜると
    #:   GAE がブートストラップを切り、価値目標が `r + γV(s')` から `γV(s')` ぶん
    #:   ずれる（code_review L-08）。
    truncated: np.ndarray | None = None
    episodes: list[EpisodeResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ユーザー介入イベント（WebSocket -> runtime -> sim）
# ---------------------------------------------------------------------------


@dataclass
class InterventionEvent:
    """memo 5章「介入も現実の交通現象の一部」。学習を止めずステップ境界で適用する。

    kind に取り得る値:
        "spawn_vehicle" / "despawn_vehicle" / "add_obstacle"
        "remove_obstacle" / "clear_obstacles" / "reset_episode"
    """

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
