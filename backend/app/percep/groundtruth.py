# -*- coding: utf-8 -*-
"""world の真値から「理想の検出結果」を作る。

★ **この関数は 2 つの役割を兼ねる。** 互換のための逃げ道ではない。

  1. 認識器（`percep/detector.py`）の**教師データ**
  2. 認識器がまだ学習できていないときの**フォールバック**
     （`config.PERCEP_FALLBACK_GROUND_TRUTH`）

  同じ関数を両方に使うのが要点で、分けてはいけない。分けると
  「フォールバックでは走れるのに、学習した認識器に差し替えると走れない」
  という食い違いが生まれ、原因が認識器なのか教師データなのか切り分けられなくなる。
  この設計のおかげで「まず走らせる → 教師データを集める → 認識器を学習する →
  差し替える」という順序で立ち上げられる。

投影の作法
----------
`percep/camera.py` の `PseudoCamera` と**同じ透視投影**を使う。camera.py が
未完成のあいだ実装が止まらないよう、投影のヘルパーはこのモジュールに自己完結で
書いてある（後で共通化する前提）。両者がずれると、画像に写っていない場所に
正解の箱が付く＝**学習不能な教師データ**になるので、共通化のときは
`camera_pose()` / `project_points()` をそのまま移すこと。

座標系（docs/protocol.md 1章）:
    ENU 平面・メートル・x=東 / y=北 / heading は +x 軸から反時計回り。
    自車座標系は「前方 +x / 左 +y」。カメラは heading 方向を向き、
    `CameraSpec.pitch` だけ下を向く。

3D の寸法はフロントエンドの `scene/signalGeometry.ts` / `scene/signGeometry.ts`
と一致させてある。**片方だけ変えると、画面の見た目と検出枠がずれる。**
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from app import config
from app.percep.types import (
    CLASS_QUOTA,
    DEFAULT_CAMERA,
    LANE_LOOKAHEAD_M,
    LANE_POLYLINE_POINTS,
    CameraSpec,
    DetClass,
    Detection,
    PerceptionResult,
    pack_by_class_quota,
)

if TYPE_CHECKING:  # 実行時に import しない（sim -> percep の循環を避けるため）
    from app.sim.world import World

__all__ = [
    "CameraPose",
    "camera_pose",
    "clear_static_cache",
    "detect_ground_truth",
    "detect_ground_truth_batch",
    "freespace_ground_truth",
    "project_points",
]


# ---------------------------------------------------------------------------
# 3D の寸法（frontend/src/scene/*.ts と一致させること）
# ---------------------------------------------------------------------------

# --- 信号機（signalGeometry.ts）---
SIGNAL_HOUSING_W = 1.16          # 灯器筐体の幅 [m]（3 灯 + 縁）
SIGNAL_HOUSING_H = 0.44          # 灯器筐体の高さ [m]
SIGNAL_MOUNT_HEIGHT = 5.0        # 灯器下端の路面からの高さ [m]
SIGNAL_HEAD_Z = SIGNAL_MOUNT_HEIGHT + SIGNAL_HOUSING_H * 0.5  # 筐体中心の高さ
SIGNAL_BEYOND_MARGIN = 2.0       # 交差点中心から灯器までの余白 [m]

# --- 最高速度標識（signGeometry.ts）---
SIGN_RADIUS = 0.3                # 標示板の半径 [m]（直径 60cm）
SIGN_BOTTOM_HEIGHT = 1.8         # 標示板下端の高さ [m]
SIGN_BOARD_Z = SIGN_BOTTOM_HEIGHT + SIGN_RADIUS  # 標示板中心の高さ

# ---------------------------------------------------------------------------
# 検出の可視条件
# ---------------------------------------------------------------------------

#: 灯器・標示板が運転者に正対していると認める角度差 [rad]。
#: これを超えると裏側を見ていることになり、灯色も数字も読めない。
#: **「箱は見えるが色は読めない」を検出扱いにしない**のは、
#: `Detection.phase` が必ず埋まっている契約にするため。
FACING_TOLERANCE = math.radians(75.0)

#: これより小さい箱は検出扱いにしない [px]。
#: 幅 192px・水平画角 68 度では焦点距離 142.3px なので、
#: 1.16m の灯器は 60m 先で 2.75px、0.6m の標識は 20m 先で 4.3px になる。
#: ここを 0 にすると「1px に満たない＝画像に情報が無いもの」まで正解に含まれ、
#: 認識器に**原理的に学習できない目標**を与えることになる。
MIN_BOX_PX = 1.5

# ★ `LANE_LOOKAHEAD_M` と `LANE_POLYLINE_POINTS` は `percep/types.py` にある
#   （code_review Q-10）。認識器（`detector.py`）が同じ値を別に持っていて
#   「揃えること」というコメントだけで担保されていたのを 1 か所へまとめた。
#   ずれると「認識器の車線だけ長さや点数が違う」という形で出るが、
#   型でもビルドでも捕まらない。
#: 車線の見た目の半幅 [m]（車線幅 3.2m 相当）
LANE_HALF_WIDTH_M = 1.6
#: 車線をサンプルする間隔 [m]
LANE_SAMPLE_M = 2.5

#: 遮蔽判定のサンプル間隔 [m]。建物の裏の信号を「見えている」ことにしないため。
OCCLUSION_STEP_M = 2.0
#: 遮蔽判定の最大サンプル数（遠方でも打ち切る）
OCCLUSION_MAX_SAMPLES = 64


#: 走行可能領域を測る向き（heading からの相対角 [rad]）。
#: config.OBS_FREESPACE_DIM 本を前方 ±90 度に等分する。
#: ★ ±90 度は水平画角 68 度より広い。**両端は画像に写っていない**ので、
#:   認識器は端の 2〜3 本を画像から推定できない（報告書の懸念を参照）。
FREESPACE_ANGLES = np.linspace(
    -math.pi / 2.0, math.pi / 2.0, config.OBS_FREESPACE_DIM, dtype=np.float32
)

#: 車両を走行可能領域のレイで遮る円の半径 [m]。
#: 長方形（4.4 x 1.8）を円で近似する。長さで取ると横に広がりすぎ、
#: 幅で取ると前後に短すぎるので平均を使う。
VEHICLE_BLOCK_RADIUS = (config.VEHICLE_LENGTH + config.VEHICLE_WIDTH) * 0.25


# ---------------------------------------------------------------------------
# カメラ（camera.py が出来たら共通化する）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraPose:
    """擬似カメラの外部パラメータ（1 台ぶん）。

    `CameraSpec` が内部パラメータ（画角・解像度）で、こちらが姿勢。
    """

    eye_x: float
    eye_y: float
    eye_z: float
    cos_yaw: float
    sin_yaw: float
    cos_pitch: float
    sin_pitch: float


def camera_pose(x: float, y: float, heading: float, spec: CameraSpec) -> CameraPose:
    """車両の姿勢から運転席カメラの姿勢を作る。

    **`frontend/src/scene/cameraMath.ts` の `driverEye()` と同じ式。**
    日本車は右ハンドルなので進行方向の右へずらす（ENU では heading - 90 度）。
    """
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    # 進行方向の右手 = (sin(h), -cos(h))
    eye_x = x + cos_h * spec.forward + sin_h * spec.right
    eye_y = y + sin_h * spec.forward - cos_h * spec.right
    pitch = spec.pitch
    return CameraPose(
        eye_x=float(eye_x),
        eye_y=float(eye_y),
        eye_z=float(spec.eye_height),
        cos_yaw=float(cos_h),
        sin_yaw=float(sin_h),
        cos_pitch=float(math.cos(pitch)),
        sin_pitch=float(math.sin(pitch)),
    )


def project_points(
    pose: CameraPose, spec: CameraSpec, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ワールド座標 (N, 3) をカメラ画像へ透視投影する。

    Returns:
        (u, v, depth)。u / v は**画素**（左上原点）、depth は光軸方向の距離 [m]。
        depth <= 0 はカメラの背後なので、呼び出し側で必ず捨てること。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    dx = pts[:, 0] - pose.eye_x
    dy = pts[:, 1] - pose.eye_y
    dz = pts[:, 2] - pose.eye_z

    # 自車座標系（前方 +x / 左 +y / 上 +z）へ
    forward = dx * pose.cos_yaw + dy * pose.sin_yaw
    left = -dx * pose.sin_yaw + dy * pose.cos_yaw

    # 俯角（左軸まわりの回転）。pitch は下向きが負。
    depth = forward * pose.cos_pitch + dz * pose.sin_pitch
    up = -forward * pose.sin_pitch + dz * pose.cos_pitch

    focal = spec.focal_px
    safe = np.where(np.abs(depth) < 1e-6, np.float64(1e-6), depth)
    # 画像 x は右が正なので、カメラ左方向 `left` は符号を反転して足す
    u = spec.width * 0.5 - focal * (left / safe)
    v = spec.height * 0.5 - focal * (up / safe)
    return u, v, depth


def _box_from_points(
    pose: CameraPose,
    spec: CameraSpec,
    points: np.ndarray,
    *,
    require_all_in_front: bool = True,
) -> tuple[float, float, float, float] | None:
    """点群を囲む正規化 BBox を返す。視野外・背後・小さすぎるものは None。

    `require_all_in_front=True` は「近接平面をまたぐ物体は捨てる」という意味。
    信号・標識・車両のような小さい物体では、またいだ時点でほぼ画面外なので
    素直に捨てたほうが正しい。車線のように大きく広がるものだけ False にする。
    """
    u, v, depth = project_points(pose, spec, points)
    front = depth > spec.near
    if require_all_in_front:
        if not bool(front.all()):
            return None
    else:
        if not bool(front.any()):
            return None
        u, v = u[front], v[front]

    x0 = float(np.min(u))
    x1 = float(np.max(u))
    y0 = float(np.min(v))
    y1 = float(np.max(v))

    # 画面外（完全に外れている）
    if x1 <= 0.0 or y1 <= 0.0 or x0 >= spec.width or y0 >= spec.height:
        return None
    # 小さすぎて画像に情報が残らない
    if (x1 - x0) < MIN_BOX_PX and (y1 - y0) < MIN_BOX_PX:
        return None

    # 画面内へ切り詰めてから正規化座標にする
    x0 = min(max(x0, 0.0), float(spec.width))
    x1 = min(max(x1, 0.0), float(spec.width))
    y0 = min(max(y0, 0.0), float(spec.height))
    y1 = min(max(y1, 0.0), float(spec.height))
    if x1 - x0 <= 0.0 or y1 - y0 <= 0.0:
        return None
    return (x0 / spec.width, y0 / spec.height, x1 / spec.width, y1 / spec.height)


# ---------------------------------------------------------------------------
# マップ由来の静的な 3D 形状（毎ステップ組み直さないためのキャッシュ）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StaticScene:
    """信号・標識の 3D 位置をあらかじめ配列にしたもの。

    毎ステップ・毎車両で `MapData.signals` を走査すると、金沢（灯器 2,736 基）
    では Python のループだけで予算を使い切る。マップは走行中変わらないので
    一度だけ作る。
    """

    # 信号: 灯器筐体の中心 (S, 3) と、幅方向の単位ベクトル (S, 2)
    signal_head: np.ndarray
    signal_across: np.ndarray
    signal_heading: np.ndarray
    signal_stop: np.ndarray       # 停止線の位置 (S, 2)
    # 標識: 標示板の中心 (G, 3)、幅方向 (G, 2)、規制速度 (G,)
    sign_board: np.ndarray
    sign_across: np.ndarray
    sign_heading: np.ndarray
    sign_limit: np.ndarray


#: (map_index, _StaticScene) の 1 件キャッシュ。マップは同時に 1 つしか
#: 読まれないので 1 件で足りる。タプルの差し替えは CPython では不可分なので、
#: 別スレッドから読まれても owner と scene がちぐはぐになることはない。
#:
#: ★ **これは「1 プロセスに `SimulationEnv` は 1 つ」という運用の前提に乗った
#:   設計で、コードはその制約を強制していない**（code_review P-06）。
#:   当たり外れは `map_index` の identity 比較だけで決まるので、テストや
#:   `train_detector.py` の中で複数の `map_index` を同時に扱うと、片方の env が
#:   別の env のキャッシュを黙って再利用しうる。**複数マップを行き来する
#:   コードを書くときは切り替えのたびに `clear_static_cache()` を呼ぶこと。**
#:   恒久的に複数マップを同時に扱うなら、キャッシュを `World` か
#:   `SimulationEnv` のインスタンスへ紐づける設計に変えること。
_STATIC_CACHE: tuple[object, _StaticScene] | None = None


def clear_static_cache() -> None:
    """静的シーンのキャッシュを捨てる（テストとマップ入れ替えの検証用）。"""
    global _STATIC_CACHE
    _STATIC_CACHE = None


def _build_static_scene(map_index) -> _StaticScene:
    """`MapData` から信号・標識の 3D 形状を組み立てる。

    信号の灯器位置は `frontend/src/scene/signalGeometry.ts` と同じ式で置く:
        交差点中心から進行方向へ (road_width/2 + 2.0)、さらに左へ road_width*0.25。
    """
    data = getattr(map_index, "data", None)
    signals = list(getattr(data, "signals", []) or [])
    signs = list(getattr(data, "signs", []) or [])
    nodes = list(getattr(data, "nodes", []) or [])
    node_xy = {int(n.id): (float(n.x), float(n.y)) for n in nodes}

    head = np.zeros((len(signals), 3), dtype=np.float64)
    across_sig = np.zeros((len(signals), 2), dtype=np.float64)
    heading_sig = np.zeros(len(signals), dtype=np.float64)
    stop = np.zeros((len(signals), 2), dtype=np.float64)
    for i, sig in enumerate(signals):
        cx, cy = node_xy.get(int(sig.node_id), (float(sig.x), float(sig.y)))
        cos_h = math.cos(float(sig.heading))
        sin_h = math.sin(float(sig.heading))
        # 進行方向の左（ENU では反時計回りに 90 度）
        left_x, left_y = -sin_h, cos_h
        beyond = float(sig.road_width) * 0.5 + SIGNAL_BEYOND_MARGIN
        offset = float(sig.road_width) * 0.25
        head[i, 0] = cx + cos_h * beyond + left_x * offset
        head[i, 1] = cy + sin_h * beyond + left_y * offset
        head[i, 2] = SIGNAL_HEAD_Z
        across_sig[i] = (left_x, left_y)
        heading_sig[i] = float(sig.heading)
        stop[i] = (float(sig.x), float(sig.y))

    board = np.zeros((len(signs), 3), dtype=np.float64)
    across_sgn = np.zeros((len(signs), 2), dtype=np.float64)
    heading_sgn = np.zeros(len(signs), dtype=np.float64)
    limit = np.zeros(len(signs), dtype=np.float64)
    for i, sgn in enumerate(signs):
        cos_h = math.cos(float(sgn.heading))
        sin_h = math.sin(float(sgn.heading))
        board[i] = (float(sgn.x), float(sgn.y), SIGN_BOARD_Z)
        across_sgn[i] = (-sin_h, cos_h)
        heading_sgn[i] = float(sgn.heading)
        limit[i] = float(sgn.speed_limit)

    return _StaticScene(
        signal_head=head,
        signal_across=across_sig,
        signal_heading=heading_sig,
        signal_stop=stop,
        sign_board=board,
        sign_across=across_sgn,
        sign_heading=heading_sgn,
        sign_limit=limit,
    )


def _static_scene(map_index) -> _StaticScene:
    global _STATIC_CACHE
    cached = _STATIC_CACHE
    if cached is not None and cached[0] is map_index:
        return cached[1]
    scene = _build_static_scene(map_index)
    _STATIC_CACHE = (map_index, scene)
    return scene


# ---------------------------------------------------------------------------
# 遮蔽（建物の裏にあるものを「見えている」ことにしない）
# ---------------------------------------------------------------------------


def _line_of_sight(map_index, eye: tuple[float, float], targets: np.ndarray) -> np.ndarray:
    """視線が建物を貫いていないターゲットを True で返す。shape (M,)。

    占有グリッド（`OccupancyGrid.building`）を線分上で数点だけ拾う近似。
    厳密な遮蔽ではないが、**遮蔽を一切考えないと「建物の裏の信号が見える」
    教師データ**になり、画像には写っていないものを検出しろと教えることになる。
    症状は「認識器の精度が上がらない」だけで、原因が教師データ側にあることに
    外から気づけないので、粗くても入れておく。

    グリッドが無い / 失敗した場合はすべて可視とみなす（走行は続けられる）。
    """
    m = int(targets.shape[0])
    if m == 0:
        return np.zeros(0, dtype=bool)
    grid = getattr(map_index, "occupancy", None)
    if grid is None:
        return np.ones(m, dtype=bool)

    dx = targets[:, 0] - eye[0]
    dy = targets[:, 1] - eye[1]
    dist = np.hypot(dx, dy)
    steps = int(min(OCCLUSION_MAX_SAMPLES, max(2, math.ceil(float(np.max(dist)) / OCCLUSION_STEP_M))))
    # 端点（カメラ自身と対象そのもの）は含めない。対象は建物に接して立つため
    t = np.linspace(0.0, 1.0, steps + 2, dtype=np.float64)[1:-1]
    sample_x = eye[0] + dx[:, None] * t[None, :]
    sample_y = eye[1] + dy[:, None] * t[None, :]
    try:
        blocked = grid.sample_building(sample_x, sample_y)
    except Exception:
        return np.ones(m, dtype=bool)
    return ~np.asarray(blocked, dtype=bool).reshape(m, -1).any(axis=1)


# ---------------------------------------------------------------------------
# クラスごとの検出
# ---------------------------------------------------------------------------


def _angle_diff(a: np.ndarray, b: float) -> np.ndarray:
    """角度差を (-pi, pi] に畳んで返す。"""
    d = np.asarray(a, dtype=np.float64) - float(b)
    return np.arctan2(np.sin(d), np.cos(d))


def _detect_signals(
    world: "World", slot: int, spec: CameraSpec, pose: CameraPose, scene: _StaticScene
) -> list[tuple[float, Detection]]:
    """前方の信号機。灯色は `world.signal_phases` の真値をそのまま入れる。"""
    out: list[tuple[float, Detection]] = []
    head = scene.signal_head
    if head.shape[0] == 0:
        return out

    eye = (pose.eye_x, pose.eye_y)
    dx = scene.signal_stop[:, 0] - eye[0]
    dy = scene.signal_stop[:, 1] - eye[1]
    dist = np.hypot(dx, dy)
    heading = float(world.fleet.heading[slot])
    # 正対しているものだけ（裏を向いた灯器は灯色が読めない）
    facing = np.abs(_angle_diff(scene.signal_heading, heading)) <= FACING_TOLERANCE
    candidates = np.flatnonzero((dist <= spec.far) & facing)
    if candidates.size == 0:
        return out

    visible = _line_of_sight(world.map_index, eye, scene.signal_stop[candidates])
    candidates = candidates[visible]

    phases = world.signal_phases
    half_w = SIGNAL_HOUSING_W * 0.5
    half_h = SIGNAL_HOUSING_H * 0.5
    for i in candidates:
        center = head[i]
        across = scene.signal_across[i]
        corners = np.array(
            [
                [center[0] - across[0] * half_w, center[1] - across[1] * half_w, center[2] - half_h],
                [center[0] + across[0] * half_w, center[1] + across[1] * half_w, center[2] - half_h],
                [center[0] - across[0] * half_w, center[1] - across[1] * half_w, center[2] + half_h],
                [center[0] + across[0] * half_w, center[1] + across[1] * half_w, center[2] + half_h],
            ],
            dtype=np.float64,
        )
        box = _box_from_points(pose, spec, corners)
        if box is None:
            continue
        idx = int(i)
        phase = int(phases[idx]) if 0 <= idx < len(phases) else 2
        out.append(
            (
                float(dist[idx]),
                Detection(
                    cls=DetClass.TRAFFIC_LIGHT,
                    x0=box[0], y0=box[1], x1=box[2], y1=box[3],
                    confidence=1.0,
                    phase=phase,
                    # ★ 停止線までの水平距離。灯器そのものまでの距離ではない
                    #   （観測は「停止線までの距離」を必要とするため）
                    distance=float(dist[idx]),
                ),
            )
        )
    return out


def _detect_speed_signs(
    world: "World", slot: int, spec: CameraSpec, pose: CameraPose, scene: _StaticScene
) -> list[tuple[float, Detection]]:
    """最高速度標識。規制速度は `MapSign.speed_limit` の真値。"""
    out: list[tuple[float, Detection]] = []
    board = scene.sign_board
    if board.shape[0] == 0:
        return out

    eye = (pose.eye_x, pose.eye_y)
    dx = board[:, 0] - eye[0]
    dy = board[:, 1] - eye[1]
    dist = np.hypot(dx, dy)
    heading = float(world.fleet.heading[slot])
    facing = np.abs(_angle_diff(scene.sign_heading, heading)) <= FACING_TOLERANCE
    candidates = np.flatnonzero((dist <= spec.far) & facing)
    if candidates.size == 0:
        return out

    visible = _line_of_sight(world.map_index, eye, board[candidates, :2])
    candidates = candidates[visible]

    for i in candidates:
        center = board[i]
        across = scene.sign_across[i]
        corners = np.array(
            [
                [center[0] - across[0] * SIGN_RADIUS, center[1] - across[1] * SIGN_RADIUS, center[2] - SIGN_RADIUS],
                [center[0] + across[0] * SIGN_RADIUS, center[1] + across[1] * SIGN_RADIUS, center[2] - SIGN_RADIUS],
                [center[0] - across[0] * SIGN_RADIUS, center[1] - across[1] * SIGN_RADIUS, center[2] + SIGN_RADIUS],
                [center[0] + across[0] * SIGN_RADIUS, center[1] + across[1] * SIGN_RADIUS, center[2] + SIGN_RADIUS],
            ],
            dtype=np.float64,
        )
        box = _box_from_points(pose, spec, corners)
        if box is None:
            continue
        idx = int(i)
        out.append(
            (
                float(dist[idx]),
                Detection(
                    cls=DetClass.SPEED_SIGN,
                    x0=box[0], y0=box[1], x1=box[2], y1=box[3],
                    confidence=1.0,
                    speed_limit=float(scene.sign_limit[idx]),
                    distance=float(dist[idx]),
                ),
            )
        )
    return out


def _detect_vehicles(
    world: "World", slot: int, spec: CameraSpec, pose: CameraPose
) -> list[tuple[float, Detection]]:
    """他車両。車体の 8 頂点（路面と屋根の 4 隅）を投影して箱にする。"""
    out: list[tuple[float, Detection]] = []
    fleet = world.fleet
    others = np.flatnonzero(fleet.active)
    others = others[others != int(slot)]
    if others.size == 0:
        return out

    eye = (pose.eye_x, pose.eye_y)
    dx = fleet.x[others].astype(np.float64) - eye[0]
    dy = fleet.y[others].astype(np.float64) - eye[1]
    dist = np.hypot(dx, dy)
    near = others[dist <= spec.far]
    if near.size == 0:
        return out

    corners_all = fleet.corners()  # (N, 4, 2)
    height = float(config.VEHICLE_HEIGHT)
    for other in near:
        flat = corners_all[int(other)].astype(np.float64)  # (4, 2)
        pts = np.empty((8, 3), dtype=np.float64)
        pts[:4, :2] = flat
        pts[:4, 2] = 0.0
        pts[4:, :2] = flat
        pts[4:, 2] = height
        box = _box_from_points(pose, spec, pts, require_all_in_front=False)
        if box is None:
            continue
        d = float(math.hypot(float(fleet.x[other]) - eye[0], float(fleet.y[other]) - eye[1]))
        out.append(
            (
                d,
                Detection(
                    cls=DetClass.VEHICLE,
                    x0=box[0], y0=box[1], x1=box[2], y1=box[3],
                    confidence=1.0,
                    distance=d,
                ),
            )
        )
    return out


def _detect_obstacles(
    world: "World", spec: CameraSpec, pose: CameraPose
) -> list[tuple[float, Detection]]:
    """ユーザーが置いたパイロン。円柱を視線に垂直な板で近似する。"""
    out: list[tuple[float, Detection]] = []
    xy = world.obstacle_xy
    if xy.shape[0] == 0:
        return out

    eye = (pose.eye_x, pose.eye_y)
    dx = xy[:, 0].astype(np.float64) - eye[0]
    dy = xy[:, 1].astype(np.float64) - eye[1]
    dist = np.hypot(dx, dy)
    height = float(config.OBSTACLE_HEIGHT)
    for i in np.flatnonzero(dist <= spec.far):
        i = int(i)
        d = float(dist[i])
        if d < 1e-3:
            continue
        radius = float(world.obstacles[i].radius) if i < len(world.obstacles) else config.OBSTACLE_RADIUS
        # 視線に垂直な水平方向（円柱の見かけの幅はこの向きに広がる）
        ax, ay = -dy[i] / d, dx[i] / d
        cx, cy = float(xy[i, 0]), float(xy[i, 1])
        pts = np.array(
            [
                [cx - ax * radius, cy - ay * radius, 0.0],
                [cx + ax * radius, cy + ay * radius, 0.0],
                [cx - ax * radius, cy - ay * radius, height],
                [cx + ax * radius, cy + ay * radius, height],
            ],
            dtype=np.float64,
        )
        box = _box_from_points(pose, spec, pts, require_all_in_front=False)
        if box is None:
            continue
        out.append(
            (
                d,
                Detection(
                    cls=DetClass.OBSTACLE,
                    x0=box[0], y0=box[1], x1=box[2], y1=box[3],
                    confidence=1.0,
                    distance=d,
                ),
            )
        )
    return out


def _detect_lane(
    world: "World", slot: int, spec: CameraSpec, pose: CameraPose
) -> tuple[float, Detection] | None:
    """走行車線。経路（＝車線中心線）への射影から作る。

    `lateral` には `world.lateral[slot]`（左が正の符号付き横方向偏差 [m]）を
    そのまま入れる。ここが観測の車線項の真値になる。
    """
    state = world.slots[int(slot)]
    route = state.route
    if route.shape[0] < 2:
        return None
    cum = state.route_cum
    arc = float(world.arc[int(slot)])

    offsets = np.arange(0.0, LANE_LOOKAHEAD_M + 1e-6, LANE_SAMPLE_M, dtype=np.float64)
    targets = arc + offsets
    # ★ 経路の総延長を超えるサンプルは捨てる（code_review P-04）。
    #   `np.interp` は外挿せず終端の値でクランプするので、目的地の直前
    #   （残り 25m 未満）ではサンプル後半が同じ 1 点へ収束し、そこから取る
    #   接線 `np.gradient` がほぼ 0 になる。`norm` の下駄で例外にはならないが、
    #   車線の左右方向 `lx, ly` が不安定になって帯の向きと終端がぶれる。
    usable = int(np.count_nonzero(targets <= cum[-1]))
    if usable < 2:
        return None  # 目的地に着く寸前。車線としてたどれる長さが残っていない
    targets = targets[:usable]
    cx = np.interp(targets, cum, route[:, 0])
    cy = np.interp(targets, cum, route[:, 1])
    if cx.size < 2:
        return None

    # 各サンプル点の接線から左右へ広げて「車線の帯」にする
    tx = np.gradient(cx)
    ty = np.gradient(cy)
    norm = np.maximum(np.hypot(tx, ty), 1e-6)
    lx, ly = -ty / norm, tx / norm
    left = np.stack([cx + lx * LANE_HALF_WIDTH_M, cy + ly * LANE_HALF_WIDTH_M], axis=1)
    right = np.stack([cx - lx * LANE_HALF_WIDTH_M, cy - ly * LANE_HALF_WIDTH_M], axis=1)
    pts = np.concatenate([left, right], axis=0)
    pts3 = np.concatenate([pts, np.zeros((pts.shape[0], 1), dtype=np.float64)], axis=1)

    box = _box_from_points(pose, spec, pts3, require_all_in_front=False)
    if box is None:
        return None

    # 見えている車線の前方距離。交差点やカーブで短くなるので、
    # 「どこまで車線をたどれるか」の目安になる
    _, _, depth = project_points(pose, spec, pts3)
    visible_depth = float(np.max(depth[depth > spec.near])) if np.any(depth > spec.near) else 0.0

    # --- 路面へ重ねて描くための中心線（自車座標系 前方 +x / 左 +y）---
    # ★ ここを送らないと、フロントは車線を**矩形でしか描けない**。
    #   車線は細長い曲線なので、ボックスでは認識のずれが見えない。
    fleet = world.fleet
    ox = float(fleet.x[int(slot)])
    oy = float(fleet.y[int(slot)])
    heading = float(fleet.heading[int(slot)])
    cos_h = float(np.cos(heading))
    sin_h = float(np.sin(heading))
    stride = max(1, int(round(cx.size / max(LANE_POLYLINE_POINTS, 1))))
    keep = np.arange(0, cx.size, stride)
    if keep[-1] != cx.size - 1:
        keep = np.append(keep, cx.size - 1)  # 終端は必ず残す（線が途中で切れて見えるため）
    ddx = cx[keep] - ox
    ddy = cy[keep] - oy
    lane_points = [
        (float(fx), float(fy))
        for fx, fy in zip(ddx * cos_h + ddy * sin_h, -ddx * sin_h + ddy * cos_h)
    ]

    return (
        0.0,  # 車線は常に自車の足元にあるので優先度の距離は 0
        Detection(
            cls=DetClass.LANE,
            x0=box[0], y0=box[1], x1=box[2], y1=box[3],
            confidence=1.0,
            distance=min(visible_depth, float(spec.far)),
            lateral=float(world.lateral[int(slot)]),
            lane_points=lane_points,
        ),
    )


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------


def detect_ground_truth(
    world: "World", slot: int, spec: CameraSpec = DEFAULT_CAMERA
) -> PerceptionResult:
    """world の真値から「理想の検出結果」を作る。

    教師データ生成とフォールバックの**共通経路**。信頼度は常に 1.0 で、
    `phase` / `speed_limit` / `distance` / `lateral` も真値で埋まる。

    並び順は `PerceptionResult` の約束どおり信頼度の降順だが、真値では
    すべて 1.0 なので同着になる。同着の中は
    **「観測化に必要なものが必ず残る順」**（車線 → 信号 → 標識 → 車両 → 障害物 の
    ラウンドロビン、各クラス内は近い順）に並べる。`best()` が最寄りの信号を
    返すこと、`config.PERCEP_MAX_DETECTIONS`（12）で切っても内訳が欠けないことを
    保証するため。
    """
    slot = int(slot)
    result = PerceptionResult(slot=slot)
    if not (0 <= slot < len(world.slots)):
        return result
    if not bool(world.fleet.active[slot]):
        # 走っていないスロットには何も見えない（観測もゼロ埋めされる）
        return result

    pose = camera_pose(
        float(world.fleet.x[slot]),
        float(world.fleet.y[slot]),
        float(world.fleet.heading[slot]),
        spec,
    )
    scene = _static_scene(world.map_index)

    per_class: dict[DetClass, list[Detection]] = {c: [] for c in DetClass}
    buckets: list[tuple[DetClass, list[tuple[float, Detection]]]] = [
        (DetClass.TRAFFIC_LIGHT, _detect_signals(world, slot, spec, pose, scene)),
        (DetClass.SPEED_SIGN, _detect_speed_signs(world, slot, spec, pose, scene)),
        (DetClass.VEHICLE, _detect_vehicles(world, slot, spec, pose)),
        (DetClass.OBSTACLE, _detect_obstacles(world, spec, pose)),
    ]
    lane = _detect_lane(world, slot, spec, pose)
    if lane is not None:
        per_class[DetClass.LANE] = [lane[1]]
    for cls, items in buckets:
        items.sort(key=lambda pair: pair[0])  # 近い順
        per_class[cls] = [det for _, det in items[: CLASS_QUOTA[cls]]]

    # クラス枠つきの優先度ラウンドロビンで詰める（上限で切っても内訳が欠けない）。
    # ★ 認識器（`detector.decode_detections`）も**同じ関数**を呼ぶ。
    #   片方だけ直すと同じ食い違いがまた生まれる（code_review Q-01）。
    result.detections = pack_by_class_quota(
        per_class, int(config.PERCEP_MAX_DETECTIONS)
    )
    return result


def detect_ground_truth_batch(
    world: "World", slots: Sequence[int], spec: CameraSpec = DEFAULT_CAMERA
) -> list[PerceptionResult]:
    """複数スロットぶんまとめて作る（教師データ収集用の薄いラッパ）。"""
    return [detect_ground_truth(world, int(s), spec) for s in slots]


def freespace_ground_truth(
    world: "World",
    slot: int,
    spec: CameraSpec = DEFAULT_CAMERA,
    max_distance: float = float(config.OBS_FREESPACE_MAX_DISTANCE),
) -> np.ndarray:
    """走行可能領域の真値。前方 ±90 度を `OBS_FREESPACE_DIM` 本に分けた距離 [m]。

    ★ `Detection` にも `PerceptionResult` にも走行可能領域を入れる場所が無い
      （`DetClass` は箱で表せる 5 クラスしか持たない）。そこで**検出とは別の
      戻り値**として扱い、認識器側も専用の出力ヘッドを持たせてある。
      `types.py` に持たせるかどうかは要相談（報告の「要確認」を参照）。

    建物は占有グリッドのレイキャスト、他車両と障害物は円との交差で測る。
    観測の元になっていた建物レイキャストと同じ値に、動く障害物を足したもの。
    """
    slot = int(slot)
    out = np.full(config.OBS_FREESPACE_DIM, float(max_distance), dtype=np.float32)
    if not (0 <= slot < len(world.slots)) or not bool(world.fleet.active[slot]):
        return out

    heading = float(world.fleet.heading[slot])
    pose = camera_pose(
        float(world.fleet.x[slot]), float(world.fleet.y[slot]), heading, spec
    )
    ox, oy = pose.eye_x, pose.eye_y
    angles = (heading + FREESPACE_ANGLES).astype(np.float64)

    # --- 建物 ---
    try:
        hit = world.map_index.raycast(
            np.array([ox], dtype=np.float32),
            np.array([oy], dtype=np.float32),
            angles.reshape(1, -1).astype(np.float32),
            float(max_distance),
            1.0,
        )
        out[:] = np.asarray(hit, dtype=np.float32).reshape(-1)
    except Exception:
        # レイキャストが失敗しても走行は続けられる。ここで例外を投げると
        # 教師データ生成が止まるので、最大距離のまま続ける
        pass

    # --- 他車両・障害物（円で近似してレイと交差させる） ---
    blockers: list[tuple[float, float, float]] = []
    fleet = world.fleet
    for other in np.flatnonzero(fleet.active):
        if int(other) == slot:
            continue
        blockers.append(
            (float(fleet.x[other]), float(fleet.y[other]), VEHICLE_BLOCK_RADIUS)
        )
    for obstacle in world.obstacles:
        blockers.append((float(obstacle.x), float(obstacle.y), float(obstacle.radius)))
    if blockers:
        circles = np.asarray(blockers, dtype=np.float64)  # (M, 3)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (R, 2)
        rel = circles[None, :, :2] - np.array([[ox, oy]], dtype=np.float64)[:, None, :]
        # 各レイ（行）× 各円（列）の投影長と垂線距離
        along = rel[..., 0] * dirs[:, None, 0] + rel[..., 1] * dirs[:, None, 1]
        perp2 = (rel * rel).sum(axis=2) - along * along
        radius2 = circles[None, :, 2] ** 2
        crosses = (along > 0.0) & (perp2 <= radius2)
        if crosses.any():
            back = np.sqrt(np.maximum(radius2 - perp2, 0.0))
            entry = np.where(crosses, np.maximum(along - back, 0.0), np.inf)
            out[:] = np.minimum(out, entry.min(axis=1).astype(np.float32))

    return np.clip(out, 0.0, float(max_distance)).astype(np.float32)
