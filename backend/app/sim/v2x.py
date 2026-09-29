"""車車間通信（V2X）。各車が 4 次元のメッセージを出し、30m 以内の近い 2 台から受け取ったものを平均して観測の末尾に足す。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from app import config
from app.percep.encoder import SURROUND_CLASSES, local_xy
from app.percep.types import DEFAULT_CAMERA, SURROUND_CAMERAS, CameraSpec, DetClass, PerceptionResult

if TYPE_CHECKING:
    from app.sim.world import World

__all__ = [
    "V2XMessageRouter",
    "vehicle_message",
    "DANGER_RANGE_M",
    "JUNCTION_RANGE_M",
]

#: 危険（歩行者・障害物）を知らせる距離 [m]。これより近いほど 1.0 に近づく
DANGER_RANGE_M = 30.0
#: 交差点への近さを知らせる距離 [m]。交差点の直前ほど 1.0 に近づく
JUNCTION_RANGE_M = 30.0

#: 危険として知らせるクラス。車両は受け取る側が自分のカメラで見るので載せない
_DANGER_CLASSES: tuple[DetClass, ...] = tuple(c for c in SURROUND_CLASSES if c != DetClass.VEHICLE)


def _nearness(distance_m: float, reach_m: float) -> float:
    """近いほど 1.0・reach_m 以上で 0.0。交差点を越えた直後の負の距離でも 1.0 で止める。"""
    return min(1.0, max(0.0, 1.0 - float(distance_m) / reach_m))


def vehicle_message(speed_ratio: float, turn: int, danger_m: float, junction_m: float) -> np.ndarray:
    """1 台が出すメッセージ（車速・右左折の意図・危険・交差点への近さ）。どの要素も -1..1。"""
    return np.array(
        [
            min(max(float(speed_ratio), 0.0), 1.0),
            float(np.sign(turn)),
            _nearness(danger_m, DANGER_RANGE_M),
            _nearness(junction_m, JUNCTION_RANGE_M),
        ],
        dtype=np.float32,
    )


def _nearest_danger(
    front: PerceptionResult | None,
    spec: CameraSpec,
    surround: dict[str, PerceptionResult] | None,
) -> float:
    """自車のカメラ（前方と周囲）に写った、いちばん近い歩行者・障害物までの距離 [m]。無ければ inf。"""
    best = float("inf")
    views: list[tuple[PerceptionResult, CameraSpec]] = []
    if front is not None:
        views.append((front, spec))
    if surround:
        for cam in SURROUND_CAMERAS:
            result = surround.get(cam.key)
            if result is not None:
                views.append((result, cam))
    for result, cam in views:
        for det in result.detections:
            if det.cls not in _DANGER_CLASSES:
                continue
            _fx, _fy, dist = local_xy(det, cam)
            if dist < best:
                best = dist
    return best


class V2XMessageRouter:
    """メッセージを作り、近傍の車へ配って平均する。台数が少ない（最大 `MAX_VEHICLES`）ので距離は総当たりで測る。"""

    def __init__(
        self,
        num_agents: int = config.MAX_VEHICLES,
        range_m: float = config.V2X_RANGE_M,
        max_peers: int = config.V2X_MAX_PEERS,
    ) -> None:
        self.num_agents = int(num_agents)
        self.range_m = float(range_m)
        self.max_peers = int(max_peers)

    def compute_messages(
        self,
        world: "World",
        max_speed: float,
        perceptions: dict[int, PerceptionResult],
        surround: dict[int, dict[str, PerceptionResult]] | None = None,
        spec: CameraSpec = DEFAULT_CAMERA,
    ) -> np.ndarray:
        """走っている車ごとのメッセージ (N, 4)。走っていない車は 0。"""
        n = self.num_agents
        out = np.zeros((n, config.OBS_V2X_DIM), dtype=np.float32)
        fleet = world.fleet
        top = max(float(max_speed), 1e-3)
        for slot in np.flatnonzero(fleet.active):
            s = int(slot)
            signal_m, _phase = world.next_signal(s)
            junction_m = min(float(world.next_junction(s)), float(signal_m))
            danger_m = _nearest_danger(
                perceptions.get(s), spec, None if surround is None else surround.get(s)
            )
            out[s] = vehicle_message(
                float(fleet.speed[s]) / top, int(world.turn_signal[s]), danger_m, junction_m
            )
        return out

    def route_and_aggregate(
        self, xy: np.ndarray, active: np.ndarray, messages: np.ndarray
    ) -> tuple[np.ndarray, dict[int, tuple[int, ...]]]:
        """30m 以内の近い 2 台から受け取ったメッセージの平均 (N, 4) と、受け取った相手。相手がいなければ 0。"""
        n = self.num_agents
        inbox = np.zeros((n, config.OBS_V2X_DIM), dtype=np.float32)
        links: dict[int, tuple[int, ...]] = {}
        idx = np.flatnonzero(np.asarray(active, dtype=bool).reshape(n))
        if idx.size < 2 or self.max_peers <= 0:
            return inbox, links
        pts = np.asarray(xy, dtype=np.float64).reshape(n, 2)[idx]
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.hypot(diff[..., 0], diff[..., 1])
        np.fill_diagonal(dist, np.inf)
        order = np.argsort(dist, axis=1, kind="stable")[:, : self.max_peers]
        for row, slot in enumerate(idx):
            near = order[row][dist[row, order[row]] <= self.range_m]
            if near.size == 0:
                continue
            peers = idx[near]
            inbox[int(slot)] = messages[peers].mean(axis=0)
            links[int(slot)] = tuple(int(p) for p in peers)
        return inbox, links
