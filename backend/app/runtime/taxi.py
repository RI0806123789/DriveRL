"""実用モード（自動運転タクシー）の配車・徴用の管理。"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from app import config
from app.contracts import (
    TAXI_PHASE_APPROACHING,
    TAXI_PHASE_ARRIVED,
    TAXI_PHASE_IDLE,
    TAXI_PHASE_RIDING,
    TAXI_PHASE_WAITING,
    TaxiStatus,
)

if TYPE_CHECKING:
    from app.sim.env import SimulationEnv

__all__ = ["TaxiService"]

logger = logging.getLogger(__name__)

ARRIVE_DISTANCE_M = 4.0
ARRIVE_SPEED_MPS = 0.8

MIN_TRIP_M = 50.0

ASSIGN_CANDIDATES = 4

ETA_MIN_SPEED_MPS = 3.5
ETA_SPEED_SMOOTH = 0.08


class TaxiService:
    """1 台だけ徴用してタクシーとして走らせる。**同時に 1 件だけ**（決定 8）。"""

    def __init__(self) -> None:
        self.status = TaxiStatus()
        self._speed_avg = 0.0

    @property
    def busy(self) -> bool:
        """配車を抱えているか。"""
        return self.status.phase != TAXI_PHASE_IDLE

    @property
    def vehicle_id(self) -> int:
        return int(self.status.vehicle_id)

    def request(
        self,
        env: "SimulationEnv",
        pickup: tuple[float, float],
        dropoff: tuple[float, float],
    ) -> str | None:
        """配車を受け付ける。断る場合はその理由を返す（成功なら None）。"""
        if self.busy:
            return "すでに配車中です。降車してからもう一度呼んでください"

        world = env.world
        # ★ 乗降地点は**道路ノード**へ寄せる（決定 7）。区間の途中に置くと、
        #   経路がそこを通り過ぎてから折り返す形になり、目的地の前で止まらない
        pick = world.snap_to_road_node(*pickup)
        drop = world.snap_to_road_node(*dropoff)
        if pick is None or drop is None:
            return "地図の道路に寄せられませんでした。道路に近い地点を選んでください"
        if math.hypot(drop[0] - pick[0], drop[1] - pick[1]) < MIN_TRIP_M:
            return f"乗車地点と降車地点が近すぎます（{MIN_TRIP_M:.0f}m 以上離してください）"

        slot, route = self._assign(world, pick)
        if slot < 0 or route is None:
            return "乗車地点まで来られる車両が見つかりませんでした"

        env.commandeer_vehicle(slot)
        if not world.install_route(slot, route, keep_pose=True):
            env.release_vehicle(slot)
            return "迎車の経路を設定できませんでした"
        world.set_stop_target(slot, float(world.route_total[slot]))

        self._speed_avg = float(world.fleet.speed[slot])
        self.status = TaxiStatus(
            phase=TAXI_PHASE_APPROACHING,
            vehicle_id=slot,
            pickup=pick,
            dropoff=drop,
            route=[(float(px), float(py)) for px, py in route],
            route_revision=self.status.route_revision + 1,
            message=f"車両 #{slot} が迎えに向かっています",
        )
        self._refresh_progress(env)
        return None

    def board(self, env: "SimulationEnv") -> str | None:
        """乗車する。降車地点への経路へ差し替える。"""
        if self.status.phase != TAXI_PHASE_WAITING:
            return "いまは乗車できません"

        slot = self.vehicle_id
        world = env.world
        dropoff = self.status.dropoff
        if dropoff is None or not world.fleet.active[slot]:
            self.cancel(env, "配車が失われました")
            return "配車が失われました"

        route = world.route_between(
            (float(world.fleet.x[slot]), float(world.fleet.y[slot])),
            dropoff,
            heading=float(world.fleet.heading[slot]),
        )
        if route is None or not world.install_route(slot, route, keep_pose=True):
            self.cancel(env, "降車地点までの経路を作れませんでした")
            return "降車地点までの経路を作れませんでした"
        world.set_stop_target(slot, float(world.route_total[slot]))

        self.status.phase = TAXI_PHASE_RIDING
        self.status.route = [(float(px), float(py)) for px, py in route]
        self.status.route_revision += 1
        self.status.message = "目的地へ向かっています"
        self._refresh_progress(env)
        return None

    def alight(self, env: "SimulationEnv") -> str | None:
        """降車する。徴用を解いて PPO の走行へ戻す（決定 3・15）。"""
        if self.status.phase not in (TAXI_PHASE_RIDING, TAXI_PHASE_ARRIVED):
            return "いまは降車できません"
        arrived = self.status.phase == TAXI_PHASE_ARRIVED
        self._release(env)
        self.status = TaxiStatus(
            route_revision=self.status.route_revision + 1,
            message="降車しました" if arrived else "目的地の手前で降車しました",
        )
        return None

    def cancel(self, env: "SimulationEnv | None", reason: str, *, halt: bool = False) -> None:
        """配車を打ち切る。`halt` なら車両をその場で止めてから解く（緊急停止）。"""
        if not self.busy:
            return
        slot = self.vehicle_id
        logger.info("配車を終了します（車両 #%d、%s）", slot, reason)
        if env is not None and halt and 0 <= slot < config.MAX_VEHICLES:
            env.world.fleet.speed[slot] = np.float32(0.0)
            env.world.fleet.steer[slot] = np.float32(0.0)
        self._release(env)
        self.status = TaxiStatus(
            route_revision=self.status.route_revision + 1, message=reason
        )

    def update(self, env: "SimulationEnv") -> None:
        """毎ステップ呼ぶ。到着判定と ETA の再計算（決定 14）。"""
        if not self.busy:
            return
        slot = self.vehicle_id
        world = env.world
        if not (0 <= slot < config.MAX_VEHICLES) or not world.fleet.active[slot]:
            self.cancel(env, "配車していた車両がいなくなりました")
            return
        if bool(world.collided_flags[slot]):
            self.cancel(env, "事故が起きたため配車を打ち切りました", halt=True)
            return

        self._refresh_progress(env)

        if self.status.phase == TAXI_PHASE_APPROACHING and self._has_arrived(world, slot):
            self._halt_here(world, slot)
            self.status.phase = TAXI_PHASE_WAITING
            self.status.message = "乗車地点に到着しました。[Enter] で乗車できます"
        elif self.status.phase == TAXI_PHASE_RIDING and self._has_arrived(world, slot):
            self._halt_here(world, slot)
            self.status.phase = TAXI_PHASE_ARRIVED
            self.status.message = "目的地に到着しました。[Enter] で降車できます"

    def _assign(
        self, world, pickup: tuple[float, float]
    ) -> tuple[int, np.ndarray | None]:
        """乗車地点に最も近く、かつそこへ来られる車両を選ぶ（決定 1）。"""
        fleet = world.fleet
        idx = np.flatnonzero(fleet.active)
        if idx.size == 0:
            return -1, None
        dx = fleet.x[idx].astype(np.float64) - pickup[0]
        dy = fleet.y[idx].astype(np.float64) - pickup[1]
        order = idx[np.argsort(dx * dx + dy * dy)]

        for slot in order[:ASSIGN_CANDIDATES]:
            slot = int(slot)
            route = world.route_between(
                (float(fleet.x[slot]), float(fleet.y[slot])),
                pickup,
                heading=float(fleet.heading[slot]),
            )
            if route is not None and route.shape[0] >= 2:
                return slot, route
        return -1, None

    def _release(self, env: "SimulationEnv | None") -> None:
        slot = self.vehicle_id
        if env is None or not (0 <= slot < config.MAX_VEHICLES):
            return
        env.release_vehicle(slot)

    @staticmethod
    def _halt_here(world, slot: int) -> None:
        """いまいる弧長を停車目標にして完全に止める（乗降中に車を動かさない）。"""
        world.set_stop_target(slot, float(world.arc[slot]))

    @staticmethod
    def _has_arrived(world, slot: int) -> bool:
        remaining = float(world.route_total[slot]) - float(world.arc[slot])
        return (
            remaining <= ARRIVE_DISTANCE_M
            and float(world.fleet.speed[slot]) <= ARRIVE_SPEED_MPS
        )

    def _refresh_progress(self, env: "SimulationEnv") -> None:
        """残り距離と ETA を更新する。**停車中でも ETA が無限にならないよう均す。**"""
        slot = self.vehicle_id
        world = env.world
        remaining = max(0.0, float(world.route_total[slot]) - float(world.arc[slot]))
        speed = float(world.fleet.speed[slot])
        self._speed_avg += (speed - self._speed_avg) * ETA_SPEED_SMOOTH

        self.status.remaining_distance_m = remaining
        if self.status.phase in (TAXI_PHASE_WAITING, TAXI_PHASE_ARRIVED):
            self.status.eta_seconds = 0.0
            return
        self.status.eta_seconds = remaining / max(self._speed_avg, ETA_MIN_SPEED_MPS)
