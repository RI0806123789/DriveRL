"""運動学または任意の動的自転車モデルで車両群を更新する。"""

from __future__ import annotations

import numpy as np

from app import config
from app.sim.dynamics import VehicleDynamics

__all__ = ["VehicleFleet", "wrap_angle"]


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    """角度を (-pi, pi] に正規化する。"""
    a = np.asarray(angle, dtype=np.float32)
    return np.arctan2(np.sin(a), np.cos(a)).astype(np.float32)


_HALF_LENGTH = float(config.VEHICLE_LENGTH) * 0.5
_HALF_WIDTH = float(config.VEHICLE_WIDTH) * 0.5
_LOCAL_CORNERS = np.array(
    [
        [+_HALF_LENGTH, +_HALF_WIDTH],
        [+_HALF_LENGTH, -_HALF_WIDTH],
        [-_HALF_LENGTH, -_HALF_WIDTH],
        [-_HALF_LENGTH, +_HALF_WIDTH],
    ],
    dtype=np.float32,
)


class VehicleFleet:
    """MAX_VEHICLES 台分の状態を numpy 配列で保持する。"""

    __slots__ = (
        "size", "x", "y", "heading", "speed", "steer", "active", "gear",
        "dynamics", "lateral_speed", "yaw_rate", "throttle", "brake",
        "longitudinal_accel", "lateral_accel",
    )

    def __init__(
        self, size: int = config.MAX_VEHICLES, *, dynamics: VehicleDynamics | None = None
    ) -> None:
        self.size = int(size)
        self.x = np.zeros(self.size, dtype=np.float32)
        self.y = np.zeros(self.size, dtype=np.float32)
        self.heading = np.zeros(self.size, dtype=np.float32)
        #: 符号つきの車体縦速度 [m/s]。詳細モデルの横滑りではギアと逆符号もあり得る
        self.speed = np.zeros(self.size, dtype=np.float32)
        self.steer = np.zeros(self.size, dtype=np.float32)
        self.active = np.zeros(self.size, dtype=bool)
        #: +1 = 前進（D）/ -1 = 後退（R）。切り替えは止まってから（`set_gear`）
        self.gear = np.ones(self.size, dtype=np.int8)
        self.dynamics = dynamics
        self.lateral_speed = np.zeros(self.size, dtype=np.float32)
        self.yaw_rate = np.zeros(self.size, dtype=np.float32)
        self.throttle = np.zeros(self.size, dtype=np.float32)
        self.brake = np.zeros(self.size, dtype=np.float32)
        self.longitudinal_accel = np.zeros(self.size, dtype=np.float32)
        self.lateral_accel = np.zeros(self.size, dtype=np.float32)

    def set_gear(self, slot: int, gear: int) -> bool:
        """ギアを切り替える。止まっていなければ切り替えず False を返す。"""
        slot = int(slot)
        want = np.int8(1 if gear >= 0 else -1)
        if self.gear[slot] == want:
            return True
        if abs(float(self.speed[slot])) > 1e-3:
            return False
        if self.dynamics is not None and (
            abs(float(self.lateral_speed[slot])) > 1e-3
            or abs(float(self.yaw_rate[slot])) > 1e-3
        ):
            return False
        self.gear[slot] = want
        self.speed[slot] = np.float32(0.0)
        if self.dynamics is not None:
            self._reset_dynamics(slot)
        return True

    def step(
        self,
        accel_cmd: np.ndarray,
        steer_cmd: np.ndarray,
        dt: float,
        max_speed: float,
    ) -> None:
        """全スロットを 1 ステップ進める。指令の正はギアの向きへの加速、負は 0 へ向けた制動。"""
        if self.dynamics is not None:
            self._step_dynamic(accel_cmd, steer_cmd, dt, max_speed)
            return
        accel_cmd = np.clip(
            np.asarray(accel_cmd, dtype=np.float32).reshape(self.size), -1.0, 1.0
        )
        steer_cmd = np.clip(
            np.asarray(steer_cmd, dtype=np.float32).reshape(self.size), -1.0, 1.0
        )
        dt = float(dt)
        limit_speed = max(float(max_speed), 0.0)
        active = self.active

        accel = np.where(
            accel_cmd >= 0.0,
            accel_cmd * np.float32(config.MAX_ACCEL),
            accel_cmd * np.float32(abs(config.MAX_DECEL)),
        ).astype(np.float32)
        # 速さの大きさで積分して、ギアの向きの符号を付け直す（制動は 0 で止まり、逆へは走らない）
        sign = self.gear.astype(np.float32)
        cap = np.where(
            self.gear > 0, np.float32(limit_speed), np.float32(config.REVERSE_MAX_SPEED)
        ).astype(np.float32)
        magnitude = np.clip(self.speed * sign + accel * dt, 0.0, cap).astype(np.float32)
        new_speed = (magnitude * sign).astype(np.float32)

        target_steer = steer_cmd * np.float32(config.MAX_STEER)
        max_delta = np.float32(config.STEER_RATE * dt)
        new_steer = self.steer + np.clip(target_steer - self.steer, -max_delta, max_delta)

        steer_limit = np.arctan(
            np.float32(config.MAX_LATERAL_ACCEL * config.WHEELBASE)
            / np.maximum(new_speed * new_speed, np.float32(1e-3))
        ).astype(np.float32)
        steer_limit = np.minimum(steer_limit, np.float32(config.MAX_STEER))

        new_steer = np.clip(new_steer, -steer_limit, steer_limit).astype(np.float32)

        new_heading = self.heading + (new_speed / np.float32(config.WHEELBASE)) * np.tan(
            new_steer
        ) * np.float32(dt)
        new_heading = wrap_angle(new_heading)
        new_x = self.x + new_speed * np.cos(new_heading) * np.float32(dt)
        new_y = self.y + new_speed * np.sin(new_heading) * np.float32(dt)

        self.steer = np.where(active, new_steer, self.steer).astype(np.float32)
        self.speed = np.where(active, new_speed, self.speed).astype(np.float32)
        self.heading = np.where(active, new_heading, self.heading).astype(np.float32)
        self.x = np.where(active, new_x, self.x).astype(np.float32)
        self.y = np.where(active, new_y, self.y).astype(np.float32)

    def corners(self) -> np.ndarray:
        """車両外形の 4 隅を返す。shape (MAX_VEHICLES, 4, 2) float32。"""
        cos_h = np.cos(self.heading)[:, None]
        sin_h = np.sin(self.heading)[:, None]
        lx = _LOCAL_CORNERS[None, :, 0]
        ly = _LOCAL_CORNERS[None, :, 1]
        out = np.empty((self.size, 4, 2), dtype=np.float32)
        out[:, :, 0] = self.x[:, None] + lx * cos_h - ly * sin_h
        out[:, :, 1] = self.y[:, None] + lx * sin_h + ly * cos_h
        return out

    def reset_slot(self, slot: int, x: float, y: float, heading: float) -> None:
        """1 スロットを指定姿勢で初期化する（速度・舵角はゼロ）。"""
        slot = int(slot)
        self.x[slot] = np.float32(x)
        self.y[slot] = np.float32(y)
        self.heading[slot] = wrap_angle(np.float32(heading))
        self.speed[slot] = np.float32(0.0)
        self.steer[slot] = np.float32(0.0)
        self.gear[slot] = np.int8(1)
        self._reset_dynamics(slot)

    def _reset_dynamics(self, slot: int) -> None:
        if self.dynamics is not None:
            self.steer[slot] = 0.0
        self.lateral_speed[slot] = 0.0
        self.yaw_rate[slot] = 0.0
        self.throttle[slot] = 0.0
        self.brake[slot] = 0.0
        self.longitudinal_accel[slot] = 0.0
        self.lateral_accel[slot] = 0.0

    def _step_dynamic(
        self, accel_cmd: np.ndarray, steer_cmd: np.ndarray, dt: float, max_speed: float
    ) -> None:
        parameters = self.dynamics
        assert parameters is not None
        if not np.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError("dt は 0 秒超・0.1 秒以下の有限数が必要です")
        if not np.isfinite(max_speed) or max_speed < 0:
            raise ValueError("max_speed は有限の非負数が必要です")
        accel_cmd = np.clip(np.asarray(accel_cmd, dtype=np.float32).reshape(self.size), -1, 1)
        steer_cmd = np.clip(np.asarray(steer_cmd, dtype=np.float32).reshape(self.size), -1, 1)
        if not np.all(np.isfinite(accel_cmd)) or not np.all(np.isfinite(steer_cmd)):
            raise ValueError("操作指令は有限の数値が必要です")
        indices = np.flatnonzero(self.active)
        if not indices.size:
            return
        count = int(np.ceil(dt / parameters.substep_s))
        h = float(dt) / count
        mass = parameters.mass_kg
        front = parameters.front_axle_m
        rear = config.WHEELBASE - front
        mu_g = parameters.friction * 9.81
        sign = self.gear[indices].astype(np.float32)
        cap = np.where(sign > 0, max_speed, config.REVERSE_MAX_SPEED)
        throttle_target = np.maximum(accel_cmd[indices], 0)
        brake_target = np.maximum(-accel_cmd[indices], 0)
        steer_target = steer_cmd[indices] * config.MAX_STEER
        throttle_gain = -np.expm1(-h / parameters.throttle_tau) if parameters.throttle_tau else 1.0
        brake_gain = -np.expm1(-h / parameters.brake_tau) if parameters.brake_tau else 1.0
        steer_gain = -np.expm1(-h / parameters.steering_tau) if parameters.steering_tau else 1.0
        for _ in range(count):
            self.throttle[indices] += throttle_gain * (throttle_target - self.throttle[indices])
            self.brake[indices] += brake_gain * (brake_target - self.brake[indices])
            steer = self.steer[indices] + np.clip(
                steer_gain * (steer_target - self.steer[indices]),
                -config.STEER_RATE * h, config.STEER_RATE * h,
            )
            vx = self.speed[indices]
            vy = self.lateral_speed[indices]
            yaw = self.yaw_rate[indices]
            braking = self.brake[indices] * np.minimum(abs(config.MAX_DECEL), np.abs(vx) / h)
            ax_request = np.clip(
                sign * self.throttle[indices] * config.MAX_ACCEL - np.sign(vx) * braking,
                -mu_g, mu_g,
            )
            front_load = np.clip(
                mass * (9.81 * rear - ax_request * parameters.cg_height_m) / config.WHEELBASE,
                mass * 9.81 * 0.05, mass * 9.81 * 0.95,
            )
            rear_load = mass * 9.81 - front_load
            fx_front = ax_request * front_load / 9.81
            fx_rear = ax_request * rear_load / 9.81
            front_budget = np.sqrt(np.maximum((parameters.friction * front_load) ** 2 - fx_front ** 2, 0))
            rear_budget = np.sqrt(np.maximum((parameters.friction * rear_load) ** 2 - fx_rear ** 2, 0))
            slip_speed = np.maximum(np.abs(vx), parameters.low_speed_m_s)
            slip_front = steer * vx / slip_speed - np.arctan2(vy + front * yaw, slip_speed)
            slip_rear = -np.arctan2(vy - rear * yaw, slip_speed)
            fy_front = front_budget * np.tanh(
                parameters.front_cornering_stiffness * slip_front / np.maximum(front_budget, 1e-6)
            )
            fy_rear = rear_budget * np.tanh(
                parameters.rear_cornering_stiffness * slip_rear / np.maximum(rear_budget, 1e-6)
            )
            front_longitudinal = fx_front * np.cos(steer) - fy_front * np.sin(steer)
            front_lateral = fy_front * np.cos(steer) + fx_front * np.sin(steer)
            ax_force = (front_longitudinal + fx_rear) / mass
            ay_force = (front_lateral + fy_rear) / mass
            yaw_dot = (front * front_lateral - rear * fy_rear) / parameters.yaw_inertia_kg_m2
            new_yaw = yaw + h * yaw_dot
            rotation = h * new_yaw
            cos_rotation, sin_rotation = np.cos(rotation), np.sin(rotation)
            integrated_vx = vx + h * ax_force
            integrated_vy = vy + h * ay_force
            magnitude = np.hypot(integrated_vx, integrated_vy)
            current_cap = np.maximum(cap, np.hypot(vx, vy))
            scale = np.minimum(1, current_cap / np.maximum(magnitude, 1e-12))
            integrated_vx *= scale
            integrated_vy *= scale
            new_vx = integrated_vx * cos_rotation + integrated_vy * sin_rotation
            new_vy = -integrated_vx * sin_rotation + integrated_vy * cos_rotation
            heading = wrap_angle(self.heading[indices] + rotation)
            self.x[indices] += h * (new_vx * np.cos(heading) - new_vy * np.sin(heading))
            self.y[indices] += h * (new_vx * np.sin(heading) + new_vy * np.cos(heading))
            self.heading[indices] = heading
            self.speed[indices] = new_vx
            self.steer[indices] = steer
            self.lateral_speed[indices] = new_vy
            self.yaw_rate[indices] = new_yaw
            self.longitudinal_accel[indices] = (integrated_vx - vx) / h
            self.lateral_accel[indices] = (integrated_vy - vy) / h
