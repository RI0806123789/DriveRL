"""キネマティック自転車モデルによる車両群のベクトル化更新。

MAX_VEHICLES 台分の状態を numpy 配列で保持し、1 回の `step()` で全スロットを
まとめて更新する（memo 5章「擬似固定エージェント数方式」）。
非アクティブなスロットは一切更新しない。

座標系は docs/protocol.md 1章に従う:
    x = 東 [m] / y = 北 [m] / heading = +x 軸から反時計回りのラジアン。
車両外形は「前方が +X 軸」（docs/protocol.md 1.3）。
"""

from __future__ import annotations

import numpy as np

from app import config

__all__ = ["VehicleFleet", "wrap_angle"]


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    """角度を (-pi, pi] に正規化する。"""
    a = np.asarray(angle, dtype=np.float32)
    return np.arctan2(np.sin(a), np.cos(a)).astype(np.float32)


# 車両外形（自車座標系）の 4 隅。前方 +X / 左 +Y。
_HALF_LENGTH = float(config.VEHICLE_LENGTH) * 0.5
_HALF_WIDTH = float(config.VEHICLE_WIDTH) * 0.5
_LOCAL_CORNERS = np.array(
    [
        [+_HALF_LENGTH, +_HALF_WIDTH],  # 左前
        [+_HALF_LENGTH, -_HALF_WIDTH],  # 右前
        [-_HALF_LENGTH, -_HALF_WIDTH],  # 右後
        [-_HALF_LENGTH, +_HALF_WIDTH],  # 左後
    ],
    dtype=np.float32,
)


class VehicleFleet:
    """MAX_VEHICLES 台分の状態を numpy 配列で保持する。

    すべての配列は shape (MAX_VEHICLES,)。`active` のみ bool、他は float32。
    位置 (x, y) は車両の幾何中心として扱う（描画側のメッシュ原点と一致させるため）。
    キネマティック自転車モデル自体は後輪中心基準の式をそのまま用いる。
    """

    __slots__ = ("size", "x", "y", "heading", "speed", "steer", "active")

    def __init__(self, size: int = config.MAX_VEHICLES) -> None:
        self.size = int(size)
        self.x = np.zeros(self.size, dtype=np.float32)
        self.y = np.zeros(self.size, dtype=np.float32)
        self.heading = np.zeros(self.size, dtype=np.float32)
        self.speed = np.zeros(self.size, dtype=np.float32)
        self.steer = np.zeros(self.size, dtype=np.float32)
        self.active = np.zeros(self.size, dtype=bool)

    # ------------------------------------------------------------------
    # 物理更新
    # ------------------------------------------------------------------

    def step(
        self,
        accel_cmd: np.ndarray,
        steer_cmd: np.ndarray,
        dt: float,
        max_speed: float,
    ) -> None:
        """全スロットを 1 ステップ進める。非アクティブなスロットは変化しない。

        Args:
            accel_cmd: shape (MAX_VEHICLES,) の加減速指令。[-1, 1]。
            steer_cmd: shape (MAX_VEHICLES,) の操舵指令。[-1, 1]。
            dt: ステップ時間 [s]。
            max_speed: この時点での最高速度 [m/s]（SimParams.max_speed）。
        """
        accel_cmd = np.clip(
            np.asarray(accel_cmd, dtype=np.float32).reshape(self.size), -1.0, 1.0
        )
        steer_cmd = np.clip(
            np.asarray(steer_cmd, dtype=np.float32).reshape(self.size), -1.0, 1.0
        )
        dt = float(dt)
        limit_speed = max(float(max_speed), 0.0)
        active = self.active

        # --- 速度: 指令の符号で加速側／減速側の上限を切り替える ---
        # 舵角の上限が速度で決まるので、先に速度を確定させる。
        accel = np.where(
            accel_cmd >= 0.0,
            accel_cmd * np.float32(config.MAX_ACCEL),
            accel_cmd * np.float32(abs(config.MAX_DECEL)),
        ).astype(np.float32)
        new_speed = np.clip(self.speed + accel * dt, 0.0, limit_speed).astype(np.float32)

        # --- 舵角: 目標舵角へレート制限付きで追従させる（急舵を防ぐ） ---
        target_steer = steer_cmd * np.float32(config.MAX_STEER)
        max_delta = np.float32(config.STEER_RATE * dt)
        new_steer = self.steer + np.clip(target_steer - self.steer, -max_delta, max_delta)

        # --- 速度に応じた舵角の上限 ---
        # a_lat = v^2 * tan(steer) / L を MAX_LATERAL_ACCEL 以下に保つ。
        # 速度が上がるほど切れる角度が小さくなり、高速での急ハンドルを防ぐ。
        # 低速では atan の中身が大きくなって MAX_STEER に張り付く。
        steer_limit = np.arctan(
            np.float32(config.MAX_LATERAL_ACCEL * config.WHEELBASE)
            / np.maximum(new_speed * new_speed, np.float32(1e-3))
        ).astype(np.float32)
        steer_limit = np.minimum(steer_limit, np.float32(config.MAX_STEER))

        new_steer = np.clip(new_steer, -steer_limit, steer_limit).astype(np.float32)

        # --- 姿勢と位置（キネマティック自転車モデル） ---
        new_heading = self.heading + (new_speed / np.float32(config.WHEELBASE)) * np.tan(
            new_steer
        ) * np.float32(dt)
        new_heading = wrap_angle(new_heading)
        new_x = self.x + new_speed * np.cos(new_heading) * np.float32(dt)
        new_y = self.y + new_speed * np.sin(new_heading) * np.float32(dt)

        # --- 非アクティブスロットは据え置き ---
        self.steer = np.where(active, new_steer, self.steer).astype(np.float32)
        self.speed = np.where(active, new_speed, self.speed).astype(np.float32)
        self.heading = np.where(active, new_heading, self.heading).astype(np.float32)
        self.x = np.where(active, new_x, self.x).astype(np.float32)
        self.y = np.where(active, new_y, self.y).astype(np.float32)

    # ------------------------------------------------------------------
    # 形状
    # ------------------------------------------------------------------

    def corners(self) -> np.ndarray:
        """車両外形の 4 隅を返す。shape (MAX_VEHICLES, 4, 2) float32。"""
        cos_h = np.cos(self.heading)[:, None]  # (N, 1)
        sin_h = np.sin(self.heading)[:, None]
        lx = _LOCAL_CORNERS[None, :, 0]  # (1, 4)
        ly = _LOCAL_CORNERS[None, :, 1]
        out = np.empty((self.size, 4, 2), dtype=np.float32)
        out[:, :, 0] = self.x[:, None] + lx * cos_h - ly * sin_h
        out[:, :, 1] = self.y[:, None] + lx * sin_h + ly * cos_h
        return out

    # ------------------------------------------------------------------
    # スロット操作
    # ------------------------------------------------------------------

    def reset_slot(self, slot: int, x: float, y: float, heading: float) -> None:
        """1 スロットを指定姿勢で初期化する（速度・舵角はゼロ）。"""
        slot = int(slot)
        self.x[slot] = np.float32(x)
        self.y[slot] = np.float32(y)
        self.heading[slot] = wrap_angle(np.float32(heading))
        self.speed[slot] = np.float32(0.0)
        self.steer[slot] = np.float32(0.0)
