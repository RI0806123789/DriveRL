"""任意で使う動的自転車モデルの車両設定。"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

from app import config


_BOUNDS = {
    "mass_kg": (500.0, 5000.0),
    "yaw_inertia_kg_m2": (200.0, 20000.0),
    "front_axle_m": (0.4, config.WHEELBASE - 0.4),
    "cg_height_m": (0.0, 2.0),
    "front_cornering_stiffness": (10000.0, 250000.0),
    "rear_cornering_stiffness": (10000.0, 250000.0),
    "friction": (0.05, 1.5),
    "throttle_tau": (0.0, 3.0),
    "brake_tau": (0.0, 3.0),
    "steering_tau": (0.0, 3.0),
    "substep_s": (0.001, 0.01),
    "low_speed_m_s": (0.5, 5.0),
}


@dataclass(frozen=True, slots=True)
class VehicleDynamics:
    """単位は kg・m・s・rad、剛性は N/rad とする。"""

    mass_kg: float = 1500.0
    yaw_inertia_kg_m2: float = 2500.0
    front_axle_m: float = 1.2
    cg_height_m: float = 0.55
    front_cornering_stiffness: float = 65000.0
    rear_cornering_stiffness: float = 65000.0
    friction: float = 0.9
    throttle_tau: float = 0.25
    brake_tau: float = 0.12
    steering_tau: float = 0.15
    substep_s: float = 0.01
    low_speed_m_s: float = 3.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field.name} は有限の数値が必要です")
            minimum, maximum = _BOUNDS[field.name]
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{field.name} は {minimum}〜{maximum} の有限数が必要です")
