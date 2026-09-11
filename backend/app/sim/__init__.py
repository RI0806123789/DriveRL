"""シミュレーションコア（物理・世界・MARL 環境）。

外部（runtime / rl）からはこのモジュールの re-export だけを使う想定::

    from app.sim import SimulationEnv
"""

from __future__ import annotations

from app.sim.env import SimulationEnv
from app.sim.vehicle import VehicleFleet, wrap_angle
from app.sim.world import ObstacleState, SlotState, World

__all__ = [
    "SimulationEnv",
    "VehicleFleet",
    "wrap_angle",
    "World",
    "ObstacleState",
    "SlotState",
]
