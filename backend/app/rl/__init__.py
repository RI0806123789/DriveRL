"""強化学習パッケージ（共有ポリシー・PPO）。

外部（runtime）からはこのモジュールの re-export だけを使う想定::

    from app.rl import PPOTrainer
"""

from __future__ import annotations

from app.rl.buffer import RolloutBuffer
from app.rl.policy import ActorCritic
from app.rl.ppo import CHECKPOINT_FORMAT, PPOTrainer

__all__ = [
    "PPOTrainer",
    "ActorCritic",
    "RolloutBuffer",
    "CHECKPOINT_FORMAT",
]
