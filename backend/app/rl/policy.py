"""共有 Actor-Critic。いまは階層型（`rl/hierarchical_policy.py`）そのもの。"""

from __future__ import annotations

from app.rl.hierarchical_policy import HierarchicalActorCritic

ActorCritic = HierarchicalActorCritic

__all__ = ["ActorCritic"]
