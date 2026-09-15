"""GAE 付きロールアウトバッファ（エージェントスロット次元あり）。"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["RolloutBuffer"]


class RolloutBuffer:
    """1 回の PPO 更新分の遷移を貯める固定長バッファ。"""

    def __init__(self, capacity: int, num_agents: int, obs_dim: int, action_dim: int) -> None:
        self.capacity = max(int(capacity), 1)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        shape = (self.capacity, self.num_agents)
        self.obs = np.zeros((*shape, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((*shape, self.action_dim), dtype=np.float32)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.dones = np.zeros(shape, dtype=bool)
        self.truncated = np.zeros(shape, dtype=bool)
        self.active = np.zeros(shape, dtype=bool)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)

        self.ptr = 0
        self._ready = False

    @property
    def full(self) -> bool:
        return self.ptr >= self.capacity

    @property
    def size(self) -> int:
        return self.ptr

    def clear(self) -> None:
        self.ptr = 0
        self._ready = False

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        active: np.ndarray,
        truncated: np.ndarray | None = None,
    ) -> None:
        """1 ステップ分を追加する。満杯なら何もしない（実行ループを止めないため）。"""
        if self.full:
            return
        i = self.ptr
        n = self.num_agents
        self.obs[i] = np.asarray(obs, dtype=np.float32).reshape(n, self.obs_dim)
        self.actions[i] = np.asarray(actions, dtype=np.float32).reshape(n, self.action_dim)
        self.log_probs[i] = np.asarray(log_probs, dtype=np.float32).reshape(n)
        self.values[i] = np.asarray(values, dtype=np.float32).reshape(n)
        self.rewards[i] = np.asarray(rewards, dtype=np.float32).reshape(n)
        self.dones[i] = np.asarray(dones, dtype=bool).reshape(n)
        self.truncated[i] = (
            np.zeros(n, dtype=bool)
            if truncated is None
            else np.asarray(truncated, dtype=bool).reshape(n)
        )
        self.active[i] = np.asarray(active, dtype=bool).reshape(n)
        self.ptr = i + 1

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        last_active: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """スロットごとに独立に GAE を時間方向へ逆順計算する。"""
        size = self.ptr
        if size == 0:
            self._ready = False
            return

        gamma = np.float32(gamma)
        lam = np.float32(gae_lambda)
        n = self.num_agents

        adv = np.zeros(n, dtype=np.float32)
        next_values = np.asarray(last_values, dtype=np.float32).reshape(n)
        next_active = np.asarray(last_active, dtype=bool).reshape(n)

        for t in range(size - 1, -1, -1):
            non_terminal = ((~self.dones[t]) & next_active).astype(np.float32)
            bootstrap = (
                ((~self.dones[t]) | self.truncated[t]) & next_active
            ).astype(np.float32)
            delta = self.rewards[t] + gamma * next_values * bootstrap - self.values[t]
            adv = delta + gamma * lam * non_terminal * adv
            adv = np.where(self.active[t], adv, np.float32(0.0)).astype(np.float32)
            self.advantages[t] = adv
            next_values = self.values[t]
            next_active = self.active[t]

        self.returns[:size] = self.advantages[:size] + self.values[:size]
        self._ready = True

    def flat_dataset(self) -> dict[str, torch.Tensor] | None:
        """active=True のサンプルだけを平坦化した学習データを 1 つ返す。"""
        size = self.ptr
        if size == 0 or not self._ready:
            return None

        mask = self.active[:size].reshape(-1)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return None

        obs = self.obs[:size].reshape(-1, self.obs_dim)[idx]
        actions = self.actions[:size].reshape(-1, self.action_dim)[idx]
        log_probs = self.log_probs[:size].reshape(-1)[idx]
        values = self.values[:size].reshape(-1)[idx]
        returns = self.returns[:size].reshape(-1)[idx]
        advantages = self.advantages[:size].reshape(-1)[idx]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "obs": torch.from_numpy(np.ascontiguousarray(obs)),
            "actions": torch.from_numpy(np.ascontiguousarray(actions)),
            "log_probs": torch.from_numpy(np.ascontiguousarray(log_probs)),
            "values": torch.from_numpy(np.ascontiguousarray(values)),
            "returns": torch.from_numpy(np.ascontiguousarray(returns)),
            "advantages": torch.from_numpy(
                np.ascontiguousarray(advantages.astype(np.float32))
            ),
        }

    @staticmethod
    def minibatch_indices(
        sample_count: int, num_minibatches: int, rng: np.random.Generator
    ) -> list[np.ndarray]:
        """1 エポック分のミニバッチ添字を、重複なしのシャッフルで作る。"""
        if sample_count <= 0:
            return []
        count = max(1, min(int(num_minibatches), int(sample_count)))
        perm = rng.permutation(sample_count)
        return [c for c in np.array_split(perm, count) if c.size > 0]
