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
        self.truncated_values = np.zeros(shape, dtype=np.float32)
        self.active = np.zeros(shape, dtype=bool)
        self.learn = np.zeros(shape, dtype=bool)
        self.expert_actions = np.zeros((*shape, self.action_dim), dtype=np.float32)
        self.assisted = np.zeros(shape, dtype=bool)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        # 階層型の方策。`rewards` は上位（環境の報酬そのもの）、`sub_rewards` は下位（整形を足したもの）
        self.sub_rewards = np.zeros(shape, dtype=np.float32)
        self.options = np.zeros(shape, dtype=np.int64)
        self.option_start = np.zeros(shape, dtype=bool)
        self.meta_log_probs = np.zeros(shape, dtype=np.float32)
        self.meta_values = np.zeros(shape, dtype=np.float32)
        self.truncated_meta_values = np.zeros(shape, dtype=np.float32)
        self.expert_options = np.full(shape, -1, dtype=np.int64)
        self.meta_advantages = np.zeros(shape, dtype=np.float32)
        self.meta_returns = np.zeros(shape, dtype=np.float32)

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
        truncated_values: np.ndarray | None = None,
        learn: np.ndarray | None = None,
        expert_actions: np.ndarray | None = None,
        assisted: np.ndarray | None = None,
        *,
        sub_rewards: np.ndarray | None = None,
        options: np.ndarray | None = None,
        option_start: np.ndarray | None = None,
        meta_log_probs: np.ndarray | None = None,
        meta_values: np.ndarray | None = None,
        truncated_meta_values: np.ndarray | None = None,
        expert_options: np.ndarray | None = None,
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
        self.truncated_values[i] = (
            np.zeros(n, dtype=np.float32)
            if truncated_values is None
            else np.asarray(truncated_values, dtype=np.float32).reshape(n)
        )
        self.active[i] = np.asarray(active, dtype=bool).reshape(n)
        # 生きていても、方策が出していない操作（安全ギミックの引き受け）は損失に入れない
        self.learn[i] = (
            self.active[i]
            if learn is None
            else self.active[i] & np.asarray(learn, dtype=bool).reshape(n)
        )
        # エキスパートが運転したステップ。方策の勾配には入れず、価値と模倣の損失にだけ使う
        if assisted is None or expert_actions is None:
            self.assisted[i] = False
            self.expert_actions[i] = 0.0
        else:
            self.assisted[i] = self.learn[i] & np.asarray(assisted, dtype=bool).reshape(n)
            self.expert_actions[i] = np.asarray(expert_actions, dtype=np.float32).reshape(
                n, self.action_dim
            )

        def fill(dest: np.ndarray, value: np.ndarray | None, default: object, dtype: type) -> None:
            if value is None:
                dest[i] = default
            else:
                dest[i] = np.asarray(value, dtype=dtype).reshape(n)

        fill(self.sub_rewards, sub_rewards, self.rewards[i], np.float32)
        fill(self.options, options, 0, np.int64)
        fill(self.option_start, option_start, False, bool)
        self.option_start[i] &= self.active[i]
        fill(self.meta_log_probs, meta_log_probs, 0.0, np.float32)
        fill(self.meta_values, meta_values, 0.0, np.float32)
        fill(self.truncated_meta_values, truncated_meta_values, 0.0, np.float32)
        fill(self.expert_options, expert_options, -1, np.int64)
        self.expert_options[i] = np.where(self.assisted[i], self.expert_options[i], -1)
        self.ptr = i + 1

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        last_active: np.ndarray,
        gamma: float,
        gae_lambda: float,
        last_meta_values: np.ndarray | None = None,
    ) -> None:
        """スロットごとに独立に GAE を時間方向へ逆順計算する（下位は毎ステップ、上位は意図ごと）。"""
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
            # 続いているかは `active`（生きているか）で見る。学習から外したステップ
            # （安全ギミックの引き受け）の手前は、次の状態の価値で補う。連鎖は外した
            # ステップの `adv` が 0 なのでそこで切れる（打ち切りと同じ形）
            continues = (~self.dones[t]) & next_active
            non_terminal = continues.astype(np.float32)
            # 打ち切りは「続いていたはずの先」の価値で補う。ただし `next_values` は
            # 再スポーン後（別のエピソード）の価値なので、打ち切った時点で評価した
            # 値に差し替える
            bootstrap = (continues | self.truncated[t]).astype(np.float32)
            boot_values = np.where(
                self.truncated[t], self.truncated_values[t], next_values
            ).astype(np.float32)
            delta = self.sub_rewards[t] + gamma * boot_values * bootstrap - self.values[t]
            adv = delta + gamma * lam * non_terminal * adv
            adv = np.where(self.learn[t], adv, np.float32(0.0)).astype(np.float32)
            self.advantages[t] = adv
            next_values = self.values[t]
            next_active = self.active[t]

        self.returns[:size] = self.advantages[:size] + self.values[:size]
        self._compute_meta(
            np.zeros(n, dtype=np.float32) if last_meta_values is None else last_meta_values,
            np.asarray(last_active, dtype=bool).reshape(n),
            gamma,
            lam,
        )
        self._ready = True

    def _compute_meta(
        self, last_meta_values: np.ndarray, last_active: np.ndarray, gamma: np.float32, lam: np.float32
    ) -> None:
        """上位方策の GAE。意図 1 つ（選んでから次に選ぶまで）を 1 手として、区間の割引報酬で数える。"""
        size = self.ptr
        n = self.num_agents
        last_v = np.asarray(last_meta_values, dtype=np.float32).reshape(n)
        seg_return = np.zeros(n, dtype=np.float32)
        discount = np.ones(n, dtype=np.float32)
        pending = np.zeros(n, dtype=np.float32)
        self.meta_advantages[:size] = 0.0
        self.meta_returns[:size] = 0.0
        for t in range(size - 1, -1, -1):
            if t == size - 1:
                next_active = last_active
                next_values = last_v
                next_start = np.zeros(n, dtype=bool)
                next_adv = np.zeros(n, dtype=np.float32)
            else:
                next_active = self.active[t + 1]
                next_values = self.meta_values[t + 1]
                next_start = self.option_start[t + 1]
                next_adv = self.meta_advantages[t + 1]
            done = self.dones[t]
            ends = done | ~next_active | next_start | (t == size - 1)
            # 打ち切りは打ち切った時点の上位の価値で補う（再スポーン後の価値は別のエピソード）
            boundary = np.where(
                done,
                np.where(self.truncated[t], self.truncated_meta_values[t], np.float32(0.0)),
                np.where(next_active, next_values, np.float32(0.0)),
            ).astype(np.float32)
            chained = next_start & next_active & ~done
            reward = self.rewards[t]
            seg_return = np.where(
                ends, reward + gamma * boundary, reward + gamma * seg_return
            ).astype(np.float32)
            discount = np.where(ends, gamma, discount * gamma).astype(np.float32)
            pending = np.where(ends, np.where(chained, next_adv, np.float32(0.0)), pending).astype(
                np.float32
            )
            start = self.option_start[t]
            adv = seg_return - self.meta_values[t] + lam * discount * pending
            self.meta_advantages[t] = np.where(start, adv, np.float32(0.0))
            self.meta_returns[t] = np.where(start, adv + self.meta_values[t], np.float32(0.0))

    def flat_dataset(self) -> dict[str, torch.Tensor] | None:
        """学習に使うサンプル（`learn`）だけを平坦化した学習データを 1 つ返す。"""
        size = self.ptr
        if size == 0 or not self._ready:
            return None

        mask = self.learn[:size].reshape(-1)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return None

        obs = self.obs[:size].reshape(-1, self.obs_dim)[idx]
        actions = self.actions[:size].reshape(-1, self.action_dim)[idx]
        log_probs = self.log_probs[:size].reshape(-1)[idx]
        values = self.values[:size].reshape(-1)[idx]
        returns = self.returns[:size].reshape(-1)[idx]
        advantages = self.advantages[:size].reshape(-1)[idx]
        assisted = self.assisted[:size].reshape(-1)[idx]
        expert_actions = self.expert_actions[:size].reshape(-1, self.action_dim)[idx]
        options = self.options[:size].reshape(-1)[idx]
        option_start = self.option_start[:size].reshape(-1)[idx]
        meta_log_probs = self.meta_log_probs[:size].reshape(-1)[idx]
        meta_values = self.meta_values[:size].reshape(-1)[idx]
        meta_returns = self.meta_returns[:size].reshape(-1)[idx]
        meta_advantages = self.meta_advantages[:size].reshape(-1)[idx]
        expert_options = self.expert_options[:size].reshape(-1)[idx]

        # 正規化は方策の勾配に入れるサンプル（エキスパートが運転していない）だけで行う
        policy = ~assisted
        if policy.any():
            ref = advantages[policy]
            advantages = (advantages - ref.mean()) / (ref.std() + 1e-8)
        else:
            advantages = np.zeros_like(advantages)
        # 上位も同じ。エキスパートが運転し始めたときに選んだ意図は、上位方策の勾配に入れない
        meta_policy = option_start & ~assisted
        if meta_policy.any():
            ref = meta_advantages[meta_policy]
            meta_advantages = np.where(
                meta_policy, (meta_advantages - ref.mean()) / (ref.std() + 1e-8), 0.0
            )
        else:
            meta_advantages = np.zeros_like(meta_advantages)

        def tensor(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(a))

        return {
            "obs": torch.from_numpy(np.ascontiguousarray(obs)),
            "actions": torch.from_numpy(np.ascontiguousarray(actions)),
            "log_probs": torch.from_numpy(np.ascontiguousarray(log_probs)),
            "values": torch.from_numpy(np.ascontiguousarray(values)),
            "returns": torch.from_numpy(np.ascontiguousarray(returns)),
            "advantages": torch.from_numpy(
                np.ascontiguousarray(advantages.astype(np.float32))
            ),
            "assisted": torch.from_numpy(np.ascontiguousarray(assisted)),
            "expert_actions": torch.from_numpy(np.ascontiguousarray(expert_actions)),
            "options": tensor(options),
            "option_start": tensor(option_start),
            "meta_log_probs": tensor(meta_log_probs),
            "meta_values": tensor(meta_values),
            "meta_returns": tensor(meta_returns),
            "meta_advantages": tensor(meta_advantages.astype(np.float32)),
            "expert_options": tensor(expert_options),
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
