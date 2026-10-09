"""PPO 学習器（CPU・共有ポリシー・オンライン学習を常時継続）。"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from app import config
from app.contracts import DriveState, SimParams
from app.rl.buffer import RolloutBuffer
from app.rl.checkpoint import (
    CHECKPOINT_FORMAT, FLAT_CHECKPOINT_FORMATS, KNOWN_CHECKPOINT_FORMATS,
    is_flat_state, prepare_checkpoint, upgrade_flat_state, widen_observation,
)
from app.rl.hierarchical_policy import (
    NUM_OPTIONS,
    OptionScheduler,
    sub_reward_shaping,
)
from app.rl.policy import ActorCritic

__all__ = [
    "PPOTrainer",
    "CHECKPOINT_FORMAT",
    "KNOWN_CHECKPOINT_FORMATS",
    "peek_hidden_sizes",
]

logger = logging.getLogger(__name__)

def peek_hidden_sizes(path: Path) -> tuple[int, ...] | None:
    """チェックポイントに保存された隠れ層構成だけを覗き見る。"""
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        sizes = tuple(int(h) for h in payload["hidden_sizes"])
    except Exception:
        return None
    return sizes if sizes else None


@dataclass
class _PendingUpdate:
    """ステップ境界に分散して実行している途中の PPO 更新。"""

    data: dict[str, torch.Tensor]
    schedule: list[np.ndarray]
    cursor: int = 0
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    kls: list[float] = field(default_factory=list)
    bc_losses: list[float] = field(default_factory=list)
    meta_policy_losses: list[float] = field(default_factory=list)
    meta_value_losses: list[float] = field(default_factory=list)
    meta_entropies: list[float] = field(default_factory=list)
    grad_sums: dict[str, float] = field(default_factory=dict)
    grad_counts: dict[str, int] = field(default_factory=dict)
    grad_totals: list[float] = field(default_factory=list)
    grad_clipped: int = 0


class PPOTrainer:
    """ロールアウト収集と PPO 更新を担う。"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        params: SimParams,
        num_agents: int,
        seed: int = 0,
        hidden_sizes: Sequence[int] | None = None,
    ) -> None:
        torch.set_num_threads(int(config.TORCH_NUM_THREADS))
        torch.manual_seed(int(seed))

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.num_agents = int(num_agents)
        self.device = torch.device("cpu")
        self._seed = int(seed)
        self._rng = np.random.default_rng(int(seed))

        self.learning_rate = float(params.learning_rate)
        self.gamma = float(params.gamma)
        self.clip_range = float(params.clip_range)
        self.entropy_coef = float(params.entropy_coef)
        self.rollout_length = max(int(params.rollout_length), 1)

        source = hidden_sizes if hidden_sizes is not None else config.PPO_HIDDEN_SIZES
        self.hidden_sizes: tuple[int, ...] = tuple(int(h) for h in source)
        self.policy = ActorCritic(
            self.obs_dim, self.action_dim, self.hidden_sizes
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.learning_rate, eps=1e-5
        )
        self.buffer = RolloutBuffer(
            self.rollout_length, self.num_agents, self.obs_dim, self.action_dim
        )

        self._updates = 0
        self._last_raw_actions: np.ndarray | None = None
        self.scheduler = OptionScheduler(self.num_agents, int(config.HRL_OPTION_STEPS))
        self._last_meta: dict[str, np.ndarray] | None = None
        self._pending: _PendingUpdate | None = None

        self._last_grad_norms: dict[str, float] = {}
        self._last_grad_total: float = 0.0
        self._last_grad_clip_rate: float = 0.0
        self._weights_before_update: dict[str, torch.Tensor] = {}
        self._last_delta_norms: dict[str, float] = {}
        #: 直前の `load` が観測の次元を広げて読み込んだなら、元の次元
        self.widened_from: int | None = None
        #: 直前の `load` が階層型にする前（平らな方策）の重みを移して読み込んだか
        self.upgraded_flat: bool = False

    @property
    def updates(self) -> int:
        return self._updates

    @property
    def experience_steps(self) -> int:
        """これまでに学習へ積んだステップ数の目安（更新回数 × ロールアウト長 + 収集中の分）。"""
        return int(self._updates) * int(self.rollout_length) + int(self.buffer.size)

    @property
    def current_options(self) -> np.ndarray:
        """スロットごとにいま保っている意図（`config.HRL_OPTIONS` の添字）。"""
        return self.scheduler.options.copy()

    def end_options(self, mask: np.ndarray) -> None:
        """エピソードが終わった車は次のステップで意図を選び直す。"""
        self.scheduler.restart(mask)

    def act(
        self, obs: np.ndarray, active: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """行動をサンプリングする。意図は `HRL_OPTION_STEPS` ごと（とエピソードの頭）にだけ選び直す。"""
        n = self.num_agents
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(n, self.obs_dim)
        active_arr = np.asarray(active, dtype=bool).reshape(n)

        with torch.no_grad():
            t_obs = torch.from_numpy(np.ascontiguousarray(obs_arr))
            due = self.scheduler.due(active_arr)
            # 上位の対数確率と価値が要るのは選び直したステップだけ（上位の GAE の区切りはそこにしか来ない）
            meta_log_probs = np.zeros(n, dtype=np.float32)
            meta_values = np.zeros(n, dtype=np.float32)
            if due.any():
                meta_dist, t_meta_values = self.policy.meta(t_obs)
                sampled = meta_dist.sample()
                self.scheduler.assign(due, sampled.numpy())
                meta_log_probs = meta_dist.log_prob(sampled).numpy().astype(np.float32)
                meta_values = t_meta_values.numpy().astype(np.float32)
            options = self.scheduler.options.copy()
            raw_actions, log_probs, values = self.policy.get_action(t_obs, torch.from_numpy(options))
        self._last_meta = {
            "options": options,
            "option_start": due,
            "meta_log_probs": np.where(due, meta_log_probs, np.float32(0.0)),
            "meta_values": np.where(due, meta_values, np.float32(0.0)),
        }
        self.scheduler.tick(active_arr)

        raw = raw_actions.numpy().astype(np.float32, copy=True)
        self._last_raw_actions = raw

        actions = np.clip(raw, -1.0, 1.0).astype(np.float32)
        actions[~active_arr] = 0.0
        return (
            actions,
            log_probs.numpy().astype(np.float32, copy=True),
            values.numpy().astype(np.float32, copy=True),
        )

    def store(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        active: np.ndarray,
        truncated: np.ndarray | None = None,
        final_obs: np.ndarray | None = None,
        learn: np.ndarray | None = None,
        expert_actions: np.ndarray | None = None,
        assisted: np.ndarray | None = None,
        expert_options: np.ndarray | None = None,
        drive: DriveState | None = None,
    ) -> None:
        """1 ステップ分をバッファに積む。下位方策の報酬は `drive` から整形を足したもの。"""
        raw = self._last_raw_actions
        actions_arr = np.asarray(actions, dtype=np.float32).reshape(
            self.num_agents, self.action_dim
        )
        if raw is not None and raw.shape == actions_arr.shape:
            actions_arr = raw
        self._last_raw_actions = None
        n = self.num_agents
        meta = self._last_meta or {
            "options": np.zeros(n, dtype=np.int64),
            "option_start": np.zeros(n, dtype=bool),
            "meta_log_probs": np.zeros(n, dtype=np.float32),
            "meta_values": np.zeros(n, dtype=np.float32),
        }
        self._last_meta = None
        env_rewards = np.asarray(rewards, dtype=np.float32).reshape(n)
        sub_rewards = env_rewards
        if drive is not None:
            sub_rewards = env_rewards + sub_reward_shaping(meta["options"], drive, active)

        truncated_values: np.ndarray | None = None
        truncated_meta_values: np.ndarray | None = None
        if final_obs is not None and truncated is not None:
            # 打ち切ったステップでしか作られない（`SimulationEnv.step`）ので、
            # 評価は 200 秒に 1 度ほど。毎ステップの推論は増えない
            with torch.no_grad():
                t_obs = torch.from_numpy(
                    np.ascontiguousarray(
                        np.asarray(final_obs, dtype=np.float32).reshape(
                            self.num_agents, self.obs_dim
                        )
                    )
                )
                final_values = self.policy.sub_value(t_obs, torch.from_numpy(meta["options"]))
                _meta_dist, final_meta_values = self.policy.meta(t_obs)
            truncated_values = final_values.numpy().astype(np.float32)
            truncated_meta_values = final_meta_values.numpy().astype(np.float32)

        self.buffer.add(
            obs,
            actions_arr,
            log_probs,
            values,
            rewards,
            dones,
            active,
            truncated,
            truncated_values,
            learn,
            expert_actions,
            assisted,
            sub_rewards=sub_rewards,
            options=meta["options"],
            option_start=meta["option_start"],
            meta_log_probs=meta["meta_log_probs"],
            meta_values=meta["meta_values"],
            truncated_meta_values=truncated_meta_values,
            expert_options=expert_options,
        )
        self.end_options(np.asarray(dones, dtype=bool).reshape(n))

    def maybe_update(
        self, last_obs: np.ndarray, last_active: np.ndarray
    ) -> dict[str, float] | None:
        """PPO 更新を進める。1 回ぶんが完了したときだけ統計を返す。"""
        if self._pending is None:
            if not self.buffer.full:
                return None
            self._begin_update(last_obs, last_active)
            if self._pending is None:
                return None

        budget = int(config.PPO_MINIBATCHES_PER_STEP)
        if budget <= 0:
            budget = len(self._pending.schedule)
        return self._run_pending(budget)

    def _begin_update(self, last_obs: np.ndarray, last_active: np.ndarray) -> None:
        """GAE を計算し、学習データと勾配更新の予定表を作ってバッファを解放する。"""
        n = self.num_agents
        last_obs_arr = np.asarray(last_obs, dtype=np.float32).reshape(n, self.obs_dim)
        last_active_arr = np.asarray(last_active, dtype=bool).reshape(n)

        with torch.no_grad():
            t_obs = torch.from_numpy(np.ascontiguousarray(last_obs_arr))
            last_values = self.policy.sub_value(t_obs, torch.from_numpy(self.scheduler.options))
            _meta_dist, last_meta_values = self.policy.meta(t_obs)
        last_values_np = last_values.numpy().astype(np.float32)

        self.buffer.compute_returns_and_advantages(
            last_values_np,
            last_active_arr,
            self.gamma,
            config.PPO_GAE_LAMBDA,
            last_meta_values=last_meta_values.numpy().astype(np.float32),
        )

        with torch.no_grad():
            self._weights_before_update = {
                name: prm.detach().clone()
                for name, prm in self.policy.named_parameters()
                if name.endswith(".weight")
            }

        data = self.buffer.flat_dataset()
        self.buffer.clear()
        if data is None:
            self._pending = None
            return

        total = int(data["obs"].shape[0])
        schedule: list[np.ndarray] = []
        for _epoch in range(int(config.PPO_EPOCHS)):
            schedule.extend(
                self.buffer.minibatch_indices(
                    total, int(config.PPO_MINIBATCHES), self._rng
                )
            )
        if not schedule:
            self._pending = None
            return

        self._pending = _PendingUpdate(data=data, schedule=schedule)

    def _run_pending(self, budget: int) -> dict[str, float] | None:
        """予定表のミニバッチを最大 budget 個ぶん進める。終わったら統計を返す。"""
        pending = self._pending
        assert pending is not None

        clip = float(self.clip_range)
        data = pending.data
        end = min(pending.cursor + max(1, int(budget)), len(pending.schedule))

        for i in range(pending.cursor, end):
            chunk = pending.schedule[i]
            sel = torch.from_numpy(np.ascontiguousarray(chunk.astype(np.int64)))
            batch = {k: v[sel] for k, v in data.items()}

            new_log_probs, entropy, values = self.policy.evaluate(
                batch["obs"], batch["actions"], batch["options"]
            )
            log_ratio = new_log_probs - batch["log_probs"]
            ratio = torch.exp(log_ratio)

            adv = batch["advantages"]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
            surrogate = torch.min(surr1, surr2)
            # エキスパートが運転したステップの行動は方策から引いたものではないので、方策の勾配に入れない
            assisted = batch["assisted"]
            own = ~assisted
            own_count = int(own.sum())
            if own_count == int(own.numel()):
                policy_loss = -surrogate.mean()
            elif own_count > 0:
                policy_loss = -surrogate[own].mean()
            else:
                policy_loss = surrogate.sum() * 0.0

            old_values = batch["values"]
            returns = batch["returns"]
            v_clipped = old_values + torch.clamp(values - old_values, -clip, clip)
            v_loss_unclipped = (values - returns) ** 2
            v_loss_clipped = (v_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_mean = entropy.mean()
            loss = (
                policy_loss
                + float(config.PPO_VALUE_COEF) * value_loss
                - float(self.entropy_coef) * entropy_mean
            )
            meta_policy_loss, meta_value_loss, meta_entropy, meta_bc = self._meta_losses(batch, clip)
            loss = (
                loss
                + meta_policy_loss
                + float(config.PPO_VALUE_COEF) * meta_value_loss
                - float(config.HRL_META_ENTROPY_COEF) * meta_entropy
            )
            if bool(assisted.any()):
                # 下位はエキスパートの操作を、そのときの状況に合う意図（`expert_options`）の下で真似る
                labels = batch["expert_options"][assisted]
                chosen = batch["options"][assisted]
                mean_action = self.policy.mean_action(
                    batch["obs"][assisted], torch.where(labels >= 0, labels, chosen)
                )
                target = torch.clamp(
                    batch["expert_actions"][assisted],
                    -float(config.PPO_BC_TEACH_LIMIT),
                    float(config.PPO_BC_TEACH_LIMIT),
                )
                bc_loss = ((mean_action - target) ** 2).sum(dim=-1).mean() + meta_bc
                loss = loss + float(config.PPO_BC_COEF) * bc_loss
                pending.bc_losses.append(float(bc_loss.detach()))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                for name, prm in self.policy.named_parameters():
                    if prm.grad is None:
                        continue
                    g = float(prm.grad.norm())
                    pending.grad_sums[name] = pending.grad_sums.get(name, 0.0) + g
                    pending.grad_counts[name] = pending.grad_counts.get(name, 0) + 1
            max_grad_norm = float(config.PPO_MAX_GRAD_NORM)
            total_norm = float(
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
            )
            pending.grad_totals.append(total_norm)
            if total_norm > max_grad_norm:
                pending.grad_clipped += 1
            self.optimizer.step()
            self.policy.clamp_log_std()

            with torch.no_grad():
                kl_terms = (ratio - 1.0) - log_ratio
                approx_kl = kl_terms[own].mean() if own_count > 0 else kl_terms.sum() * 0.0
            pending.policy_losses.append(float(policy_loss.detach()))
            pending.value_losses.append(float(value_loss.detach()))
            pending.entropies.append(float(entropy_mean.detach()))
            pending.kls.append(float(approx_kl))
            pending.meta_policy_losses.append(float(meta_policy_loss.detach()))
            pending.meta_value_losses.append(float(meta_value_loss.detach()))
            pending.meta_entropies.append(float(meta_entropy.detach()))

        pending.cursor = end
        if pending.cursor < len(pending.schedule):
            return None

        self._pending = None
        self._updates += 1
        self._last_grad_norms = {
            k: pending.grad_sums[k] / max(1, pending.grad_counts.get(k, 1))
            for k in pending.grad_sums
        }
        batches = len(pending.grad_totals)
        self._last_grad_total = (
            float(np.mean(pending.grad_totals)) if batches else 0.0
        )
        self._last_grad_clip_rate = pending.grad_clipped / max(1, batches)
        with torch.no_grad():
            self._last_delta_norms = {
                name: float((prm.detach() - before).norm())
                for name, prm in self.policy.named_parameters()
                if (before := self._weights_before_update.get(name)) is not None
            }
        return {
            "policy_loss": float(np.mean(pending.policy_losses)),
            "value_loss": float(np.mean(pending.value_losses)),
            "entropy": float(np.mean(pending.entropies)),
            "approx_kl": float(np.mean(pending.kls)),
            "bc_loss": float(np.mean(pending.bc_losses)) if pending.bc_losses else 0.0,
            "meta_policy_loss": float(np.mean(pending.meta_policy_losses)),
            "meta_value_loss": float(np.mean(pending.meta_value_losses)),
            "meta_entropy": float(np.mean(pending.meta_entropies)),
        }

    def _meta_losses(
        self, batch: dict[str, torch.Tensor], clip: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """上位方策の (PPO の代理損失, 価値の損失, エントロピー, 意図の模倣の交差エントロピー)。"""
        obs = batch["obs"]
        dist, meta_values = self.policy.meta(obs)
        zero = meta_values.sum() * 0.0
        starts = batch["option_start"]
        own = starts & ~batch["assisted"]

        policy_loss = zero
        if bool(own.any()):
            new_log_probs = dist.log_prob(batch["options"])[own]
            ratio = torch.exp(new_log_probs - batch["meta_log_probs"][own])
            adv = batch["meta_advantages"][own]
            surrogate = torch.min(ratio * adv, torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv)
            policy_loss = -surrogate.mean()

        value_loss = zero
        if bool(starts.any()):
            old = batch["meta_values"][starts]
            returns = batch["meta_returns"][starts]
            values = meta_values[starts]
            clipped = old + torch.clamp(values - old, -clip, clip)
            value_loss = 0.5 * torch.max((values - returns) ** 2, (clipped - returns) ** 2).mean()

        labeled = batch["expert_options"] >= 0
        bc = zero
        if bool(labeled.any()):
            bc = -dist.log_prob(batch["expert_options"].clamp(min=0))[labeled].mean()
        return policy_loss, value_loss, dist.entropy().mean(), bc

    def network_snapshot(self) -> dict[str, Any]:
        """層ごとの重み・勾配・変化量を返す。"""
        layers: list[dict[str, Any]] = []
        with torch.no_grad():
            for name, prm in self.policy.named_parameters():
                if not name.endswith(".weight"):
                    continue
                w = prm.detach()
                delta = float(self._last_delta_norms.get(name, 0.0))
                out_dim, in_dim = (w.shape + (1,))[:2] if w.dim() >= 2 else (w.numel(), 1)
                layers.append(
                    {
                        "name": name.removesuffix(".weight"),
                        "role": "value" if "value" in name else "policy",
                        "inDim": int(in_dim),
                        "outDim": int(out_dim),
                        "weightAbsMean": float(w.abs().mean()),
                        "weightStd": float(w.std()) if w.numel() > 1 else 0.0,
                        "gradNorm": float(self._last_grad_norms.get(name, 0.0)),
                        "deltaNorm": delta,
                    }
                )

            log_std = self.policy.log_std.detach()
            std = torch.exp(
                torch.clamp(log_std, config.PPO_LOG_STD_MIN, config.PPO_LOG_STD_MAX)
            )

        return {
            "updates": int(self._updates),
            "obsDim": int(self.obs_dim),
            "actionDim": int(self.action_dim),
            "hiddenSizes": [int(h) for h in self.policy.hidden_sizes],
            "layers": layers,
            "logStd": [float(x) for x in log_std],
            "actionStd": [float(x) for x in std],
            "logStdMin": float(config.PPO_LOG_STD_MIN),
            "logStdMax": float(config.PPO_LOG_STD_MAX),
            "gradTotalNorm": float(self._last_grad_total),
            "gradClipRate": float(self._last_grad_clip_rate),
            "gradMaxNorm": float(config.PPO_MAX_GRAD_NORM),
        }

    def _drop_pending(self) -> None:
        """進行中の更新を捨てる。重みを差し替えたときは続きを回してはいけない。"""
        self._pending = None

    def reset_rollout(self) -> None:
        """収集中のロールアウトを捨てる。**世界が不連続に変わったときに呼ぶ。**"""
        self.buffer.clear()
        self._last_raw_actions = None
        self._last_meta = None
        self.scheduler.restart()
        self._drop_pending()

    def apply_params(self, params: SimParams) -> None:
        """学習率・gamma・clip などの実行時変更を反映する。"""
        self.gamma = float(params.gamma)
        self.clip_range = float(params.clip_range)
        self.entropy_coef = float(params.entropy_coef)

        new_lr = float(params.learning_rate)
        if new_lr != self.learning_rate:
            self.learning_rate = new_lr
            for group in self.optimizer.param_groups:
                group["lr"] = new_lr

        new_length = max(int(params.rollout_length), 1)
        if new_length != self.rollout_length:
            self.rollout_length = new_length
            self.buffer = RolloutBuffer(
                new_length, self.num_agents, self.obs_dim, self.action_dim
            )
            self._drop_pending()

    def reset_policy(self) -> None:
        """重みを初期化し直し、バッファを空にする。"""
        torch.manual_seed(self._seed)
        self.policy = ActorCritic(
            self.obs_dim, self.action_dim, self.hidden_sizes
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.learning_rate, eps=1e-5
        )
        self.buffer.clear()
        self._updates = 0
        self._last_raw_actions = None
        self._last_meta = None
        self.scheduler.restart()
        self._drop_pending()
        self._last_grad_norms = {}
        self._last_grad_total = 0.0
        self._last_grad_clip_rate = 0.0
        self._last_delta_norms = {}
        self._weights_before_update = {}

    def set_hidden_sizes(self, hidden_sizes: Sequence[int]) -> bool:
        """隠れ層の構成を変えて作り直す。変わったら True。"""
        wanted = tuple(int(h) for h in hidden_sizes)
        if wanted == self.hidden_sizes:
            return False
        self.hidden_sizes = wanted
        self.reset_policy()
        return True

    def snapshot_state(self) -> dict[str, Any]:
        """重み・オプティマイザの状態・更新回数を手元に複製する（自動探索で試行ごとに巻き戻すため）。"""
        return {
            "hidden_sizes": tuple(self.policy.hidden_sizes),
            "policy": copy.deepcopy(self.policy.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "updates": int(self._updates),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """`snapshot_state` の複製へ戻す。溜めかけのロールアウトと進行中の更新は捨てる。"""
        if tuple(state["hidden_sizes"]) != tuple(self.policy.hidden_sizes):
            raise ValueError("隠れ層の構成が複製したときと違うため巻き戻せません")
        self.policy.load_state_dict(state["policy"])
        # Adam の統計は load_state_dict で複製されず、手元の複製と共有されてその場で書き換わる
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
        self._updates = int(state["updates"])
        self.reset_rollout()

    def save(self, path: Path) -> None:
        """チェックポイントをアトミックに書き出す（.tmp -> os.replace）。"""
        self._write_checkpoint(
            path,
            self.policy.hidden_sizes,
            self._updates,
            self.policy.state_dict(),
            self.optimizer.state_dict(),
        )

    def save_state(self, state: dict[str, Any], path: Path) -> None:
        """`snapshot_state` の複製を `save` と同じ形で書き出す。複製なので学習を回しているスレッドの外から呼んでよい。"""
        self._write_checkpoint(path, state["hidden_sizes"], state["updates"], state["policy"], state["optimizer"])

    def _write_checkpoint(
        self,
        path: Path,
        hidden_sizes: Sequence[int],
        updates: int,
        policy: dict[str, Any],
        optimizer: dict[str, Any],
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": [int(h) for h in hidden_sizes],
            "updates": int(updates),
            "policy": policy,
            "optimizer": optimizer,
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def load(self, path: Path) -> bool:
        """検証済みの方策と Adam をまとめて切り替え、失敗時は現在の状態を保つ。"""
        path = Path(path)
        if not path.exists():
            return False
        try:
            payload = torch.load(path, map_location=self.device, weights_only=True)
            prepared = prepare_checkpoint(
                payload, obs_dim=self.obs_dim, action_dim=self.action_dim,
                hidden_sizes=self.hidden_sizes, seed=self._seed,
                learning_rate=self.learning_rate, reference_policy=self.policy,
            )
        except Exception as exc:
            logger.warning("チェックポイントを読み込めませんでした: %s（%s）", path.name, exc)
            return False
        self.policy = prepared.policy.to(self.device)
        self.optimizer = prepared.optimizer
        self._updates = prepared.updates
        self.reset_rollout()
        self.widened_from = prepared.obs_dim if prepared.obs_dim != self.obs_dim else None
        self.upgraded_flat = prepared.flat
        if prepared.flat:
            logger.info("旧方策を階層型へ移しました（意図の入力はゼロ・上位方策と Adam は初期状態）")
        if self.widened_from is not None:
            logger.info("観測 %d 次元を %d 次元へ広げました（追加入力はゼロ・Adam は初期状態）", self.widened_from, self.obs_dim)
        return True
