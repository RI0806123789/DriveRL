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
from app.contracts import SimParams
from app.rl.buffer import RolloutBuffer
from app.rl.policy import ActorCritic

__all__ = [
    "PPOTrainer",
    "CHECKPOINT_FORMAT",
    "KNOWN_CHECKPOINT_FORMATS",
    "peek_hidden_sizes",
]

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT = "autoware-sim-ppo-1"

KNOWN_CHECKPOINT_FORMATS: frozenset[object] = frozenset({CHECKPOINT_FORMAT, 1})


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
    grad_sums: dict[str, float] = field(default_factory=dict)
    grad_counts: dict[str, int] = field(default_factory=dict)


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
        self._pending: _PendingUpdate | None = None

        self._last_grad_norms: dict[str, float] = {}
        self._weights_before_update: dict[str, torch.Tensor] = {}
        self._last_delta_norms: dict[str, float] = {}

    @property
    def updates(self) -> int:
        return self._updates

    def act(
        self, obs: np.ndarray, active: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """行動をサンプリングする。"""
        n = self.num_agents
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(n, self.obs_dim)
        active_arr = np.asarray(active, dtype=bool).reshape(n)

        with torch.no_grad():
            t_obs = torch.from_numpy(np.ascontiguousarray(obs_arr))
            raw_actions, log_probs, values = self.policy.act(t_obs)

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
    ) -> None:
        """1 ステップ分をバッファに積む。"""
        raw = self._last_raw_actions
        actions_arr = np.asarray(actions, dtype=np.float32).reshape(
            self.num_agents, self.action_dim
        )
        if raw is not None and raw.shape == actions_arr.shape:
            actions_arr = raw
        self._last_raw_actions = None

        self.buffer.add(
            obs, actions_arr, log_probs, values, rewards, dones, active, truncated
        )

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
            _dist, last_values = self.policy.forward(t_obs)
        last_values_np = last_values.numpy().astype(np.float32)

        self.buffer.compute_returns_and_advantages(
            last_values_np, last_active_arr, self.gamma, config.PPO_GAE_LAMBDA
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
                batch["obs"], batch["actions"]
            )
            log_ratio = new_log_probs - batch["log_probs"]
            ratio = torch.exp(log_ratio)

            adv = batch["advantages"]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

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

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), float(config.PPO_MAX_GRAD_NORM)
            )
            with torch.no_grad():
                for name, prm in self.policy.named_parameters():
                    if prm.grad is None:
                        continue
                    g = float(prm.grad.norm())
                    pending.grad_sums[name] = pending.grad_sums.get(name, 0.0) + g
                    pending.grad_counts[name] = pending.grad_counts.get(name, 0) + 1
            self.optimizer.step()
            self.policy.clamp_log_std()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
            pending.policy_losses.append(float(policy_loss.detach()))
            pending.value_losses.append(float(value_loss.detach()))
            pending.entropies.append(float(entropy_mean.detach()))
            pending.kls.append(float(approx_kl))

        pending.cursor = end
        if pending.cursor < len(pending.schedule):
            return None

        self._pending = None
        self._updates += 1
        self._last_grad_norms = {
            k: pending.grad_sums[k] / max(1, pending.grad_counts.get(k, 1))
            for k in pending.grad_sums
        }
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
        }

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
                        "role": "value" if name.startswith("value") else "policy",
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
        }

    def _drop_pending(self) -> None:
        """進行中の更新を捨てる。重みを差し替えたときは続きを回してはいけない。"""
        self._pending = None

    def reset_rollout(self) -> None:
        """収集中のロールアウトを捨てる。**世界が不連続に変わったときに呼ぶ。**"""
        self.buffer.clear()
        self._last_raw_actions = None
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
        self._drop_pending()
        self._last_grad_norms = {}
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

    def save(self, path: Path) -> None:
        """チェックポイントをアトミックに書き出す（.tmp -> os.replace）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": list(self.policy.hidden_sizes),
            "updates": self._updates,
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def load(self, path: Path) -> bool:
        """チェックポイントを読み込む。読めたら True、形状不一致等なら False。"""
        path = Path(path)
        if not path.exists():
            return False
        try:
            payload = torch.load(path, map_location=self.device, weights_only=True)
        except Exception as exc:  # noqa: BLE001 - 拒否した理由を残す
            logger.warning(
                "チェックポイントを安全モードで読み込めませんでした: %s（%s: %s）",
                path.name,
                type(exc).__name__,
                str(exc).splitlines()[0] if str(exc) else "",
            )
            return False
        if not isinstance(payload, dict):
            return False
        fmt = payload.get("format")
        if fmt is not None and fmt not in KNOWN_CHECKPOINT_FORMATS:
            logger.warning(
                "見覚えのないチェックポイント形式です: %r（%s）", fmt, path.name
            )
        try:
            if int(payload.get("obs_dim", -1)) != self.obs_dim:
                return False
            if int(payload.get("action_dim", -1)) != self.action_dim:
                return False
            if tuple(payload.get("hidden_sizes", ())) != tuple(self.policy.hidden_sizes):
                return False
        except (TypeError, ValueError):
            logger.warning("チェックポイントのモデル定義が壊れています: %s", path.name)
            return False
        before = copy.deepcopy(self.policy.state_dict())
        try:
            self._drop_pending()
            self.policy.load_state_dict(payload["policy"])
            if "optimizer" in payload:
                self.optimizer.load_state_dict(payload["optimizer"])
                for group in self.optimizer.param_groups:
                    group["lr"] = self.learning_rate
        except Exception:
            logger.exception("チェックポイントの適用に失敗しました: %s", path.name)
            try:
                self.policy.load_state_dict(before)
            except Exception:
                logger.exception("方策の巻き戻しにも失敗しました。方策を初期化します")
                self.reset_policy()
            return False
        finally:
            self.buffer.clear()
            self._last_raw_actions = None
        self.policy.clamp_log_std()
        self._updates = int(payload.get("updates", 0))
        return True
