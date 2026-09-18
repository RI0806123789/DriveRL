"""共有 Actor-Critic（対角ガウス方策）。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from app import config

_LOG_STD_EPS = 1e-3

__all__ = ["ActorCritic"]


def _init_linear(layer: nn.Linear, gain: float) -> nn.Linear:
    """直交初期化。バイアスはゼロ。"""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _build_trunk(in_dim: int, hidden_sizes: Sequence[int]) -> tuple[nn.Sequential, int]:
    """Tanh 活性の MLP 本体を作る。戻り値は (モジュール, 出力次元)。"""
    layers: list[nn.Module] = []
    last = int(in_dim)
    for size in hidden_sizes:
        layers.append(_init_linear(nn.Linear(last, int(size)), gain=2.0**0.5))
        layers.append(nn.Tanh())
        last = int(size)
    return nn.Sequential(*layers), last


class ActorCritic(nn.Module):
    """方策（平均のみ NN・対数標準偏差は状態非依存）と価値関数。"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = config.PPO_HIDDEN_SIZES,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)

        self.policy_trunk, policy_out = _build_trunk(self.obs_dim, self.hidden_sizes)
        self.value_trunk, value_out = _build_trunk(self.obs_dim, self.hidden_sizes)
        self.mu_head = _init_linear(nn.Linear(policy_out, self.action_dim), gain=0.01)
        self.value_head = _init_linear(nn.Linear(value_out, 1), gain=1.0)

        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(config.PPO_LOG_STD_INIT))
        )
        self.clamp_log_std()

    def _distribution(self, obs: torch.Tensor) -> Normal:
        mu = torch.tanh(self.mu_head(self.policy_trunk(obs)))
        log_std = torch.clamp(self.log_std, config.PPO_LOG_STD_MIN, config.PPO_LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mu)
        return Normal(mu, std)

    @torch.no_grad()
    def clamp_log_std(self) -> None:
        """log_std を可動域の**内側**へ丸める。**optimizer.step() の直後に呼ぶこと。**"""
        self.log_std.clamp_(
            config.PPO_LOG_STD_MIN + _LOG_STD_EPS,
            config.PPO_LOG_STD_MAX - _LOG_STD_EPS,
        )

    def forward(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        """行動分布と状態価値を返す。値の shape は (B,)。"""
        dist = self._distribution(obs)
        values = self.value_head(self.value_trunk(obs)).squeeze(-1)
        return dist, values

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """行動をサンプリングする。戻り値 (actions, log_probs, values)。"""
        dist, values = self.forward(obs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        return actions, log_probs, values

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """既存の行動を再評価する。戻り値 (log_probs, entropy, values)。"""
        dist, values = self.forward(obs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy, values
