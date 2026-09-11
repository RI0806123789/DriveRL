"""共有 Actor-Critic（対角ガウス方策）。

memo 5章「全車両が同一の共有ポリシーを使用する（parameter sharing）」に従い、
全スロットの観測をバッチ次元に積んで 1 つのネットワークで処理する。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from app import config

# log_std を可動域の境界ぴったりに置くと torch.clamp の勾配が 0 になるので、
# これだけ内側へ寄せる。exp(1e-3) は std にして 0.1% の差でしかなく、
# 探索の広さには実質影響しない。
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
    """方策（平均のみ NN・対数標準偏差は状態非依存）と価値関数。

    行動は環境側で [-1, 1] にクリップされる前提で、ここではクリップしない
    （クリップすると log_prob が実際のサンプル分布と食い違うため）。
    """

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
        # 出力層の gain: 方策は小さく（初期行動をほぼゼロに）、価値は 1.0
        self.mu_head = _init_linear(nn.Linear(policy_out, self.action_dim), gain=0.01)
        self.value_head = _init_linear(nn.Linear(value_out, 1), gain=1.0)

        # 対数標準偏差は状態非依存の学習パラメータ
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(config.PPO_LOG_STD_INIT))
        )
        self.clamp_log_std()

    # ------------------------------------------------------------------

    def _distribution(self, obs: torch.Tensor) -> Normal:
        mu = self.mu_head(self.policy_trunk(obs))
        # 保険のクランプ。**これだけに頼ってはいけない。**
        # torch.clamp は範囲外の入力に対する勾配を 0 にするので、一度でも外へ
        # 出ると上げることも下げることもできなくなる（実際にそれで方策が死んだ）。
        # 実効的な歯止めは clamp_log_std() が optimizer.step() の後に掛ける。
        log_std = torch.clamp(self.log_std, config.PPO_LOG_STD_MIN, config.PPO_LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mu)
        return Normal(mu, std)

    @torch.no_grad()
    def clamp_log_std(self) -> None:
        """log_std を可動域の**内側**へ丸める。**optimizer.step() の直後に呼ぶこと。**

        境界ちょうどに置いてはいけない。torch.clamp の逆伝播は
         のときだけ勾配を通すので、**上限ぴったりでも勾配が 0** になり
        結局そこから動けなくなる（検証スクリプトで実測した）。
        そこで境界より _EPS だけ内側へ寄せる。こうすると常に勾配が流れ、
        エントロピー報酬が押し上げても毎回押し戻される「壁」として機能する。
        """
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
