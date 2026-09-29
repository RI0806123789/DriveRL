"""階層型の Actor-Critic（上位: 意図を選ぶ離散方策 / 下位: 意図の下で操作を出す連続方策）。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from app import config
from app.contracts import DriveState

_LOG_STD_EPS = 1e-3

NUM_OPTIONS = len(config.HRL_OPTIONS)
OPTION_CRUISE, OPTION_FOLLOW, OPTION_YIELD, OPTION_STOP = range(NUM_OPTIONS)

__all__ = [
    "HierarchicalActorCritic",
    "OptionScheduler",
    "NUM_OPTIONS",
    "OPTION_CRUISE",
    "OPTION_FOLLOW",
    "OPTION_YIELD",
    "OPTION_STOP",
    "META_MODULES",
    "option_name",
    "sub_reward_shaping",
]

#: 上位方策の部品。平らな方策のチェックポイントには無いので、読み込むときは初期値のまま残す
META_MODULES = ("meta_trunk", "meta_head", "meta_value_trunk", "meta_value_head")


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


def option_name(index: int) -> str:
    """意図の添字から名前（`config.HRL_OPTIONS`）を引く。範囲外は空文字。"""
    i = int(index)
    return config.HRL_OPTIONS[i] if 0 <= i < NUM_OPTIONS else ""


class HierarchicalActorCritic(nn.Module):
    """上位（観測 → 意図 4 値と価値）と下位（観測 + 意図の one-hot → 操作 2 値と価値）。"""

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
        self.num_options = NUM_OPTIONS
        self.sub_in_dim = self.obs_dim + self.num_options

        # 下位の部品の名前は平らな方策のころのまま（重みを引き継ぐため）
        self.policy_trunk, policy_out = _build_trunk(self.sub_in_dim, self.hidden_sizes)
        self.value_trunk, value_out = _build_trunk(self.sub_in_dim, self.hidden_sizes)
        self.mu_head = _init_linear(nn.Linear(policy_out, self.action_dim), gain=0.01)
        self.value_head = _init_linear(nn.Linear(value_out, 1), gain=1.0)
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(config.PPO_LOG_STD_INIT))
        )
        self.option_bias = nn.Parameter(torch.zeros(self.num_options, self.action_dim))
        with torch.no_grad():
            self.option_bias[:, 0] = torch.tensor(
                config.HRL_OPTION_ACCEL_PRIOR, dtype=torch.float32
            )

        self.meta_trunk, meta_out = _build_trunk(self.obs_dim, self.hidden_sizes)
        self.meta_head = _init_linear(nn.Linear(meta_out, self.num_options), gain=0.01)
        self.meta_value_trunk, meta_value_out = _build_trunk(self.obs_dim, self.hidden_sizes)
        self.meta_value_head = _init_linear(nn.Linear(meta_value_out, 1), gain=1.0)

        self.clamp_log_std()

    @torch.no_grad()
    def clamp_log_std(self) -> None:
        """log_std を可動域の**内側**へ丸める。**optimizer.step() の直後に呼ぶこと。**"""
        self.log_std.clamp_(
            config.PPO_LOG_STD_MIN + _LOG_STD_EPS,
            config.PPO_LOG_STD_MAX - _LOG_STD_EPS,
        )

    def meta(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        """上位方策の意図の分布と、上位の状態価値 (B,)。"""
        logits = self.meta_head(self.meta_trunk(obs))
        values = self.meta_value_head(self.meta_value_trunk(obs)).squeeze(-1)
        return Categorical(logits=logits), values

    @torch.no_grad()
    def greedy_options(self, obs: torch.Tensor) -> torch.Tensor:
        """いちばん確率の高い意図 (B,)。意図を渡されなかったときに使う。"""
        return self.meta_head(self.meta_trunk(obs)).argmax(dim=-1)

    def _options(self, obs: torch.Tensor, options: torch.Tensor | None) -> torch.Tensor:
        if options is None:
            return self.greedy_options(obs)
        return torch.as_tensor(options, dtype=torch.long).reshape(obs.shape[:-1])

    def _sub_input(self, obs: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(options, self.num_options).to(obs.dtype)
        return torch.cat([obs, one_hot], dim=-1)

    def mean_action(self, obs: torch.Tensor, options: torch.Tensor | None = None) -> torch.Tensor:
        """下位方策の行動分布の平均（-1..1）。模倣の損失で教師の操作と比べる。"""
        opts = self._options(obs, options)
        raw = self.mu_head(self.policy_trunk(self._sub_input(obs, opts)))
        return torch.tanh(raw + self.option_bias[opts])

    def _distribution(self, obs: torch.Tensor, options: torch.Tensor | None = None) -> Normal:
        mu = self.mean_action(obs, options)
        log_std = torch.clamp(self.log_std, config.PPO_LOG_STD_MIN, config.PPO_LOG_STD_MAX)
        return Normal(mu, torch.exp(log_std).expand_as(mu))

    def sub_value(self, obs: torch.Tensor, options: torch.Tensor | None = None) -> torch.Tensor:
        """下位の状態価値 (B,)。"""
        opts = self._options(obs, options)
        return self.value_head(self.value_trunk(self._sub_input(obs, opts))).squeeze(-1)

    def forward(
        self, obs: torch.Tensor, options: torch.Tensor | None = None
    ) -> tuple[Normal, torch.Tensor]:
        """下位方策の行動分布と状態価値 (B,)。意図を省くと上位方策のいちばん確率の高い意図を使う。"""
        opts = self._options(obs, options)
        return self._distribution(obs, opts), self.sub_value(obs, opts)

    @torch.no_grad()
    def select_option(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """上位方策から意図を引く。戻り値 (options, log_probs, meta_values)。"""
        dist, values = self.meta(obs)
        options = dist.sample()
        return options, dist.log_prob(options), values

    @torch.no_grad()
    def get_action(
        self, obs: torch.Tensor, options: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """意図の下で下位方策から操作を引く。戻り値 (actions, log_probs, values)。"""
        dist, values = self.forward(obs, options)
        actions = dist.sample()
        return actions, dist.log_prob(actions).sum(dim=-1), values

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, options: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`get_action` と同じ。意図を省くと上位方策のいちばん確率の高い意図を使う。"""
        return self.get_action(obs, self._options(obs, options))

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor, options: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """既存の操作を再評価する。戻り値 (log_probs, entropy, values)。"""
        dist, values = self.forward(obs, options)
        return dist.log_prob(actions).sum(dim=-1), dist.entropy().sum(dim=-1), values

    def evaluate_options(
        self, obs: torch.Tensor, options: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """既存の意図を上位方策で再評価する。戻り値 (log_probs, entropy, meta_values)。"""
        dist, values = self.meta(obs)
        opts = torch.as_tensor(options, dtype=torch.long)
        return dist.log_prob(opts), dist.entropy(), values


class OptionScheduler:
    """スロットごとに意図を `period` ステップ保つ。エピソードが終わった車は次のステップで選び直す。"""

    def __init__(self, num_agents: int, period: int = config.HRL_OPTION_STEPS) -> None:
        self.num_agents = int(num_agents)
        self.period = max(1, int(period))
        self.options = np.zeros(self.num_agents, dtype=np.int64)
        self.age = np.full(self.num_agents, self.period, dtype=np.int64)
        self._was_active = np.zeros(self.num_agents, dtype=bool)

    def due(self, active: np.ndarray) -> np.ndarray:
        """このステップで意図を選び直すスロット。走り出したばかりの車も選び直す。"""
        act = np.asarray(active, dtype=bool).reshape(self.num_agents)
        return act & ((self.age >= self.period) | ~self._was_active)

    def assign(self, mask: np.ndarray, options: np.ndarray) -> None:
        m = np.asarray(mask, dtype=bool).reshape(self.num_agents)
        self.options[m] = np.asarray(options, dtype=np.int64).reshape(self.num_agents)[m]
        self.age[m] = 0

    def tick(self, active: np.ndarray) -> None:
        act = np.asarray(active, dtype=bool).reshape(self.num_agents)
        self.age[act] += 1
        self._was_active = act.copy()

    def restart(self, mask: np.ndarray | None = None) -> None:
        """次のステップで意図を選び直させる（エピソードの終わり・世界の不連続な変化）。"""
        if mask is None:
            self.age[:] = self.period
            return
        self.age[np.asarray(mask, dtype=bool).reshape(self.num_agents)] = self.period


def sub_reward_shaping(
    options: np.ndarray, drive: DriveState, active: np.ndarray
) -> np.ndarray:
    """下位方策の報酬へ足す整形（0 以下）。操作の急な変化・車線中心からの横ずれ・意図の速度帯からの外れ。"""
    opts = np.asarray(options, dtype=np.int64)
    speed = np.abs(np.asarray(drive.speed, dtype=np.float64))
    lateral = np.asarray(drive.lateral, dtype=np.float64)
    lead = np.asarray(drive.lead_speed, dtype=np.float64)
    jerk = float(config.HRL_JERK_COEF) * np.asarray(drive.action_delta_sq, dtype=np.float64)
    lane = float(config.HRL_LANE_COEF) * lateral**2

    cruise_short = np.maximum(0.0, float(config.HRL_CRUISE_MIN_MPS) - speed)
    follow_gap = np.where(np.isfinite(lead), np.abs(speed - np.nan_to_num(lead)), cruise_short)
    off_band = np.select(
        [opts == OPTION_CRUISE, opts == OPTION_FOLLOW, opts == OPTION_YIELD, opts == OPTION_STOP],
        [
            cruise_short,
            follow_gap,
            np.maximum(0.0, speed - float(config.HRL_YIELD_MAX_MPS)),
            np.maximum(0.0, speed - float(config.HRL_STOP_MAX_MPS)),
        ],
        default=0.0,
    )
    consistency = float(config.HRL_CONSISTENCY_COEF) * np.minimum(
        1.0, off_band / float(config.HRL_CONSISTENCY_SPAN_MPS)
    )
    shaping = -(jerk + lane + consistency)
    return (shaping * np.asarray(active, dtype=bool)).astype(np.float32)
