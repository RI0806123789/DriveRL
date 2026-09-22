"""経路追従の操作を教師にして方策を初期化する（行動クローニング）。"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from app import config

__all__ = [
    "ExpertData",
    "WarmstartResult",
    "collect_expert",
    "fit_policy",
    "fit_value",
    "set_exploration",
]

logger = logging.getLogger(__name__)

#: 教師の行動は tanh の外側に置かない（±1 には決して届かないので学習が終わらない）
TEACH_LIMIT = 0.98

#: ウォームスタート直後の探索ノイズ。**広すぎても狭すぎても損をする。**
#: 既定の 0.6 のままだとステアに乗って車線を外れ、衝突の罰（-100）で「止まる方が得」へ
#: 戻る。逆に 0.15 まで絞ると衝突は止まるが探索が狭く、学習の伸びが鈍るうえ、
#: エントロピーが -0.96 に張り付いて画面上は「学習が動いていない」ように見える
#: （実測・銀座 60 秒の報酬: 0.6 が +61〜+78 / 0.15 が +148 / 0.37 が +126〜+264）
WARMSTART_STD = 0.3

DEFAULT_STEPS = 3000
DEFAULT_EPOCHS = 8
DEFAULT_BATCH = 256
DEFAULT_LR = 3e-4


@dataclass(slots=True)
class ExpertData:
    """経路追従で走らせて集めた、観測・操作・そのときのリターン。"""

    obs: np.ndarray
    actions: np.ndarray
    returns: np.ndarray

    def __len__(self) -> int:
        return int(self.obs.shape[0])


@dataclass(slots=True)
class WarmstartResult:
    """回帰の結果。"""

    samples: int
    first_loss: float
    last_loss: float


def collect_expert(env, steps: int = DEFAULT_STEPS, *, gamma: float | None = None) -> ExpertData:
    """経路追従で走らせながら、そのときの観測・操作・リターンを集める。"""
    discount = float(env.params.gamma if gamma is None else gamma)
    restore = env.autopilot_all
    env.autopilot_all = True

    idle = np.zeros((config.MAX_VEHICLES, config.ACTION_DIM), dtype=np.float32)
    idle[:, 0] = -1.0

    n = config.MAX_VEHICLES
    obs_seq: list[list[np.ndarray]] = [[] for _ in range(n)]
    act_seq: list[list[np.ndarray]] = [[] for _ in range(n)]
    rew_seq: list[list[float]] = [[] for _ in range(n)]
    end_seq: list[list[bool]] = [[] for _ in range(n)]

    try:
        for _ in range(int(steps)):
            obs = np.asarray(env.observations, dtype=np.float32)
            active = np.asarray(env.active_mask, dtype=bool)
            slots = np.flatnonzero(active)
            teach = {}
            for slot in slots:
                teach[int(slot)] = np.asarray(env._autopilot(int(slot)), dtype=np.float32)

            result = env.step(idle)

            for slot in slots:
                slot = int(slot)
                obs_seq[slot].append(obs[slot].copy())
                act_seq[slot].append(teach[slot])
                rew_seq[slot].append(float(result.rewards[slot]))
                # 打ち切り（時間切れ）も区切りとして扱う。続きの価値を知らないので同じこと
                truncated = result.truncated is not None and bool(result.truncated[slot])
                end_seq[slot].append(bool(result.dones[slot]) or truncated)
    finally:
        env.autopilot_all = restore

    obs_out: list[np.ndarray] = []
    act_out: list[np.ndarray] = []
    ret_out: list[np.ndarray] = []
    for slot in range(n):
        if not obs_seq[slot]:
            continue
        rewards = np.asarray(rew_seq[slot], dtype=np.float32)
        ends = np.asarray(end_seq[slot], dtype=bool)
        returns = np.zeros_like(rewards)
        running = 0.0
        # 末尾は「この先いくらもらえるか」を知らないまま 0 から積むので、
        #   後ろほど実力より低く出る。捨てずに使うと価値関数が悲観側へ寄る
        for t in range(rewards.shape[0] - 1, -1, -1):
            running = float(rewards[t]) + (0.0 if ends[t] else discount * running)
            returns[t] = running
        obs_out.append(np.asarray(obs_seq[slot], dtype=np.float32))
        act_out.append(np.asarray(act_seq[slot], dtype=np.float32))
        ret_out.append(returns)

    if not obs_out:
        return ExpertData(
            obs=np.zeros((0, config.OBS_DIM), dtype=np.float32),
            actions=np.zeros((0, config.ACTION_DIM), dtype=np.float32),
            returns=np.zeros((0,), dtype=np.float32),
        )

    actions = np.concatenate(act_out, axis=0)
    np.clip(actions, -TEACH_LIMIT, TEACH_LIMIT, out=actions)
    return ExpertData(
        obs=np.concatenate(obs_out, axis=0),
        actions=actions,
        returns=np.concatenate(ret_out, axis=0),
    )


def _regress(
    modules: list[torch.nn.Module],
    forward,
    t_obs: torch.Tensor,
    target: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    label: str,
) -> WarmstartResult:
    n = int(t_obs.shape[0])
    params: list[torch.nn.Parameter] = []
    for m in modules:
        params.extend(m.parameters())
    opt = torch.optim.Adam(params, lr=float(lr))
    rng = np.random.default_rng(int(seed))

    first = last = 0.0
    for epoch in range(int(epochs)):
        order = rng.permutation(n)
        total = 0.0
        batches = 0
        for start in range(0, n, int(batch_size)):
            idx = torch.from_numpy(order[start : start + int(batch_size)].astype(np.int64))
            loss = F.mse_loss(forward(t_obs[idx]), target[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            batches += 1
        last = total / max(batches, 1)
        if epoch == 0:
            first = last
        logger.info("%s %d/%d: 損失 %.5f", label, epoch + 1, int(epochs), last)

    return WarmstartResult(samples=n, first_loss=first, last_loss=last)


def fit_policy(
    trainer,
    data: ExpertData,
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    seed: int = 0,
) -> WarmstartResult:
    """集めた操作へ方策の平均を回帰する。`log_std` には触らない。"""
    if len(data) == 0:
        return WarmstartResult(samples=0, first_loss=0.0, last_loss=0.0)

    policy = trainer.policy
    return _regress(
        [policy.policy_trunk, policy.mu_head],
        lambda x: torch.tanh(policy.mu_head(policy.policy_trunk(x))),
        torch.from_numpy(np.ascontiguousarray(data.obs)),
        torch.from_numpy(np.ascontiguousarray(data.actions)),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        label="方策",
    )


def set_exploration(trainer, std: float = WARMSTART_STD) -> float:
    """探索ノイズを揃える。戻り値は実際に設定された標準偏差。"""
    policy = trainer.policy
    with torch.no_grad():
        policy.log_std.data.fill_(math.log(max(float(std), 1e-6)))
    policy.clamp_log_std()
    return float(torch.exp(policy.log_std.detach())[0])


def fit_value(
    trainer,
    data: ExpertData,
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    seed: int = 0,
) -> WarmstartResult:
    """集めたリターンへ価値関数を回帰する。**方策と必ずセットで呼ぶこと。**"""
    if len(data) == 0:
        return WarmstartResult(samples=0, first_loss=0.0, last_loss=0.0)

    policy = trainer.policy
    return _regress(
        [policy.value_trunk, policy.value_head],
        lambda x: policy.value_head(policy.value_trunk(x)).squeeze(-1),
        torch.from_numpy(np.ascontiguousarray(data.obs)),
        torch.from_numpy(np.ascontiguousarray(data.returns)),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        label="価値",
    )
