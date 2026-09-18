"""経路追従の操作を教師にして方策を初期化する（行動クローニング）。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from app import config

__all__ = ["WarmstartResult", "collect_expert", "fit_policy"]

logger = logging.getLogger(__name__)

#: 教師の行動は tanh の外側に置かない（±1 には決して届かないので学習が終わらない）
TEACH_LIMIT = 0.98

DEFAULT_STEPS = 3000
DEFAULT_EPOCHS = 8
DEFAULT_BATCH = 256
DEFAULT_LR = 3e-4


@dataclass(slots=True)
class WarmstartResult:
    """回帰の結果。"""

    samples: int
    first_loss: float
    last_loss: float


def collect_expert(env, steps: int = DEFAULT_STEPS) -> tuple[np.ndarray, np.ndarray]:
    """経路追従で走らせながら、そのときの観測と操作を集める。

    ★ **観測は `step()` の前に取ること。** 教師の操作も同じ時点の状態から出すので、
    ずらすと「1 ステップ先の状況に対する操作」を教えることになる。
    """
    restore = env.autopilot_all
    env.autopilot_all = True

    idle = np.zeros((config.MAX_VEHICLES, config.ACTION_DIM), dtype=np.float32)
    idle[:, 0] = -1.0

    obs_chunks: list[np.ndarray] = []
    act_chunks: list[np.ndarray] = []
    try:
        for _ in range(int(steps)):
            obs = np.asarray(env.observations, dtype=np.float32)
            active = np.asarray(env.active_mask, dtype=bool)
            slots = np.flatnonzero(active)
            if slots.size:
                teach = np.zeros((slots.size, config.ACTION_DIM), dtype=np.float32)
                for i, slot in enumerate(slots):
                    teach[i] = env._autopilot(int(slot))
                obs_chunks.append(obs[slots].copy())
                act_chunks.append(teach)
            env.step(idle)
    finally:
        env.autopilot_all = restore

    if not obs_chunks:
        return (
            np.zeros((0, config.OBS_DIM), dtype=np.float32),
            np.zeros((0, config.ACTION_DIM), dtype=np.float32),
        )
    actions = np.concatenate(act_chunks, axis=0)
    np.clip(actions, -TEACH_LIMIT, TEACH_LIMIT, out=actions)
    return np.concatenate(obs_chunks, axis=0), actions


def fit_policy(
    trainer,
    obs: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    seed: int = 0,
) -> WarmstartResult:
    """集めた操作へ方策の平均を回帰する。価値関数と log_std には触らない。"""
    n = int(obs.shape[0])
    if n == 0:
        return WarmstartResult(samples=0, first_loss=0.0, last_loss=0.0)

    policy = trainer.policy
    # ★ 価値関数まで巻き込まないこと。ここで教えられるのは「どう操作するか」だけで、
    #   「その状態の価値」は経路追従の走りからは出てこない（PPO が学び直す）
    params = list(policy.policy_trunk.parameters()) + list(policy.mu_head.parameters())
    opt = torch.optim.Adam(params, lr=float(lr))

    t_obs = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32))
    t_act = torch.from_numpy(np.ascontiguousarray(actions, dtype=np.float32))
    rng = np.random.default_rng(int(seed))

    first = last = 0.0
    for epoch in range(int(epochs)):
        order = rng.permutation(n)
        total = 0.0
        batches = 0
        for start in range(0, n, int(batch_size)):
            idx = torch.from_numpy(order[start : start + int(batch_size)].astype(np.int64))
            mu = torch.tanh(policy.mu_head(policy.policy_trunk(t_obs[idx])))
            loss = F.mse_loss(mu, t_act[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            batches += 1
        last = total / max(batches, 1)
        if epoch == 0:
            first = last
        logger.info("ウォームスタート %d/%d: 損失 %.5f", epoch + 1, int(epochs), last)

    return WarmstartResult(samples=n, first_loss=first, last_loss=last)
