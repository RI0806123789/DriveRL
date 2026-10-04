"""学習の 1 ステップ（行動 → お手本の割り込み → 環境 → バッファ → 更新）。エンジンと tune_hyperparams.py が同じものを通る。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.contracts import StepResult

if TYPE_CHECKING:
    from app.rl.online_assist import OnlineAssistController
    from app.rl.ppo import PPOTrainer
    from app.sim.env import SimulationEnv


def learning_step(
    env: "SimulationEnv",
    trainer: "PPOTrainer",
    assist: "OnlineAssistController",
    *,
    learn: bool = True,
) -> tuple[StepResult, dict[str, float] | None]:
    """1 ステップ進める。`learn` が偽（実用モード）なら推論だけで、重みは触らない。PPO 更新が終わったときだけ統計を返す。"""
    obs = env.observations
    active = env.active_mask

    actions, log_probs, values = trainer.act(obs, active)
    env.current_options = trainer.current_options if learn else None
    expert = None
    if learn and env.params.online_assist:
        expert = assist.decide(trainer.experience_steps, active, env.assist_danger(active))
    result = env.step(actions, expert=expert)

    if not learn:
        # 学習しない間も、エピソードが終わった車は次のステップで意図を選び直す
        trainer.end_options(result.dones)
        return result, None

    trainer.store(
        obs=obs,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=result.rewards,
        dones=result.dones,
        active=result.active,
        truncated=result.truncated,
        final_obs=result.final_obs,
        # 安全ギミックが操作を丸ごと引き受けた車は、方策の経験として積まない
        learn=result.learn,
        # エキスパートが運転した車は、方策の勾配に入れず模倣の損失に使う
        expert_actions=result.expert_actions,
        assisted=result.assisted,
        expert_options=result.expert_options,
        drive=result.drive,
    )
    return result, trainer.maybe_update(result.obs, result.active)
