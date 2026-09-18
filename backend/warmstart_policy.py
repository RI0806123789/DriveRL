"""経路追従の走りを教師にして、方策を「とりあえず走る」状態へ初期化する。

使い方（backend/ から）:
    .venv\\Scripts\\python.exe warmstart_policy.py --preset ginza
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys

import numpy as np

from app import config
from app.contracts import SimParams
from app.map import build_map_index, get_preset, load_map
from app.rl.ppo import PPOTrainer, peek_hidden_sizes
from app.rl.warmstart import (
    DEFAULT_BATCH,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_STEPS,
    collect_expert,
    fit_policy,
)
from app.sim.env import SimulationEnv

logger = logging.getLogger("warmstart")


def _check_drive(env: SimulationEnv, trainer: PPOTrainer, steps: int) -> None:
    """方策だけで走らせて、どれだけ動くかを見る。"""
    env.reset_all()
    env.autopilot_all = False
    world = env.world
    start = np.stack([world.fleet.x.copy(), world.fleet.y.copy()], axis=1)
    accel = []
    for _ in range(steps):
        obs = env.observations
        active = env.active_mask
        actions, _, _ = trainer.act(obs, active)
        if active.any():
            accel.append(float(np.mean(actions[active][:, 0])))
        env.step(actions)

    moved = np.hypot(world.fleet.x - start[:, 0], world.fleet.y - start[:, 1])
    idx = np.flatnonzero(world.fleet.active)
    print(f"  移動: " + " / ".join(f"#{s} {moved[s]:5.0f}m" for s in idx))
    print(f"  止まったままの台数: {int((moved[idx] < 1.0).sum())} / {idx.size}")
    print(f"  アクセル指令の平均: {np.mean(accel):+.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="ginza")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--vehicles", type=int, default=8)
    parser.add_argument("--check-steps", type=int, default=1200)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    params = SimParams(vehicle_count=int(args.vehicles))
    index = build_map_index(load_map(get_preset(args.preset)))
    env = SimulationEnv(index, params, seed=0)
    env.reset_all()

    print(f"教師データを集めます（{args.preset} / {args.steps} ステップ）")
    obs, actions = collect_expert(env, int(args.steps))
    if obs.shape[0] == 0:
        print("走れる車両がいませんでした")
        return 1
    print(f"  {obs.shape[0]} 件（アクセルの平均 {actions[:, 0].mean():+.3f}）")

    hidden = peek_hidden_sizes(config.CHECKPOINT_PATH) if config.CHECKPOINT_PATH.exists() else None
    trainer = PPOTrainer(
        obs_dim=config.OBS_DIM,
        action_dim=config.ACTION_DIM,
        params=params,
        num_agents=config.MAX_VEHICLES,
        seed=0,
        hidden_sizes=tuple(hidden) if hidden else None,
    )
    restored = trainer.load(config.CHECKPOINT_PATH)
    print(f"いまの重み: {'読めた' if restored else '無い（新規）'} / 更新回数 {trainer.updates}")

    if restored and not args.no_backup:
        backup = config.CHECKPOINT_PATH.with_suffix(".pt.before-warmstart")
        shutil.copy2(config.CHECKPOINT_PATH, backup)
        print(f"  元の重みを {backup.name} へ控えました")

    print("\n回帰します")
    result = fit_policy(
        trainer,
        obs,
        actions,
        epochs=int(args.epochs),
        batch_size=int(args.batch),
        lr=float(args.lr),
    )
    print(f"  損失 {result.first_loss:.5f} → {result.last_loss:.5f}")

    # ★ 集めた分の経験は方策と食い違うので必ず捨てる（PPO は方策オン）
    trainer.reset_rollout()
    trainer.save(config.CHECKPOINT_PATH)
    print(f"  {config.CHECKPOINT_PATH.name} へ保存しました")

    if int(args.check_steps) > 0:
        print(f"\n方策だけで走らせます（{int(args.check_steps) * config.DT:.0f} 秒）")
        _check_drive(env, trainer, int(args.check_steps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
