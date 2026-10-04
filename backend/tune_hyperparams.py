"""Optuna で PPO と報酬の重みのハイパーパラメータを自動探索する（サーバーも画面も使わない）。"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from app import config
from app.contracts import Bounds, MapData, MapEdge, MapNode, MapIndex, SimParams
from app.map import build_map_index, get_preset, load_map
from app.rl.online_assist import OnlineAssistController
from app.rl.ppo import PPOTrainer, peek_hidden_sizes
from app.runtime import autotune
from app.runtime.learning_step import learning_step
from app.sim.env import SimulationEnv

logger = logging.getLogger("tune_hyperparams")

#: 地図のキャッシュを読まない合成の碁盤の目（短い動作確認とテスト用。信号も建物も無い）
SYNTHETIC_PRESET = "grid"


def synthetic_grid(k: int = 5, step: float = 120.0) -> MapData:
    """建物も信号も無い k×k の碁盤の目（両方向 1 車線ずつ）。"""
    nodes = [MapNode(i * k + j, j * step, i * step) for i in range(k) for j in range(k)]
    edges: list[MapEdge] = []
    for i in range(k):
        for j in range(k):
            for di, dj in ((0, 1), (1, 0)):
                if i + di >= k or j + dj >= k:
                    continue
                a, b = nodes[i * k + j], nodes[(i + di) * k + j + dj]
                for u, v in ((a, b), (b, a)):
                    edges.append(
                        MapEdge(len(edges), u.id, v.id, 1, 7.0, False, 11.1, [(u.x, u.y), (v.x, v.y)], step)
                    )
    edge = (k - 1) * step
    return MapData(
        SYNTHETIC_PRESET, "合成の碁盤の目", 0.0, 0.0, edge,
        Bounds(-20.0, -20.0, edge + 20.0, edge + 20.0), nodes, edges, [],
    )


def load_index(preset_id: str) -> MapIndex:
    """プリセットの地図（`grid` なら合成の碁盤の目）を読む。"""
    if preset_id == SYNTHETIC_PRESET:
        return build_map_index(synthetic_grid())
    preset = get_preset(preset_id)
    if preset is None:
        raise SystemExit(f"未知のプリセットです: {preset_id}（grid か、app/map/presets.py の ID を指定してください）")
    return build_map_index(load_map(preset))


@dataclass
class BestTrial:
    number: int
    score: autotune.TrialScore
    patch: dict[str, float]


class HeadlessTuner:
    """study の 1 試行を走らせる目的関数。試行ごとに探索開始時の重みへ巻き戻してから学習させる。"""

    def __init__(
        self,
        env: SimulationEnv,
        trainer: PPOTrainer,
        base_params: SimParams,
        *,
        trial_steps: int,
        report_every: int,
        best_policy_path: Path,
        best_params_path: Path,
        summary_path: Path,
    ) -> None:
        self.env = env
        self.trainer = trainer
        self.base_params = base_params
        self.trial_steps = int(trial_steps)
        self.report_every = int(report_every)
        self.best_policy_path = Path(best_policy_path)
        self.best_params_path = Path(best_params_path)
        self.summary_path = Path(summary_path)
        self.assist = OnlineAssistController(config.MAX_VEHICLES, seed=0)
        self.baseline = trainer.snapshot_state()
        self.best: BestTrial | None = None

    def objective(self, trial: Any) -> float:
        import optuna

        raw = {
            d.wire: trial.suggest_float(d.wire, d.low, d.high, log=d.log, step=d.step)
            for d in autotune.SEARCH_SPACE
        }
        patch = autotune.validated_patch(raw, self.base_params)
        params = replace(self.base_params)
        params.apply_wire(patch)

        trainer = self.trainer
        env = self.env
        trainer.restore_state(self.baseline)
        env.apply_params(params)
        trainer.apply_params(params)
        env.reset_all()
        self.assist.reset()

        monitor = autotune.TrialMonitor(self.trial_steps, params.max_speed, self.report_every)
        started = time.perf_counter()
        while not monitor.done:
            result, _stats = learning_step(env, trainer, self.assist)
            monitor.observe(result, env.world.fleet.speed)
            if not autotune.is_policy_healthy(trainer.policy):
                trial.set_user_attr("outcome", "diverged")
                trial.set_user_attr("steps", monitor.steps)
                print(f"  #{trial.number:<3d} 重みが NaN / Inf になったため打ち切り（{monitor.steps} ステップ目）")
                return autotune.SCORE_DIVERGED
            k = monitor.checkpoint()
            if k is not None:
                trial.report(monitor.interim().value, k)
                if trial.should_prune():
                    score = monitor.interim()
                    for key, value in score.attrs().items():
                        trial.set_user_attr(key, value)
                    trial.set_user_attr("outcome", "pruned")
                    trial.set_user_attr("steps", monitor.steps)
                    print(f"  #{trial.number:<3d} 見込みが薄いので打ち切り（途中のスコア {score.value:+.3f}・{monitor.steps} ステップ）")
                    raise optuna.TrialPruned()

        score = monitor.final()
        for key, value in score.attrs().items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("outcome", "complete")
        trial.set_user_attr("steps", monitor.steps)
        improved = self.best is None or score.value > self.best.score.value
        if improved:
            self.best = BestTrial(trial.number, score, patch)
            if autotune.save_policy_checked(trainer, self.best_policy_path):
                autotune.write_best_params(self.best_params_path, patch)
        print(
            f"  #{trial.number:<3d} スコア {score.value:+.3f}"
            f"（到達 {score.goal_rate:.0%} / 衝突 {score.collision_rate:.0%} / 逸脱 {score.offroad_rate:.0%}"
            f" / 走行 {score.moving:.0%} / エピソード {score.episodes}・{time.perf_counter() - started:.0f} 秒）"
            + ("  ← 最良を更新" if improved else "")
        )
        return score.value

    def after_trial(self, study: Any, _trial: Any) -> None:
        try:
            autotune.write_trial_summary(study, self.summary_path)
        except Exception:
            logger.exception("試行の一覧（CSV）を書き出せませんでした（探索は続けます）")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="ginza", help="探索に使う地図（grid は合成の碁盤の目）")
    parser.add_argument("--trials", type=int, default=20, help="この実行で回す試行の数")
    parser.add_argument("--trial-steps", type=int, default=autotune.CLI_TRIAL_STEPS, help="1 試行のステップ数（20Hz）")
    parser.add_argument("--report-every", type=int, default=autotune.REPORT_EVERY_STEPS, help="枝刈りの判断に途中経過を報告する間隔")
    parser.add_argument("--study-name", default=None, help="同じ名前を指定すると続きから再開する（既定 cli-<preset>）")
    parser.add_argument("--vehicles", type=int, default=8)
    parser.add_argument("--pedestrians", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", choices=("checkpoint", "fresh"), default="checkpoint", help="各試行の出発点の重み")
    parser.add_argument("--no-prune", action="store_true", help="見込みの薄い試行を途中で打ち切らない")
    parser.add_argument("--timeout", type=float, default=None, help="この秒数を過ぎたら新しい試行を始めない")
    parser.add_argument("--tuning-dir", type=Path, default=config.TUNING_DIR, help="履歴の DB・最良のパラメータ・試行の一覧の置き場")
    parser.add_argument("--best-policy", type=Path, default=config.BEST_TUNED_POLICY_PATH, help="最良の試行の重みの保存先")
    parser.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT_PATH, help="--init checkpoint で読む重み（書き換えない）")
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not autotune.optuna_available():
        print(autotune.OPTUNA_MISSING_MESSAGE)
        return 1

    tuning_dir = Path(args.tuning_dir)
    db_path = tuning_dir / autotune.DB_PATH.name
    best_params_path = tuning_dir / autotune.BEST_PARAMS_PATH.name
    summary_path = tuning_dir / autotune.TRIAL_SUMMARY_PATH.name
    study_name = args.study_name or f"cli-{args.preset}"

    params = SimParams(vehicle_count=int(args.vehicles), pedestrian_count=int(args.pedestrians))
    print(f"地図を読み込みます（{args.preset}）")
    env = SimulationEnv(load_index(args.preset), params, seed=int(args.seed))

    hidden = None
    checkpoint = Path(args.checkpoint)
    if args.init == "checkpoint" and checkpoint.exists():
        peeked = peek_hidden_sizes(checkpoint)
        hidden = tuple(peeked) if peeked else None
    trainer = PPOTrainer(
        obs_dim=config.OBS_DIM,
        action_dim=config.ACTION_DIM,
        params=params,
        num_agents=config.MAX_VEHICLES,
        seed=int(args.seed),
        hidden_sizes=hidden,
    )
    if args.init == "checkpoint":
        if trainer.load(checkpoint):
            print(f"出発点の重み: {checkpoint.name}（更新回数 {trainer.updates}）")
        else:
            print(f"出発点の重み: {checkpoint.name} を読めないため、初期値から始めます")
    else:
        print("出発点の重み: 初期値")
    if not autotune.is_policy_healthy(trainer.policy):
        print("出発点の重みに NaN / Inf が混じっています。--init fresh で始めるか、重みを直してください")
        return 1

    study, storage = autotune.open_study(db_path, study_name, seed=int(args.seed), prune=not args.no_prune)
    tuner = HeadlessTuner(
        env,
        trainer,
        params,
        trial_steps=int(args.trial_steps),
        report_every=int(args.report_every),
        best_policy_path=Path(args.best_policy),
        best_params_path=best_params_path,
        summary_path=summary_path,
    )
    prior = autotune.completed_trials(study)
    print(
        f"探索を始めます（study {study_name} / {int(args.trials)} 試行 × {int(args.trial_steps)} ステップ"
        + (f" / 前回までの完了 {prior} 件から再開" if prior else "")
        + "）"
    )
    if prior:
        print("  ※ 最良の比べ直しはこの実行の中だけで行います（前回の最良の重みとは出発点が違うことがあるため）")

    interrupted = False
    try:
        study.optimize(
            tuner.objective,
            n_trials=int(args.trials),
            timeout=args.timeout,
            callbacks=[tuner.after_trial],
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n中断しました。同じ --study-name を指定すると続きから再開できます")
    finally:
        try:
            autotune.write_trial_summary(study, summary_path)
        except Exception:
            logger.exception("試行の一覧（CSV）を書き出せませんでした")
        autotune.close_storage(storage)

    best = tuner.best
    if best is None:
        print("この実行では完了した試行がありませんでした")
    else:
        print(f"\n最良: 試行 #{best.number}（スコア {best.score.value:+.3f}）")
        for d in autotune.SEARCH_SPACE:
            print(f"  {d.wire:<16s} {best.patch[d.wire]:.6g}")
        print(f"  重み: {Path(args.best_policy)}")
        print(f"  パラメータ: {best_params_path}")
    print(f"  試行の一覧: {summary_path}")
    print(f"  可視化: optuna-dashboard sqlite:///{db_path.resolve().as_posix()}")
    return 130 if interrupted else 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return run()


if __name__ == "__main__":
    sys.exit(main())
