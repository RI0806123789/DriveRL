"""同じ重みを複数の交通・物理・天候条件で評価する CLI。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch

from app import config
from app.contracts import SimParams
from app.map import build_map_index, get_preset, load_map
from app.map.loader import CACHE_VERSION, cache_path_for
from app.percep.groundtruth import clear_static_cache
from app.percep.weather import PRESETS as WEATHER_PRESETS
from app.rl.hierarchical_policy import OptionScheduler
from app.rl.ppo import PPOTrainer, peek_hidden_sizes
from app.rl.online_assist import OnlineAssistController
from app.runtime.learning_step import learning_step
from app.sim.env import SimulationEnv
from app.sim.evaluation import EvaluationMetrics
from app.sim.scenario import Scenario, load_scenario
from app.sim.vehicle import VehicleFleet
from tune_hyperparams import synthetic_grid


def load_index(preset_id: str):
    if preset_id == "grid":
        return build_map_index(synthetic_grid())
    preset = get_preset(preset_id)
    if preset is None:
        raise ValueError(f"未知のプリセットです: {preset_id}")
    path = cache_path_for(preset)
    if not path.exists():
        raise ValueError(f"{preset_id} のマップを先に app.map.prefetch で取得してください")
    with path.open(encoding="utf-8") as stream:
        head = stream.read(64)
    if not head.startswith(f'{{"version":{CACHE_VERSION},'):
        raise ValueError(f"{preset_id} のキャッシュが古いため app.map.prefetch で更新してください")
    return build_map_index(load_map(preset))


def make_env(index, scenario: Scenario, weather: str, seed: int, max_speed: float) -> SimulationEnv:
    rain = WEATHER_PRESETS[weather]
    params = SimParams(
        vehicle_count=scenario.learner_vehicles + scenario.background_vehicles,
        pedestrian_count=16, max_speed=max_speed, weather_auto=False,
        weather_rain=rain.rain, weather_fog=rain.fog,
        online_assist=False, incident_curriculum=False, safety_assist=False,
    )
    clear_static_cache()
    return SimulationEnv(index, params, seed=seed, scenario=scenario)


def evaluate(index, trainer: PPOTrainer, scenario: Scenario, weather: str, seed: int, steps: int, max_speed: float) -> dict:
    env = make_env(index, scenario, weather, seed, max_speed)
    learners = np.zeros(config.MAX_VEHICLES, dtype=bool)
    learners[:scenario.learner_vehicles] = True
    scheduler = OptionScheduler(config.MAX_VEHICLES)
    metrics = EvaluationMetrics()
    durations = []
    policy = trainer.policy
    policy.eval()
    for _ in range(steps):
        active = env.active_mask & learners
        obs = torch.from_numpy(env.observations)
        with torch.no_grad():
            due = scheduler.due(active)
            if due.any():
                scheduler.assign(due, policy.greedy_options(obs).numpy())
            actions = policy.mean_action(obs, torch.from_numpy(scheduler.options.copy())).numpy()
        actions[~active] = 0.0
        scheduler.tick(active)
        started = time.perf_counter()
        result = env.step(actions)
        durations.append((time.perf_counter() - started) * 1000.0)
        metrics.observe(result, learners)
        scheduler.restart(result.dones)
    output = metrics.to_dict()
    output.update({
        "scenario": scenario.name, "model": "dynamic" if scenario.dynamics else "kinematic",
        "weather": weather, "seed": seed, "steps": steps,
        "max_speed_m_s": env.params.max_speed, "pedestrian_count": env.params.pedestrian_count,
        "obey_signals": env.params.obey_signals, "obey_speed_signs": env.params.obey_speed_signs,
        "sim_seconds": steps * config.DT,
        "unfinished_episodes": int((env.active_mask & learners).sum()),
        "step_ms_median": float(np.median(durations)), "step_ms_p99": float(np.percentile(durations, 99)),
        "cnn_active": env.detector_active,
        "background_active": int((env.active_mask & env.traffic.mask).sum()),
        "signal_sources": dict(Counter(s.source for s in index.data.signals)),
    })
    return output


def train(index, trainer: PPOTrainer, scenario: Scenario, weather: str, seed: int, steps: int, max_speed: float) -> None:
    """比較条件ごとに同じ初期重みから学習し、本番の保存先には書かない。"""
    torch.manual_seed(seed)
    env = make_env(index, scenario, weather, seed, max_speed)
    trainer.apply_params(env.params)
    assist = OnlineAssistController(config.MAX_VEHICLES, seed=seed)
    trainer.policy.train()
    for _ in range(steps):
        learning_step(env, trainer, assist)


def maneuvers(scenario: Scenario) -> dict:
    fleet = VehicleFleet(1, dynamics=scenario.dynamics)
    fleet.active[:] = True
    fleet.speed[:] = 20.0
    for i in range(2000):
        fleet.step(np.array([-1.0]), np.zeros(1), config.DT, 30.0)
        if abs(float(fleet.speed[0])) < 0.01:
            break
    brake = {"distance_m": float(fleet.x[0]), "seconds": (i + 1) * config.DT, "stopped": abs(float(fleet.speed[0])) < 0.01}
    fleet.reset_slot(0, 0.0, 0.0, 0.0)
    fleet.speed[:] = 20.0
    slip = 0.0
    lateral = 0.0
    for i in range(200):
        steer = 0.5 * np.sin(i * config.DT * np.pi)
        fleet.step(np.zeros(1), np.array([steer]), config.DT, 30.0)
        slip = max(slip, abs(float(fleet.lateral_speed[0])))
        lateral = max(lateral, abs(float(fleet.lateral_accel[0])))
    return {"scenario": scenario.name, "initial_speed_m_s": 20.0, "braking": brake, "slalom_peak_lateral_speed_m_s": slip, "dynamic_peak_lateral_accel_m_s2": lateral if scenario.dynamics else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT_PATH)
    parser.add_argument("--scenarios", type=Path, nargs="+", required=True)
    parser.add_argument("--presets", nargs="+", default=["ginza"])
    parser.add_argument("--weather", nargs="+", choices=list(WEATHER_PRESETS), default=["clear"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--train-steps", type=int, default=0, help="条件ごとに初期重みを巻き戻して指定ステップ学習してから評価する")
    parser.add_argument("--max-speed", type=float, default=config.MAX_SPEED)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--maneuvers", action="store_true", help="20m/s からの急制動と急操舵も比較する")
    args = parser.parse_args()
    if args.steps <= 0 or args.train_steps < 0 or not np.isfinite(args.max_speed) or not 0 < args.max_speed <= 40 or any(seed < 0 for seed in args.seeds):
        parser.error("steps は正整数、train-steps / seeds は非負整数、max-speed は 0 より大きく 40m/s 以下にしてください")
    if args.output is not None and args.output.exists():
        parser.error("output は既存ファイルを上書きしない別の名前にしてください")
    try:
        scenarios = [load_scenario(path) for path in args.scenarios]
        trainer = PPOTrainer(config.OBS_DIM, config.ACTION_DIM, params=SimParams(), num_agents=config.MAX_VEHICLES, hidden_sizes=peek_hidden_sizes(args.checkpoint))
        if not trainer.load(args.checkpoint):
            raise ValueError("チェックポイントを読み込めませんでした")
        baseline = trainer.snapshot_state()
        report = {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "policy": "greedy_hierarchical", "metric_definition": "率は完了した学習車のエピソード数が分母。未完了なら null。平均速度は停止を含む車両ステップ平均。", "scenario_settings": [asdict(s) for s in scenarios], "results": [], "maneuvers": []}
        for preset in args.presets:
            index = load_index(preset)
            for scenario in scenarios:
                for weather in args.weather:
                    for seed in args.seeds:
                        trainer.restore_state(baseline)
                        if args.train_steps:
                            train(index, trainer, scenario, weather, seed, args.train_steps, args.max_speed)
                        result = evaluate(index, trainer, scenario, weather, seed, args.steps, args.max_speed)
                        result["train_steps"] = args.train_steps
                        result["training_updates"] = trainer.updates - baseline["updates"]
                        result["preset"] = preset
                        report["results"].append(result)
                        print(json.dumps(result, ensure_ascii=False))
            clear_static_cache()
        if args.maneuvers:
            report["maneuvers"] = [maneuvers(scenario) for scenario in scenarios]
            print(json.dumps(report["maneuvers"], ensure_ascii=False))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
