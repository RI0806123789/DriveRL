"""同じ方策を理想・CNN・誤認識の観測で学習せずに比較する。"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from app import config
from app.contracts import MapIndex, SimParams, StepResult
from app.percep.groundtruth import clear_static_cache
from app.percep.types import CameraSpec, DetClass, PerceptionResult
from app.rl.hierarchical_policy import OptionScheduler
from app.rl.ppo import PPOTrainer, peek_hidden_sizes
from app.sim.env import SimulationEnv
from app.sim.evaluation import EvaluationMetrics
from app.sim.scenario import Scenario


@dataclass(frozen=True)
class NoiseConfig:
    """理想の検出にだけ加える欠落率・距離の相対標準偏差・信号現示の誤分類率。"""

    drop_probability: float = 0.1
    distance_std: float = 0.1
    phase_error_probability: float = 0.1

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0 or (name != "distance_std" and value > 1):
                raise ValueError(f"認識ノイズの設定が範囲外です: {name}")


class PerceptionNoise:
    """世界の乱数と真値には触れず、各カメラの新しい検出を複製して揺らす。"""

    def __init__(self, seed: int, settings: NoiseConfig) -> None:
        self.rng = np.random.default_rng([int(seed), 102])
        self.settings = settings

    def __call__(self, result: PerceptionResult, spec: CameraSpec) -> PerceptionResult:
        detections = []
        settings = self.settings
        for original in result.detections:
            if settings.drop_probability and self.rng.random() < settings.drop_probability:
                continue
            detection = replace(original, lane_points=None if original.lane_points is None else list(original.lane_points))
            if detection.distance is not None and settings.distance_std:
                detection.distance = max(0.0, float(detection.distance) * (1.0 + self.rng.normal(0.0, settings.distance_std)))
            if detection.cls == DetClass.TRAFFIC_LIGHT and detection.phase is not None and settings.phase_error_probability:
                if self.rng.random() < settings.phase_error_probability:
                    detection.phase = (int(detection.phase) + int(self.rng.integers(1, 3))) % 3
            detections.append(detection)
        return PerceptionResult(slot=result.slot, detections=detections)


def load_policy(checkpoint: Path) -> tuple[Any, dict[str, Any]]:
    """既存の安全な移行処理で方策を読むだけで、保存や学習は行わない。"""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise ValueError("方策のチェックポイントがありません")
    trainer = PPOTrainer(
        config.OBS_DIM, config.ACTION_DIM, SimParams(), config.MAX_VEHICLES,
        hidden_sizes=peek_hidden_sizes(checkpoint),
    )
    if not trainer.load(checkpoint):
        raise ValueError("方策のチェックポイントを安全に読み込めませんでした")
    trainer.policy.eval()
    return trainer.policy, {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "hidden_sizes": list(trainer.hidden_sizes),
        "training_updates": trainer.updates,
        "observation_dim": config.OBS_DIM,
        "widened_from": trainer.widened_from,
        "upgraded_flat": trainer.upgraded_flat,
        "action_selection": "greedy_hierarchical_mean",
    }


@dataclass
class PolicyMetrics:
    """完了したエピソードの率と評価打ち切り時の未完了分を分けて数える。"""

    episodes: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    completed_seconds: float = 0.0
    arrived_seconds: float = 0.0
    signal_events: int = 0
    lane_events: int = 0
    outcomes: Counter = field(default_factory=Counter)

    def observe(self, result: StepResult, learners: np.ndarray) -> None:
        self.episodes.observe(result, learners)
        for episode in result.episodes:
            if learners[episode.slot]:
                seconds = episode.length * config.DT
                self.completed_seconds += seconds
                self.arrived_seconds += seconds if episode.reason == "goal" else 0.0
                self.signal_events += episode.signal_violations
                self.lane_events += episode.lane_departures
                self.outcomes[episode.reason] += 1

    def to_dict(self, unfinished: list[tuple[int, int, int]]) -> dict[str, Any]:
        output = self.episodes.to_dict()
        completed = self.episodes.completed
        arrivals = self.outcomes["goal"]
        seconds = self.episodes.vehicle_steps * config.DT
        red_events = self.signal_events + sum(violations for _, violations, _ in unfinished)
        lane_events = self.lane_events + sum(departures for _, _, departures in unfinished)
        output.update({
            "offroad_rate": self.outcomes["offroad"] / completed if completed else None,
            "mean_completed_episode_seconds": self.completed_seconds / completed if completed else None,
            "mean_arrival_seconds": self.arrived_seconds / arrivals if arrivals else None,
            "unfinished_episodes": len(unfinished),
            "unfinished_episode_seconds": sum(steps for steps, _, _ in unfinished) * config.DT,
            "vehicle_seconds": seconds,
            "signal_violation_events": red_events,
            "lane_departure_events": lane_events,
            "signal_violations_per_vehicle_hour": red_events * 3600 / seconds if seconds else None,
            "lane_departures_per_vehicle_hour": lane_events * 3600 / seconds if seconds else None,
        })
        return output


def timing_summary(milliseconds: list[float]) -> dict[str, float | int]:
    """計測した区間の分布と 50ms 予算の超過率を返す。"""
    values = np.asarray(milliseconds, dtype=np.float64)
    if values.size == 0:
        return {"samples": 0}
    return {
        "samples": len(milliseconds),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
        "over_50ms_fraction": float(np.mean(values > 50.0)),
    }


def evaluate_policy(
    index: MapIndex,
    policy: Any,
    params: SimParams,
    *,
    mode: str,
    seed: int,
    steps: int,
    detector: Any = None,
    noise: NoiseConfig | None = None,
    scenario: Scenario | None = None,
    profiler: Callable[[SimulationEnv], Any] | None = None,
) -> dict[str, Any]:
    """同じ初期世界から平均操作で走らせ、認識の出所以外の条件を固定する。"""
    if mode not in {"oracle", "cnn", "noisy"} or seed < 0 or steps <= 0:
        raise ValueError("mode は oracle/cnn/noisy、seed は非負、steps は正整数にしてください")
    evaluation_params = replace(params, online_assist=False, incident_curriculum=False, safety_assist=False)
    noise = noise if noise is not None else NoiseConfig()
    clear_static_cache()
    env = SimulationEnv(
        index, evaluation_params, seed=seed, scenario=scenario,
        perception_mode="oracle" if mode == "noisy" else mode,
        detector=detector if mode == "cnn" else None,
        perception_transform=PerceptionNoise(seed, noise) if mode == "noisy" else None,
        deterministic_respawn=True,
    )
    measured = profiler(env) if profiler is not None else None
    learners = np.ones(config.MAX_VEHICLES, dtype=bool)
    if env.traffic is not None:
        learners &= ~env.traffic.mask
    scheduler = OptionScheduler(config.MAX_VEHICLES)
    metrics = PolicyMetrics()
    inference_ms: list[float] = []
    step_ms: list[float] = []
    total_ms: list[float] = []
    was_training = policy.training
    policy.eval()
    try:
        with torch.inference_mode():
            for _ in range(steps):
                started = time.perf_counter()
                active = env.active_mask & learners
                obs = torch.from_numpy(env.observations)
                due = scheduler.due(active)
                if due.any():
                    scheduler.assign(due, policy.greedy_options(obs).cpu().numpy())
                actions = policy.mean_action(obs, torch.from_numpy(scheduler.options.copy())).cpu().numpy()
                actions[~active] = 0.0
                scheduler.tick(active)
                inferred = time.perf_counter()
                result = env.step(actions)
                ended = time.perf_counter()
                inference_ms.append((inferred - started) * 1000)
                step_ms.append((ended - inferred) * 1000)
                total_ms.append((ended - started) * 1000)
                metrics.observe(result, learners)
                scheduler.restart(result.dones)
    finally:
        policy.train(was_training)
        if measured is not None and hasattr(measured, "close"):
            measured.close()
        clear_static_cache()
    unfinished = [
        (slot.steps, slot.violations, slot.lane_departures)
        for number, slot in enumerate(env.world.slots)
        if learners[number] and env.active_mask[number] and slot.steps > 0
    ]
    output = metrics.to_dict(unfinished)
    output.update({
        "mode": mode, "seed": seed, "steps": steps,
        "params": asdict(env.params),
        "sim_seconds": steps * config.DT,
        "cnn_active": env.detector_active,
        "headless": True,
        "learning_updates": 0,
        "noise": asdict(noise) if mode == "noisy" else None,
        "noise_scope": "detections_only_freespace_unchanged" if mode == "noisy" else None,
        "metric_definition": "エピソード率は完了数が分母で、完了なしは null。平均完了時間は全終了理由、平均到達時間は到達のみ。イベント数と車両時間は評価終了時の未完了分も含む。平均速度は停止を含む車両ステップ平均。",
        "timing": {"policy": timing_summary(inference_ms), "environment": timing_summary(step_ms), "total": timing_summary(total_ms)},
        "steps_per_second": 1000 / float(np.mean(total_ms)),
        "vehicle_steps_per_second": metrics.episodes.vehicle_steps / (sum(total_ms) / 1000),
    })
    if measured is not None:
        output["timing"]["stages"] = measured.to_dict()
    return output
