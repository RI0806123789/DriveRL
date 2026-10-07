"""通常の実行時には計測を載せず、独立した環境で学習速度を測る。"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

from app import config
from app.contracts import SimParams
from app.percep.types import DEFAULT_CAMERA
from app.percep.weather import PRESETS
from app.sim.env import RESPAWN_BUDGET_SEC, SimulationEnv


def timing_summary(values: list[float]) -> dict:
    """ミリ秒の分布を返す。未計測は成功率と取り違えないよう None にする。"""
    if not values:
        return {"calls": 0, "total_ms": 0.0, "median_ms": None, "p95_ms": None, "p99_ms": None}
    return {
        "calls": len(values), "total_ms": float(np.sum(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


class PipelineProfiler:
    """ベンチマークのインスタンスだけに計測を取り付け、終了時に元へ戻す。"""

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)
        self._restore: list[tuple[Any, str, bool, Any]] = []

    def attach(self, target: Any, name: str, phase: str) -> None:
        original = getattr(target, name)
        owned = name in vars(target)
        stored = vars(target).get(name)

        @wraps(original)
        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.samples[phase].append((time.perf_counter() - started) * 1000.0)

        self._restore.append((target, name, owned, stored))
        setattr(target, name, timed)

    def attach_env(self, env: SimulationEnv) -> None:
        for target, name, phase in (
            (env, "step", "environment"),
            (env, "_compute_observations", "observation"),
            (env, "_update_occlusion", "occlusion"),
            (env, "_run_safety", "safety"),
            (env, "_drain_respawn_queue", "respawn"),
            (env.world, "project_all", "route_projection"),
            (env.world, "check_collisions", "collision_detection"),
        ):
            self.attach(target, name, phase)
        if env._camera is not None:
            self.attach(env._camera, "render", "camera_render")
        if env._detector is not None:
            self.attach(env._detector, "_forward", "cnn_forward")
            self.attach(env._detector, "detect_with_freespace", "cnn_forward_and_decode")

    def clear(self) -> None:
        self.samples.clear()

    def close(self) -> None:
        for target, name, owned, original in reversed(self._restore):
            if owned:
                setattr(target, name, original)
            else:
                delattr(target, name)
        self._restore.clear()

    def to_dict(self) -> dict:
        return {key: timing_summary(values) for key, values in sorted(self.samples.items())}


def benchmark_speed(
    index, params: SimParams, *, mode: str, detector=None, checkpoint: Path | None = None,
    seed: int = 0, steps: int = 256, warmup: int = 16, train: bool = True,
    publish_frame: bool = False, deterministic_respawn: bool = False,
) -> dict:
    """画面とサーバーを起動せず、観測を維持したまま実際の PPO 学習を計測する。"""
    from app.rl.online_assist import OnlineAssistController
    from app.rl.ppo import PPOTrainer, peek_hidden_sizes
    from app.runtime.learning_step import learning_step

    if mode not in ("oracle", "cnn") or steps <= 0 or warmup < 0 or seed < 0:
        raise ValueError("認識モード・ステップ数・乱数種が不正です")
    if not 1 <= params.vehicle_count <= config.MAX_VEHICLES:
        raise ValueError("台数は車両スロット数以内にしてください")
    trainer = PPOTrainer(
        config.OBS_DIM, config.ACTION_DIM, params, config.MAX_VEHICLES, seed=seed,
        hidden_sizes=peek_hidden_sizes(checkpoint) if checkpoint else None,
    )
    if checkpoint is not None and not trainer.load(checkpoint):
        raise ValueError("チェックポイントを読み込めませんでした")
    env = SimulationEnv(index, params, seed=seed, perception_mode=mode, detector=detector,
                        deterministic_respawn=deterministic_respawn)
    assist = OnlineAssistController(config.MAX_VEHICLES, seed=seed)
    profile = PipelineProfiler()
    profile.attach_env(env)
    for name, phase in (("act", "policy_action"), ("store", "rollout_store"), ("maybe_update", "ppo_update")):
        profile.attach(trainer, name, phase)
    durations: list[float] = []
    vehicle_steps = 0
    initial_updates = trainer.updates
    wall_started = cpu_started = 0.0
    frame_bytes = 0
    try:
        for step in range(warmup + steps):
            if step == warmup:
                profile.clear()
                initial_updates = trainer.updates
                wall_started = time.perf_counter()
                cpu_started = time.process_time()
            started = time.perf_counter()
            result, _ = learning_step(env, trainer, assist, learn=train)
            if publish_frame:
                frame_started = time.perf_counter()
                frame = env.snapshot(step, env.sim_time)
                encoded = json.dumps(frame.to_wire(), ensure_ascii=False, separators=(",", ":"))
                profile.samples["frame_serialization"].append((time.perf_counter() - frame_started) * 1000.0)
                if step >= warmup:
                    frame_bytes += len(encoded.encode("utf-8"))
                env.world.clear_route_dirty(frame.routed_slots)
            elapsed = (time.perf_counter() - started) * 1000.0
            if step >= warmup:
                durations.append(elapsed)
                vehicle_steps += int(result.active.sum())
        wall_sec = time.perf_counter() - wall_started
        cpu_sec = time.process_time() - cpu_started
    finally:
        profile.close()
    budget = config.DT * 1000.0
    phases = profile.to_dict()
    forward = phases.get("cnn_forward", {}).get("total_ms", 0.0)
    detected = phases.get("cnn_forward_and_decode", {}).get("total_ms", 0.0)
    environment = phases.get("environment", {}).get("total_ms", 0.0)
    observation = phases.get("observation", {}).get("total_ms", 0.0)
    return {
        "mode": mode, "detector_backend": env._detector.backend if env.detector_active else None,
        "vehicles": params.vehicle_count, "seed": seed, "steps": steps, "warmup": warmup,
        "training": train, "headless": not publish_frame, "frame_bytes": frame_bytes,
        "policy_initialization": "checkpoint" if checkpoint else "seeded_random",
        "step": timing_summary(durations), "budget_ms": budget,
        "budget_exceeded_steps": sum(value > budget for value in durations),
        "budget_exceeded_fraction": sum(value > budget for value in durations) / steps,
        "wall_sec": wall_sec, "process_cpu_sec": cpu_sec,
        "steps_per_sec": steps / wall_sec, "vehicle_steps_per_sec": vehicle_steps / wall_sec,
        "vehicle_steps": vehicle_steps, "sim_seconds_per_wall_second": steps * config.DT / wall_sec,
        "training_updates": trainer.updates - initial_updates,
        "ppo_update_pending": trainer._pending is not None,
        "respawn_budget_ms": None if deterministic_respawn else RESPAWN_BUDGET_SEC * 1000.0,
        "deterministic_respawn": deterministic_respawn,
        "phases": phases, "cnn_decode_total_ms": max(0.0, detected - forward),
        "physics_rewards_respawn_total_ms": max(0.0, environment - observation),
        "phase_definition": "各フェーズは入れ子を含むため合計しない。step は方策・物理・観測・更新・指定時の配信データ作成を含む。",
    }


def collect_image_samples(index, *, count: int, weather: str, seed: int = 0) -> list:
    """CNN 評価とは独立した固定の合成画像と正解ラベルを集める。"""
    from app.percep.benchmark import ImageSample
    from app.percep.camera import PseudoCamera
    from app.percep.groundtruth import detect_ground_truth_batch, freespace_ground_truth_views
    from app.percep.trainer import cluster_vehicles, scatter_props

    if count <= 0 or weather not in PRESETS or seed < 0:
        raise ValueError("画像数・天候・乱数種が不正です")
    condition = PRESETS[weather]
    params = SimParams(vehicle_count=config.MAX_VEHICLES, pedestrian_count=16, weather_auto=False,
                       weather_rain=condition.rain, weather_fog=condition.fog)
    env = SimulationEnv(index, params, seed=seed, compute_observations=False, deterministic_respawn=True)
    camera = PseudoCamera(index)
    rng = np.random.default_rng(seed)
    samples = []
    step = 0
    while len(samples) < count:
        if step % 12 == 0:
            if step:
                env.reset_all(relocate_walkers=False)
            cluster_vehicles(env, rng)
            scatter_props(env, rng)
        slots = np.flatnonzero(env.active_mask)
        if not len(slots):
            raise ValueError("評価画像を作る車両を配置できませんでした")
        images = camera.render(env.world, slots, condition, step)
        labels = detect_ground_truth_batch(env.world, slots, DEFAULT_CAMERA, condition)
        reach = min(config.OBS_FREESPACE_MAX_DISTANCE, condition.visibility_m(DEFAULT_CAMERA.far))
        frees = freespace_ground_truth_views(env.world, slots, (DEFAULT_CAMERA,), reach)[:, 0]
        for image, label, free in zip(images, labels, frees):
            if len(samples) == count:
                break
            samples.append(ImageSample(image, label, free, source="synthetic", sample_id=f"{weather}:{step}:{label.slot}", condition=weather))
        actions = rng.uniform(-0.2, 1.0, (config.MAX_VEHICLES, config.ACTION_DIM)).astype(np.float32)
        env.step(actions)
        step += 1
    return samples


def file_fingerprint(path: Path) -> str:
    """ローカルパスを出力せずモデルや入力ファイルの内容を識別する。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weather_params(weather: str, vehicles: int, pedestrians: int = 16, rollout_length: int = 128) -> SimParams:
    """速度計測と方策評価が共有する条件を作る。"""
    condition = PRESETS[weather]
    return replace(
        SimParams(), vehicle_count=vehicles, pedestrian_count=pedestrians, rollout_length=rollout_length,
        weather_auto=False, weather_rain=condition.rain, weather_fog=condition.fog,
        online_assist=False, incident_curriculum=False, safety_assist=False,
    )
