"""学習速度・認識器・運転方策を同一条件とモデルの指紋で比較する CLI。"""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import asdict, replace
from pathlib import Path

from app import config
from app.map import get_preset
from app.map.loader import CACHE_VERSION, cache_path_for
from app.percep.benchmark import DEFAULT_CONDITIONS, evaluate_images, load_external_dataset
from app.percep.detector import Detector
from app.percep.groundtruth import clear_static_cache
from app.percep.weather import PRESETS
from app.runtime.policy_benchmark import NoiseConfig, evaluate_policy, load_policy
from app.runtime.pipeline_benchmark import benchmark_speed, collect_image_samples, file_fingerprint, weather_params
from evaluate_scenarios import load_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("speed", "policy", "images"), default=["speed", "policy", "images"])
    parser.add_argument("--presets", nargs="+", default=["grid"], help="grid または取得済みの実地図。自動再取得しない")
    parser.add_argument("--vehicles", nargs="+", type=int, default=[1, 4, config.MAX_VEHICLES])
    parser.add_argument("--weather", nargs="+", choices=tuple(PRESETS), default=["clear", "rain", "fog"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--modes", nargs="+", choices=("oracle", "cnn", "noisy"), default=["oracle", "cnn", "noisy"])
    parser.add_argument("--detector", type=Path, default=config.DETECTOR_PATH)
    parser.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT_PATH)
    parser.add_argument("--device", choices=("auto", "cpu", "keras"), default="auto", help="CNN の推論先。GPU は既存の方針どおり選ばない")
    parser.add_argument("--steps", type=int, default=256, help="速度計測のステップ数")
    parser.add_argument("--policy-steps", type=int, default=4000, help="方策評価のステップ数")
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--pedestrians", type=int, default=16)
    parser.add_argument("--samples", type=int, default=64, help="各地図・天候・乱数種の合成画像数")
    parser.add_argument("--external", type=Path, help="外部 RGB の JSON manifest または NPZ")
    parser.add_argument("--external-only", action="store_true", help="地図を読まず外部 RGB だけ採点する（tasks images 専用）")
    parser.add_argument("--inference-only", action="store_true", help="速度計測時も PPO の重みを更新しない")
    parser.add_argument("--publish-frame", action="store_true", help="配信データ作成も計測する。ブラウザ描画時間は含まない")
    parser.add_argument("--evaluate-unconstrained", action="store_true", help="方策評価だけ真値の信号・規制速度制限を外す（報酬・終了判定は真値のまま）")
    parser.add_argument("--deterministic-respawn", action="store_true", help="速度計測も再スポーンの実時間予算を外して状態列を固定する（通常運転との比較には外す）")
    parser.add_argument("--noise-drop", type=float, default=0.1)
    parser.add_argument("--noise-distance", type=float, default=0.1)
    parser.add_argument("--noise-phase", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True, help="比較結果の JSON。既存ファイルは上書きしない")
    args = parser.parse_args(argv)
    if any(value <= 0 for value in (args.steps, args.policy_steps, args.rollout_length, args.samples)) or args.warmup < 0:
        parser.error("ステップ数・画像数・rollout-length は正整数、warmup は非負整数にしてください")
    if any(value < 0 for value in args.seeds) or any(not 1 <= value <= config.MAX_VEHICLES for value in args.vehicles):
        parser.error("seeds は非負整数、vehicles は車両スロット数以内にしてください")
    if not 0 <= args.pedestrians <= config.MAX_PEDESTRIANS:
        parser.error("pedestrians は歩行者の上限以内にしてください")
    if args.output.exists():
        parser.error("output は既存ファイルを上書きしない別の名前にしてください")
    if args.external_only and (set(args.tasks) != {"images"} or args.external is None):
        parser.error("external-only は tasks images と external を指定してください")
    if args.external is not None and "images" not in args.tasks:
        parser.error("external は tasks images と指定してください")
    if "speed" in args.tasks and not any(mode in ("oracle", "cnn") for mode in args.modes):
        parser.error("速度計測には modes oracle または cnn を指定してください")
    if os.getenv("DRIVERL_SCENARIO", "").strip():
        parser.error("条件を固定するため DRIVERL_SCENARIO を外して実行してください")
    try:
        noise = NoiseConfig(args.noise_drop, args.noise_distance, args.noise_phase)
        needs_cnn = "images" in args.tasks or ("cnn" in args.modes and bool(set(args.tasks) & {"speed", "policy"}))
        config.PERCEP_DEVICE = args.device
        detector = Detector.load(args.detector, accelerate=args.device != "keras") if needs_cnn else None
        if needs_cnn and detector is None:
            raise ValueError("読み込める認識器を指定してください。CNN 評価は真値で代用しません")
        policy = None
        policy_meta = None
        if "policy" in args.tasks:
            policy, policy_meta = load_policy(args.checkpoint)
        external = load_external_dataset(args.external) if args.external else None
        checkpoint = args.checkpoint if args.checkpoint.exists() else None
        report = {
            "schema_version": 1, "issue": 102,
            "platform": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine()},
            "detector_sha256": file_fingerprint(args.detector) if needs_cnn else None,
            "detector_backend": detector.backend if detector else None,
            "checkpoint_sha256": file_fingerprint(checkpoint) if checkpoint else None,
            "policy": policy_meta, "noise": asdict(noise),
            "conditions": [asdict(condition) for condition in DEFAULT_CONDITIONS],
            "settings": {"vehicles": args.vehicles, "weather": args.weather, "seeds": args.seeds,
                         "steps": args.steps, "policy_steps": args.policy_steps, "warmup": args.warmup,
                         "rollout_length": args.rollout_length, "pedestrians": args.pedestrians,
                         "max_vehicles": config.MAX_VEHICLES, "surround_images_per_step": config.SURROUND_CNN_IMAGES_PER_STEP},
            "maps": [], "speed": [], "driving": [], "perception": [],
            "evaluation_scope": "外部画像は認識器の評価。運転指標はシミュレータ内の方策評価で、実環境の運転性能を示すものではない。",
            "driving_metric_definition": "率の分母は完了エピソード。完了ゼロ件は null。mean_arrival_seconds は到達のみ、mean_completed_episode_seconds は全終了理由の平均。イベント/車両時間は未完了も含む。",
            "perception_comparison": "oracle/cnn/noisy は同じ周囲カメラの更新予算。noisy は検出だけを揺らし走行可能距離は維持する。oracle の速度は理想ラベル生成の参考値。",
            "traffic_constraints": "通常は真値の信号・規制速度の制約が有効。evaluate-unconstrained 指定時の方策評価だけ外す。各結果の params に記録する。",
        }
        for preset in ([] if args.external_only else args.presets):
            index = load_index(preset)
            report["maps"].append(
                {"preset": preset, "synthetic_grid": {"k": 5, "step_m": 120.0}}
                if preset == "grid" else
                {"preset": preset, "cache_version": CACHE_VERSION, "cache_sha256": file_fingerprint(cache_path_for(get_preset(preset)))}
            )
            for weather in args.weather:
                for seed in args.seeds:
                    for vehicles in args.vehicles:
                        params = weather_params(weather, vehicles, args.pedestrians, args.rollout_length)
                        if "speed" in args.tasks:
                            for mode in dict.fromkeys(m for m in args.modes if m != "noisy"):
                                result = benchmark_speed(index, params, mode=mode, detector=detector if mode == "cnn" else None,
                                                         checkpoint=checkpoint, seed=seed, steps=args.steps, warmup=args.warmup,
                                                         train=not args.inference_only, publish_frame=args.publish_frame,
                                                         deterministic_respawn=args.deterministic_respawn)
                                result.update(preset=preset, weather=weather, params=asdict(params))
                                report["speed"].append(result)
                                print(f"速度: {preset} / {weather} / {vehicles}台 / {mode}: 中央値 {result['step']['median_ms']:.2f}ms、50ms超過 {result['budget_exceeded_steps']}/{args.steps}")
                        if "policy" in args.tasks:
                            policy_params = replace(params, obey_signals=False, obey_speed_signs=False) if args.evaluate_unconstrained else params
                            for mode in args.modes:
                                result = evaluate_policy(index, policy, policy_params, mode=mode, seed=seed, steps=args.policy_steps,
                                                         detector=detector if mode == "cnn" else None, noise=noise)
                                result.update(preset=preset, weather=weather, vehicles=vehicles)
                                report["driving"].append(result)
                                print(f"方策: {preset} / {weather} / {vehicles}台 / {mode}: 評価完了")
                    if "images" in args.tasks:
                        samples = collect_image_samples(index, count=args.samples, weather=weather, seed=seed)
                        result = evaluate_images(detector, samples, seed=seed)
                        result.update(preset=preset, weather=weather, seed=seed)
                        report["perception"].append(result)
                        print(f"認識: {preset} / {weather}: {len(samples)}枚を採点")
            clear_static_cache()
        if external is not None:
            result = evaluate_images(detector, external, seed=args.seeds[0])
            result["dataset_sha256"] = file_fingerprint(args.external)
            report["perception"].append(result)
            print(f"外部認識: {len(external)}枚を採点")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    finally:
        clear_static_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
