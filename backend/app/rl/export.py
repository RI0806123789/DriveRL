"""学習済みモデルの書き出し。"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from app import config
from app.rl.policy import ActorCritic
from app.rl.ppo import CHECKPOINT_FORMAT

__all__ = [
    "ExportResult",
    "ExportError",
    "export_model",
    "preload_keras",
    "prune_exports",
    "EXPORT_KINDS",
    "MAX_EXPORT_FILES",
]

logger = logging.getLogger(__name__)

EXPORT_KINDS = ("checkpoint", "torchscript", "keras")

METADATA_VERSION = 2

MAX_EXPORT_FILES = 20


class ExportError(RuntimeError):
    """書き出しに失敗したときに投げる。"""


@dataclass
class ExportResult:
    path: Path
    filename: str
    size_bytes: int
    media_type: str
    kind: str


class InferencePolicy(nn.Module):
    """観測から決定論的な行動を出すだけのモジュール。"""

    def __init__(self, policy: ActorCritic) -> None:
        super().__init__()
        self.policy_trunk = copy.deepcopy(policy.policy_trunk)
        self.mu_head = copy.deepcopy(policy.mu_head)
        self.value_trunk = copy.deepcopy(policy.value_trunk)
        self.value_head = copy.deepcopy(policy.value_head)
        log_std = torch.clamp(
            policy.log_std.detach().clone(),
            float(config.PPO_LOG_STD_MIN),
            float(config.PPO_LOG_STD_MAX),
        )
        self.register_buffer("log_std", log_std)

        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: obs (B, obs_dim) -> Returns: (action (B, action_dim), value (B,))"""
        mu = self.mu_head(self.policy_trunk(obs))
        action = torch.clamp(mu, -1.0, 1.0)
        value = self.value_head(self.value_trunk(obs)).squeeze(-1)
        return action, value


def _observation_layout() -> list[dict[str, Any]]:
    """観測ベクトルの構成。app/percep/encoder.py の連結順と一致していること。"""
    return [
        {
            "name": "self",
            "size": config.OBS_SELF_DIM,
            "source": "vehicle_sensor",
            "description": "速度計と舵角センサー。"
            f"[速度 / max_speed, 舵角 / {config.MAX_STEER}rad]",
        },
        {
            "name": "goal",
            "size": config.OBS_GOAL_DIM,
            "source": "navigation",
            "description": "ナビが与える目的地の自車座標系相対位置 "
            f"(dx, dy) / {config.OBS_GOAL_RANGE}m と正規化した距離。"
            "カメラでは原理的に得られないので真値を使う",
        },
        {
            "name": "route",
            "size": config.OBS_ROUTE_DIM,
            "source": "navigation",
            "description": f"ナビの経路案内点 {config.OBS_ROUTE_POINTS} 個を "
            f"{config.OBS_ROUTE_SPACING}m 間隔でサンプルし、自車座標系"
            "（前方 +x / 左 +y）へ変換して正規化したもの",
        },
        {
            "name": "lane",
            "size": config.OBS_LANE_DIM,
            "source": "camera",
            "description": "認識した走行車線。[車線中心からの横方向偏差 / "
            f"{config.OBS_LATERAL_RANGE}m, 車線方向とのずれ sin, cos, 信頼度]。"
            "検出できなければ信頼度 0",
        },
        {
            "name": "signal",
            "size": config.OBS_SIGNAL_DIM,
            "source": "camera",
            "description": "認識した前方の信号。[停止線までの推定距離 / "
            f"{config.OBS_SIGNAL_RANGE}m, 青, 黄, 赤, 信頼度]。"
            "検出できなければ距離 1.0・灯色フラグはすべて 0・信頼度 0。"
            "**「信号が無い」と「青」は別物**",
        },
        {
            "name": "sign",
            "size": config.OBS_SIGN_DIM,
            "source": "camera",
            "description": "認識した最高速度標識（道交法 22 条）。"
            "[規制速度 / max_speed, (speed - 規制速度) / max_speed, 信頼度]。"
            "2 番目は正なら超過。標識を検出できていない間は "
            "規制速度 = max_speed として扱う",
        },
        {
            "name": "vehicles",
            "size": config.OBS_VEHICLE_DIM,
            "source": "camera",
            "description": f"認識した前方車両 {config.OBS_VEHICLE_COUNT} 台について "
            f"(dx, dy) / {config.OBS_VEHICLE_RANGE}m、推定距離、信頼度。"
            "距離は既知の車幅とバウンディングボックスの大きさから推定した単眼測距",
        },
        {
            "name": "obstacles",
            "size": config.OBS_OBSTACLE_DIM,
            "source": "camera",
            "description": f"認識した障害物 {config.OBS_OBSTACLE_COUNT} 個について "
            f"(dx, dy) / {config.OBS_OBSTACLE_RANGE}m と信頼度",
        },
        {
            "name": "pedestrians",
            "size": config.OBS_PEDESTRIAN_DIM,
            "source": "camera",
            "description": f"認識した歩行者 {config.OBS_PEDESTRIAN_COUNT} 人について "
            f"(dx, dy) / {config.OBS_PEDESTRIAN_RANGE}m と信頼度。"
            "距離は既知の肩幅とバウンディングボックスの大きさから推定した単眼測距",
        },
        {
            "name": "freespace",
            "size": config.OBS_FREESPACE_DIM,
            "source": "camera",
            "description": "認識した走行可能領域。自車 heading を中心に ±90 度を "
            f"{config.OBS_FREESPACE_DIM} 方向に等分し、各方向へ進める距離 / "
            f"{config.OBS_FREESPACE_MAX_DISTANCE}m。"
            "以前の建物レイキャストに相当するが、値は画像からの推定",
        },
    ]


def build_metadata(
    trainer: Any,
    *,
    kind: str,
    preset_id: str | None,
    preset_name: str | None,
    metrics: dict[str, Any] | None,
    params: Any | None = None,
) -> dict[str, Any]:
    """書き出しに同梱するメタデータを組み立てる。"""
    max_speed = float(getattr(params, "max_speed", config.MAX_SPEED))
    return {
        "metadataVersion": METADATA_VERSION,
        "application": "DriveRL",
        "kind": kind,
        "exportedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "torchVersion": str(torch.__version__),
        "model": {
            "type": "shared ActorCritic (parameter sharing)",
            "obsDim": int(trainer.obs_dim),
            "actionDim": int(trainer.action_dim),
            "hiddenSizes": [int(h) for h in trainer.policy.hidden_sizes],
            "activation": "tanh",
        },
        "observation": {
            "dim": int(trainer.obs_dim),
            "dtype": "float32",
            "note": "すべての要素は概ね [-1, 1] に正規化済み。下の layout の順に連結されている。",
            "layout": _observation_layout(),
        },
        "action": {
            "dim": int(trainer.action_dim),
            "range": [-1.0, 1.0],
            "fields": [
                {
                    "name": "accel",
                    "index": 0,
                    "description": "加減速指令。正なら加速、負なら減速",
                    "scaleToPhysical": {
                        "positive": f"accel * {config.MAX_ACCEL} [m/s^2]",
                        "negative": f"accel * {abs(config.MAX_DECEL)} [m/s^2]",
                    },
                },
                {
                    "name": "steer",
                    "index": 1,
                    "description": "操舵指令（前輪目標舵角）",
                    "scaleToPhysical": f"steer * {config.MAX_STEER} [rad]",
                },
            ],
            "note": "TorchScript 版と Keras 版はいずれも分布の平均を [-1, 1] にクリップした"
            "決定論的な行動を返す。学習時と同じ確率的な行動が欲しい場合は "
            "policy.logStd を使って Normal(action, exp(logStd)) からサンプリングすること。",
        },
        "policy": {
            "logStd": [
                float(
                    min(
                        max(float(v), float(config.PPO_LOG_STD_MIN)),
                        float(config.PPO_LOG_STD_MAX),
                    )
                )
                for v in trainer.policy.log_std.detach().cpu().numpy()
            ],
            "logStdRange": [
                float(config.PPO_LOG_STD_MIN),
                float(config.PPO_LOG_STD_MAX),
            ],
        },
        "vehicle": {
            "length": config.VEHICLE_LENGTH,
            "width": config.VEHICLE_WIDTH,
            "wheelbase": config.WHEELBASE,
            "maxSteer": config.MAX_STEER,
            "maxAccel": config.MAX_ACCEL,
            "maxDecel": config.MAX_DECEL,
            "dt": config.DT,
            "maxSpeed": max_speed,
            "steerRate": config.STEER_RATE,
            "maxLateralAccel": config.MAX_LATERAL_ACCEL,
        },
        "training": {
            "updates": int(trainer.updates),
            "gamma": float(trainer.gamma),
            "clipRange": float(trainer.clip_range),
            "entropyCoef": float(trainer.entropy_coef),
            "learningRate": float(trainer.learning_rate),
            "rolloutLength": int(trainer.rollout_length),
            "numAgentSlots": int(trainer.num_agents),
        },
        "map": {"presetId": preset_id, "presetName": preset_name},
        "metrics": metrics or {},
    }


def preload_keras() -> None:
    """Keras を先に読み込んでおく。"""
    _import_keras()


def _import_keras():
    """Keras 3 を torch バックエンドで読み込む。"""
    os.environ.setdefault("KERAS_BACKEND", "torch")
    try:
        import keras  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 依存が無い環境向け
        raise ExportError(
            "Keras 形式で書き出すには keras が必要です。"
            "`pip install -r requirements.txt` で導入してください"
        ) from exc
    return keras


def _build_keras_model(trainer: Any):
    """PyTorch の ActorCritic と同じ構造の Keras モデルを作り、重みを移す。"""
    keras = _import_keras()

    policy = trainer.policy
    hidden = [int(h) for h in policy.hidden_sizes]
    obs_dim = int(policy.obs_dim)
    action_dim = int(policy.action_dim)
    state = policy.state_dict()

    inputs = keras.Input(shape=(obs_dim,), dtype="float32", name="observation")

    def trunk(x, prefix: str):
        for i, units in enumerate(hidden):
            x = keras.layers.Dense(units, activation="tanh", name=f"{prefix}_dense_{i}")(x)
        return x

    action = keras.layers.Dense(
        action_dim, activation="hard_tanh", name="action"
    )(trunk(inputs, "policy"))
    value_dense = keras.layers.Dense(1, activation=None, name="value")(trunk(inputs, "value"))
    value = keras.layers.Reshape((), name="value_squeezed")(value_dense)

    model = keras.Model(
        inputs=inputs, outputs=[action, value], name="autoware_sim_policy"
    )

    def weight(name: str):
        return state[f"{name}.weight"].detach().cpu().numpy()

    def bias(name: str):
        return state[f"{name}.bias"].detach().cpu().numpy()

    for i in range(len(hidden)):
        for prefix, source in (("policy", "policy_trunk"), ("value", "value_trunk")):
            model.get_layer(f"{prefix}_dense_{i}").set_weights(
                [weight(f"{source}.{2 * i}").T, bias(f"{source}.{2 * i}")]
            )
    model.get_layer("action").set_weights([weight("mu_head").T, bias("mu_head")])
    model.get_layer("value").set_weights([weight("value_head").T, bias("value_head")])

    return model


def _attach_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """.keras（zip）へ独自のメタデータを追記する。"""
    with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            KERAS_METADATA_ENTRY,
            json.dumps(metadata, ensure_ascii=False, indent=2),
        )


KERAS_METADATA_ENTRY = "autoware_sim_metadata.json"


def _safe_slug(text: str | None, fallback: str = "nomap") -> str:
    """ファイル名に使える文字だけに落とす。"""
    if not text:
        return fallback
    slug = re.sub(r"[^0-9A-Za-z_-]+", "-", text).strip("-")
    return slug or fallback


def _atomic_save(write: Any, path: Path) -> None:
    """一時ファイルへ書いてから差し替える。中断しても壊れたファイルを残さない。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    write(tmp_path)
    os.replace(tmp_path, path)


def prune_exports(
    directory: Path | None = None,
    *,
    keep: int = MAX_EXPORT_FILES,
    protect: Path | None = None,
) -> list[str]:
    """書き出しディレクトリを新しい順に `keep` 件だけ残し、古いものを消す。"""
    directory = Path(directory) if directory is not None else config.EXPORT_DIR
    if not directory.is_dir():
        return []

    keep = max(1, int(keep))
    protect_name = protect.name if protect is not None else None

    entries: list[tuple[float, Path]] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.name.endswith(".tmp") or path.name.endswith(".partial.keras"):
            continue
        try:
            entries.append((path.stat().st_mtime, path))
        except OSError:
            continue

    entries.sort(key=lambda item: item[0], reverse=True)
    removed: list[str] = []
    for _mtime, path in entries[keep:]:
        if path.name == protect_name:
            continue
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("古い書き出しを削除できませんでした: %s（%s）", path.name, exc)
            continue
        removed.append(path.name)

    if removed:
        logger.info(
            "古い書き出しを %d 件削除しました（最新 %d 件を保持）", len(removed), keep
        )
    return removed


def export_model(
    trainer: Any,
    kind: str,
    *,
    out_dir: Path | None = None,
    preset_id: str | None = None,
    preset_name: str | None = None,
    metrics: dict[str, Any] | None = None,
    label: str | None = None,
    params: Any | None = None,
) -> ExportResult:
    """モデルを書き出してファイルの情報を返す。"""
    if kind not in EXPORT_KINDS:
        raise ExportError(f"未知の書き出し形式です: {kind}")

    directory = Path(out_dir) if out_dir is not None else config.EXPORT_DIR
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"_{_safe_slug(label, 'x')}" if label else ""
    base = f"autoware-sim_{_safe_slug(preset_id)}_upd{int(trainer.updates)}{tag}_{stamp}"

    metadata = build_metadata(
        trainer,
        kind=kind,
        preset_id=preset_id,
        preset_name=preset_name,
        metrics=metrics,
        params=params,
    )

    try:
        if kind == "checkpoint":
            path = directory / f"{base}.pt"
            payload = {
                "format": CHECKPOINT_FORMAT,
                "obs_dim": int(trainer.obs_dim),
                "action_dim": int(trainer.action_dim),
                "hidden_sizes": [int(h) for h in trainer.policy.hidden_sizes],
                "updates": int(trainer.updates),
                "policy": trainer.policy.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "metadata": metadata,
            }
            _atomic_save(lambda p: torch.save(payload, p), path)
            media_type = "application/octet-stream"

        elif kind == "keras":
            path = directory / f"{base}.keras"
            model = _build_keras_model(trainer)
            tmp_path = path.with_name(path.stem + ".partial.keras")
            model.save(tmp_path)
            _attach_metadata(tmp_path, metadata)
            os.replace(tmp_path, path)
            media_type = "application/octet-stream"

        else:
            path = directory / f"{base}.torchscript.pt"
            module = InferencePolicy(trainer.policy)
            try:
                scripted = torch.jit.script(module)
            except Exception:
                example = torch.zeros(1, int(trainer.obs_dim), dtype=torch.float32)
                scripted = torch.jit.trace(module, example, check_trace=False)

            extra = {"metadata.json": json.dumps(metadata, ensure_ascii=False, indent=2)}
            _atomic_save(lambda p: torch.jit.save(scripted, str(p), _extra_files=extra), path)
            media_type = "application/octet-stream"

    except ExportError:
        raise
    except Exception as exc:  # noqa: BLE001 - 失敗理由をそのまま画面に出したい
        raise ExportError(f"モデルの書き出しに失敗しました: {exc}") from exc

    if directory == config.EXPORT_DIR:
        try:
            prune_exports(directory, protect=path)
        except Exception:  # noqa: BLE001
            logger.exception("古い書き出しの整理に失敗しました（書き出し自体は成功しています）")

    return ExportResult(
        path=path,
        filename=path.name,
        size_bytes=path.stat().st_size,
        media_type=media_type,
        kind=kind,
    )
