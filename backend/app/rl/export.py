"""学習済みモデルの書き出し。"""

from __future__ import annotations

import copy
import importlib
import io
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
from app.warn import warn_once

__all__ = [
    "ExportResult",
    "ExportError",
    "export_model",
    "preload_keras",
    "preload_torch_export",
    "prune_exports",
    "EXPORT_KINDS",
    "IMPORT_BACKUP_LABEL",
    "MAX_EXPORT_FILES",
    "MAX_IMPORT_BACKUPS",
]

logger = logging.getLogger(__name__)

EXPORT_KINDS = ("checkpoint", "torchscript", "pt2", "keras")

#: torch.export の初回に読み込まれる重いモジュール。HTTP 側で先に読み、エンジンスレッドでの初回の書き出しを軽くする
PT2_PRELOAD_MODULES = (
    "torch.export",
    "torch._dynamo",
    "sympy",
    "torch.fx.experimental.symbolic_shapes",
    "torch._subclasses.fake_tensor",
    "torch._functorch.aot_autograd",
)

METADATA_VERSION = 2

MAX_EXPORT_FILES = 20

#: 読み込む前の重みを退避するときの印（ファイル名に `_before-import_` として入る）
IMPORT_BACKUP_LABEL = "before-import"
#: 退避した重みは書き出しの世代（`MAX_EXPORT_FILES`）に数えず、退避どうしでこれだけ残す
MAX_IMPORT_BACKUPS = 10


#: Keras が無いときに画面へ出す文。例外の文をそのまま HTTP の応答へ載せないよう、main.py もこれを返す
KERAS_MISSING_MESSAGE = (
    "Keras 形式で書き出すには keras が必要です。`pip install -r requirements.txt` で導入してください"
)


class ExportError(RuntimeError):
    """書き出しに失敗したときに投げる。"""


@dataclass
class ExportResult:
    path: Path
    filename: str
    size_bytes: int
    media_type: str
    kind: str


#: Keras で上位方策のいちばん確率の高い意図を one-hot にするときの温度（ロジットをこれで割って softmax）
KERAS_OPTION_TEMPERATURE = 1e-8


class InferencePolicy(nn.Module):
    """観測から決定論的な行動を出すだけのモジュール。意図は呼ばれるたびに上位方策のいちばん確率の高いものを選ぶ。"""

    def __init__(self, policy: ActorCritic) -> None:
        super().__init__()
        self.num_options = int(policy.num_options)
        self.meta_trunk = copy.deepcopy(policy.meta_trunk)
        self.meta_head = copy.deepcopy(policy.meta_head)
        self.register_buffer("option_bias", policy.option_bias.detach().clone())
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
        option = self.meta_head(self.meta_trunk(obs)).argmax(dim=-1)
        one_hot = torch.nn.functional.one_hot(option, self.num_options).to(obs.dtype)
        x = torch.cat([obs, one_hot], dim=-1)
        action = torch.tanh(self.mu_head(self.policy_trunk(x)) + one_hot @ self.option_bias)
        value = self.value_head(self.value_trunk(x)).squeeze(-1)
        return action, value


def _layout_notes() -> dict[str, tuple[str, str]]:
    """観測の区画ごとの (出どころ, 説明)。並びと大きさは `config.OBS_LAYOUT` が持つ。"""
    return {
        "self": (
            "vehicle_sensor",
            "速度計と舵角センサー。"
            f"[速度 / max_speed, 舵角 / {config.MAX_STEER}rad]",
        ),
        "goal": (
            "navigation",
            "ナビが与える目的地の自車座標系相対位置 "
            f"(dx, dy) / {config.OBS_GOAL_RANGE}m と正規化した距離。"
            "カメラでは原理的に得られないので真値を使う",
        ),
        "route": (
            "navigation",
            f"ナビの経路案内点 {config.OBS_ROUTE_POINTS} 個を "
            f"{config.OBS_ROUTE_SPACING}m 間隔でサンプルし、自車座標系"
            "（前方 +x / 左 +y）へ変換して正規化したもの",
        ),
        "lane": (
            "camera",
            "認識した走行車線。[車線中心からの横方向偏差 / "
            f"{config.OBS_LATERAL_RANGE}m, 車線方向とのずれ sin, cos, 信頼度]。"
            "検出できなければ信頼度 0",
        ),
        "signal": (
            "camera",
            "認識した前方の信号。[停止線までの推定距離 / "
            f"{config.OBS_SIGNAL_RANGE}m, 青, 黄, 赤, 信頼度]。"
            "検出できなければ距離 1.0・灯色フラグはすべて 0・信頼度 0。"
            "**「信号が無い」と「青」は別物**",
        ),
        "sign": (
            "camera",
            "認識した最高速度標識（道交法 22 条）。"
            "[規制速度 / max_speed, (speed - 規制速度) / max_speed, 信頼度]。"
            "2 番目は正なら超過。標識を検出できていない間は "
            "規制速度 = max_speed として扱う",
        ),
        "vehicles": (
            "camera",
            f"認識した前方車両 {config.OBS_VEHICLE_COUNT} 台について "
            f"(dx, dy) / {config.OBS_VEHICLE_RANGE}m、推定距離、信頼度。"
            "距離は既知の車幅とバウンディングボックスの大きさから推定した単眼測距",
        ),
        "obstacles": (
            "camera",
            f"認識した障害物 {config.OBS_OBSTACLE_COUNT} 個について "
            f"(dx, dy) / {config.OBS_OBSTACLE_RANGE}m と信頼度",
        ),
        "pedestrians": (
            "camera",
            f"認識した歩行者 {config.OBS_PEDESTRIAN_COUNT} 人について "
            f"(dx, dy) / {config.OBS_PEDESTRIAN_RANGE}m と信頼度。"
            "距離は既知の肩幅とバウンディングボックスの大きさから推定した単眼測距",
        ),
        "freespace": (
            "camera",
            "認識した走行可能領域。自車 heading を中心に ±90 度を "
            f"{config.OBS_FREESPACE_DIM} 方向に等分し、各方向へ進める距離 / "
            f"{config.OBS_FREESPACE_MAX_DISTANCE}m。"
            "以前の建物レイキャストに相当するが、値は画像からの推定",
        ),
        "surround": (
            "camera",
            f"周囲カメラ（後方・左・右の順）の Late Fusion。カメラごとに、いちばん近い車両・障害物・歩行者の "
            f"(dx, dy) / {config.OBS_SURROUND_RANGE}m（自車座標系: 前方 +x / 左 +y）と信頼度。"
            "写っていなければ 3 つとも 0。CNN で走るときは周囲カメラを 1 ステップに "
            f"{config.SURROUND_CNN_IMAGES_PER_STEP} 枚ずつ古い順に撮り直すので、少し前の画から作った値が入る",
        ),
        "v2x": (
            "v2x",
            f"車車間通信。{config.V2X_RANGE_M:.0f}m 以内の近い {config.V2X_MAX_PEERS} 台から受け取ったメッセージの平均。"
            "[車速 / max_speed, 右左折の意図（方向指示器と同じ -1/0/+1）, 送り手のカメラに写った歩行者・障害物の近さ "
            "max(0, 1 - d/30), 交差点（信号の無い交差点の入口か次の停止線）への近さ max(0, 1 - d/30)]。"
            "近くに車がいないか、v2xComm を切ったときは 4 つとも 0",
        ),
        "traffic_signs": (
            "camera",
            "前方カメラが認識した一時停止・横断歩道・一方通行・指定方向外進行禁止・駐車禁止・駐停車禁止の順に "
            f"[距離 / {config.OBS_SIGNAL_RANGE}m, 方位 / (pi/2), 信頼度]。未検出は 0。"
            f"末尾は指定方向の one-hot（{', '.join(config.OBS_SIGN_DIRECTIONS)}）。地図の真値は使わない",
        ),
        "occlusion": (
            "camera",
            f"4 台のカメラの検出と走行可能距離だけから作った見通しと死角（半径 {config.OBS_FREESPACE_MAX_DISTANCE:.0f}m）。"
            f"[左カメラの見通し距離 / {config.OBS_FREESPACE_MAX_DISTANCE:.0f}m, 右カメラの見通し距離 / "
            f"{config.OBS_FREESPACE_MAX_DISTANCE:.0f}m, 前方カメラの画角のうち検出した車両の陰の割合, "
            "いちばん近い遮蔽の角までの距離 / 20m（無ければ 1）, 前・後・左・右の 90 度の扇のうち見えている面積の割合]。"
            "見通し距離はそのカメラの画角で見えている奥行きの最大。まだ撮っていないカメラの欄は 0",
        ),
    }


def _observation_layout() -> list[dict[str, Any]]:
    """観測ベクトルの構成。`config.OBS_LAYOUT` の並び（`percep/encoder.py` の添字と同じ出典）。"""
    notes = _layout_notes()
    return [
        {"name": name, "size": size, "source": notes[name][0], "description": notes[name][1]}
        for name, size in config.OBS_LAYOUT
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
            "type": "shared hierarchical ActorCritic (parameter sharing)",
            "obsDim": int(trainer.obs_dim),
            "actionDim": int(trainer.action_dim),
            "hiddenSizes": [int(h) for h in trainer.policy.hidden_sizes],
            "activation": "tanh",
            "options": {
                "names": list(config.HRL_OPTIONS),
                "periodSteps": int(config.HRL_OPTION_STEPS),
                "note": (
                    "上位方策が意図を選び、下位方策が観測と意図の one-hot から操作を出す。"
                    "書き出したモデルは呼ばれるたびに、上位方策のいちばん確率の高い意図を選び直す"
                    "（アプリの中では periodSteps ごとに引き直して保つ）"
                ),
            },
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
            "note": "TorchScript 版・torch.export 版・Keras 版はいずれも分布の平均（tanh で [-1, 1] に収めたもの）を"
            "決定論的な行動として返す。学習時と同じ確率的な行動が欲しい場合は "
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


def preload_torch_export() -> None:
    """torch.export が初回に読むモジュールを先に読み込む（トレースはしない）。HTTP 側のスレッドから呼ぶ。"""
    for name in PT2_PRELOAD_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            warn_once(
                "rl.export.preload_torch_export",
                f"torch.export の先読みで {name} を読み込めませんでした（書き出しはできますが、初回が遅くなります）",
            )


def _export_program(policy: ActorCritic, obs_dim: int) -> Any:
    """推論モデルを torch.export でトレースする。バッチ数は 1 以上の任意の値で呼べる。"""
    batch = torch.export.Dim("batch", min=1)
    example = (torch.zeros(2, int(obs_dim), dtype=torch.float32),)
    return torch.export.export(InferencePolicy(policy), example, dynamic_shapes={"obs": {0: batch}})


def _import_keras():
    """Keras 3 を torch バックエンドで読み込む。"""
    os.environ.setdefault("KERAS_BACKEND", "torch")
    try:
        import keras  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 依存が無い環境向け
        raise ExportError(KERAS_MISSING_MESSAGE) from exc
    return keras


def _build_keras_model(trainer: Any):
    """PyTorch の ActorCritic と同じ構造の Keras モデルを作り、重みを移す。"""
    keras = _import_keras()

    policy = trainer.policy
    hidden = [int(h) for h in policy.hidden_sizes]
    obs_dim = int(policy.obs_dim)
    action_dim = int(policy.action_dim)
    state = policy.state_dict()

    num_options = int(policy.num_options)
    inputs = keras.Input(shape=(obs_dim,), dtype="float32", name="observation")

    def trunk(x, prefix: str):
        for i, units in enumerate(hidden):
            x = keras.layers.Dense(units, activation="tanh", name=f"{prefix}_dense_{i}")(x)
        return x

    # いちばん確率の高い意図の one-hot を、温度の低い softmax で標準の層だけで作る（Lambda を使わない）
    option = keras.layers.Dense(num_options, activation="softmax", name="option")(
        trunk(inputs, "meta")
    )
    sub_inputs = keras.layers.Concatenate(name="observation_option")([inputs, option])
    raw_action = keras.layers.Dense(action_dim, activation=None, name="action_raw")(
        trunk(sub_inputs, "policy")
    )
    option_bias = keras.layers.Dense(action_dim, use_bias=False, name="option_bias")(option)
    action = keras.layers.Activation("tanh", name="action")(
        keras.layers.Add(name="action_biased")([raw_action, option_bias])
    )
    value_dense = keras.layers.Dense(1, activation=None, name="value")(trunk(sub_inputs, "value"))
    value = keras.layers.Reshape((), name="value_squeezed")(value_dense)

    model = keras.Model(
        inputs=inputs, outputs=[action, value], name="autoware_sim_policy"
    )

    def weight(name: str):
        return state[f"{name}.weight"].detach().cpu().numpy()

    def bias(name: str):
        return state[f"{name}.bias"].detach().cpu().numpy()

    for i in range(len(hidden)):
        for prefix, source in (
            ("meta", "meta_trunk"),
            ("policy", "policy_trunk"),
            ("value", "value_trunk"),
        ):
            model.get_layer(f"{prefix}_dense_{i}").set_weights(
                [weight(f"{source}.{2 * i}").T, bias(f"{source}.{2 * i}")]
            )
    scale = 1.0 / float(KERAS_OPTION_TEMPERATURE)
    model.get_layer("option").set_weights(
        [weight("meta_head").T * scale, bias("meta_head") * scale]
    )
    model.get_layer("action_raw").set_weights([weight("mu_head").T, bias("mu_head")])
    model.get_layer("option_bias").set_weights(
        [state["option_bias"].detach().cpu().numpy()]
    )
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


def _is_import_backup(name: str) -> bool:
    return f"_{IMPORT_BACKUP_LABEL}_" in name


def prune_exports(
    directory: Path | None = None,
    *,
    keep: int = MAX_EXPORT_FILES,
    protect: Path | None = None,
    backups: int | None = None,
) -> list[str]:
    """書き出しディレクトリを新しい順に `keep` 件だけ残し、古いものを消す。"""
    directory = Path(directory) if directory is not None else config.EXPORT_DIR
    if not directory.is_dir():
        return []

    keep = max(1, int(keep))
    protect_name = protect.name if protect is not None else None

    entries: list[tuple[float, Path]] = []
    kept_backups: list[tuple[float, Path]] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.name.endswith(".tmp") or path.name.endswith(".partial.keras"):
            continue
        try:
            item = (path.stat().st_mtime, path)
        except OSError:
            continue
        # 読み込み前の退避は書き出しの世代に数えない。数えると 20 回書き出しただけで消える
        (kept_backups if backups is not None and _is_import_backup(path.name) else entries).append(item)

    entries.sort(key=lambda item: item[0], reverse=True)
    kept_backups.sort(key=lambda item: item[0], reverse=True)
    stale = entries[keep:]
    if backups is not None:
        stale += kept_backups[max(1, int(backups)) :]
    removed: list[str] = []
    for _mtime, path in stale:
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

        elif kind == "pt2":
            path = directory / f"{base}.pt2"
            program = _export_program(trainer.policy, int(trainer.obs_dim))
            buffer = io.BytesIO()
            extra = {"metadata.json": json.dumps(metadata, ensure_ascii=False, indent=2)}
            torch.export.save(program, buffer, extra_files=extra)
            _atomic_save(lambda p: p.write_bytes(buffer.getvalue()), path)
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
            prune_exports(directory, protect=path, backups=MAX_IMPORT_BACKUPS)
        except Exception:  # noqa: BLE001
            logger.exception("古い書き出しの整理に失敗しました（書き出し自体は成功しています）")

    return ExportResult(
        path=path,
        filename=path.name,
        size_bytes=path.stat().st_size,
        media_type=media_type,
        kind=kind,
    )
