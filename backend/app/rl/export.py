"""学習済みモデルの書き出し。

3 つの形式を用意する。

- **checkpoint (.pt)**: `PPOTrainer.save()` と同じ中身（重み＋オプティマイザ状態）に、
  人が読めるメタデータを足したもの。このプロジェクトに読み戻せる、いわば「続きから学習できる」形式。
- **TorchScript (.torchscript.pt)**: 推論だけを切り出した自己完結の形式。
  このリポジトリのコードが無くても `torch.jit.load()` だけで動く。
- **Keras (.keras)**: 同じネットワークを Keras 3 のモデルとして組み直し、重みを移したもの。
  `keras.saving.load_model()` で読める。PyTorch の外（Keras/TensorFlow 系の資産）で
  扱いたい場合向け。標準の Dense 層だけで構成しているので custom_objects は不要。

どちらにもメタデータ（観測ベクトルの構成、行動のスケール、学習の進み具合）を必ず埋め込む。
これが無いと、書き出したモデルを受け取った側が「54 次元の入力に何を入れればよいか」を
再現できず、ファイルとしては読めても実際には使えない。

**観測の構成を変えたら `_observation_layout()` も必ず直すこと。**
ここがずれると、次元数だけ合っていて中身の説明が嘘になる。
"""

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

# メタデータのスキーマ版。中身の意味を変えたら上げること。
# 2: 観測を画像認識ベースへ移行。`observation.layout` の各項に `source` が付き、
#    値の意味が「真値」から「CNN 認識器の出力」へ変わった。
METADATA_VERSION = 2

# `data/exports/` に残す世代数（1 ファイル = 1 世代）。
# 読み込みのたびに `before-import` の退避が増えるので、上限が無いと際限なく貯まる
# （実測で 48 ファイル 21MB）。古いものから消す。
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


# ---------------------------------------------------------------------------
# 推論専用ラッパー（TorchScript 用）
# ---------------------------------------------------------------------------


class InferencePolicy(nn.Module):
    """観測から決定論的な行動を出すだけのモジュール。

    学習時の方策は対角ガウス分布からサンプリングするが、書き出し先で欲しいのは
    普通「一番よいと思っている行動」なので、分布の平均を採用して [-1, 1] に
    クリップする。探索用のノイズを再現したい場合のために log_std も同梱する。
    """

    def __init__(self, policy: ActorCritic) -> None:
        super().__init__()
        # 学習中のモジュールを共有すると、書き出した後の重み更新が
        # TorchScript 側にも影響してしまう。必ず複製してから固める。
        self.policy_trunk = copy.deepcopy(policy.policy_trunk)
        self.mu_head = copy.deepcopy(policy.mu_head)
        self.value_trunk = copy.deepcopy(policy.value_trunk)
        self.value_head = copy.deepcopy(policy.value_head)
        # 学習時と同じクランプを掛けた値を定数として持たせる
        log_std = torch.clamp(policy.log_std.detach().clone(), -5.0, 1.0)
        self.register_buffer("log_std", log_std)

        # 推論専用なので勾配追跡を外す。付けたままだと、受け取った側が
        # 出力を float() するだけで警告が出るなど、使い勝手が悪くなる。
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: obs (B, obs_dim) -> Returns: (action (B, action_dim), value (B,))"""
        mu = self.mu_head(self.policy_trunk(obs))
        action = torch.clamp(mu, -1.0, 1.0)
        value = self.value_head(self.value_trunk(obs)).squeeze(-1)
        return action, value


# ---------------------------------------------------------------------------
# メタデータ
# ---------------------------------------------------------------------------


def _observation_layout() -> list[dict[str, Any]]:
    """観測ベクトルの構成。app/percep/encoder.py の連結順と一致していること。

    ★ **この観測は CNN 認識器の出力でできている**（`app/percep/`）。
      受け取った側が同じ入力を再現するには、車載カメラ相当の画像を同じ認識器に
      通す必要がある。真値をそのまま入れると学習時と分布が変わる
      （学習時の観測は**誤認識と見落としを含んでいる**）。
      `source` がその区別で、"camera" は認識結果、それ以外は
      カメラを通さずに得られる値（速度計・ナビ）。
    """
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
) -> dict[str, Any]:
    """書き出しに同梱するメタデータを組み立てる。"""
    return {
        "metadataVersion": METADATA_VERSION,
        "application": "DriveRL",
        "kind": kind,
        "exportedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        # ★ str() で包むこと。torch.__version__ は str のサブクラス TorchVersion であり、
        #   そのまま保存すると torch.load(weights_only=True) が
        #   「Unsupported global: torch.torch_version.TorchVersion」で読み込みを拒否する。
        #   読み込み側を安全モードで動かすために、素の str に落とす。
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
            # 探索用のノイズ。決定論的な行動が欲しいだけなら使わなくてよい。
            # 学習時と同じ確率的な行動を再現するときに Normal(action, exp(log_std)) とする。
            "logStd": [float(v) for v in trainer.policy.log_std.detach().cpu().numpy()],
        },
        "vehicle": {
            "length": config.VEHICLE_LENGTH,
            "width": config.VEHICLE_WIDTH,
            "wheelbase": config.WHEELBASE,
            "maxSteer": config.MAX_STEER,
            "maxAccel": config.MAX_ACCEL,
            "maxDecel": config.MAX_DECEL,
            "dt": config.DT,
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



# ---------------------------------------------------------------------------
# Keras 形式
# ---------------------------------------------------------------------------


def preload_keras() -> None:
    """Keras を先に読み込んでおく。

    `import keras` は数秒かかる。書き出し自体はシミュレーションスレッドの
    ステップ境界で行うので、そこでインポートすると学習が数秒止まってしまう。
    HTTP ハンドラ側から別スレッドで先に呼んでおくためのフック。
    """
    _import_keras()


def _import_keras():
    """Keras 3 を torch バックエンドで読み込む。

    Keras 3 は既定で TensorFlow を探すが、この環境には入れていない。
    torch は学習で既に使っているので、それをバックエンドにする。
    環境変数は `import keras` より前に設定しなければ効かない。
    """
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
    """PyTorch の ActorCritic と同じ構造の Keras モデルを作り、重みを移す。

    層は標準の Dense だけで組む。カスタム層を使うと読み込み側にこのコードが
    必要になり、「Keras だけで読める」という利点が消えてしまうため。
    行動のクリップ [-1, 1] は `hard_tanh` 活性で表す（clip と厳密に一致する）。
    """
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
    value = keras.layers.Dense(1, activation=None, name="value")(trunk(inputs, "value"))

    model = keras.Model(
        inputs=inputs, outputs=[action, value], name="autoware_sim_policy"
    )

    # --- 重みの移植 ---
    # PyTorch の Linear は (out, in)、Keras の Dense カーネルは (in, out) なので転置する。
    def weight(name: str):
        return state[f"{name}.weight"].detach().cpu().numpy()

    def bias(name: str):
        return state[f"{name}.bias"].detach().cpu().numpy()

    for i in range(len(hidden)):
        # trunk は Linear と Tanh の交互なので、i 番目の Linear は添字 2i
        for prefix, source in (("policy", "policy_trunk"), ("value", "value_trunk")):
            model.get_layer(f"{prefix}_dense_{i}").set_weights(
                [weight(f"{source}.{2 * i}").T, bias(f"{source}.{2 * i}")]
            )
    model.get_layer("action").set_weights([weight("mu_head").T, bias("mu_head")])
    model.get_layer("value").set_weights([weight("value_head").T, bias("value_head")])

    return model


def _attach_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """.keras（zip）へ独自のメタデータを追記する。

    Keras は自分が知っているエントリしか読まないので、追記しても
    `keras.saving.load_model()` はそのまま通る。モデル単体で
    「観測に何を入れるか」が分かるようにするための同梱。
    """
    with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            KERAS_METADATA_ENTRY,
            json.dumps(metadata, ensure_ascii=False, indent=2),
        )


#: .keras の中に入れる独自メタデータのファイル名
KERAS_METADATA_ENTRY = "autoware_sim_metadata.json"


# ---------------------------------------------------------------------------
# 書き出し
# ---------------------------------------------------------------------------


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
    """書き出しディレクトリを新しい順に `keep` 件だけ残し、古いものを消す。

    書き出しと読み込み前の自動退避で増え続けるため、書き出しのたびに呼ぶ。
    消せなかったファイル（ダウンロード中で掴まれている等）は黙って飛ばす。
    削除に失敗しても書き出し自体は成功しているので、例外は外へ出さない。

    Args:
        protect: 何があっても消さないファイル（いま書き出したもの）
    Returns:
        実際に消したファイル名。
    """
    directory = Path(directory) if directory is not None else config.EXPORT_DIR
    if not directory.is_dir():
        return []

    keep = max(1, int(keep))
    protect_name = protect.name if protect is not None else None

    entries: list[tuple[float, Path]] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        # 書き出し途中の一時ファイルは世代に数えない（掴まれている可能性がある）
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
) -> ExportResult:
    """モデルを書き出してファイルの情報を返す。

    Args:
        trainer: `PPOTrainer`
        kind: "checkpoint" か "torchscript"
        label: ファイル名に挟む目印（例: "before-import"）。あとから探しやすくするため
    Raises:
        ExportError: 未知の形式、または書き出しに失敗したとき
    """
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
    )

    try:
        if kind == "checkpoint":
            path = directory / f"{base}.pt"
            payload = {
                # PPOTrainer.load() が読める形を保つ（続きから学習できるようにするため）。
                # 形式の識別子は save() と同じ定数を使う（経路によって型が違わないように）
                "format": CHECKPOINT_FORMAT,
                "obs_dim": int(trainer.obs_dim),
                "action_dim": int(trainer.action_dim),
                "hidden_sizes": [int(h) for h in trainer.policy.hidden_sizes],
                "updates": int(trainer.updates),
                "policy": trainer.policy.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                # 書き出し版だけの追加情報
                "metadata": metadata,
            }
            _atomic_save(lambda p: torch.save(payload, p), path)
            media_type = "application/octet-stream"

        elif kind == "keras":
            path = directory / f"{base}.keras"
            model = _build_keras_model(trainer)
            # Keras は保存先の拡張子を見て形式を決めるので、.tmp では保存できない。
            # 一度別名の .keras に書いてから差し替える。
            tmp_path = path.with_name(path.stem + ".partial.keras")
            model.save(tmp_path)
            _attach_metadata(tmp_path, metadata)
            os.replace(tmp_path, path)
            media_type = "application/octet-stream"

        else:  # torchscript
            path = directory / f"{base}.torchscript.pt"
            module = InferencePolicy(trainer.policy)
            try:
                scripted = torch.jit.script(module)
            except Exception:
                # script が通らない環境向けの保険。バッチ次元は動的のままにする。
                example = torch.zeros(1, int(trainer.obs_dim), dtype=torch.float32)
                scripted = torch.jit.trace(module, example, check_trace=False)

            extra = {"metadata.json": json.dumps(metadata, ensure_ascii=False, indent=2)}
            _atomic_save(lambda p: torch.jit.save(scripted, str(p), _extra_files=extra), path)
            media_type = "application/octet-stream"

    except ExportError:
        raise
    except Exception as exc:  # noqa: BLE001 - 失敗理由をそのまま画面に出したい
        raise ExportError(f"モデルの書き出しに失敗しました: {exc}") from exc

    # 世代上限を超えたぶんを片付ける。失敗しても書き出しは成功しているので握る。
    # 掃除するのは既定の書き出し先だけ。out_dir を指定された場合は、呼び出し側の
    # ディレクトリに関係の無いファイルが入っていることがあるので触らない。
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
