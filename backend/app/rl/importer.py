"""書き出したモデルファイルの読み込み（学習の再開）。

`app/rl/export.py` が書き出した `.pt` を受け取り、中身を検証してから
`PPOTrainer` へ載せ替えるための下ごしらえを行う。

扱えるのは `checkpoint` 形式（`.pt`）だけ。TorchScript や Keras 形式は推論専用で
オプティマイザ状態を持たないため学習を再開できない。取り違えたときに
「壊れています」ではなく理由が分かるよう、形式を見分けて案内する。

このモジュールが扱うファイルは **ユーザーがアップロードしたもの** である。
`torch.load` は既定でピクルを実行するため、任意コード実行の入口になりうる。
そこでここでは必ず `weights_only=True` で解析し、テンソルと素の Python 値しか
入っていないことを確認してから先へ進める。

（`export.py` が `torch.__version__` を `str()` で包んでいるのはこのため。
  TorchVersion オブジェクトのままだと安全モードで弾かれてしまう。）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import zipfile

import torch

from app.rl.ppo import KNOWN_CHECKPOINT_FORMATS

__all__ = ["CheckpointImportError", "CheckpointInfo", "inspect_checkpoint", "MAX_UPLOAD_BYTES"]

logger = logging.getLogger(__name__)

# アップロードを受け付ける最大サイズ。実測で 1 ファイル約 0.6MB なので十分な余裕。
MAX_UPLOAD_BYTES = 256 * 1024 * 1024


class CheckpointImportError(RuntimeError):
    """読み込めないファイルを渡されたときに投げる。文言はそのまま画面に出す。"""


@dataclass
class CheckpointInfo:
    """検証を通ったチェックポイントの概要。UI に「何を読み込むのか」を見せるために使う。"""

    updates: int
    obs_dim: int
    action_dim: int
    hidden_sizes: list[int]
    has_optimizer: bool
    metadata: dict[str, Any] | None

    def to_wire(self) -> dict[str, Any]:
        meta = self.metadata or {}
        return {
            "updates": self.updates,
            "obsDim": self.obs_dim,
            "actionDim": self.action_dim,
            "hiddenSizes": self.hidden_sizes,
            "hasOptimizer": self.has_optimizer,
            "exportedAt": meta.get("exportedAt"),
            "presetId": (meta.get("map") or {}).get("presetId"),
            "presetName": (meta.get("map") or {}).get("presetName"),
            "metrics": meta.get("metrics") or {},
        }


def _looks_like_keras(path: Path) -> bool:
    """Keras の .keras（zip）を渡されたかどうかを見分ける。

    keras を import せずに済むよう、zip の中身だけで判定する。
    """
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except Exception:
        return False
    return "config.json" in names and "model.weights.h5" in names


def _looks_like_torchscript(path: Path) -> bool:
    """TorchScript ファイルを間違って渡されたかどうかを見分ける。

    ユーザーの手元には `.pt` と `.torchscript.pt` の 2 つが並んでいるので、
    取り違えは十分起こりうる。「読めません」で終わらせず、どちらを選ぶべきか伝えたい。
    """
    try:
        torch.jit.load(str(path), map_location="cpu")
        return True
    except Exception:
        return False


def inspect_checkpoint(
    path: Path,
    *,
    expected_obs_dim: int,
    expected_action_dim: int,
    expected_hidden_sizes: Sequence[int],
) -> CheckpointInfo:
    """チェックポイントを安全に解析し、このアプリに載せられるか検証する。

    Raises:
        CheckpointImportError: 読めない、または形が合わないとき（理由は日本語で入る）
    """
    path = Path(path)
    if not path.exists():
        raise CheckpointImportError("ファイルが見つかりません")

    size = path.stat().st_size
    if size == 0:
        raise CheckpointImportError("ファイルが空です")
    if size > MAX_UPLOAD_BYTES:
        raise CheckpointImportError(
            f"ファイルが大きすぎます（{size / 1024 / 1024:.0f} MB）。"
            f"上限は {MAX_UPLOAD_BYTES // 1024 // 1024} MB です"
        )

    # --- 安全モードで解析する（ピクルの任意コード実行を通さない） ---
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        if _looks_like_keras(path):
            raise CheckpointImportError(
                "これは Keras 形式（.keras、推論専用）のファイルです。"
                "学習を再開するには「重み一式（.pt）」で書き出したファイルを選んでください"
            ) from exc
        if _looks_like_torchscript(path):
            raise CheckpointImportError(
                "これは TorchScript 形式（推論専用）のファイルです。"
                "学習を再開するには「重み一式（.pt）」で書き出したファイルを選んでください"
            ) from exc
        message = str(exc)
        if "Unsupported global" in message or "WeightsUnpickler" in message:
            raise CheckpointImportError(
                "このアプリが書き出したものではない可能性があります"
                "（重み以外のオブジェクトが含まれているため、安全のため読み込みを中止しました）"
            ) from exc
        raise CheckpointImportError(
            "PyTorch のチェックポイントとして読めませんでした。ファイルが壊れていないか確認してください"
        ) from exc

    if not isinstance(payload, dict):
        raise CheckpointImportError("チェックポイントの形式が違います（辞書ではありません）")

    missing = [k for k in ("obs_dim", "action_dim", "hidden_sizes", "policy") if k not in payload]
    if missing:
        raise CheckpointImportError(
            f"必要な項目が入っていません: {', '.join(missing)}。"
            "このアプリが書き出した「重み一式（.pt）」を選んでください"
        )

    # 形式の識別子。**未知でも欠けていても拒否はしない。**
    # 以前は保存経路によって int の 1 と文字列が混在しており、
    # 手元に両方のファイルが残っているため（形が合わなければ下の次元チェックで落ちる）
    fmt = payload.get("format")
    if fmt is None:
        logger.warning("形式の識別子が入っていないチェックポイントです: %s", path.name)
    elif fmt not in KNOWN_CHECKPOINT_FORMATS:
        logger.warning(
            "見覚えのないチェックポイント形式です: %r（%s）", fmt, path.name
        )

    try:
        obs_dim = int(payload["obs_dim"])
        action_dim = int(payload["action_dim"])
        hidden_sizes = [int(h) for h in payload["hidden_sizes"]]
    except (TypeError, ValueError) as exc:
        raise CheckpointImportError("チェックポイントのモデル定義が壊れています") from exc

    # --- このアプリのネットワークと形が一致するか ---
    if obs_dim != expected_obs_dim:
        raise CheckpointImportError(
            f"観測ベクトルの次元が違います（ファイル: {obs_dim} / このアプリ: {expected_obs_dim}）。"
            "観測の作り方を変更した後のモデルは読み込めません"
        )
    if action_dim != expected_action_dim:
        raise CheckpointImportError(
            f"行動の次元が違います（ファイル: {action_dim} / このアプリ: {expected_action_dim}）"
        )
    if tuple(hidden_sizes) != tuple(int(h) for h in expected_hidden_sizes):
        raise CheckpointImportError(
            f"ネットワークの層構成が違います"
            f"（ファイル: {hidden_sizes} / このアプリ: {list(expected_hidden_sizes)}）"
        )

    if not isinstance(payload.get("policy"), dict):
        raise CheckpointImportError("重み（policy）が入っていません")

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = None

    return CheckpointInfo(
        updates=int(payload.get("updates", 0)),
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        has_optimizer=isinstance(payload.get("optimizer"), dict),
        metadata=metadata,
    )
