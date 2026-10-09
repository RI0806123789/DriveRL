"""書き出したモデルファイルの読み込み（学習の再開）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import zipfile

import torch

from app.model_upload import MAX_UPLOAD_BYTES
from app.rl.checkpoint import CheckpointValidationError, prepare_checkpoint

__all__ = ["CheckpointImportError", "CheckpointInfo", "inspect_checkpoint", "MAX_UPLOAD_BYTES"]

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
        map_info = meta.get("map")
        if not isinstance(map_info, dict):
            map_info = {}
        return {
            "updates": self.updates,
            "obsDim": self.obs_dim,
            "actionDim": self.action_dim,
            "hiddenSizes": self.hidden_sizes,
            "hasOptimizer": self.has_optimizer,
            "exportedAt": meta.get("exportedAt"),
            "presetId": map_info.get("presetId"),
            "presetName": map_info.get("presetName"),
            "metrics": meta.get("metrics") or {},
        }


def _looks_like_keras(path: Path) -> bool:
    """Keras の .keras（zip）を渡されたかどうかを見分ける。"""
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except Exception:
        return False
    return "config.json" in names and "model.weights.h5" in names


def _looks_like_torchscript(path: Path) -> bool:
    """TorchScript ファイルを間違って渡されたかどうかを見分ける。"""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except Exception:
        return False
    return any(n.endswith("constants.pkl") for n in names) and any(
        "/code/" in n for n in names
    )


def _looks_like_pt2(path: Path) -> bool:
    """torch.export の .pt2（zip）を間違って渡されたかどうかを見分ける。"""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except Exception:
        return False
    return any(n.endswith("/archive_format") for n in names) and any(
        "/models/" in n and n.endswith(".json") for n in names
    )


def inspect_checkpoint(
    path: Path,
    *,
    expected_obs_dim: int,
    expected_action_dim: int,
    expected_hidden_sizes: Sequence[int],
) -> CheckpointInfo:
    """チェックポイントを安全に解析し、このアプリに載せられるか検証する。"""
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
        if _looks_like_pt2(path):
            raise CheckpointImportError(
                "これは torch.export 形式（.pt2、推論専用）のファイルです。"
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

    try:
        prepared = prepare_checkpoint(
            payload, obs_dim=expected_obs_dim, action_dim=expected_action_dim,
            hidden_sizes=expected_hidden_sizes,
        )
    except CheckpointValidationError as exc:
        raise CheckpointImportError(str(exc)) from exc

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = None

    return CheckpointInfo(
        updates=prepared.updates,
        obs_dim=prepared.obs_dim,
        action_dim=prepared.action_dim,
        hidden_sizes=list(prepared.hidden_sizes),
        has_optimizer=prepared.has_optimizer,
        metadata=metadata,
    )
