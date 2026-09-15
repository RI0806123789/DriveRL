"""画像認識パッケージ（擬似カメラ → CNN 認識器 → 検出結果）。"""

from __future__ import annotations

from app.percep.types import (
    DEFAULT_CAMERA,
    CameraSpec,
    DetClass,
    Detection,
    PerceptionResult,
)

__all__ = [
    "CameraSpec",
    "DEFAULT_CAMERA",
    "DetClass",
    "Detection",
    "PerceptionResult",
]
