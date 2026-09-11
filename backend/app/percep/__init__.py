"""画像認識パッケージ（擬似カメラ → CNN 認識器 → 検出結果）。

型はこのモジュールから使う::

    from app.percep import CameraSpec, Detection, DetClass

**実装（camera / detector / groundtruth / encoder）はここで re-export しない。**
`detector` は Keras を引き込むし、`camera` も観測を作る瞬間まで要らない。
起動を待たせないよう、利用側（`app/sim/env.py` の `_ensure_percep()`）が
必要になった時点で直接 import する。

学習の観測も画面のバウンディングボックスも**同じ検出結果**から作る。
`types.py` がその契約なので、片方の都合だけで型を変えないこと。
"""

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
