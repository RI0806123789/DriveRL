"""地図データの取得・正規化・実行時インデックス構築を担うパッケージ。"""

from __future__ import annotations

import importlib
from typing import Any

from app.map.presets import DEFAULT_PRESET_ID, PRESETS, get_preset, list_presets

__all__ = [
    "MapLoadError",
    "MapIndexImpl",
    "PRESETS",
    "DEFAULT_PRESET_ID",
    "list_presets",
    "get_preset",
    "load_map",
    "cache_path_for",
    "build_map_index",
]

_LAZY_EXPORTS: dict[str, str] = {
    "MapIndexImpl": "app.map.index",
    "build_map_index": "app.map.index",
    "MapLoadError": "app.map.loader",
    "cache_path_for": "app.map.loader",
    "load_map": "app.map.loader",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
