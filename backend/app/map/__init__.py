"""地図データの取得・正規化・実行時インデックス構築を担うパッケージ。

外部（`sim` / `runtime`）からはこのモジュールの re-export だけを使う想定::

    from app.map import PRESETS, get_preset, load_map, build_map_index, MapLoadError

`index` の import は shapely / networkx を引き込むので、プリセット一覧だけが
欲しい場面（起動直後の init メッセージなど）でも重くならないよう、
実装本体は各サブモジュールに置いている。

★ **その狙いは再エクスポートを遅延させて初めて達成される**（code_review R-08）。
  以前はここで `app.map.index` を無条件に import していたため、
  `from app.map import list_presets` だけでも networkx と shapely が引き込まれ、
  WebSocket の接続ごとに実測 256.4ms 止まっていた。**この docstring が
  「重くならないようにしてある」と書いている当のことが、実装では起きていなかった。**
  いまは軽い `presets` だけを先に読み、残りは属性アクセス時に解決する（PEP 562）。
"""

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

#: 遅延解決する名前 -> 実装モジュール。
#: `index` は shapely / networkx を、`loader` は osmnx を引き込むので、
#: 実際に使う瞬間まで触らない。
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
    globals()[name] = value  # 2 回目以降は __getattr__ を通らない
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
