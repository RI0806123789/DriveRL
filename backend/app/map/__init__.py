"""地図データの取得・正規化・実行時インデックス構築を担うパッケージ。

外部（`sim` / `runtime`）からはこのモジュールの re-export だけを使う想定::

    from app.map import PRESETS, get_preset, load_map, build_map_index, MapLoadError

`index` の import は shapely / networkx を引き込むので、プリセット一覧だけが
欲しい場面（起動直後の init メッセージなど）でも重くならないよう、
実装本体は各サブモジュールに置いている。
"""

from __future__ import annotations

from app.map.index import MapIndexImpl, build_map_index
from app.map.loader import MapLoadError, cache_path_for, load_map
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
