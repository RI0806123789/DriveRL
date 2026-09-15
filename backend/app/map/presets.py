"""読み込み対象エリアの固定プリセット定義。"""

from __future__ import annotations

from app.contracts import MapPreset

_PRESET_LIST: tuple[MapPreset, ...] = (
    MapPreset(
        id="ginza",
        name="東京・銀座",
        description="中央通りと晴海通りが交差する、高層ビルが密集する銀座中心部",
        center_lat=35.6717,
        center_lon=139.7650,
        radius_m=400.0,
    ),
    MapPreset(
        id="umeda",
        name="大阪・梅田",
        description="阪急とJRの大阪駅に隣接する、大通りと商業ビルが集中する梅田",
        center_lat=34.7025,
        center_lon=135.4959,
        radius_m=400.0,
    ),
    MapPreset(
        id="sakae",
        name="名古屋・栄",
        description="久屋大通と広小路通が交わる、碁盤の目状の街路が広がる名古屋・栄",
        center_lat=35.1681,
        center_lon=136.9083,
        radius_m=400.0,
    ),
    MapPreset(
        id="kanazawa",
        name="石川・金沢",
        description="金沢市街地の全域（12km 四方）。香林坊を中心に金沢駅・兼六園・野々市市境まで",
        center_lat=36.5606,
        center_lon=136.6555,
        radius_m=6000.0,
        signals_at_all_intersections=False,
    ),
)

PRESETS: dict[str, MapPreset] = {p.id: p for p in _PRESET_LIST}

DEFAULT_PRESET_ID: str = _PRESET_LIST[0].id


def list_presets() -> list[MapPreset]:
    """定義順のプリセット一覧を返す（docs/protocol.md 2.1 の init.presets 用）。"""
    return list(_PRESET_LIST)


def get_preset(preset_id: str) -> MapPreset | None:
    """ID からプリセットを引く。未知の ID なら None（呼び出し側でエラー応答を作る）。"""
    return PRESETS.get(preset_id)
