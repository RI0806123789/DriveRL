"""全プリセットのマップを先読みして JSON キャッシュを温める CLI。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from app.contracts import MapData, MapPreset
from app.map.loader import MapLoadError, cache_path_for, load_map
from app.map.presets import get_preset, list_presets


def _format_size(path: Path) -> str:
    try:
        return f"{path.stat().st_size / 1024.0:.1f} KB"
    except OSError:
        return "不明"


def _report(preset: MapPreset, data: MapData, elapsed: float) -> None:
    b = data.bounds
    total_edge_length = sum(e.length for e in data.edges)
    cache = cache_path_for(preset)
    print(f"  ノード数      : {len(data.nodes)}")
    print(f"  エッジ数      : {len(data.edges)}")
    print(f"  建物数        : {len(data.buildings)}")
    print(f"  道路総延長    : {total_edge_length:,.0f} m")
    print(
        "  bounds [m]    : "
        f"x [{b.min_x:.1f}, {b.max_x:.1f}] / y [{b.min_y:.1f}, {b.max_y:.1f}]"
        f"  （幅 {b.max_x - b.min_x:.1f} × 高さ {b.max_y - b.min_y:.1f}）"
    )
    print(f"  所要時間      : {elapsed:.2f} 秒")
    print(f"  キャッシュ    : {cache}（{_format_size(cache)}）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="app.map.prefetch",
        description="全プリセットのマップを取得して JSON キャッシュを作成する",
    )
    parser.add_argument(
        "preset_ids",
        nargs="*",
        help="対象プリセット ID（省略時は全件）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存のキャッシュを無視して OSM から取り直す",
    )
    args = parser.parse_args(argv)

    if args.preset_ids:
        targets: list[MapPreset] = []
        for preset_id in args.preset_ids:
            preset = get_preset(preset_id)
            if preset is None:
                print(f"[警告] 未知のプリセット ID をスキップします: {preset_id}")
                continue
            targets.append(preset)
    else:
        targets = list_presets()

    if not targets:
        print("[エラー] 対象プリセットがありません。")
        return 1

    print(f"マップ先読みを開始します（対象 {len(targets)} 件 / force={args.force}）\n")

    succeeded = 0
    failed: list[tuple[str, str]] = []

    for i, preset in enumerate(targets, start=1):
        print(f"[{i}/{len(targets)}] {preset.id} — {preset.name}")
        print(
            f"  中心 (lat, lon) : ({preset.center_lat}, {preset.center_lon})"
            f" / 半径 {preset.radius_m:.0f} m"
        )
        started = time.perf_counter()
        try:
            data = load_map(preset, force_refresh=args.force)
        except MapLoadError as exc:
            elapsed = time.perf_counter() - started
            print(f"  [失敗] {exc}（{elapsed:.2f} 秒）\n")
            failed.append((preset.id, str(exc)))
            continue
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(f"  [失敗] 想定外のエラー: {type(exc).__name__}: {exc}（{elapsed:.2f} 秒）\n")
            failed.append((preset.id, f"{type(exc).__name__}: {exc}"))
            continue

        elapsed = time.perf_counter() - started
        _report(preset, data, elapsed)
        print()
        succeeded += 1

    print("=" * 60)
    print(f"完了: 成功 {succeeded} 件 / 失敗 {len(failed)} 件")
    for preset_id, message in failed:
        print(f"  - {preset_id}: {message}")

    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
