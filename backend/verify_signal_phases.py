# -*- coding: utf-8 -*-
"""交差する流れが同じ現示に入っていないことを、プリセットのマップで検証する。"""
from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
for st in (sys.stdout, sys.stderr):
    try:
        st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


import numpy as np

from app.map.loader import SIGNAL_CONFLICT_ANGLE, MapLoadError, _axis_angle, load_map
from app.map.presets import get_preset, list_presets
from app.sim.signals import RED, SignalController

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if ok else 'NG  '}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def longest_red(ctrl: SignalController, count: int) -> float:
    """どれか 1 基でも赤が続く最長の秒数（2 サイクル分を 0.5 秒刻みで見る）。"""
    span = ctrl.max_cycle * 2.0
    run = np.zeros(count)
    best = np.zeros(count)
    for k in range(int(span / 0.5)):
        red = np.array(ctrl.phases(k * 0.5)) == RED
        run = np.where(red, run + 0.5, 0.0)
        best = np.maximum(best, run)
    return float(best.max())


targets = sys.argv[1:] or [p.id for p in list_presets()]

for preset_id in targets:
    preset = get_preset(preset_id)
    if preset is None:
        print(f"未知のプリセット: {preset_id}")
        sys.exit(2)

    print("=" * 72)
    print(f"{preset.id}（{preset.name}）")
    print("=" * 72)
    try:
        data = load_map(preset)
    except MapLoadError as exc:
        print(f"  マップを読めません（{exc}）。prefetch してから実行すること")
        sys.exit(2)

    by_key: dict[int, list] = defaultdict(list)
    for sg in data.signals:
        by_key[sg.phase_key].append(sg)

    conflicts = 0
    conflict_nodes = 0
    worst = 0.0
    phase_counts: Counter[int] = Counter()
    gaps = 0
    for members in by_key.values():
        groups = sorted({m.group for m in members})
        phase_counts[len(groups)] += 1
        if groups != list(range(len(groups))):
            gaps += 1
        hit = False
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if a.group != b.group:
                    continue
                angle = _axis_angle(a.heading, b.heading)
                if angle >= SIGNAL_CONFLICT_ANGLE:
                    conflicts += 1
                    worst = max(worst, angle)
                    hit = True
        conflict_nodes += hit

    total = len(by_key)
    print(f"  灯器 {len(data.signals)} 基 / 交差点 {total} か所")
    print(f"  現示の数: {dict(sorted(phase_counts.items()))}")

    check(
        "同じ現示に、交差する流れ（軸差 30 度以上）が入っていない",
        conflicts == 0,
        f"交差点 {conflict_nodes} / 組 {conflicts} / 最大 {math.degrees(worst):.0f} 度",
    )
    check("群番号が 0 から連番になっている", gaps == 0, f"連番でない交差点 {gaps} か所")

    if data.signals:
        ctrl = SignalController(data.signals)
        print(f"  {ctrl.describe()}")
        red = longest_red(ctrl, len(data.signals))
        check("赤で待たされるのは 60 秒未満", red < 60.0, f"最長 {red:.1f} 秒")
    print()

print("=" * 72)
if FAILURES:
    print(f"結果: {len(FAILURES)} 件の不合格")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("結果: すべて合格")
