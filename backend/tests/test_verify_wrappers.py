"""既存の verify_*.py を子プロセスで回し、終了コードで合否を決める。"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass

import pytest

from app.map.loader import CACHE_VERSION, cache_path_for
from app.map.presets import get_preset, list_presets

from .conftest import BACKEND_DIR

LARGE_MAP_RADIUS_M = 1000.0
TIMEOUT_SEC = 3600
_VERSION_HEAD = re.compile(r'^\{"version":(\d+),')

SCRIPTS = sorted(p.name for p in BACKEND_DIR.glob("verify_*.py"))
ALL_PRESETS = tuple(p.id for p in list_presets())
SMALL_PRESETS = tuple(p.id for p in list_presets() if p.radius_m <= LARGE_MAP_RADIUS_M)


@dataclass(frozen=True)
class Plan:
    """1 本のスクリプトの回し方。`per_preset` なら引数にプリセット名を 1 つずつ渡して分けて回す。"""

    maps: tuple[str, ...] = ()
    per_preset: bool = False
    fast_presets: tuple[str, ...] = ()


PLANS: dict[str, Plan] = {
    "verify_log_std.py": Plan(),
    "verify_publish_routes.py": Plan(maps=("ginza",)),
    "verify_signal_phases.py": Plan(maps=ALL_PRESETS, per_preset=True, fast_presets=SMALL_PRESETS),
    "verify_route_start.py": Plan(maps=ALL_PRESETS, per_preset=True),
    "verify_route_signals.py": Plan(maps=ALL_PRESETS, per_preset=True),
    "verify_safety_gimmicks.py": Plan(maps=("ginza", "umeda", "sakae")),
}


def _cases() -> list[pytest.ParameterSet]:
    cases: list[pytest.ParameterSet] = []
    for script in SCRIPTS:
        plan = PLANS.get(script, Plan())
        stem = script.removesuffix(".py")
        if plan.per_preset:
            for preset in plan.maps:
                marks = () if preset in plan.fast_presets else (pytest.mark.slow,)
                cases.append(pytest.param(script, [preset], (preset,), id=f"{stem}-{preset}", marks=marks))
        else:
            cases.append(pytest.param(script, [], plan.maps, id=stem, marks=(pytest.mark.slow,)))
    return cases


def _stale_maps(presets: tuple[str, ...]) -> list[str]:
    """キャッシュが無いか版が古いプリセット（読むと Overpass から取り直しになる）。"""
    stale: list[str] = []
    for preset_id in presets:
        path = cache_path_for(get_preset(preset_id))
        try:
            with path.open("r", encoding="utf-8") as fp:
                head = _VERSION_HEAD.match(fp.read(64))
        except OSError:
            head = None
        if head is None or int(head.group(1)) != CACHE_VERSION:
            stale.append(preset_id)
    return stale


def test_every_script_has_a_plan() -> None:
    unplanned = [s for s in SCRIPTS if s not in PLANS]
    assert unplanned == [], "新しい verify_*.py は PLANS に読むマップを書き足すこと"
    gone = [s for s in PLANS if s not in SCRIPTS]
    assert gone == [], "消えたスクリプトが PLANS に残っている"


@pytest.mark.parametrize(("script", "args", "maps"), _cases())
def test_verify_script(script: str, args: list[str], maps: tuple[str, ...]) -> None:
    stale = _stale_maps(maps)
    if stale:
        pytest.skip(
            f"マップのキャッシュが無いか古い（{', '.join(stale)}）。"
            f"backend で python -m app.map.prefetch を実行してから回す"
        )
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.run(
        [sys.executable, script, *args],
        cwd=BACKEND_DIR,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT_SEC,
    )
    if proc.returncode == 0:
        return
    lines = (proc.stdout + proc.stderr).splitlines()
    failed = [line for line in lines if "[NG" in line]
    tail = "\n".join(lines[-30:])
    pytest.fail(
        f"{script} {' '.join(args)} が終了コード {proc.returncode} で終わった\n"
        + "\n".join(failed)
        + f"\n--- 末尾 ---\n{tail}",
        pytrace=False,
    )
