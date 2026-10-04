"""学習タブの設定値（ハイパーパラメータ・報酬の重み・スイッチ）をディスクに控え、起動時に戻す。"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from app.contracts import SimParams
from app.runtime.autotune import atomic_write_json
from app.warn import warn_once

#: 保存する params のキー（ワイヤ名）。学習タブのスライダーとスイッチだけ。台数・天候などは対象外
PERSISTED_WIRE_KEYS: tuple[str, ...] = (
    "learningRate",
    "gamma",
    "clipRange",
    "entropyCoef",
    "rewardGoal",
    "rewardCollision",
    "rewardProgress",
    "rewardOffroad",
    "rewardSignal",
    "rewardOverspeed",
    "rewardTime",
    "onlineAssist",
    "incidentCurriculum",
    "v2xComm",
)


class ParamStore:
    """学習タブの設定値を 1 つの JSON に控える。書き込みは呼び出し側のスレッドで行う（小さいファイルなので軽い）。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def load(self) -> dict[str, Any]:
        """控えた値（ワイヤ名）。無い・読めないときは空。壊れたファイルは握りつぶさず初回だけログに残す。"""
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            warn_once("runtime.param_store.load", f"{self.path.name} を読めないため、設定値は既定から始めます")
            return {}
        if not isinstance(raw, dict):
            warn_once("runtime.param_store.load", f"{self.path.name} の形が不正なため、設定値は既定から始めます")
            return {}
        return {key: raw[key] for key in PERSISTED_WIRE_KEYS if key in raw}

    def restore_into(self, params: SimParams) -> list[str]:
        """控えた値を検証して `params` へ戻す。戻したキー（ワイヤ名）を返す。範囲外は丸め、読めない値は既定のまま。"""
        saved = self.load()
        if not saved:
            return []
        params.apply_wire(saved)
        return list(saved)

    def save(self, patch: dict[str, Any]) -> bool:
        """`patch` のうち保存対象のキーを、いまの控えへ混ぜて書く。書けなければ False（初回だけログ）。"""
        wanted = {key: patch[key] for key in PERSISTED_WIRE_KEYS if key in patch}
        if not wanted:
            return True
        with self._lock:
            merged = {**self.load(), **wanted}
            try:
                atomic_write_json(self.path, merged)
            except OSError:
                warn_once(
                    "runtime.param_store.save",
                    f"{self.path.name} へ設定値を保存できませんでした。再起動すると既定値に戻ります",
                )
                return False
        return True
