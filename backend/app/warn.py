"""握りつぶした失敗を初回だけログに残す（code_review B-15）。依存を持たない中立地帯。"""

from __future__ import annotations

import logging
import sys

__all__ = ["warn_once"]

logger = logging.getLogger("autoware_sim")

_WARNED: set[str] = set()


def warn_once(key: str, message: str) -> None:
    """鍵（`モジュール.何の失敗か`）ごとに初回だけ記録する。例外の処理中なら traceback も残す。"""
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.error(message, exc_info=sys.exc_info()[1] is not None)
