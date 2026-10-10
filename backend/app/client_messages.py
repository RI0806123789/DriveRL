"""WebSocket で受けた 1 通を、種類ごとの処理へ渡す前に検証する（不正な通で受信ループを終わらせない）。"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

import orjson

logger = logging.getLogger("autoware_sim")

INVALID_JSON_MESSAGE = "JSON として解釈できません"
NOT_OBJECT_MESSAGE = "オブジェクトを送ってください"
INVALID_TYPE_MESSAGE = "type には文字列のメッセージ種別を指定してください"
#: 種類ごとの処理が想定外の例外を出したときに返す文（例外の文は載せず、ログにだけ残す）
HANDLER_FAILED_MESSAGE = "メッセージを処理できませんでした"
#: メッセージ種別として受け付ける長さの上限（いちばん長い種別は 30 文字に届かない）
MAX_TYPE_LENGTH = 64


def invalid_message(text: str) -> dict[str, Any]:
    """`INVALID_MESSAGE` の `error` を作る（`docs/protocol.md` 2.8）。"""
    return {"type": "error", "code": "INVALID_MESSAGE", "message": text}


def parse_client_message(raw: str | bytes) -> tuple[dict[str, Any] | None, str | None]:
    """1 通を読む。読めれば (メッセージ, None)、読めなければ (None, 返す文)。`type` は空でない文字列だけを通す。"""
    try:
        message = orjson.loads(raw)
    except orjson.JSONDecodeError:
        return None, INVALID_JSON_MESSAGE
    if not isinstance(message, dict):
        return None, NOT_OBJECT_MESSAGE
    kind = message.get("type")
    if not isinstance(kind, str) or not kind or len(kind) > MAX_TYPE_LENGTH:
        return None, INVALID_TYPE_MESSAGE
    return message, None


async def dispatch_client_text(
    raw: str | bytes,
    handle: Callable[[dict[str, Any]], Awaitable[None]],
    reply: Callable[[dict[str, Any]], Awaitable[None]],
) -> bool:
    """1 通を検証して `handle` へ渡す。不正な通・処理の失敗は `reply` で知らせて False を返し、例外を外へ出さない。"""
    message, problem = parse_client_message(raw)
    if message is None:
        await reply(invalid_message(problem or INVALID_JSON_MESSAGE))
        return False
    try:
        await handle(message)
    except Exception:
        logger.exception("WebSocket のメッセージ（type=%r）の処理で例外が発生しました", message["type"])
        await reply(invalid_message(HANDLER_FAILED_MESSAGE))
        return False
    return True
