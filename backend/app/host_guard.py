"""Host と Origin の確認（DNS リバインディングと、別のサイトからの WebSocket・POST を断る ASGI ミドルウェア）。"""

from __future__ import annotations

import ipaddress
import logging
from typing import Any, Awaitable, Callable, Iterable
from urllib.parse import urlsplit

__all__ = [
    "HostOriginGuard",
    "host_allowed",
    "origin_allowed",
    "split_host",
]

logger = logging.getLogger(__name__)

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

#: 状態を変えるので、Origin を確かめるメソッド
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

LOCAL_NAMES = frozenset({"localhost"})


def split_host(value: str) -> str:
    """Host ヘッダ（`名前:ポート` / `[IPv6]:ポート`）から名前だけを小文字で取り出す。"""
    text = value.strip().lower()
    if text.startswith("["):
        end = text.find("]")
        return text[1:end] if end > 0 else ""
    if text.count(":") == 1:
        text = text.split(":", 1)[0]
    return text.rstrip(".")


def _is_ip(name: str) -> bool:
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


def host_allowed(host_header: str, extra_hosts: Iterable[str] = ()) -> bool:
    """IP アドレスそのもの・localhost・設定で足した名前だけを通す（DNS リバインディングは名前を使うため）。"""
    name = split_host(host_header)
    if not name:
        return False
    if _is_ip(name) or name in LOCAL_NAMES:
        return True
    return name in {split_host(h) for h in extra_hosts if h.strip()}


def origin_allowed(origin: str, host_header: str, allowed_origins: Iterable[str]) -> bool:
    """同じオリジン（Origin の `ホスト:ポート` が Host と同じ）か、許したオリジンだけを通す。"""
    text = origin.strip()
    if text in {o.rstrip("/") for o in allowed_origins}:
        return True
    parts = urlsplit(text)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return False
    return parts.netloc.lower() == host_header.strip().lower()


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            return value.decode("latin-1")
    return None


class HostOriginGuard:
    """HTTP と WebSocket の両方で Host を確かめ、WebSocket と状態を変える HTTP では Origin も確かめる。"""

    def __init__(
        self,
        app: ASGIApp,
        *,
        allowed_origins: Iterable[str],
        extra_hosts: Iterable[str] = (),
    ) -> None:
        self.app = app
        self.allowed_origins = tuple(allowed_origins)
        self.extra_hosts = tuple(extra_hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        kind = scope.get("type")
        if kind not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        host = _header(scope, b"host") or ""
        if not host_allowed(host, self.extra_hosts):
            await self._reject(scope, receive, send, "Host", host)
            return
        origin = _header(scope, b"origin")
        needs_origin = kind == "websocket" or str(scope.get("method", "")).upper() in UNSAFE_METHODS
        # Origin を付けないのはブラウザ以外（curl やスクリプト）だけなので通す。ブラウザは必ず付ける
        if needs_origin and origin is not None and not origin_allowed(origin, host, self.allowed_origins):
            await self._reject(scope, receive, send, "Origin", origin)
            return
        await self.app(scope, receive, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send, what: str, value: str) -> None:
        logger.warning("許していない %s からの接続を断りました: %r（%s %s）", what, value[:200], scope.get("type"), scope.get("path"))
        if scope.get("type") == "websocket":
            message = await receive()
            if message.get("type") == "websocket.connect":
                await send({"type": "websocket.close", "code": 1008})
            return
        body = '{"error":"このアドレスからの接続は許可されていません"}'.encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
