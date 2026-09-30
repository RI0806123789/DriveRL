"""Host と Origin の確認（app/host_guard.py）。DNS リバインディングと、別のサイトからの WebSocket・POST を断るか。"""

from __future__ import annotations

import asyncio

import pytest

from app.host_guard import HostOriginGuard, host_allowed, origin_allowed, split_host

ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


class TestHost:
    @pytest.mark.parametrize(
        "value, name",
        [("127.0.0.1:8000", "127.0.0.1"), ("[::1]:8000", "::1"), ("LocalHost", "localhost"), ("::1", "::1"), ("a.example.:80", "a.example")],
    )
    def test_split(self, value: str, name: str) -> None:
        assert split_host(value) == name

    @pytest.mark.parametrize("value", ["127.0.0.1:8000", "localhost:5173", "[::1]:8000", "192.168.1.5:8000", "10.0.0.2"])
    def test_ip_and_localhost_pass(self, value: str) -> None:
        assert host_allowed(value)

    @pytest.mark.parametrize("value", ["evil.example:8000", "localhost.evil.example", "", "127.0.0.1.nip.io:8000"])
    def test_names_are_rejected(self, value: str) -> None:
        assert not host_allowed(value)

    def test_extra_names_from_settings(self) -> None:
        assert host_allowed("mypc.local:8000", ["mypc.local"])
        assert not host_allowed("other.local:8000", ["mypc.local"])


class TestOrigin:
    def test_same_origin_and_dev_server(self) -> None:
        assert origin_allowed("http://127.0.0.1:8000", "127.0.0.1:8000", ORIGINS)
        assert origin_allowed("http://192.168.1.5:8000", "192.168.1.5:8000", ORIGINS)
        assert origin_allowed("http://localhost:5173", "127.0.0.1:8000", ORIGINS)

    @pytest.mark.parametrize(
        "origin", ["https://evil.example", "http://203.0.113.5", "http://127.0.0.1:9999", "null", "file://"]
    )
    def test_other_sites_are_rejected(self, origin: str) -> None:
        assert not origin_allowed(origin, "127.0.0.1:8000", ORIGINS)


def run(scope: dict, messages: list[dict] | None = None) -> tuple[list[dict], bool]:
    reached = []
    sent: list[dict] = []
    inbox = list(messages or [{"type": "websocket.connect"}])

    async def app(scope, receive, send):
        reached.append(scope)

    async def receive():
        return inbox.pop(0)

    async def send(message):
        sent.append(message)

    guard = HostOriginGuard(app, allowed_origins=ORIGINS)
    asyncio.run(guard(scope, receive, send))
    return sent, bool(reached)


def headers(**kw: str) -> list[tuple[bytes, bytes]]:
    return [(k.encode(), v.encode()) for k, v in kw.items()]


class TestMiddleware:
    def test_websocket_from_another_site_is_closed(self) -> None:
        sent, reached = run({"type": "websocket", "path": "/ws", "headers": headers(host="127.0.0.1:8000", origin="https://evil.example")})
        assert not reached
        assert sent == [{"type": "websocket.close", "code": 1008}]

    def test_websocket_from_the_app_passes(self) -> None:
        _, reached = run({"type": "websocket", "path": "/ws", "headers": headers(host="127.0.0.1:8000", origin="http://127.0.0.1:8000")})
        assert reached

    def test_rebound_name_is_rejected_even_without_origin(self) -> None:
        sent, reached = run({"type": "http", "method": "GET", "path": "/api/export/checkpoint", "headers": headers(host="evil.example:8000")})
        assert not reached
        assert sent[0]["status"] == 403

    def test_cross_site_post_is_rejected(self) -> None:
        sent, reached = run({"type": "http", "method": "POST", "path": "/api/import", "headers": headers(host="127.0.0.1:8000", origin="https://evil.example")})
        assert not reached and sent[0]["status"] == 403

    def test_get_and_scripts_without_origin_pass(self) -> None:
        assert run({"type": "http", "method": "GET", "path": "/", "headers": headers(host="127.0.0.1:8000", origin="https://evil.example")})[1]
        assert run({"type": "http", "method": "POST", "path": "/api/import", "headers": headers(host="127.0.0.1:8000")})[1]

    def test_lifespan_is_untouched(self) -> None:
        assert run({"type": "lifespan"})[1]
