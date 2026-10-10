"""WebSocket の 1 通を分岐の前に検証し、不正な通で受信ループを終わらせないか（#129）。main.py は import せず ast で読む。"""
from __future__ import annotations

import ast
import asyncio
import math
import time
from pathlib import Path
from typing import Any

import orjson
import pytest

from app import client_messages as cm
from app.contracts import InterventionEvent, coerce_bool, validate_hidden_sizes

MAIN_PY = Path(__file__).resolve().parents[1] / "app" / "main.py"

BAD_TYPES = [None, [], {}, 5, 1.5, True, "", "x" * (cm.MAX_TYPE_LENGTH + 1)]


@pytest.mark.parametrize("kind", BAD_TYPES, ids=repr)
def test_parse_rejects_non_string_types(kind) -> None:
    message, problem = cm.parse_client_message(orjson.dumps({"type": kind, "presetId": "ginza"}))
    assert message is None and problem == cm.INVALID_TYPE_MESSAGE


def test_parse_rejects_missing_type_bad_json_and_non_objects() -> None:
    assert cm.parse_client_message(b'{"presetId": "ginza"}') == (None, cm.INVALID_TYPE_MESSAGE)
    assert cm.parse_client_message("{not json") == (None, cm.INVALID_JSON_MESSAGE)
    for raw in ("[]", "5", '"ping"', "null"):
        assert cm.parse_client_message(raw) == (None, cm.NOT_OBJECT_MESSAGE)
    assert cm.parse_client_message('{"type": "ping"}') == ({"type": "ping"}, None)


def test_dispatch_never_raises_and_keeps_going(caplog) -> None:
    handled: list[str] = []
    replies: list[dict[str, Any]] = []

    async def handle(message):
        if message["type"] == "boom":
            raise RuntimeError("内部の失敗 C:/secret/path")
        handled.append(message["type"])

    async def reply(payload):
        replies.append(payload)

    async def main():
        results = []
        for raw in ('{"type": []}', '{"type": "boom"}', "oops", '{"type": "ping"}'):
            results.append(await cm.dispatch_client_text(raw, handle, reply))
        return results

    assert asyncio.run(main()) == [False, False, False, True]
    assert handled == ["ping"]
    assert [r["code"] for r in replies] == ["INVALID_MESSAGE"] * 3
    assert replies[1]["message"] == cm.HANDLER_FAILED_MESSAGE
    assert all("secret" not in r["message"] for r in replies), "応答に例外の文を載せない"
    assert "secret" in caplog.text, "例外はログに残す"


class _RecordingEngine:
    """呼ばれた属性を記録する偽のエンジン。不正な通では何も呼ばれないはず。"""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.detector_job = type("Job", (), {"running": False})()

    def autotune_running(self) -> bool:
        return False

    def __getattr__(self, name: str):
        def record(*args, **kwargs):
            self.calls.append(name)
            return None

        return record


def _load_handler():
    sent: list[dict[str, Any]] = []
    engine = _RecordingEngine()
    namespace: dict[str, Any] = {
        "Any": Any, "WebSocket": object, "math": math, "time": time, "engine": engine,
        "manager": type("M", (), {"send": staticmethod(lambda ws, payload: sent.append(payload))})(),
        "InterventionEvent": InterventionEvent, "coerce_bool": coerce_bool,
        "validate_hidden_sizes": validate_hidden_sizes, "TUNED_WIRE_KEYS": set(),
        "invalid_message": cm.invalid_message, "INVALID_TYPE_MESSAGE": cm.INVALID_TYPE_MESSAGE,
    }
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
    names = {"_EVENT_KINDS", "_COMMAND_KINDS", "_TAXI_COMMANDS"}
    body = [
        node for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {"handle_client_message", "send_json", "send_notice", "_detector_training_error", "_parse_point"})
        or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) in names for t in node.targets))
    ]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(MAIN_PY), "exec"), namespace)
    return namespace["handle_client_message"], sent, engine


@pytest.mark.parametrize("kind", [[], {}, 5, None, True], ids=repr)
def test_main_handler_rejects_bad_types_and_serves_the_next_ping(kind) -> None:
    handle, sent, engine = _load_handler()

    async def main():
        reply = lambda payload: _append(sent, payload)  # noqa: E731
        await cm.dispatch_client_text(orjson.dumps({"type": kind}), lambda m: handle(object(), m), reply)
        # 入口を通さずに直接呼ばれても TypeError を出さない
        await handle(object(), {"type": kind})
        await cm.dispatch_client_text('{"type": "ping"}', lambda m: handle(object(), m), reply)

    asyncio.run(main())
    assert [m["type"] for m in sent] == ["error", "error", "pong"]
    assert all(m["code"] == "INVALID_MESSAGE" for m in sent[:2])
    assert engine.calls == [], "不正な通をエンジンへ渡さない"


def test_main_handler_answers_unknown_string_types() -> None:
    handle, sent, engine = _load_handler()
    asyncio.run(handle(object(), {"type": "no_such_command"}))
    assert sent and sent[0]["code"] == "INVALID_MESSAGE" and "未知" in sent[0]["message"]
    assert engine.calls == []


def test_receive_loop_goes_through_the_validating_entry() -> None:
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
    endpoint = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "websocket_endpoint")
    loop = next(n for n in ast.walk(endpoint) if isinstance(n, ast.While))
    called = {getattr(c.func, "id", getattr(c.func, "attr", None)) for c in ast.walk(loop) if isinstance(c, ast.Call)}
    assert "dispatch_client_text" in called
    assert "handle_client_message" not in called, "受信ループから検証を飛ばして直接呼ばない"


async def _append(sink: list, payload) -> None:
    sink.append(payload)
