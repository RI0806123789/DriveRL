"""WebSocket の配信を接続ごとに分けたか（遅い接続が他を止めない・順序・未送信の上限）を検査する。"""

from __future__ import annotations

import ast
import asyncio
import time
from pathlib import Path

import orjson

from app.connections import ConnectionManager, frame_carries_route

MAIN_PY = Path(__file__).resolve().parents[1] / "app" / "main.py"


class FakeSocket:
    """送った通を記録する。`gate` を閉じている間は send_text が返らない（読み取りの遅い相手）。"""

    def __init__(self, *, stall: bool = False, fail: bool = False) -> None:
        self.sent: list[dict] = []
        self.sent_at: list[float] = []
        self.closed_with: int | None = None
        self.gate = asyncio.Event()
        if not stall:
            self.gate.set()
        self.fail = fail
        self.started = 0

    async def send_text(self, data: str) -> None:
        self.started += 1
        if self.fail:
            raise RuntimeError("disconnected")
        await self.gate.wait()
        self.sent.append(orjson.loads(data))
        self.sent_at.append(time.perf_counter())

    async def close(self, code: int = 1000) -> None:
        self.closed_with = code


async def settle(rounds: int = 5) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)


def types(ws: FakeSocket) -> list[str]:
    return [m["type"] + (f":{m['n']}" if "n" in m else "") for m in ws.sent]


def test_slow_connections_do_not_delay_healthy_one():
    async def main():
        manager = ConnectionManager(send_timeout=0.2)
        slow = [FakeSocket(stall=True), FakeSocket(stall=True)]
        healthy = FakeSocket()
        for ws in (*slow, healthy):
            manager.add(ws)
        started = time.perf_counter()
        for n in range(3):
            await manager.broadcast({"type": "metrics", "n": n})
        returned = time.perf_counter() - started
        await settle()
        delivered = healthy.sent_at[-1] - started
        assert types(healthy) == ["metrics:0", "metrics:1", "metrics:2"]
        # 以前は遅い接続ごとに時間切れ（ここでは 0.2 秒）まで待ってから次の接続へ進んでいた
        assert returned < 0.02, returned
        assert delivered < 0.05, delivered

        await asyncio.sleep(0.3)
        assert [ws.closed_with for ws in slow] == [1011, 1011]
        assert manager.count == 1
        await manager.broadcast({"type": "metrics", "n": 3})
        await settle()
        assert types(healthy)[-1] == "metrics:3"
        await manager.close_all()

    asyncio.run(main())


def test_order_is_kept_within_a_connection():
    async def main():
        manager = ConnectionManager()
        ws = FakeSocket()
        manager.add(ws)
        manager.send(ws, {"type": "init"})
        manager.send(ws, {"type": "detector"})
        await manager.broadcast({"type": "map"})
        await manager.broadcast({"type": "frame", "n": 1}, replaceable=True)
        manager.send(ws, {"type": "pong"})
        await manager.broadcast({"type": "status"})
        await settle(10)
        assert types(ws) == ["init", "detector", "map", "frame:1", "pong", "status"]
        await manager.close_all()

    asyncio.run(main())


def test_unsent_frames_are_replaced_but_routes_and_other_messages_are_kept():
    async def main():
        manager = ConnectionManager(send_timeout=5.0)
        ws = FakeSocket(stall=True)
        manager.add(ws)
        await manager.broadcast({"type": "frame", "n": 0}, replaceable=True)
        await settle()
        assert ws.started == 1  # 0 番は送っている最中（もう差し替えられない）
        for n in range(1, 4):
            await manager.broadcast({"type": "frame", "n": n}, replaceable=True)
        await manager.broadcast({"type": "frame", "n": 4, "route": True})
        await manager.broadcast({"type": "taxi", "n": 5})
        for n in range(6, 40):
            await manager.broadcast({"type": "frame", "n": n}, replaceable=True)
        await manager.broadcast({"type": "metrics", "n": 40})
        await manager.broadcast({"type": "frame", "n": 41}, replaceable=True)
        assert manager.pending(ws) == 4
        ws.gate.set()
        await settle(20)
        assert types(ws) == ["frame:0", "frame:4", "taxi:5", "metrics:40", "frame:41"]
        await manager.close_all()

    asyncio.run(main())


def test_pending_messages_are_bounded():
    async def main():
        manager = ConnectionManager(send_timeout=5.0, max_pending_messages=8)
        slow = FakeSocket(stall=True)
        healthy = FakeSocket()
        manager.add(slow)
        manager.add(healthy)
        await manager.broadcast({"type": "status", "n": -1})
        await settle()
        # 配信ループと同じく 1 通ごとに譲る（譲らずに上限を超えて積むと、健全な接続も送る前に切れる）
        for n in range(30):
            await manager.broadcast({"type": "status", "n": n})
            assert manager.pending(slow) <= 8
            await settle()
        await settle()
        assert slow.closed_with == 1011
        assert manager.count == 1
        assert len(healthy.sent) == 31
        await manager.close_all()

    asyncio.run(main())


def test_pending_size_is_bounded_but_a_large_message_is_always_accepted():
    async def main():
        manager = ConnectionManager(send_timeout=5.0, max_pending_chars=1000)
        ws = FakeSocket(stall=True)
        manager.add(ws)
        # 新しい接続は init から map まで譲らずに積む。map が上限より大きくても断らない
        manager.send(ws, {"type": "init"})
        manager.send(ws, {"type": "map", "blob": "x" * 5000})
        assert manager.count == 1
        await settle()
        assert manager.pending(ws) == 1  # init は送っている最中、map は未送信
        await manager.broadcast({"type": "status"})  # 未送信 5,000 文字 > 1,000 なので断って切る
        await settle()
        assert ws.closed_with == 1011
        assert manager.count == 0

        other = FakeSocket(stall=True)
        manager.add(other)
        await manager.broadcast({"type": "map", "blob": "y" * 5000})
        await settle()  # 1 通目は送っている最中なので数えない
        await manager.broadcast({"type": "status"})
        await manager.broadcast({"type": "map", "blob": "z" * 2000})
        assert manager.count == 1
        await manager.broadcast({"type": "status"})
        await settle()
        assert other.closed_with == 1011
        await manager.close_all()

    asyncio.run(main())


def test_send_error_drops_only_that_connection():
    async def main():
        manager = ConnectionManager()
        broken = FakeSocket(fail=True)
        healthy = FakeSocket()
        manager.add(broken)
        manager.add(healthy)
        await manager.broadcast({"type": "status"})
        await settle()
        assert manager.count == 1
        manager.send(broken, {"type": "pong"})  # 切断済みには積まない
        await manager.broadcast({"type": "status"})
        await settle()
        assert len(healthy.sent) == 2
        await manager.close_all()

    asyncio.run(main())


def test_remove_and_close_all_stop_sender_tasks():
    async def main():
        manager = ConnectionManager()
        ws = FakeSocket(stall=True)
        manager.add(ws)
        await manager.broadcast({"type": "status"})
        await settle()
        manager.remove(ws)
        assert manager.count == 0
        await manager.close_all()
        assert not manager._tasks

    asyncio.run(main())


def test_frame_carries_route():
    assert frame_carries_route({"vehicles": [{"id": 0}, {"id": 1, "route": [[0, 0]]}]})
    assert not frame_carries_route({"vehicles": [{"id": 0}, {"id": 1}]})
    assert not frame_carries_route({})


def _main_tree() -> ast.Module:
    return ast.parse(MAIN_PY.read_text(encoding="utf-8"))


def test_main_sends_only_through_the_manager():
    calls = [
        node.func.attr
        for node in ast.walk(_main_tree())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "send_text" not in calls
    assert "send_json" not in calls  # websocket.send_json も使わない（キューを迂回する）


def test_endpoint_queues_initial_messages_without_awaiting():
    tree = _main_tree()
    endpoint = next(
        node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "websocket_endpoint"
    )
    lines: dict[str, int] = {}
    for node in ast.walk(endpoint):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            name = f"{owner.id}.{node.func.attr}" if isinstance(owner, ast.Name) else node.func.attr
            lines.setdefault(name, node.lineno)
    start = lines["manager.add"]
    end = lines["engine.request_full_frame"]
    awaits = [node.lineno for node in ast.walk(endpoint) if isinstance(node, ast.Await) and start <= node.lineno <= end]
    assert awaits == [], f"init から map までの間に await がある（{awaits} 行目）"
