"""接続中の WebSocket への送信を接続ごとのキューに分け、遅い接続が他の接続の配信を止めないようにする。"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any, Protocol

import orjson

logger = logging.getLogger("autoware_sim")

#: 1 通の送信がこれを超えたら、読み取りが滞っている接続として切断する [秒]
SEND_TIMEOUT_SEC = 5.0
#: 切断の通知（close フレーム）を待つ上限 [秒]
CLOSE_TIMEOUT_SEC = 1.0
#: 1 接続に溜めてよい未送信の通数。置き換えられる frame は 1 通しか溜まらないので、届くのは状態の通知ばかり
MAX_PENDING_MESSAGES = 512
#: 未送信の文字数がこれを超えた接続には積まない（JSON はほぼ ASCII なのでバイト数とほぼ同じ）。溜まるのは最大でこれ + 1 通
MAX_PENDING_CHARS = 64 * 1024 * 1024

#: 切断の理由に付ける WebSocket の終了コード（Internal Error。以前と同じ）
CLOSE_CODE = 1011


class _Socket(Protocol):
    async def send_text(self, data: str) -> None: ...

    async def close(self, code: int = 1000) -> None: ...


class _Item:
    """キューの 1 通。比較は同一性で行う（deque.remove のため）。"""

    __slots__ = ("text", "replaceable")

    def __init__(self, text: str, replaceable: bool) -> None:
        self.text = text
        self.replaceable = replaceable


class _Outbox:
    """1 接続ぶんの送信キューと、それを順に送るタスク。"""

    def __init__(self, manager: ConnectionManager, ws: _Socket) -> None:
        self.manager = manager
        self.ws = ws
        self.queue: deque[_Item] = deque()
        self.pending_chars = 0
        self.replaceable: _Item | None = None
        self.wake = asyncio.Event()
        self.closed = False
        self.task = manager.spawn(self._run(), name="ws-sender")

    def put(self, text: str, *, replaceable: bool) -> None:
        if self.closed:
            return
        if replaceable and self.replaceable is not None:
            # まだ送っていない前の frame は誰にも読まれないので捨て、最新を末尾へ積む
            self.queue.remove(self.replaceable)
            self.pending_chars -= len(self.replaceable.text)
            self.replaceable = None
        # 文字数は積む前の未送信で見る（新しい 1 通の大きさでは断らない。map 1 通が上限より大きくても接続できる）
        if (
            len(self.queue) >= self.manager.max_pending_messages
            or self.pending_chars > self.manager.max_pending_chars
        ):
            self.abort(
                f"未送信が {len(self.queue)} 通・{self.pending_chars / 1e6:.1f}MB に溜まったため"
            )
            return
        item = _Item(text, replaceable)
        self.queue.append(item)
        self.pending_chars += len(text)
        if replaceable:
            self.replaceable = item
        self.wake.set()

    async def _run(self) -> None:
        while not self.closed:
            if not self.queue:
                self.wake.clear()
                await self.wake.wait()
                continue
            item = self.queue.popleft()
            self.pending_chars -= len(item.text)
            if item is self.replaceable:
                self.replaceable = None
            try:
                await asyncio.wait_for(self.ws.send_text(item.text), timeout=self.manager.send_timeout)
            except TimeoutError:
                self.abort(f"送信が {self.manager.send_timeout:.1f} 秒で完了しなかったため")
                return
            except Exception:
                # 相手が先に切った（WebSocketDisconnect など）。受信側のループも同じ切断で抜ける
                self.abort(None)
                return

    def abort(self, reason: str | None) -> None:
        """この接続への送信をやめ、接続を閉じる。どこから呼んでもよい（待たない）。"""
        if self.closed:
            return
        self.closed = True
        self.queue.clear()
        self.pending_chars = 0
        self.replaceable = None
        self.manager._forget(self)
        if reason is not None:
            logger.warning("WebSocket の読み取りが滞っている接続を切断します（%s）", reason)
        if self.task is not asyncio.current_task():
            self.task.cancel()
        self.manager.spawn(self._close(), name="ws-close")

    async def _close(self) -> None:
        try:
            await asyncio.wait_for(self.ws.close(code=CLOSE_CODE), timeout=CLOSE_TIMEOUT_SEC)
        except Exception:
            pass


class ConnectionManager:
    """接続中の WebSocket をまとめて扱う。送信は接続ごとのキューに積むだけで、送り終わりを待たない。"""

    def __init__(
        self,
        *,
        send_timeout: float = SEND_TIMEOUT_SEC,
        max_pending_messages: int = MAX_PENDING_MESSAGES,
        max_pending_chars: int = MAX_PENDING_CHARS,
    ) -> None:
        self.send_timeout = send_timeout
        self.max_pending_messages = max_pending_messages
        self.max_pending_chars = max_pending_chars
        self._outboxes: dict[Any, _Outbox] = {}
        self._tasks: set[asyncio.Task[Any]] = set()

    def spawn(self, coro: Any, *, name: str) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def add(self, websocket: _Socket) -> None:
        """接続を登録する。続けて await を挟まずに積んだ通は、これ以降の配信より先に届く。"""
        if websocket not in self._outboxes:
            self._outboxes[websocket] = _Outbox(self, websocket)

    def remove(self, websocket: _Socket) -> None:
        """切断した接続を外し、未送信の通を捨てる。"""
        outbox = self._outboxes.pop(websocket, None)
        if outbox is not None and not outbox.closed:
            outbox.closed = True
            outbox.queue.clear()
            outbox.task.cancel()

    def _forget(self, outbox: _Outbox) -> None:
        if self._outboxes.get(outbox.ws) is outbox:
            del self._outboxes[outbox.ws]

    @property
    def count(self) -> int:
        return len(self._outboxes)

    def pending(self, websocket: _Socket) -> int:
        """未送信の通数（送っている最中の 1 通は含まない）。"""
        outbox = self._outboxes.get(websocket)
        return 0 if outbox is None else len(outbox.queue)

    def send(self, websocket: _Socket, payload: dict[str, Any]) -> None:
        """1 接続へ 1 通積む。切断済みなら捨てる。"""
        outbox = self._outboxes.get(websocket)
        if outbox is not None:
            outbox.put(orjson.dumps(payload).decode("utf-8"), replaceable=False)

    async def broadcast(self, payload: dict[str, Any], *, replaceable: bool = False) -> None:
        """全接続へ 1 通積む。`replaceable` の通は、まだ送っていない前の置き換えられる通と差し替える。"""
        if not self._outboxes:
            return
        text = orjson.dumps(payload).decode("utf-8")
        for outbox in list(self._outboxes.values()):
            outbox.put(text, replaceable=replaceable)

    async def close_all(self) -> None:
        """送信タスクを止める（終了時）。"""
        for websocket in list(self._outboxes):
            self.remove(websocket)
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def frame_carries_route(wire: dict[str, Any]) -> bool:
    """frame が経路を運んでいるか。運んでいる通は置き換えない（経路は版が変わったときしか送らない）。"""
    return any("route" in vehicle for vehicle in wire.get("vehicles", ()))
