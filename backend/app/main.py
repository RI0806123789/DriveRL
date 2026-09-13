"""FastAPI アプリケーション本体。

役割は 3 つだけ：

1. WebSocket 接続を受け付け、`docs/protocol.md` のメッセージを送受信する
2. クライアントからのコマンドを `SimulationEngine` へ受け渡す
3. エンジンが持つ最新スナップショットを一定周期で全接続へ配信する

重い処理（OSM の取得、物理、学習）はここには一切書かない。asyncio のイベントループを
止めないことが最優先で、地図取得は `asyncio.to_thread`、物理と学習は専用スレッド
（`app.runtime.engine`）が担当する。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import PurePath
from typing import Any

import orjson
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app import config
from app.contracts import InterventionEvent
from app.contracts import coerce_bool, validate_hidden_sizes
from app.runtime.engine import SimulationEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("autoware_sim")

engine = SimulationEngine()

# 介入イベントとしてそのまま流せるメッセージ種別
_EVENT_KINDS = {
    "spawn_vehicle",
    "despawn_vehicle",
    "add_obstacle",
    "remove_obstacle",
    "clear_obstacles",
    "reset_episode",
}

# エンジンへ素通しするコマンド
_COMMAND_KINDS = {"save_checkpoint", "load_checkpoint", "reset_policy"}

# モデル書き出しの待ち時間の上限 [s]。PPO 更新中に依頼が来ると 1 ステップ分待たされる。
EXPORT_TIMEOUT_SEC = 60.0

# モデル読み込みの待ち時間の上限 [s]。検証・バックアップ・載せ替えを含む。
IMPORT_TIMEOUT_SEC = 120.0

# 実行中のバックグラウンドタスク。
# asyncio はタスクへの強参照を持たないので、ここに入れておかないと実行途中で
# GC に回収されうる（マップ読込は数十秒かかるため最も起こりやすい）。
_background_tasks: set[asyncio.Task[Any]] = set()


def _spawn_background(coro: Any, *, name: str) -> "asyncio.Task[Any]":
    """待たないタスクを、参照を保持したまま起動する。"""
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _safe_upload_name(name: str | None) -> str:
    """アップロードされたファイル名をそのまま使わず、安全な形に落とす。

    ディレクトリ区切りや `..` を含む名前を信用すると、保存先を抜け出して
    任意の場所へ書き込まれてしまう。
    """
    base = PurePath(name or "model.pt").name
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", base).strip("-.")
    return (cleaned or "model.pt")[:120]


# ---------------------------------------------------------------------------
# 接続管理
# ---------------------------------------------------------------------------

#: 1 接続への送信をここで打ち切る [秒]（code_review R-05）。
#: フレームは 20Hz、金沢の map は 18.5MB。ローカル接続で 5 秒かかるのは
#: 「クライアントが読んでいない」以外に考えにくいので、切って他を守る。
_SEND_TIMEOUT_SEC = 5.0


class ConnectionManager:
    """接続中の WebSocket をまとめて扱う。

    memo 5章の通り単一ユーザー想定だが、ページのリロードで一時的に 2 本になることは
    あるので集合で持っておく。
    """

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def add(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.add(websocket)

    async def remove(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    @property
    def count(self) -> int:
        return len(self._connections)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        """全接続へ 1 通送る。**読み取りの遅い接続では時間切れで切断する。**

        ★ タイムアウトを外さないこと（code_review R-05）。素の
          `await ws.send_text(...)` には上限が無いので、受信を止めた
          クライアントが 1 本いると TCP の送信ウィンドウが埋まって
          `drain()` が無期限に待つ。すると配信ループ全体が止まり、
          **正常なクライアントにも何も届かなくなる**（学習は裏で進むので
          画面だけ凍る）。`handle_load_map` もここを await するのでマップ読込も
          止まり、詰まっている間は例外が出ないので掃除もされない。
          CLAUDE.md が勧める検証手順（`websockets` で /ws に繋ぐ）で
          `recv()` を回さないスクリプトを繋ぎっぱなしにすると数秒で再現する。
        """
        if not self._connections:
            return
        text = orjson.dumps(payload).decode("utf-8")
        async with self._lock:
            targets = list(self._connections)
        dead: list[WebSocket] = []
        for ws in targets:
            try:
                await asyncio.wait_for(ws.send_text(text), timeout=_SEND_TIMEOUT_SEC)
            except TimeoutError:
                logger.warning(
                    "WebSocket への送信が %.1f 秒で完了しませんでした。"
                    "読み取りが滞っている接続として切断します",
                    _SEND_TIMEOUT_SEC,
                )
                dead.append(ws)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)
            for ws in dead:
                try:
                    await ws.close(code=1011)
                except Exception:
                    pass  # 既に切れている接続。閉じられなくても掃除は済んでいる


manager = ConnectionManager()


async def send_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
    await websocket.send_text(orjson.dumps(payload).decode("utf-8"))


# ---------------------------------------------------------------------------
# 配信ループ
# ---------------------------------------------------------------------------


async def broadcast_loop() -> None:
    """フレーム・指標・通知を全接続へ配信する常駐タスク。

    フレームは「エンジンが新しいものを作ったときだけ」送る。`render_paused` のときは
    エンジンが None を返すので自然に配信が止まる（学習は裏で進み続ける）。
    """
    last_seq = -1
    last_metrics_at = 0.0
    poll_interval = 1.0 / 60.0
    metrics_interval = 1.0 / config.METRICS_HZ

    while True:
        try:
            await asyncio.sleep(poll_interval)

            if manager.count == 0:
                continue

            last_seq, frame = engine.take_frame(last_seq)
            if frame is not None:
                await manager.broadcast({"type": "frame", **frame.to_wire()})

            now = time.perf_counter()
            if now - last_metrics_at >= metrics_interval:
                last_metrics_at = now
                await manager.broadcast({"type": "metrics", **engine.metrics_payload()})
                network = engine.network_payload()
                if network is not None:
                    await manager.broadcast({"type": "network", **network})

            # エンジン側でパラメータが動いたとき（手動スポーンで台数が増えた等）は
            # そのまま配信する。送らないと UI の「車両数」が実態とずれたままになり、
            # 次に利用者がスライダーを触った瞬間に足した車両が消える
            changed_params = engine.take_params_update()
            if changed_params is not None:
                await manager.broadcast(
                    {"type": "params", "params": changed_params.to_wire()}
                )

            for notice in engine.drain_notices():
                payload = {"type": "status", **engine.status_payload()}
                payload["message"] = notice["message"]
                await manager.broadcast(payload)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("配信ループで例外が発生しました")
            await asyncio.sleep(0.5)


# ---------------------------------------------------------------------------
# マップ読み込み
# ---------------------------------------------------------------------------

_map_load_lock = asyncio.Lock()

# 直近に配信した map メッセージ。あとから接続してきたクライアントへ再送するために保持する
# （これが無いと、マップ読込後にページをリロードしたとき 3D が空のままになる）。
_current_map_wire: dict[str, Any] | None = None


def _load_map_blocking(preset_id: str):
    """別スレッドで実行される同期処理。OSM 取得とインデックス構築。"""
    # 依存が重いのでここで初めて読み込む
    from app.map import build_map_index, get_preset, load_map

    preset = get_preset(preset_id)
    if preset is None:
        raise ValueError(f"未知のプリセット: {preset_id}")
    data = load_map(preset)
    index = build_map_index(data)
    return preset, data, index


async def handle_load_map(preset_id: str) -> None:
    """マップ読み込みを非同期に進め、完了したらエンジンへ差し込む。"""
    if _map_load_lock.locked():
        await manager.broadcast(
            {
                "type": "status",
                **engine.status_payload(),
                "message": "別のマップを読み込み中です。完了までお待ちください",
            }
        )
        return

    global _current_map_wire

    async with _map_load_lock:
        from app.map import get_preset

        preset = get_preset(preset_id)
        if preset is None:
            await manager.broadcast(
                {"type": "error", "code": "MAP_LOAD_FAILED", "message": f"未知のプリセット: {preset_id}"}
            )
            return

        engine.set_loading(preset.id, preset.name)
        await manager.broadcast({"type": "status", **engine.status_payload()})

        started = time.perf_counter()
        try:
            preset, data, index = await asyncio.to_thread(_load_map_blocking, preset_id)
        except Exception as exc:
            logger.exception("マップの読み込みに失敗しました: %s", preset_id)
            engine.set_error(f"マップの読み込みに失敗しました: {exc}")
            await manager.broadcast(
                {
                    "type": "error",
                    "code": "MAP_LOAD_FAILED",
                    "message": f"マップの読み込みに失敗しました: {exc}",
                }
            )
            await manager.broadcast({"type": "status", **engine.status_payload()})
            return

        elapsed = time.perf_counter() - started
        logger.info(
            "マップ %s を読み込みました（ノード %d / エッジ %d / 建物 %d, %.1f 秒）",
            preset.id,
            len(data.nodes),
            len(data.edges),
            len(data.buildings),
            elapsed,
        )

        # ★ ワイヤ変換もイベントループから追い出す（code_review R-01）。
        #   全ノード・全エッジ・全建物の点に round() を回す純 Python ループで、
        #   金沢（ノード 19,493 / 建物 35,607）で実測 781.7ms。この間フレーム配信・
        #   metrics・ping/pong・他クライアントの HTTP がすべて止まる。
        #   銀座は 6.9ms なので、400m プリセットしか触っていないと気づけない。
        wire = await asyncio.to_thread(data.to_wire)

        # ★ フレーム配信は map を**送り始める前**に止める（code_review R-02）。
        #   `engine.set_map()` の中で止めていたときは、18.5MB の map を送っている
        #   最中に配信ループが**古いマップのフレームを map の後ろへ追記**していた。
        #   止めるのは broadcast を始める前でなければ意味が無い（この broadcast 自体が
        #   await なので、その間に配信ループが走る）。
        engine.pause_frames()
        _current_map_wire = wire
        await manager.broadcast({"type": "map", **_current_map_wire})
        engine.set_map(index, preset.id, preset.name)

        # エンジンがマップを取り込むまで少しだけ待ってから status を送る
        await asyncio.sleep(0.15)
        await manager.broadcast({"type": "status", **engine.status_payload()})


# ---------------------------------------------------------------------------
# クライアントメッセージの処理
# ---------------------------------------------------------------------------


async def handle_client_message(websocket: WebSocket, message: dict[str, Any]) -> None:
    kind = message.get("type")

    if kind == "ping":
        await send_json(websocket, {"type": "pong", "t": int(time.time() * 1000)})
        return

    if kind == "load_map":
        preset_id = message.get("presetId")
        if not isinstance(preset_id, str):
            await send_json(
                websocket,
                {"type": "error", "code": "INVALID_MESSAGE", "message": "presetId が指定されていません"},
            )
            return
        _spawn_background(handle_load_map(preset_id), name=f"load-map-{preset_id}")
        return

    if kind == "set_params":
        patch = message.get("params")
        if not isinstance(patch, dict):
            await send_json(
                websocket,
                {"type": "error", "code": "INVALID_MESSAGE", "message": "params が不正です"},
            )
            return
        params, patch_result = engine.update_params(patch)
        if patch_result.has_problem:
            # 黙って捨てると「動かしたのに効かない」としか分からない。
            # 反映した値は下の params メッセージで返すので、UI はそちらで直る
            details: list[str] = []
            if patch_result.rejected:
                details.append(f"無視した項目: {', '.join(patch_result.rejected)}")
            if patch_result.clamped:
                details.append(f"範囲内に丸めた項目: {', '.join(patch_result.clamped)}")
            await send_json(
                websocket,
                {
                    "type": "error",
                    "code": "INVALID_MESSAGE",
                    "message": "params の値が許容範囲外です。" + " / ".join(details),
                },
            )
        await manager.broadcast({"type": "params", "params": params.to_wire()})
        return

    if kind == "set_render_paused":
        # ★ `bool()` に任せないこと（code_review R-09）。`bool("false")` は True に
        #   なるので、文字列 `"false"` が**一時停止**として通っていた。
        #   必須項目の欠落は protocol.md 2.8 どおり INVALID_MESSAGE で返す。
        paused = coerce_bool(message.get("paused"))
        if paused is None:
            await send_json(
                websocket,
                {
                    "type": "error",
                    "code": "INVALID_MESSAGE",
                    "message": "paused には真偽値を指定してください",
                },
            )
            return
        engine.set_render_paused(paused)
        payload = {"type": "status", **engine.status_payload()}
        payload["message"] = (
            "描画を一時停止しました（学習は継続しています）" if paused else "描画を再開しました"
        )
        await manager.broadcast(payload)
        return

    if kind in _EVENT_KINDS:
        payload = {k: v for k, v in message.items() if k != "type"}
        engine.submit_event(InterventionEvent(kind=kind, payload=payload))
        return

    if kind == "set_network":
        sizes, why = validate_hidden_sizes(message.get("hiddenSizes"))
        if sizes is None:
            await send_json(
                websocket,
                {"type": "error", "code": "INVALID_MESSAGE", "message": why},
            )
            return
        engine.set_hidden_sizes(sizes)
        return

    if kind in _COMMAND_KINDS:
        engine.command(kind)
        return

    await send_json(
        websocket,
        {"type": "error", "code": "INVALID_MESSAGE", "message": f"未知のメッセージ種別: {kind}"},
    )


# ---------------------------------------------------------------------------
# アプリケーション
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.start()
    task = asyncio.create_task(broadcast_loop(), name="broadcast-loop")
    logger.info("サーバーを起動しました（http://%s:%d）", config.HOST, config.PORT)
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        engine.stop()


app = FastAPI(title="DriveRL", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "protocolVersion": config.PROTOCOL_VERSION,
            "connections": manager.count,
            "engine": engine.status_payload(),
        }
    )


@app.get("/api/export/{kind}")
async def export_model_endpoint(kind: str):
    """学習済みモデルを書き出してダウンロードさせる。

    kind:
        "checkpoint"  — 重み＋オプティマイザ状態＋メタデータ。このアプリに読み戻せる
        "torchscript" — 推論だけを切り出した自己完結形式。torch.jit.load() で読める

    実際の書き出しはエンジンスレッドがステップ境界で行う（学習中の中途半端な重みを
    掴まないため）。ここはその完了を待つだけで、イベントループはブロックしない。
    """
    from app.rl.export import EXPORT_KINDS

    if kind not in EXPORT_KINDS:
        return JSONResponse(
            {"error": f"未知の書き出し形式です: {kind}", "supported": list(EXPORT_KINDS)},
            status_code=400,
        )

    if kind == "keras":
        # `import keras` は数秒かかる。書き出し自体はシミュレーションスレッドの
        # ステップ境界で行うので、そこで初回インポートすると学習が数秒止まる。
        # 先にこのリクエストのスレッドで済ませておく。
        from app.rl.export import ExportError, preload_keras

        try:
            await asyncio.to_thread(preload_keras)
        except ExportError as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    ticket = engine.request_export(kind)
    finished = await asyncio.to_thread(ticket.done.wait, EXPORT_TIMEOUT_SEC)

    if not finished:
        return JSONResponse(
            {"error": "書き出しが時間内に完了しませんでした。少し待って再試行してください"},
            status_code=504,
        )
    if ticket.error is not None or ticket.result is None:
        return JSONResponse(
            {"error": ticket.error or "モデルの書き出しに失敗しました"}, status_code=500
        )

    result = ticket.result
    return FileResponse(
        path=str(result.path),
        filename=result.filename,
        media_type=result.media_type,
        headers={
            # フロントがサイズと形式を読めるようにしておく
            "X-Export-Kind": result.kind,
            "X-Export-Size": str(result.size_bytes),
        },
    )


@app.post("/api/import")
async def import_model_endpoint(file: UploadFile = File(...)):
    """書き出したモデルを受け取り、その状態から学習を再開する。

    受け取ったファイルは **ユーザーがアップロードしたもの** なので、
    `app.rl.importer` が `weights_only=True` で安全に解析してから載せ替える。

    実際の載せ替えはエンジンスレッドのステップ境界で行い、その直前に
    現在のモデルを `backend/data/exports/` へ自動バックアップする。

    受け取ったファイルは **成否によらず必ず消す**。読み込みが済めば用済みで、
    失敗したもの（形式違い・空・サイズ超過で打ち切った部分ファイル）を
    残しておく理由も無い。載せ替え前の状態は `data/exports/` の
    `..._before-import_...pt` に退避されているので、これで失うものは無い。
    """
    from app.rl.importer import MAX_UPLOAD_BYTES

    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_name = _safe_upload_name(file.filename)
    dest = config.UPLOAD_DIR / f"{stamp}_{safe_name}"

    # サイズ上限を見ながら少しずつ書き出す（丸ごとメモリに載せない）
    written = 0
    try:
        try:
            with dest.open("wb") as out:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        return JSONResponse(
                            {
                                "ok": False,
                                "error": f"ファイルが大きすぎます。上限は "
                                f"{MAX_UPLOAD_BYTES // 1024 // 1024} MB です",
                            },
                            status_code=413,
                        )
                    out.write(chunk)
        except Exception as exc:
            logger.exception("アップロードの保存に失敗しました")
            return JSONResponse(
                {"ok": False, "error": f"アップロードの保存に失敗しました: {exc}"},
                status_code=500,
            )
        finally:
            await file.close()

        if written == 0:
            return JSONResponse({"ok": False, "error": "ファイルが空です"}, status_code=400)

        ticket = engine.request_import(dest)
        finished = await asyncio.to_thread(ticket.done.wait, IMPORT_TIMEOUT_SEC)

        if not finished:
            return JSONResponse(
                {"ok": False, "error": "読み込みが時間内に完了しませんでした"}, status_code=504
            )
        if ticket.error is not None or ticket.info is None:
            # 形が合わないなどの「利用者が直せる」失敗なので 400 で返す
            return JSONResponse(
                {"ok": False, "error": ticket.error or "モデルの読み込みに失敗しました"},
                status_code=400,
            )

        payload: dict[str, Any] = {
            "ok": True,
            "filename": file.filename,
            "sizeBytes": written,
            "checkpoint": ticket.info.to_wire(),
            "backup": (
                {
                    "filename": ticket.backup.filename,
                    "sizeBytes": ticket.backup.size_bytes,
                }
                if ticket.backup is not None
                else None
            ),
            "message": f"学習回数 {ticket.info.updates} 回の状態から学習を再開します",
        }

        # 他の接続にも状態変化を知らせる
        await manager.broadcast({"type": "status", **engine.status_payload()})
        return JSONResponse(payload)

    finally:
        # 通常はエンジンスレッドが読み終えた後（ticket.done 待ちの後）なので、
        # 掴まれたまま消すことはない。時間切れ（504）だけは engine が後から
        # 読みに来る可能性があるが、その場合は「ファイルが見つかりません」で
        # 失敗するだけで、消せなければ下の警告が出る
        try:
            dest.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "アップロードファイルを削除できませんでした: %s（%s）", dest.name, exc
            )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    await manager.add(websocket)
    logger.info("WebSocket 接続を受け付けました（接続数 %d）", manager.count)

    try:
        # ★ `app.map` ではなく `app.map.presets` から引く（code_review R-08）。
        #   パッケージ側も遅延化したので実害は消えているが、ここは
        #   「プリセット一覧しか要らない」ことを import で明示しておく
        from app.map.presets import list_presets

        presets = [p.to_wire() for p in list_presets()]
    except Exception:
        logger.exception("プリセット一覧の取得に失敗しました")
        presets = []

    try:
        await send_json(
            websocket,
            {
                "type": "init",
                "protocolVersion": config.PROTOCOL_VERSION,
                "presets": presets,
                "config": {
                    "maxVehicles": config.MAX_VEHICLES,
                    "simHz": config.SIM_HZ,
                    "obsDim": config.OBS_DIM,
                    "actionDim": config.ACTION_DIM,
                },
                "params": engine.snapshot_params().to_wire(),
                "status": engine.status_payload(),
            },
        )

        # 接続時点で既にマップが読み込まれていれば、それも送って画面を復元させる。
        # （ページをリロードしただけで 3D が空になるのを防ぐ）
        if _current_map_wire is not None:
            await send_json(websocket, {"type": "map", **_current_map_wire})
            # 経路は差分配信なので、次のフレームだけ全スロット分を載せてもらう
            engine.request_full_frame()

        while True:
            raw = await websocket.receive_text()
            try:
                message = orjson.loads(raw)
            except orjson.JSONDecodeError:
                await send_json(
                    websocket,
                    {"type": "error", "code": "INVALID_MESSAGE", "message": "JSON として解釈できません"},
                )
                continue
            if not isinstance(message, dict):
                await send_json(
                    websocket,
                    {"type": "error", "code": "INVALID_MESSAGE", "message": "オブジェクトを送ってください"},
                )
                continue
            await handle_client_message(websocket, message)

    except WebSocketDisconnect:
        logger.info("WebSocket が切断されました")
    except Exception:
        logger.exception("WebSocket ハンドラで例外が発生しました")
    finally:
        await manager.remove(websocket)


# 本番用：フロントエンドをビルド済みなら静的配信する。
# 開発時は Vite の dev サーバー（5173）を使うのでここは通らない。
_dist_dir = config.PROJECT_DIR / "frontend" / "dist"

import os as _os  # noqa: E402

from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.responses import Response as _Response  # noqa: E402
from starlette.types import Scope as _Scope  # noqa: E402


class _FrontendStatic(StaticFiles):
    """index.html だけキャッシュさせない静的配信。

    Vite の出力は `index-<ハッシュ>.js` のようにファイル名が内容で変わるので、
    アセットは永続キャッシュして構わない。一方 index.html はファイル名が固定なので、
    ブラウザにキャッシュされると **再ビルドしても古いバンドルを指したまま**になる
    （実際に、古い index.html が既に消えた JS を参照し続ける事故が起きた）。
    そのため HTML だけ毎回サーバーへ確認させる。
    """

    def file_response(
        self,
        full_path: "_os.PathLike[str] | str",
        stat_result: _os.stat_result,
        scope: _Scope,
        status_code: int = 200,
    ) -> _Response:
        response = super().file_response(full_path, stat_result, scope, status_code)
        name = _os.fspath(full_path).replace("\\", "/")
        if name.endswith(".html"):
            response.headers["Cache-Control"] = "no-cache"
        elif "/assets/" in name:
            # 内容が変わればファイル名が変わるので、長期キャッシュして問題ない
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


if _dist_dir.is_dir():
    app.mount("/", _FrontendStatic(directory=str(_dist_dir), html=True), name="frontend")
    logger.info("ビルド済みフロントエンドを配信します: %s", _dist_dir)
