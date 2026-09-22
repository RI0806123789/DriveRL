"""FastAPI アプリケーション本体。"""

from __future__ import annotations

import asyncio
import logging
import math
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

_EVENT_KINDS = {
    "spawn_vehicle",
    "despawn_vehicle",
    "add_obstacle",
    "remove_obstacle",
    "clear_obstacles",
    "reset_episode",
}

_COMMAND_KINDS = {"save_checkpoint", "load_checkpoint", "reset_policy"}

_TAXI_COMMANDS = {"board_taxi": "board", "alight_taxi": "alight"}

EXPORT_TIMEOUT_SEC = 60.0

IMPORT_TIMEOUT_SEC = 120.0

_background_tasks: set[asyncio.Task[Any]] = set()


def _spawn_background(coro: Any, *, name: str) -> "asyncio.Task[Any]":
    """待たないタスクを、参照を保持したまま起動する。"""
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _safe_upload_name(name: str | None) -> str:
    """アップロードされたファイル名をそのまま使わず、安全な形に落とす。"""
    base = PurePath(name or "model.pt").name
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", base).strip("-.")
    return (cleaned or "model.pt")[:120]


_SEND_TIMEOUT_SEC = 5.0


class ConnectionManager:
    """接続中の WebSocket をまとめて扱う。"""

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
        """全接続へ 1 通送る。**読み取りの遅い接続では時間切れで切断する。**"""
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
                    pass


manager = ConnectionManager()


def _detector_training_error(
    what: str = "認識器の学習中は物理が止まっているため、配車できません",
) -> dict[str, Any]:
    """学習中に弾いたことを伝えるエラー応答を作る。"""
    return {
        "type": "error",
        "code": "DETECTOR_TRAINING",
        "message": f"{what}。中止するか、終わるまで待ってください",
    }


async def send_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
    await websocket.send_text(orjson.dumps(payload).decode("utf-8"))


async def broadcast_loop() -> None:
    """フレーム・指標・通知を全接続へ配信する常駐タスク。"""
    last_seq = -1
    last_taxi_seq = -1
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

            last_taxi_seq, taxi = engine.take_taxi(last_taxi_seq)
            if taxi is not None:
                await manager.broadcast({"type": "taxi", **taxi})

            now = time.perf_counter()
            if now - last_metrics_at >= metrics_interval:
                last_metrics_at = now
                await manager.broadcast({"type": "metrics", **engine.metrics_payload()})
                network = engine.network_payload()
                if network is not None:
                    await manager.broadcast({"type": "network", **network})

                await broadcast_detector_if_changed()

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


_last_detector_payload: dict[str, Any] | None = None


async def broadcast_detector(force: bool = False) -> None:
    """`detector` メッセージを全接続へ送る（中身が変わったときだけ）。"""
    global _last_detector_payload
    payload = engine.detector_job.snapshot()
    if not force and payload == _last_detector_payload:
        return
    _last_detector_payload = payload
    await manager.broadcast({"type": "detector", **payload})


async def broadcast_detector_if_changed() -> None:
    await broadcast_detector(force=False)


_map_load_lock = asyncio.Lock()

_current_map_wire: dict[str, Any] | None = None


def _load_map_blocking(preset_id: str):
    """別スレッドで実行される同期処理。OSM 取得とインデックス構築。"""
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

        wire = await asyncio.to_thread(data.to_wire)

        engine.pause_frames()
        _current_map_wire = wire
        await manager.broadcast({"type": "map", **_current_map_wire})
        engine.set_map(index, preset.id, preset.name)

        await asyncio.sleep(0.15)
        await manager.broadcast({"type": "status", **engine.status_payload()})


def _parse_point(value: Any) -> tuple[float, float] | None:
    """`[x, y]` を有限な座標として読む。読めなければ None。"""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


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
        if engine.detector_job.running:
            await send_json(
                websocket,
                _detector_training_error("認識器の学習中はエリアを変えられません"),
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

    if kind == "set_app_mode":
        mode = message.get("mode")
        if mode == "taxi" and engine.detector_job.running:
            await send_json(websocket, _detector_training_error())
            return
        if mode not in ("dev", "taxi"):
            await send_json(
                websocket,
                {
                    "type": "error",
                    "code": "INVALID_MESSAGE",
                    "message": 'mode には "dev" か "taxi" を指定してください',
                },
            )
            return
        engine.set_practical_mode(mode == "taxi")
        return

    if kind == "request_taxi":
        if engine.detector_job.running:
            await send_json(websocket, _detector_training_error())
            return
        pickup = _parse_point(message.get("pickup"))
        dropoff = _parse_point(message.get("dropoff"))
        if pickup is None or dropoff is None:
            await send_json(
                websocket,
                {
                    "type": "error",
                    "code": "INVALID_MESSAGE",
                    "message": "pickup / dropoff は [x, y] の有限な数値で指定してください",
                },
            )
            return
        engine.taxi_command(
            "request",
            {
                "pickupX": pickup[0],
                "pickupY": pickup[1],
                "dropoffX": dropoff[0],
                "dropoffY": dropoff[1],
            },
        )
        return

    if kind in _TAXI_COMMANDS:
        if kind == "board_taxi" and engine.detector_job.running:
            await send_json(websocket, _detector_training_error())
            return
        engine.taxi_command(_TAXI_COMMANDS[kind])
        return

    if kind == "player_pose":
        # 10Hz で届くので、値が読めないときは黙って街から消す。
        # エラーを返すと同じ頻度で返し続けることになる
        engine.submit_player_pose(_parse_point(message.get("at")))
        return

    if kind == "cancel_taxi":
        halt = coerce_bool(message.get("halt")) or False
        engine.taxi_command("halt" if halt else "cancel")
        return

    if kind == "start_detector_training":
        from app.runtime.detector_job import parse_request

        request, why = parse_request(message.get("request"))
        if request is None:
            await send_json(
                websocket,
                {"type": "error", "code": "INVALID_MESSAGE", "message": why},
            )
            return
        problem = engine.detector_job.start(request)
        if problem:
            await manager.broadcast(
                {"type": "status", **engine.status_payload(), "message": problem}
            )
        await broadcast_detector(force=True)
        return

    if kind == "cancel_detector_training":
        problem = engine.detector_job.cancel()
        if problem:
            await manager.broadcast(
                {"type": "status", **engine.status_payload(), "message": problem}
            )
        await broadcast_detector(force=True)
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


MAX_UPLOAD_FILES = 3


def _prune_uploads() -> None:
    """起動時にアップロードの残骸を片付ける（code_review E-07）。"""
    from app.rl.export import prune_exports

    try:
        removed = prune_exports(config.UPLOAD_DIR, keep=MAX_UPLOAD_FILES)
    except Exception:
        logger.exception("アップロードの整理に失敗しました（起動は続行します）")
        return
    if removed:
        logger.info("古いアップロードを %d 件片付けました", len(removed))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _prune_uploads()
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
    """学習済みモデルを書き出してダウンロードさせる。"""
    from app.rl.export import EXPORT_KINDS

    if kind not in EXPORT_KINDS:
        return JSONResponse(
            {"error": f"未知の書き出し形式です: {kind}", "supported": list(EXPORT_KINDS)},
            status_code=400,
        )

    if kind == "keras":
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
            "X-Export-Kind": result.kind,
            "X-Export-Size": str(result.size_bytes),
        },
    )


@app.post("/api/import")
async def import_model_endpoint(file: UploadFile = File(...)):
    """書き出したモデルを受け取り、その状態から学習を再開する。"""
    from app.rl.importer import MAX_UPLOAD_BYTES

    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_name = _safe_upload_name(file.filename)
    dest = config.UPLOAD_DIR / f"{stamp}_{safe_name}"

    written = 0
    handed_off = False
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
        handed_off = True
        finished = await asyncio.to_thread(ticket.done.wait, IMPORT_TIMEOUT_SEC)

        if not finished:
            return JSONResponse(
                {"ok": False, "error": "読み込みが時間内に完了しませんでした"}, status_code=504
            )
        if ticket.error is not None or ticket.info is None:
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

        await manager.broadcast({"type": "status", **engine.status_payload()})
        return JSONResponse(payload)

    finally:
        # 依頼を積めたなら削除はエンジン側（_handle_import）の責任（code_review E-05）。
        # 積む前に抜けた場合だけここで消す
        if not handed_off:
            try:
                dest.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "アップロードファイルを削除できませんでした: %s（%s）", dest.name, exc
                )


def _weather_presets_wire() -> list[dict[str, Any]]:
    """天候プリセットの (rain, fog)。**数値の出典は percep/weather.py の PRESETS だけ。**"""
    from app.percep.weather import PRESETS

    return [
        {"id": name, "rain": round(w.rain, 3), "fog": round(w.fog, 3)}
        for name, w in PRESETS.items()
    ]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    await manager.add(websocket)
    logger.info("WebSocket 接続を受け付けました（接続数 %d）", manager.count)

    try:
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
                    "maxPedestrians": config.MAX_PEDESTRIANS,
                    "simHz": config.SIM_HZ,
                    "obsDim": config.OBS_DIM,
                    "actionDim": config.ACTION_DIM,
                },
                "weatherPresets": _weather_presets_wire(),
                "params": engine.snapshot_params().to_wire(),
                "status": engine.status_payload(),
            },
        )

        await send_json(
            websocket, {"type": "detector", **engine.detector_job.snapshot()}
        )

        # 配車は 1 件しか無い（決定 8）ので、経路つきの 1 通を全接続へ配り直す
        engine.request_full_taxi()

        if _current_map_wire is not None:
            await send_json(websocket, {"type": "map", **_current_map_wire})
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


_dist_dir = config.PROJECT_DIR / "frontend" / "dist"

import os as _os  # noqa: E402

from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.responses import Response as _Response  # noqa: E402
from starlette.types import Scope as _Scope  # noqa: E402

_NO_CACHE_FILES = {"sw.js", "manifest.webmanifest"}


class _FrontendStatic(StaticFiles):
    """index.html と Service Worker だけキャッシュさせない静的配信。"""

    def file_response(
        self,
        full_path: "_os.PathLike[str] | str",
        stat_result: _os.stat_result,
        scope: _Scope,
        status_code: int = 200,
    ) -> _Response:
        response = super().file_response(full_path, stat_result, scope, status_code)
        name = _os.fspath(full_path).replace("\\", "/")
        if name.endswith(".html") or name.rsplit("/", 1)[-1] in _NO_CACHE_FILES:
            response.headers["Cache-Control"] = "no-cache"
        elif "/assets/" in name:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


if _dist_dir.is_dir():
    app.mount("/", _FrontendStatic(directory=str(_dist_dir), html=True), name="frontend")
    logger.info("ビルド済みフロントエンドを配信します: %s", _dist_dir)
