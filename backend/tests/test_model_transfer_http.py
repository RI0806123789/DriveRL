"""モデル入出力の HTTP 境界と multipart の容量・後片付けを検査する。"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePath
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette import formparsers
from starlette.requests import ClientDisconnect
from starlette.datastructures import UploadFile as StarletteUploadFile

from app import model_upload
from app.contracts import MODEL_EXPORT_KINDS
from app.host_guard import HostOriginGuard
from app.model_upload import UploadRejected, read_model_upload
from app.rl import export

MAIN_PY = Path(__file__).resolve().parents[1] / "app" / "main.py"
ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
CONTENT_TYPE = "multipart/form-data; boundary=transfer-test"


def multipart(data: bytes, *, name: str = "file", filename: str = "model.pt", end: bool = True) -> bytes:
    prefix = (
        '--transfer-test\r\n'
        f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        'Content-Type: application/octet-stream\r\n\r\n'
    ).encode()
    return prefix + data + (b"\r\n--transfer-test--\r\n" if end else b"")


@pytest.fixture
def transfer_app(tmp_path, monkeypatch):
    calls = {"exports": [], "imports": [], "preloads": []}
    exported = tmp_path / "export.pt"
    exported.write_bytes(b"model-result")
    done = threading.Event()
    done.set()

    def request_export(kind):
        calls["exports"].append(kind)
        return SimpleNamespace(done=done, error=None, result=SimpleNamespace(
            path=exported, filename="export.pt", media_type="application/octet-stream",
            kind=kind, size_bytes=exported.stat().st_size,
        ))

    def request_import(path):
        calls["imports"].append((path, path.read_bytes()))
        info = SimpleNamespace(updates=12, to_wire=lambda: {"updates": 12})
        return SimpleNamespace(done=done, error=None, info=info, backup=None)

    async def broadcast(payload):
        pass

    monkeypatch.setattr(export, "preload_torch_export", lambda: calls["preloads"].append("pt2"))
    monkeypatch.setattr(export, "preload_keras", lambda: calls["preloads"].append("keras"))
    app = FastAPI()
    namespace = {
        "app": app, "asyncio": asyncio, "Request": Request, "UploadFile": UploadFile,
        "JSONResponse": JSONResponse, "FileResponse": FileResponse, "time": time,
        "PurePath": PurePath, "re": re, "Any": object, "uuid4": uuid4,
        "logger": logging.getLogger(__name__), "config": SimpleNamespace(UPLOAD_DIR=tmp_path),
        "engine": SimpleNamespace(request_export=request_export, request_import=request_import,
                                  status_payload=lambda: {"state": "running"}),
        "manager": SimpleNamespace(broadcast=broadcast),
        "MAX_UPLOAD_BYTES": model_upload.MAX_UPLOAD_BYTES, "UploadRejected": UploadRejected,
        "UPLOAD_ERROR_MESSAGES": model_upload.UPLOAD_ERROR_MESSAGES,
        "read_model_upload": read_model_upload, "EXPORT_TIMEOUT_SEC": 1.0, "IMPORT_TIMEOUT_SEC": 1.0,
    }
    names = {"_safe_upload_name", "export_model_endpoint", "import_model_endpoint", "_import_model_file"}
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
    body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    assert len(body) == len(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(MAIN_PY), "exec"), namespace)
    app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(HostOriginGuard, allowed_origins=ORIGINS)
    return app, calls, namespace


def request(app, method, path, chunks=(), headers=None):
    messages = list(chunks) or [b""]
    consumed = 0
    sent = []
    raw_headers = {"host": "127.0.0.1:8000", **(headers or {})}
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": b"", "headers": [(k.encode(), v.encode()) for k, v in raw_headers.items()],
        "server": ("127.0.0.1", 8000), "client": ("127.0.0.1", 12345), "root_path": "",
    }

    async def receive():
        nonlocal consumed
        assert consumed < len(messages), "本文の最後より先まで受信しました"
        body = messages[consumed]
        consumed += 1
        if isinstance(body, BaseException):
            raise body
        if body is None:
            return {"type": "http.disconnect"}
        return {"type": "http.request", "body": body, "more_body": consumed < len(messages)}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, body, consumed


@pytest.fixture
def small_limits(monkeypatch):
    monkeypatch.setattr(model_upload, "MAX_UPLOAD_BYTES", 32)
    monkeypatch.setattr(model_upload, "MAX_IMPORT_BODY_BYTES", 512)


@pytest.fixture
def temporary_files(monkeypatch):
    files = []
    original = formparsers.SpooledTemporaryFile

    def create(*args, **kwargs):
        kwargs["max_size"] = 8
        file = original(*args, **kwargs)
        files.append(file)
        return file

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", create)
    return files


def test_declared_oversize_never_receives_or_creates_files(transfer_app, small_limits, temporary_files):
    app, calls, _ = transfer_app
    status, body, consumed = request(app, "POST", "/api/import", [b"unused"],
        {"content-type": CONTENT_TYPE, "content-length": "513"})
    assert status == 413 and consumed == 0 and temporary_files == []
    assert json.loads(body)["ok"] is False and calls["imports"] == []


@pytest.mark.parametrize("length", [None, "1"])
def test_stream_limit_precedes_multipart_write(transfer_app, small_limits, temporary_files, length):
    app, calls, _ = transfer_app
    prefix = multipart(b"", end=False)
    headers = {"content-type": CONTENT_TYPE}
    if length is not None:
        headers["content-length"] = length
    chunks = [prefix + b"x" * 16, b"x" * 512, b"not-received"]
    status, _, consumed = request(app, "POST", "/api/import", chunks, headers)
    assert status == 413 and consumed == 2 and calls["imports"] == []
    assert len(temporary_files) == 1 and temporary_files[0].closed and temporary_files[0]._rolled


def test_file_limit_precedes_writing_offending_chunk(transfer_app, small_limits, temporary_files, monkeypatch):
    app, calls, _ = transfer_app
    writes = []
    original = StarletteUploadFile.write

    async def write(file, data):
        writes.append(bytes(data))
        return await original(file, data)

    monkeypatch.setattr(StarletteUploadFile, "write", write)
    status, _, consumed = request(app, "POST", "/api/import",
        [multipart(b"x" * 16, end=False), b"x" * 17, b"not-received"], {"content-type": CONTENT_TYPE})
    assert status == 413 and consumed == 2 and calls["imports"] == []
    assert b"".join(writes) == b"x" * 16
    assert len(temporary_files) == 1 and temporary_files[0].closed


@pytest.mark.parametrize("length", [False, True])
def test_exact_file_limit_imports_and_closes_temp(transfer_app, small_limits, temporary_files, length):
    app, calls, _ = transfer_app
    data = bytes(range(32))
    raw = multipart(data)
    headers = {"content-type": CONTENT_TYPE}
    if length:
        headers["content-length"] = str(len(raw))
    status, body, _ = request(app, "POST", "/api/import", [raw], headers)
    assert status == 200 and json.loads(body)["sizeBytes"] == 32
    assert calls["imports"][0][1] == data
    assert all(file.closed for file in temporary_files)


@pytest.mark.parametrize("name", ["file", "other"])
def test_second_file_is_rejected_before_its_body(transfer_app, temporary_files, name):
    app, calls, _ = transfer_app
    first = multipart(b"first").replace(b"--transfer-test--\r\n", b"")
    second = multipart(b"", name=name, end=False)
    status, _, consumed = request(app, "POST", "/api/import", [first + second, b"not-received"],
        {"content-type": CONTENT_TYPE})
    assert status == 400 and consumed == 1 and calls["imports"] == []
    assert len(temporary_files) == 1 and temporary_files[0].closed


@pytest.mark.parametrize("raw", [multipart(b"", name="other"), multipart(b"data", end=False),
    b'--transfer-test\r\nContent-Disposition: form-data; name="field"\r\n\r\nvalue\r\n--transfer-test--\r\n',
    b"invalid multipart"])
def test_invalid_form_cleans_temp(transfer_app, temporary_files, raw):
    app, calls, _ = transfer_app
    status, _, _ = request(app, "POST", "/api/import", [raw], {"content-type": CONTENT_TYPE})
    assert status == 400 and calls["imports"] == []
    assert all(file.closed for file in temporary_files)


@pytest.mark.parametrize("interruption", [None, asyncio.CancelledError()])
def test_interrupted_body_closes_partial_file(transfer_app, temporary_files, interruption):
    app, calls, _ = transfer_app
    with pytest.raises(ClientDisconnect if interruption is None else asyncio.CancelledError):
        request(app, "POST", "/api/import", [multipart(b"x" * 16, end=False), interruption],
            {"content-type": CONTENT_TYPE})
    assert len(temporary_files) == 1 and temporary_files[0].closed and temporary_files[0]._rolled
    assert calls["imports"] == []


def test_save_failure_removes_destination(transfer_app, temporary_files, monkeypatch):
    app, calls, namespace = transfer_app
    original = StarletteUploadFile.read

    async def broken_read(file, size=-1):
        if size == 1024 * 1024:
            raise OSError("保存の検査")
        return await original(file, size)

    monkeypatch.setattr(StarletteUploadFile, "read", broken_read)
    status, _, _ = request(app, "POST", "/api/import", [multipart(b"content")], {"content-type": CONTENT_TYPE})
    assert status == 500 and calls["imports"] == []
    assert list(namespace["config"].UPLOAD_DIR.glob("*_model.pt")) == []
    assert all(file.closed for file in temporary_files)


@pytest.mark.parametrize("headers", [{}, {"origin": "https://evil.example"}, {"sec-fetch-site": "cross-site"}])
@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_export_get_cannot_generate(transfer_app, headers, method):
    app, calls, _ = transfer_app
    status, _, consumed = request(app, method, "/api/export/pt2", headers=headers)
    assert status == 405 and consumed == 0
    assert calls["exports"] == [] and calls["preloads"] == []


@pytest.mark.parametrize("origin", [None, "http://127.0.0.1:8000", *ORIGINS])
@pytest.mark.parametrize("kind", MODEL_EXPORT_KINDS)
def test_export_post_downloads_from_ui_dev_and_cli(transfer_app, origin, kind):
    app, calls, _ = transfer_app
    headers = {"content-type": "application/json"}
    if origin:
        headers["origin"] = origin
    status, body, _ = request(app, "POST", f"/api/export/{kind}", [b"{}"], headers)
    assert status == 200 and body == b"model-result" and calls["exports"] == [kind]


@pytest.mark.parametrize("content_type", [None, "text/plain", "application/x-www-form-urlencoded", "multipart/form-data"])
def test_export_without_json_cannot_generate_even_without_origin(transfer_app, content_type):
    app, calls, _ = transfer_app
    headers = {"content-type": content_type} if content_type else {}
    status, _, _ = request(app, "POST", "/api/export/pt2", headers=headers)
    assert status == 415 and calls["exports"] == [] and calls["preloads"] == []


def test_cross_site_export_and_preflight_are_rejected(transfer_app):
    app, calls, _ = transfer_app
    headers = {"origin": "https://evil.example", "content-type": "application/json"}
    assert request(app, "POST", "/api/export/pt2", headers=headers)[0] == 403
    headers["access-control-request-method"] = "POST"
    headers["access-control-request-headers"] = "content-type"
    assert request(app, "OPTIONS", "/api/export/pt2", headers=headers)[0] == 400
    assert calls["exports"] == [] and calls["preloads"] == []


@pytest.mark.parametrize("origin", ORIGINS)
def test_vite_preflight_allows_json_export(transfer_app, origin):
    app, calls, _ = transfer_app
    headers = {"origin": origin, "access-control-request-method": "POST",
               "access-control-request-headers": "content-type"}
    assert request(app, "OPTIONS", "/api/export/pt2", headers=headers)[0] == 200
    assert calls["exports"] == []


@pytest.mark.parametrize("length", ["-1", "invalid", "", "1.5"])
def test_invalid_content_length_is_rejected_before_receive(transfer_app, temporary_files, length):
    app, calls, _ = transfer_app
    status, _, consumed = request(app, "POST", "/api/import", [b"unused"],
        {"content-type": CONTENT_TYPE, "content-length": length})
    assert status == 400 and consumed == 0 and temporary_files == [] and calls["imports"] == []


def test_exact_body_limit_allows_file_and_framing(transfer_app, temporary_files, monkeypatch):
    app, calls, _ = transfer_app
    raw = multipart(b"x" * 32)
    monkeypatch.setattr(model_upload, "MAX_IMPORT_BODY_BYTES", len(raw))
    status, _, _ = request(app, "POST", "/api/import", [raw], {"content-type": CONTENT_TYPE})
    assert status == 200 and calls["imports"][0][1] == b"x" * 32
    assert all(file.closed for file in temporary_files)


def test_originless_cross_site_json_export_cannot_generate(transfer_app):
    app, calls, _ = transfer_app
    headers = {"content-type": "application/json", "sec-fetch-site": "cross-site"}
    status, _, consumed = request(app, "POST", "/api/export/pt2", [b"{}"], headers)
    assert status == 403 and consumed == 0 and calls["exports"] == [] and calls["preloads"] == []


def test_http_timeout_keeps_engine_owned_upload(transfer_app, temporary_files):
    app, calls, namespace = transfer_app
    original = namespace["engine"].request_import

    def pending_import(path):
        ticket = original(path)
        ticket.done = SimpleNamespace(wait=lambda timeout: False)
        return ticket

    namespace["engine"].request_import = pending_import
    status, _, _ = request(app, "POST", "/api/import", [multipart(b"content")], {"content-type": CONTENT_TYPE})
    assert status == 504 and calls["imports"][0][0].read_bytes() == b"content"
    assert all(file.closed for file in temporary_files)


@pytest.mark.parametrize("names", [("model.pt", "model.pt"), ("model?.pt", "model*.pt")])
@pytest.mark.parametrize("timeout", [False, True])
def test_parallel_uploads_keep_separate_engine_owned_files(transfer_app, temporary_files, names, timeout):
    app, calls, namespace = transfer_app
    namespace["time"] = SimpleNamespace(strftime=lambda fmt: "fixed-second")
    assert namespace["_safe_upload_name"](names[0]) == namespace["_safe_upload_name"](names[1])
    barrier = threading.Barrier(2)
    original = namespace["engine"].request_import

    def queued_import(path):
        barrier.wait(timeout=5)
        ticket = original(path)
        if timeout:
            ticket.done = SimpleNamespace(wait=lambda seconds: False)
        return ticket

    namespace["engine"].request_import = queued_import
    data = [b"first-policy", b"second-policy"]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(request, app, "POST", "/api/import", [multipart(content, filename=name)],
                   {"content-type": CONTENT_TYPE}) for content, name in zip(data, names)]
        results = [future.result(timeout=10) for future in futures]
    assert all(status == (504 if timeout else 200) for status, _, _ in results)
    imports = calls["imports"]
    assert len(imports) == 2 and imports[0][0] != imports[1][0]
    assert {content for _, content in imports} == set(data)
    assert {path.read_bytes() for path, _ in imports} == set(data)
    if not timeout:
        assert [json.loads(body)["filename"] for _, body, _ in results] == list(names)
    assert all(file.closed for file in temporary_files)


def test_exclusive_creation_failure_keeps_existing_upload(transfer_app, temporary_files):
    app, calls, namespace = transfer_app
    namespace["time"] = SimpleNamespace(strftime=lambda fmt: "fixed-second")
    namespace["uuid4"] = lambda: SimpleNamespace(hex="fixed-id")
    existing = namespace["config"].UPLOAD_DIR / "fixed-second_fixed-id_model.pt"
    existing.write_bytes(b"queued-policy")
    status, body, _ = request(app, "POST", "/api/import", [multipart(b"replacement")], {"content-type": CONTENT_TYPE})
    assert status == 500 and "保存に失敗" in json.loads(body)["error"]
    assert existing.read_bytes() == b"queued-policy" and calls["imports"] == []
    assert all(file.closed for file in temporary_files)


def test_partial_save_failure_removes_only_its_own_upload(transfer_app, temporary_files, monkeypatch):
    app, calls, namespace = transfer_app
    foreign = namespace["config"].UPLOAD_DIR / "queued_model.pt"
    foreign.write_bytes(b"queued-policy")
    original = StarletteUploadFile.read
    reads = 0

    async def broken_read(file, size=-1):
        nonlocal reads
        if size == 1024 * 1024:
            reads += 1
            if reads == 2:
                raise OSError("部分ファイルの検査")
        return await original(file, size)

    monkeypatch.setattr(StarletteUploadFile, "read", broken_read)
    assert request(app, "POST", "/api/import", [multipart(b"partial")], {"content-type": CONTENT_TYPE})[0] == 500
    assert calls["imports"] == [] and list(namespace["config"].UPLOAD_DIR.glob("*_model.pt")) == [foreign]
    assert foreign.read_bytes() == b"queued-policy" and all(file.closed for file in temporary_files)


def test_queue_failure_cleans_only_unhanded_upload(transfer_app, temporary_files):
    app, calls, namespace = transfer_app
    foreign = namespace["config"].UPLOAD_DIR / "queued_model.pt"
    foreign.write_bytes(b"queued-policy")

    def broken_queue(path):
        raise RuntimeError("キューの検査")

    namespace["engine"].request_import = broken_queue
    with pytest.raises(RuntimeError, match="キューの検査"):
        request(app, "POST", "/api/import", [multipart(b"content")], {"content-type": CONTENT_TYPE})
    assert calls["imports"] == [] and list(namespace["config"].UPLOAD_DIR.glob("*_model.pt")) == [foreign]
    assert foreign.read_bytes() == b"queued-policy" and all(file.closed for file in temporary_files)



def test_model_export_kinds_match_frontend_contract(protocol_ts_source):
    declaration = re.search(r"export type ExportKind = ([^\r\n]+)", protocol_ts_source)
    assert declaration is not None
    assert tuple(re.findall(r"'([^']+)'", declaration.group(1))) == MODEL_EXPORT_KINDS
