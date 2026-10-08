"""モデルの multipart アップロードを容量制限つきで受け取る。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from python_multipart.exceptions import MultipartParseError
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import Request

MAX_UPLOAD_BYTES = 256 * 1024 * 1024
MAX_MULTIPART_OVERHEAD_BYTES = 64 * 1024
MAX_IMPORT_BODY_BYTES = MAX_UPLOAD_BYTES + MAX_MULTIPART_OVERHEAD_BYTES
UPLOAD_ERROR_MESSAGES = {
    400: "アップロードが不正です。file 欄にモデルファイルを 1 つだけ送ってください",
    413: f"ファイルまたは本文が大きすぎます。ファイルの上限は {MAX_UPLOAD_BYTES // 1024 // 1024} MB です",
    415: "Content-Type は multipart/form-data にしてください",
}


class UploadRejected(MultiPartException):
    """アップロードを拒否した理由と HTTP ステータスを保持する。"""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


async def _bounded_stream(request: Request) -> AsyncIterator[bytes]:
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > MAX_IMPORT_BODY_BYTES:
            raise UploadRejected("アップロードの本文が大きすぎます", 413)
        yield chunk


class _ModelMultipartParser(MultiPartParser):
    """ファイル単体の上限と file 欄を解析中に検査する。"""

    def __init__(self, request: Request) -> None:
        super().__init__(request.headers, _bounded_stream(request), max_files=1, max_fields=0)
        self.uploads: list[UploadFile] = []
        self._file_bytes = 0
        self.complete = False

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self._file_bytes = 0

    def on_headers_finished(self) -> None:
        super().on_headers_finished()
        upload = self._current_part.file
        if upload is not None:
            self.uploads.append(upload)
        if upload is None or self._current_part.field_name != "file":
            raise UploadRejected("file 欄にモデルファイルを 1 つだけ送ってください")

    def on_end(self) -> None:
        self.complete = True

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._current_part.file is not None:
            self._file_bytes += end - start
            if self._file_bytes > MAX_UPLOAD_BYTES:
                raise UploadRejected(
                    f"ファイルが大きすぎます。上限は {MAX_UPLOAD_BYTES // 1024 // 1024} MB です", 413
                )
        super().on_part_data(data, start, end)


@asynccontextmanager
async def read_model_upload(request: Request) -> AsyncIterator[UploadFile]:
    """本文を制限して解析し、正常時も中断時も一時ファイルを閉じる。"""
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "multipart/form-data":
        raise UploadRejected("Content-Type は multipart/form-data にしてください", 415)
    raw_length = request.headers.get("content-length")
    if raw_length is not None:
        try:
            if not raw_length.isascii() or not raw_length.isdecimal():
                raise ValueError
            length = int(raw_length)
        except ValueError:
            raise UploadRejected("Content-Length が不正です") from None
        if length > MAX_IMPORT_BODY_BYTES:
            raise UploadRejected("アップロードの本文が大きすぎます", 413)

    parser = _ModelMultipartParser(request)
    try:
        try:
            form = await parser.parse()
        except UploadRejected:
            raise
        except (MultiPartException, MultipartParseError):
            raise UploadRejected("file 欄にモデルファイルを 1 つだけ送ってください（multipart が不正です）") from None
        items = form.multi_items()
        if not parser.complete or len(items) != 1 or items[0][0] != "file" or not isinstance(items[0][1], UploadFile):
            raise UploadRejected("file 欄にモデルファイルを 1 つだけ送ってください")
        yield items[0][1]
    finally:
        for upload in parser.uploads:
            await upload.close()
