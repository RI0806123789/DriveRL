"""環境変数の読み方（`app/config.py`）の契約テスト（#94）。"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.config import _env_port

BACKEND_DIR = Path(__file__).resolve().parents[1]
NAME = "DRIVERL_TEST_PORT"


@pytest.mark.parametrize(("raw", "port"), [("8001", 8001), (" 8080 ", 8080), ("1", 1), ("65535", 65535)])
def test_valid_port_is_used(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str, port: int) -> None:
    monkeypatch.setenv(NAME, raw)
    with caplog.at_level(logging.WARNING, logger="app.config"):
        assert _env_port(NAME, 8000) == port
    assert caplog.records == []


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_unset_or_empty_falls_back_quietly(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str | None) -> None:
    if raw is None:
        monkeypatch.delenv(NAME, raising=False)
    else:
        monkeypatch.setenv(NAME, raw)
    with caplog.at_level(logging.WARNING, logger="app.config"):
        assert _env_port(NAME, 8000) == 8000
    assert caplog.records == []


@pytest.mark.parametrize("raw", ["abc", "80.5", "0x1f90", "0", "-1", "65536", "8000abc"])
def test_invalid_port_falls_back_with_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str) -> None:
    monkeypatch.setenv(NAME, raw)
    with caplog.at_level(logging.WARNING, logger="app.config"):
        assert _env_port(NAME, 8000) == 8000
    assert len(caplog.records) == 1
    assert NAME in caplog.records[0].getMessage() and "8000" in caplog.records[0].getMessage()


def test_server_config_survives_a_broken_port() -> None:
    env = {**os.environ, "DRIVERL_PORT": "abc", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.run(
        [sys.executable, "-c", "from app import config; print(config.PORT)"],
        cwd=BACKEND_DIR,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "8000"
    assert "DRIVERL_PORT='abc'" in proc.stderr
