"""バックエンドの起動スクリプト。フロントエンドもここから一緒に起動できる。"""

from __future__ import annotations

import argparse
import re
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn  # noqa: E402

from app import config  # noqa: E402

FRONTEND_DIR = config.PROJECT_DIR / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"
VITE_PORT = 5173


def _force_utf8_console() -> None:
    """Windows のコンソールを UTF-8 にして、日本語ログの文字化けを防ぐ。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass


def _find_npm() -> str | None:
    """npm の実行ファイル。Windows では `npm.cmd` になる。"""
    return shutil.which("npm")


def _frontend_ready(npm: str | None) -> str | None:
    """フロントを扱える状態か。駄目な理由を返す（問題なければ None）。"""
    if npm is None:
        return "npm が PATH に見つかりません（Node.js 22.18 以降を入れてください）"
    if not FRONTEND_DIR.is_dir():
        return f"フロントエンドのディレクトリがありません: {FRONTEND_DIR}"
    if not (FRONTEND_DIR / "node_modules").is_dir():
        return "依存が入っていません。`cd frontend; npm install` を先に実行してください"
    return None


def _pump_output(proc: subprocess.Popen[str]) -> None:
    """Vite の出力に `[vite]` を付けて流す。"""
    stream = proc.stdout
    if stream is None:
        return
    try:
        for line in stream:
            text = line.rstrip()
            if text:
                print(f"[vite] {text}", flush=True)
    except (ValueError, OSError):
        pass


def _spawn_vite(npm: str) -> subprocess.Popen[str] | None:
    """Vite の開発サーバーを子プロセスとして起動する。"""
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        return subprocess.Popen(
            [npm, "run", "dev"],
            cwd=str(FRONTEND_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
    except OSError as exc:
        print(f"[vite] 起動できませんでした: {exc}", flush=True)
        return None


def _stop_vite(proc: subprocess.Popen[str] | None) -> None:
    """Vite を確実に止める。"""
    if proc is None or proc.poll() is not None:
        return

    print("[vite] 停止します", flush=True)
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        proc.terminate()

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _build_frontend(npm: str) -> bool:
    """`npm run build` を実行する。型チェックも含まれるので失敗しうる。"""
    print("フロントエンドをビルドします（型チェック込み。1 分ほどかかります）", flush=True)
    result = subprocess.run([npm, "run", "build"], cwd=str(FRONTEND_DIR), check=False)
    if result.returncode != 0:
        print("ビルドに失敗しました。上の出力を確認してください。", flush=True)
        return False
    print(f"ビルドできました: {DIST_DIR}", flush=True)
    return True


def _port_taken(port: int) -> bool:
    """待ち受けているプロセスがあるか。"""
    targets = (
        (socket.AF_INET, ("127.0.0.1", port)),
        (socket.AF_INET6, ("::1", port)),
    )
    for family, address in targets:
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.4)
                if sock.connect_ex(address) == 0:
                    return True
        except OSError:
            continue
    return False


def _listening_pid(port: int) -> str:
    """そのポートを掴んでいるプロセスの PID。分からなければ空文字。"""
    if sys.platform != "win32":
        return ""
    try:
        out = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    pattern = re.compile(rf"^\s*TCP\s+\S*:{port}\s+\S+\s+LISTENING\s+(\d+)", re.M)
    m = pattern.search(out)
    return m.group(1) if m else ""


def _check_ports(dev: bool) -> bool:
    """空いていれば True。埋まっていたら理由を出して False。"""
    busy: list[tuple[str, int]] = []
    if _port_taken(config.PORT):
        busy.append(("バックエンド", config.PORT))
    if dev and _port_taken(VITE_PORT):
        busy.append(("Vite", VITE_PORT))
    if not busy:
        return True

    print("=" * 62, flush=True)
    print("  起動できません: ポートが既に使われています", flush=True)
    print("", flush=True)
    for name, port in busy:
        pid = _listening_pid(port)
        suffix = f"（PID {pid}）" if pid else ""
        print(f"    {port:<6}{name}{suffix}", flush=True)
    print("", flush=True)
    print("  すでに DriveRL を起動している可能性があります。", flush=True)
    if dev:
        print("  その場合はブラウザで http://localhost:5173 を開いてください。", flush=True)
    else:
        print(f"  その場合はブラウザで http://{config.HOST}:{config.PORT} を開いてください。", flush=True)
    print("", flush=True)
    print("  動いている窓が見当たらないなら、次で止めてから実行し直してください。", flush=True)
    for name, port in busy:
        pid = _listening_pid(port)
        if pid:
            print(f"    taskkill /F /T /PID {pid}    # {port} を掴んでいるもの", flush=True)
    print("=" * 62, flush=True)
    return False


def _print_banner(dev: bool) -> None:
    """どの URL を開けばよいかを最初にはっきり出す。"""
    backend_url = f"http://{config.HOST}:{config.PORT}"
    print("=" * 62, flush=True)
    if dev:
        print(f"  ブラウザで開く : http://localhost:{VITE_PORT}", flush=True)
        print(f"  バックエンド   : {backend_url}（Vite が /ws と /api を中継）", flush=True)
        print("  フロントの変更は保存すると即反映される（学習は途切れない）", flush=True)
    else:
        print(f"  ブラウザで開く : {backend_url}", flush=True)
        if not DIST_DIR.is_dir():
            print("  ※ frontend/dist がありません。この状態では画面が出ません。", flush=True)
            print("     --build を付けるか、--dev で開発サーバーを使ってください。", flush=True)
    print("  終了する       : Ctrl+C", flush=True)
    print("=" * 62, flush=True)


def main() -> None:
    _force_utf8_console()

    parser = argparse.ArgumentParser(
        description="DriveRL のバックエンド（と、必要ならフロントエンド）を起動する。",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Vite の開発サーバーも一緒に起動する。ブラウザでは 5173 を開く",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="起動前に `npm run build` を実行する。以降はバックエンド単体で配信できる",
    )
    args = parser.parse_args()

    if args.dev and args.build:
        parser.error("--dev と --build は同時に指定できません（--dev はビルドを使いません）")

    if not _check_ports(dev=args.dev):
        sys.exit(1)

    npm = _find_npm()
    vite: subprocess.Popen[str] | None = None

    if args.build:
        problem = _frontend_ready(npm)
        if problem is not None:
            print(f"ビルドできません: {problem}", flush=True)
            sys.exit(1)
        assert npm is not None
        if not _build_frontend(npm):
            sys.exit(1)

    if args.dev:
        problem = _frontend_ready(npm)
        if problem is not None:
            print(f"[vite] 起動を飛ばします: {problem}", flush=True)
        else:
            assert npm is not None
            vite = _spawn_vite(npm)
            if vite is not None:
                threading.Thread(target=_pump_output, args=(vite,), daemon=True).start()

    _print_banner(dev=args.dev and vite is not None)

    try:
        uvicorn.run(
            "app.main:app",
            host=config.HOST,
            port=config.PORT,
            log_level="info",
            ws_ping_interval=20.0,
            ws_ping_timeout=20.0,
        )
    finally:
        _stop_vite(vite)


if __name__ == "__main__":
    main()
