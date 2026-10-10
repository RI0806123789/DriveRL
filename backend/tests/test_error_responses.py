"""HTTP の応答に例外の文を載せない（CodeQL の「例外による情報の露出」）。main.py は import せず ast で読む。"""
from __future__ import annotations

import ast

from .conftest import BACKEND_DIR

MAIN_PY = BACKEND_DIR / "app" / "main.py"

#: 応答の文を作る所（ticket.error を経由して HTTP の応答に載る）
RESPONSE_SOURCES = [
    BACKEND_DIR / "app" / "runtime" / "engine.py",
    BACKEND_DIR / "app" / "rl" / "export.py",
    BACKEND_DIR / "app" / "rl" / "importer.py",
    BACKEND_DIR / "app" / "rl" / "checkpoint.py",
]

#: 文を画面へそのまま出す例外。どれも固定の文と値だけで組み立てる約束
PUBLIC_ERRORS = {"ExportError", "CheckpointImportError", "CheckpointValidationError"}


def _leaks(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for handler in ast.walk(tree):
        if not isinstance(handler, ast.ExceptHandler) or handler.name is None:
            continue
        for node in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "JSONResponse"):
                continue
            if any(isinstance(n, ast.Name) and n.id == handler.name for n in ast.walk(node)):
                found.append(f"main.py:{node.lineno}（except ... as {handler.name}）")
    return found


def _caught_names(handler: ast.ExceptHandler) -> set[str]:
    kinds = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return {k.id if isinstance(k, ast.Name) else getattr(k, "attr", "?") for k in kinds if k is not None}


def _ticket_leaks(tree: ast.AST, label: str) -> list[str]:
    """`ticket.error = ...` と公開用の例外の組み立てに、捕まえた内部の例外を使っている所。"""
    found: list[str] = []
    for handler in ast.walk(tree):
        if not isinstance(handler, ast.ExceptHandler) or handler.name is None:
            continue
        if _caught_names(handler) <= PUBLIC_ERRORS:
            continue  # 公開用の例外の文はもともと画面に出す文
        for node in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Attribute) and t.attr == "error" for t in node.targets
            ):
                value = node.value
            elif isinstance(node, ast.Call) and getattr(node.func, "id", None) in PUBLIC_ERRORS:
                value = ast.Tuple(elts=[*node.args, *(k.value for k in node.keywords)], ctx=ast.Load())
            else:
                continue
            if any(isinstance(n, ast.Name) and n.id == handler.name for n in ast.walk(value)):
                found.append(f"{label}:{node.lineno}（except ... as {handler.name}）")
    return found


def test_json_responses_do_not_include_exception_text() -> None:
    leaks = _leaks(ast.parse(MAIN_PY.read_text(encoding="utf-8")))
    assert leaks == [], "例外の文は logger に残し、応答には固定の文を返すこと: " + ", ".join(leaks)


def test_ticket_errors_and_public_errors_do_not_include_internal_exception_text() -> None:
    leaks: list[str] = []
    for path in RESPONSE_SOURCES:
        leaks += _ticket_leaks(ast.parse(path.read_text(encoding="utf-8")), path.name)
    assert leaks == [], "書き出し・読み込みの応答に内部の例外の文が載る: " + ", ".join(leaks)


def test_detects_a_leak() -> None:
    src = (
        "try:\n    pass\nexcept Exception as exc:\n"
        "    JSONResponse({'error': f'失敗: {exc}'})\n"
    )
    assert _leaks(ast.parse(src)) != []


def test_detects_a_ticket_leak() -> None:
    for body in (
        "    ticket.error = f'失敗: {exc}'\n",
        "    ticket.error = str(exc)\n",
        "    raise ExportError(f'失敗: {exc}') from exc\n",
        "    raise CheckpointImportError(message=str(exc))\n",
    ):
        src = "try:\n    pass\nexcept OSError as exc:\n" + body
        assert _ticket_leaks(ast.parse(src), "x") != [], body
    passthrough = "try:\n    pass\nexcept ExportError as exc:\n    ticket.error = str(exc)\n"
    assert _ticket_leaks(ast.parse(passthrough), "x") == []
