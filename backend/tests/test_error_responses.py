"""HTTP の応答に例外の文を載せない（CodeQL の「例外による情報の露出」）。main.py は import せず ast で読む。"""
from __future__ import annotations

import ast

from .conftest import BACKEND_DIR

MAIN_PY = BACKEND_DIR / "app" / "main.py"


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


def test_json_responses_do_not_include_exception_text() -> None:
    leaks = _leaks(ast.parse(MAIN_PY.read_text(encoding="utf-8")))
    assert leaks == [], "例外の文は logger に残し、応答には固定の文を返すこと: " + ", ".join(leaks)


def test_detects_a_leak() -> None:
    src = (
        "try:\n    pass\nexcept Exception as exc:\n"
        "    JSONResponse({'error': f'失敗: {exc}'})\n"
    )
    assert _leaks(ast.parse(src)) != []
