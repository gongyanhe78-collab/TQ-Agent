"""仓库运行产物和模块依赖边界测试。"""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_runtime_outputs_are_ignored() -> None:
    """运行数据、日志、截图和临时目录必须已写入忽略规则。"""
    ignore_text = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    expected_rules = (
        "runtime/",
        "logs/",
        ".codex_tmp/",
        "backend/tests/outputs/",
        "tmp/",
        "data/numba_cache/",
    )
    missing = [rule for rule in expected_rules if rule not in ignore_text]
    assert missing == []


def test_router_modules_do_not_import_main() -> None:
    """Router 只能通过依赖注入访问应用服务。"""
    route_dir = PROJECT_ROOT / "backend/app/api/routes"
    offenders: list[str] = []

    for path in route_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported = _imported_main_module(node)
            if imported:
                offenders.append(f"{path.name}: {imported}")

    assert offenders == []


def _imported_main_module(node: ast.AST) -> str | None:
    """返回节点中导入主应用模块的名称。"""
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if module == "backend.app" and any(alias.name == "main" for alias in node.names):
            return "backend.app.main"
        if module.startswith("backend.app.main"):
            return module
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.startswith("backend.app.main"):
                return alias.name
    return None
