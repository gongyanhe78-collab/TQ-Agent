"""应用工厂、路由依赖和 Agent 包边界的结构测试。"""
from __future__ import annotations

import ast
import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.main import ROUTERS, create_app


ROUTES_DIR = Path(__file__).resolve().parents[3] / "app" / "api" / "routes"
AGENT_PACKAGES = (
    "backend.app.agents.case_multidim_search",
    "backend.app.agents.smart_case_match",
    "backend.app.agents.query_intent",
    "backend.app.agents.unified_chat",
)
QUERY_PATHS = (
    "/api/query",
    "/api/agent/query",
    "/api/query/stream",
)


class _FakeChatService:
    """记录三个入口的调用，避免测试进入真实检索和模型链路。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def query_document(self, payload) -> dict:
        self.calls.append(("document", payload.model_dump()))
        return {"source": "document", "question": payload.question}

    def query_agent(self, payload) -> dict:
        self.calls.append(("agent", payload.model_dump()))
        return {"source": "agent", "question": payload.question}

    async def stream(self, payload, _is_disconnected):
        self.calls.append(("stream", payload.model_dump()))
        yield {"type": "delta", "content": "统一流响应"}


class _FakeServices:
    """仅提供查询路由所需门面，并记录生命周期关闭。"""

    def __init__(self) -> None:
        self.chat_service = _FakeChatService()
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class _TestSettings:
    """替代真实配置读取，避免结构测试依赖本地环境变量。"""

    def __init__(self, root: Path) -> None:
        self.numba_cache_dir = root / "numba-cache"

    def ensure_directories(self) -> None:
        self.numba_cache_dir.mkdir(parents=True, exist_ok=True)


class ApplicationArchitectureTests(unittest.TestCase):
    """验证应用装配边界、主接口和兼容接口均可用。"""

    def test_openapi_registers_each_query_path_once(self) -> None:
        """三个公开查询入口应通过统一应用工厂注册且不重复。"""
        app = create_app()
        paths = app.openapi()["paths"]

        for path in QUERY_PATHS:
            with self.subTest(path=path):
                self.assertIn(path, paths)
                route_count = sum(
                    getattr(route, "path", None) == path
                    for router in ROUTERS
                    for route in router.routes
                )
                self.assertEqual(route_count, 1)

    def test_route_modules_do_not_depend_on_application_main(self) -> None:
        """接口模块只能依赖服务容器，不能反向导入 main 或编排器状态。"""
        violations: list[str] = []
        for source_path in ROUTES_DIR.glob("*.py"):
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                imported = ""
                if isinstance(node, ast.ImportFrom):
                    imported = node.module or ""
                elif isinstance(node, ast.Import):
                    imported = ",".join(alias.name for alias in node.names)
                if imported.startswith("backend.app.main"):
                    violations.append(f"{source_path.name}: {imported}")
                if imported.startswith("backend.app.agents.unified_chat.orchestrator"):
                    violations.append(f"{source_path.name}: {imported}")

        self.assertEqual(violations, [])

    def test_agent_packages_are_importable(self) -> None:
        """重构后的 Agent 目录必须都是可正常导入的 Python 包。"""
        for package_name in AGENT_PACKAGES:
            with self.subTest(package=package_name):
                module = importlib.import_module(package_name)
                self.assertIsNotNone(module)

    def test_create_app_does_not_build_services_during_import(self) -> None:
        """创建应用对象时只装配路由，数据库和模型建设延迟到生命周期。"""
        with patch(
            "backend.app.services.container.build_services",
            side_effect=AssertionError("创建应用时不应构造服务"),
        ) as build_services:
            app = create_app()
            build_services.assert_not_called()

        self.assertIsNone(getattr(app.state, "services", None))

    def test_query_routes_delegate_to_chat_service_and_close_services(self) -> None:
        """公开接口应统一委托 ChatService，并由生命周期关闭服务容器。"""
        services = _FakeServices()

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = _TestSettings(Path(temp_dir))
            with patch(
                "backend.app.core.lifecycle.get_settings",
                return_value=settings,
            ):
                app = create_app(services_factory=lambda: services)
                with TestClient(app) as client:
                    document = client.post(
                        "/api/query",
                        json={"question": "查询文档证据", "top_k": 3, "top_n": 2},
                    )
                    agent = client.post(
                        "/api/agent/query",
                        json={"question": "查询结构化个例"},
                    )
                    stream = client.post(
                        "/api/query/stream",
                        json={"question": "流式分析个例"},
                    )

        self.assertEqual(document.status_code, 200)
        self.assertEqual(document.json()["source"], "document")
        self.assertEqual(agent.status_code, 200)
        self.assertEqual(agent.json()["source"], "agent")
        self.assertEqual(stream.status_code, 200)
        self.assertIn("统一流响应", stream.text)
        self.assertEqual(
            [call[0] for call in services.chat_service.calls],
            ["document", "agent", "stream"],
        )
        self.assertEqual(services.close_count, 1)


if __name__ == "__main__":
    unittest.main()
