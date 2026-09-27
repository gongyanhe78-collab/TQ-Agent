"""Smart Case Match 单例、连接池与 Router 生命周期测试。"""
from __future__ import annotations

import asyncio
import threading
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ..llm.llm_service import SmartCaseLlmService
from ..router import _SmartCaseRuntime, _router_lifespan


class _FakeAgent:
    """记录创建和关闭次数，避免生命周期测试加载真实模型与本地资料。"""

    created = 0
    closed = 0

    def __init__(self) -> None:
        type(self).created += 1

    def close(self) -> None:
        type(self).closed += 1


class _AvailableClient:
    """模拟可用模型客户端，验证关闭后服务状态不会恢复。"""

    def is_available(self) -> bool:
        return True


class SmartCaseLifecycleTests(unittest.TestCase):
    """验证迁移后无需 main.py 介入也能安全管理资源。"""

    def setUp(self) -> None:
        _FakeAgent.created = 0
        _FakeAgent.closed = 0

    def test_runtime_does_not_create_agent_while_closing_unused_service(self):
        """服务从未收到请求时，关闭流程不能反向创建一个 Agent。"""
        runtime = _SmartCaseRuntime()
        runtime.close()
        self.assertEqual(_FakeAgent.created, 0)

    def test_concurrent_first_access_creates_one_agent(self):
        """并发首个请求只能创建一个单例，避免重复连接池和重复加载资料。"""
        runtime = _SmartCaseRuntime()
        agents = []
        with patch(
            "backend.app.services.agent.Smart_Case_Match.router.SmartCaseMatchAgent",
            _FakeAgent,
        ):
            threads = [threading.Thread(target=lambda: agents.append(runtime.get_agent())) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            runtime.close()

        self.assertEqual(_FakeAgent.created, 1)
        self.assertEqual(_FakeAgent.closed, 1)
        self.assertTrue(all(agent is agents[0] for agent in agents))

    def test_closed_runtime_rejects_new_agent_until_next_start(self):
        """关闭后不返回已停用实例，新生命周期启动后才允许重新创建。"""
        runtime = _SmartCaseRuntime()
        with patch(
            "backend.app.services.agent.Smart_Case_Match.router.SmartCaseMatchAgent",
            _FakeAgent,
        ):
            runtime.get_agent()
            runtime.close()
            runtime.close()
            with self.assertRaises(RuntimeError):
                runtime.get_agent()
            runtime.start()
            runtime.get_agent()
            runtime.close()

        self.assertEqual(_FakeAgent.created, 2)
        self.assertEqual(_FakeAgent.closed, 2)

    def test_router_lifespan_closes_runtime(self):
        """模拟 Ctrl+C 的 lifespan 退出，确保 Router 自己完成资源释放。"""
        from .. import router as router_module

        runtime = _SmartCaseRuntime()

        async def run_lifespan() -> None:
            with patch.object(router_module, "_runtime", runtime), patch.object(
                router_module,
                "SmartCaseMatchAgent",
                _FakeAgent,
            ):
                async with _router_lifespan(None):
                    runtime.get_agent()

        asyncio.run(run_lifespan())
        self.assertEqual(_FakeAgent.created, 1)
        self.assertEqual(_FakeAgent.closed, 1)
        with self.assertRaises(RuntimeError):
            runtime.get_agent()

    def test_included_router_runs_its_own_lifespan(self):
        """宿主只需 include_router，Ctrl+C 对应的退出阶段就会关闭本目录的单例。"""
        from .. import router as router_module

        runtime = _SmartCaseRuntime()
        app = FastAPI()
        app.include_router(router_module.router)
        with patch.object(router_module, "_runtime", runtime), patch.object(
            router_module,
            "SmartCaseMatchAgent",
            _FakeAgent,
        ):
            with TestClient(app):
                runtime.get_agent()

        self.assertEqual(_FakeAgent.created, 1)
        self.assertEqual(_FakeAgent.closed, 1)
        with self.assertRaises(RuntimeError):
            runtime.get_agent()

    def test_llm_service_close_is_idempotent_and_prevents_reuse(self):
        """LLM 服务可重复关闭，且关闭后不能向已停用线程池提交任务。"""
        service = SmartCaseLlmService(_AvailableClient())
        self.assertTrue(service.available())
        service.close()
        service.close()
        self.assertFalse(service.available())
        with self.assertRaises(RuntimeError):
            service._answer_with_timeout("测试", {}, 16, None)


if __name__ == "__main__":
    unittest.main()
