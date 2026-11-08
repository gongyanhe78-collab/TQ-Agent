"""应用服务容器与 Smart 模型服务的生命周期测试。"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from backend.app.agents.smart_case_match.llm.llm_service import SmartCaseLlmService
from backend.app.services.container import AppServices


class _ClosableResource:
    """记录关闭次数，验证应用服务容器执行幂等资源释放。"""

    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class _AvailableClient:
    """模拟可用模型客户端，验证关闭后服务状态不会恢复。"""

    def is_available(self) -> bool:
        return True


class SmartCaseLifecycleTests(unittest.TestCase):
    """验证服务容器负责资源释放，多次关闭保持空操作。"""

    def test_app_services_close_is_idempotent(self) -> None:
        """重复关闭服务容器时，反馈任务和 Smart Agent 只关闭一次。"""
        feedback_tasks = _ClosableResource()
        smart_agent = _ClosableResource()
        services = SimpleNamespace(
            _closed=False,
            _feedback_tasks={"test-user": feedback_tasks},
            smart_agent=smart_agent,
            intent_agent=SimpleNamespace(),
            engine=SimpleNamespace(dispose=lambda: None),
        )

        AppServices.close(services)
        AppServices.close(services)

        self.assertTrue(services._closed)
        self.assertEqual(feedback_tasks.close_count, 1)
        self.assertEqual(smart_agent.close_count, 1)

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
