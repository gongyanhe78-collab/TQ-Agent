"""查询幂等注册表、去重事件与路由级幂等测试。"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.routes import agent
from backend.app.observability.middleware import ObservabilityMiddleware
from backend.app.services.query.idempotency import (
    QueryIdempotencyRegistry,
    duplicate_event,
)


def _make_registry() -> QueryIdempotencyRegistry:
    """构造不连接 Redis 的内存幂等注册表。"""
    return QueryIdempotencyRegistry(
        url="",
        prefix="risk",
        ttl_seconds=1800,
        enabled=False,
    )


class QueryIdempotencyRegistryTests(unittest.TestCase):
    """验证幂等键占位、回读与用户隔离。"""

    def setUp(self) -> None:
        self.registry = _make_registry()

    def test_claim_lookup_update_roundtrip(self) -> None:
        self.assertTrue(
            self.registry.claim("key-1", "user-1", run_id="")
        )
        entry = self.registry.lookup("key-1", "user-1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["status"], "running")
        self.registry.update(
            "key-1",
            "user-1",
            run_id="run-1",
            status="completed",
        )
        entry = self.registry.lookup("key-1", "user-1")
        self.assertEqual(entry["run_id"], "run-1")
        self.assertEqual(entry["status"], "completed")

    def test_duplicate_claim_is_rejected(self) -> None:
        self.assertTrue(self.registry.claim("key-1", "user-1"))
        self.assertFalse(self.registry.claim("key-1", "user-1"))

    def test_keys_are_isolated_by_user(self) -> None:
        self.assertTrue(self.registry.claim("key-1", "user-1"))
        self.assertIsNone(self.registry.lookup("key-1", "user-2"))

    def test_unregistered_key_returns_none(self) -> None:
        self.assertIsNone(self.registry.lookup("key-missing", "user-1"))


class DuplicateEventTests(unittest.TestCase):
    """验证去重事件只在键已登记时返回。"""

    def setUp(self) -> None:
        self.registry = _make_registry()

    def test_no_key_returns_none(self) -> None:
        self.assertIsNone(duplicate_event(self.registry, "", "user-1"))

    def test_unregistered_key_returns_none(self) -> None:
        self.assertIsNone(
            duplicate_event(self.registry, "key-1", "user-1")
        )

    def test_running_duplicate_event(self) -> None:
        self.registry.claim("key-1", "user-1")
        event = duplicate_event(self.registry, "key-1", "user-1")
        self.assertIsNotNone(event)
        self.assertEqual(event["type"], "duplicate")
        self.assertEqual(event["status"], "running")

    def test_completed_duplicate_event_keeps_run_id(self) -> None:
        self.registry.claim("key-1", "user-1")
        self.registry.update(
            "key-1",
            "user-1",
            run_id="run-9",
            status="completed",
        )
        event = duplicate_event(self.registry, "key-1", "user-1")
        self.assertEqual(event["run_id"], "run-9")
        self.assertEqual(event["status"], "completed")


class _AuthServices:
    """为查询路由提供最小认证服务组合。"""

    def __init__(self, chat_service) -> None:
        self.chat_service = chat_service

    @property
    def auth_service(self):
        return SimpleNamespace(
            get_current_user=lambda token: {
                "user_id": "idem-user",
                "role": "user",
                "token": token,
            }
        )

    def chat_service_for(self, _user_id):
        return self.chat_service


class _CountingChatService:
    """记录被调用次数并回放固定事件，用于验证幂等去重。"""

    def __init__(self) -> None:
        self.calls = 0

    async def stream(
        self,
        _payload,
        _is_disconnected,
        execution_context=None,
    ):
        self.calls += 1
        yield {
            "type": "metadata",
            "status": "running",
            "run_id": "run-idem-1",
        }
        yield {
            "type": "done",
            "status": "completed",
            "run_id": "run-idem-1",
        }


def _make_client(chat_service) -> TestClient:
    """挂载真实查询路由，供幂等去重测试使用。"""
    app = FastAPI()
    app.state.services = _AuthServices(chat_service)
    app.include_router(agent.router)
    app.add_middleware(ObservabilityMiddleware)
    return TestClient(
        app,
        headers={"Authorization": "Bearer idem-token"},
    )


class RouteIdempotencyTests(unittest.TestCase):
    """验证同一幂等键的第二次请求不触发编排器。"""

    def test_second_request_returns_duplicate_and_skips_chat(self) -> None:
        chat_service = _CountingChatService()
        registry = _make_registry()
        client = _make_client(chat_service)
        payload = {
            "question": "今天重庆天气怎么样",
            "idempotency_key": "dup-1",
        }
        try:
            with patch(
                "backend.app.api.routes.agent.get_query_idempotency",
                return_value=registry,
            ):
                first = client.post("/api/query/stream", json=payload)
                second = client.post("/api/query/stream", json=payload)
        finally:
            client.close()

        self.assertEqual(first.status_code, 200)
        self.assertIn("run-idem-1", first.text)
        self.assertIn("completed", first.text)
        # 第二次请求被去重，不再调用聊天服务。
        self.assertEqual(chat_service.calls, 1)
        self.assertEqual(second.status_code, 200)
        self.assertIn("duplicate", second.text)
        self.assertIn("未重复执行", second.text)

    def test_no_key_requests_are_not_deduplicated(self) -> None:
        chat_service = _CountingChatService()
        registry = _make_registry()
        client = _make_client(chat_service)
        payload = {"question": "山西今天天气怎么样"}
        try:
            with patch(
                "backend.app.api.routes.agent.get_query_idempotency",
                return_value=registry,
            ):
                first = client.post("/api/query/stream", json=payload)
                second = client.post("/api/query/stream", json=payload)
        finally:
            client.close()

        self.assertEqual(chat_service.calls, 2)
        self.assertIn("completed", first.text)
        self.assertIn("completed", second.text)
