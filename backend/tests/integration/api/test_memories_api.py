"""当前用户记忆读取与删除 HTTP 接口测试。"""
from __future__ import annotations

import importlib

import unittest
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.api.routes import memories
from backend.app.db.base import Base
from backend.app.db.models import User


def _load_module(module_name: str):
    """导入待实现模块，并把缺失实现明确报告为测试失败。"""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise AssertionError(f"待实现模块缺失：{module_name}：{exc}") from exc


class _FakeMemoryService:
    """提供接口所需的最小记忆服务替身。"""

    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []
        self.contexts: dict[str, dict] = {}

    def context_for_user(self, user_id):
        return self.contexts.get(
            user_id,
            {
                "user_id": user_id,
                "memories": [],
                "fallback_structured": False,
            },
        )

    def delete(self, user_id, memory_id):
        self.deleted.append((user_id, memory_id))
        return {"status": "deleted", "memory_id": memory_id, "user_id": user_id}


def _auth_services(memory_service, *, user_id="api-user") -> SimpleNamespace:
    """构造只允许当前用户的记忆服务组合。"""
    return SimpleNamespace(
        auth_service=SimpleNamespace(
            get_current_user=lambda token: {
                "user_id": user_id,
                "role": "user",
                "token": token,
            }
        ),
        memory_service_for=lambda uid: memory_service,
    )


class MemoriesApiTests(unittest.TestCase):
    """GET 与 DELETE /api/memories 只能访问当前用户数据。"""

    def setUp(self) -> None:
        self.memory_service = _FakeMemoryService()
        app = FastAPI()
        app.state.services = _auth_services(self.memory_service)
        app.include_router(memories.router)
        self.client = TestClient(
            app,
            headers={"Authorization": "Bearer api-token"},
        )

    def test_get_memories_returns_current_user_fields(self) -> None:
        """GET 返回当前用户可见字段，不暴露内部向量。"""
        self.memory_service.contexts["api-user"] = {
            "user_id": "api-user",
            "memories": [
                {
                    "memory_id": "memory-001",
                    "memory_type": "preference",
                    "content": "偏好关注暴雨",
                    "confidence": 0.9,
                    "sensitivity": "normal",
                    "status": "active",
                    "version": 1,
                    "updated_at": "2026-11-08T00:00:00+00:00",
                }
            ],
            "fallback_structured": False,
        }
        response = self.client.get("/api/memories")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["user_id"], "api-user")
        self.assertEqual(len(payload["memories"]), 1)
        serialized = json_dumps(payload["memories"][0])
        self.assertNotIn("dense_vector", serialized)
        self.assertNotIn("embedding", serialized)

    def test_delete_memory_calls_service_and_returns_deleted(self) -> None:
        """DELETE 应委托记忆服务删除并返回明确状态。"""
        response = self.client.delete("/api/memories/memory-001")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "deleted")
        self.assertEqual(self.memory_service.deleted, [("api-user", "memory-001")])

    def test_delete_missing_memory_returns_404(self) -> None:
        """不存在的记忆删除应返回 404。"""
        self.memory_service.contexts["api-user"] = {
            "user_id": "api-user",
            "memories": [],
            "fallback_structured": False,
        }
        self.memory_service.delete = lambda user_id, memory_id: None
        response = self.client.delete("/api/memories/memory-404")
        self.assertEqual(response.status_code, 404)


def json_dumps(value) -> str:
    """把对象序列化为 JSON 字符串用于字段检查。"""
    import json

    return json.dumps(value, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
