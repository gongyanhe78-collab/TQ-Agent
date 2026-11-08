"""长期记忆服务的删除、上下文召回与候选抽取单元测试。"""
from __future__ import annotations

import importlib

import unittest

from backend.app.agents.unified_chat.orchestration_models import (
    OrchestratorDependencies,
)


def _load_module(module_name: str):
    """导入待实现模块，并把缺失实现明确报告为测试失败。"""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise AssertionError(f"待实现模块缺失：{module_name}：{exc}") from exc


class _FakeVectorStore:
    """记录删除调用并按用户返回空命中的记忆向量占位。"""

    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []

    def delete_memories(self, memory_ids, *, user_id):
        self.deleted.append((user_id, memory_ids[0]))

    def query_by_user(self, user_id, embedding, top_k, query=None, filters=None):
        return []


class _FakeCache:
    """记录失效用户的记忆上下文缓存。"""

    def __init__(self) -> None:
        self.invalidated_user_ids: list[str] = []

    def invalidate_user(self, user_id):
        self.invalidated_user_ids.append(user_id)


class _FakeRepository:
    """提供删除与 active 列表的最小仓储替身。"""

    def __init__(self) -> None:
        self.deleted_pairs: list[tuple[str, str]] = []
        self.active_by_user: dict[str, list[dict]] = {}

    def soft_delete(self, user_id, memory_id):
        self.deleted_pairs.append((user_id, memory_id))
        return True

    def list_active(self, user_id, limit=20, memory_type=None):
        return list(self.active_by_user.get(user_id, []))


class MemoryServiceUnitTests(unittest.TestCase):
    """删除必须同时标记 PostgreSQL、删除 Milvus 向量并清理 Redis。"""

    def test_delete_marks_postgres_and_removes_milvus_and_redis(self) -> None:
        """删除应联动权威库、向量库和缓存。"""
        module = _load_module("backend.app.services.memory_service")
        vector_store = _FakeVectorStore()
        cache = _FakeCache()
        repository = _FakeRepository()
        service = module.MemoryService(
            repository=repository,
            vector_store=vector_store,
            cache=cache,
        )

        result = service.delete(user_id="user-a", memory_id="memory-001")

        assert result["status"] == "deleted"
        assert vector_store.deleted == [("user-a", "memory-001")]
        assert cache.invalidated_user_ids == ["user-a"]
        assert repository.deleted_pairs == [("user-a", "memory-001")]

    def test_context_for_user_marks_fallback_structured_when_vector_empty(self) -> None:
        """向量召回为空时必须回退到 PostgreSQL 结构化记忆。"""
        module = _load_module("backend.app.services.memory_service")
        vector_store = _FakeVectorStore()
        cache = _FakeCache()
        repository = _FakeRepository()
        repository.active_by_user["user-a"] = [
            {
                "memory_id": "memory-001",
                "user_id": "user-a",
                "memory_type": "preference",
                "content": "偏好关注暴雨",
                "status": "active",
            }
        ]
        service = module.MemoryService(
            repository=repository,
            vector_store=vector_store,
            cache=cache,
        )

        context = service.context_for_user("user-a")

        assert context["user_id"] == "user-a"
        assert context["fallback_structured"] is True
        assert any(
            item["content"] == "偏好关注暴雨" for item in context["memories"]
        )

    def test_extract_candidates_only_keeps_explicit_memory_signals(self) -> None:
        """只从明确偏好、区域、灾种和分析习惯中提取候选。"""
        module = _load_module("backend.app.services.memory_service")
        service = module.MemoryService(
            repository=_FakeRepository(),
            vector_store=_FakeVectorStore(),
            cache=_FakeCache(),
        )
        messages = [
            {"role": "user", "content": "我以后重点关注暴雨灾害，请每次带上山西北部。"},
            {"role": "assistant", "content": "这是单次分析的临时结论，不是偏好。"},
        ]
        candidates = service.extract_candidates(
            user_id="user-a",
            session_id="session-1",
            messages=messages,
        )
        assert len(candidates) >= 1
        assert all(item["user_id"] == "user-a" for item in candidates)
        assert all(item["status"] == "candidate" for item in candidates)


class OrchestratorDependenciesMemoryTests(unittest.TestCase):
    """编排器依赖应能携带长期记忆服务。"""

    def test_orchestrator_dependencies_accept_memory_service(self) -> None:
        """OrchestratorDependencies 必须支持 memory_service 字段。"""
        deps = OrchestratorDependencies(
            session_store=object(),
            document_qa=object(),
            structured_query=object(),
            context_cases=object(),
            presenter=object(),
            standard_case_store=object(),
            llm_client=object(),
            multidim_agent=object(),
            multidim_store=object(),
            smart_agent=object(),
            intent_agent=object(),
            memory_service=object(),
        )
        assert deps.memory_service is not None


if __name__ == "__main__":
    unittest.main()
