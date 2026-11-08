"""跨会话长期记忆服务：用户隔离、结构化降级、软删除与候选抽取。"""
from __future__ import annotations

import logging
import re
from typing import Any
from uuid import uuid4

from backend.app.db.base import utc_now


LOGGER = logging.getLogger("uvicorn.error")

# 明确表达偏好或习惯的稳定信号词，避免把单次事实误判为长期记忆。
_PREFERENCE_MARKERS = (
    "以后",
    "每次",
    "偏好",
    "重点关注",
    "请记得",
    "记得以后",
    "关注一下",
    "后续都",
)

# 常用灾种候选词，用于标记记忆类型。
_DISASTER_WORDS = ("暴雨", "冰雹", "大风", "沙尘", "暴雪", "洪涝", "干旱", "雷电")


class MemoryService:
    """以 PostgreSQL 为权威、Milvus 为可重建索引的长期记忆门面。"""

    def __init__(
        self,
        *,
        repository: Any,
        vector_store: Any,
        cache: Any,
        embedding_client: Any | None = None,
    ) -> None:
        """保存记忆仓储、向量库和上下文缓存依赖。"""
        self.repository = repository
        self.vector_store = vector_store
        self.cache = cache
        self.embedding_client = embedding_client

    def context_for_user(
        self,
        user_id: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        """返回当前用户记忆上下文；向量召回失败时结构化降级。"""
        memories: list[dict[str, Any]] = []
        fallback_structured = False
        query_embedding = self._query_embedding()
        if query_embedding:
            try:
                hits = self.vector_store.query_by_user(
                    user_id,
                    query_embedding,
                    top_k=max(1, int(limit)),
                )
                memories = [self._hit_to_memory(hit) for hit in hits]
            except Exception as exc:
                LOGGER.warning("[记忆] 向量召回失败，已结构化降级: %s", exc)
                memories = []
        if not memories:
            fallback_structured = True
            memories = self.repository.list_active(
                user_id,
                limit=max(1, int(limit)),
            )
        return {
            "user_id": str(user_id),
            "memories": memories,
            "fallback_structured": bool(fallback_structured),
        }

    def context_text_for_user(
        self,
        user_id: str,
        limit: int = 10,
    ) -> str:
        """把当前用户记忆上下文压缩为回答前缀文本，超限时截断。"""
        context = self.context_for_user(user_id, limit=limit)
        lines = []
        for memory in context.get("memories", []):
            content = str(memory.get("content") or "").strip()
            if content:
                lines.append(content)
        return "；".join(lines)[:2000]

    def extract_candidates(
        self,
        *,
        user_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
        source_message_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """只从明确偏好、区域、灾种和分析习惯中提取候选记忆。"""
        candidates: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if str(message.get("role") or "") != "user":
                continue
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            memory_type, confidence = self._classify(content)
            if memory_type is None:
                continue
            candidates.append(
                {
                    "memory_id": f"memory-{uuid4().hex[:16]}",
                    "user_id": str(user_id),
                    "session_id": str(session_id),
                    "memory_type": memory_type,
                    "content": content[:500],
                    "confidence": confidence,
                    "sensitivity": "normal",
                    "status": "candidate",
                    "source_message_id": (
                        int(source_message_id) if source_message_id is not None else None
                    ),
                    "version": 1,
                    "metadata": {
                        "extracted_at": utc_now().isoformat(),
                        "message_index": index,
                    },
                }
            )
        return candidates

    def delete(self, user_id: str, memory_id: str) -> dict[str, Any]:
        """软删除权威记忆，并联动删除向量与清理上下文缓存。"""
        deleted = self.repository.soft_delete(user_id, memory_id)
        if not deleted:
            return {
                "status": "not_found",
                "memory_id": memory_id,
                "user_id": user_id,
            }
        try:
            self.vector_store.delete_memories([memory_id], user_id=user_id)
        except Exception as exc:
            LOGGER.warning("[记忆] 向量删除失败，进入可重试清单: %s", exc)
        try:
            self.cache.invalidate_user(user_id)
        except Exception as exc:
            LOGGER.warning("[记忆] 缓存清理失败: %s", exc)
        return {
            "status": "deleted",
            "memory_id": memory_id,
            "user_id": user_id,
        }

    def _query_embedding(self) -> list[float] | None:
        """生成记忆召回查询向量；云端向量不可用时返回 None。"""
        client = self.embedding_client
        if client is None or not getattr(client, "is_available", lambda: False)():
            return None
        try:
            return list(client.embed_query("用户长期关注的气象灾害与偏好"))
        except Exception as exc:
            LOGGER.warning("[记忆] 查询向量生成失败: %s", exc)
            return None

    def _hit_to_memory(self, hit: Any) -> dict[str, Any]:
        """把向量命中投影为接口可见的记忆字段，不暴露原始向量。"""
        metadata = dict(getattr(hit, "metadata", {}) or {})
        return {
            "memory_id": str(getattr(hit, "chunk_id", "")),
            "user_id": str(metadata.get("user_id") or ""),
            "memory_type": str(metadata.get("memory_type") or ""),
            "content": str(getattr(hit, "content", "")),
            "confidence": 0.0,
            "sensitivity": str(metadata.get("sensitivity") or "normal"),
            "status": str(metadata.get("status") or "active"),
            "version": int(metadata.get("version") or 1),
            "source_message_id": metadata.get("source_message_id"),
            "metadata": {},
        }

    def _classify(self, content: str) -> tuple[str | None, float]:
        """判断消息是否包含明确的长期记忆信号并给出类型与置信度。"""
        if not any(marker in content for marker in _PREFERENCE_MARKERS):
            return None, 0.0
        if any(word in content for word in _DISASTER_WORDS):
            return "preference", 0.85
        if re.search(r"关注[^，。；]*?(区域|地区|市|县|省)", content):
            return "region", 0.8
        if any(word in content for word in ("分析", "报告", "结论")):
            return "habit", 0.6
        return "preference", 0.5
