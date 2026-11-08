"""固定后端降级链与不可用 Milvus 占位实现。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.app.models import DocumentChunk
from backend.app.services.vector.contracts import (
    CollectionInfo,
    DocumentVectorStore,
    VectorHit,
    VectorQuery,
)


@dataclass(slots=True)
class FallbackEvent:
    """记录一次真实后端降级及可诊断原因。"""

    backend: str
    reason: str
    health_status: str = "unavailable"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为健康接口可序列化的普通字典。"""
        return {
            "backend": self.backend,
            "reason": self.reason,
            "health_status": self.health_status,
            "metadata": dict(self.metadata),
        }


class UnavailableMilvusDocumentStore:
    """Task 5 的 Milvus 占位实现，只报告不可用且不伪造任何命中。"""

    backend = "milvus"

    def __init__(
        self,
        *,
        collection_name: str,
        uri: str,
        schema_version: str = "v1",
    ) -> None:
        """记录后续 Task 6 所需配置，但本任务不建立网络连接。"""
        self.collection_name = str(collection_name)
        self.uri = str(uri)
        self.schema_version = str(schema_version)
        self._closed = False

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """拒绝在占位后端写入索引，避免造成已持久化的假象。"""
        self._ensure_open()
        raise RuntimeError("Milvus 文档向量后端在 Task 5 中不可用")

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """拒绝在占位后端执行删除。"""
        self._ensure_open()
        raise RuntimeError("Milvus 文档向量后端在 Task 5 中不可用")

    def query(
        self,
        embedding: list[float],
        top_k: int,
        query: VectorQuery | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """拒绝查询，而不是返回空列表伪装成无命中。"""
        self._ensure_open()
        raise RuntimeError("Milvus 文档向量后端在 Task 5 中不可用")

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """占位后端不持有正文，也不读取权威 JSON。"""
        self._ensure_open()
        raise RuntimeError("Milvus 文档向量后端在 Task 5 中不可用")

    def list_chunks(self) -> list[DocumentChunk]:
        """占位后端不持有正文。"""
        self._ensure_open()
        raise RuntimeError("Milvus 文档向量后端在 Task 5 中不可用")

    def list_chunk_ids(self) -> list[str]:
        """占位后端没有真实索引标识。"""
        self._ensure_open()
        raise RuntimeError("Milvus 文档向量后端在 Task 5 中不可用")

    def collection_info(self) -> CollectionInfo:
        """返回不可用集合描述，数量始终为零。"""
        detail = "Milvus 适配器在 Task 5 中仅作不可用占位"
        if self._closed:
            detail = f"{detail}，占位实例已关闭但仍不可用"
        return CollectionInfo(
            backend=self.backend,
            collection=self.collection_name,
            count=0,
            dimension=None,
            schema_version=self.schema_version,
            health_status="unavailable",
            detail=detail,
            metadata={"uri": self.uri},
        )

    def rebuild(self) -> None:
        """拒绝重建占位后端。"""
        self._ensure_open()
        raise RuntimeError("Milvus 文档向量后端在 Task 5 中不可用")

    def health(self) -> dict[str, Any]:
        """返回稳定且可重复读取的不可用健康状态。"""
        info = self.collection_info().to_dict()
        return {**info, "status": info["health_status"]}

    def close(self) -> None:
        """幂等关闭无外部连接的占位后端。"""
        self._closed = True

    def _ensure_open(self) -> None:
        """关闭后仍明确拒绝操作。"""
        if self._closed:
            raise RuntimeError("Milvus document vector placeholder is closed")


class FallbackDocumentVectorStore:
    """把真实终端适配器包装成携带固定降级审计信息的统一存储。"""

    def __init__(
        self,
        *,
        active_store: DocumentVectorStore,
        requested_backend: str,
        fallbacks: list[FallbackEvent] | None = None,
    ) -> None:
        """保存已选中的真实终端适配器和降级事件。"""
        self._active_store = active_store
        self.requested_backend = str(requested_backend)
        self._fallbacks = list(fallbacks or [])
        self._closed = False

    @property
    def backend(self) -> str:
        """返回最终实际选中的后端。"""
        return str(getattr(self._active_store, "backend", ""))

    @property
    def collection_name(self) -> str:
        """返回最终实际集合名。"""
        collection_name = getattr(
            self._active_store,
            "collection_name",
            "",
        )
        if collection_name:
            return str(collection_name)
        return str(self.collection_info().collection)

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """委托活动后端写入索引副本。"""
        self._ensure_open()
        self._active_store.upsert_chunks(chunks)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """委托活动后端幂等删除索引记录。"""
        self._ensure_open()
        self._active_store.delete_chunks(chunk_ids)

    def query(
        self,
        embedding: list[float],
        top_k: int,
        query: VectorQuery | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """委托活动后端查询。"""
        self._ensure_open()
        return self._active_store.query(
            embedding,
            top_k,
            query=query,
            filters=filters,
        )

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """从活动后端对应的权威 JSON 读取正文。"""
        self._ensure_open()
        return self._active_store.get_chunk(chunk_id)

    def list_chunks(self) -> list[DocumentChunk]:
        """从活动后端对应的权威 JSON 列出正文。"""
        self._ensure_open()
        return self._active_store.list_chunks()

    def list_chunk_ids(self) -> list[str]:
        """列出活动后端中的真实索引标识。"""
        self._ensure_open()
        return self._active_store.list_chunk_ids()

    def collection_info(self) -> CollectionInfo:
        """返回活动后端的真实集合信息。"""
        return self._active_store.collection_info()

    def rebuild(self) -> None:
        """委托活动后端按权威正文重建索引。"""
        self._ensure_open()
        self._active_store.rebuild()

    def health(self) -> dict[str, Any]:
        """在终端健康状态中追加固定链降级记录。"""
        if self._closed:
            return {
                **CollectionInfo(
                    backend=self.backend,
                    collection=self.collection_name,
                    health_status="closed",
                    schema_version="v1",
                    detail="文档向量降级存储已关闭",
                ).to_dict(),
                "status": "closed",
                "requested_backend": self.requested_backend,
                "fallbacks": [event.to_dict() for event in self._fallbacks],
            }
        health = dict(self._active_store.health())
        health["requested_backend"] = self.requested_backend
        health["fallbacks"] = [event.to_dict() for event in self._fallbacks]
        return health

    def close(self) -> None:
        """只关闭一次活动后端，重复关闭保持空操作。"""
        if self._closed:
            return
        self._closed = True
        self._active_store.close()

    def _ensure_open(self) -> None:
        """阻止关闭后的降级包装器继续使用。"""
        if self._closed:
            raise RuntimeError("fallback document vector store is closed")
