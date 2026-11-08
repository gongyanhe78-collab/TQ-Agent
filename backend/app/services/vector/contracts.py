"""文档向量库与长期记忆库共享的数据契约。"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from backend.app.models import DocumentChunk


@dataclass(slots=True)
class CollectionInfo(Mapping[str, Any]):
    """描述一个真实向量集合，并兼容既有字典式读取方式。"""

    backend: str
    collection: str
    count: int = 0
    dimension: int | None = None
    schema_version: str = "v1"
    health_status: str = "healthy"
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """展开为可直接返回给现有接口的普通字典。"""
        return {
            "backend": self.backend,
            "collection": self.collection,
            "name": self.collection,
            "count": int(self.count),
            "dimension": self.dimension,
            "schema_version": self.schema_version,
            "health_status": self.health_status,
            "detail": self.detail,
            "metadata": dict(self.metadata),
            # 兼容旧知识库和系统状态接口使用的 source 字段。
            "source": self.backend,
        }

    def __getitem__(self, key: str) -> Any:
        """支持现有调用方通过键读取集合信息。"""
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        """按稳定顺序遍历集合信息键。"""
        return iter(self.to_dict())

    def __len__(self) -> int:
        """返回集合信息字段数量。"""
        return len(self.to_dict())


@dataclass(slots=True)
class VectorQuery:
    """统一描述一次文档或记忆向量查询。"""

    query_text: str = ""
    embedding: list[float] = field(default_factory=list)
    top_k: int = 5
    filters: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    trace_id: str = ""
    user_id: str = ""


@dataclass(slots=True)
class VectorHit:
    """统一返回真实后端、检索模式和得分可追溯的向量命中。"""

    chunk_id: str
    content: str
    raw_score: float
    fused_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    backend: str = ""
    collection: str = ""
    retrieval_mode: str = ""
    request_id: str = ""
    trace_id: str = ""

    @property
    def id(self) -> str:
        """返回对外稳定的文档块标识。"""
        return self.chunk_id

    @property
    def score(self) -> float:
        """优先返回融合得分，兼容既有检索结果只读取 score 的调用方。"""
        return (
            float(self.fused_score)
            if self.fused_score is not None
            else float(self.raw_score)
        )

    @property
    def chunk(self) -> DocumentChunk:
        """把统一命中投影为既有 DocumentChunk 调用形状。"""
        try:
            chunk_no = int(self.metadata.get("chunk_no") or 0)
        except (TypeError, ValueError):
            chunk_no = 0
        return DocumentChunk(
            source_pdf=str(self.metadata.get("source_pdf") or ""),
            chunk_id=self.chunk_id,
            chunk_no=chunk_no,
            content=self.content,
            file_path=str(self.metadata.get("file_path") or "") or None,
            embedding=None,
        )

    def to_dict(self) -> dict[str, Any]:
        """展开为可持久化或透传给上层的普通字典。"""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "raw_score": float(self.raw_score),
            "fused_score": (
                float(self.fused_score)
                if self.fused_score is not None
                else None
            ),
            "metadata": dict(self.metadata),
            "backend": self.backend,
            "collection": self.collection,
            "retrieval_mode": self.retrieval_mode,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
        }


@runtime_checkable
class DocumentVectorStore(Protocol):
    """统一文档向量存储的固定操作接口。"""

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """写入或更新索引副本，不修改权威正文。"""

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """幂等删除索引副本，不删除权威正文。"""

    def query(
        self,
        embedding: list[float],
        top_k: int,
        query: VectorQuery | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """按真实向量或词面模式查询统一命中。"""

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """从权威 chunk JSON 读取单个正文块。"""

    def list_chunks(self) -> list[DocumentChunk]:
        """从权威 chunk JSON 列出正文块。"""

    def list_chunk_ids(self) -> list[str]:
        """列出权威正文中的稳定块标识，索引删除不改变该全集。"""

    def collection_info(self) -> CollectionInfo:
        """返回当前后端的集合状态。"""

    def rebuild(self) -> None:
        """依据权威 JSON 和既有向量副本重建索引。"""

    def health(self) -> dict[str, Any]:
        """返回真实健康状态，重复调用保持幂等。"""

    def close(self) -> None:
        """释放客户端资源，重复调用保持幂等。"""


@runtime_checkable
class MemoryVectorStore(Protocol):
    """长期记忆向量库的最小占位契约。"""

    def upsert_memories(self, memories: list[dict[str, Any]]) -> None:
        """保存或更新记忆索引副本。"""

    def save(self, memories: list[dict[str, Any]]) -> None:
        """兼容直接保存记忆的调用语义。"""

    def delete_memories(
        self,
        memory_ids: list[str],
        *,
        user_id: str,
    ) -> None:
        """按用户删除记忆索引。"""

    def delete(self, memory_ids: list[str], *, user_id: str) -> None:
        """兼容直接删除记忆的调用语义。"""

    def query_by_user(
        self,
        user_id: str,
        embedding: list[float],
        top_k: int,
        query: VectorQuery | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """只在指定用户的记忆范围内检索。"""

    def rebuild(self, user_id: str | None = None) -> None:
        """重建全部或指定用户的记忆索引。"""

    def health(self) -> dict[str, Any]:
        """返回记忆后端健康状态。"""

    def close(self) -> None:
        """释放记忆后端资源。"""


class EmptyMemoryVectorStore:
    """不接入业务数据的可用空记忆实现，便于后续 Milvus 适配替换。"""

    backend = "memory"

    def __init__(self) -> None:
        """初始化无状态占位实现。"""
        self._closed = False

    def upsert_memories(self, memories: list[dict[str, Any]]) -> None:
        """空实现不持久化任何记忆。"""
        self._ensure_open()

    def save(self, memories: list[dict[str, Any]]) -> None:
        """兼容保存别名并保持空行为。"""
        self.upsert_memories(memories)

    def delete_memories(
        self,
        memory_ids: list[str],
        *,
        user_id: str,
    ) -> None:
        """空实现不执行删除副作用。"""
        self._ensure_open()

    def delete(self, memory_ids: list[str], *, user_id: str) -> None:
        """兼容删除别名并保持空行为。"""
        self.delete_memories(memory_ids, user_id=user_id)

    def query_by_user(
        self,
        user_id: str,
        embedding: list[float],
        top_k: int,
        query: VectorQuery | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """占位实现始终返回空结果，不伪造记忆命中。"""
        self._ensure_open()
        return []

    def rebuild(self, user_id: str | None = None) -> None:
        """空实现无需重建。"""
        self._ensure_open()

    def health(self) -> dict[str, Any]:
        """返回占位实现当前生命周期状态。"""
        return CollectionInfo(
            backend=self.backend,
            collection="memory_vectors",
            count=0,
            dimension=None,
            schema_version="v1",
            health_status="closed" if self._closed else "healthy",
            detail="长期记忆占位实现",
        ).to_dict()

    def close(self) -> None:
        """幂等关闭占位实现。"""
        self._closed = True

    def _ensure_open(self) -> None:
        """阻止关闭后的空实现继续被业务误用。"""
        if self._closed:
            raise RuntimeError("memory vector store is closed")

def count_collection_rows(client: Any, collection_name: str) -> int:
    """返回 Milvus 集合真实行数，优先用 count(*) 查询，失败时回退统计接口。"""
    try:
        result = client.query(
            collection_name=collection_name,
            output_fields=["count(*)"],
        )
        rows = list(result or [])
        if rows:
            first = rows[0]
            if isinstance(first, dict):
                return int(first.get("count(*)") or 0)
    except Exception:
        pass
    try:
        return int(
            client.get_collection_stats(collection_name).get("row_count", 0)
        )
    except Exception:
        return 0
