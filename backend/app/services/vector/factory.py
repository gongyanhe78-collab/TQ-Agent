"""按固定 Milvus 到 Chroma 再到 JSON 顺序构造文档向量存储。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.core.config import Settings
from backend.app.services.vector.chroma_document_store import (
    ChromaDocumentVectorStore,
)
from backend.app.services.vector.contracts import DocumentVectorStore
from backend.app.services.vector.fallback import (
    FallbackDocumentVectorStore,
    FallbackEvent,
)
from backend.app.services.vector.json_document_store import (
    JsonDocumentVectorStore,
    read_authority_chunks,
)
from backend.app.services.vector.milvus_document_store import (
    MilvusDocumentStore,
)

_BACKEND_ORDER = ("milvus", "chroma", "json")


def build_document_vector_store(
    settings: Settings,
    embedding_client: Any,
) -> DocumentVectorStore:
    """按配置起点沿固定降级链选择第一个真实健康后端。"""
    requested_backend = str(settings.vector_backend or "chroma").strip().lower()
    if requested_backend not in _BACKEND_ORDER:
        raise ValueError(
            "VECTOR_BACKEND 只能是 milvus、chroma 或 json，"
            f"当前为 {requested_backend!r}"
        )

    authority_path = settings.document_index_dir / "chunks.json"
    start_index = _BACKEND_ORDER.index(requested_backend)
    fallbacks: list[FallbackEvent] = []
    for backend in _BACKEND_ORDER[start_index:]:
        if backend == "milvus":
            candidate = MilvusDocumentStore(
                uri=settings.milvus_uri,
                collection_name=settings.milvus_collection_name,
                authority_path=authority_path,
                dimension=_infer_dimension(authority_path),
                token=settings.milvus_token,
                database=settings.milvus_database,
                schema_version=settings.vector_schema_version,
                dense_top_k=settings.milvus_dense_top_k,
                sparse_top_k=settings.milvus_sparse_top_k,
                rrf_k=settings.milvus_rrf_k,
                timeout_seconds=settings.milvus_timeout_seconds,
                standard_cases_path=settings.standard_cases_path,
                embedding_model=settings.embedding_model,
                document_version="v1",
            )
        elif backend == "chroma":
            candidate = ChromaDocumentVectorStore(
                authority_path=authority_path,
                persist_dir=settings.document_index_dir,
                collection_name=settings.document_collection_name,
                schema_version=settings.vector_schema_version,
                embedding_client=embedding_client,
            )
        else:
            candidate = JsonDocumentVectorStore(
                authority_path=authority_path,
                index_path=settings.document_index_dir / "document_vector_index.json",
                collection_name=settings.document_collection_name,
                schema_version=settings.vector_schema_version,
                embedding_client=embedding_client,
            )

        healthy, status, reason = _candidate_health(candidate)
        if healthy:
            if fallbacks:
                return FallbackDocumentVectorStore(
                    active_store=candidate,
                    requested_backend=requested_backend,
                    fallbacks=fallbacks,
                )
            return candidate

        fallbacks.append(
            FallbackEvent(
                backend=backend,
                reason=reason,
                health_status=status,
            )
        )
        candidate.close()

    raise RuntimeError("固定降级链未产生可用的文档向量后端")


def _infer_dimension(authority_path: Path) -> int:
    """从权威 chunk JSON 推断嵌入维度；无法推断时返回 0 表示不可用。"""
    for chunk in read_authority_chunks(authority_path):
        embedding = chunk.embedding or []
        if embedding:
            return len(embedding)
    return 0


def build_memory_vector_store(
    settings: Settings,
) -> Any:
    """按配置构造记忆向量存储；Milvus 不可用时使用可用空实现。"""
    from backend.app.services.vector.contracts import EmptyMemoryVectorStore
    from backend.app.services.vector.milvus_memory_store import (
        MilvusMemoryStore,
    )

    requested_backend = str(settings.vector_backend or "chroma").strip().lower()
    if requested_backend != "milvus":
        return EmptyMemoryVectorStore()
    dimension = _infer_dimension(settings.document_index_dir / "chunks.json")
    if dimension <= 0:
        return EmptyMemoryVectorStore()
    store = MilvusMemoryStore(
        uri=settings.milvus_uri,
        collection_name=settings.milvus_memory_collection_name,
        dimension=dimension,
        token=settings.milvus_token,
        database=settings.milvus_database,
        schema_version=settings.vector_schema_version,
        dense_top_k=settings.milvus_dense_top_k,
        sparse_top_k=settings.milvus_sparse_top_k,
        rrf_k=settings.milvus_rrf_k,
        timeout_seconds=settings.milvus_timeout_seconds,
    )
    health = store.health()
    status = str(health.get("health_status") or health.get("status") or "")
    if status in {"healthy", "ready", "ok"}:
        return store
    store.close()
    return EmptyMemoryVectorStore()


def _candidate_health(
    candidate: DocumentVectorStore,
) -> tuple[bool, str, str]:
    """把候选健康结果归一化为是否可用以及可记录原因。"""
    try:
        health = dict(candidate.health())
    except Exception as exc:
        return False, "unavailable", f"健康检查异常：{exc}"
    status = str(
        health.get("health_status")
        or health.get("status")
        or "unavailable"
    ).strip().lower()
    healthy = (
        health.get("healthy") is True
        or status in {"healthy", "ready", "ok"}
    )
    reason = str(health.get("detail") or health.get("reason") or "").strip()
    if not healthy and not reason:
        reason = f"后端健康状态为 {status or 'unknown'}"
    return healthy, status or "unknown", reason
