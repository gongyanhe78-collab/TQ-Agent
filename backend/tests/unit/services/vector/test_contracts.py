"""统一文档向量存储契约和 JSON 适配器行为测试。"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from backend.app.models import DocumentChunk


def _load_module(module_name: str):
    """导入待实现模块，并把缺失实现明确报告为测试失败。"""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        pytest.fail(f"待实现模块缺失：{module_name}：{exc}")


def _seed_authority(path: Path, chunks: list[DocumentChunk]) -> None:
    """按现有 chunk JSON 格式写入权威正文夹具。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sample_chunk(
    *,
    chunk_id: str = "chunk-001",
    content: str = "2025年7月出现暴雨天气过程。",
    source_pdf: str = "case-2025-07.pdf",
    embedding: list[float] | None = None,
) -> DocumentChunk:
    """构造具备稳定来源、顺序号和真实向量的文档块。"""
    return DocumentChunk(
        source_pdf=source_pdf,
        chunk_id=chunk_id,
        chunk_no=1,
        content=content,
        file_path=f"resource/{source_pdf}",
        embedding=embedding if embedding is not None else [1.0, 0.0],
    )


def test_contracts_expose_complete_hit_collection_query_and_memory_shapes() -> None:
    """契约对象应携带检索模式、追踪字段、双得分和来源过滤元数据。"""
    contracts = _load_module("backend.app.services.vector.contracts")

    info = contracts.CollectionInfo(
        backend="json",
        collection="weather_document_chunks",
        count=1,
        dimension=2,
        schema_version="v1",
        health_status="healthy",
    )
    query = contracts.VectorQuery(
        query_text="暴雨",
        embedding=[1.0, 0.0],
        top_k=3,
        filters={"source_pdf": "case-2025-07.pdf"},
        request_id="request-001",
        trace_id="trace-001",
    )
    hit = contracts.VectorHit(
        chunk_id="chunk-001",
        content="权威正文",
        raw_score=0.8,
        fused_score=0.9,
        metadata={
            "source_pdf": "case-2025-07.pdf",
            "chunk_no": 1,
            "file_path": "resource/case-2025-07.pdf",
            "applied_filters": {"source_pdf": "case-2025-07.pdf"},
        },
        backend="json",
        collection="weather_document_chunks",
        retrieval_mode="vector/json",
        request_id="request-001",
        trace_id="trace-001",
    )

    assert info["schema_version"] == "v1"
    assert info.get("backend") == "json"
    assert query.filters == {"source_pdf": "case-2025-07.pdf"}
    assert hit.id == "chunk-001"
    assert hit.score == 0.9
    assert hit.chunk.content == "权威正文"
    assert hit.chunk.source_pdf == "case-2025-07.pdf"
    assert hit.metadata["applied_filters"] == {"source_pdf": "case-2025-07.pdf"}
    assert contracts.DocumentVectorStore is not None
    assert contracts.MemoryVectorStore is not None

    memory_store = contracts.EmptyMemoryVectorStore()
    assert memory_store.query_by_user("user-a", [1.0, 0.0], top_k=3) == []
    assert memory_store.health()["health_status"] == "healthy"
    memory_store.close()
    memory_store.close()
    assert memory_store.health()["health_status"] == "closed"


def test_document_contract_declares_every_required_operation() -> None:
    """文档存储协议必须逐项暴露 Task 5 固定接口。"""
    contracts = _load_module("backend.app.services.vector.contracts")

    required_methods = {
        "upsert_chunks",
        "delete_chunks",
        "query",
        "get_chunk",
        "list_chunks",
        "list_chunk_ids",
        "collection_info",
        "rebuild",
        "health",
        "close",
    }
    assert required_methods <= set(contracts.DocumentVectorStore.__dict__)


def test_json_adapter_reads_authority_body_scores_filters_and_trace(
    tmp_path: Path,
) -> None:
    """JSON 索引只提供向量，正文、得分、过滤和追踪信息必须来自真实链路。"""
    json_store_module = _load_module("backend.app.services.vector.json_document_store")
    contracts = _load_module("backend.app.services.vector.contracts")
    authority_path = tmp_path / "chunks.json"
    authority_chunk = _sample_chunk(content="权威 JSON 正文：暴雨导致城市内涝。")
    _seed_authority(authority_path, [authority_chunk])

    store = json_store_module.JsonDocumentVectorStore(
        authority_path=authority_path,
        index_path=tmp_path / "document_vector_index.json",
        collection_name="weather_document_chunks",
        schema_version="v1",
    )
    store.upsert_chunks(
        [
            _sample_chunk(content="不得进入查询结果的索引副本正文。"),
            _sample_chunk(
                chunk_id="chunk-other",
                content="其他来源的降雪过程。",
                source_pdf="other.pdf",
                embedding=[0.0, 1.0],
            ),
        ]
    )

    query = contracts.VectorQuery(
        query_text="暴雨",
        embedding=[1.0, 0.0],
        top_k=2,
        filters={"source_pdf": "case-2025-07.pdf"},
        request_id="request-json",
        trace_id="trace-json",
    )
    hits = store.query([1.0, 0.0], top_k=2, query=query)

    assert len(hits) == 1
    assert hits[0].content == authority_chunk.content
    assert hits[0].raw_score == pytest.approx(1.0)
    assert hits[0].fused_score == pytest.approx(1.0)
    assert hits[0].backend == "json"
    assert hits[0].collection == "weather_document_chunks"
    assert hits[0].retrieval_mode == "vector/json"
    assert hits[0].request_id == "request-json"
    assert hits[0].trace_id == "trace-json"
    assert hits[0].metadata["source_pdf"] == "case-2025-07.pdf"
    assert hits[0].metadata["applied_filters"] == {
        "source_pdf": "case-2025-07.pdf"
    }
    assert "不得进入查询结果的索引副本正文" not in hits[0].content


def test_json_adapter_lexical_mode_is_honest_and_returns_empty_without_matches(
    tmp_path: Path,
) -> None:
    """JSON 词面查询必须标记 lexical/json，零命中时不得扩大返回范围。"""
    json_store_module = _load_module("backend.app.services.vector.json_document_store")
    contracts = _load_module("backend.app.services.vector.contracts")
    authority_path = tmp_path / "chunks.json"
    authority_chunk = _sample_chunk(content="权威 JSON 正文：暴雨导致城市内涝。")
    _seed_authority(authority_path, [authority_chunk])
    store = json_store_module.JsonDocumentVectorStore(
        authority_path=authority_path,
        index_path=tmp_path / "document_vector_index.json",
        collection_name="weather_document_chunks",
        schema_version="v1",
    )
    store.rebuild()

    query = contracts.VectorQuery(
        query_text="暴雨 内涝",
        embedding=[],
        top_k=5,
        request_id="request-lexical",
        trace_id="trace-lexical",
    )
    hits = store.query([], top_k=5, query=query)
    assert len(hits) == 1
    assert hits[0].retrieval_mode == "lexical/json"
    assert hits[0].raw_score > 0
    assert hits[0].fused_score == hits[0].raw_score
    assert hits[0].metadata["source_pdf"] == "case-2025-07.pdf"

    no_match = contracts.VectorQuery(
        query_text="沙尘",
        embedding=[],
        top_k=5,
        request_id="request-empty",
        trace_id="trace-empty",
    )
    assert store.query([], top_k=5, query=no_match) == []


def test_json_adapter_delete_is_index_only_and_lifecycle_is_idempotent(
    tmp_path: Path,
) -> None:
    """删除只移除索引副本，权威正文保留；关闭和健康检查可重复调用。"""
    json_store_module = _load_module("backend.app.services.vector.json_document_store")
    authority_path = tmp_path / "chunks.json"
    chunk = _sample_chunk()
    _seed_authority(authority_path, [chunk])
    store = json_store_module.JsonDocumentVectorStore(
        authority_path=authority_path,
        index_path=tmp_path / "document_vector_index.json",
        collection_name="weather_document_chunks",
        schema_version="v1",
    )
    store.upsert_chunks([chunk])

    assert store.get_chunk(chunk.chunk_id).content == chunk.content
    store.delete_chunks([chunk.chunk_id])
    store.delete_chunks([chunk.chunk_id])
    assert chunk.chunk_id in store.list_chunk_ids()
    assert store.collection_info()["count"] == 0
    assert store.get_chunk(chunk.chunk_id).content == chunk.content

    store.close()
    store.close()
    health = store.health()
    assert health["health_status"] == "closed"
    assert health["backend"] == "json"


def test_json_adapter_uses_authority_ids_and_keeps_unembedded_chunks_lexical_only(
    tmp_path: Path,
) -> None:
    """无向量权威块应保留过滤元数据和词面可见性，但不得进入向量查询。"""
    json_store_module = _load_module(
        "backend.app.services.vector.json_document_store"
    )
    contracts = _load_module("backend.app.services.vector.contracts")
    authority_path = tmp_path / "authority" / "chunks.json"
    index_path = tmp_path / "authority" / "document_vector_index.json"
    embedded = _sample_chunk(
        chunk_id="chunk-embedded",
        content="暴雨过程造成城市内涝。",
    )
    unembedded = _sample_chunk(
        chunk_id="chunk-unembedded",
        content="暴雨过程伴随短时大风。",
        source_pdf="case-other.pdf",
        embedding=[],
    )
    _seed_authority(authority_path, [embedded, unembedded])
    store = json_store_module.JsonDocumentVectorStore(
        authority_path=authority_path,
        index_path=index_path,
        collection_name="weather_document_chunks",
        schema_version="v1",
    )

    store.upsert_chunks([embedded, unembedded])
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    unembedded_item = index_payload["items"].get(unembedded.chunk_id)

    assert unembedded_item is not None
    assert "embedding" not in unembedded_item
    assert unembedded_item["metadata"]["source_pdf"] == "case-other.pdf"
    assert store.list_chunk_ids() == [
        embedded.chunk_id,
        unembedded.chunk_id,
    ]
    assert store.collection_info()["count"] == 1

    lexical_query = contracts.VectorQuery(
        query_text="暴雨",
        embedding=[],
        top_k=5,
    )
    lexical_hits = store.query([], top_k=5, query=lexical_query)
    assert {hit.chunk_id for hit in lexical_hits} == {
        embedded.chunk_id,
        unembedded.chunk_id,
    }
    assert all(
        hit.retrieval_mode == "lexical/json"
        for hit in lexical_hits
    )

    vector_query = contracts.VectorQuery(
        query_text="暴雨",
        embedding=[1.0, 0.0],
        top_k=5,
    )
    vector_hits = store.query(
        [1.0, 0.0],
        top_k=5,
        query=vector_query,
    )
    assert [hit.chunk_id for hit in vector_hits] == [embedded.chunk_id]


def test_unavailable_milvus_placeholder_stays_unavailable_after_close() -> None:
    """Milvus 占位关闭后仍须表达不可用，而不是伪装成普通已关闭集合。"""
    fallback_module = _load_module(
        "backend.app.services.vector.fallback"
    )
    store = fallback_module.UnavailableMilvusDocumentStore(
        collection_name="weather_document_chunks",
        uri="http://127.0.0.1:19530",
        schema_version="v1",
    )

    store.close()
    store.close()

    assert store.collection_info()["health_status"] == "unavailable"
    assert store.health()["health_status"] == "unavailable"
    assert store.health()["status"] == "unavailable"

