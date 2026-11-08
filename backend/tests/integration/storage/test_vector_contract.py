"""统一文档向量存储的 Chroma、JSON 和固定降级链集成测试。"""
from __future__ import annotations

import importlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.core.config import Settings
from backend.app.models import DocumentChunk, StandardCase


def _load_module(module_name: str):
    """导入待实现模块，并把缺失实现明确报告为测试失败。"""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        pytest.fail(f"待实现模块缺失：{module_name}：{exc}")


def _sample_chunk(
    *,
    chunk_id: str = "chunk-001",
    content: str = "权威 JSON 正文：2025年7月暴雨过程。",
    source_pdf: str = "case-2025-07.pdf",
    embedding: list[float] | None = None,
) -> DocumentChunk:
    """构造两个适配器共同使用的文档块。"""
    return DocumentChunk(
        source_pdf=source_pdf,
        chunk_id=chunk_id,
        chunk_no=1,
        content=content,
        file_path=f"resource/{source_pdf}",
        embedding=embedding if embedding is not None else [1.0, 0.0],
    )


def _seed_authority(path: Path, chunks: list[DocumentChunk]) -> None:
    """写入权威 chunk JSON，避免索引副本反向定义正文。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def make_contract_stores(tmp_path: Path):
    """创建共享契约的 JSON 与真实 Chroma 适配器。"""
    from backend.app.services.vector.chroma_document_store import (
        ChromaDocumentVectorStore,
    )
    from backend.app.services.vector.json_document_store import (
        JsonDocumentVectorStore,
    )

    sample_chunk = _sample_chunk()
    json_root = tmp_path / "json-store"
    chroma_root = tmp_path / "chroma-store"
    json_authority = json_root / "chunks.json"
    chroma_authority = chroma_root / "chunks.json"
    _seed_authority(json_authority, [sample_chunk])
    _seed_authority(chroma_authority, [sample_chunk])

    return (
        sample_chunk,
        [
            JsonDocumentVectorStore(
                authority_path=json_authority,
                index_path=json_root / "document_vector_index.json",
                collection_name="weather_document_chunks",
                schema_version="v1",
            ),
            ChromaDocumentVectorStore(
                authority_path=chroma_authority,
                persist_dir=chroma_root / "chroma",
                collection_name="weather_document_chunks_test",
                schema_version="v1",
            ),
        ],
    )


def test_json_and_chroma_adapters_satisfy_the_same_document_contract(
    tmp_path: Path,
) -> None:
    """JSON 与 Chroma 必须实现同一生命周期和正文读取契约。"""
    sample_chunk, stores = make_contract_stores(tmp_path)
    for store in stores:
        store.upsert_chunks([sample_chunk])
        assert store.get_chunk(sample_chunk.chunk_id).content == sample_chunk.content
        assert sample_chunk.chunk_id in store.list_chunk_ids()
        assert store.collection_info()["schema_version"] == "v1"
        store.close()
        store.close()


def test_chroma_query_uses_authoritative_body_real_scores_and_filters(
    tmp_path: Path,
) -> None:
    """Chroma 只负责向量命中，正文和过滤元数据来自权威 JSON。"""
    from backend.app.services.vector.contracts import VectorQuery
    from backend.app.services.vector.chroma_document_store import (
        ChromaDocumentVectorStore,
    )

    authority_path = tmp_path / "chroma" / "chunks.json"
    authority_chunk = _sample_chunk(content="权威正文，不得被 Chroma document 覆盖。")
    other_chunk = _sample_chunk(
        chunk_id="chunk-other",
        content="另一个来源的沙尘过程。",
        source_pdf="other.pdf",
        embedding=[0.0, 1.0],
    )
    _seed_authority(authority_path, [authority_chunk, other_chunk])
    store = ChromaDocumentVectorStore(
        authority_path=authority_path,
        persist_dir=tmp_path / "chroma" / "db",
        collection_name="weather_filter_test",
        schema_version="v1",
    )
    store.upsert_chunks(
        [
            _sample_chunk(content="不可信 Chroma 索引正文。"),
            other_chunk,
        ]
    )
    query = VectorQuery(
        query_text="暴雨",
        embedding=[1.0, 0.0],
        top_k=5,
        filters={"source_pdf": "case-2025-07.pdf"},
        request_id="request-chroma",
        trace_id="trace-chroma",
    )
    hits = store.query([1.0, 0.0], top_k=5, query=query)

    assert len(hits) == 1
    assert hits[0].content == authority_chunk.content
    assert hits[0].raw_score == pytest.approx(1.0)
    assert hits[0].fused_score == pytest.approx(1.0)
    assert hits[0].backend == "chroma"
    assert hits[0].retrieval_mode == "vector/chroma"
    assert hits[0].request_id == "request-chroma"
    assert hits[0].trace_id == "trace-chroma"
    assert hits[0].metadata["applied_filters"] == {
        "source_pdf": "case-2025-07.pdf"
    }

    no_match = VectorQuery(
        query_text="暴雨",
        embedding=[1.0, 0.0],
        top_k=5,
        filters={"source_pdf": "missing.pdf"},
    )
    assert store.query([1.0, 0.0], top_k=5, query=no_match) == []
    store.close()


def test_json_and_chroma_deletion_is_idempotent_and_index_only(
    tmp_path: Path,
) -> None:
    """两个适配器重复删除均不抛错，且权威正文仍可读取。"""
    sample_chunk, stores = make_contract_stores(tmp_path)
    for store in stores:
        store.upsert_chunks([sample_chunk])
        store.delete_chunks([sample_chunk.chunk_id])
        store.delete_chunks([sample_chunk.chunk_id])
        assert store.list_chunk_ids() == [sample_chunk.chunk_id]
        assert store.collection_info()["count"] == 0
        assert store.get_chunk(sample_chunk.chunk_id).content == sample_chunk.content
        store.close()


def test_chroma_uses_authority_ids_and_keeps_unembedded_chunks_lexical_only(
    tmp_path: Path,
) -> None:
    """Chroma 词面范围应来自权威正文，向量查询只使用真实 Chroma 索引。"""
    from backend.app.services.vector.chroma_document_store import (
        ChromaDocumentVectorStore,
    )
    from backend.app.services.vector.contracts import VectorQuery

    authority_path = tmp_path / "chroma-authority" / "chunks.json"
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
    store = ChromaDocumentVectorStore(
        authority_path=authority_path,
        persist_dir=tmp_path / "chroma-authority" / "db",
        collection_name="weather_authority_scope_test",
        schema_version="v1",
    )

    store.upsert_chunks([embedded, unembedded])

    assert store.list_chunk_ids() == [
        embedded.chunk_id,
        unembedded.chunk_id,
    ]
    assert store.collection_info()["count"] == 1

    lexical_hits = store.query(
        [],
        top_k=5,
        query=VectorQuery(
            query_text="暴雨",
            embedding=[],
            top_k=5,
        ),
    )
    assert {hit.chunk_id for hit in lexical_hits} == {
        embedded.chunk_id,
        unembedded.chunk_id,
    }
    assert all(
        hit.retrieval_mode == "lexical/json"
        for hit in lexical_hits
    )

    vector_hits = store.query(
        [1.0, 0.0],
        top_k=5,
        query=VectorQuery(
            query_text="暴雨",
            embedding=[1.0, 0.0],
            top_k=5,
        ),
    )
    assert [hit.chunk_id for hit in vector_hits] == [embedded.chunk_id]
    store.close()


def test_legacy_document_store_names_keep_authority_and_query_api(
    tmp_path: Path,
) -> None:
    """旧类名应写入权威 JSON，并保留 replace、正文读取和查询调用形状。"""
    from backend.app.services.document_store import (
        ChromaDocumentChunkStore,
        JsonDocumentChunkStore,
    )

    sample_chunk = _sample_chunk()
    stores = (
        (
            JsonDocumentChunkStore(tmp_path / "legacy-json" / "chunks.json"),
            tmp_path / "legacy-json" / "chunks.json",
        ),
        (
            ChromaDocumentChunkStore(
                tmp_path / "legacy-chroma",
                "weather_document_chunks_legacy_test",
            ),
            tmp_path / "legacy-chroma" / "chunks.json",
        ),
    )
    for store, authority_path in stores:
        store.replace_chunks([sample_chunk])
        assert store.get_chunk(sample_chunk.chunk_id).content == sample_chunk.content
        assert store.list_chunk_ids() == [sample_chunk.chunk_id]
        assert store.get_chunks([sample_chunk.chunk_id])[0].content == sample_chunk.content
        hits = store.query([1.0, 0.0], top_k=1)
        assert len(hits) == 1
        assert hits[0].chunk.chunk_id == sample_chunk.chunk_id
        assert hits[0].score == pytest.approx(1.0)
        payload = json.loads(authority_path.read_text(encoding="utf-8"))
        assert payload[0]["content"] == sample_chunk.content
        store.close()
        store.close()


def test_legacy_json_replace_reports_authority_ids_for_unembedded_chunk(
    tmp_path: Path,
) -> None:
    """旧 JSON 包装的 ID 全集不得因重建跳过无向量正文而缩小。"""
    from backend.app.services.document_store import JsonDocumentChunkStore
    from backend.app.services.vector.contracts import VectorQuery

    chunk = _sample_chunk(
        chunk_id="chunk-without-embedding",
        content="暴雨过程未生成向量的权威正文。",
        embedding=[],
    )
    store = JsonDocumentChunkStore(
        tmp_path / "legacy-unembedded" / "chunks.json"
    )

    store.replace_chunks([chunk])

    assert store.list_chunk_ids() == [chunk.chunk_id]
    assert store.collection_info()["count"] == 0
    hits = store.query(
        [],
        top_k=5,
        query=VectorQuery(
            query_text="暴雨",
            embedding=[],
            top_k=5,
        ),
    )
    assert [hit.chunk.chunk_id for hit in hits] == [chunk.chunk_id]
    store.close()


def test_settings_default_vector_backend_preserves_legacy_chroma(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task 5 修复阶段不得把默认部署切到尚未构建的空 JSON 索引。"""
    monkeypatch.delenv("VECTOR_BACKEND", raising=False)

    settings = Settings(document_index_dir=tmp_path)

    assert settings.vector_backend == "chroma"


def test_factory_uses_fixed_milvus_chroma_json_order_and_records_reasons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """工厂从 Milvus 起步时只允许依次降级到 Chroma、JSON。"""
    factory_module = _load_module("backend.app.services.vector.factory")
    contracts = _load_module("backend.app.services.vector.contracts")
    authority_path = tmp_path / "chunks.json"
    _seed_authority(authority_path, [_sample_chunk()])

    class _UnavailableStore:
        """模拟指定后端健康检查失败且不能执行查询。"""

        backend = "chroma"
        collection_name = "weather_document_chunks"

        def health(self) -> dict:
            return {
                "backend": self.backend,
                "health_status": "unavailable",
                "detail": "simulated unavailable",
            }

        def close(self) -> None:
            return None

        def query(self, *_args, **_kwargs):
            raise RuntimeError("simulated unavailable")

    monkeypatch.setattr(
        factory_module,
        "ChromaDocumentVectorStore",
        lambda **_kwargs: _UnavailableStore(),
    )
    settings = Settings(
        document_index_dir=tmp_path,
        document_collection_name="weather_document_chunks",
        vector_backend="milvus",
        vector_schema_version="v1",
        milvus_uri="http://127.0.0.1:19530",
        milvus_collection_name="weather_document_chunks",
    )
    store = factory_module.build_document_vector_store(settings, embedding_client=object())

    assert isinstance(store, contracts.DocumentVectorStore)
    health = store.health()
    assert health["backend"] == "json"
    assert [event["backend"] for event in health["fallbacks"]] == [
        "milvus",
        "chroma",
    ]
    assert all(event["reason"] for event in health["fallbacks"])
    assert store.collection_info()["backend"] == "json"
    store.close()


def test_factory_starting_at_json_never_uses_an_upward_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VECTOR_BACKEND=json 时固定链应只构造 JSON 终端适配器。"""
    factory_module = _load_module("backend.app.services.vector.factory")
    chroma_calls: list[dict] = []
    monkeypatch.setattr(
        factory_module,
        "ChromaDocumentVectorStore",
        lambda **kwargs: chroma_calls.append(kwargs),
    )
    settings = Settings(
        document_index_dir=tmp_path,
        document_collection_name="weather_document_chunks",
        vector_backend="json",
        vector_schema_version="v1",
    )
    store = factory_module.build_document_vector_store(settings, embedding_client=object())

    assert store.collection_info()["backend"] == "json"
    assert chroma_calls == []
    store.close()


def test_factory_passes_embedding_client_into_chroma_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """工厂参数应进入真实适配器构造，而不是被静默丢弃。"""
    factory_module = _load_module("backend.app.services.vector.factory")
    chroma_calls: list[dict] = []

    class _HealthyChromaStore:
        """提供工厂健康探测所需的最小真实适配器形状。"""

        backend = "chroma"
        collection_name = "weather_document_chunks"

        def health(self) -> dict:
            return {
                "backend": self.backend,
                "health_status": "healthy",
                "detail": "simulated healthy",
            }

        def close(self) -> None:
            return None

    def _build_chroma(**kwargs):
        chroma_calls.append(dict(kwargs))
        return _HealthyChromaStore()

    monkeypatch.setattr(
        factory_module,
        "ChromaDocumentVectorStore",
        _build_chroma,
    )
    settings = Settings(
        document_index_dir=tmp_path,
        document_collection_name="weather_document_chunks",
        vector_backend="chroma",
        vector_schema_version="v1",
    )
    embedding_client = object()

    store = factory_module.build_document_vector_store(
        settings,
        embedding_client=embedding_client,
    )

    assert chroma_calls[0]["embedding_client"] is embedding_client
    store.close()


def test_build_services_uses_vector_factory_and_closes_document_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """组合根必须通过统一工厂注入文档存储，并在关闭容器时释放一次。"""
    from backend.app.services import container

    class _DocumentStore:
        """提供组合根构造所需的最小统一存储接口。"""

        backend = "json"
        collection_name = "weather_document_chunks"

        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    document_store = _DocumentStore()
    factory_calls: list[tuple[Settings, object]] = []

    def _build_store(settings: Settings, embedding_client: object):
        factory_calls.append((settings, embedding_client))
        return document_store

    fake_embedding_client = object()
    fake_llm_client = object()
    fake_engine = SimpleNamespace(dispose=lambda: None)
    monkeypatch.setattr(container, "build_document_vector_store", _build_store)
    monkeypatch.setattr(container, "build_engine", lambda _settings: fake_engine)
    monkeypatch.setattr(
        container,
        "get_embedding_client",
        lambda: fake_embedding_client,
    )
    monkeypatch.setattr(container, "get_llm_client", lambda: fake_llm_client)
    monkeypatch.setattr(container, "get_rerank_client", lambda: object())
    monkeypatch.setattr(container, "get_intent_llm_client", lambda: object())
    monkeypatch.setattr(
        container,
        "ensure_non_thinking_client",
        lambda client: client,
    )
    monkeypatch.setattr(
        container,
        "_build_multidim_agent",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        container,
        "SmartCaseMatchAgent",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        container,
        "QueryIntentAgent",
        lambda **_kwargs: SimpleNamespace(),
    )
    settings = Settings(
        document_index_dir=tmp_path,
        smart_case_data_dir=tmp_path,
        case_multidim_runtime_dir=tmp_path,
        image_metadata_path=tmp_path / "image_metadata.json",
        standard_cases_path=tmp_path / "standard_cases.json",
        database_url="sqlite+pysqlite:///:memory:",
    )

    services = container.build_services(settings)

    assert services.document_store is document_store
    assert len(factory_calls) == 1
    assert factory_calls[0] == (settings, fake_embedding_client)
    services.close()
    services.close()
    assert document_store.close_count == 1


class _UnavailableEmbeddingClient:
    """模拟向量模型不可用，确保检索器走真实词面降级。"""

    def is_available(self) -> bool:
        return False

    def expected_dimension(self) -> int | None:
        return None


class _AvailableEmbeddingClient:
    """为检索器故障矩阵提供真实可用的向量客户端。"""

    def __init__(
        self,
        *,
        expected_dimension: int,
        embedding: list[float],
    ) -> None:
        self._expected_dimension = expected_dimension
        self._embedding = list(embedding)

    def is_available(self) -> bool:
        return True

    def expected_dimension(self) -> int:
        return self._expected_dimension

    def embed_query(self, _question: str) -> list[float]:
        return list(self._embedding)


class _FailingBackendStore:
    """模拟活动后端可读取正文但查询接口失败的故障矩阵。"""

    backend = "chroma"
    collection_name = "weather_document_chunks"

    def __init__(self, chunk: DocumentChunk) -> None:
        self._chunk = chunk

    def collection_info(self) -> dict:
        return {
            "backend": self.backend,
            "collection": self.collection_name,
            "count": 1,
            "dimension": 2,
            "health_status": "healthy",
        }

    def query(self, *_args, **_kwargs):
        raise RuntimeError("模拟选定后端服务不可达")

    def list_chunks(self) -> list[DocumentChunk]:
        return [self._chunk]


class _StandardCaseStore:
    """为检索器回归提供固定个例范围。"""

    def __init__(self) -> None:
        self.cases = [
            StandardCase(
                case_id="case-001",
                title="暴雨过程",
                date_range="2025年7月",
                source_pdf="case-2025-07.pdf",
                source_chunk_ids=["chunk-001"],
            )
        ]

    def list_cases(self) -> list[StandardCase]:
        return list(self.cases)


def test_document_retriever_preserves_lexical_fallback_and_marks_json(
    tmp_path: Path,
) -> None:
    """模型不可用时保持原词面召回，并返回诚实的 lexical/json 元数据。"""
    from backend.app.observability.context import bind_request_context
    from backend.app.services.query.document_retrieval import DocumentRetriever
    from backend.app.services.vector.json_document_store import (
        JsonDocumentVectorStore,
    )

    authority_path = tmp_path / "chunks.json"
    chunk = _sample_chunk()
    _seed_authority(authority_path, [chunk])
    store = JsonDocumentVectorStore(
        authority_path=authority_path,
        index_path=tmp_path / "document_vector_index.json",
        collection_name="weather_document_chunks",
        schema_version="v1",
    )
    store.rebuild()
    retriever = DocumentRetriever(
        embedding_client=_UnavailableEmbeddingClient(),
        document_store=store,
        standard_case_store=_StandardCaseStore(),
    )

    with bind_request_context(
        request_id="request-retriever",
        trace_id="a" * 32,
    ):
        mode, hits = retriever.retrieve("暴雨过程", top_k=5, top_n=3)

    assert mode == "lexical/json"
    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == "chunk-001"
    assert hits[0].backend == "json"
    assert hits[0].retrieval_mode == "lexical/json"
    assert hits[0].request_id == "request-retriever"
    assert hits[0].trace_id == "a" * 32
    store.close()


def test_document_retriever_reports_dimension_mismatch_without_blocking_answer(
    tmp_path: Path,
) -> None:
    """索引维度不匹配时应保留可诊断原因，同时继续返回权威 JSON 词面证据。"""
    from backend.app.services.query.document_retrieval import (
        DocumentRetriever,
    )
    from backend.app.services.vector.json_document_store import (
        JsonDocumentVectorStore,
    )

    authority_path = tmp_path / "dimension" / "chunks.json"
    chunk = _sample_chunk(content="暴雨过程造成城市内涝。")
    _seed_authority(authority_path, [chunk])
    store = JsonDocumentVectorStore(
        authority_path=authority_path,
        index_path=tmp_path / "dimension" / "document_vector_index.json",
        collection_name="weather_document_chunks",
        schema_version="v1",
    )
    store.rebuild()
    retriever = DocumentRetriever(
        embedding_client=_AvailableEmbeddingClient(
            expected_dimension=3,
            embedding=[1.0, 0.0, 0.0],
        ),
        document_store=store,
        standard_case_store=_StandardCaseStore(),
    )

    mode, hits = retriever.retrieve("暴雨过程", top_k=5, top_n=3)

    diagnostics = getattr(
        retriever,
        "last_retrieval_diagnostics",
        None,
    )
    assert mode == "lexical/json"
    assert diagnostics is not None
    assert diagnostics["category"] == "dimension_mismatch"
    assert diagnostics["backend"] == "json"
    assert "expected 3, got 2" in diagnostics["reason"]
    assert diagnostics["exception_type"] == "RuntimeError"
    assert hits[0].metadata["retrieval_fallback"] == diagnostics
    store.close()


def test_document_retriever_reports_backend_failure_with_safe_metric_reason(
    tmp_path: Path,
) -> None:
    """后端异常原因应进入结构化元数据，指标标签不得携带用户问题。"""
    from backend.app.observability import metrics
    from backend.app.services.query.document_retrieval import (
        DocumentRetriever,
    )

    question = "暴雨唯一用户问题-98765"
    chunk = _sample_chunk(content="暴雨过程造成城市内涝。")
    store = _FailingBackendStore(chunk)
    retriever = DocumentRetriever(
        embedding_client=_AvailableEmbeddingClient(
            expected_dimension=2,
            embedding=[1.0, 0.0],
        ),
        document_store=store,
        standard_case_store=_StandardCaseStore(),
    )

    metrics.reset_metrics()
    mode, hits = retriever.retrieve(question, top_k=5, top_n=3)

    diagnostics = getattr(
        retriever,
        "last_retrieval_diagnostics",
        None,
    )
    assert mode == "lexical/json"
    assert diagnostics is not None
    assert diagnostics["category"] == "backend_error"
    assert diagnostics["backend"] == "chroma"
    assert "模拟选定后端服务不可达" in diagnostics["reason"]
    assert diagnostics["exception_type"] == "RuntimeError"
    assert hits[0].backend == "json"
    assert hits[0].retrieval_mode == "lexical/json"
    assert hits[0].metadata["retrieval_fallback"] == diagnostics
    rendered_metrics = metrics.render_prometheus()
    assert (
        'vector_fallbacks_total{backend="chroma",reason="backend_error"} 1'
        in rendered_metrics
    )
    assert "唯一用户问题-98765" not in rendered_metrics


def test_document_retriever_context_case_path_keeps_order_and_marks_source(
    tmp_path: Path,
) -> None:
    """指定个例路径应保持原文顺序，并标记真实词面来源和追踪字段。"""
    from backend.app.observability.context import bind_request_context
    from backend.app.services.query.document_retrieval import DocumentRetriever
    from backend.app.services.vector.json_document_store import (
        JsonDocumentVectorStore,
    )

    first = _sample_chunk(chunk_id="chunk-001")
    second = DocumentChunk(
        source_pdf="case-2025-07.pdf",
        chunk_id="chunk-002",
        chunk_no=2,
        content="权威 JSON 正文：强对流造成短时大风。",
        file_path="resource/case-2025-07.pdf",
        embedding=[0.0, 1.0],
    )
    class_store = _StandardCaseStore()
    class_store.cases[0].source_chunk_ids = ["chunk-002", "chunk-001"]
    authority_path = tmp_path / "context-chunks.json"
    _seed_authority(authority_path, [second, first])
    store = JsonDocumentVectorStore(
        authority_path=authority_path,
        index_path=tmp_path / "context-vector-index.json",
        collection_name="weather_document_chunks",
        schema_version="v1",
    )
    store.rebuild()
    retriever = DocumentRetriever(
        embedding_client=_UnavailableEmbeddingClient(),
        document_store=store,
        standard_case_store=class_store,
    )

    with bind_request_context(
        request_id="request-context",
        trace_id="b" * 32,
    ):
        mode, hits = retriever.retrieve(
            "暴雨过程",
            top_k=5,
            top_n=3,
            context_case_ids=["case-001"],
        )

    assert mode == "document_context_cases"
    assert [hit.chunk.chunk_id for hit in hits] == ["chunk-001", "chunk-002"]
    assert all(hit.backend == "json" for hit in hits)
    assert all(hit.retrieval_mode == "lexical/context_cases" for hit in hits)
    assert all(hit.request_id == "request-context" for hit in hits)
    assert all(hit.trace_id == "b" * 32 for hit in hits)
    store.close()


if __name__ == "__main__":
    unittest.main()
