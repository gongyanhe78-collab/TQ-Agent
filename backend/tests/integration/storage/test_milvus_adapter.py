"""Milvus 文档与记忆适配器的真实混合检索集成测试。"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

from backend.app.models import DocumentChunk


def _load_module(module_name: str):
    """导入待实现模块，并把缺失实现明确报告为测试失败。"""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        pytest.fail(f"待实现模块缺失：{module_name}：{exc}")


def _test_uri() -> str:
    """返回 Milvus 测试地址，默认本机 19530。"""
    return os.getenv("MILVUS_TEST_URI", "http://127.0.0.1:19530").strip()


def _milvus_available(uri: str) -> bool:
    """探测 Milvus 是否可连接；不可用时集成用例整体跳过。"""
    try:
        import pymilvus
    except ImportError:
        return False
    try:
        client = pymilvus.MilvusClient(uri=uri, timeout=2)
        return bool(client.list_collections() is not None)
    except Exception:
        return False


URI = _test_uri()
COLLECTION = "codex_test_document_chunks_v1"
MEMORY_COLLECTION = "codex_test_user_memories_v1"


def _sample_chunk(
    *,
    chunk_id: str = "chunk-001",
    content: str = "2025年7月山西出现暴雨天气过程。",
    source_pdf: str = "case-2025-07.pdf",
    embedding: list[float] | None = None,
) -> DocumentChunk:
    """构造 Milvus 文档适配器使用的文档块。"""
    return DocumentChunk(
        source_pdf=source_pdf,
        chunk_id=chunk_id,
        chunk_no=1,
        content=content,
        file_path=f"resource/{source_pdf}",
        embedding=embedding if embedding is not None else [1.0, 0.0, 0.0, 0.0],
    )


def _seed_authority(path: Path, chunks: list[DocumentChunk]) -> None:
    """写入权威 chunk JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@pytest.mark.skipif(not _milvus_available(URI), reason="本机 Milvus 服务不可用，跳过真实适配器验证")
class TestMilvusDocumentAdapter:
    """文档适配器需要真实 Milvus 时执行。"""

    def test_milvus_document_store_roundtrip_and_case_filter(self, tmp_path: Path) -> None:
        """写入、过滤查询、正文回读和关闭必须符合文档契约。"""
        module = _load_module(
            "backend.app.services.vector.milvus_document_store"
        )
        authority_path = tmp_path / "chunks.json"
        first = _sample_chunk(
            chunk_id="chunk-hail",
            content="山西冰雹灾害造成农业损失",
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        second = _sample_chunk(
            chunk_id="chunk-rain",
            content="山西暴雨引发山洪和城乡积涝",
            source_pdf="case-rain.pdf",
            embedding=[0.0, 1.0, 0.0, 0.0],
        )
        _seed_authority(authority_path, [first, second])
        store = module.MilvusDocumentStore(
            uri=URI,
            collection_name=COLLECTION,
            authority_path=authority_path,
            dimension=4,
            dense_top_k=10,
            sparse_top_k=10,
            rrf_k=60,
        )
        try:
            store.rebuild()
            assert set(store.list_chunk_ids()) >= {"chunk-hail", "chunk-rain"}
            assert store.get_chunk("chunk-hail").content == first.content
            hits = store.query(
                [1.0, 0.0, 0.0, 0.0],
                top_k=5,
                filters={"case_ids": ["case-2025-07.pdf"]},
            )
            assert all(hit.chunk_id != "chunk-rain" for hit in hits)
            info = store.collection_info()
            assert info.backend == "milvus"
        finally:
            store.close()
            store.close()

    def test_milvus_document_store_rebuild_matches_authority_ids(
        self, tmp_path: Path
    ) -> None:
        """重建后 chunk ID 集合必须与权威 JSON 一致。"""
        module = _load_module(
            "backend.app.services.vector.milvus_document_store"
        )
        authority_path = tmp_path / "chunks.json"
        chunks = [
            _sample_chunk(
                chunk_id=f"chunk-{index:03d}",
                content=f"权威第 {index} 段：暴雨过程。",
                embedding=[float(value) for value in [1.0, 0.0, 0.0, float(index)]],
            )
            for index in range(1, 6)
        ]
        _seed_authority(authority_path, chunks)
        store = module.MilvusDocumentStore(
            uri=URI,
            collection_name=COLLECTION,
            authority_path=authority_path,
            dimension=4,
        )
        try:
            store.rebuild()
            assert set(store.list_chunk_ids()) == {item.chunk_id for item in chunks}
        finally:
            store.close()


@pytest.mark.skipif(not _milvus_available(URI), reason="本机 Milvus 服务不可用，跳过真实记忆适配器验证")
class TestMilvusMemoryAdapter:
    """记忆适配器需要真实 Milvus 时执行。"""

    def test_memory_store_user_scoped_upsert_query_and_delete(
        self, tmp_path: Path
    ) -> None:
        """记忆必须按 user_id 隔离检索，删除后不再返回。"""
        module = _load_module(
            "backend.app.services.vector.milvus_memory_store"
        )
        store = module.MilvusMemoryStore(
            uri=URI,
            collection_name=MEMORY_COLLECTION,
            dimension=4,
        )
        try:
            store.rebuild(user_id=None)
            store.upsert_memories(
                [
                    {
                        "memory_id": "memory-user-a-1",
                        "user_id": "user-a",
                        "session_id": "session-a",
                        "memory_type": "preference",
                        "sensitivity": "normal",
                        "status": "active",
                        "summary": "偏好关注暴雨灾害",
                        "dense_vector": [1.0, 0.0, 0.0, 0.0],
                        "sparse_vector": {"1": 1.0},
                        "source_message_id": 1,
                        "version": 1,
                        "created_at": 1700000000,
                    }
                ]
            )
            hits = store.query_by_user(
                "user-a",
                [1.0, 0.0, 0.0, 0.0],
                top_k=5,
            )
            assert any(hit.chunk_id == "memory-user-a-1" for hit in hits)

            other_hits = store.query_by_user(
                "user-b",
                [1.0, 0.0, 0.0, 0.0],
                top_k=5,
            )
            assert all(
                hit.metadata.get("user_id") != "user-a" for hit in other_hits
            )

            store.delete_memories(["memory-user-a-1"], user_id="user-a")
            after = store.query_by_user(
                "user-a",
                [1.0, 0.0, 0.0, 0.0],
                top_k=5,
            )
            assert all(hit.chunk_id != "memory-user-a-1" for hit in after)
        finally:
            store.close()
            store.close()
