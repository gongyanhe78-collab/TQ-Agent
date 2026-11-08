"""兼容既有调用的文档片段权威存储与统一向量适配层。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.models import DocumentChunk, DocumentRetrievalHit
from backend.app.services.vector.chroma_document_store import (
    ChromaDocumentVectorStore,
)
from backend.app.services.vector.contracts import VectorHit
from backend.app.services.vector.json_document_store import (
    JsonDocumentVectorStore,
    read_authority_chunks,
)

_DEFAULT_COLLECTION_NAME = "weather_document_chunks"
_DEFAULT_SCHEMA_VERSION = "v1"


def _write_authority_chunks(path: Path, chunks: list[DocumentChunk]) -> None:
    """按稳定顺序写入权威 chunk JSON，正文不再由向量索引承担。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        chunk.to_dict()
        for chunk in sorted(
            chunks,
            key=lambda item: (
                item.source_pdf,
                item.chunk_no,
                item.chunk_id,
            ),
        )
    ]
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _merge_authority_chunks(
    path: Path,
    chunks: list[DocumentChunk],
) -> list[DocumentChunk]:
    """把增量块合并进权威 JSON，并返回合并后的完整正文集合。"""
    existing = {
        chunk.chunk_id: chunk for chunk in read_authority_chunks(path)
    }
    for chunk in chunks:
        chunk_id = str(chunk.chunk_id or "").strip()
        if chunk_id:
            existing[chunk_id] = chunk
    merged = list(existing.values())
    _write_authority_chunks(path, merged)
    return merged


def _legacy_hits(hits: list[VectorHit]) -> list[DocumentRetrievalHit]:
    """把统一命中投影为旧检索调用方消费的正文与得分形状。"""
    return [
        DocumentRetrievalHit(chunk=hit.chunk, score=float(hit.score))
        for hit in hits
    ]


class JsonDocumentChunkStore:
    """兼容旧类名的 JSON 权威正文与轻量向量索引存储。"""

    def __init__(self, storage_path: Path):
        """以旧 chunks.json 路径建立正文权威文件和独立索引副本。"""
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._vector_store = JsonDocumentVectorStore(
            authority_path=self.storage_path,
            index_path=self.storage_path.parent / "document_vector_index.json",
            collection_name=_DEFAULT_COLLECTION_NAME,
            schema_version=_DEFAULT_SCHEMA_VERSION,
        )

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """先更新权威正文，再增量维护仅含向量和元数据的 JSON 索引。"""
        _merge_authority_chunks(self.storage_path, chunks)
        self._vector_store.upsert_chunks(chunks)

    def replace_chunks(self, chunks: list[DocumentChunk]) -> None:
        """全量替换权威正文，并按新正文和既有向量重建 JSON 索引。"""
        _write_authority_chunks(self.storage_path, chunks)
        self._vector_store.rebuild()

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """只删除 JSON 索引记录，权威正文保持不变。"""
        self._vector_store.delete_chunks(chunk_ids)

    def query(
        self,
        embedding: list[float],
        top_k: int,
        query: Any = None,
        filters: dict[str, Any] | None = None,
    ) -> list[DocumentRetrievalHit]:
        """执行统一 JSON 查询，并把结果投影为旧命中形状。"""
        return _legacy_hits(
            self._vector_store.query(
                embedding,
                top_k,
                query=query,
                filters=filters,
            )
        )

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """始终从权威 chunk JSON 读取单个正文块。"""
        return self._vector_store.get_chunk(chunk_id)

    def get_chunks(self, chunk_ids: list[str]) -> list[DocumentChunk]:
        """按传入顺序批量返回权威正文块。"""
        chunks = []
        for chunk_id in dict.fromkeys(
            str(item) for item in chunk_ids or [] if str(item)
        ):
            chunk = self.get_chunk(chunk_id)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def list_chunks(self) -> list[DocumentChunk]:
        """列出权威 chunk JSON 中的全部正文块。"""
        return self._vector_store.list_chunks()

    def list_chunk_ids(self) -> list[str]:
        """列出权威 chunk JSON 中的全部稳定标识。"""
        return self._vector_store.list_chunk_ids()

    def collection_info(self) -> dict[str, Any]:
        """返回旧接口可直接读取的 JSON 索引集合信息。"""
        return self._vector_store.collection_info().to_dict()

    def rebuild(self) -> None:
        """依据权威正文和既有向量副本重建 JSON 索引。"""
        self._vector_store.rebuild()

    def health(self) -> dict[str, Any]:
        """返回 JSON 索引的可重复健康状态。"""
        return self._vector_store.health()

    def close(self) -> None:
        """幂等关闭 JSON 索引。"""
        self._vector_store.close()


class ChromaDocumentChunkStore:
    """兼容旧类名的 Chroma 纯向量索引与 JSON 权威正文适配层。"""

    def __init__(self, persist_dir: Path, collection_name: str):
        """在旧持久化目录中建立 Chroma 索引和 chunks.json 权威文件。"""
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = str(collection_name)
        self.storage_path = self.persist_dir / "chunks.json"
        self._vector_store = ChromaDocumentVectorStore(
            authority_path=self.storage_path,
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
            schema_version=_DEFAULT_SCHEMA_VERSION,
        )

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """先更新权威正文，再只向 Chroma 写入向量和过滤元数据。"""
        _merge_authority_chunks(self.storage_path, chunks)
        self._vector_store.upsert_chunks(chunks)

    def replace_chunks(self, chunks: list[DocumentChunk]) -> None:
        """全量替换权威正文，并按新正文和既有向量重建 Chroma 集合。"""
        _write_authority_chunks(self.storage_path, chunks)
        self._vector_store.rebuild()

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """只删除 Chroma 索引记录，权威正文保持不变。"""
        self._vector_store.delete_chunks(chunk_ids)

    def query(
        self,
        embedding: list[float],
        top_k: int,
        query: Any = None,
        filters: dict[str, Any] | None = None,
    ) -> list[DocumentRetrievalHit]:
        """执行统一 Chroma 查询，并把结果投影为旧命中形状。"""
        return _legacy_hits(
            self._vector_store.query(
                embedding,
                top_k,
                query=query,
                filters=filters,
            )
        )

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """从权威 JSON 读取单个正文块，避免从 Chroma document 反推正文。"""
        return self._vector_store.get_chunk(chunk_id)

    def get_chunks(self, chunk_ids: list[str]) -> list[DocumentChunk]:
        """按传入顺序批量返回权威正文块。"""
        chunks = []
        for chunk_id in dict.fromkeys(
            str(item) for item in chunk_ids or [] if str(item)
        ):
            chunk = self.get_chunk(chunk_id)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def list_chunks(self) -> list[DocumentChunk]:
        """从权威 JSON 列出全部正文块。"""
        return self._vector_store.list_chunks()

    def list_chunk_ids(self) -> list[str]:
        """列出权威 chunk JSON 中的全部稳定标识。"""
        return self._vector_store.list_chunk_ids()

    def collection_info(self) -> dict[str, Any]:
        """返回旧接口可直接读取的 Chroma 集合信息。"""
        return self._vector_store.collection_info().to_dict()

    def rebuild(self) -> None:
        """依据权威正文和既有向量副本重建 Chroma 集合。"""
        self._vector_store.rebuild()

    def health(self) -> dict[str, Any]:
        """返回 Chroma 适配器的可重复健康状态。"""
        return self._vector_store.health()

    def close(self) -> None:
        """幂等关闭 Chroma 索引。"""
        self._vector_store.close()
