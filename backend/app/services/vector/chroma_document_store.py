"""只保存向量索引、正文回读权威 JSON 的 Chroma 文档适配器。"""
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from backend.app.models import DocumentChunk
from backend.app.services.vector.contracts import (
    CollectionInfo,
    VectorHit,
    VectorQuery,
)
from backend.app.services.vector.json_document_store import (
    build_lexical_hits,
    chunk_metadata,
    filter_matches,
    merge_vector_query,
    read_authority_chunks,
)


class ChromaDocumentVectorStore:
    """把 Chroma 作为纯向量索引，权威正文永远从 chunk JSON 获取。"""

    backend = "chroma"

    def __init__(
        self,
        *,
        authority_path: Path,
        persist_dir: Path,
        collection_name: str,
        schema_version: str = "v1",
        embedding_client: Any | None = None,
    ) -> None:
        """记录权威正文路径和 Chroma 持久化目录。"""
        self.authority_path = Path(authority_path)
        self.persist_dir = Path(persist_dir)
        self.collection_name = str(collection_name)
        self.schema_version = str(schema_version)
        self.embedding_client = embedding_client
        self._client = None
        self._collection = None
        self._collection_lock = Lock()
        self._closed = False

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """只写入 Chroma 向量和过滤元数据，不写入正文文档。"""
        self._ensure_open()
        indexed_chunks = [
            chunk for chunk in chunks if chunk.embedding and chunk.chunk_id
        ]
        if not indexed_chunks:
            return
        collection = self._get_collection()
        payload = {
            "ids": [chunk.chunk_id for chunk in indexed_chunks],
            "embeddings": [
                [float(value) for value in (chunk.embedding or [])]
                for chunk in indexed_chunks
            ],
            "metadatas": [chunk_metadata(chunk) for chunk in indexed_chunks],
        }
        try:
            collection.upsert(**payload)
        except Exception as exc:
            if "dimension" not in str(exc).lower():
                raise
            collection = self._reset_collection()
            collection.upsert(**payload)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """幂等删除 Chroma 索引记录，不修改权威正文。"""
        self._ensure_open()
        normalized = [
            str(chunk_id)
            for chunk_id in dict.fromkeys(chunk_ids)
            if str(chunk_id)
        ]
        if not normalized:
            return
        self._get_collection().delete(ids=normalized)

    def query(
        self,
        embedding: list[float],
        top_k: int,
        query: VectorQuery | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """查询真实 Chroma 距离；无向量时使用权威 JSON 的诚实词面模式。"""
        self._ensure_open()
        resolved = merge_vector_query(
            embedding=embedding,
            top_k=top_k,
            query=query,
            filters=filters,
        )
        authority = {
            chunk.chunk_id: chunk for chunk in read_authority_chunks(self.authority_path)
        }
        authority_ids = set(authority)
        if not resolved.embedding:
            return build_lexical_hits(
                chunks=list(authority.values()),
                allowed_ids=authority_ids,
                query=resolved,
                collection_name=self.collection_name,
            )
        indexed_ids = set(self._indexed_chunk_ids())
        if resolved.top_k <= 0 or not indexed_ids:
            return []

        collection = self._get_collection()
        request: dict[str, Any] = {
            "query_embeddings": [resolved.embedding],
            "n_results": min(resolved.top_k, max(1, collection.count())),
            "include": ["distances"],
        }
        chroma_where = _chroma_where(resolved.filters)
        if chroma_where:
            request["where"] = chroma_where
        result = collection.query(**request)
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[VectorHit] = []
        for chunk_id, distance in zip(ids, distances):
            chunk = authority.get(str(chunk_id))
            if chunk is None:
                continue
            metadata = chunk_metadata(chunk)
            if not filter_matches(metadata, resolved.filters):
                continue
            score = (
                float(1.0 - float(distance))
                if distance is not None
                else 0.0
            )
            hits.append(
                VectorHit(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    raw_score=score,
                    fused_score=score,
                    metadata={
                        **metadata,
                        "applied_filters": dict(resolved.filters),
                    },
                    backend=self.backend,
                    collection=self.collection_name,
                    retrieval_mode="vector/chroma",
                    request_id=resolved.request_id,
                    trace_id=resolved.trace_id,
                )
            )
        return hits[: resolved.top_k]

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """从权威 chunk JSON 读取单个正文块。"""
        self._ensure_open()
        normalized = str(chunk_id)
        return next(
            (
                chunk
                for chunk in read_authority_chunks(self.authority_path)
                if chunk.chunk_id == normalized
            ),
            None,
        )

    def list_chunks(self) -> list[DocumentChunk]:
        """从权威 chunk JSON 列出全部正文块。"""
        self._ensure_open()
        return read_authority_chunks(self.authority_path)

    def list_chunk_ids(self) -> list[str]:
        """列出权威 chunk JSON 中的全部稳定标识。"""
        self._ensure_open()
        return sorted(
            {
                chunk.chunk_id
                for chunk in read_authority_chunks(self.authority_path)
            }
        )

    def collection_info(self) -> CollectionInfo:
        """返回真实 Chroma 集合数量、维度和 schema 版本。"""
        if self._closed:
            return self._closed_info()
        try:
            collection = self._get_collection()
            count = int(collection.count())
            dimension = None
            if count:
                peek = collection.get(limit=1, include=["embeddings"])
                embeddings = peek.get("embeddings")
                if embeddings is not None and len(embeddings):
                    dimension = len(embeddings[0])
            return CollectionInfo(
                backend=self.backend,
                collection=self.collection_name,
                count=count,
                dimension=dimension,
                schema_version=self.schema_version,
                health_status="healthy",
                detail="Chroma 纯向量索引，正文来自权威 chunk JSON",
            )
        except Exception as exc:
            return CollectionInfo(
                backend=self.backend,
                collection=self.collection_name,
                count=0,
                dimension=None,
                schema_version=self.schema_version,
                health_status="unavailable",
                detail=f"Chroma 不可用：{exc}",
            )

    def rebuild(self) -> None:
        """用权威 JSON 中已有向量，或沿用旧索引向量，重建 Chroma 集合。"""
        self._ensure_open()
        previous_embeddings: dict[str, list[float]] = {}
        try:
            collection = self._get_collection()
            if collection.count():
                existing = collection.get(include=["embeddings"])
                ids = existing.get("ids") or []
                embeddings = existing.get("embeddings")
                if embeddings is not None:
                    for chunk_id, embedding in zip(ids, embeddings):
                        if embedding is not None:
                            previous_embeddings[str(chunk_id)] = [
                                float(value) for value in embedding
                            ]
        except Exception:
            previous_embeddings = {}

        rebuilt_chunks: list[DocumentChunk] = []
        for chunk in read_authority_chunks(self.authority_path):
            embedding = [float(value) for value in (chunk.embedding or [])]
            if not embedding:
                embedding = previous_embeddings.get(chunk.chunk_id, [])
            if embedding:
                rebuilt_chunks.append(
                    DocumentChunk(
                        source_pdf=chunk.source_pdf,
                        chunk_id=chunk.chunk_id,
                        chunk_no=chunk.chunk_no,
                        content=chunk.content,
                        file_path=chunk.file_path,
                        embedding=embedding,
                    )
                )
        self._reset_collection()
        if rebuilt_chunks:
            self.upsert_chunks(rebuilt_chunks)

    def health(self) -> dict[str, Any]:
        """返回可重复调用的 Chroma 健康状态。"""
        if self._closed:
            info = self._closed_info().to_dict()
        else:
            info = self.collection_info().to_dict()
        return {**info, "status": info["health_status"]}

    def close(self) -> None:
        """幂等释放 Chroma 客户端引用。"""
        if self._closed:
            return
        self._closed = True
        with self._collection_lock:
            self._collection = None
            self._client = None

    def _get_collection(self):
        """延迟创建 Chroma 客户端和余弦集合。"""
        self._ensure_open()
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("缺少 chromadb 依赖") from exc
        with self._collection_lock:
            if self._collection is None:
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(self.persist_dir))
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "schema_version": self.schema_version,
                    },
                )
            return self._collection

    def _indexed_chunk_ids(self) -> list[str]:
        """只列出 Chroma 中真正存在向量记录的稳定标识。"""
        self._ensure_open()
        result = self._get_collection().get(include=[])
        return sorted(str(item) for item in (result.get("ids") or []))

    def _reset_collection(self):
        """删除并重新创建当前 Chroma 集合。"""
        self._ensure_open()
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("缺少 chromadb 依赖") from exc
        with self._collection_lock:
            client = self._client or chromadb.PersistentClient(
                path=str(self.persist_dir)
            )
            try:
                client.delete_collection(name=self.collection_name)
            except Exception:
                pass
            self._client = client
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "schema_version": self.schema_version,
                },
            )
            return self._collection

    def _closed_info(self) -> CollectionInfo:
        """构造关闭状态集合描述。"""
        return CollectionInfo(
            backend=self.backend,
            collection=self.collection_name,
            count=0,
            dimension=None,
            schema_version=self.schema_version,
            health_status="closed",
            detail="Chroma 文档向量适配器已关闭",
        )

    def _ensure_open(self) -> None:
        """阻止关闭后的 Chroma 资源继续使用。"""
        if self._closed:
            raise RuntimeError("Chroma document vector store is closed")


def _chroma_where(filters: dict[str, Any]) -> dict[str, Any] | None:
    """把稳定精确过滤转换为 Chroma where 条件。"""
    clauses = []
    for key, expected in filters.items():
        if isinstance(expected, (str, int, float, bool)):
            clauses.append({key: expected})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
