"""
文档片段向量存储模块
提供 JSON 元数据备份和 ChromaDB 向量存储，支持文档 chunk 的向量索引、
相似度检索、增量更新和全量重建操作。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from threading import Lock

from backend.app.models import DocumentChunk, DocumentRetrievalHit


class JsonDocumentChunkStore:
    """
    文档片段 JSON 元数据备份存储
    提供 DocumentChunk 的增删改查和余弦相似度检索功能，作为 ChromaDB 的降级方案。
    """

    def __init__(self, storage_path: Path):
        """
        初始化 JSON 文档片段存储

        Args:
            storage_path: JSON 备份文件路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_lock = Lock()
        self._cached_signature: tuple[int, int] | None = None
        self._cached_chunks: list[DocumentChunk] = []
        self._cached_by_id: dict[str, DocumentChunk] = {}

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """
        增量更新文档片段

        Args:
            chunks: DocumentChunk 对象列表
        """
        existing = {chunk.chunk_id: chunk for chunk in self._load_chunks()}
        for chunk in chunks:
            existing[chunk.chunk_id] = chunk
        self._save_chunks(list(existing.values()))

    def replace_chunks(self, chunks: list[DocumentChunk]) -> None:
        """
        全量替换文档片段（重建索引）

        Args:
            chunks: DocumentChunk 对象列表
        """
        self._save_chunks(chunks)

    def list_chunk_ids(self) -> list[str]:
        """列出 JSON 备份中保存的所有 chunk key。"""
        return [chunk.chunk_id for chunk in self._load_chunks()]

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """根据 chunk_id 返回单个文档 chunk，找不到时返回 None。"""
        self._load_chunks()
        with self._cache_lock:
            return self._cached_by_id.get(chunk_id)

    def list_chunks(self) -> list[DocumentChunk]:
        """返回所有文档 chunk。"""
        return self._load_chunks()

    def query(self, embedding: list[float], top_k: int) -> list[DocumentRetrievalHit]:
        """使用 JSON 备份中的向量执行余弦相似度查询。"""
        hits = []
        for chunk in self._load_chunks():
            if not chunk.embedding:
                continue
            hits.append(
                DocumentRetrievalHit(
                    chunk=chunk,
                    score=self._cosine_similarity(embedding, chunk.embedding),
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def _load_chunks(self) -> list[DocumentChunk]:
        """从 JSON 文件读取所有文档 chunk。"""
        if not self.storage_path.exists():
            return []
        stat = self.storage_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        with self._cache_lock:
            if signature != self._cached_signature:
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
                self._cached_chunks = [DocumentChunk.from_dict(item) for item in data]
                self._cached_by_id = {item.chunk_id: item for item in self._cached_chunks}
                self._cached_signature = signature
            return list(self._cached_chunks)

    def _save_chunks(self, chunks: list[DocumentChunk]) -> None:
        """将文档 chunk 按 key 排序后持久化到 JSON 文件。"""
        data = [chunk.to_dict() for chunk in sorted(chunks, key=lambda item: item.chunk_id)]
        self.storage_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with self._cache_lock:
            self._cached_signature = None
            self._cached_chunks = []
            self._cached_by_id = {}

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        """计算两个向量的余弦相似度。"""
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)


class ChromaDocumentChunkStore:
    """
    文档片段 ChromaDB 向量存储
    独立的文档 chunk 向量索引，不与原案例向量库混用，支持向量检索、
    增量更新和全量重建，集成 JSON 元数据备份。
    """

    def __init__(self, persist_dir: Path, collection_name: str):
        """准备新文档向量库目录和 JSON 备份存储。"""
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._json_fallback = JsonDocumentChunkStore(self.persist_dir / "chunks.json")
        self._client = None
        self._collection = None
        self._collection_lock = Lock()

    def _get_collection(self):
        """获取文档 chunk 的 ChromaDB 集合，未安装 chromadb 时返回 None。"""
        try:
            import chromadb
        except ImportError:
            return None
        with self._collection_lock:
            if self._collection is None:
                self._client = chromadb.PersistentClient(path=str(self.persist_dir))
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            return self._collection

    def _reset_collection(self):
        """删除并重建文档 chunk 的 ChromaDB 集合。"""
        try:
            import chromadb
        except ImportError:
            return None
        with self._collection_lock:
            client = self._client or chromadb.PersistentClient(path=str(self.persist_dir))
            try:
                client.delete_collection(name=self.collection_name)
            except Exception:
                pass
            self._client = client
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return self._collection

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """在 JSON 备份和 ChromaDB 中插入或更新文档 chunk。"""
        collection = self._get_collection()
        self._json_fallback.upsert_chunks(chunks)
        if collection is None or not chunks:
            return
        payload = self._build_upsert_payload(chunks)
        try:
            collection.upsert(**payload)
        except Exception as exc:
            if "dimension" not in str(exc).lower():
                raise
            collection = self._reset_collection()
            if collection is not None:
                collection.upsert(**payload)

    def replace_chunks(self, chunks: list[DocumentChunk]) -> None:
        """完整重建新文档向量库，适合扫描目录后批量生成 chunk。"""
        self._json_fallback.replace_chunks(chunks)
        collection = self._reset_collection()
        if collection is None or not chunks:
            return
        collection.upsert(**self._build_upsert_payload(chunks))

    def collection_info(self) -> dict:
        """返回文档 chunk 集合的数量、维度和当前存储来源。"""
        collection = self._get_collection()
        if collection is None:
            return {
                "count": len(self._json_fallback.list_chunk_ids()),
                "dimension": None,
                "source": "json_fallback",
            }
        count = collection.count()
        dimension = None
        if count:
            peek = collection.get(limit=1, include=["embeddings"])
            embeddings = peek.get("embeddings")
            if embeddings is not None and len(embeddings):
                dimension = len(embeddings[0])
        return {
            "count": count,
            "dimension": dimension,
            "source": "chroma",
        }

    def list_chunk_ids(self) -> list[str]:
        """列出新文档向量库中的所有 chunk key。"""
        collection = self._get_collection()
        if collection is None:
            return self._json_fallback.list_chunk_ids()
        result = collection.get(include=[])
        ids = result.get("ids") or []
        if ids:
            return sorted(ids)
        return self._json_fallback.list_chunk_ids()

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """从 JSON 备份中读取完整 chunk 元数据和正文。"""
        return self._json_fallback.get_chunk(chunk_id)

    def get_chunks(self, chunk_ids: list[str]) -> list[DocumentChunk]:
        """按传入顺序批量读取正文块，供多维检索复用同一本地知识库。"""
        chunks = []
        for chunk_id in dict.fromkeys(str(item) for item in chunk_ids or [] if str(item)):
            chunk = self.get_chunk(chunk_id)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def list_chunks(self) -> list[DocumentChunk]:
        """从 JSON 备份中读取完整 chunk 数据。"""
        return self._json_fallback.list_chunks()

    def query(self, embedding: list[float], top_k: int) -> list[DocumentRetrievalHit]:
        """首先查询 ChromaDB，失败时回退到 JSON 余弦相似度查询。"""
        collection = self._get_collection()
        if collection is None:
            return self._json_fallback.query(embedding, top_k)
        result = collection.query(query_embeddings=[embedding], n_results=top_k)
        hits: list[DocumentRetrievalHit] = []
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for chunk_id, distance in zip(ids, distances):
            chunk = self.get_chunk(chunk_id)
            if chunk is None:
                continue
            score = 1 - float(distance) if distance is not None else 0.0
            hits.append(DocumentRetrievalHit(chunk=chunk, score=score))
        if hits:
            return hits
        return self._json_fallback.query(embedding, top_k)

    def _build_upsert_payload(self, chunks: list[DocumentChunk]) -> dict:
        """构建 ChromaDB upsert 所需的 ids、documents、metadatas 和 embeddings。"""
        return {
            "ids": [chunk.chunk_id for chunk in chunks],
            "documents": [chunk.content for chunk in chunks],
            "metadatas": [
                {
                    "source_pdf": chunk.source_pdf,
                    "chunk_no": chunk.chunk_no,
                    "file_path": chunk.file_path or "",
                    "text_length": len(chunk.content),
                }
                for chunk in chunks
            ],
            "embeddings": [chunk.embedding or [] for chunk in chunks],
        }
