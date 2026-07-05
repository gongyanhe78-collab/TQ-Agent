from __future__ import annotations

import json
import math
from pathlib import Path

from backend.app.models import DocumentChunk, DocumentRetrievalHit


class JsonDocumentChunkStore:
    """文档 chunk 的 JSON 元数据备份存储。"""

    def __init__(self, storage_path: Path):
        """准备 JSON 备份文件路径，父目录不存在时自动创建。"""
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """按 chunk_id 插入或更新文档 chunk 元数据。"""
        existing = {chunk.chunk_id: chunk for chunk in self._load_chunks()}
        for chunk in chunks:
            existing[chunk.chunk_id] = chunk
        self._save_chunks(list(existing.values()))

    def replace_chunks(self, chunks: list[DocumentChunk]) -> None:
        """用传入的 chunk 列表完整替换 JSON 备份内容。"""
        self._save_chunks(chunks)

    def list_chunk_ids(self) -> list[str]:
        """列出 JSON 备份中保存的所有 chunk key。"""
        return [chunk.chunk_id for chunk in self._load_chunks()]

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """根据 chunk_id 返回单个文档 chunk，找不到时返回 None。"""
        for chunk in self._load_chunks():
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

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
        data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        return [DocumentChunk.from_dict(item) for item in data]

    def _save_chunks(self, chunks: list[DocumentChunk]) -> None:
        """将文档 chunk 按 key 排序后持久化到 JSON 文件。"""
        data = [chunk.to_dict() for chunk in sorted(chunks, key=lambda item: item.chunk_id)]
        self.storage_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        """计算两个向量的余弦相似度。"""
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)


class ChromaDocumentChunkStore:
    """独立的文档 chunk ChromaDB 存储，不与原个例向量库混用。"""

    def __init__(self, persist_dir: Path, collection_name: str):
        """准备新文档向量库目录和 JSON 备份存储。"""
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._json_fallback = JsonDocumentChunkStore(self.persist_dir / "chunks.json")

    def _get_collection(self):
        """获取文档 chunk 的 ChromaDB 集合，未安装 chromadb 时返回 None。"""
        try:
            import chromadb
        except ImportError:
            return None
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        return client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _reset_collection(self):
        """删除并重建文档 chunk 的 ChromaDB 集合。"""
        try:
            import chromadb
        except ImportError:
            return None
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        try:
            client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        return client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

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
