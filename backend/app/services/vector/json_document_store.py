"""基于权威 chunk JSON 的轻量文档向量索引适配器。"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from threading import Lock
from typing import Any

from backend.app.models import DocumentChunk
from backend.app.observability.context import current_context
from backend.app.services.vector.contracts import (
    CollectionInfo,
    VectorHit,
    VectorQuery,
)


_TOKEN_SEPARATOR = re.compile(
    r"[\s，。！？、；：,.!?;:（）()\-_/]+"
)
_CJK_RUN = re.compile(r"[\u3400-\u9fff]+")


def read_authority_chunks(path: Path) -> list[DocumentChunk]:
    """从权威 chunk JSON 读取正文；文件缺失时返回空列表。"""
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"权威 chunk JSON 必须是对象数组：{path}")
    chunks: list[DocumentChunk] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"权威 chunk JSON 包含非对象记录：{path}")
        chunk = DocumentChunk.from_dict(item)
        if chunk.chunk_id:
            chunks.append(chunk)
    return chunks


def chunk_metadata(chunk: DocumentChunk) -> dict[str, Any]:
    """提取可过滤的索引元数据，正文不进入索引。"""
    return {
        "source_pdf": chunk.source_pdf,
        "chunk_no": int(chunk.chunk_no),
        "file_path": chunk.file_path or "",
        "text_length": len(chunk.content),
    }


def merge_vector_query(
    *,
    embedding: list[float],
    top_k: int,
    query: VectorQuery | None,
    filters: dict[str, Any] | None,
) -> VectorQuery:
    """合并显式参数和当前请求上下文，保留调用方显式值。"""
    context = current_context()
    resolved = query or VectorQuery()
    merged_filters = dict(resolved.filters)
    if filters:
        merged_filters.update(filters)
    return VectorQuery(
        query_text=str(resolved.query_text or ""),
        embedding=[float(value) for value in (embedding or resolved.embedding)],
        top_k=max(0, int(top_k if top_k is not None else resolved.top_k)),
        filters=merged_filters,
        request_id=str(resolved.request_id or context.request_id),
        trace_id=str(resolved.trace_id or context.trace_id),
        user_id=str(resolved.user_id or context.user_id),
    )


def filter_matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    """按索引元数据执行稳定的精确或集合过滤。"""
    for key, expected in filters.items():
        actual = metadata.get(key)
        if isinstance(expected, dict):
            if "$in" in expected and actual not in expected["$in"]:
                return False
            if "$ne" in expected and actual == expected["$ne"]:
                return False
            if "$eq" in expected and actual != expected["$eq"]:
                return False
            continue
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
            continue
        if actual != expected:
            return False
    return True


def lexical_terms(text: str) -> list[str]:
    """切分查询词，并为连续中文补充稳定的二字片段。"""
    terms: list[str] = []
    for item in _TOKEN_SEPARATOR.split(str(text or "")):
        normalized = item.replace(" ", "")
        if not normalized:
            continue
        terms.append(normalized)
        for run in _CJK_RUN.findall(normalized):
            if len(run) < 2:
                continue
            terms.extend(
                run[index : index + 2]
                for index in range(len(run) - 1)
            )
    return list(dict.fromkeys(terms))


def lexical_score(query_text: str, content: str) -> float:
    """计算与现有检索器一致的轻量词面覆盖得分。"""
    normalized = str(content or "").replace(" ", "")
    return float(
        sum(
            max(1, len(token))
            for token in lexical_terms(query_text)
            if token and token in normalized
        )
    )


def build_lexical_hits(
    *,
    chunks: list[DocumentChunk],
    allowed_ids: set[str],
    query: VectorQuery,
    collection_name: str,
    backend: str = "json",
) -> list[VectorHit]:
    """从权威正文产生诚实的 JSON 词面命中，零命中时保持空列表。"""
    scored: list[tuple[DocumentChunk, float]] = []
    for chunk in chunks:
        if chunk.chunk_id not in allowed_ids:
            continue
        metadata = chunk_metadata(chunk)
        if not filter_matches(metadata, query.filters):
            continue
        score = lexical_score(query.query_text, chunk.content)
        if score <= 0:
            continue
        scored.append((chunk, score))
    scored.sort(
        key=lambda item: (
            -item[1],
            item[0].source_pdf,
            item[0].chunk_no,
            item[0].chunk_id,
        )
    )
    return [
        VectorHit(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            raw_score=score,
            fused_score=score,
            metadata={
                **chunk_metadata(chunk),
                "applied_filters": dict(query.filters),
            },
            backend=backend,
            collection=collection_name,
            retrieval_mode="lexical/json",
            request_id=query.request_id,
            trace_id=query.trace_id,
        )
        for chunk, score in scored[: query.top_k]
    ]


class JsonDocumentVectorStore:
    """使用 JSON 索引副本查询向量或词面结果，正文始终读取权威文件。"""

    backend = "json"

    def __init__(
        self,
        *,
        authority_path: Path,
        index_path: Path,
        collection_name: str,
        schema_version: str = "v1",
        embedding_client: Any | None = None,
    ) -> None:
        """记录权威正文路径和独立向量索引路径。"""
        self.authority_path = Path(authority_path)
        self.index_path = Path(index_path)
        self.collection_name = str(collection_name)
        self.schema_version = str(schema_version)
        self.embedding_client = embedding_client
        self._lock = Lock()
        self._closed = False

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """只更新 JSON 索引副本，绝不改写权威正文文件。"""
        self._ensure_open()
        with self._lock:
            index = self._load_index()
            items = dict(index.get("items") or {})
            for chunk in chunks:
                chunk_id = str(chunk.chunk_id or "").strip()
                if not chunk_id:
                    continue
                previous = dict(items.get(chunk_id) or {})
                embedding = [float(value) for value in (chunk.embedding or [])]
                if embedding:
                    previous["embedding"] = embedding
                previous["metadata"] = chunk_metadata(chunk)
                items[chunk_id] = previous
            self._write_index(items)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """幂等删除索引记录，保留权威正文。"""
        self._ensure_open()
        with self._lock:
            items = dict(self._load_index().get("items") or {})
            for chunk_id in chunk_ids:
                items.pop(str(chunk_id), None)
            self._write_index(items)

    def query(
        self,
        embedding: list[float],
        top_k: int,
        query: VectorQuery | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """执行余弦向量查询，缺少向量时只执行诚实的 JSON 词面查询。"""
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
        index = self._load_index()
        items = dict(index.get("items") or {})
        allowed_ids = set(authority)
        if not resolved.embedding:
            return build_lexical_hits(
                chunks=list(authority.values()),
                allowed_ids=allowed_ids,
                query=resolved,
                collection_name=self.collection_name,
            )

        scored: list[tuple[DocumentChunk, float]] = []
        for chunk_id, item in items.items():
            chunk = authority.get(chunk_id)
            if chunk is None:
                continue
            metadata = chunk_metadata(chunk)
            if not filter_matches(metadata, resolved.filters):
                continue
            indexed_embedding = [
                float(value) for value in (item.get("embedding") or [])
            ]
            if not indexed_embedding:
                continue
            scored.append(
                (
                    chunk,
                    _cosine_similarity(resolved.embedding, indexed_embedding),
                )
            )
        scored.sort(
            key=lambda item: (
                -item[1],
                item[0].source_pdf,
                item[0].chunk_no,
                item[0].chunk_id,
            )
        )
        return [
            VectorHit(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                raw_score=score,
                fused_score=score,
                metadata={
                    **chunk_metadata(chunk),
                    "applied_filters": dict(resolved.filters),
                },
                backend=self.backend,
                collection=self.collection_name,
                retrieval_mode="vector/json",
                request_id=resolved.request_id,
                trace_id=resolved.trace_id,
            )
            for chunk, score in scored[: resolved.top_k]
        ]

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """始终从权威 chunk JSON 读取单个正文块。"""
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
        """始终从权威 chunk JSON 列出正文块。"""
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
        """返回 JSON 索引副本的真实数量、维度和生命周期状态。"""
        index = self._load_index()
        items = dict(index.get("items") or {})
        dimension = None
        indexed_count = 0
        for item in items.values():
            embedding = item.get("embedding") or []
            if embedding:
                indexed_count += 1
                if dimension is None:
                    dimension = len(embedding)
        return CollectionInfo(
            backend=self.backend,
            collection=self.collection_name,
            count=indexed_count,
            dimension=dimension,
            schema_version=self.schema_version,
            health_status="closed" if self._closed else "healthy",
            detail="JSON 轻量向量索引，正文来自权威 chunk JSON",
        )

    def rebuild(self) -> None:
        """以权威正文中的向量为优先来源重建 JSON 索引副本。"""
        self._ensure_open()
        with self._lock:
            previous = dict(self._load_index().get("items") or {})
            items: dict[str, dict[str, Any]] = {}
            for chunk in read_authority_chunks(self.authority_path):
                embedding = [float(value) for value in (chunk.embedding or [])]
                if not embedding:
                    embedding = [
                        float(value)
                        for value in (
                            (previous.get(chunk.chunk_id) or {}).get("embedding") or []
                        )
                    ]
                item = {"metadata": chunk_metadata(chunk)}
                if embedding:
                    item["embedding"] = embedding
                items[chunk.chunk_id] = item
            self._write_index(items)

    def health(self) -> dict[str, Any]:
        """返回可重复调用的 JSON 索引健康状态。"""
        info = self.collection_info().to_dict()
        return {
            **info,
            "status": info["health_status"],
            "detail": (
                "JSON 向量索引已关闭"
                if self._closed
                else info["detail"]
            ),
        }

    def close(self) -> None:
        """幂等关闭无外部连接的 JSON 索引。"""
        self._closed = True

    def _load_index(self) -> dict[str, Any]:
        """读取索引文件；缺失时返回空索引。"""
        if not self.index_path.is_file():
            return {"items": {}}
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"向量索引必须是 JSON 对象：{self.index_path}")
        items = payload.get("items")
        if items is not None and not isinstance(items, dict):
            raise ValueError(f"向量索引 items 必须是对象：{self.index_path}")
        return payload

    def _write_index(self, items: dict[str, Any]) -> None:
        """原子写入不含正文的稳定 JSON 索引副本。"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        dimension = None
        for item in items.values():
            embedding = item.get("embedding") or []
            if embedding:
                dimension = len(embedding)
                break
        payload = {
            "schema_version": self.schema_version,
            "collection": self.collection_name,
            "dimension": dimension,
            "items": dict(sorted(items.items())),
        }
        temporary_path = self.index_path.with_suffix(
            f"{self.index_path.suffix}.tmp"
        )
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self.index_path)

    def _ensure_open(self) -> None:
        """阻止关闭后的索引继续执行持久化操作。"""
        if self._closed:
            raise RuntimeError("JSON document vector store is closed")


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个有效等维向量的余弦相似度。"""
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)
