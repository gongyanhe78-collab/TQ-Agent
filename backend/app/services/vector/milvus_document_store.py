"""Milvus 文档向量适配器：只保存可重建索引，正文永远回读权威 JSON。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Lock
from typing import Any

from backend.app.models import DocumentChunk
from backend.app.services.vector.contracts import (
    CollectionInfo,
    VectorHit,
    VectorQuery,
    count_collection_rows,
)
from backend.app.services.vector.json_document_store import (
    chunk_metadata,
    merge_vector_query,
    read_authority_chunks,
)


def build_case_filter(target_case_ids: list[str]) -> str:
    """构造参数化个例过滤表达式，真实个例 ID 不进入表达式字符串。"""
    normalized = [
        str(case_id).strip()
        for case_id in (target_case_ids or [])
        if str(case_id).strip()
    ]
    if not normalized:
        return ""
    return "ARRAY_CONTAINS_ANY(case_ids, {target_case_ids})"


def build_case_filter_params(target_case_ids: list[str]) -> dict[str, Any]:
    """返回个例过滤表达式参数，只携带目标个例 ID。"""
    normalized = [
        str(case_id).strip()
        for case_id in (target_case_ids or [])
        if str(case_id).strip()
    ]
    return {"target_case_ids": normalized}


def _content_hash(content: str) -> str:
    """计算正文稳定哈希，用于索引与权威正文的一致性校验。"""
    return hashlib.sha256(str(content or "").encode("utf-8")).hexdigest()


def _chunk_text(chunk: DocumentChunk) -> str:
    """返回可写入 Milvus 中文分析器字段的正文，超长时按字段上限截断。"""
    return str(chunk.content or "")[:4096]


class MilvusDocumentStore:
    """把 Milvus 作为可重建的文档混合索引，正文从权威 chunk JSON 读取。"""

    backend = "milvus"

    def __init__(
        self,
        *,
        uri: str,
        collection_name: str,
        authority_path: Path,
        dimension: int,
        token: str = "",
        database: str = "default",
        schema_version: str = "v1",
        dense_top_k: int = 30,
        sparse_top_k: int = 30,
        rrf_k: int = 60,
        timeout_seconds: float = 10.0,
        standard_cases_path: Path | None = None,
        embedding_model: str = "",
        document_version: str = "",
    ) -> None:
        """记录 Milvus 连接、权威正文路径和固定检索参数。"""
        self.uri = str(uri)
        self.collection_name = str(collection_name)
        self.authority_path = Path(authority_path)
        self.dimension = max(1, int(dimension))
        self.token = str(token or "")
        self.database = str(database or "default")
        self.schema_version = str(schema_version or "v1")
        self.dense_top_k = max(1, int(dense_top_k))
        self.sparse_top_k = max(1, int(sparse_top_k))
        self.rrf_k = max(1, int(rrf_k))
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.standard_cases_path = (
            Path(standard_cases_path) if standard_cases_path else None
        )
        self.embedding_model = str(embedding_model or "")
        self.document_version = str(document_version or "")
        self._client: Any = None
        self._collection_lock = Lock()
        self._closed = False
        self._last_error: str = ""

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """只写入 Milvus 向量和过滤元数据，不写入权威正文。"""
        self._ensure_open()
        rows = [
            row
            for chunk in chunks
            if chunk.chunk_id and chunk.embedding
            for row in [self._chunk_row(chunk)]
            if row is not None
        ]
        if not rows:
            return
        client = self._ensure_collection()
        client.upsert(collection_name=self.collection_name, data=rows)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """幂等删除 Milvus 索引记录，不修改权威正文。"""
        self._ensure_open()
        normalized = [
            str(chunk_id)
            for chunk_id in dict.fromkeys(chunk_ids)
            if str(chunk_id)
        ]
        if not normalized:
            return
        client = self._ensure_collection()
        client.delete(
            collection_name=self.collection_name,
            filter="chunk_id in " + str(normalized),
        )

    def query(
        self,
        embedding: list[float],
        top_k: int,
        query: VectorQuery | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """执行稠密与 BM25 双路混合检索，并按权威 JSON 回读正文。"""
        self._ensure_open()
        resolved = merge_vector_query(
            embedding=embedding,
            top_k=top_k,
            query=query,
            filters=filters,
        )
        authority = {
            chunk.chunk_id: chunk
            for chunk in read_authority_chunks(self.authority_path)
        }
        authority_ids = set(authority)
        if not resolved.embedding:
            return []
        if resolved.top_k <= 0 or not authority_ids:
            return []
        client = self._ensure_collection()

        expr = ""
        expr_params: dict[str, Any] | None = None
        target_case_ids = list(resolved.filters.get("case_ids") or [])
        if target_case_ids:
            expr = build_case_filter(target_case_ids)
            expr_params = build_case_filter_params(target_case_ids)

        try:
            from pymilvus import AnnSearchRequest, RRFRanker

            requests = [
                AnnSearchRequest(
                    data=[resolved.embedding],
                    anns_field="dense_vector",
                    param={"metric_type": "COSINE"},
                    limit=self.dense_top_k,
                    expr=expr or None,
                    expr_params=expr_params,
                ),
                AnnSearchRequest(
                    data=[str(resolved.query_text or "")],
                    anns_field="sparse_vector",
                    param={"metric_type": "BM25"},
                    limit=self.sparse_top_k,
                    expr=expr or None,
                    expr_params=expr_params,
                ),
            ]
            results = client.hybrid_search(
                collection_name=self.collection_name,
                reqs=requests,
                ranker=RRFRanker(k=self.rrf_k),
                limit=resolved.top_k,
                output_fields=[
                    "chunk_id",
                    "case_ids",
                    "source_pdf",
                    "chunk_no",
                    "file_path",
                ],
            )
        except Exception:
            # 查询失败必须上抛，由固定降级链决定是否切换到 Chroma 或 JSON。
            raise

        hits: list[VectorHit] = []
        for row in (results or [[]])[0]:
            entity = dict(row.get("entity") or {})
            chunk_id = str(entity.get("chunk_id") or row.get("id") or "")
            if not chunk_id or chunk_id not in authority_ids:
                continue
            chunk = authority[chunk_id]
            metadata = chunk_metadata(chunk)
            metadata["case_ids"] = list(entity.get("case_ids") or [])
            metadata["applied_filters"] = dict(resolved.filters)
            distance = row.get("distance")
            score = float(distance) if distance is not None else 0.0
            hits.append(
                VectorHit(
                    chunk_id=chunk_id,
                    content=chunk.content,
                    raw_score=score,
                    fused_score=score,
                    metadata=metadata,
                    backend=self.backend,
                    collection=self.collection_name,
                    retrieval_mode="milvus/hybrid_rrf",
                    request_id=resolved.request_id,
                    trace_id=resolved.trace_id,
                )
            )
        return hits

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """从权威 chunk JSON 读取单个正文块。"""
        self._ensure_open()
        authority = {
            chunk.chunk_id: chunk
            for chunk in read_authority_chunks(self.authority_path)
        }
        return authority.get(str(chunk_id))

    def list_chunks(self) -> list[DocumentChunk]:
        """从权威 chunk JSON 列出正文块。"""
        self._ensure_open()
        return read_authority_chunks(self.authority_path)

    def list_chunk_ids(self) -> list[str]:
        """列出 Milvus 集合中真实存在的索引标识。"""
        self._ensure_open()
        client = self._ensure_collection()
        # Milvus 单次查询窗口上限为 16384，必须分页遍历；分页步长取 1000，
        # 集合规模超出窗口时也保持 offset+limit 不越过上限。
        page_size = 1000
        offset = 0
        chunk_ids: list[str] = []
        while True:
            rows = client.query(
                collection_name=self.collection_name,
                filter="chunk_id != ''",
                output_fields=["chunk_id"],
                limit=page_size,
                offset=offset,
            )
            chunk_ids.extend(
                str(item.get("chunk_id") or "")
                for item in rows
                if str(item.get("chunk_id") or "")
            )
            if len(rows) < page_size:
                break
            offset += page_size
        return sorted(chunk_ids)

    def collection_info(self) -> CollectionInfo:
        """返回当前集合的后端、数量、维度和健康状态。"""
        count = 0
        detail = ""
        health_status = "healthy"
        if self.dimension <= 0:
            return CollectionInfo(
                backend=self.backend,
                collection=self.collection_name,
                count=0,
                dimension=None,
                schema_version=self.schema_version,
                health_status="unavailable",
                detail="无法从权威数据推断嵌入维度，Milvus 集合不可用",
                metadata={
                    "uri": self.uri,
                    "dense_top_k": self.dense_top_k,
                    "sparse_top_k": self.sparse_top_k,
                    "rrf_k": self.rrf_k,
                },
            )
        if not self._closed:
            try:
                client = self._get_client()
                if client is not None and client.has_collection(
                    self.collection_name
                ):
                    count = count_collection_rows(
                        client,
                        self.collection_name,
                    )
            except Exception as exc:
                health_status = "unavailable"
                detail = f"Milvus 集合状态读取失败：{exc}"
                self._last_error = str(exc)
        return CollectionInfo(
            backend=self.backend,
            collection=self.collection_name,
            count=count,
            dimension=self.dimension,
            schema_version=self.schema_version,
            health_status=health_status,
            detail=detail,
            metadata={
                "uri": self.uri,
                "dense_top_k": self.dense_top_k,
                "sparse_top_k": self.sparse_top_k,
                "rrf_k": self.rrf_k,
            },
        )

    def rebuild(self) -> None:
        """按权威 JSON 重新构建整个 Milvus 集合。"""
        self._ensure_open()
        chunks = read_authority_chunks(self.authority_path)
        client = self._get_client()
        if client.has_collection(self.collection_name):
            client.drop_collection(self.collection_name)
        client = self._ensure_collection()
        rows = [
            row
            for chunk in chunks
            if chunk.chunk_id and chunk.embedding
            for row in [self._chunk_row(chunk)]
            if row is not None
        ]
        if rows:
            client.insert(collection_name=self.collection_name, data=rows)

    def health(self) -> dict[str, Any]:
        """返回真实健康状态，连接失败时报告不可用而不是抛出异常。"""
        info = self.collection_info().to_dict()
        return {**info, "status": info["health_status"]}

    def close(self) -> None:
        """幂等关闭 Milvus 客户端。"""
        if self._closed:
            return
        self._closed = True
        with self._collection_lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    def _chunk_row(self, chunk: DocumentChunk) -> dict[str, Any] | None:
        """构造 Milvus 插入行，正文只进入可重建的索引字段。"""
        if not chunk.chunk_id or not chunk.embedding:
            return None
        metadata = chunk_metadata(chunk)
        return {
            "chunk_id": str(chunk.chunk_id),
            "text": _chunk_text(chunk),
            "dense_vector": [float(value) for value in (chunk.embedding or [])],
            "case_ids": self._case_ids_for(chunk.chunk_id),
            "source_pdf": str(chunk.source_pdf or ""),
            "file_path": str(chunk.file_path or ""),
            "chunk_no": int(chunk.chunk_no or 0),
            "text_length": int(metadata.get("text_length") or 0),
            "content_hash": _content_hash(chunk.content),
            "document_version": self.document_version,
            "embedding_model": self.embedding_model,
        }

    def _case_ids_for(self, chunk_id: str) -> list[str]:
        """从权威标准个例反向生成 chunk 关联的个例 ID。"""
        if self.standard_cases_path is None or not self.standard_cases_path.is_file():
            return []
        try:
            import json

            payload = json.loads(
                self.standard_cases_path.read_text(encoding="utf-8")
            )
        except Exception:
            return []
        matched: list[str] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            chunk_ids = item.get("source_chunk_ids") or []
            if chunk_id in chunk_ids:
                matched.append(str(item.get("case_id") or ""))
        return [case_id for case_id in matched if case_id]

    def _ensure_collection(self) -> Any:
        """延迟创建 Milvus 客户端并保证集合存在。"""
        client = self._get_client()
        if client.has_collection(self.collection_name):
            return client
        with self._collection_lock:
            if client.has_collection(self.collection_name):
                return client
            from pymilvus import DataType, Function, FunctionType

            schema = client.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(
                field_name="chunk_id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=128,
            )
            schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                max_length=4096,
                enable_analyzer=True,
                analyzer_params={"type": "chinese"},
            )
            schema.add_field(
                field_name="dense_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=self.dimension,
            )
            schema.add_field(
                field_name="sparse_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )
            schema.add_field(
                field_name="case_ids",
                datatype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=64,
                max_length=128,
            )
            schema.add_field(
                field_name="source_pdf",
                datatype=DataType.VARCHAR,
                max_length=256,
            )
            schema.add_field(
                field_name="file_path",
                datatype=DataType.VARCHAR,
                max_length=512,
            )
            schema.add_field(
                field_name="chunk_no",
                datatype=DataType.INT64,
            )
            schema.add_field(
                field_name="text_length",
                datatype=DataType.INT64,
            )
            schema.add_field(
                field_name="content_hash",
                datatype=DataType.VARCHAR,
                max_length=64,
            )
            schema.add_field(
                field_name="document_version",
                datatype=DataType.VARCHAR,
                max_length=64,
            )
            schema.add_field(
                field_name="embedding_model",
                datatype=DataType.VARCHAR,
                max_length=128,
            )
            schema.add_function(
                Function(
                    name="text_bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=["text"],
                    output_field_names=["sparse_vector"],
                )
            )
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_vector_idx",
                index_type="FLAT",
                metric_type="COSINE",
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_vector_idx",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={"bm25_k1": 1.2, "bm25_b": 0.75},
            )
            client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
                consistency_level="Strong",
            )
            return client

    def _get_client(self) -> Any:
        """延迟创建 Milvus 客户端，启动阶段不强制连接。"""
        self._ensure_open()
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(
                uri=self.uri,
                token=self.token,
                db_name=self.database,
                timeout=self.timeout_seconds,
            )
        return self._client

    def _ensure_open(self) -> None:
        """阻止关闭后的 Milvus 资源继续被业务使用。"""
        if self._closed:
            raise RuntimeError("Milvus document vector store is closed")
