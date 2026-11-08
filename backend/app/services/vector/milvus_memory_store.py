"""Milvus 长期记忆向量适配器：所有查询强制绑定 user_id 过滤。"""
from __future__ import annotations

from threading import Lock
from typing import Any

from backend.app.services.vector.contracts import (
    CollectionInfo,
    VectorHit,
    VectorQuery,
    count_collection_rows,
)


def build_user_filter(user_id: str) -> str:
    """构造参数化用户过滤表达式，真实 user_id 不进入表达式字符串。"""
    normalized = str(user_id or "").strip()
    if not normalized:
        raise ValueError("user_id 不能为空")
    return "user_id == {user_id}"


def build_user_filter_params(user_id: str) -> dict[str, Any]:
    """返回用户过滤表达式参数。"""
    normalized = str(user_id or "").strip()
    if not normalized:
        raise ValueError("user_id 不能为空")
    return {"user_id": normalized}


class MilvusMemoryStore:
    """把 Milvus 作为长期记忆的向量索引，权威记忆事实保留在 PostgreSQL。"""

    backend = "milvus"

    def __init__(
        self,
        *,
        uri: str,
        collection_name: str,
        dimension: int,
        token: str = "",
        database: str = "default",
        schema_version: str = "v1",
        dense_top_k: int = 30,
        sparse_top_k: int = 30,
        rrf_k: int = 60,
        timeout_seconds: float = 10.0,
    ) -> None:
        """记录 Milvus 连接参数和固定检索参数。"""
        self.uri = str(uri)
        self.collection_name = str(collection_name)
        self.dimension = max(1, int(dimension))
        self.token = str(token or "")
        self.database = str(database or "default")
        self.schema_version = str(schema_version or "v1")
        self.dense_top_k = max(1, int(dense_top_k))
        self.sparse_top_k = max(1, int(sparse_top_k))
        self.rrf_k = max(1, int(rrf_k))
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._client: Any = None
        self._collection_lock = Lock()
        self._closed = False
        self._last_error: str = ""

    def upsert_memories(self, memories: list[dict[str, Any]]) -> None:
        """保存或更新记忆索引副本，不修改 PostgreSQL 权威事实。"""
        self._ensure_open()
        rows = [
            row
            for memory in memories
            for row in [self._memory_row(memory)]
            if row is not None
        ]
        if not rows:
            return
        client = self._ensure_collection()
        client.upsert(collection_name=self.collection_name, data=rows)

    def save(self, memories: list[dict[str, Any]]) -> None:
        """兼容直接保存记忆的调用语义。"""
        self.upsert_memories(memories)

    def delete_memories(
        self,
        memory_ids: list[str],
        *,
        user_id: str,
    ) -> None:
        """按用户删除记忆索引，跨用户 ID 不产生任何副作用。"""
        self._ensure_open()
        normalized = [
            str(memory_id)
            for memory_id in dict.fromkeys(memory_ids)
            if str(memory_id)
        ]
        if not normalized:
            return
        client = self._ensure_collection()
        expr = build_user_filter(user_id)
        expr_params = build_user_filter_params(user_id)
        client.delete(
            collection_name=self.collection_name,
            filter=f"memory_id in {normalized} and {expr}",
            filter_params=expr_params,
        )

    def delete(self, memory_ids: list[str], *, user_id: str) -> None:
        """兼容直接删除记忆的调用语义。"""
        self.delete_memories(memory_ids, user_id=user_id)

    def query_by_user(
        self,
        user_id: str,
        embedding: list[float],
        top_k: int,
        query: VectorQuery | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """只在指定用户的记忆范围内执行稠密与 BM25 混合检索。"""
        self._ensure_open()
        resolved = query or VectorQuery()
        resolved_embedding = [float(value) for value in (embedding or resolved.embedding)]
        if not resolved_embedding:
            return []
        if int(top_k or resolved.top_k) <= 0:
            return []
        client = self._ensure_collection()
        expr = build_user_filter(user_id)
        expr_params = build_user_filter_params(user_id)

        try:
            from pymilvus import AnnSearchRequest, RRFRanker

            requests = [
                AnnSearchRequest(
                    data=[resolved_embedding],
                    anns_field="dense_vector",
                    param={"metric_type": "COSINE"},
                    limit=self.dense_top_k,
                    expr=expr,
                    expr_params=expr_params,
                ),
                AnnSearchRequest(
                    data=[str(resolved.query_text or "")],
                    anns_field="sparse_vector",
                    param={"metric_type": "BM25"},
                    limit=self.sparse_top_k,
                    expr=expr,
                    expr_params=expr_params,
                ),
            ]
            results = client.hybrid_search(
                collection_name=self.collection_name,
                reqs=requests,
                ranker=RRFRanker(k=self.rrf_k),
                limit=int(top_k or resolved.top_k),
                output_fields=[
                    "memory_id",
                    "user_id",
                    "memory_type",
                    "sensitivity",
                    "status",
                    "summary",
                    "source_message_id",
                    "version",
                ],
            )
        except Exception:
            raise

        hits: list[VectorHit] = []
        for row in (results or [[]])[0]:
            entity = dict(row.get("entity") or {})
            memory_id = str(entity.get("memory_id") or row.get("id") or "")
            if not memory_id:
                continue
            distance = row.get("distance")
            score = float(distance) if distance is not None else 0.0
            hits.append(
                VectorHit(
                    chunk_id=memory_id,
                    content=str(entity.get("summary") or ""),
                    raw_score=score,
                    fused_score=score,
                    metadata={
                        "memory_id": memory_id,
                        "user_id": str(entity.get("user_id") or ""),
                        "memory_type": str(entity.get("memory_type") or ""),
                        "sensitivity": str(entity.get("sensitivity") or ""),
                        "status": str(entity.get("status") or ""),
                        "source_message_id": entity.get("source_message_id"),
                        "version": entity.get("version"),
                        "applied_filters": {"user_id": user_id},
                    },
                    backend=self.backend,
                    collection=self.collection_name,
                    retrieval_mode="milvus/memory_hybrid_rrf",
                    request_id=resolved.request_id,
                    trace_id=resolved.trace_id,
                )
            )
        return hits

    def rebuild(self, user_id: str | None = None) -> None:
        """重建全部或指定用户的记忆索引；本适配器只负责重建后的写入。"""
        self._ensure_open()
        client = self._get_client()
        if user_id is None:
            if client.has_collection(self.collection_name):
                client.drop_collection(self.collection_name)
            self._ensure_collection()

    def health(self) -> dict[str, Any]:
        """返回真实健康状态，连接失败时报告不可用而不是抛出异常。"""
        count = 0
        detail = ""
        health_status = "healthy"
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
                detail = f"Milvus 记忆集合状态读取失败：{exc}"
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
        ).to_dict()

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

    def _memory_row(self, memory: dict[str, Any]) -> dict[str, Any] | None:
        """构造 Milvus 记忆插入行，正文只进入可重建的索引字段。"""
        memory_id = str(memory.get("memory_id") or "")
        user_id = str(memory.get("user_id") or "")
        dense_vector = memory.get("dense_vector") or []
        if not memory_id or not user_id or not dense_vector:
            return None
        return {
            "memory_id": memory_id,
            "user_id": user_id,
            "session_id": str(memory.get("session_id") or ""),
            "memory_type": str(memory.get("memory_type") or ""),
            "sensitivity": str(memory.get("sensitivity") or ""),
            "status": str(memory.get("status") or "active"),
            "summary": str(memory.get("summary") or memory.get("content") or "")[:4096],
            "dense_vector": [float(value) for value in dense_vector],
            "source_message_id": int(memory.get("source_message_id") or 0),
            "version": int(memory.get("version") or 1),
            "created_at": int(memory.get("created_at") or 0),
        }

    def _ensure_collection(self) -> Any:
        """延迟创建 Milvus 客户端并保证记忆集合存在。"""
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
                field_name="memory_id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=128,
            )
            schema.add_field(
                field_name="user_id",
                datatype=DataType.VARCHAR,
                max_length=64,
            )
            schema.add_field(
                field_name="session_id",
                datatype=DataType.VARCHAR,
                max_length=64,
            )
            schema.add_field(
                field_name="memory_type",
                datatype=DataType.VARCHAR,
                max_length=32,
            )
            schema.add_field(
                field_name="sensitivity",
                datatype=DataType.VARCHAR,
                max_length=16,
            )
            schema.add_field(
                field_name="status",
                datatype=DataType.VARCHAR,
                max_length=16,
            )
            schema.add_field(
                field_name="summary",
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
                field_name="source_message_id",
                datatype=DataType.INT64,
            )
            schema.add_field(
                field_name="version",
                datatype=DataType.INT64,
            )
            schema.add_field(
                field_name="created_at",
                datatype=DataType.INT64,
            )
            schema.add_function(
                Function(
                    name="memory_bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=["summary"],
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
            raise RuntimeError("Milvus memory vector store is closed")
