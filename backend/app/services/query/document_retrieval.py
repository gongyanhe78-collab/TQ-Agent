"""文档片段检索、个例范围读取和上下文压缩。"""
from __future__ import annotations

import inspect
import re
from typing import Any

from backend.app.models import DocumentChunk, DocumentRetrievalHit
from backend.app.observability.context import current_context
from backend.app.observability import metrics
from backend.app.services.vector.contracts import VectorHit, VectorQuery
from backend.app.services.vector.json_document_store import lexical_terms


_CONTEXT_EVIDENCE_SECTIONS: tuple[tuple[str, ...], ...] = (
    (
        "天气实况",
        "实况特征",
        "过程降水",
        "累计降水",
        "小时雨强",
        "最大风速",
        "极大风速",
        "最高气温",
        "最低气温",
        "降温幅度",
        "积雪深度",
        "能见度",
        "国家站",
        "区域站",
    ),
    (
        "环流形势",
        "演变特征",
        "天气成因",
        "成因分析",
        "500hPa",
        "700hPa",
        "850hPa",
        "高压脊",
        "冷涡",
        "低涡",
        "切变线",
        "冷锋",
        "副热带高压",
        "水汽",
        "垂直运动",
    ),
    ("影响", "灾情", "农业", "交通", "电力", "人体健康", "服务", "风险", "预警区域"),
    (
        "数值预报",
        "模式预报",
        "预报产品",
        "预报着眼点",
        "预报效果",
        "预报偏差",
        "EC模式",
        "CMA模式",
        "GFS",
        "预警",
        "漏报",
        "空报",
        "订正",
    ),
)


class DocumentRetriever:
    """封装向量检索及不可用时的词面召回。"""

    def __init__(
        self,
        *,
        embedding_client: Any,
        document_store: Any,
        standard_case_store: Any,
    ) -> None:
        self.embedding_client = embedding_client
        self.document_store = document_store
        self.standard_case_store = standard_case_store
        self.last_retrieval_diagnostics: dict[str, str] | None = None

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        top_n: int = 3,
        context_case_ids: list[str] | None = None,
    ) -> tuple[str, list[DocumentRetrievalHit]]:
        """召回与问题相关的文档片段，保持原检索和排序规则。"""
        self.last_retrieval_diagnostics = None
        scoped_chunks = self._document_chunks_for_context_cases(context_case_ids)
        if scoped_chunks:
            return "document_context_cases", self._rank_context_document_chunks(
                question,
                scoped_chunks,
                context_case_ids or [],
                top_n,
            )
        try:
            if not self.embedding_client.is_available():
                diagnostics = {
                    "category": "embedding_unavailable",
                    "backend": self._store_backend(),
                    "reason": "embedding client is unavailable",
                    "exception_type": "",
                }
                self.last_retrieval_diagnostics = diagnostics
                metrics.record_fallback(
                    "vector",
                    backend=diagnostics["backend"],
                    reason=diagnostics["category"],
                )
                return self._lexical_result(
                    question,
                    top_k,
                    top_n,
                    fallback_diagnostics=diagnostics,
                )
            self._ensure_document_index_compatible()
            query_embedding = self.embedding_client.embed_query(question)
            hits = self._query_store(
                query_embedding,
                top_k,
                self._vector_query(question, query_embedding, top_k),
            )
            mode, normalized = self._normalize_hits(
                hits,
                "vector",
                question,
                top_k,
            )
        except Exception as exc:
            diagnostics = self._retrieval_diagnostics(exc)
            self.last_retrieval_diagnostics = diagnostics
            metrics.record_fallback(
                "vector",
                backend=diagnostics["backend"],
                reason=diagnostics["category"],
            )
            return self._lexical_result(
                question,
                top_k,
                top_n,
                fallback_diagnostics=diagnostics,
            )
        return mode, normalized[:top_n]

    def _document_chunks_for_context_cases(self, context_case_ids: list[str] | None = None) -> list:
        if not context_case_ids:
            return []
        wanted_case_ids = set(context_case_ids)
        context_cases = [
            case for case in self.standard_case_store.list_cases() if case.case_id in wanted_case_ids
        ]
        if not context_cases:
            return []
        wanted_chunk_ids = {
            chunk_id
            for case in context_cases
            for chunk_id in case.source_chunk_ids
            if chunk_id
        }
        wanted_sources = {case.source_pdf for case in context_cases if case.source_pdf}
        chunks = []
        for chunk in self.document_store.list_chunks():
            if wanted_chunk_ids and chunk.chunk_id in wanted_chunk_ids:
                chunks.append(chunk)
            elif not wanted_chunk_ids and chunk.source_pdf in wanted_sources:
                chunks.append(chunk)
        return chunks

    def _rank_context_document_chunks(
        self,
        question: str,
        chunks: list,
        context_case_ids: list[str],
        top_n: int,
    ) -> list[VectorHit]:
        cases_by_id = {
            case.case_id: case
            for case in self.standard_case_store.list_cases()
            if case.case_id in set(context_case_ids)
        }
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        tokens = [part for part in re.split(r"[\s，。！？、；：,.!?;:（）()]+", question) if part]
        context = current_context()
        backend = self._store_backend()
        collection = self._store_collection()
        applied_filters = {"context_case_ids": list(context_case_ids)}

        def make_hit(chunk: DocumentChunk) -> VectorHit:
            normalized = chunk.content.replace(" ", "")
            score = sum(
                max(1, len(token)) for token in tokens if token.replace(" ", "") in normalized
            )
            return VectorHit(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                raw_score=float(score),
                fused_score=float(score),
                metadata={
                    "source_pdf": chunk.source_pdf,
                    "chunk_no": int(chunk.chunk_no),
                    "file_path": chunk.file_path or "",
                    "text_length": len(chunk.content),
                    "applied_filters": applied_filters,
                },
                backend=backend,
                collection=collection,
                retrieval_mode="lexical/context_cases",
                request_id=context.request_id,
                trace_id=context.trace_id,
            )

        ordered_groups: list[list[DocumentChunk]] = []
        seen_chunk_ids: set[str] = set()
        for case_id in context_case_ids:
            case = cases_by_id.get(case_id)
            if case is None:
                continue
            group = []
            for chunk_id in case.source_chunk_ids:
                chunk = chunk_by_id.get(chunk_id)
                if chunk is not None and chunk.chunk_id not in seen_chunk_ids:
                    group.append(chunk)
                    seen_chunk_ids.add(chunk.chunk_id)
            group.sort(key=lambda item: (item.source_pdf, item.chunk_no, item.chunk_id))
            if group:
                ordered_groups.append(group)

        if len(cases_by_id) <= 2:
            ungrouped = [
                chunk
                for chunk in sorted(chunks, key=lambda item: (item.source_pdf, item.chunk_no, item.chunk_id))
                if chunk.chunk_id not in seen_chunk_ids
            ]
            compression_groups = list(ordered_groups)
            if ungrouped:
                compression_groups.append(ungrouped)
            selected = [
                chunk
                for group in compression_groups
                for chunk in compress_context_chunk_group(question, group, limit=12)
            ]
            return [make_hit(chunk) for chunk in selected]

        per_group = [
            sorted(group, key=lambda item: make_hit(item).score, reverse=True)
            for group in ordered_groups
        ]
        limit = min(30, max(int(top_n), int(top_n) * 3))
        selected = []
        index = 0
        while len(selected) < limit and any(index < len(group) for group in per_group):
            for group in per_group:
                if index < len(group):
                    selected.append(group[index])
                    if len(selected) >= limit:
                        break
            index += 1
        return [make_hit(chunk) for chunk in selected]

    def _ensure_document_index_compatible(self) -> None:
        if not self.embedding_client.is_available():
            raise RuntimeError("embedding client is unavailable")
        expected = self.embedding_client.expected_dimension()
        if expected is None:
            return
        actual = self.document_store.collection_info().get("dimension")
        if actual != expected:
            raise RuntimeError(
                f"document vector index dimension mismatch: expected {expected}, got {actual}; "
                "rebuild document_index"
            )

    def _vector_query(
        self,
        question: str,
        embedding: list[float],
        top_k: int,
    ) -> VectorQuery:
        """合并当前请求追踪字段，构造统一向量查询。"""
        context = current_context()
        return VectorQuery(
            query_text=str(question or ""),
            embedding=[float(value) for value in embedding],
            top_k=max(0, int(top_k)),
            request_id=context.request_id,
            trace_id=context.trace_id,
            user_id=context.user_id,
        )

    def _lexical_result(
        self,
        question: str,
        top_k: int,
        top_n: int,
        *,
        fallback_diagnostics: dict[str, str] | None = None,
    ) -> tuple[str, list[VectorHit]]:
        """优先让统一存储产生诚实的 JSON 词面命中，再回退本地词面扫描。"""
        query = self._vector_query(question, [], top_k)
        try:
            hits = self._query_store([], top_k, query)
            mode, normalized = self._normalize_hits(
                hits,
                "lexical",
                question,
                top_k,
                backend_override="json",
                metadata_extra=(
                    {"retrieval_fallback": fallback_diagnostics}
                    if fallback_diagnostics
                    else None
                ),
            )
        except Exception:
            mode, normalized = self._normalize_hits(
                self._lexical_recall(question, top_k),
                "lexical",
                question,
                top_k,
                backend_override="json",
                metadata_extra=(
                    {"retrieval_fallback": fallback_diagnostics}
                    if fallback_diagnostics
                    else None
                ),
            )
        return mode, normalized[:top_n]

    def _query_store(
        self,
        embedding: list[float],
        top_k: int,
        query: VectorQuery,
    ) -> list[Any]:
        """调用统一查询签名，同时兼容只接受两个参数的既有存储。"""
        if self._store_supports_vector_query():
            return self.document_store.query(
                embedding,
                top_k=top_k,
                query=query,
            )
        return self.document_store.query(embedding, top_k)

    def _store_supports_vector_query(self) -> bool:
        """检查旧存储是否已实现可选的统一查询参数。"""
        try:
            parameters = inspect.signature(self.document_store.query).parameters
        except (TypeError, ValueError):
            return True
        return "query" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

    def _normalize_hits(
        self,
        hits: list[Any],
        default_mode: str,
        question: str,
        top_k: int,
        *,
        backend_override: str = "",
        metadata_extra: dict[str, Any] | None = None,
    ) -> tuple[str, list[VectorHit]]:
        """把旧命中补齐为真实后端、集合、模式和追踪字段。"""
        context = current_context()
        backend = backend_override or self._store_backend()
        collection = self._store_collection()
        normalized: list[VectorHit] = []
        for hit in hits:
            chunk = getattr(hit, "chunk", None)
            if chunk is None:
                continue
            score = float(getattr(hit, "score", 0.0) or 0.0)
            raw_score = float(getattr(hit, "raw_score", score) or 0.0)
            fused_score = getattr(hit, "fused_score", score)
            metadata = dict(getattr(hit, "metadata", {}) or {})
            metadata.setdefault("source_pdf", chunk.source_pdf)
            metadata.setdefault("chunk_no", int(chunk.chunk_no))
            metadata.setdefault("file_path", chunk.file_path or "")
            metadata.setdefault("text_length", len(chunk.content))
            metadata.setdefault("applied_filters", {})
            for key, value in (metadata_extra or {}).items():
                metadata[key] = (
                    dict(value)
                    if isinstance(value, dict)
                    else list(value)
                    if isinstance(value, list)
                    else value
                )
            normalized.append(
                VectorHit(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    raw_score=raw_score,
                    fused_score=(
                        float(fused_score)
                        if fused_score is not None
                        else None
                    ),
                    metadata=metadata,
                    backend=str(getattr(hit, "backend", "") or backend),
                    collection=str(
                        getattr(hit, "collection", "") or collection
                    ),
                    retrieval_mode=str(
                        getattr(hit, "retrieval_mode", "")
                        or self._mode_name(default_mode, backend)
                    ),
                    request_id=str(
                        getattr(hit, "request_id", "") or context.request_id
                    ),
                    trace_id=str(
                        getattr(hit, "trace_id", "") or context.trace_id
                    ),
                )
            )
        mode = (
            normalized[0].retrieval_mode
            if normalized
            else self._mode_name(default_mode, backend)
        )
        return mode, normalized

    def _retrieval_diagnostics(self, exc: Exception) -> dict[str, str]:
        """生成可观测且不进入指标标签的真实降级诊断。"""
        reason = str(exc) or type(exc).__name__
        category = (
            "dimension_mismatch"
            if "dimension mismatch" in reason.lower()
            else "backend_error"
        )
        return {
            "category": category,
            "backend": self._store_backend(),
            "reason": reason,
            "exception_type": type(exc).__name__,
        }

    def _store_backend(self) -> str:
        """读取活动存储的真实后端名称。"""
        backend = str(getattr(self.document_store, "backend", "") or "")
        if backend:
            return backend
        try:
            return str(self.document_store.collection_info().get("backend") or "")
        except Exception:
            return "document"

    def _store_collection(self) -> str:
        """读取活动存储的真实集合名称。"""
        collection = str(
            getattr(self.document_store, "collection_name", "") or ""
        )
        if collection:
            return collection
        try:
            info = self.document_store.collection_info()
            return str(info.get("collection") or info.get("name") or "")
        except Exception:
            return ""

    @staticmethod
    def _mode_name(category: str, backend: str) -> str:
        """按真实检索类别和后端拼装稳定模式名。"""
        return f"{category}/{backend}" if backend else category

    def _lexical_recall(self, question: str, top_k: int) -> list[DocumentRetrievalHit]:
        tokens = lexical_terms(question)
        hits = []
        for chunk in self.document_store.list_chunks():
            normalized = chunk.content.replace(" ", "")
            score = sum(
                max(1, len(token)) for token in tokens if token and token.replace(" ", "") in normalized
            )
            if score > 0:
                hits.append(DocumentRetrievalHit(chunk=chunk, score=float(score)))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]


def compress_context_chunk_group(
    question: str,
    chunks: list[DocumentChunk],
    limit: int = 12,
) -> list[DocumentChunk]:
    """覆盖关键业务章节并保留相邻上下文，压缩已知个例的模型输入。"""
    ordered = sorted(chunks, key=lambda item: (item.source_pdf, item.chunk_no, item.chunk_id))
    if len(ordered) <= limit:
        return ordered

    selected_indices = {0, len(ordered) - 1}

    def section_score(index: int, terms: tuple[str, ...]) -> int:
        content = ordered[index].content
        return sum(content.count(term) * max(1, len(term)) for term in terms)

    for terms in _CONTEXT_EVIDENCE_SECTIONS:
        ranked = sorted(
            range(len(ordered)),
            key=lambda index: (section_score(index, terms), -index),
            reverse=True,
        )
        for index in [item for item in ranked if section_score(item, terms) > 0][:2]:
            selected_indices.add(index)

    focus_terms = tuple(
        term for terms in _CONTEXT_EVIDENCE_SECTIONS for term in terms if term in question
    )
    if focus_terms:
        focused = sorted(
            range(len(ordered)),
            key=lambda index: (section_score(index, focus_terms), -index),
            reverse=True,
        )
        selected_indices.update(
            [item for item in focused if section_score(item, focus_terms) > 0][:2]
        )

    for index in sorted(tuple(selected_indices)):
        if len(selected_indices) >= limit:
            break
        if index + 1 < len(ordered):
            selected_indices.add(index + 1)

    all_terms = tuple(term for terms in _CONTEXT_EVIDENCE_SECTIONS for term in terms)
    remaining = sorted(
        (index for index in range(len(ordered)) if index not in selected_indices),
        key=lambda index: (section_score(index, all_terms), -index),
        reverse=True,
    )
    selected_indices.update(remaining[: max(0, limit - len(selected_indices))])
    return [ordered[index] for index in sorted(selected_indices)[:limit]]
