"""云端文本重排序客户端。"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from threading import Lock

from backend.app.config import settings


def _rerank_document_text(hit) -> str:
    """把普通 RAG 和多维检索的不同命中对象转换为重排序文本。"""
    if hit is None:
        return ""
    if isinstance(hit, str):
        return hit
    document_text = getattr(hit, "document_text", None)
    if document_text:
        return str(document_text)
    chunk = getattr(hit, "chunk", None)
    if chunk is not None:
        return "\n".join(
            (
                f"chunk_id: {getattr(chunk, 'chunk_id', '')}",
                f"source_pdf: {getattr(chunk, 'source_pdf', '')}",
                f"content: {getattr(chunk, 'content', '')}",
            )
        )
    case = getattr(hit, "case", None)
    if case is not None:
        return "\n".join(
            (
                f"case_id: {getattr(case, 'case_id', '')}",
                f"title: {getattr(case, 'title', '')}",
                f"date_range: {getattr(case, 'date_range', '')}",
            )
        )
    return str(getattr(hit, "content", None) or hit)


class DashScopeRerankClient:
    """调用 DashScope 云端文本重排序服务。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
        timeout: float = 60.0,
        http_client=None,
    ):
        self.api_key = api_key or settings.dashscope_api_key
        self.base_url = base_url or settings.dashscope_base_url
        self.model = model or settings.dashscope_rerank_model
        self.endpoint = endpoint or settings.dashscope_rerank_endpoint or self._default_endpoint(self.base_url)
        self.timeout = timeout
        self.http_client = http_client
        self._shared_http_client = None
        self._client_lock = Lock()
        self._result_cache: OrderedDict[str, list[int]] = OrderedDict()
        self._cache_lock = Lock()
        self._cache_limit = 128

    def is_available(self) -> bool:
        """仅在配置云端 API Key 和重排序地址后报告可用。"""
        return bool(self.api_key and self.endpoint)

    def rerank(self, question, hits, top_n):
        """调用云端接口，并按返回索引重排原始命中对象。"""
        if not hits:
            return []
        if not self.is_available():
            raise RuntimeError("云端 rerank 客户端不可用")
        payload = {
            "model": self.model,
            "input": {
                "query": question,
                "documents": [_rerank_document_text(hit) for hit in hits],
            },
            "parameters": {"top_n": min(top_n, len(hits)), "return_documents": False},
        }
        cache_key = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        with self._cache_lock:
            cached_indices = self._result_cache.pop(cache_key, None)
            if cached_indices is not None:
                self._result_cache[cache_key] = cached_indices
                return [hits[index] for index in cached_indices if 0 <= index < len(hits)][:top_n]
        response = self._post(payload)
        results = self._extract_results(response.json())
        if not results:
            raise RuntimeError("rerank 响应没有返回有效结果")
        ordered = []
        ordered_indices = []
        seen = set()
        for item in sorted(results, key=lambda value: value["score"], reverse=True):
            index = item["index"]
            if 0 <= index < len(hits) and index not in seen:
                seen.add(index)
                ordered.append(hits[index])
                ordered_indices.append(index)
            if len(ordered) >= top_n:
                break
        with self._cache_lock:
            self._result_cache[cache_key] = ordered_indices
            while len(self._result_cache) > self._cache_limit:
                self._result_cache.popitem(last=False)
        return ordered

    def _post(self, payload):
        """发送带鉴权的云端重排序请求。"""
        client = self.http_client
        if client is None:
            import httpx

            with self._client_lock:
                if self._shared_http_client is None:
                    self._shared_http_client = httpx.Client()
                client = self._shared_http_client
        response = client.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response

    def _extract_results(self, payload: dict) -> list[dict]:
        """兼容 DashScope 常见的结果包装层。"""
        raw_results = payload.get("output", {}).get("results") or payload.get("results") or payload.get("data", [])
        results = []
        for item in raw_results:
            score = item.get("relevance_score") if item.get("relevance_score") is not None else item.get("score")
            if item.get("index") is None or score is None:
                continue
            results.append({"index": int(item["index"]), "score": float(score)})
        return results

    def _document_text(self, hit) -> str:
        """保留旧调用点使用的实例方法。"""
        return _rerank_document_text(hit)

    def _default_endpoint(self, base_url: str) -> str:
        """返回 DashScope 官方重排序地址，不能从专属 Chat MaaS 域名拼接。"""
        return "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


class ModelFirstReranker:
    """优先调用云端 Rerank，服务异常时回退到本地关键词排序。"""

    def __init__(self, model_reranker, fallback_reranker):
        self.model_reranker = model_reranker
        self.fallback_reranker = fallback_reranker
        self.last_mode = "not_requested"

    def rerank(self, question, hits, top_n):
        """执行云端重排序并隔离网络故障。"""
        try:
            reranked = self.model_reranker.rerank(question, hits, top_n)
        except Exception as exc:
            self.last_mode = f"keyword_fallback: {exc}"
            return self.fallback_reranker.rerank(question, hits, top_n)
        self.last_mode = "qwen3-rerank"
        return reranked


def get_rerank_client() -> DashScopeRerankClient:
    """返回项目唯一的云端重排序客户端。"""
    return DashScopeRerankClient(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        model=settings.dashscope_rerank_model,
        endpoint=settings.dashscope_rerank_endpoint,
    )
