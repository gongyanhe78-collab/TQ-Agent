"""云端文本向量客户端。"""
from __future__ import annotations

from collections import OrderedDict
from threading import Lock

from backend.app.config import settings


class DashScopeEmbeddingClient:
    """通过 OpenAI 兼容接口调用 DashScope 文本向量模型。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        batch_size: int = 4,
    ):
        self.api_key = api_key or settings.dashscope_api_key
        self.base_url = base_url or settings.dashscope_base_url
        self.model = model or settings.dashscope_embedding_model
        self.batch_size = batch_size
        self._openai_client = None
        self._client_lock = Lock()
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._query_cache_lock = Lock()
        self._query_cache_limit = 256

    def is_available(self) -> bool:
        """仅在配置云端 API Key 后报告可用。"""
        return bool(self.api_key)

    def expected_dimension(self) -> int | None:
        """返回已知模型的向量维度，用于启动时校验主向量库。"""
        return 1024 if self.model == "text-embedding-v4" else None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """按批次调用云端 Embedding 接口。"""
        if not self.is_available():
            raise RuntimeError("未配置 DashScope API Key，无法调用云端向量模型")
        client = self._client()
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            response = client.embeddings.create(model=self.model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """为单条用户查询生成云端向量。"""
        key = f"{self.model}\n{str(text or '').strip()}"
        with self._query_cache_lock:
            cached = self._query_cache.pop(key, None)
            if cached is not None:
                self._query_cache[key] = cached
                return list(cached)
        vector = self.embed_documents([text])[0]
        with self._query_cache_lock:
            self._query_cache[key] = list(vector)
            while len(self._query_cache) > self._query_cache_limit:
                self._query_cache.popitem(last=False)
        return list(vector)

    def _client(self):
        """复用向量接口连接池，降低短查询的固定网络开销。"""
        with self._client_lock:
            if self._openai_client is None:
                from openai import OpenAI

                self._openai_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            return self._openai_client


def get_embedding_client() -> DashScopeEmbeddingClient:
    """返回项目唯一的云端向量客户端。"""
    return DashScopeEmbeddingClient(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        model=settings.dashscope_embedding_model,
    )
