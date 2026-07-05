"""
通义千问向量嵌入客户端模块
将文本转换为向量表示，用于相似度检索
"""
from __future__ import annotations

import hashlib

from backend.app.config import settings


class DashScopeEmbeddingClient:
    """
    阿里云 DashScope 向量嵌入客户端
    通过 OpenAI 兼容 API 调用通义千问的嵌入模型，将文本转换为向量
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        batch_size: int = 4,
    ):
        """
        初始化向量嵌入客户端

        Args:
            api_key: API 密钥，不传则从配置中读取
            base_url: API 基础地址，不传则使用默认配置
            model: 嵌入模型名称，不传则使用默认配置
            batch_size: 批量处理时每批的文本数量
        """
        self.api_key = api_key or settings.api_key
        self.base_url = base_url or settings.base_url
        self.model = model or settings.embedding_model
        self.batch_size = batch_size

    def is_available(self) -> bool:
        """
        检查客户端是否可用（是否配置了 API 密钥）

        Returns:
            可用返回 True，否则返回 False
        """
        return bool(self.api_key)

    def expected_dimension(self) -> int | None:
        """
        返回嵌入向量的预期维度

        Returns:
            向量维度整数（如 1024），未知模型则返回 None
        """
        if self.model == "text-embedding-v4":
            return 1024
        return None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量将文本列表转换为向量
        会按 batch_size 分批调用 API，避免请求过大

        Args:
            texts: 待转换的文本列表

        Returns:
            向量列表，每个向量是 float 数组
        """
        # 如果没有配置 API Key，使用降级的哈希向量
        if not self.is_available():
            return [self._fallback_embedding(text) for text in texts]

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for DashScope embedding calls.") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        embeddings: list[list[float]] = []
        # 分批调用嵌入 API
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            response = client.embeddings.create(model=self.model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        将单条查询文本转换为向量
        内部调用 embed_documents 实现

        Args:
            text: 查询文本

        Returns:
            向量数组
        """
        return self.embed_documents([text])[0]

    def _fallback_embedding(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = [int(byte) / 255 for byte in digest[:32]]
        return vector
