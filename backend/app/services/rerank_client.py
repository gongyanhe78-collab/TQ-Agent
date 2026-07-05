"""
通义千问重排序客户端模块
使用 qwen3-rerank 模型对向量检索结果进行二次排序，提升检索准确率
包含模型重排序客户端和降级机制封装
"""
from __future__ import annotations

from urllib.parse import urlparse

from backend.app.config import settings


class DashScopeRerankClient:
    """
    DashScope 通义千问文本重排序客户端
    调用 qwen3-rerank 模型，对向量检索的初步结果进行精排
    根据问题与案例的相关性重新排序，提升 RAG 系统的检索准确率
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
        timeout: float = 60.0,
        http_client=None,
    ):
        """
        初始化重排序客户端

        Args:
            api_key: API 密钥，不传则从配置中读取
            base_url: API 基础地址，不传则使用默认配置
            model: 重排序模型名称，不传则使用默认 qwen3-rerank
            endpoint: 重排序接口完整地址，不传则自动拼接
            timeout: HTTP 请求超时时间，默认 60 秒
            http_client: 自定义 HTTP 客户端，不传则自动使用 httpx
        """
        self.api_key = api_key or settings.api_key
        self.base_url = base_url or settings.base_url
        self.model = model or settings.rerank_model
        self.endpoint = endpoint or settings.rerank_endpoint or self._default_endpoint(self.base_url)
        self.timeout = timeout
        self.http_client = http_client

    def is_available(self) -> bool:
        """
        检查客户端是否可用（是否配置了 API 密钥）

        Returns:
            可用返回 True，否则返回 False
        """
        return bool(self.api_key)

    def rerank(self, question, hits, top_n):
        """
        对检索结果进行重排序

        Args:
            question: 用户问题
            hits: 向量检索返回的命中结果列表
            top_n: 返回前 N 个结果

        Returns:
            重排序后的命中结果列表

        Raises:
            RuntimeError: 客户端不可用或响应无结果时抛出
        """
        # 空输入直接返回空列表
        if not hits:
            return []
        if not self.is_available():
            raise RuntimeError("rerank client is unavailable")

        # 为每个案例构建重排序用的文本文档
        documents = [self._document_text(hit) for hit in hits]
        # 构建 API 请求 payload
        payload = {
            "model": self.model,
            "input": {
                "query": question,
                "documents": documents,
            },
            "parameters": {
                "top_n": min(top_n, len(hits)),
                "return_documents": False,  # 不返回原始文档，只返回排序索引和分数
            },
        }
        # 发送 HTTP 请求
        response = self._post(payload)
        # 解析响应，提取排序结果
        results = self._extract_results(response.json())
        if not results:
            raise RuntimeError("rerank response did not include results")

        # 根据返回的索引和分数重新排序结果
        ordered = []
        seen = set()  # 记录已处理的索引，防止重复
        for result in sorted(results, key=lambda item: item["score"], reverse=True):
            index = result["index"]
            # 校验索引范围，确保不越界
            if 0 <= index < len(hits) and index not in seen:
                ordered.append(hits[index])
                seen.add(index)
            # 达到需要的数量后停止
            if len(ordered) >= top_n:
                break
        return ordered

    def _post(self, payload):
        """
        发送 HTTP POST 请求到重排序 API（内部方法）

        Args:
            payload: 请求数据字典

        Returns:
            httpx.Response 响应对象

        Raises:
            RuntimeError: httpx 包未安装时抛出
            HTTPError: 请求失败时抛出
        """
        client = self.http_client
        # 如果没有传入自定义客户端，动态导入 httpx
        if client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("httpx package is required for DashScope rerank calls.") from exc
            client = httpx

        # 发送 POST 请求
        response = client.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        # 如果状态码不是 2xx，抛出异常
        response.raise_for_status()
        return response

    def _extract_results(self, payload: dict) -> list[dict]:
        """
        从 API 响应中提取重排序结果（内部方法）
        兼容多种响应格式（output.results / results / data）

        Args:
            payload: API 响应的 JSON 字典

        Returns:
            包含 index 和 score 的字典列表，格式如 [{"index": 0, "score": 0.95}, ...]
        """
        # 尝试多种可能的响应字段位置
        raw_results = (
            payload.get("output", {}).get("results")
            or payload.get("results")
            or payload.get("data", [])
        )
        results = []
        for item in raw_results:
            index = item.get("index")
            # 兼容两种分数字段名：relevance_score 或 score
            score = (
                item.get("relevance_score")
                if item.get("relevance_score") is not None
                else item.get("score")
            )
            # 跳过不完整的结果
            if index is None or score is None:
                continue
            results.append({"index": int(index), "score": float(score)})
        return results

    def _document_text(self, hit) -> str:
        """
        构建用于重排序的案例文本文档（内部方法）

        Args:
            hit: 检索命中对象（包含 case 属性）

        Returns:
            拼接好的案例文本（每行一个字段）
        """
        case = hit.case
        # 用换行连接各个字段（过滤空字段）
        return "\n".join(
            part
            for part in [
                f"case_id: {case.case_id}",
                f"title: {case.title}",
                f"date_range: {case.date_range}",
                f"content: {case.content}",
            ]
            if part
        )

    def _default_endpoint(self, base_url: str) -> str:
        """
        根据 base_url 构建默认的重排序 API 端点（内部方法）

        Args:
            base_url: API 基础地址

        Returns:
            完整的重排序接口 URL
        """
        parsed = urlparse(base_url)
        # 如果是有效的 URL，使用相同的域名拼接端点
        if parsed.scheme and parsed.netloc:
            root = f"{parsed.scheme}://{parsed.netloc}"
            return f"{root}/api/v1/services/rerank/text-rerank/text-rerank"
        # 否则返回官方默认端点
        return "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


class ModelFirstReranker:
    """
    模型优先的重排序封装器
    优先使用 qwen3-rerank 模型重排序，失败时自动降级到关键词重排序
    保证系统在 API 故障时仍能正常工作
    """

    def __init__(self, model_reranker, fallback_reranker):
        """
        初始化重排序封装器

        Args:
            model_reranker: 模型重排序器（DashScopeRerankClient）
            fallback_reranker: 降级重排序器（KeywordReranker）
        """
        self.model_reranker = model_reranker
        self.fallback_reranker = fallback_reranker
        self.last_mode = "not_requested"  # 记录最后一次使用的重排序模式（用于调试）

    def rerank(self, question, hits, top_n):
        """
        执行重排序
        先尝试使用模型重排序，失败则降级到关键词重排序

        Args:
            question: 用户问题
            hits: 向量检索返回的命中结果列表
            top_n: 返回前 N 个结果

        Returns:
            重排序后的命中结果列表
        """
        try:
            # 优先使用模型重排序
            reranked = self.model_reranker.rerank(question, hits, top_n)
        except Exception as exc:
            # 模型重排序失败，降级到关键词重排序，并记录失败原因
            self.last_mode = f"keyword_fallback: {exc}"
            return self.fallback_reranker.rerank(question, hits, top_n)
        # 模型重排序成功
        self.last_mode = "qwen3-rerank"
        return reranked
