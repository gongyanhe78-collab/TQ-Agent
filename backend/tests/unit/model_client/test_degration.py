"""模型客户端熔断与降级故障注入测试。"""
from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from backend.app.core.resilience import (
    CircuitOpenError,
    get_circuit_breaker,
    reset_circuit_breakers,
)
from backend.app.services.model_client.chat import DashScopeChatClient
from backend.app.services.model_client.embedding import DashScopeEmbeddingClient
from backend.app.services.model_client.rerank import (
    DashScopeRerankClient,
    ModelFirstReranker,
)


class _FakeCompletions:
    """模拟 OpenAI 聊天补全端点。"""

    def __init__(self, owner: _FakeChat) -> None:
        self.owner = owner
        # OpenAI 客户端路径为 client.chat.completions.create，补齐命名空间层级。
        self.completions = self

    def create(self, **_kwargs):
        """按配置抛错或返回固定文本。"""
        if self.owner.raise_error:
            raise RuntimeError("simulated chat failure")
        return _FakeCompletion("ok")


class _FakeChat:
    """模拟 OpenAI 客户端。"""

    def __init__(self, raise_error: bool = False) -> None:
        self.raise_error = raise_error
        self.chat = _FakeCompletions(self)


class _FakeCompletion:
    """模拟补全响应。"""

    def __init__(self, text: str) -> None:
        self.choices = [_FakeChoice(text)]


class _FakeChoice:
    """模拟响应选择项。"""

    def __init__(self, text: str) -> None:
        self.message = _FakeMessage(text)


class _FakeMessage:
    """模拟响应消息。"""

    def __init__(self, content: str) -> None:
        self.content = content


class _FakeEmbeddings:
    """模拟 OpenAI 向量端点。"""

    def __init__(self, owner: _FakeEmbeddingClient) -> None:
        self.owner = owner

    def create(self, **_kwargs):
        if self.owner.raise_error:
            raise RuntimeError("simulated embedding failure")
        return _FakeEmbeddingResponse([[0.1, 0.2, 0.3]])


class _FakeEmbeddingClient:
    """模拟向量客户端。"""

    def __init__(self, raise_error: bool = False) -> None:
        self.raise_error = raise_error
        self.embeddings = _FakeEmbeddings(self)


class _FakeEmbeddingResponse:
    """模拟向量响应。"""

    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [_FakeEmbeddingItem(vector) for vector in vectors]


class _FakeEmbeddingItem:
    """模拟单个向量结果。"""

    def __init__(self, vector: list[float]) -> None:
        self.embedding = vector


class _FakeRerankResponse:
    """模拟重排序 HTTP 响应。"""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class ChatCircuitBreakerTests(unittest.TestCase):
    """验证聊天客户端连续失败触发熔断并快速失败。"""

    def setUp(self) -> None:
        reset_circuit_breakers()

    def test_continuous_failures_open_breaker_and_fail_fast(self) -> None:
        client = DashScopeChatClient(api_key="test-key", base_url="http://localhost")
        breaker = get_circuit_breaker("chat")
        fake = _FakeChat(raise_error=True)
        with patch.object(client, "_client", return_value=fake):
            for _ in range(breaker.failure_threshold):
                with self.assertRaises(RuntimeError):
                    client.answer_with_context("问题", ["证据片段"])
        self.assertEqual(breaker.snapshot()["state"], "open")
        # 熔断打开后快速失败，不再触碰底层客户端。
        with patch.object(client, "_client", side_effect=AssertionError("不应调用底层")):
            with self.assertRaises(CircuitOpenError):
                client.answer_with_context("问题", ["证据片段"])

    def test_breaker_recovers_after_cooldown(self) -> None:
        reset_circuit_breakers()
        breaker = get_circuit_breaker("chat", failure_threshold=2, cooldown_seconds=0.01)
        client = DashScopeChatClient(api_key="test-key", base_url="http://localhost")
        fake_fail = _FakeChat(raise_error=True)
        with patch.object(client, "_client", return_value=fake_fail):
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    client.answer_with_context("问题", ["证据"])
        self.assertEqual(breaker.snapshot()["state"], "open")
        time.sleep(0.02)
        fake_ok = _FakeChat(raise_error=False)
        with patch.object(client, "_client", return_value=fake_ok):
            result = client.answer_with_context("问题", ["证据"])
        self.assertEqual(result, "ok")
        self.assertEqual(breaker.snapshot()["state"], "closed")

    def test_stream_general_chat_marks_failure(self) -> None:
        client = DashScopeChatClient(api_key="test-key", base_url="http://localhost")
        breaker = get_circuit_breaker("chat")
        fake = _FakeChat(raise_error=True)
        with patch.object(client, "_client", return_value=fake):
            with self.assertRaises(RuntimeError):
                for _text in client.stream_general_chat("重庆天气怎么样"):
                    pass
        self.assertEqual(breaker.snapshot()["failure_count"], 1)


class EmbeddingCircuitBreakerTests(unittest.TestCase):
    """验证向量客户端熔断与快速失败。"""

    def setUp(self) -> None:
        reset_circuit_breakers()

    def test_embedding_failures_open_breaker(self) -> None:
        client = DashScopeEmbeddingClient(api_key="test-key", base_url="http://localhost")
        breaker = get_circuit_breaker("embedding")
        fake = _FakeEmbeddingClient(raise_error=True)
        with patch.object(client, "_client", return_value=fake):
            for _ in range(breaker.failure_threshold):
                with self.assertRaises(RuntimeError):
                    client.embed_documents(["文本"])
        self.assertEqual(breaker.snapshot()["state"], "open")
        with patch.object(client, "_client", side_effect=AssertionError("不应调用底层")):
            with self.assertRaises(CircuitOpenError):
                client.embed_documents(["文本"])


class RerankCircuitBreakerTests(unittest.TestCase):
    """验证重排序熔断与关键词回退。"""

    def setUp(self) -> None:
        reset_circuit_breakers()

    def test_rerank_failures_open_breaker(self) -> None:
        client = DashScopeRerankClient(api_key="test-key", endpoint="http://localhost/rerank")
        breaker = get_circuit_breaker("rerank")
        hits = ["甲", "乙", "丙"]
        with patch.object(client, "_post", side_effect=RuntimeError("simulated rerank failure")):
            for _ in range(breaker.failure_threshold):
                with self.assertRaises(RuntimeError):
                    client.rerank("问题", hits, 2)
        self.assertEqual(breaker.snapshot()["state"], "open")
        with patch.object(client, "_post", side_effect=AssertionError("不应调用底层")):
            with self.assertRaises(CircuitOpenError):
                client.rerank("问题", hits, 2)

    def test_model_first_reranker_falls_back_to_keyword(self) -> None:
        hits = ["甲", "乙", "丙"]

        class _FailingReranker:
            """模拟云端重排序持续失败。"""

            def rerank(self, question, items, top_n):
                raise RuntimeError("simulated rerank failure")

        class _KeywordReranker:
            """模拟本地关键词回退。"""

            def rerank(self, question, items, top_n):
                return items[:top_n]

        reranker = ModelFirstReranker(_FailingReranker(), _KeywordReranker())
        result = reranker.rerank("问题", hits, 2)
        self.assertEqual(result, ["甲", "乙"])
        self.assertTrue(reranker.last_mode.startswith("keyword_fallback"))


class RerankSuccessRecordsTests(unittest.TestCase):
    """验证重排序成功后熔断复位。"""

    def setUp(self) -> None:
        reset_circuit_breakers()

    def test_rerank_success_records_success(self) -> None:
        client = DashScopeRerankClient(api_key="test-key", endpoint="http://localhost/rerank")
        breaker = get_circuit_breaker("rerank")
        hits = ["甲", "乙"]
        payload = {
            "output": {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.5},
                ]
            }
        }
        with patch.object(client, "_post", return_value=_FakeRerankResponse(payload)):
            result = client.rerank("问题", hits, 2)
        self.assertEqual(result, ["乙", "甲"])
        self.assertEqual(breaker.snapshot()["failure_count"], 0)

