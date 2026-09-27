"""查询向量与重排结果缓存测试。"""
from __future__ import annotations

import unittest

from backend.app.services.model_client.chat import DashScopeChatClient, get_intent_llm_client
from backend.app.services.model_client.embedding import DashScopeEmbeddingClient
from backend.app.services.model_client.rerank import DashScopeRerankClient


class _CountingEmbeddingClient(DashScopeEmbeddingClient):
    """用固定向量替代外部接口并记录调用次数。"""

    def __init__(self) -> None:
        super().__init__(api_key="test-key", model="test-model")
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [[float(len(text)), 1.0] for text in texts]


class _Response:
    """模拟 httpx 响应。"""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"output": {"results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.8},
        ]}}


class _CountingHttpClient:
    """记录云端重排请求次数。"""

    def __init__(self) -> None:
        self.calls = 0

    def post(self, *_args, **_kwargs):
        self.calls += 1
        return _Response()


class _ChatCompletionResponse:
    """模拟 OpenAI 兼容接口的聊天响应。"""

    class _Message:
        content = '{"intent":"rag"}'

    class _Choice:
        message = None

    def __init__(self) -> None:
        choice = self._Choice()
        choice.message = self._Message()
        self.choices = [choice]


class _CapturingChatCompletions:
    """记录聊天请求参数，验证意图专用参数不会丢失。"""

    def __init__(self) -> None:
        self.options = {}

    def create(self, **kwargs):
        self.options = kwargs
        return _ChatCompletionResponse()


class _CapturingOpenAIClient:
    """提供测试所需的 chat.completions 结构。"""

    def __init__(self) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = _CapturingChatCompletions()


class ModelClientCacheTests(unittest.TestCase):
    """重复的完全相同输入应避免再次访问云端接口。"""

    def test_stream_answer_prompt_allows_natural_closing_without_follow_up_call(self) -> None:
        """主 RAG 提示应包含自然承接语规则，而非依赖第二次模型调用。"""
        client = DashScopeChatClient(api_key="test-key", model="test-model")

        with_closing = client._answer_messages("分析暴雨过程", ["过程证据"], include_closing=True)
        without_closing = client._answer_messages("分析暴雨过程", ["过程证据"])

        self.assertIn("自然、简短的承接语", with_closing[0]["content"])
        self.assertIn("不要使用‘接下来你可以继续了解’", with_closing[0]["content"])
        self.assertNotIn("自然、简短的承接语", without_closing[0]["content"])

    def test_query_embedding_uses_lru_cache(self) -> None:
        client = _CountingEmbeddingClient()

        first = client.embed_query("山西暴雨")
        second = client.embed_query("山西暴雨")

        self.assertEqual(first, second)
        self.assertEqual(client.calls, 1)

    def test_rerank_uses_lru_cache(self) -> None:
        http_client = _CountingHttpClient()
        client = DashScopeRerankClient(
            api_key="test-key",
            endpoint="https://example.invalid/rerank",
            http_client=http_client,
        )
        hits = ["第一个片段", "第二个片段"]

        first = client.rerank("暴雨", hits, 2)
        second = client.rerank("暴雨", hits, 2)

        self.assertEqual(first, ["第二个片段", "第一个片段"])
        self.assertEqual(second, first)
        self.assertEqual(http_client.calls, 1)

    def test_intent_client_disables_thinking_without_changing_default_client(self) -> None:
        intent_client = get_intent_llm_client()
        default_client = DashScopeChatClient(api_key="test-key", model="test-model")
        capture = _CapturingOpenAIClient()
        intent_client._openai_client = capture

        answer = intent_client.answer_with_context(
            "只做意图判断", ["短上下文"], timeout_seconds=15, json_mode=True,
        )

        self.assertEqual(answer, '{"intent":"rag"}')
        self.assertEqual(capture.chat.completions.options["extra_body"], {"enable_thinking": False})
        self.assertEqual(capture.chat.completions.options["response_format"], {"type": "json_object"})
        self.assertEqual(intent_client.max_retries, 0)
        self.assertIsNone(default_client.enable_thinking)
        self.assertIsNone(default_client.max_retries)


if __name__ == "__main__":
    unittest.main()
