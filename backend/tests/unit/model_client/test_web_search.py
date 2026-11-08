"""DashScope 聊天客户端联网搜索参数测试。"""
from __future__ import annotations

import unittest

from backend.app.services.model_client.chat import (
    DashScopeChatClient,
    general_chat_system_prompt,
)


class _FakeDelta:
    """模拟流式响应中的增量内容。"""

    def __init__(self, content: str) -> None:
        self.content = content


class _FakeMessage:
    """模拟完整响应中的消息。"""

    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    """模拟单个流式或完整响应选择项。"""

    def __init__(self, content: str) -> None:
        self.delta = _FakeDelta(content)
        self.message = _FakeMessage(content)


class _FakeChunk:
    """模拟一个流式响应分片。"""

    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletion:
    """模拟非流式完整响应。"""

    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """记录请求参数并按 stream 标志返回流式或完整响应。"""

    def __init__(self) -> None:
        self.last_kwargs: dict = {}

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if kwargs.get("stream"):
            return [_FakeChunk("联网回答内容")]
        return _FakeCompletion("联网回答内容")


class _FakeChat:
    """模拟 OpenAI 兼容客户端的 chat 命名空间。"""

    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeOpenAIClient:
    """模拟 OpenAI 兼容客户端顶层对象。"""

    def __init__(self) -> None:
        self.chat = _FakeChat()


class DashScopeWebSearchTests(unittest.TestCase):
    """验证联网搜索参数只在显式开启时写入请求体。"""

    def setUp(self) -> None:
        self.client = DashScopeChatClient(
            api_key="test-key",
            model="qwen3.7-flash",
        )
        self.fake = _FakeOpenAIClient()
        self.client._openai_client = self.fake

    def test_stream_answer_with_context_does_not_search_by_default(self) -> None:
        list(self.client.stream_answer_with_context("山西暴雨", []))
        body = self.fake.chat.completions.last_kwargs.get("extra_body") or {}
        self.assertNotIn("enable_search", body)

    def test_stream_answer_with_context_enables_search(self) -> None:
        list(
            self.client.stream_answer_with_context(
                "今天重庆天气怎么样",
                [],
                enable_search=True,
                forced_search=True,
            )
        )
        body = self.fake.chat.completions.last_kwargs.get("extra_body") or {}
        self.assertEqual(body["enable_search"], True)
        self.assertEqual(body["search_options"], {"forced_search": True})

    def test_stream_general_chat_uses_web_search_without_knowledge_context(self) -> None:
        text = "".join(self.client.stream_general_chat("今天重庆天气怎么样"))
        kwargs = self.fake.chat.completions.last_kwargs
        body = kwargs.get("extra_body") or {}
        self.assertEqual(body["enable_search"], True)
        self.assertEqual(body["search_options"], {"forced_search": True})
        self.assertEqual(text, "联网回答内容")
        messages = kwargs.get("messages") or []
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "今天重庆天气怎么样")
        self.assertNotIn("历史个例材料", messages[0]["content"])

    def test_general_chat_system_prompt_is_chinese(self) -> None:
        prompt = general_chat_system_prompt()
        self.assertTrue(prompt)
        self.assertIsInstance(prompt, str)


if __name__ == "__main__":
    unittest.main()
