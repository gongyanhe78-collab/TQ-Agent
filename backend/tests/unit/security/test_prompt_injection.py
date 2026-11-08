"""提示词注入防护的回归测试：系统提示词必须禁止执行材料内指令。"""
from __future__ import annotations

import unittest

from backend.app.services.model_client.chat import (
    DashScopeChatClient,
    general_chat_system_prompt,
)

GUARD_MARKERS = ("一律不得执行", "不得泄露", "系统提示词")


class PromptInjectionGuardTests(unittest.TestCase):
    """验证开放域与 RAG 提示词都包含防注入指令。"""

    def test_general_chat_prompt_contains_injection_guard(self) -> None:
        prompt = general_chat_system_prompt()
        for marker in GUARD_MARKERS:
            self.assertIn(marker, prompt)

    def test_answer_messages_prompt_contains_injection_guard(self) -> None:
        client = DashScopeChatClient(
            api_key="test-key",
            base_url="http://localhost:1",
            model="test-model",
        )
        messages = client._answer_messages(  # noqa: SLF001
            question="测试问题",
            context_blocks=["测试材料"],
        )
        system_prompt = messages[0]["content"]
        for marker in GUARD_MARKERS:
            self.assertIn(marker, system_prompt)
        self.assertEqual(messages[0]["role"], "system")


if __name__ == "__main__":
    unittest.main()
