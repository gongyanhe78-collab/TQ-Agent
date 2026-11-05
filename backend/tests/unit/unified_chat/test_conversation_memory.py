"""统一聊天分层记忆的隔离与压缩测试。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.services.agent.unified_chat.conversation_memory import ConversationMemoryManager
from backend.app.services.session_store import SessionStore


class ConversationMemoryTests(unittest.TestCase):
    """验证记忆只在当前 session_id 内生效，并保留最近轮次细节。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = SessionStore(Path(self.temp_dir.name) / "sessions.sqlite3")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _append_turn(self, session_id: str, index: int) -> None:
        """写入带有可追溯个例实体的一轮测试消息。"""
        self.store.add_message(session_id, "user", f"第{index}轮问题" + "天气过程" * 80)
        self.store.add_message(
            session_id,
            "assistant",
            f"第{index}轮回答" + "实况环流成因预报复盘" * 100,
            metadata={
                "agent_type": "document_rag",
                "run_id": f"run-{index}",
                "evidence_cases": [{"case_id": f"case-{index}", "title": f"{index}月天气过程"}],
            },
        )

    def test_memory_is_isolated_by_session(self) -> None:
        first = self.store.create_session()["session_id"]
        second = self.store.create_session()["session_id"]
        self._append_turn(first, 1)
        manager = ConversationMemoryManager(self.store)

        first_context = manager.refresh(first)
        second_context = manager.refresh(second)

        self.assertEqual(first_context["previous_turn"]["case_refs"][0]["case_id"], "case-1")
        self.assertEqual(second_context["previous_turn"], {})
        self.assertEqual(second_context["available_case_refs"], [])

    def test_old_turns_are_compressed_but_previous_turn_stays_detailed(self) -> None:
        session_id = self.store.create_session()["session_id"]
        manager = ConversationMemoryManager(self.store, token_budget=1200, recent_turn_limit=3)
        for index in range(1, 8):
            self._append_turn(session_id, index)

        context = manager.refresh(session_id)

        self.assertTrue(context["budget"]["compressed"])
        self.assertTrue(context["long_term_summary"])
        self.assertEqual(context["previous_turn"]["run_id"], "run-7")
        self.assertEqual(context["previous_turn"]["case_refs"][0]["case_id"], "case-7")
        self.assertLessEqual(len(context["recent_turns"]), 2)

    def test_invalid_case_ids_are_removed_against_current_knowledge(self) -> None:
        """记忆只保留当前主库真实存在的个例，旧版 jan-* 和 all-* 引用必须被清理。"""
        session_id = self.store.create_session()["session_id"]
        self.store.add_message(session_id, "user", "列出一月个例")
        self.store.add_message(
            session_id,
            "assistant",
            "包含一个有效个例和两个旧版引用。",
            metadata={
                "agent_type": "document_rag",
                "evidence_cases": [
                    {"case_id": "valid-case", "title": "1月23-26日雨雪寒潮大风天气过程"},
                    {"case_id": "jan-2", "title": "旧版引用"},
                    {"case_id": "all-15", "title": "旧版聚合引用"},
                ],
            },
        )
        manager = ConversationMemoryManager(
            self.store,
            case_provider=lambda: [{"case_id": "valid-case", "title": "1月23-26日雨雪寒潮大风天气过程"}],
        )

        context = manager.refresh(session_id)

        self.assertEqual([item["case_id"] for item in context["available_case_refs"]], ["valid-case"])
        self.assertEqual(context["previous_turn"]["reference_type"], "knowledge_case")
        self.assertTrue(context["knowledge_version"])

    def test_context_before_message_excludes_old_answer_and_later_turns(self) -> None:
        """重新生成的临时记忆只保留旧回答之前已经完成的上下文。"""
        session_id = self.store.create_session()["session_id"]
        self._append_turn(session_id, 1)
        self.store.add_message(session_id, "user", "需要重新回答的原问题")
        old_answer = self.store.add_message(
            session_id,
            "assistant",
            "这是一条必须排除的旧回答",
            metadata={"agent_type": "document_rag", "run_id": "old-answer"},
        )
        self._append_turn(session_id, 3)
        manager = ConversationMemoryManager(self.store)

        context = manager.context_before_message(session_id, old_answer["message_id"])

        serialized = str(context)
        self.assertIn("第1轮回答", serialized)
        self.assertNotIn("必须排除的旧回答", serialized)
        self.assertNotIn("第3轮回答", serialized)


if __name__ == "__main__":
    unittest.main()
