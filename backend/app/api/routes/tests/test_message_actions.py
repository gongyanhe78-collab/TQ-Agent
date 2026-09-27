"""消息级操作接口与会话存储的回归测试。"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.routes import sessions
from backend.app.services.session_store import SessionStore


class _PdfAgent:
    """记录消息级导出输入，避免测试调用真实 PDF 分析链路。"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.previous_result: dict = {}

    def export_pdf_from_previous_result(self, previous_result, filename="", report_title=""):
        self.previous_result = dict(previous_result)
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4\n%%EOF")
        return path


class _FeedbackNormalizer:
    """模拟反馈归一化模型，验证接口异步落库而不依赖云端。"""

    def is_available(self) -> bool:
        return True

    def normalize_feedback_guidance(self, _question, _answer, _feedback, **_kwargs) -> str:
        return (
            '{"decision":"apply",'
            '"guidance_prompt":"回答灾害过程时补充已有证据支持的影响范围。",'
            '"guidance_type":["answer_completeness"],'
            '"requested_dimensions":["影响范围"],'
            '"needs_same_scope_retrieval":true,"scope_change":false,"confidence":0.94}'
        )


class MessageActionTests(unittest.TestCase):
    """验证消息 ID 权限边界以及导出、反馈、分享和上报行为。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.store = SessionStore(self.root / "sessions.sqlite3")
        self.session_id = self.store.create_session("测试会话")["session_id"]
        self.other_session_id = self.store.create_session("其他会话")["session_id"]
        self.user_message = self.store.add_message(
            self.session_id,
            "user",
            "请分析2025年5月至7月的气象灾害",
        )
        self.assistant_message = self.store.add_message(
            self.session_id,
            "assistant",
            "页面可见的分析正文。",
            metadata={
                "agent_type": "document_rag",
                "run_id": "run-visible",
                "evidence_chunks": [{"chunk_id": "secret-chunk", "content": "隐藏证据"}],
                "audit": {"internal": True},
                "reports": ["旧版损坏项"],
                "status": "completed",
            },
        )
        self.main = SimpleNamespace(session_store=self.store)
        app = FastAPI()
        app.include_router(sessions.router)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_message_lookup_and_metadata_update_are_session_scoped(self) -> None:
        """消息编号必须同时绑定会话，元数据补丁不能覆盖已有证据。"""
        message_id = self.assistant_message["message_id"]

        self.assertIsNone(self.store.get_message(self.other_session_id, message_id))
        updated = self.store.update_message_metadata(
            self.session_id,
            message_id,
            {"reports": [{"report_id": "report-1"}]},
        )

        self.assertEqual(updated["run_id"], "run-visible")
        self.assertEqual(updated["evidence_chunks"][0]["chunk_id"], "secret-chunk")
        self.assertEqual(updated["reports"][0]["report_id"], "report-1")

    def test_feedback_is_mutually_exclusive_and_can_be_removed(self) -> None:
        """同一客户端对同一消息只保留一个反馈，none 会取消反馈。"""
        message_id = self.assistant_message["message_id"]
        self.store.set_message_feedback(self.session_id, message_id, "browser-1", "love")
        self.store.set_message_feedback(
            self.session_id,
            message_id,
            "browser-1",
            "needs_improvement",
            "遗漏影响范围",
        )
        with self.store._connect() as conn:
            rows = conn.execute(
                "SELECT rating, reason FROM message_feedback WHERE session_id = ? AND message_id = ?",
                (self.session_id, message_id),
            ).fetchall()
        self.assertEqual([(row["rating"], row["reason"]) for row in rows], [
            ("needs_improvement", "遗漏影响范围"),
        ])

        self.store.set_message_feedback(self.session_id, message_id, "browser-1", "none")
        with self.store._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM message_feedback").fetchone()[0]
        self.assertEqual(count, 0)

    def test_message_list_restores_feedback_for_current_client(self) -> None:
        """重新打开会话时只恢复当前浏览器保存的反馈和原因。"""
        message_id = self.assistant_message["message_id"]
        self.store.set_message_feedback(
            self.session_id,
            message_id,
            "browser-1",
            "needs_improvement",
            "遗漏关键信息：缺少影响范围",
        )
        self.store.set_message_feedback(
            self.session_id,
            message_id,
            "browser-2",
            "love",
        )

        with patch.object(sessions, "_main", return_value=self.main):
            response = self.client.get(
                f"/api/sessions/{self.session_id}/messages",
                params={"client_id": "browser-1"},
            )

        self.assertEqual(response.status_code, 200)
        assistant = next(item for item in response.json()["messages"] if item["role"] == "assistant")
        self.assertEqual(assistant["feedback"], "needs_improvement")
        self.assertEqual(assistant["feedback_reason"], "遗漏关键信息：缺少影响范围")

    def test_issue_and_share_snapshot_do_not_expose_internal_metadata(self) -> None:
        """问题上报独立存储，共享快照只能返回可见的一问一答。"""
        message_id = self.assistant_message["message_id"]
        issue = self.store.add_message_issue(
            self.session_id,
            message_id,
            "browser-1",
            "missing_fact",
            "遗漏关键事实",
        )
        snapshot = self.store.create_shared_snapshot(
            self.session_id,
            message_id,
            self.user_message["content"],
            self.assistant_message["content"],
            "气象灾害分析",
        )
        restored = self.store.get_shared_snapshot(snapshot["token"])

        self.assertTrue(issue["report_id"].startswith("issue_"))
        self.assertEqual(
            set(restored or {}),
            {"token", "title", "question", "answer", "created_at"},
        )
        self.assertNotIn("secret-chunk", str(restored))
        self.assertNotIn("internal", str(restored))

    def test_pdf_endpoint_uses_only_visible_answer_and_merges_report(self) -> None:
        """消息级 PDF 只能消费正文，并保存直接下载记录且不生成缩略图。"""
        fake_agent = _PdfAgent(self.root / "reports")

        with (
            patch.object(sessions, "_main", return_value=self.main),
            patch("backend.app.services.agent.case_multidim_search.router._agent", return_value=fake_agent),
        ):
            response = self.client.post(
                f"/api/sessions/{self.session_id}/messages/"
                f"{self.assistant_message['message_id']}/export-pdf"
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake_agent.previous_result, {"answer": "页面可见的分析正文。"})
        self.assertNotIn("secret-chunk", str(fake_agent.previous_result))
        report = response.json()
        self.assertTrue(report["download_only"])
        self.assertEqual(report["report_mode"], "message_export")
        self.assertNotIn("thumbnail_url", report)
        restored = self.store.get_message(self.session_id, self.assistant_message["message_id"])
        self.assertEqual(restored["run_id"], "run-visible")
        self.assertEqual(restored["reports"], [report])

    def test_message_action_endpoints_reject_cross_session_access(self) -> None:
        """不同会话不能凭全局消息编号导出、反馈或分享回答。"""
        message_id = self.assistant_message["message_id"]
        with patch.object(sessions, "_main", return_value=self.main):
            response = self.client.post(
                f"/api/sessions/{self.other_session_id}/messages/{message_id}/share"
            )
        self.assertEqual(response.status_code, 404)

class FeedbackGuidanceEndpointTests(unittest.IsolatedAsyncioTestCase):
    """验证“仅提交反馈”异步生成临时要求并按会话隔离。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = SessionStore(Path(self.temp_dir.name) / "sessions.sqlite3")
        self.session_id = self.store.create_session("反馈会话")["session_id"]
        self.user = self.store.add_message(self.session_id, "user", "分析历史暴雨过程")
        self.assistant = self.store.add_message(
            self.session_id,
            "assistant",
            "上一条回答。",
            metadata={"agent_type": "document_rag", "status": "completed"},
        )
        self.main = SimpleNamespace(
            session_store=self.store,
            llm_client=_FeedbackNormalizer(),
        )

    async def asyncTearDown(self) -> None:
        for task in list(getattr(self.main, "_feedback_guidance_tasks", set())):
            await task
        self.temp_dir.cleanup()

    async def test_feedback_is_normalized_asynchronously_and_new_session_is_empty(self) -> None:
        with patch.object(sessions, "_main", return_value=self.main):
            response = await sessions.save_message_feedback(
                self.session_id,
                self.assistant["message_id"],
                sessions.MessageFeedbackRequest(
                    rating="needs_improvement",
                    client_id="browser-1",
                    reason="遗漏关键信息：请补充影响范围",
                ),
            )
        self.assertEqual(response["guidance_status"], "pending")
        # 等待后台线程完成；提交接口本身没有等待模型调用。
        await asyncio.sleep(0.05)
        active = self.store.list_feedback_guidances(self.session_id)
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["status"], "active")
        self.assertIn("补充已有证据支持的影响范围", active[0]["guidance_prompt"])

        new_session = self.store.create_session("新会话")["session_id"]
        self.assertEqual(self.store.list_feedback_guidances(new_session), [])


if __name__ == "__main__":
    unittest.main()
