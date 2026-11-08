"""失败运行重放服务集成测试：复用原会话且不重复创建。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.app.services.query.run_replay import (
    list_failed_runs,
    replay_failed_run,
)
from backend.app.services.session_store import SessionStore


class _RecordingStream:
    """记录收到的请求并回放固定事件。"""

    def __init__(self) -> None:
        self.requests: list = []
        self.fail = False

    async def __call__(self, request, _is_disconnected=None):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("模拟重放失败")
        yield {"type": "delta", "text": "重放回答"}
        yield {"type": "done", "status": "completed"}


@pytest.mark.asyncio
class FailedRunReplayTests(unittest.TestCase):
    """验证失败运行查询与重放不创建新会话。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.store = SessionStore(root / "sessions.sqlite3")
        self.session_id = self.store.create_session(
            title="重放测试会话"
        )["session_id"]
        self.store.add_message(self.session_id, "user", "山西暴雨过程分析")
        self.store.start_agent_run(
            "run-failed-1",
            self.session_id,
            "document_rag",
            "山西暴雨过程分析",
            status="failed",
        )
        self.store.finish_agent_run(
            "run-failed-1",
            status="failed",
            answer="",
            metadata={"error": "模拟失败"},
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_list_failed_runs_returns_failed_only(self) -> None:
        async def scenario() -> None:
            runs = await list_failed_runs(self.store)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["run_id"], "run-failed-1")
            self.assertEqual(runs[0]["session_id"], self.session_id)

        self._run_async(scenario)

    def test_replay_reuses_session_and_reports_status(self) -> None:
        async def scenario() -> None:
            stream = _RecordingStream()
            result = await replay_failed_run(
                self.store,
                "run-failed-1",
                stream_fn=stream,
            )
            self.assertFalse(result["skipped"])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["session_id"], self.session_id)
            self.assertEqual(len(stream.requests), 1)
            # 重放请求复用原会话，不新建会话。
            self.assertEqual(stream.requests[0].session_id, self.session_id)
            sessions = self.store.list_sessions()
            self.assertEqual(len(sessions), 1)

        self._run_async(scenario)

    def test_replay_missing_run_is_skipped(self) -> None:
        async def scenario() -> None:
            stream = _RecordingStream()
            result = await replay_failed_run(
                self.store,
                "run-missing",
                stream_fn=stream,
            )
            self.assertTrue(result["skipped"])
            self.assertEqual(result["reason"], "运行不存在")
            self.assertEqual(len(stream.requests), 0)

        self._run_async(scenario)

    def test_replay_non_failed_run_is_skipped(self) -> None:
        async def scenario() -> None:
            self.store.start_agent_run(
                "run-ok-1",
                self.session_id,
                "document_rag",
                "问题",
                status="completed",
            )
            stream = _RecordingStream()
            result = await replay_failed_run(
                self.store,
                "run-ok-1",
                stream_fn=stream,
            )
            self.assertTrue(result["skipped"])
            self.assertIn("不是 failed", result["reason"])

        self._run_async(scenario)

    def test_replay_missing_session_is_skipped(self) -> None:
        async def scenario() -> None:
            self.store.start_agent_run(
                "run-orphan-1",
                self.session_id,
                "document_rag",
                "问题",
                status="failed",
            )
            stream = _RecordingStream()
            # 模拟运行存在但会话已被外部清理的防御分支。
            with patch.object(
                self.store,
                "get_session",
                return_value=None,
            ):
                result = await replay_failed_run(
                    self.store,
                    "run-orphan-1",
                    stream_fn=stream,
                )
            self.assertTrue(result["skipped"])
            self.assertEqual(result["reason"], "会话不存在")

        self._run_async(scenario)

    def test_replay_stream_error_is_reported(self) -> None:
        async def scenario() -> None:
            stream = _RecordingStream()
            stream.fail = True
            result = await replay_failed_run(
                self.store,
                "run-failed-1",
                stream_fn=stream,
            )
            self.assertFalse(result["skipped"])
            self.assertEqual(result["status"], "failed")
            self.assertIn("模拟重放失败", result["error"])

        self._run_async(scenario)

    def _run_async(self, scenario) -> None:
        import asyncio

        asyncio.run(scenario())
