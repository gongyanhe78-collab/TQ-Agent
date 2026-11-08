"""查询取消时运行收口为 cancelled 并标记 partial 的测试。"""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend.app.agents.unified_chat.orchestrator import (
    OrchestratorDependencies,
    UnifiedChatOrchestrator,
)
from backend.app.agents.unified_chat.persistence import (
    ResultPersistenceMixin,
)
from backend.app.agents.unified_chat.stream_events import (
    StreamEventMixin,
)
from backend.app.services.session_store import SessionStore


class _StandardCaseStore:
    """提供最小标准个例集合，供意图 Agent 使用。"""

    def list_cases(self):
        return []


class _SlowGeneralChatLlm:
    """模拟有停顿的联网聊天流，便于中途取消。"""

    def is_available(self) -> bool:
        return True

    def stream_general_chat(
        self,
        question,
        *,
        system_prompt="",
        enable_search=True,
        forced_search=True,
    ):
        yield "第一段"
        time.sleep(0.3)
        yield "第二段"


class _GeneralChatIntentAgent:
    """模拟意图模型把开放域问题路由到普通聊天。"""

    case_provider = None

    @staticmethod
    def route(_request):
        return SimpleNamespace(
            intent="general_chat",
            rag_strategy="vector",
            normalized_message="今天重庆天气怎么样",
            clarification_question="",
            referenced_case_ids=[],
            conditions={},
            audit_id="intent-general",
            context_related=False,
            relation_type="standalone",
            reference_type="none",
            reference_clues=[],
            context_confidence=1.0,
            context_resolution_source="llm",
            context_message_ids=[],
            routing_source="llm",
            reason="开放域实时天气查询",
        )


class _Stub:
    """为未参与通用聊天的依赖提供空实现。"""

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def pending_proposal(self, _session_id):
        return None


class _PersistenceHost(StreamEventMixin, ResultPersistenceMixin):
    """组合持久化与事件混入，为测试提供运行时元数据合并能力。"""


class CancelPersistenceTests(unittest.TestCase):
    """验证取消时把运行收口为 cancelled 并写入 partial 标记。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.store = SessionStore(root / "sessions.sqlite3")
        self.session_id = self.store.create_session()["session_id"]

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_cancelled_run_is_persisted_with_partial_flag(self) -> None:
        host = _PersistenceHost()
        host.session_store = self.store
        host.runtime = None
        host._start_persisted_run(
            "run-cancel-1",
            self.session_id,
            "general_chat",
            "今天重庆天气怎么样",
            status="running",
        )
        host._finish_active_run(
            status="cancelled",
            error="客户端断开或请求取消",
        )
        run = self.store.get_agent_run("run-cancel-1")
        self.assertEqual(run["status"], "cancelled")
        self.assertIs(run["metadata"]["partial"], True)
        self.assertEqual(
            run["metadata"]["termination_reason"],
            "客户端断开或请求取消",
        )


class OrchestratorCancelTests(unittest.IsolatedAsyncioTestCase):
    """验证流中途断开会把通用聊天运行收口为 cancelled。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.session_store = SessionStore(root / "sessions.sqlite3")
        self.session_id = self.session_store.create_session()["session_id"]
        self.dependencies = OrchestratorDependencies(
            session_store=self.session_store,
            document_qa=_Stub(document_store=object()),
            structured_query=_Stub(),
            context_cases=_Stub(),
            presenter=_Stub(),
            standard_case_store=_StandardCaseStore(),
            llm_client=_SlowGeneralChatLlm(),
            multidim_agent=_Stub(output_dir=root / "reports"),
            multidim_store=_Stub(),
            smart_agent=_Stub(),
            intent_agent=_GeneralChatIntentAgent(),
            memory_service=None,
        )
        self.orchestrator = UnifiedChatOrchestrator(
            self.dependencies
        )

    async def asyncTearDown(self) -> None:
        await self.orchestrator.wait_for_background_tasks()
        self.temp_dir.cleanup()

    async def test_cancel_mid_stream_marks_run_cancelled(self) -> None:
        payload = SimpleNamespace(
            question="今天重庆天气怎么样",
            session_id=self.session_id,
            top_k=10,
            top_n=5,
        )
        stream = self.orchestrator.stream(payload)
        first_delta_seen = False
        async for event in stream:
            if event.get("type") == "delta":
                first_delta_seen = True
                break
        self.assertTrue(first_delta_seen)
        # 模拟客户端断开：关闭异步生成器触发取消信号。
        await stream.aclose()
        runs = self.session_store.list_agent_runs(self.session_id)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "cancelled")
        self.assertIs(runs[0]["metadata"]["partial"], True)
