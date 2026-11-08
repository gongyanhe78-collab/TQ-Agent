"""验证 Agent Runtime 与 SSE 路由、会话运行元数据的端到端契约。"""
from __future__ import annotations

import asyncio
import importlib
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.agents.unified_chat.orchestration_models import (
    AgentBudget,
    ExecutionContext,
)
from backend.app.agents.unified_chat.orchestrator import (
    UnifiedChatOrchestrator,
)
from backend.app.agents.unified_chat.performance import PerformanceTrace
from backend.app.agents.unified_chat.persistence import ResultPersistenceMixin
from backend.app.agents.unified_chat.stream_events import StreamEventMixin
from backend.app.api.routes import agent
from backend.app.observability.middleware import ObservabilityMiddleware
from backend.app.services.query.chat_service import ChatService
from backend.app.services.session_store import SessionStore


def _runtime_class():
    """延迟导入运行时，确保 RED 阶段失败原因指向功能缺失。"""
    module = importlib.import_module(
        "backend.app.agents.unified_chat.runtime"
    )
    return module.AgentRuntime


def _auth_services(chat_service) -> SimpleNamespace:
    """构造查询路由依赖的最小认证服务组合。"""
    return SimpleNamespace(
        auth_service=SimpleNamespace(
            get_current_user=lambda token: {
                "user_id": "runtime-user",
                "role": "user",
                "token": token,
            }
        ),
        chat_service_for=lambda user_id: chat_service,
    )


def _make_client(chat_service) -> TestClient:
    """挂载真实查询路由和观测中间件，验证实际 SSE 边界。"""
    app = FastAPI()
    app.state.services = _auth_services(chat_service)
    app.include_router(agent.router)
    app.add_middleware(ObservabilityMiddleware)
    return TestClient(
        app,
        headers={"Authorization": "Bearer runtime-token"},
    )


class _LegacyChatService:
    """保留旧两参数 stream 签名，验证路由兼容性。"""

    async def stream(self, _payload, _is_disconnected):
        yield {"type": "delta", "text": "旧接口仍可用"}


class _ContextChatService:
    """接收执行上下文并记录路由关闭时设置的取消事件。"""

    def __init__(self) -> None:
        self.execution_context = None

    async def stream(
        self,
        _payload,
        _is_disconnected,
        execution_context,
    ):
        self.execution_context = execution_context
        yield {"type": "delta", "text": "上下文已传入"}


class _SessionStoreRecorder:
    """只提供无效会话判断，并阻止意外创建运行记录。"""

    def __init__(self) -> None:
        self.start_calls = 0

    def get_session(self, _session_id):
        return None

    def start_agent_run(self, *_args, **_kwargs):
        self.start_calls += 1
        raise AssertionError("非法会话不得创建 agent_runs")


class _MinimalIntentAgent:
    """取消发生前不会被调用，只提供编排器构造所需属性。"""

    case_provider = None


def _minimal_orchestrator(runtime) -> UnifiedChatOrchestrator:
    """构造可在首次运行时检查处终止的最小编排器。"""
    dependencies = SimpleNamespace(
        session_store=_SessionStoreRecorder(),
        document_qa=SimpleNamespace(document_store=object()),
        structured_query=object(),
        context_cases=object(),
        presenter=object(),
        standard_case_store=object(),
        llm_client=object(),
        multidim_agent=object(),
        multidim_store=object(),
        smart_agent=object(),
        intent_agent=_MinimalIntentAgent(),
    )
    return UnifiedChatOrchestrator(dependencies, runtime=runtime)


class _Settings:
    """提供 ChatService 创建预算和运行时限所需的配置。"""

    agent_max_seconds = 30
    agent_max_steps = 5
    agent_max_model_calls = 3
    agent_token_budget = 100
    agent_step_timeout_seconds = 10


class _NoopOrchestrator:
    """记录 ChatService 注入的运行时。"""

    def __init__(self, runtime=None) -> None:
        self.runtime = runtime

    async def stream(self, _payload, _is_disconnected):
        yield {
            "type": "metadata",
            "request_id": self.runtime.execution_context.request_id,
            "trace_id": self.runtime.execution_context.trace_id,
            "termination_reason": "",
            "subtask_status": [],
        }


def test_stream_route_keeps_legacy_two_argument_chat_service() -> None:
    """新增执行上下文不能破坏旧 ChatService 的两参数签名。"""
    client = _make_client(_LegacyChatService())

    try:
        response = client.post(
            "/api/query/stream",
            json={"question": "兼容旧签名"},
        )
    finally:
        client.close()

    assert response.status_code == 200
    assert "旧接口仍可用" in response.text


def test_stream_route_passes_context_and_sets_cancel_on_close() -> None:
    """真实服务支持第三参数时，路由应传入关联上下文并在结束时取消。"""
    chat_service = _ContextChatService()
    client = _make_client(chat_service)

    try:
        response = client.post(
            "/api/query/stream",
            json={"question": "验证请求上下文", "session_id": "session-1"},
        )
    finally:
        client.close()

    assert response.status_code == 200
    execution_context = chat_service.execution_context
    assert isinstance(execution_context, ExecutionContext)
    assert execution_context.request_id == response.headers["X-Request-ID"]
    assert execution_context.trace_id == response.headers["X-Trace-ID"]
    assert execution_context.session_id == "session-1"
    assert execution_context.cancel_event is not None
    assert execution_context.cancel_event.is_set()


def test_cancelled_client_terminates_before_route_without_model_call() -> None:
    """已取消请求不得进入意图路由或增加模型调用计数。"""

    async def scenario():
        cancel_event = asyncio.Event()
        cancel_event.set()
        runtime_class = _runtime_class()
        runtime = runtime_class(
            AgentBudget(max_steps=5, max_model_calls=5),
            ExecutionContext(
                request_id="request-cancel-1",
                trace_id="b" * 32,
                cancel_event=cancel_event,
            ),
        )
        orchestrator = _minimal_orchestrator(runtime)
        payload = SimpleNamespace(
            question="分析暴雨过程",
            session_id="",
            top_k=5,
            top_n=3,
        )
        return [
            event
            async for event in orchestrator.stream(payload)
        ], runtime

    events, runtime = asyncio.run(scenario())

    assert events[0]["request_id"] == "request-cancel-1"
    assert events[0]["trace_id"] == "b" * 32
    assert events[-1]["type"] == "error"
    assert events[-1]["termination_reason"] == "cancelled_by_client"
    assert runtime.model_calls == 0


def test_invalid_session_returns_runtime_fields_without_agent_run() -> None:
    """非法会话必须返回结构化错误，且不能创建运行记录。"""
    payload = SimpleNamespace(
        question="分析暴雨过程",
        session_id="missing-session",
        top_k=5,
        top_n=3,
    )
    runtime_class = _runtime_class()
    runtime = runtime_class(
        AgentBudget(),
        ExecutionContext(
            request_id="request-invalid-1",
            trace_id="c" * 32,
        ),
    )
    orchestrator = _minimal_orchestrator(runtime)

    async def scenario():
        return [
            event
            async for event in orchestrator.stream(payload)
        ]

    events = asyncio.run(scenario())

    assert events[-1]["type"] == "error"
    assert events[-1]["request_id"] == "request-invalid-1"
    assert events[-1]["trace_id"] == "c" * 32
    assert events[-1]["termination_reason"] == "invalid_session"
    assert orchestrator.session_store.start_calls == 0


def test_stream_event_helpers_add_runtime_fields_without_renaming() -> None:
    """首轮、完成 metadata 和 done 只扩展四个运行时字段。"""
    runtime_class = _runtime_class()
    runtime = runtime_class(
        AgentBudget(),
        ExecutionContext(
            request_id="request-events-1",
            trace_id="d" * 32,
        ),
    )

    class _Builder(StreamEventMixin):
        """使用真实事件构造逻辑的最小宿主。"""

    builder = _Builder()
    builder.runtime = runtime
    performance = PerformanceTrace()
    first = builder._progress_metadata(
        "query_intent",
        "",
        "understanding",
        2,
        "正在理解问题",
        performance,
    )
    common = {
        "agent_type": "document_rag",
        "run_id": "run-events-1",
        "message_id": 42,
        "performance": performance.snapshot(),
    }
    runtime.finish_subtask()
    runtime.mark_completed()
    completion_metadata = builder._completion_metadata(
        common,
        status="completed",
    )
    done = builder._done(common, status="completed")

    required = {
        "request_id",
        "trace_id",
        "termination_reason",
        "subtask_status",
    }
    for event in (first, completion_metadata, done):
        assert required <= set(event)
        assert event["request_id"] == "request-events-1"
        assert event["trace_id"] == "d" * 32
    assert completion_metadata["type"] == "metadata"
    assert completion_metadata["message_id"] == 42
    assert done["type"] == "done"
    assert done["status"] == "completed"
    assert done["termination_reason"] == "completed"


def test_chat_service_injects_request_runtime_into_new_factory() -> None:
    """ChatService 应创建带截止时间的运行时并传给支持它的工厂。"""
    captured = {}

    def factory(runtime):
        captured["runtime"] = runtime
        return _NoopOrchestrator(runtime)

    service = ChatService(
        settings=_Settings(),
        session_store=object(),
        document_qa=object(),
        structured_query=object(),
        context_cases=object(),
        agent_query=object(),
        presenter=object(),
        orchestrator_factory=factory,
    )
    cancel_event = asyncio.Event()
    execution_context = ExecutionContext(
        request_id="request-chat-service-1",
        trace_id="e" * 32,
        cancel_event=cancel_event,
    )
    payload = SimpleNamespace(
        question="测试工厂注入",
        session_id="session-chat-1",
    )

    async def scenario():
        return [
            event
            async for event in service.stream(
                payload,
                lambda: asyncio.sleep(0, result=False),
                execution_context,
            )
        ]

    events = asyncio.run(scenario())

    runtime = captured["runtime"]
    assert runtime.execution_context.deadline is not None
    assert runtime.execution_context.session_id == "session-chat-1"
    assert events[0]["request_id"] == "request-chat-service-1"
    assert events[0]["trace_id"] == "e" * 32


def test_chat_service_keeps_zero_argument_factory_compatibility() -> None:
    """旧工厂不接受运行时参数时仍应正常创建编排器。"""
    service = ChatService(
        settings=_Settings(),
        session_store=object(),
        document_qa=object(),
        structured_query=object(),
        context_cases=object(),
        agent_query=object(),
        presenter=object(),
        orchestrator_factory=lambda: _NoopOrchestrator(
            _runtime_class()(
                AgentBudget(),
                ExecutionContext(
                    request_id="request-zero-factory",
                    trace_id="f" * 32,
                ),
            )
        ),
    )
    payload = SimpleNamespace(question="旧工厂", session_id="")

    async def scenario():
        return [
            event
            async for event in service.stream(
                payload,
                lambda: asyncio.sleep(0, result=False),
            )
        ]

    events = asyncio.run(scenario())

    assert events[0]["request_id"] == "request-zero-factory"


def test_persist_complete_writes_runtime_snapshot_to_agent_run() -> None:
    """完成落库时必须收口子任务并写入预算、Token 和终止快照。"""
    runtime_class = _runtime_class()
    runtime = runtime_class(
        AgentBudget(
            max_seconds=30,
            max_steps=2,
            max_model_calls=3,
            token_budget=100,
            step_timeout_seconds=10,
        ),
        ExecutionContext(
            request_id="request-persist-1",
            trace_id="9" * 32,
        ),
    )
    runtime.begin_subtask("route_rag", "rag")
    runtime.count_step()
    runtime.count_tokens("气象")

    with tempfile.TemporaryDirectory() as temp_dir:
        store = SessionStore(Path(temp_dir) / "sessions.sqlite3")
        session_id = store.create_session("运行快照")["session_id"]

        class _PersistenceBuilder(
            ResultPersistenceMixin,
            StreamEventMixin,
        ):
            """组合落库和运行时快照能力的最小宿主。"""

        builder = _PersistenceBuilder()
        builder.runtime = runtime
        builder.session_store = store
        builder._regeneration_context = {}
        common = {
            "agent_type": "document_rag",
            "run_id": "run-persist-1",
            "question": "气象快照",
            "answer": "回答",
        }
        builder._persist_complete(
            session_id,
            common,
            status="completed",
        )
        saved = store.get_agent_run("run-persist-1")

    metadata = saved["metadata"]
    assert metadata["request_id"] == "request-persist-1"
    assert metadata["trace_id"] == "9" * 32
    assert metadata["termination_reason"] == "completed"
    assert metadata["budget"]["max_steps"] == 2
    assert metadata["token_stats"]["total"] == 2
    assert metadata["token_usage_source"] == "estimated"
    assert metadata["subtask_status"][0]["status"] == "completed"


def test_sqlite_agent_run_migrates_and_persists_trace_id() -> None:
    """旧 SQLite 运行表应自动补 trace_id 列并在开始与完成时写入。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "sessions.sqlite3"
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                """
                CREATE TABLE agent_runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL DEFAULT '',
                    evidence_cases TEXT NOT NULL DEFAULT '[]',
                    evidence_chunks TEXT NOT NULL DEFAULT '[]',
                    images TEXT NOT NULL DEFAULT '[]',
                    reports TEXT NOT NULL DEFAULT '[]',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
        finally:
            # Windows 下连接未显式关闭会锁住临时数据库，掩盖真实迁移结果。
            connection.close()
        store = SessionStore(path)
        session_id = store.create_session("运行迁移")["session_id"]
        metadata = {
            "request_id": "request-sqlite-1",
            "trace_id": "1" * 32,
            "termination_reason": "completed",
            "subtask_status": [],
            "budget": {"max_steps": 5},
            "token_stats": {"total": 3, "input": 2, "output": 1},
            "token_usage_source": "estimated",
        }

        run = store.start_agent_run(
            "run-sqlite-1",
            session_id,
            "document_rag",
            "写入运行元数据",
            metadata=metadata,
        )
        store.finish_agent_run(
            "run-sqlite-1",
            status="completed",
            answer="回答",
            metadata=metadata,
        )
        saved = store.get_agent_run("run-sqlite-1")

    assert run["trace_id"] == "1" * 32
    assert saved["trace_id"] == "1" * 32
    assert saved["metadata"]["request_id"] == "request-sqlite-1"
    assert saved["metadata"]["termination_reason"] == "completed"
    assert saved["metadata"]["budget"]["max_steps"] == 5
    assert saved["metadata"]["token_stats"]["total"] == 3
