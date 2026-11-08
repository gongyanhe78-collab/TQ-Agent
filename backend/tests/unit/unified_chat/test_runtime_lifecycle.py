"""验证 Task 4 各业务流的 Agent 运行生命周期契约。"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.agents.unified_chat.multidim_stream import (
    MultidimStreamMixin,
)
from backend.app.agents.unified_chat.orchestration_models import (
    AgentBudget,
    BudgetTermination,
    ExecutionContext,
    OrchestratorDependencies,
    TerminationReason,
    UnifiedRouteDecision,
)
from backend.app.agents.unified_chat.orchestrator import (
    UnifiedChatOrchestrator,
)
from backend.app.agents.unified_chat.performance import PerformanceTrace
from backend.app.agents.unified_chat.persistence import (
    ResultPersistenceMixin,
)
from backend.app.agents.unified_chat.rag_stream import RagStreamMixin
from backend.app.agents.unified_chat.report_stream import (
    ReportStreamMixin,
)
from backend.app.agents.unified_chat.smart_stream import SmartStreamMixin
from backend.app.agents.unified_chat.stream_events import StreamEventMixin
from backend.app.db.base import Base
from backend.app.db.models import User
from backend.app.db.session_repository import SqlAlchemySessionStore
from backend.app.services.session_store import SessionStore


_RUNTIME_FIELDS = {
    "request_id",
    "trace_id",
    "termination_reason",
    "subtask_status",
}


class _LifecycleBuilder(
    RagStreamMixin,
    MultidimStreamMixin,
    SmartStreamMixin,
    ReportStreamMixin,
    ResultPersistenceMixin,
    StreamEventMixin,
):
    """组合真实业务流和真实落库能力的最小宿主。"""

    @staticmethod
    def _conversation_report_title(_previous_result):
        """为承接式报告提供不依赖完整路由宿主的标题。"""
        return "气象灾害分析报告"


class _StaticRouteOrchestrator(UnifiedChatOrchestrator):
    """固定返回同一路由，隔离意图模型对生命周期测试的影响。"""

    def __init__(self, dependencies, runtime, decision) -> None:
        self._fixed_decision = decision
        super().__init__(dependencies, runtime=runtime)

    def _route(self, _question, _has_pending, _memory_context):
        """返回测试指定的统一路由决策。"""
        return self._fixed_decision


class _ExplodingSmartAgent:
    """接受请求后抛出普通异常，验证由统一编排器负责收口。"""

    async def stream_natural_match(self, _request, _is_disconnected=None):
        """先发出 accepted，再模拟 Smart 内部异常。"""
        yield "accepted", {
            "run_id": "smart-runtime-failed",
            "stage": "parse",
            "percent": 2,
        }
        raise RuntimeError("Smart 内部异常")


class _ErrorSmartAgent:
    """通过业务 error 事件结束 Smart 流。"""

    async def stream_natural_match(self, _request, _is_disconnected=None):
        """返回 accepted 和 error，不抛 Python 异常。"""
        yield "accepted", {
            "run_id": "smart-runtime-error",
            "stage": "parse",
            "percent": 2,
        }
        yield "error", {"message": "相似个例匹配失败"}


class _BudgetExhaustedLlm:
    """在正文流输出时触发请求 Token 预算终止。"""

    @staticmethod
    def is_available() -> bool:
        """声明正文模型可用。"""
        return True

    def stream_answer_with_context(self, *_args, **_kwargs):
        """返回足够长的中文片段，使测试预算在消费时越界。"""
        yield "这是一段会超过请求Token预算的模型回答"


class _BudgetMultidimAgent:
    """在多维分析线程内耗尽模型调用预算。"""

    def __init__(self, runtime) -> None:
        self.runtime = runtime

    def analyze_structured(self, _request):
        """模拟首次业务模型调用越过预算边界。"""
        return self.runtime.count_model_call()


class _BudgetReportAgent:
    """在承接式 PDF 导出线程内耗尽模型调用预算。"""

    def __init__(self, runtime) -> None:
        self.runtime = runtime

    def export_pdf_from_previous_result(self, *_args, **_kwargs):
        """第一次调用占用最后一个名额，第二次调用触发终止。"""
        self.runtime.count_model_call()
        return self.runtime.count_model_call()


class _RaisingMultidimStore:
    """为确认动作提供固定请求。"""

    @staticmethod
    def confirm(_session_id, _proposal_id, _run_id):
        """返回可直接落库的确认条件。"""
        request = SimpleNamespace(
            model_dump=lambda: {"years": [2025], "months": [3]},
        )
        return request, "统计2025年3月个例并生成报告"


def _dependencies(store, *, smart_agent=None, llm_client=None):
    """构造统一编排器所需的最小依赖组合。"""
    structured_result = {
        "answer": "2025年3月共有3个灾害天气个例。",
        "retrieval_mode": "structured_cases:statistics",
        "hit_count": 3,
        "evidence_cases": [],
        "evidence_chunks": [],
        "images": [],
        "visuals": [],
        "llm_used": False,
    }
    return OrchestratorDependencies(
        session_store=store,
        document_qa=SimpleNamespace(document_store=object()),
        structured_query=SimpleNamespace(
            answer=lambda *_args, **_kwargs: dict(structured_result),
            needs_metric_documents=lambda _result: False,
            metric_document_result=lambda *_args, **_kwargs: None,
        ),
        context_cases=SimpleNamespace(
            from_messages=lambda *_args, **_kwargs: [],
        ),
        presenter=object(),
        standard_case_store=SimpleNamespace(list_cases=lambda: []),
        llm_client=llm_client or _BudgetExhaustedLlm(),
        multidim_agent=object(),
        multidim_store=SimpleNamespace(
            pending_proposal=lambda _session_id: None,
        ),
        smart_agent=smart_agent or object(),
        intent_agent=SimpleNamespace(case_provider=lambda: []),
    )


def _orchestrator(
    store,
    runtime,
    decision,
    *,
    smart_agent=None,
    llm_client=None,
):
    """创建使用固定路由的测试编排器。"""
    return _StaticRouteOrchestrator(
        _dependencies(
            store,
            smart_agent=smart_agent,
            llm_client=llm_client,
        ),
        runtime,
        decision,
    )


def _build_lifecycle_host(store, runtime):
    """构造能直接执行任意业务流的最小混合宿主。"""
    builder = _LifecycleBuilder()
    builder.session_store = store
    builder.runtime = runtime
    builder.document_store = object()
    builder.smart_agent = object()
    builder.multidim_agent = object()
    return builder


def test_structured_rag_budget_termination_persists_failed_run() -> None:
    """结构化 RAG 正文预算终止后，不得留下缺失或 running 的运行记录。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = SessionStore(Path(temp_dir) / "sessions.sqlite3")
        session_id = store.create_session("结构化预算")["session_id"]
        runtime = AgentRuntimeForTest(
            AgentBudget(max_model_calls=3, token_budget=8),
            ExecutionContext(
                request_id="request-structured-budget",
                trace_id="a" * 32,
            ),
        )
        orchestrator = _orchestrator(
            store,
            runtime,
            UnifiedRouteDecision(
                "rag",
                "structured",
                "统计2025年3月个例",
            ),
        )
        payload = SimpleNamespace(
            question="统计个例",
            session_id=session_id,
            top_k=5,
            top_n=3,
        )

        async def scenario():
            return [
                event
                async for event in orchestrator.stream(payload)
            ]

        events = asyncio.run(scenario())
        runs = store.list_agent_runs(session_id)

    assert events[-1]["type"] == "error"
    assert events[-1]["termination_reason"] == "token_budget_exceeded"
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    assert runs[0]["trace_id"] == "a" * 32
    assert (
        runs[0]["metadata"]["termination_reason"]
        == "token_budget_exceeded"
    )
    assert runs[0]["metadata"]["budget"]["token_budget"] == 8


def test_smart_exception_is_finished_by_orchestrator() -> None:
    """Smart 抛异常时，编排器必须失败收口并在所有 metadata 中保留运行字段。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = SessionStore(Path(temp_dir) / "sessions.sqlite3")
        session_id = store.create_session("Smart 异常")["session_id"]
        runtime = AgentRuntimeForTest(
            AgentBudget(),
            ExecutionContext(
                request_id="request-smart-error",
                trace_id="b" * 32,
            ),
        )
        orchestrator = _orchestrator(
            store,
            runtime,
            UnifiedRouteDecision(
                "similar_case_match",
                "vector",
                "匹配相似过程",
            ),
            smart_agent=_ExplodingSmartAgent(),
        )
        payload = SimpleNamespace(
            question="匹配相似过程",
            session_id=session_id,
            top_k=5,
            top_n=3,
        )

        async def scenario():
            return [
                event
                async for event in orchestrator.stream(payload)
            ]

        events = asyncio.run(scenario())
        runs = store.list_agent_runs(session_id)

    smart_metadata = [
        event
        for event in events
        if event.get("type") == "metadata"
        and event.get("agent_type") == "smart_case_match"
    ]
    assert events[-1]["type"] == "error"
    assert events[-1]["termination_reason"] == "internal_error"
    assert smart_metadata
    assert all(
        _RUNTIME_FIELDS <= set(event)
        for event in smart_metadata
    )
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    assert runs[0]["trace_id"] == "b" * 32
    assert (
        runs[0]["metadata"]["termination_reason"]
        == "internal_error"
    )


def test_smart_error_event_finishes_run_with_runtime_metadata() -> None:
    """Smart 业务 error 事件也必须收口运行并写入 trace 与预算快照。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = SessionStore(Path(temp_dir) / "sessions.sqlite3")
        session_id = store.create_session("Smart 业务错误")["session_id"]
        runtime = AgentRuntimeForTest(
            AgentBudget(max_steps=5),
            ExecutionContext(
                request_id="request-smart-business-error",
                trace_id="c" * 32,
            ),
        )
        builder = _build_lifecycle_host(store, runtime)
        builder.smart_agent = _ErrorSmartAgent()

        async def scenario():
            return [
                event
                async for event in builder._stream_smart(
                    "匹配相似过程",
                    session_id,
                    None,
                    PerformanceTrace(),
                )
            ]

        events = asyncio.run(scenario())
        runs = store.list_agent_runs(session_id)

    smart_metadata = [
        event
        for event in events
        if event.get("type") == "metadata"
        and event.get("agent_type") == "smart_case_match"
    ]
    assert events[-1]["type"] == "error"
    assert smart_metadata
    assert all(
        _RUNTIME_FIELDS <= set(event)
        for event in smart_metadata
    )
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    assert runs[0]["trace_id"] == "c" * 32
    assert (
        runs[0]["metadata"]["termination_reason"]
        == "internal_error"
    )
    assert runs[0]["metadata"]["budget"]["max_steps"] == 5


def test_multidim_budget_termination_keeps_runtime_reason() -> None:
    """多维预算终止必须穿透业务异常，并保留预算原因和 trace。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = SessionStore(Path(temp_dir) / "sessions.sqlite3")
        session_id = store.create_session("多维预算")["session_id"]
        runtime = AgentRuntimeForTest(
            AgentBudget(max_model_calls=0),
            ExecutionContext(
                request_id="request-multidim-budget",
                trace_id="d" * 32,
            ),
        )
        builder = _build_lifecycle_host(store, runtime)
        builder.multidim_store = _RaisingMultidimStore()
        builder.multidim_agent = _BudgetMultidimAgent(runtime)

        async def scenario():
            try:
                async for _event in builder._execute_multidim(
                    session_id,
                    {"proposal_id": "proposal-1"},
                    PerformanceTrace(),
                ):
                    pass
            except BudgetTermination as exc:
                return exc
            return None

        error = asyncio.run(scenario())
        runs = store.list_agent_runs(session_id)

    assert isinstance(error, BudgetTermination)
    assert error.termination_reason == "model_call_budget_exceeded"
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    assert runs[0]["trace_id"] == "d" * 32
    assert (
        runs[0]["metadata"]["termination_reason"]
        == "model_call_budget_exceeded"
    )


def test_report_budget_termination_propagates_and_finishes_run() -> None:
    """承接式报告不得吞掉预算异常；运行必须失败收口并保留预算原因。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = SessionStore(Path(temp_dir) / "sessions.sqlite3")
        session_id = store.create_session("报告预算")["session_id"]
        runtime = AgentRuntimeForTest(
            AgentBudget(max_model_calls=1),
            ExecutionContext(
                request_id="request-report-budget",
                trace_id="e" * 32,
            ),
        )
        builder = _build_lifecycle_host(store, runtime)
        builder.multidim_agent = _BudgetReportAgent(runtime)
        previous_result = {
            "question": "统计2025年3月个例",
            "answer": "上一轮个例分析结果。",
            "source_message_id": 3,
            "run_id": "previous-run",
            "evidence_cases": [],
            "evidence_chunks": [],
        }

        async def scenario():
            try:
                async for _event in (
                    builder._stream_previous_result_report(
                        "把上面的分析整理成PDF",
                        session_id,
                        previous_result,
                        PerformanceTrace(),
                    )
                ):
                    pass
            except BudgetTermination as exc:
                return exc
            return None

        error = asyncio.run(scenario())
        runs = store.list_agent_runs(session_id)

    assert isinstance(error, BudgetTermination)
    assert error.termination_reason == "model_call_budget_exceeded"
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    assert runs[0]["trace_id"] == "e" * 32
    assert (
        runs[0]["metadata"]["termination_reason"]
        == "model_call_budget_exceeded"
    )


def test_sqlalchemy_start_agent_run_refreshes_trace_id() -> None:
    """同一运行记录再次启动时必须刷新独立 trace_id 列。"""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(
            User(
                user_id="runtime-user",
                username="runtime-user",
                password_hash="not-used",
                role="user",
                status="active",
            )
        )
        db.commit()

    try:
        store = SqlAlchemySessionStore(engine, "runtime-user")
        session_id = store.create_session("运行 trace")["session_id"]
        store.start_agent_run(
            "run-trace-refresh",
            session_id,
            "document_rag",
            "第一次启动",
            metadata={"trace_id": "1" * 32},
        )
        run = store.start_agent_run(
            "run-trace-refresh",
            session_id,
            "document_rag",
            "第二次启动",
            metadata={"trace_id": "2" * 32},
        )
    finally:
        engine.dispose()

    assert run["trace_id"] == "2" * 32


# 延迟导入保持测试模块的公共导入面简洁，并让 RED 原因指向生命周期行为。
def AgentRuntimeForTest(budget, execution_context):
    """导入真实运行时而不在测试模块顶层重复实现。"""
    from backend.app.agents.unified_chat.runtime import AgentRuntime

    return AgentRuntime(budget, execution_context)
