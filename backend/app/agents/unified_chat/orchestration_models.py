"""统一聊天编排层共享的数据模型与预算定义。"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from backend.app.agents.query_intent import QueryIntentAgent
from backend.app.services.query.context_cases import ContextCaseResolver
from backend.app.services.query.document_qa import DocumentQA
from backend.app.services.query.result_presenters import QueryResultPresenter
from backend.app.services.query.structured_query import StructuredQueryService


class TerminationReason(str, Enum):
    """请求结束或预算终止的固定原因集合。"""

    COMPLETED = "completed"
    TIMEOUT = "timeout"
    CANCELLED_BY_CLIENT = "cancelled_by_client"
    STEP_BUDGET_EXCEEDED = "step_budget_exceeded"
    MODEL_CALL_BUDGET_EXCEEDED = "model_call_budget_exceeded"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    TOOL_NOT_ALLOWED = "tool_not_allowed"
    TOOL_FAILED = "tool_failed"
    INVALID_SESSION = "invalid_session"
    INTERNAL_ERROR = "internal_error"

    def __str__(self) -> str:
        """保证枚举在序列化和日志中输出稳定字符串值。"""
        return self.value


@dataclass(frozen=True)
class AgentBudget:
    """单个请求的 Agent 执行预算，字段保持不可变。"""

    max_seconds: float = 300.0
    max_steps: int = 50
    max_model_calls: int = 40
    token_budget: int = 150000
    step_timeout_seconds: float = 90.0
    allowed_tools: frozenset[str] | None = None

    @classmethod
    def from_settings(cls, settings: Any) -> AgentBudget:
        """从 Settings 读取五个 Agent 预算字段并构造固定预算。"""
        return cls(
            max_seconds=float(getattr(settings, "agent_max_seconds", 300)),
            max_steps=int(getattr(settings, "agent_max_steps", 50)),
            max_model_calls=int(
                getattr(settings, "agent_max_model_calls", 40)
            ),
            token_budget=int(getattr(settings, "agent_token_budget", 150000)),
            step_timeout_seconds=float(
                getattr(settings, "agent_step_timeout_seconds", 90)
            ),
        )


@dataclass(frozen=True)
class ExecutionContext:
    """一次请求的关联身份、截止时间和取消信号。"""

    request_id: str = ""
    trace_id: str = ""
    user_id: str = ""
    session_id: str = ""
    started_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
    deadline: datetime | None = None
    cancel_event: asyncio.Event | None = field(
        default=None,
        compare=False,
        hash=False,
        repr=False,
    )


@dataclass(slots=True)
class SubtaskExecutionState:
    """单个路由或子任务的执行状态快照。"""

    subtask_id: str = ""
    route: str = ""
    status: str = "running"
    termination_reason: str = ""
    model_calls: int = 0
    elapsed_ms: int = 0
    started_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )


class BudgetTermination(Exception):
    """表示请求越过取消、超时或预算边界。"""

    def __init__(self, reason: TerminationReason, message: str = "") -> None:
        self.reason = reason
        super().__init__(message or reason.value)

    @property
    def termination_reason(self) -> str:
        """返回可直接写入 SSE 和运行元数据的固定字符串。"""
        return self.reason.value


@dataclass
class UnifiedRouteDecision:
    """统一入口完成会话理解后的下游分发信息。"""

    intent: str
    rag_strategy: str
    resolved_question: str
    clarification: str = ""
    referenced_case_ids: list[str] | None = None
    trace: dict[str, Any] | None = None
    tasks: list[dict[str, Any]] | None = None
    # 报告模式仅描述统一入口的复用策略，不改变正式多维检索意图。
    report_mode: str = "none"
    source_message_id: int | None = None


@dataclass(slots=True)
class OrchestratorDependencies:
    """统一编排器所需的显式应用服务。"""

    session_store: Any
    document_qa: DocumentQA
    structured_query: StructuredQueryService
    context_cases: ContextCaseResolver
    presenter: QueryResultPresenter
    standard_case_store: Any
    llm_client: Any
    multidim_agent: Any
    multidim_store: Any
    smart_agent: Any
    intent_agent: QueryIntentAgent
    # 长期记忆服务在阶段二接入；旧测试直接构造编排器时不要求必填。
    memory_service: Any | None = None
