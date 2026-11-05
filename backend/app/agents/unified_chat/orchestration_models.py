"""统一聊天编排层共享的数据模型。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.agents.query_intent import QueryIntentAgent
from backend.app.services.query.context_cases import ContextCaseResolver
from backend.app.services.query.document_qa import DocumentQA
from backend.app.services.query.result_presenters import QueryResultPresenter
from backend.app.services.query.structured_query import StructuredQueryService


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
