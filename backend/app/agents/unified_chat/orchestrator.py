"""主页面统一编排器，只协调现有 Agent，不修改其内部流程。"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from uuid import uuid4

from backend.app.agents.case_multidim_search.retrieval.structured_retriever import (
    StructuredCaseRetriever,
)
from backend.app.agents.query_intent import QueryIntentAgent
from backend.app.agents.unified_chat.conversation_memory import (
    ConversationMemoryManager,
)
from backend.app.agents.unified_chat.multidim_stream import (
    MultidimStreamMixin,
)
from backend.app.agents.unified_chat.orchestration_models import (
    BudgetTermination,
    OrchestratorDependencies,
    TerminationReason,
    UnifiedRouteDecision,
)
from backend.app.agents.unified_chat.performance import PerformanceTrace
from backend.app.agents.unified_chat.persistence import (
    ResultPersistenceMixin,
)
from backend.app.agents.unified_chat.rag_stream import RagStreamMixin
from backend.app.agents.unified_chat.general_chat_stream import (
    GeneralChatStreamMixin,
)
from backend.app.agents.unified_chat.report_stream import (
    ReportStreamMixin,
)
from backend.app.agents.unified_chat.routing import RoutingMixin
from backend.app.agents.unified_chat.runtime import AgentRuntime
from backend.app.agents.unified_chat.smart_stream import SmartStreamMixin
from backend.app.agents.unified_chat.stream_events import StreamEventMixin
from backend.app.agents.unified_chat.structured_context import (
    StructuredContextMixin,
)
from backend.app.agents.unified_chat.task_retrieval import (
    TaskRetrievalMixin,
)


LOGGER = logging.getLogger("uvicorn.error")

__all__ = [
    "OrchestratorDependencies",
    "UnifiedChatOrchestrator",
    "UnifiedRouteDecision",
]


class UnifiedChatOrchestrator(
    MultidimStreamMixin,
    SmartStreamMixin,
    RagStreamMixin,
    ReportStreamMixin,
    GeneralChatStreamMixin,
    RoutingMixin,
    StructuredContextMixin,
    TaskRetrievalMixin,
    ResultPersistenceMixin,
    StreamEventMixin,
):
    """把主 RAG、多维检索和 Smart 暴露为同一个流式聊天入口。"""

    def __init__(
        self,
        dependencies: OrchestratorDependencies,
        runtime: AgentRuntime | None = None,
    ) -> None:
        """保存编排依赖并创建请求级会话记忆管理器。"""
        self.dependencies = dependencies
        # 请求级运行时既可由 ChatService 注入，也兼容旧测试直接构造编排器。
        self.runtime = runtime or AgentRuntime()
        self.session_store = dependencies.session_store
        self.document_store = dependencies.document_qa.document_store
        self.multidim_agent = dependencies.multidim_agent
        self.multidim_store = dependencies.multidim_store
        self.smart_agent = dependencies.smart_agent
        self.intent_agent = dependencies.intent_agent
        if getattr(self.intent_agent, "case_provider", None) is None:
            # 自定义意图 Agent 也使用主项目当前标准个例复核历史实体。
            self.intent_agent.case_provider = self._standard_cases
        self.memory_manager = ConversationMemoryManager(
            self.session_store,
            case_provider=self._standard_cases,
        )
        self.structured_case_retriever = StructuredCaseRetriever()
        self._background_tasks: set[asyncio.Task] = set()
        # 编排器按请求创建，重生成信息可安全保存在当前实例中。
        self._regeneration_context: dict[str, Any] = {}
        # 请求级跨会话长期记忆上下文，读取失败时保持空文本且不阻塞回答。
        self._long_term_memory_context: str = ""
        self._long_term_memory_trace: dict[str, Any] = {}

    async def stream(
        self,
        payload,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """识别意图并把选中链路转换为统一 SSE 事件。"""
        performance = PerformanceTrace()
        try:
            # 取消、总截止时间和 Token 预算必须先于会话判断与模型路由。
            self.runtime.check()
            self.runtime.count_tokens(str(payload.question or ""))
        except BudgetTermination as exc:
            yield self._runtime_error_event(str(exc))
            return
        session_id = str(payload.session_id or "")
        if (
            session_id
            and self.session_store.get_session(session_id) is None
        ):
            self.runtime.mark_terminated(
                TerminationReason.INVALID_SESSION
            )
            yield self._runtime_error_event("Session not found")
            return

        # 首个事件先于记忆读取和意图模型调用，让页面立即给出反馈。
        yield self._progress_metadata(
            "query_intent",
            "",
            "understanding",
            2,
            "正在理解问题",
            performance,
        )
        performance.start("session_memory_read_ms")
        # 每轮开始清空上一轮重生成状态，避免污染普通提问。
        self._regeneration_context = {}
        regenerate_message_id = getattr(
            payload,
            "regenerate_from_message_id",
            None,
        )
        if regenerate_message_id:
            # 重新生成回到原问题时间点，后续会话内容不参与本次推理。
            target = (
                self.session_store.get_message(
                    session_id,
                    int(regenerate_message_id),
                )
                if session_id
                else None
            )
            if (
                not target
                or str(target.get("role") or "") != "assistant"
            ):
                performance.stop("session_memory_read_ms")
                yield {
                    "type": "error",
                    "message": "需要重新生成的助手消息不存在",
                }
                return
            target_metadata = dict(target.get("metadata") or {})
            source_user_message_id = int(
                target_metadata.get("source_user_message_id") or 0
            )
            source_user = (
                self.session_store.get_message(
                    session_id,
                    source_user_message_id,
                )
                if source_user_message_id
                else None
            )
            if (
                not source_user
                or str(source_user.get("role") or "") != "user"
            ):
                source_user = (
                    self.session_store.previous_user_message(
                        session_id,
                        int(regenerate_message_id),
                    )
                )
            if not source_user:
                performance.stop("session_memory_read_ms")
                yield {
                    "type": "error",
                    "message": "没有找到这条回答对应的原问题",
                }
                return
            payload.question = str(source_user.get("content") or "")
            memory_context = (
                self.memory_manager.context_before_message(
                    session_id,
                    int(regenerate_message_id),
                )
            )
            source_user_message_id = int(
                source_user.get("message_id") or 0
            )
            version_group_id = str(
                target_metadata.get("version_group_id")
                or f"user-{source_user_message_id}"
            )
            existing_versions = [
                int(item.get("version") or 1)
                for item in self.session_store.list_messages(session_id)
                if str(item.get("version_group_id") or "")
                == version_group_id
            ]
            self._regeneration_context = {
                "regenerated_from_message_id": int(
                    regenerate_message_id
                ),
                # 从旧版本再次重生成时，从组内最大版本继续递增。
                "version": max(existing_versions or [1]) + 1,
                "version_group_id": version_group_id,
                "source_user_message_id": (
                    source_user_message_id or None
                ),
                # 反馈原因只约束本次重生成，不写入会话摘要。
                "regeneration_instruction": str(
                    getattr(
                        payload,
                        "regeneration_instruction",
                        "",
                    )
                    or ""
                ).strip(),
            }
        else:
            # 记忆包已在上一轮后台预构建，此处直接读取。
            memory_context = (
                self.memory_manager.load_context(session_id)
                if session_id
                else {}
            )
        # 临时反馈只在当前 session 内读取，不传给意图识别。
        session_guidances = (
            self.session_store.list_feedback_guidances(
                session_id,
                statuses=("active",),
                limit=5,
            )
            if session_id
            else []
        )
        performance.stop("session_memory_read_ms")
        # 跨会话长期记忆在回答前读取；失败只降级不阻塞主回答。
        self._load_long_term_memory(session_id, regenerate_message_id)
        # 历史回答重做时不消费当前会话残留的多维确认状态。
        pending = (
            self.multidim_store.pending_proposal(session_id)
            if session_id and not regenerate_message_id
            else None
        )
        performance.start("intent_ms")
        # 意图模型在线程中执行，入口首个 metadata 已先行反馈。
        try:
            self.runtime.begin_subtask(
                "intent_route",
                "query_intent",
            )
            self.runtime.count_step()
            self.runtime.count_model_call()
            decision = await asyncio.to_thread(
                self._route,
                payload.question,
                bool(pending),
                memory_context,
            )
            self.runtime.finish_subtask()
        except BudgetTermination as exc:
            performance.stop("intent_ms")
            yield self._runtime_error_event(str(exc))
            return
        # 承接式报告请求优先复用上一轮落库结果。
        previous_result = self._previous_reusable_result(
            session_id,
            payload.question,
            memory_context,
        )
        if self._is_previous_result_report_request(
            payload.question,
            memory_context,
        ):
            if previous_result:
                decision.report_mode = "from_previous_result"
                decision.source_message_id = (
                    int(previous_result.get("source_message_id") or 0)
                    or None
                )
                decision.trace = dict(decision.trace or {})
                decision.trace.update(
                    {
                        "report_mode": decision.report_mode,
                        "source_message_id": (
                            decision.source_message_id
                        ),
                    }
                )
            elif decision.intent == "multidim_search":
                decision = UnifiedRouteDecision(
                    "clarify",
                    decision.rag_strategy,
                    str(payload.question or ""),
                    "当前会话没有可复用的上一轮个例分析结果，"
                    "请先完成个例分析，或明确提供需要生成报告的个例。",
                    trace={
                        "report_mode": (
                            "from_previous_result_unavailable"
                        )
                    },
                )
        performance.stop("intent_ms")
        if session_id and not regenerate_message_id:
            self.session_store.add_message(
                session_id,
                "user",
                payload.question,
            )

        route_messages = {
            "rag": "正在检索本地知识库",
            "multidim_search": "正在准备多维检索与报告",
            "similar_case_match": "正在匹配相似历史过程",
            "clarify": "正在整理需要确认的信息",
            "general_chat": "正在联网搜索最新信息",
        }
        yield self._progress_metadata(
            decision.intent,
            "",
            "routed",
            6,
            route_messages.get(
                decision.intent,
                "正在处理问题",
            ),
            performance,
        )

        try:
            self.runtime.check()
            self.runtime.begin_subtask(
                f"route_{decision.intent}",
                decision.intent,
            )
            self.runtime.count_step()
            if (
                decision.report_mode == "from_previous_result"
                and previous_result
            ):
                async for event in self._stream_previous_result_report(
                    payload.question,
                    session_id,
                    previous_result,
                    performance,
                ):
                    yield event
                self.runtime.finish_subtask()
                return
            if decision.intent == "multidim_search":
                # 确认和取消语句原样交给状态机，其他问题使用消解表达。
                multidim_question = (
                    payload.question
                    if pending
                    else decision.resolved_question
                )
                async for event in self._stream_multidim(
                    multidim_question,
                    session_id,
                    pending,
                    performance,
                ):
                    yield event
                self.runtime.finish_subtask()
                return
            if decision.intent == "similar_case_match":
                async for event in self._stream_smart(
                    decision.resolved_question,
                    session_id,
                    is_disconnected,
                    performance,
                ):
                    yield event
                self.runtime.finish_subtask()
                return
            if decision.intent == "clarify":
                async for event in self._stream_clarification(
                    payload.question,
                    session_id,
                    decision.clarification,
                    performance,
                ):
                    yield event
                self.runtime.finish_subtask()
                return
            if decision.intent == "general_chat":
                async for event in self._stream_general_chat(
                    decision.resolved_question,
                    session_id,
                    run_id=f"general_chat_{uuid4().hex}",
                    performance=performance,
                ):
                    yield event
                self.runtime.finish_subtask()
                return
            async for event in self._stream_rag(
                payload,
                session_id,
                [],
                decision.rag_strategy,
                resolved_question=decision.resolved_question,
                referenced_case_ids=decision.referenced_case_ids,
                route_trace=decision.trace,
                task_plan=decision.tasks,
                session_guidances=session_guidances,
                performance=performance,
            ):
                yield event
            self.runtime.finish_subtask()
        except BudgetTermination as exc:
            self._finish_active_run(
                status="failed",
                error=str(exc),
            )
            yield self._runtime_error_event(str(exc))
        except Exception as exc:
            self.runtime.mark_terminated(
                TerminationReason.INTERNAL_ERROR
            )
            self._finish_active_run(
                status="failed",
                error=str(exc),
            )
            yield self._runtime_error_event(str(exc))

    def _runtime_error_event(self, message: str) -> dict[str, Any]:
        """构造带请求关联字段的结构化运行时错误事件。"""
        return {
            "type": "error",
            "message": str(message or "请求执行失败"),
            **self._runtime_fields(),
        }

    def _load_long_term_memory(
        self,
        session_id: str,
        regenerate_message_id: int | None = None,
    ) -> None:
        """按当前用户读取长期记忆；异常只记录降级并清空上下文。"""
        self._long_term_memory_context = ""
        self._long_term_memory_trace = {}
        memory_service = getattr(self.dependencies, "memory_service", None)
        if memory_service is None or not session_id:
            return
        if regenerate_message_id:
            return
        try:
            context = memory_service.context_for_user(
                self.session_store.user_id,
                limit=10,
            )
            self._long_term_memory_context = "；".join(
                str(item.get("content") or "")
                for item in context.get("memories", [])
                if str(item.get("content") or "")
            )[:2000]
            self._long_term_memory_trace = {
                "fallback_structured": bool(
                    context.get("fallback_structured")
                ),
                "memory_count": len(context.get("memories", [])),
            }
        except Exception as exc:
            LOGGER.warning(
                "[长期记忆] 读取失败，不阻塞回答 user_id=%s error=%s",
                self.session_store.user_id,
                exc,
            )

    def _schedule_memory_refresh(self, session_id: str) -> None:
        """创建受引用保护的会话记忆刷新任务。"""
        if not session_id:
            return

        async def refresh() -> None:
            """在线程池中执行 SQLite 增量摘要刷新。"""
            try:
                await asyncio.to_thread(
                    self.memory_manager.refresh,
                    session_id,
                )
            except Exception as exc:
                LOGGER.warning(
                    "[会话记忆] 后台刷新失败 session_id=%s error=%s",
                    session_id,
                    exc,
                )

        task = asyncio.create_task(refresh())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def wait_for_background_tasks(self) -> None:
        """等待当前实例的后台增强任务。"""
        tasks = list(self._background_tasks)
        if tasks:
            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )
