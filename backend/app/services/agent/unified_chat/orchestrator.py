"""主页面统一编排器，只协调现有 Agent，不修改它们的内部处理流程。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from backend.app.services.agent.case_multidim_search.core import progress as multidim_progress
from backend.app.services.agent.case_multidim_search.core.natural_query import NaturalCaseQueryParser
from backend.app.services.agent.case_multidim_search.integrations.non_thinking_llm import ensure_non_thinking_client
from backend.app.services.agent.case_multidim_search.reporting.report_preview import ensure_report_thumbnail
from backend.app.services.agent.case_multidim_search.retrieval.structured_retriever import StructuredCaseRetriever
from backend.app.services.agent.case_multidim_search.schemas import CaseSearchQuery
from backend.app.services.agent.query_intent_compat import IntentRouteRequest, QueryIntentAgent
from backend.app.services.agent.unified_chat.conversation_memory import ConversationMemoryManager
from backend.app.services.agent.unified_chat.performance import PerformanceTrace
from backend.app.services.agent.unified_chat.metric_aggregation import (
    aggregate_metric_facts,
    metric_result_text,
)
from backend.app.services.agent.unified_chat.result_adapter import (
    adapt_multidim_result,
    adapt_smart_result,
    confirmation_answer,
)
from backend.app.services.feedback_guidance import render_session_guidance
from backend.app.services.model_client.chat import get_llm_client
from backend.app.models import DocumentRetrievalHit


LOGGER = logging.getLogger("uvicorn.error")


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


class UnifiedChatOrchestrator:
    """把主 RAG、多维检索和 Smart 暴露为同一个流式聊天入口。"""

    def __init__(self, main_module, multidim_agent, multidim_store, smart_agent, intent_agent=None):
        self.main = main_module
        self.session_store = main_module.session_store
        self.document_store = main_module.document_store
        self.multidim_agent = multidim_agent
        self.multidim_store = multidim_store
        self.smart_agent = smart_agent
        # 连字符目录不能直接用 from import，这里由兼容模块延迟加载新版意图 Agent。
        self.intent_agent = intent_agent or QueryIntentAgent(case_provider=self._standard_cases)
        if getattr(self.intent_agent, "case_provider", None) is None:
            # 自定义意图 Agent 也必须使用主项目当前标准个例库复核历史实体。
            self.intent_agent.case_provider = self._standard_cases
        self.memory_manager = ConversationMemoryManager(self.session_store, case_provider=self._standard_cases)
        self.structured_case_retriever = StructuredCaseRetriever()
        self._background_tasks: set[asyncio.Task] = set()
        # 编排器按请求创建，重新生成信息可以安全地保存在当前实例中。
        self._regeneration_context: dict[str, Any] = {}

    async def stream(
        self,
        payload,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """识别意图并把选中链路转换为 metadata/delta/done/error。"""
        performance = PerformanceTrace()
        session_id = str(payload.session_id or "")
        if session_id and self.session_store.get_session(session_id) is None:
            yield {"type": "error", "message": "Session not found"}
            return

        # 首个事件必须先于记忆读取和意图模型调用，让页面立即给出可见反馈。
        yield self._progress_metadata("query_intent", "", "understanding", 2, "正在理解问题", performance)
        performance.start("session_memory_read_ms")
        # 编排器可能被应用复用，每轮开始必须清空上一轮重生成状态，避免污染普通提问。
        self._regeneration_context = {}
        regenerate_message_id = getattr(payload, "regenerate_from_message_id", None)
        if regenerate_message_id:
            # 重新生成必须回到原问题所在时间点，旧回答及其后的会话内容不能参与本次推理。
            target = self.session_store.get_message(session_id, int(regenerate_message_id)) if session_id else None
            if not target or str(target.get("role") or "") != "assistant":
                performance.stop("session_memory_read_ms")
                yield {"type": "error", "message": "需要重新生成的助手消息不存在"}
                return
            target_metadata = dict(target.get("metadata") or {})
            source_user_message_id = int(target_metadata.get("source_user_message_id") or 0)
            source_user = (
                self.session_store.get_message(session_id, source_user_message_id)
                if source_user_message_id else None
            )
            if not source_user or str(source_user.get("role") or "") != "user":
                source_user = self.session_store.previous_user_message(session_id, int(regenerate_message_id))
            if not source_user:
                performance.stop("session_memory_read_ms")
                yield {"type": "error", "message": "没有找到这条回答对应的原问题"}
                return
            payload.question = str(source_user.get("content") or "")
            memory_context = self.memory_manager.context_before_message(session_id, int(regenerate_message_id))
            source_user_message_id = int(source_user.get("message_id") or 0)
            version_group_id = str(
                target_metadata.get("version_group_id") or f"user-{source_user_message_id}"
            )
            existing_versions = [
                int(item.get("version") or 1)
                for item in self.session_store.list_messages(session_id)
                if str(item.get("version_group_id") or "") == version_group_id
            ]
            self._regeneration_context = {
                "regenerated_from_message_id": int(regenerate_message_id),
                # 即使从旧版本再次重生成，也从组内最大版本继续递增，避免出现两个“版本 2”。
                "version": max(existing_versions or [1]) + 1,
                "version_group_id": version_group_id,
                "source_user_message_id": source_user_message_id or None,
                # 反馈原因只约束本次重生成，不写入会话摘要或后续普通问题。
                "regeneration_instruction": str(getattr(payload, "regeneration_instruction", "") or "").strip(),
            }
        else:
            # 记忆包已经在上一轮后台预构建，此处直接读取可避免短任务的线程调度开销。
            memory_context = self.memory_manager.load_context(session_id) if session_id else {}
        # 临时反馈只在当前 session 内读取，且不传给意图识别和原始检索条件解析。
        session_guidances = (
            self.session_store.list_feedback_guidances(session_id, statuses=("active",), limit=5)
            if session_id
            else []
        )
        performance.stop("session_memory_read_ms")
        # 历史回答重做时不消费当前会话可能残留的多维确认状态。
        pending = self.multidim_store.pending_proposal(session_id) if session_id and not regenerate_message_id else None
        performance.start("intent_ms")
        # 意图模型仍在线程中执行，避免阻塞事件循环；入口首个 metadata 已先行反馈。
        decision = await asyncio.to_thread(self._route, payload.question, bool(pending), memory_context)
        # “上面/这些/这几个”等承接式报告请求优先复用上一轮已落库结果，
        # 避免重新向量检索和逐例调用大模型；没有可复用结果时明确提示用户。
        previous_result = self._previous_reusable_result(session_id, payload.question, memory_context)
        # 承接式简易报告是基于上一轮已完成结果的后处理动作，不能依赖意图模型
        # 是否恰好把它分类为 multidim_search；否则模型判成 rag 时会绕过复用链路。
        if self._is_previous_result_report_request(payload.question, memory_context):
            if previous_result:
                decision.report_mode = "from_previous_result"
                decision.source_message_id = int(previous_result.get("source_message_id") or 0) or None
                decision.trace = dict(decision.trace or {})
                decision.trace.update({
                    "report_mode": decision.report_mode,
                    "source_message_id": decision.source_message_id,
                })
            elif decision.intent == "multidim_search":
                decision = UnifiedRouteDecision(
                    "clarify",
                    decision.rag_strategy,
                    str(payload.question or ""),
                    "当前会话没有可复用的上一轮个例分析结果，请先完成个例分析，或明确提供需要生成报告的个例。",
                    trace={"report_mode": "from_previous_result_unavailable"},
                )
        performance.stop("intent_ms")
        if session_id and not regenerate_message_id:
            self.session_store.add_message(session_id, "user", payload.question)

        route_messages = {
            "rag": "正在检索本地知识库",
            "multidim_search": "正在准备多维检索与报告",
            "similar_case_match": "正在匹配相似历史过程",
            "clarify": "正在整理需要确认的信息",
        }
        yield self._progress_metadata(
            decision.intent,
            "",
            "routed",
            6,
            route_messages.get(decision.intent, "正在处理问题"),
            performance,
        )

        try:
            if decision.report_mode == "from_previous_result" and previous_result:
                async for event in self._stream_previous_result_report(
                    payload.question,
                    session_id,
                    previous_result,
                    performance,
                ):
                    yield event
                return
            if decision.intent == "multidim_search":
                # 确认和取消语句必须原样交给既有状态机，其他问题使用消解后的完整表达。
                multidim_question = payload.question if pending else decision.resolved_question
                async for event in self._stream_multidim(multidim_question, session_id, pending, performance):
                    yield event
                return
            if decision.intent == "similar_case_match":
                async for event in self._stream_smart(decision.resolved_question, session_id, is_disconnected, performance):
                    yield event
                return
            if decision.intent == "clarify":
                async for event in self._stream_clarification(payload.question, session_id, decision.clarification, performance):
                    yield event
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
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}

    async def _stream_multidim(
        self,
        question: str,
        session_id: str,
        pending: dict | None,
        performance: PerformanceTrace,
    ) -> AsyncIterator[dict[str, Any]]:
        """在主会话中完成多维条件提案、确认和既有分析流程。"""
        if not session_id:
            yield {"type": "error", "message": "多维检索必须先创建主页面会话"}
            return
        if pending and self._is_cancellation(question):
            self.multidim_store.reset(session_id)
            answer = "已取消待确认的多维检索条件，请重新描述需要检索的时间、地区或灾种。"
            run_id = f"case_multidim_cancel_{uuid4().hex}"
            common = self._empty_result("case_multidim_search", run_id, question, answer)
            self._persist_complete(session_id, common, status="cancelled", performance=performance)
            yield self._metadata(common, status="cancelled")
            async for event in self._stream_text_chunks(answer):
                yield event
            yield self._done(common, status="cancelled")
            self._schedule_memory_refresh(session_id)
            return

        if pending and self._is_confirmation(question):
            async for event in self._execute_multidim(session_id, pending, performance):
                yield event
            return

        parser = NaturalCaseQueryParser(ensure_non_thinking_client(get_llm_client()))
        performance.start("multidim_parse_ms")
        parse_task = asyncio.create_task(asyncio.to_thread(
            self.multidim_store.propose, session_id, question, parser,
        ))
        while True:
            try:
                proposal = await asyncio.wait_for(asyncio.shield(parse_task), timeout=0.5)
                break
            except asyncio.TimeoutError:
                yield self._progress_metadata(
                    "case_multidim_search", "", "condition_parse", 12,
                    "正在解析报告检索条件", performance,
                )
        performance.stop("multidim_parse_ms")
        run_id = proposal.proposal_id or f"case_multidim_proposal_{uuid4().hex}"
        answer = confirmation_answer(proposal)
        common = self._empty_result("case_multidim_search", run_id, question, answer)
        common.update(
            {
                "retrieval_mode": "agent:case_multidim_confirmation",
                "query_conditions": proposal.parsed_request.model_dump(),
                "proposal_id": proposal.proposal_id,
                "proposal_operation": proposal.operation,
                "proposal_question": question,
                "warnings": list(proposal.warnings or []),
            }
        )
        status = "awaiting_confirmation" if proposal.can_confirm else "needs_clarification"
        self._persist_complete(session_id, common, status=status, performance=performance)
        yield self._metadata(common, status=status)
        async for event in self._stream_text_chunks(answer):
            yield event
        yield self._done(common, status=status)
        self._schedule_memory_refresh(session_id)

    async def _execute_multidim(
        self,
        session_id: str,
        pending: dict,
        performance: PerformanceTrace,
    ) -> AsyncIterator[dict[str, Any]]:
        """确认条件后原样调用多维 Agent，并按需生成 PDF 报告。"""
        run_id = f"case_multidim_{uuid4().hex}"
        request, original_message = self.multidim_store.confirm(
            session_id,
            str(pending.get("proposal_id") or ""),
            run_id,
        )
        self.session_store.start_agent_run(
            run_id,
            session_id,
            "case_multidim_search",
            original_message,
            metadata={"query_conditions": request.model_dump()},
        )
        yield {
            "type": "metadata",
            "agent_type": "case_multidim_search",
            "run_id": run_id,
            "status": "running",
            "progress": {"stage": "search", "percent": 5, "message": "正在执行多维检索与统计分析"},
        }
        multidim_progress.start(run_id)
        try:
            performance.start("multidim_analysis_ms")
            analysis_task = asyncio.create_task(asyncio.to_thread(self.multidim_agent.analyze_structured, request))
            last_progress = None
            while not analysis_task.done():
                progress = multidim_progress.get(run_id)
                signature = self._progress_signature(progress)
                if progress and signature != last_progress:
                    last_progress = signature
                    yield self._progress_metadata(
                        "case_multidim_search", run_id, str(progress.get("stage") or "analysis"),
                        int(progress.get("percent") or 0), str(progress.get("message") or "正在分析个例"),
                        performance, extra=progress,
                    )
                await asyncio.sleep(0.5)
            result = await analysis_task
            performance.stop("multidim_analysis_ms")
            # 只有明确要求报告才会进入该链路，因此确认后始终生成可预览的 PDF。
            multidim_progress.update(run_id, stage="pdf_export", message="正在准备 PDF 报告", percent=5)

            def update_pdf_progress(percent: int, message: str) -> None:
                """把 PDF 构建线程的阶段进度写入统一进度存储。"""
                multidim_progress.update(
                    run_id,
                    stage="pdf_export",
                    message=message,
                    percent=percent,
                )

            performance.start("pdf_total_ms")
            pdf_task = asyncio.create_task(asyncio.to_thread(
                self.multidim_agent.export_pdf_from_answer,
                result.answer_id,
                filename=f"{run_id}.pdf",
                report_title=result.title,
                progress_callback=update_pdf_progress,
            ))
            last_progress = None
            while not pdf_task.done():
                progress = multidim_progress.get(run_id)
                signature = self._progress_signature(progress)
                if progress and signature != last_progress:
                    last_progress = signature
                    yield self._progress_metadata(
                        "case_multidim_search", run_id, "pdf_export",
                        int(progress.get("percent") or 0), str(progress.get("message") or "正在生成 PDF 报告"),
                        performance, extra=progress,
                    )
                await asyncio.sleep(0.5)
            report_path = await pdf_task
            performance.stop("pdf_total_ms")
            reports = [self._report_payload(
                report_path,
                report_id=result.answer_id,
                title=result.title,
            )]
            common = adapt_multidim_result(result, self.document_store, reports)
            common["run_id"] = run_id
            common["question"] = original_message
            common["answer_id"] = result.answer_id
            multidim_progress.complete(run_id)
            # 报告和结构化结果先交给页面，文本再按片段刷新，最后完成持久化并发送 done。
            yield self._metadata(common, status="running")
            async for event in self._stream_text_chunks(common["answer"]):
                yield event
            yield self._progress_metadata(
                "case_multidim_search", run_id, "persist", 98, "正在保存报告和会话记录", performance,
            )
            self._persist_complete(
                session_id, common, status="completed", already_started=True, performance=performance,
            )
            yield self._completion_metadata(common, status="completed")
            yield self._done(common, status="completed")
            self._schedule_memory_refresh(session_id)
        except Exception as exc:
            multidim_progress.fail(run_id, f"检索失败：{exc}")
            self.session_store.finish_agent_run(run_id, status="failed", metadata={"error": str(exc)})
            raise

    async def _stream_smart(
        self,
        question: str,
        session_id: str,
        is_disconnected: Callable[[], Awaitable[bool]] | None,
        performance: PerformanceTrace,
    ) -> AsyncIterator[dict[str, Any]]:
        """完整消费 Smart 原流式流程，仅转换对外事件名称和持久化位置。"""
        from backend.app.services.agent.Smart_Case_Match.schemas import NaturalLanguageMatchRequest

        run_id = ""
        started = False
        request = NaturalLanguageMatchRequest(message=question)
        performance.start("smart_total_ms")
        async for event, data in self.smart_agent.stream_natural_match(request, is_disconnected):
            if event == "accepted":
                run_id = str(data.get("run_id") or f"smart_case_{uuid4().hex}")
                if session_id:
                    self.session_store.start_agent_run(run_id, session_id, "smart_case_match", question)
                    started = True
                yield {
                    "type": "metadata",
                    "agent_type": "smart_case_match",
                    "run_id": run_id,
                    "status": "running",
                    "progress": data,
                }
                continue
            if event in {"stage", "heartbeat", "parsed_query", "matched_cases", "forecast"}:
                yield {
                    "type": "metadata",
                    "agent_type": "smart_case_match",
                    "run_id": run_id,
                    "status": "running",
                    "progress": data,
                }
                continue
            if event == "error":
                if session_id and started:
                    self.session_store.finish_agent_run(run_id, status="failed", metadata={"error": data.get("message", "")})
                yield {"type": "error", "message": data.get("message", "相似个例匹配失败"), "run_id": run_id}
                return
            if event == "completed":
                performance.stop("smart_total_ms")
                common = adapt_smart_result(data, self.document_store)
                common["question"] = question
                # Smart 结构化结果先展示；文字按片段传输但由主页面隐藏重复副本。
                yield self._metadata(common, status="running")
                async for delta_event in self._stream_text_chunks(common["answer"]):
                    yield delta_event
                yield self._progress_metadata(
                    "smart_case_match", str(common.get("run_id") or run_id), "persist", 98,
                    "正在保存匹配结果和证据", performance,
                )
                if session_id:
                    self._persist_complete(
                        session_id,
                        common,
                        status=str(data.get("status") or "completed"),
                        already_started=started,
                        performance=performance,
                    )
                yield self._completion_metadata(common, status=str(data.get("status") or "completed"))
                yield self._done(common, status=str(data.get("status") or "completed"))
                self._schedule_memory_refresh(session_id)
                return

    async def _stream_rag(
        self,
        payload,
        session_id: str,
        history: list[dict],
        rag_strategy: str = "vector",
        *,
        resolved_question: str = "",
        referenced_case_ids: list[str] | None = None,
        route_trace: dict[str, Any] | None = None,
        task_plan: list[dict[str, Any]] | None = None,
        session_guidances: list[dict[str, Any]] | None = None,
        performance: PerformanceTrace,
    ) -> AsyncIterator[dict[str, Any]]:
        """保留主项目原文档向量检索和云端流式回答链路。"""
        run_id = f"document_rag_{uuid4().hex}"
        original_question = str(payload.question or "")
        question = str(resolved_question or original_question)
        generation_question = question
        session_guidance_prompt = render_session_guidance(session_guidances)
        regeneration_instruction = str(self._regeneration_context.get("regeneration_instruction") or "").strip()
        if regeneration_instruction:
            generation_question = (
                f"{question}\n\n重新生成要求：{regeneration_instruction}。"
                "请仍严格依据本轮检索证据回答，不得编造材料外事实。"
            )
        context_case_ids = list(referenced_case_ids or [])
        rag_tasks = [task for task in (task_plan or []) if str(task.get("route") or "") == "rag"]
        rag_tasks = self._apply_same_scope_feedback_to_tasks(rag_tasks, session_guidances)
        # 即使任务描述被补充了反馈维度，底层向量检索仍使用本轮原始问题。
        for task in rag_tasks:
            if task.get("feedback_guidance_ids"):
                task["retrieval_question"] = question
        feedback_task_applied = any(task.get("feedback_guidance_ids") for task in rag_tasks)
        uses_task_plan = bool(rag_tasks) and (
            len(rag_tasks) > 1
            or any(str(task.get("type") or "") == "case_listing" for task in rag_tasks)
            or feedback_task_applied
        )
        if not context_case_ids:
            context_case_ids = self.main._context_case_ids_from_messages(history, original_question)
        if rag_strategy == "structured" and not uses_task_plan:
            yield self._progress_metadata(
                "document_rag", run_id, "structured_search", 12,
                "正在查询标准化个例", performance,
            )
            structured = await asyncio.to_thread(
                self.main._answer_structured_question, question, context_case_ids=context_case_ids,
            )
            if structured is not None and self.main._structured_result_needs_metric_documents(structured):
                structured = self.main._metric_document_result_from_structured(
                    question,
                    top_k=payload.top_k,
                    top_n=payload.top_n,
                    context_case_ids=context_case_ids,
                ) or structured
            if structured is not None:
                common = self._structured_common(structured, run_id, original_question)
                # 结构化查询只负责提供已经核验的事实，最终展示统一交给回答模型。
                # 这样“需要改进”中的表格、分组、补充维度等任意回答要求都能生效。
                guidance_audit = {
                    "applied": False,
                    "count": len(session_guidances or []),
                    "renderer": "structured_fallback",
                    "reason": "structured_result_fallback",
                }
                common.setdefault("audit", {})["session_feedback_guidance"] = guidance_audit
                self._attach_memory_trace(common, original_question, question, context_case_ids, route_trace)
                yield self._metadata(common, status="running")
                structured_answer = str(common.get("answer") or "")
                answer_parts: list[str] = []
                llm_used = False
                llm_status = "structured_fallback"
                if self.main.llm_client.is_available():
                    render_started = perf_counter()
                    performance.start("llm_total_ms")
                    first_delta = True
                    yield self._progress_metadata(
                        "document_rag", run_id, "generation", 45,
                        "正在整理结构化事实并生成回答", performance,
                    )
                    try:
                        render_context = self._structured_render_context(structured, common)
                        try:
                            answer_stream = self.main.llm_client.stream_answer_with_context(
                                generation_question,
                                render_context,
                                session_guidance_prompt=session_guidance_prompt,
                            )
                        except TypeError:
                            # 兼容旧测试客户端和外部适配器；生产客户端支持会话反馈参数。
                            answer_stream = self.main.llm_client.stream_answer_with_context(
                                generation_question,
                                render_context,
                            )
                        async for stream_kind, delta in self._stream_sync_deltas(answer_stream):
                            if stream_kind == "heartbeat":
                                yield self._progress_metadata(
                                    "document_rag", run_id, "generation", 50,
                                    "正在等待模型返回首段回答", performance,
                                )
                                continue
                            if first_delta:
                                performance.set("llm_ttft_ms", (perf_counter() - render_started) * 1000)
                                first_delta = False
                                yield self._progress_metadata(
                                    "document_rag", run_id, "streaming", 60,
                                    "模型已开始整理回答", performance,
                                )
                            for visible_delta in self._split_stream_delta(delta):
                                answer_parts.append(visible_delta)
                                yield {"type": "delta", "text": visible_delta}
                                await asyncio.sleep(0)
                        performance.stop("llm_total_ms")
                        if not answer_parts:
                            raise RuntimeError("结构化回答渲染模型返回空答案")
                        llm_used = True
                        llm_status = "called_with_structured_facts"
                        guidance_audit.update({
                            "applied": bool(session_guidance_prompt),
                            "renderer": "llm",
                            "reason": "structured_facts_rendered_by_answer_model",
                        })
                    except Exception as exc:
                        performance.stop("llm_total_ms")
                        # 首字前失败时回退核验过的结构化答案；已有增量时沿用已显示内容，避免重复拼接。
                        llm_status = f"structured_render_failed: {exc}"
                        guidance_audit.update({
                            "applied": False,
                            "renderer": "structured_fallback",
                            "reason": "structured_render_failed",
                        })
                        if not answer_parts:
                            answer_parts.append(structured_answer)
                            yield {"type": "delta", "text": structured_answer}
                else:
                    answer_parts.append(structured_answer)
                    yield {"type": "delta", "text": structured_answer}
                    guidance_audit.update({
                        "applied": False,
                        "renderer": "structured_fallback",
                        "reason": "answer_model_unavailable",
                    })
                common["answer"] = "".join(answer_parts)
                common["llm_used"] = llm_used
                common["llm_status"] = llm_status
                common["answer_renderer"] = "llm" if llm_used else "structured_fallback"
                yield self._progress_metadata(
                    "document_rag", run_id, "persist", 98, "正在保存回答和证据", performance,
                )
                if session_id:
                    self._persist_complete(session_id, common, status="completed", performance=performance)
                yield self._completion_metadata(common, status="completed")
                yield self._done(common, status="completed")
                self._schedule_memory_refresh(session_id)
                return
        if session_id:
            self.session_store.start_agent_run(run_id, session_id, "document_rag", original_question)
        yield self._progress_metadata(
            "document_rag", run_id, "retrieval", 12, "正在检索本地知识库", performance,
        )
        performance.start("retrieval_ms")
        task_context_blocks: list[str] = []
        task_results: list[dict[str, Any]] = []
        planned_evidence_cases: list[dict[str, Any]] = []
        if uses_task_plan:
            retrieval_mode, hits, task_context_blocks, task_results, planned_evidence_cases = (
                self._retrieve_rag_task_plan(
                    rag_tasks,
                    context_case_ids,
                    top_k=payload.top_k,
                    top_n=payload.top_n,
                )
            )
        else:
            # 单任务继续沿用原检索路径，避免改变已有内容生成效果。
            retrieval_mode, hits = self.main._retrieve_document_chunks(
                question,
                top_k=payload.top_k,
                top_n=payload.top_n,
                context_case_ids=context_case_ids,
            )
        performance.stop("retrieval_ms")
        yield self._progress_metadata(
            "document_rag", run_id, "retrieved", 35,
            f"已找到 {len(hits)} 个相关证据片段", performance,
        )
        analysis = self.main.AgentIntentAnalyzer().analyze(question)
        plan = self.main.AgentPlanner().plan(question, analysis)
        hit_payloads = [self.main._document_hit_to_response(hit) for hit in hits]
        images = self._images_from_hits(hit_payloads)
        # 图片继续作为可追溯证据持久化；是否展示由任务计划单独声明，避免检索命中即铺满页面。
        image_display_mode = self._rag_image_display_mode(rag_tasks)
        evidence_cases = self._context_evidence_cases(context_case_ids)
        known_case_ids = {str(item.get("case_id") or "") for item in evidence_cases}
        for case in planned_evidence_cases:
            case_id = str(case.get("case_id") or "")
            if case_id and case_id not in known_case_ids:
                evidence_cases.append(case)
                known_case_ids.add(case_id)
        if not evidence_cases:
            evidence_cases = self._evidence_cases_from_document_hits(hit_payloads)
        common = {
            "agent_type": "document_rag",
            "run_id": run_id,
            "question": original_question,
            "retrieval_mode": retrieval_mode,
            "hit_count": len(hits),
            "hits": hit_payloads,
            "evidence_cases": evidence_cases,
            "evidence_chunks": hit_payloads,
            "images": images,
            "image_display_mode": image_display_mode,
            "visuals": [],
            "reports": [],
            "analysis": {},
            "intent_trace": {
                "route": "document_rag",
                "intents": [intent.value for intent in plan.intents],
            },
            "execution_plan": plan.to_dict(),
            "audit": {"agent_runs": [{"agent": "document_retriever", "status": "ok"}]},
        }
        common["audit"]["session_feedback_guidance"] = {
            "applied": bool(session_guidance_prompt),
            "count": len(session_guidances or []),
        }
        if uses_task_plan:
            # 仅保存任务状态和证据 ID，完整正文仍由统一 evidence_chunks 字段持久化。
            common["task_plan"] = task_results
            common["audit"]["agent_runs"].extend(
                {
                    "agent": "task_evidence_retriever",
                    "task_id": item.get("id"),
                    "task_type": item.get("type"),
                    "status": item.get("status", "ok"),
                    "evidence_chunk_ids": item.get("evidence_chunk_ids", []),
                }
                for item in task_results
            )
        self._attach_memory_trace(common, original_question, question, context_case_ids, route_trace)
        yield self._metadata(common, status="running")
        answer_parts: list[str] = []
        llm_used = False
        llm_status = "skipped_no_hits"
        has_evidence = bool(hits or task_context_blocks)
        if not has_evidence:
            answer_parts.append("未检索到相关证据片段。")
            yield {"type": "delta", "text": answer_parts[-1]}
        elif self.main.llm_client.is_available():
            context_blocks = task_context_blocks or self.main._context_blocks_from_document_hits(hits)
            try:
                # 同步模型流通过队列桥接到异步 SSE，避免等待完整答案后再集中回放。
                llm_started = perf_counter()
                performance.start("llm_total_ms")
                first_delta = True
                yield self._progress_metadata(
                    "document_rag", run_id, "generation", 45, "正在基于证据生成回答", performance,
                )
                try:
                    answer_stream = self.main.llm_client.stream_answer_with_context(
                        generation_question,
                        context_blocks,
                        session_guidance_prompt=session_guidance_prompt,
                    )
                except TypeError:
                    # 兼容旧测试客户端和外部适配器；生产客户端支持独立的临时要求参数。
                    answer_stream = self.main.llm_client.stream_answer_with_context(
                        generation_question,
                        context_blocks,
                    )
                async for stream_kind, delta in self._stream_sync_deltas(answer_stream):
                    if stream_kind == "heartbeat":
                        yield self._progress_metadata(
                            "document_rag", run_id, "generation", 50,
                            "正在等待模型返回首段回答", performance,
                        )
                        continue
                    if first_delta:
                        performance.set("llm_ttft_ms", (perf_counter() - llm_started) * 1000)
                        first_delta = False
                        yield self._progress_metadata(
                            "document_rag", run_id, "streaming", 60,
                            "模型已开始生成回答", performance,
                        )
                    # 云端偶尔会返回较大的文本块，拆成小块后浏览器才能稳定呈现渐进式输出。
                    for visible_delta in self._split_stream_delta(delta):
                        answer_parts.append(visible_delta)
                        yield {"type": "delta", "text": visible_delta}
                        await asyncio.sleep(0)
                performance.stop("llm_total_ms")
                if not answer_parts:
                    raise RuntimeError("云端模型返回空答案")
                llm_used = True
                llm_status = "called_with_local_evidence"
            except Exception as exc:
                # 首字前失败仍使用原兜底答案；已有部分内容时不再拼接重复模板。
                fallback = "" if answer_parts else (
                    self.main._document_answer_fallback(question, hits)
                    if hits
                    else self._task_context_fallback(task_context_blocks)
                )
                if fallback:
                    answer_parts.append(fallback)
                llm_status = f"failed: {exc}"
                if fallback:
                    yield {"type": "delta", "text": fallback}
        else:
            fallback = (
                self.main._document_answer_fallback(question, hits)
                if hits
                else self._task_context_fallback(task_context_blocks)
            )
            answer_parts.append(fallback)
            llm_status = "skipped_no_api_key"
            yield {"type": "delta", "text": fallback}
        common["answer"] = "".join(answer_parts)
        common["llm_used"] = llm_used
        common["llm_status"] = llm_status
        common["answer_renderer"] = "llm" if llm_used else "fallback"
        common["audit"]["session_feedback_guidance"].update({
            "renderer": "llm" if llm_used else "fallback",
            "applied": bool(session_guidance_prompt and llm_used),
        })
        yield self._progress_metadata(
            "document_rag", run_id, "persist", 98, "正在保存回答和证据", performance,
        )
        if session_id:
            self._persist_complete(
                session_id, common, status="completed", already_started=True, performance=performance,
            )
        yield self._completion_metadata(common, status="completed")
        yield self._done(common, status="completed")
        self._schedule_memory_refresh(session_id)

    @staticmethod
    def _structured_render_context(structured: dict[str, Any], common: dict[str, Any]) -> list[str]:
        """把结构化事实整理成回答模型上下文，不把内部任务信息暴露给模型输出。"""
        result = structured if isinstance(structured, dict) else {}
        blocks: list[str] = []
        answer = str(result.get("answer") or "").strip()
        if answer:
            blocks.append(
                "已核验的结构化查询结果（只能整理和润色，不得改动其中的数量、时间、名称或范围）：\n"
                + answer[:12000]
            )

        # 结构化字段比已经排版的答案更适合生成表格、分组和逐个例说明。
        safe_cases: list[dict[str, Any]] = []
        for item in list(common.get("evidence_cases") or [])[:40]:
            if not isinstance(item, dict):
                continue
            safe_cases.append({
                "title": str(item.get("title") or ""),
                "date_range": str(item.get("date_range") or ""),
                "disaster_types": list(item.get("disaster_types") or [])[:12],
                "affected_areas": list(item.get("affected_areas") or [])[:12],
                "summary": str(item.get("summary") or "")[:1200],
                "weather_facts": str(item.get("weather_facts") or "")[:1600],
                "forecast_focus": str(item.get("forecast_focus") or "")[:800],
                "source_pdf": str(item.get("source_pdf") or ""),
            })
        if safe_cases:
            blocks.append(
                "已核验的标准个例字段（可按用户要求整理成表格或分组）：\n"
                + json.dumps(safe_cases, ensure_ascii=False)
            )

        safe_chunks: list[str] = []
        for item in list(common.get("evidence_chunks") or [])[:24]:
            if not isinstance(item, dict):
                continue
            chunk = item.get("chunk") if isinstance(item.get("chunk"), dict) else item
            content = str(chunk.get("content") or "").strip()
            if content:
                safe_chunks.append(
                    f"来源：{str(chunk.get('source_pdf') or '')}\n{content[:2400]}"
                )
        if safe_chunks:
            blocks.append("可追溯证据片段（仅用于支持事实，不足时不得补写）：\n" + "\n\n".join(safe_chunks))

        return blocks or ["本轮结构化检索没有返回可供回答模型使用的事实内容。"]

    @staticmethod
    def _apply_same_scope_feedback_to_tasks(
        tasks: list[dict[str, Any]],
        session_guidances: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """把需要补证据的反馈作为同范围任务约束，不改变原问题筛选条件。"""
        normalized_tasks = [dict(task) for task in tasks]
        dimensions: list[str] = []
        guidance_ids: list[str] = []
        for guidance in session_guidances or []:
            if not guidance.get("needs_same_scope_retrieval"):
                continue
            for value in guidance.get("requested_dimensions") or []:
                text = re.sub(r"\s+", " ", str(value or "")).strip()[:60]
                if text and text not in dimensions:
                    dimensions.append(text)
            if guidance.get("guidance_id"):
                guidance_ids.append(str(guidance["guidance_id"]))
        if not dimensions:
            return normalized_tasks

        # 只扩展已有 RAG 任务；没有既有任务时不新造检索范围，避免反馈越权改变路由。
        target = next(
            (
                task
                for task in normalized_tasks
                if str(task.get("type") or "") == "case_analysis"
            ),
            None,
        )
        if target is None:
            return normalized_tasks
        dimension_text = "、".join(dimensions[:5])
        # 保留原始检索文本，反馈要求只作为证据完整性提示，不覆盖原查询。
        target["retrieval_question"] = str(target.get("question") or "")[:6000]
        target["question"] = (
            f"{str(target.get('question') or '').rstrip()}\n"
            f"在原有时间、地点、灾种和证据范围内，补充用户反馈要求的维度：{dimension_text}。"
        )[:6000]
        target["feedback_guidance_ids"] = guidance_ids[:5]
        return normalized_tasks

    def _retrieve_rag_task_plan(
        self,
        tasks: list[dict[str, Any]],
        context_case_ids: list[str],
        *,
        top_k: int,
        top_n: int,
    ) -> tuple[str, list[Any], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
        """按显式任务依赖传递个例集合，独立任务之间不共享隐式检索范围。"""
        all_hits: dict[str, Any] = {}
        context_blocks: list[str] = []
        task_results: list[dict[str, Any]] = []
        evidence_cases: list[dict[str, Any]] = []
        case_catalog = {str(case.case_id): case for case in self._standard_cases()}
        initial_case_ids = [case_id for case_id in context_case_ids if case_id in case_catalog]
        task_case_ids: dict[str, list[str]] = {}
        task_types: dict[str, str] = {}
        for task in tasks:
            task_id = str(task.get("id") or f"task_{len(task_results) + 1}")
            task_type = str(task.get("type") or "knowledge_qa")
            question = str(task.get("question") or "")
            retrieval_question = str(task.get("retrieval_question") or question)
            scope = str(task.get("evidence_scope") or "vector")
            conditions = dict(task.get("conditions") or {})
            dependency_ids = [str(value) for value in task.get("depends_on") or []]
            # 指标聚合等筛选型依赖已经把宽候选集合收窄，下游必须优先消费其结果，
            # 不能再与 case_identification 的全量集合做并集。
            narrowing_dependencies = [
                dependency_id
                for dependency_id in dependency_ids
                if task_types.get(dependency_id) == "metric_aggregation"
            ]
            active_dependencies = narrowing_dependencies or dependency_ids
            dependency_scope_locked = bool(narrowing_dependencies)
            dependency_case_ids: list[str] = []
            for dependency_id in active_dependencies:
                for case_id in task_case_ids.get(str(dependency_id), []):
                    if case_id not in dependency_case_ids:
                        dependency_case_ids.append(case_id)
            scoped_case_ids = dependency_case_ids if dependency_ids else list(initial_case_ids)
            if (
                conditions
                and not scoped_case_ids
                and not dependency_scope_locked
                and scope in {"case_chunks", "multi_case_chunks", "image_index", "same_pdf_related"}
            ):
                scoped_case_ids = [case.case_id for case in self._structured_cases_for_task(conditions)]
            scoped_cases = [case_catalog[case_id] for case_id in scoped_case_ids if case_id in case_catalog]
            source_pdfs = {
                str(case.source_pdf)
                for case in scoped_cases
                if getattr(case, "source_pdf", "")
            }
            structured_found = False
            resolved_case_ids = list(scoped_case_ids)
            structured_result: dict[str, Any] = {}
            if dependency_scope_locked and not scoped_case_ids:
                # 筛选型上游没有结果时，下游保持无结果，禁止无范围向量检索污染答案。
                hits = []
                retrieval_mode = "unresolved_dependency_scope"
            elif task_type == "metric_aggregation":
                # 指标聚合读取候选个例的全部关联正文，并把极值事实作为带类型的依赖结果传给下游。
                candidate_cases = scoped_cases or self._structured_cases_for_task(conditions)
                candidate_chunk_ids = {
                    str(chunk_id)
                    for case in candidate_cases
                    for chunk_id in getattr(case, "source_chunk_ids", [])
                }
                candidate_sources = {str(getattr(case, "source_pdf", "") or "") for case in candidate_cases}
                metric_chunks = [
                    chunk for chunk in self.document_store.list_chunks()
                    if str(getattr(chunk, "chunk_id", "")) in candidate_chunk_ids
                    or str(getattr(chunk, "source_pdf", "")) in candidate_sources
                ]
                chunk_case_ids: dict[str, list[str]] = {}
                for case in candidate_cases:
                    for chunk_id in getattr(case, "source_chunk_ids", []):
                        chunk_case_ids.setdefault(str(chunk_id), []).append(str(case.case_id))
                structured_result = aggregate_metric_facts(
                    metric_chunks,
                    metric=str(conditions.get("metric") or ""),
                    operator=str(conditions.get("operator") or ""),
                    chunk_case_ids=chunk_case_ids,
                )
                resolved_case_ids = list(structured_result.get("selected_case_ids") or [])
                hits = []
                retrieval_mode = "document_metric_facts"
                structured_found = structured_result.get("status") == "ok"
                if structured_found:
                    context_blocks.append(f"检索目标：{question}\n结构化查询结果：{metric_result_text(structured_result)}")
            elif scope == "structured_cases" and conditions:
                matched_cases = self._structured_cases_for_task(conditions, scoped_case_ids or None)
                resolved_case_ids = [str(case.case_id) for case in matched_cases]
                hits = []
                retrieval_mode = f"structured_task:{task_type}"
                structured_found = bool(matched_cases)
                evidence_text = self._structured_task_evidence(task_type, matched_cases, conditions)
                # 附属任务未命中只写审计，不把失败状态交给回答模型。
                if evidence_text and matched_cases:
                    context_blocks.append(f"检索目标：{question}\n结构化查询结果：{evidence_text}")
                evidence_cases.extend(case.to_dict() for case in matched_cases)
            elif task_type == "case_analysis" and scoped_case_ids:
                # 分析任务稍后按个例逐一检索，这里不先做一次混合范围检索，避免重复调用和串案例证据。
                hits = []
                retrieval_mode = "document_case_analysis_pending"
            elif scope in {"case_chunks", "image_index"} and scoped_case_ids:
                retrieval_mode, hits = self.main._retrieve_document_chunks(
                    retrieval_question, top_k=top_k, top_n=top_n, context_case_ids=scoped_case_ids,
                )
            elif scope == "multi_case_chunks":
                if len(scoped_case_ids) >= 2:
                    retrieval_mode, hits = self.main._retrieve_document_chunks(
                        retrieval_question, top_k=top_k, top_n=top_n, context_case_ids=scoped_case_ids,
                    )
                else:
                    # 比较对象没有通过上游任务解析时，不允许回退为无范围向量检索。
                    retrieval_mode, hits = "unresolved_comparison_scope", []
            elif scope == "same_pdf_related" and source_pdfs:
                candidates = [chunk for chunk in self.document_store.list_chunks() if str(chunk.source_pdf) in source_pdfs]
                # 预警类任务优先保留包含目标日期、预报或预警词的 chunk，并扩大候选数量。
                hits = self._rank_related_evidence_chunks(question, candidates, max(top_n, 8))
                retrieval_mode = "document_same_pdf_related"
            elif scope == "structured_cases":
                hits = []
                retrieval_mode = "structured_cases"
                structured = self.main._answer_structured_question(
                    question, context_case_ids=scoped_case_ids or None,
                )
                if structured:
                    structured_found = True
                    answer = str(structured.get("answer") or "")
                    structured_cases = list(structured.get("evidence_cases") or [])
                    if answer and structured_cases:
                        context_blocks.append(f"检索目标：{question}\n结构化查询结果：{answer}")
                    for case in structured.get("evidence_cases") or []:
                        if isinstance(case, dict):
                            evidence_cases.append(case)
                            case_id = str(case.get("case_id") or "")
                            if case_id and case_id in case_catalog and case_id not in resolved_case_ids:
                                resolved_case_ids.append(case_id)
            else:
                retrieval_mode, hits = self.main._retrieve_document_chunks(
                    question, top_k=top_k, top_n=top_n, context_case_ids=None,
                )
            # case_analysis 必须按个例分别取证，避免多个过程的正文混在一个 top-n 列表中，
            # 导致某个个例没有任何 chunk 却仍被当作“已分析”。最终回答格式不变，
            # 这里只扩充内部证据上下文和审计字段。
            case_evidence: list[dict[str, Any]] = []
            if task_type == "case_analysis" and scoped_case_ids:
                per_case_hits: list[Any] = []
                for case_id in scoped_case_ids:
                    case_mode, case_hits = self.main._retrieve_document_chunks(
                        retrieval_question,
                        top_k=top_k,
                        top_n=top_n,
                        context_case_ids=[case_id],
                    )
                    if case_hits:
                        per_case_hits.extend(case_hits)
                    case_evidence.append({
                        "case_id": case_id,
                        "retrieval_mode": case_mode,
                        "evidence_chunk_ids": [str(hit.chunk.chunk_id) for hit in case_hits],
                    })
                # 保留主检索函数的去重与顺序，同时确保每个个例的结果都进入最终上下文。
                hits = per_case_hits
                retrieval_mode = "document_case_analysis_by_case"

            hit_ids = []
            for hit in hits:
                chunk_id = str(hit.chunk.chunk_id)
                all_hits.setdefault(chunk_id, hit)
                hit_ids.append(chunk_id)
                source_blocks = self.main._context_blocks_from_document_hits([hit])
                for source_block in source_blocks:
                    case_label = ""
                    if task_type == "case_analysis":
                        matched_case = next(
                            (item["case_id"] for item in case_evidence
                             if chunk_id in item.get("evidence_chunk_ids", [])),
                            "",
                        )
                        case_label = f"\n对应个例：{matched_case}" if matched_case else ""
                    block = f"检索目标：{question}{case_label}\n{source_block}"
                    if block not in context_blocks:
                        context_blocks.append(block)
            task_case_ids[task_id] = resolved_case_ids
            task_types[task_id] = task_type
            task_results.append({
                "id": task_id,
                "type": task_type,
                "route": "rag",
                "question": question,
                "depends_on": list(task.get("depends_on") or []),
                "evidence_scope": scope,
                "conditions": conditions,
                "retrieval_mode": retrieval_mode,
                "status": "ok" if hits or structured_found else "no_hits",
                "evidence_chunk_ids": hit_ids,
                "resolved_case_ids": resolved_case_ids,
                "case_evidence": case_evidence,
                "structured_result": structured_result,
            })
        for case_id in initial_case_ids:
            evidence_cases.append(case_catalog[case_id].to_dict())
        ordered_hits = list(all_hits.values())
        unique_evidence_cases: list[dict[str, Any]] = []
        seen_case_ids: set[str] = set()
        for case in evidence_cases:
            case_id = str(case.get("case_id") or "")
            if not case_id or case_id in seen_case_ids:
                continue
            seen_case_ids.add(case_id)
            unique_evidence_cases.append(case)
        return "document_task_plan", ordered_hits, context_blocks, task_results, unique_evidence_cases

    def _structured_cases_for_task(
        self,
        conditions: dict[str, Any],
        scoped_case_ids: list[str] | None = None,
    ) -> list[Any]:
        """直接使用模型归一化条件查询标准个例，不再解析任务中文。"""
        allowed = {
            key: conditions.get(key)
            for key in ("start_date", "end_date", "years", "months", "disaster_types", "cities", "areas")
            if conditions.get(key)
        }
        query = CaseSearchQuery(**allowed, limit=500)
        candidates = self._standard_cases()
        if scoped_case_ids:
            wanted = set(scoped_case_ids)
            candidates = [case for case in candidates if str(case.case_id) in wanted]
        original_order = {str(case.case_id): index for index, case in enumerate(candidates)}
        matched_cases = [match.case for match in self.structured_case_retriever.search(candidates, query)]
        return sorted(matched_cases, key=lambda case: original_order.get(str(case.case_id), len(original_order)))

    def _structured_task_evidence(
        self,
        task_type: str,
        cases: list[Any],
        conditions: dict[str, Any],
    ) -> str:
        """把同一命中集合投影成计数或清单，供最终回答模型按任务消费。"""
        if not cases:
            return "当前标准化个例库中没有命中给定结构化条件的个例。"
        if task_type == "aggregate_statistics":
            return f"命中个例总数：{len(cases)}。计数对象为上游筛选出的同一组标准化个例。"
        lines = [
            f"- {case.case_id}｜{case.date_range}｜{case.title}｜灾种：{'、'.join(case.disaster_types) or '未标注'}｜来源：{case.source_pdf}"
            for case in cases
        ]
        if task_type == "case_identification":
            heading = f"已定位 {len(cases)} 个符合条件的标准化个例，以下 case_id 作为依赖任务的输入："
        elif task_type == "case_listing":
            heading = f"按用户要求列出 {len(cases)} 个个例的名称和发生时间："
        else:
            heading = f"结构化检索共命中 {len(cases)} 个个例："
        coverage = self._structured_coverage_note(conditions)
        return "\n".join([heading, *lines, *([coverage] if coverage else [])])

    def _structured_coverage_note(self, conditions: dict[str, Any]) -> str:
        """区分用户请求范围和当前库内记录跨度，避免把部分覆盖表述成完整全年。"""
        if conditions.get("time_scope_type") != "calendar_year" or not conditions.get("years"):
            return ""
        knowledge = self.intent_agent.knowledge_range()
        years = "、".join(str(value) for value in conditions.get("years") or [])
        return (
            f"范围说明：用户请求的是 {years} 完整自然年；当前标准个例记录跨度为 "
            f"{knowledge.start_date or '未知'} 至 {knowledge.end_date or '未知'}。"
            "上述数量只代表当前库内已收录个例，不能据此宣称未覆盖时段没有相关过程。"
        )

    @staticmethod
    def _task_context_fallback(context_blocks: list[str]) -> str:
        """模型不可用时仅返回各结构化子任务的已核验结果。"""
        answers = [
            block.split("结构化查询结果：", 1)[1].strip()
            for block in context_blocks
            if "结构化查询结果：" in block
        ]
        return "\n\n".join(answers) or "已找到相关证据，但当前无法生成综合回答。"

    @staticmethod
    def _rank_related_evidence_chunks(question: str, chunks: list[Any], limit: int) -> list[Any]:
        """按日期与证据类型定位同源关联材料，并保留表格跨 chunk 的相邻行。"""
        normalized_question = re.sub(r"\s+", "", question)
        date_terms: list[str] = []
        for year, month, day in re.findall(r"(?:(20\d{2})年)?(\d{1,2})月(\d{1,2})日?", normalized_question):
            date_terms.extend([
                f"{int(month)}月{int(day)}日",
                f"{int(month):02d}月{int(day):02d}日",
                f"{int(month):02d}月{int(day):02d}",
            ])
            if year:
                date_terms.append(f"{year}年{int(month)}月{int(day)}日")
        evidence_terms = [
            term for term in (
                "预报预警服务", "预警服务", "预报", "预警", "重要气象信息", "发布时间",
                "灾害实况", "灾情", "影响区域", "环流形势", "天气成因", "图", "图片",
            )
            if term in normalized_question
        ]
        scored: list[tuple[int, int, Any]] = []
        dated_evidence_indices: list[int] = []
        for index, chunk in enumerate(chunks):
            content = re.sub(r"\s+", "", str(chunk.content or ""))
            date_score = sum(12 for term in date_terms if term in content)
            evidence_score = sum(max(2, len(term)) for term in evidence_terms if term in content)
            score = date_score + evidence_score
            if date_score > 0 and evidence_score > 0:
                dated_evidence_indices.append(index)
            scored.append((score, index, chunk))
        # 日期型预报预警查询优先返回完整表格区间，防止时间、预警类型和区域被拆散。
        if dated_evidence_indices and any(term in evidence_terms for term in ("预报", "预警", "预警服务", "预报预警服务")):
            last_anchor = max(dated_evidence_indices)
            headings = [
                index for index in range(last_anchor, -1, -1)
                if any(term in str(chunks[index].content or "") for term in ("预警服务情况", "预报预警服务"))
            ]
            start = headings[0] if headings else max(0, min(dated_evidence_indices) - 1)
            relevant_anchors = [index for index in dated_evidence_indices if index >= start]
            end = min(len(chunks), max(relevant_anchors or dated_evidence_indices) + 2)
            if end - start <= max(limit, 12):
                score_by_index = {index: score for score, index, _chunk in scored}
                return [
                    DocumentRetrievalHit(chunk=chunks[index], score=float(score_by_index.get(index, 0)))
                    for index in range(start, end)
                ]
        positive = sorted((item for item in scored if item[0] > 0), key=lambda item: (-item[0], item[1]))
        selected_indices: set[int] = set()
        for _score, index, _chunk in positive:
            selected_indices.add(index)
            # 表格的时间、预警类型和区域经常跨越前后两个 chunk，必须成组保留。
            if index > 0:
                selected_indices.add(index - 1)
            if index + 1 < len(chunks):
                selected_indices.add(index + 1)
            if len(selected_indices) >= limit:
                break
        if not selected_indices:
            selected_indices.update(range(min(limit, len(chunks))))
        ordered = sorted(selected_indices)[:limit]
        score_by_index = {index: score for score, index, _chunk in scored}
        return [
            DocumentRetrievalHit(chunk=chunks[index], score=float(score_by_index.get(index, 0)))
            for index in ordered
        ]

    def _route(self, question: str, has_pending: bool, memory_context: dict[str, Any]) -> UnifiedRouteDecision:
        """调用新版意图 Agent，并把待确认操作固定留在多维链路。"""
        if has_pending and (self._is_confirmation(question) or self._is_cancellation(question)):
            return UnifiedRouteDecision("multidim_search", "structured", question)
        try:
            result = self.intent_agent.route(IntentRouteRequest(
                message=question,
                previous_intent=memory_context.get("previous_intent"),
                memory_context=memory_context,
            ))
            # 待确认时，明确修改词或短条件表达继续交给多维条件解析器。
            modifies_pending = any(term in question for term in ("改", "换", "调整", "不要", "去掉", "不限", "增加", "补充"))
            short_condition = bool(result.conditions) and len(str(question).strip()) <= 30
            if has_pending and (modifies_pending or short_condition):
                return UnifiedRouteDecision("multidim_search", "structured", question)
            trace = {
                "audit_id": result.audit_id,
                "context_related": result.context_related,
                "relation_type": result.relation_type,
                "context_confidence": result.context_confidence,
                "context_resolution_source": result.context_resolution_source,
                "context_message_ids": result.context_message_ids,
                "reference_type": result.reference_type,
                "reference_clues": result.reference_clues,
                "routing_source": result.routing_source,
                "reason": result.reason,
            }
            tasks = [item.model_dump() for item in getattr(result, "tasks", [])]
            trace["task_plan"] = tasks
            referenced_case_ids = self._valid_knowledge_case_ids(result.referenced_case_ids)
            intent = str(result.intent)
            rag_strategy = str(result.rag_strategy or "vector")
            # 执行前最后复核：已锁定库内历史个例时绝不调用 Smart，而是限定对应 chunk 交给主 RAG。
            if referenced_case_ids and intent == "similar_case_match":
                intent = "rag"
                rag_strategy = "vector"
                trace["orchestrator_guard"] = "knowledge_case_forced_to_rag"
            trace["validated_knowledge_case_ids"] = referenced_case_ids
            return UnifiedRouteDecision(
                intent,
                rag_strategy,
                str(result.normalized_message or question),
                str(result.clarification_question or ""),
                referenced_case_ids,
                trace,
                tasks,
            )
        except Exception:
            # 兼容意图 Agent 暂不可用时，保留旧规则作为安全兜底。
            text = str(question or "")
            if "报告" in text and any(term in text for term in ("生成", "导出", "制作")):
                return UnifiedRouteDecision("multidim_search", "vector", text)
            if any(term in text for term in ("相似个例", "相似过程", "参考经验")):
                return UnifiedRouteDecision("similar_case_match", "vector", text)
            return UnifiedRouteDecision("rag", "vector", text)

    def _standard_cases(self) -> list[Any]:
        """向记忆和意图模块提供主项目当前标准个例，避免各模块维护第二份实体集合。"""
        store = getattr(self.main, "standard_case_store", None)
        return list(store.list_cases()) if store is not None else []

    def _valid_knowledge_case_ids(self, case_ids: list[str] | None) -> list[str]:
        """按主标准个例库过滤和去重模型返回的 ID，非法 ID 不得进入下游检索。"""
        valid_ids = {str(case.case_id) for case in self._standard_cases() if getattr(case, "case_id", None)}
        result: list[str] = []
        for case_id in case_ids or []:
            value = str(case_id)
            if value in valid_ids and value not in result:
                result.append(value)
        return result

    async def _stream_clarification(
        self,
        question: str,
        session_id: str,
        answer: str,
        performance: PerformanceTrace,
    ) -> AsyncIterator[dict[str, Any]]:
        """把意图冲突追问作为普通聊天消息返回并写入主会话。"""
        text = answer or "请确认本轮要生成历史个例报告，还是分析当前过程并匹配相似历史个例。"
        common = self._empty_result("query_intent", f"query_intent_{uuid4().hex}", question, text)
        common["retrieval_mode"] = "agent:query_intent_clarification"
        if session_id:
            self._persist_complete(session_id, common, status="needs_clarification", performance=performance)
        yield self._metadata(common, status="needs_clarification")
        async for event in self._stream_text_chunks(text):
            yield event
        yield self._done(common, status="needs_clarification")
        self._schedule_memory_refresh(session_id)

    @staticmethod
    def _is_previous_result_report_request(question: str, memory_context: dict[str, Any]) -> bool:
        """判断报告动作是否明确承接当前会话，而非发起新的正式检索报告。"""
        text = re.sub(r"\s+", "", str(question or "")).lower()
        report_action = any(term in text for term in ("报告", "pdf", "导出", "整理"))
        context_cues = (
            "上面", "上一轮", "刚才", "前面", "上述", "这些", "这几个", "这8", "这八",
            "它们", "前述", "刚刚",
        )
        # 结果后处理经常直接说“这个分析/结果”，不一定出现“上一轮”等时间词。
        # 这里匹配完整的动作结构，避免单独出现“这个”时误把新报告当成复用报告。
        result_phrases = (
            "这个分析", "该分析", "这个结果", "该结果", "这个汇总", "该汇总",
            "这份分析", "这份结果", "这个回答", "这个答案", "以上分析", "以上结果",
        )
        structured_reuse = any(phrase in text for phrase in result_phrases) or bool(
            re.search(r"(?:把|将).{0,16}(?:分析|结果|汇总|答案|回答).{0,16}(?:成|为|整理|导出|生成).{0,12}(?:pdf|报告)", text)
        )
        # 是否确有可复用结果由调用方通过当前 session 的原始助手消息确认；
        # 这里不依赖压缩记忆中的 case_refs，避免记忆尚未刷新时漏判承接请求。
        return report_action and (any(cue in text for cue in context_cues) or structured_reuse)

    @staticmethod
    def _is_report_action(question: str) -> bool:
        """判断一条历史用户消息是否曾要求生成报告，用于过滤失败的回退回答。"""
        text = re.sub(r"\s+", "", str(question or "")).lower()
        return any(term in text for term in ("报告", "pdf", "导出", "整理成", "汇总成"))

    def _previous_reusable_result(
        self,
        session_id: str,
        question: str,
        memory_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """从当前 session 的原始助手消息恢复上一轮完整结果，不依赖压缩记忆正文。"""
        if not session_id:
            return None
        messages = self.session_store.list_messages(session_id)
        # 统一入口尚未写入本轮用户消息，因此倒序遇到的第一条可复用助手消息就是上一轮结果。
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if str(message.get("role") or "") != "assistant":
                continue
            metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
            merged = dict(metadata)
            for key, value in message.items():
                if key not in {"metadata", "content", "role", "message_id", "session_id", "created_at"}:
                    merged.setdefault(key, value)
            status = str(merged.get("status") or "")
            if status in {"awaiting_confirmation", "needs_clarification", "cancelled"}:
                continue
            # 若该助手消息对应的用户问题本身要求 PDF，但没有落库报告，
            # 它就是一次失败的普通 RAG 回退，不能遮蔽更早的真实分析结果。
            source_question = str(
                merged.get("original_question")
                or merged.get("question")
                or (merged.get("memory") or {}).get("question")
                or ""
            )
            if not source_question:
                for prior in reversed(messages[:index]):
                    if str(prior.get("role") or "") == "user":
                        source_question = str(prior.get("content") or "")
                        break
            reports = list(merged.get("reports") or [])
            if self._is_report_action(source_question) and not reports:
                continue
            if str(merged.get("agent_type") or "") == "conversation_report" and not reports:
                continue
            cases = list(merged.get("evidence_cases") or [])
            agent_result = merged.get("agent_result")
            if not cases and isinstance(agent_result, dict):
                cases = list(agent_result.get("evidence_cases") or agent_result.get("matched_cases") or [])
            answer = str(merged.get("answer") or message.get("content") or "").strip()
            if not cases and not answer and not merged.get("analysis"):
                continue
            result = {
                "source_message_id": int(message.get("message_id") or 0),
                "answer": answer,
                "agent_type": str(merged.get("agent_type") or ""),
                "run_id": str(merged.get("run_id") or ""),
                "evidence_cases": cases,
                "evidence_chunks": list(merged.get("evidence_chunks") or []),
                "images": list(merged.get("images") or []),
                "analysis": merged.get("analysis") or {},
                "task_plan": list(merged.get("task_plan") or []),
                "agent_result": agent_result if isinstance(agent_result, dict) else {},
                "question": str(merged.get("question") or ""),
            }
            return result
        return None

    async def _stream_previous_result_report(
        self,
        question: str,
        session_id: str,
        previous_result: dict[str, Any],
        performance: PerformanceTrace,
    ) -> AsyncIterator[dict[str, Any]]:
        """只将上一轮结果排版为简易 PDF，不重新检索或调用逐例分析模型。"""
        run_id = f"conversation_report_{uuid4().hex}"
        report_title = self._conversation_report_title(previous_result)
        source_answer = str(previous_result.get("answer") or "").strip()
        common = {
            "agent_type": "conversation_report",
            "run_id": run_id,
            "question": question,
            "answer": "已复用上一轮个例分析结果，正在整理为简易 PDF 报告。",
            "retrieval_mode": "conversation_reuse",
            "hit_count": len(previous_result.get("evidence_cases") or []),
            "hits": [],
            "evidence_cases": list(previous_result.get("evidence_cases") or []),
            "evidence_chunks": list(previous_result.get("evidence_chunks") or []),
            # 报告后处理只返回 PDF，不在当前消息再次铺开上一轮图片。
            "images": [],
            "visuals": [],
            "reports": [],
            "analysis": dict(previous_result.get("analysis") or {}),
            "task_plan": list(previous_result.get("task_plan") or []),
            "agent_result": dict(previous_result.get("agent_result") or {}),
            "report_mode": "from_previous_result",
            "source_message_id": previous_result.get("source_message_id"),
            "source_run_id": previous_result.get("run_id", ""),
            "source_content_sha256": hashlib.sha256(source_answer.encode("utf-8")).hexdigest(),
            "report_title": report_title,
            "llm_used": False,
            "llm_status": "reused_previous_result",
        }
        if not session_id:
            yield {"type": "error", "message": "简易报告必须先创建主页面会话"}
            return
        self.session_store.start_agent_run(run_id, session_id, "conversation_report", question)
        yield self._metadata(common, status="running")
        # 先发送可见反馈，再在线程中执行本地 PDF 排版，避免页面等待到文件完成才有响应。
        async for event in self._stream_text_chunks(common["answer"], chunk_size=36):
            yield event
        try:
            report_path = await asyncio.to_thread(
                self.multidim_agent.export_pdf_from_previous_result,
                previous_result,
                filename=f"{run_id}.pdf",
                report_title=report_title,
            )
            common["reports"] = [self._report_payload(
                report_path,
                report_id=run_id,
                title=report_title,
                report_mode="from_previous_result",
            )]
            common["answer"] = "已根据上一轮分析结果生成简易 PDF 汇总报告。"
            self._persist_complete(
                session_id,
                common,
                status="completed",
                already_started=True,
                performance=performance,
            )
            yield self._completion_metadata(common, status="completed")
            yield self._done(common, status="completed")
            self._schedule_memory_refresh(session_id)
        except Exception as exc:
            self.session_store.finish_agent_run(run_id, status="failed", metadata={"error": str(exc)})
            yield {"type": "error", "message": f"简易 PDF 生成失败：{exc}", "run_id": run_id}

    @staticmethod
    def _conversation_report_title(previous_result: dict[str, Any]) -> str:
        """依据上一轮可见问题和回答生成简洁标题，不额外调用模型。"""
        result = previous_result if isinstance(previous_result, dict) else {}
        question = re.sub(r"\s+", "", str(result.get("question") or ""))
        answer = str(result.get("answer") or "")
        years = list(dict.fromkeys(re.findall(r"(20\d{2})年", question)))
        months = [
            int(value)
            for value in dict.fromkeys(re.findall(r"(?<!\d)(\d{1,2})月", question))
            if 1 <= int(value) <= 12
        ]
        if len(years) == 1 and len(months) >= 2:
            return f"{years[0]}年{months[0]}月至{months[-1]}月气象灾害分析报告"
        if len(years) == 1 and len(months) == 1:
            return f"{years[0]}年{months[0]}月气象灾害分析报告"
        if len(years) == 1:
            return f"{years[0]}年气象灾害分析报告"
        heading = re.search(r"^#{1,3}\s+(.+)$", answer, flags=re.MULTILINE)
        if heading:
            title = re.sub(r"[*_`#]", "", heading.group(1)).strip(" ：:。")
            if title and title not in {"核心结论", "统计结论", "分析结论", "主要过程"}:
                return title if title.endswith("报告") else f"{title}报告"
        return "气象灾害分析报告"

    @staticmethod
    def _is_confirmation(question: str) -> bool:
        text = str(question or "").strip()
        # 页面确认卡常用“确认”或“确定”，两者都必须复用当前提案，不能退回普通 RAG。
        return text in {"确认", "确定", "确认条件", "确定条件", "开始检索", "执行检索", "按此检索", "生成吧"} or text.startswith(("确认", "确定"))

    @staticmethod
    def _is_cancellation(question: str) -> bool:
        return str(question or "").strip() in {"取消", "取消检索", "放弃", "重新输入"}

    def _structured_common(self, result: dict, run_id: str, question: str) -> dict:
        """把标准个例结构化回答转换成统一聊天结果。"""
        return {
            "agent_type": "document_rag",
            "run_id": run_id,
            "question": question,
            "answer": result.get("answer", ""),
            "retrieval_mode": result.get("retrieval_mode", "structured_cases"),
            "hit_count": result.get("hit_count", 0),
            "hits": result.get("hits", []),
            "evidence_cases": result.get("evidence_cases", []),
            "evidence_chunks": result.get("evidence_chunks", []),
            "images": result.get("images", []),
            "image_display_mode": result.get("image_display_mode", "hidden"),
            "visuals": result.get("visuals", []),
            "reports": [],
            "analysis": result.get("analysis", {}),
            "query_conditions": result.get("query_conditions", {}),
            "intent_trace": result.get("intent_trace", {}),
            "execution_plan": result.get("execution_plan", {}),
            "audit": result.get("audit", {}),
            "llm_used": bool(result.get("llm_used", False)),
        }

    def _persist_complete(
        self,
        session_id: str,
        common: dict,
        *,
        status: str,
        already_started: bool = False,
        performance: PerformanceTrace | None = None,
    ) -> dict:
        """把统一结果同时写入 Agent 运行表和消息元数据。"""
        if performance:
            performance.start("persist_ms")
            common["performance"] = performance.snapshot()
        run_id = str(common.get("run_id") or f"agent_run_{uuid4().hex}")
        if not already_started:
            self.session_store.start_agent_run(
                run_id,
                session_id,
                str(common.get("agent_type") or "unknown"),
                str(common.get("question") or ""),
                status=status,
            )
        common.update(self._regeneration_context)
        if not common.get("source_user_message_id"):
            # 普通回答以本轮用户消息作为稳定版本组，后续重生成只追加版本而不新增用户消息。
            source_user = next(
                (item for item in reversed(self.session_store.list_messages(session_id)) if item.get("role") == "user"),
                None,
            )
            source_user_message_id = int((source_user or {}).get("message_id") or 0)
            common["source_user_message_id"] = source_user_message_id or None
            common["version_group_id"] = f"user-{source_user_message_id}" if source_user_message_id else ""
        metadata = self._message_metadata(common, status)
        self.session_store.finish_agent_run(
            run_id,
            status=status,
            answer=str(common.get("answer") or ""),
            evidence_cases=list(common.get("evidence_cases") or []),
            evidence_chunks=list(common.get("evidence_chunks") or []),
            images=list(common.get("images") or []),
            reports=list(common.get("reports") or []),
            metadata=metadata,
        )
        saved_message = self.session_store.add_message(
            session_id,
            "assistant",
            str(common.get("answer") or ""),
            metadata=metadata,
        )
        common["message_id"] = int(saved_message.get("message_id") or 0) or None
        if performance:
            performance.stop("persist_ms")
            common["performance"] = performance.snapshot()
            LOGGER.info("[统一聊天性能] run_id=%s timings=%s", run_id, common["performance"])
        return saved_message

    def _message_metadata(self, common: dict, status: str) -> dict:
        """构建会话恢复需要的顶层字段和统一记忆包。"""
        fields = {
            key: common.get(key, default)
            for key, default in (
                ("agent_type", ""),
                ("run_id", ""),
                # 待确认的多维提案需要把编号写入消息元数据，跨请求恢复时才能精确确认。
                ("proposal_id", ""),
                ("proposal_operation", ""),
                ("proposal_question", ""),
                ("retrieval_mode", ""),
                ("evidence_cases", []),
                ("evidence_chunks", []),
                ("images", []),
                ("image_display_mode", "hidden"),
                ("visuals", []),
                ("reports", []),
                ("analysis", {}),
                ("query_conditions", {}),
                ("intent_trace", {}),
                ("execution_plan", {}),
                ("task_plan", []),
                ("audit", {}),
                ("agent_result", {}),
                ("original_question", ""),
                ("resolved_question", ""),
                ("context_case_ids", []),
                ("memory_trace", {}),
                ("report_mode", "none"),
                ("source_message_id", None),
                ("source_run_id", ""),
                ("llm_used", False),
                ("llm_status", ""),
                ("answer_renderer", ""),
                ("performance", {}),
                ("follow_up_suggestions", []),
                ("follow_up_status", "none"),
                ("regenerated_from_message_id", None),
                ("regeneration_instruction", ""),
                ("source_user_message_id", None),
                ("version_group_id", ""),
                ("version", 1),
            )
        }
        fields["status"] = status
        fields["memory"] = {"version": 2, "question": common.get("question", ""), "answer": common.get("answer", ""), **fields}
        return fields

    def _metadata(self, common: dict, *, status: str) -> dict:
        """生成统一 metadata 事件。"""
        return {
            "type": "metadata",
            "status": status,
            **{key: value for key, value in common.items() if key != "answer"},
        }

    def _done(self, common: dict, *, status: str) -> dict:
        """生成统一 done 事件，包含运行号和报告地址。"""
        return {
            "type": "done",
            "status": status,
            "agent_type": common.get("agent_type", ""),
            "run_id": common.get("run_id", ""),
            "llm_used": bool(common.get("llm_used", False)),
            "reports": list(common.get("reports") or []),
            "message_id": common.get("message_id"),
            "follow_up_suggestions": list(common.get("follow_up_suggestions") or []),
            "follow_up_status": common.get("follow_up_status", "none"),
            "regenerated_from_message_id": common.get("regenerated_from_message_id"),
            "regeneration_instruction": common.get("regeneration_instruction", ""),
            "source_user_message_id": common.get("source_user_message_id"),
            "version_group_id": common.get("version_group_id", ""),
            "version": int(common.get("version") or 1),
            "performance": dict(common.get("performance") or {}),
        }

    @staticmethod
    def _completion_metadata(common: dict, *, status: str) -> dict:
        """正文流结束后发送轻量后处理结果，不重复传输大块证据。"""
        return {
            "type": "metadata",
            "status": status,
            "agent_type": common.get("agent_type", ""),
            "run_id": common.get("run_id", ""),
            "message_id": common.get("message_id"),
            "reports": list(common.get("reports") or []),
            "follow_up_suggestions": list(common.get("follow_up_suggestions") or []),
            "follow_up_status": common.get("follow_up_status", "none"),
            "regenerated_from_message_id": common.get("regenerated_from_message_id"),
            "regeneration_instruction": common.get("regeneration_instruction", ""),
            "source_user_message_id": common.get("source_user_message_id"),
            "version_group_id": common.get("version_group_id", ""),
            "version": int(common.get("version") or 1),
        }

    @staticmethod
    async def _stream_sync_deltas(iterator) -> AsyncIterator[tuple[str, str]]:
        """在线程中消费同步生成器，并在模型等待期间持续发出心跳。"""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=32)

        def produce() -> None:
            """阻塞读取云端流，并通过线程安全队列通知事件循环。"""
            try:
                for value in iterator:
                    asyncio.run_coroutine_threadsafe(queue.put(("delta", value)), loop).result()
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(("error", exc)), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop).result()

        producer = asyncio.create_task(asyncio.to_thread(produce))
        try:
            while True:
                try:
                    kind, value = await asyncio.wait_for(queue.get(), timeout=0.75)
                except asyncio.TimeoutError:
                    yield "heartbeat", ""
                    continue
                if kind == "done":
                    break
                if kind == "error":
                    raise value
                if value:
                    yield "delta", str(value)
        finally:
            await producer

    @staticmethod
    async def _stream_text_chunks(text: str, chunk_size: int = 48) -> AsyncIterator[dict[str, str]]:
        """把已生成答案按可读片段发送，保持拼接后的正文逐字完全一致。"""
        value = str(text or "")
        start = 0
        punctuation = "。！？!?；;\n"
        while start < len(value):
            end = min(len(value), start + chunk_size)
            if end < len(value):
                # 在片段后半段优先寻找自然边界，避免页面刷新时停在半句话中。
                boundary = max((value.rfind(mark, start + chunk_size // 2, end + 1) for mark in punctuation), default=-1)
                if boundary >= start:
                    end = boundary + 1
            yield {"type": "delta", "text": value[start:end]}
            start = end
            await asyncio.sleep(0)

    @staticmethod
    def _split_stream_delta(text: str, chunk_size: int = 32) -> list[str]:
        """将云端单次返回的大块文本拆成稳定的小增量，不改变字符顺序和内容。"""
        value = str(text or "")
        return [value[index:index + chunk_size] for index in range(0, len(value), chunk_size)] or [""]

    def _schedule_memory_refresh(self, session_id: str) -> None:
        """创建受引用保护的后台记忆刷新任务，并隔离增强功能异常。"""
        if not session_id:
            return

        async def refresh() -> None:
            """在线程池中执行 SQLite 增量摘要刷新。"""
            try:
                await asyncio.to_thread(self.memory_manager.refresh, session_id)
            except Exception as exc:
                LOGGER.warning("[会话记忆] 后台刷新失败 session_id=%s error=%s", session_id, exc)

        task = asyncio.create_task(refresh())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def wait_for_background_tasks(self) -> None:
        """等待当前实例的后台增强任务，供测试和进程优雅退出使用。"""
        tasks = list(self._background_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _progress_signature(progress: dict | None) -> tuple:
        """生成进度去重键，避免轮询期间重复推送相同状态。"""
        if not progress:
            return ()
        return (
            progress.get("stage"),
            progress.get("percent"),
            progress.get("message"),
            progress.get("completed_cases"),
            progress.get("total_cases"),
        )

    @staticmethod
    def _progress_metadata(
        agent_type: str,
        run_id: str,
        stage: str,
        percent: int,
        message: str,
        performance: PerformanceTrace,
        *,
        extra: dict | None = None,
    ) -> dict[str, Any]:
        """构建三个处理链路共用的实时进度事件。"""
        progress = {**dict(extra or {}), "stage": stage, "percent": percent, "message": message}
        return {
            "type": "metadata",
            "agent_type": agent_type,
            "run_id": run_id,
            "status": "running",
            "progress": progress,
            "performance": performance.snapshot(),
        }

    def _empty_result(self, agent_type: str, run_id: str, question: str, answer: str) -> dict:
        """创建确认和取消阶段使用的最小统一结果。"""
        return {
            "agent_type": agent_type,
            "run_id": run_id,
            "question": question,
            "answer": answer,
            "retrieval_mode": f"agent:{agent_type}",
            "hit_count": 0,
            "hits": [],
            "evidence_cases": [],
            "evidence_chunks": [],
            "images": [],
            "visuals": [],
            "reports": [],
            "analysis": {},
            "intent_trace": {"route": agent_type},
            "execution_plan": {"agent": agent_type, "flow_preserved": True},
            "audit": {},
            "llm_used": False,
            "llm_status": "not_applicable",
        }

    def _attach_memory_trace(
        self,
        common: dict[str, Any],
        original_question: str,
        resolved_question: str,
        context_case_ids: list[str],
        route_trace: dict[str, Any] | None,
    ) -> None:
        """把问题改写和历史引用链写入统一元数据，便于审计与会话恢复。"""
        common["original_question"] = original_question
        common["resolved_question"] = resolved_question
        common["context_case_ids"] = list(context_case_ids)
        common["memory_trace"] = dict(route_trace or {})

    def _context_evidence_cases(self, case_ids: list[str]) -> list[dict[str, Any]]:
        """为指代检索补回标准个例实体，保证后续轮次仍能继续引用。"""
        store = getattr(self.main, "standard_case_store", None)
        converter = getattr(self.main, "_standard_case_to_response", None)
        if not case_ids or store is None or converter is None:
            return []
        wanted = set(case_ids)
        return [converter(case) for case in store.list_cases() if case.case_id in wanted]

    def _evidence_cases_from_document_hits(self, hits: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
        """把普通向量命中的 chunk 反向关联标准个例，供后续轮次可靠指代。"""
        store = getattr(self.main, "standard_case_store", None)
        converter = getattr(self.main, "_standard_case_to_response", None)
        if store is None or converter is None:
            return []
        cases = store.list_cases()
        result = []
        seen = set()
        for hit in hits:
            chunk = hit.get("chunk") if isinstance(hit, dict) else {}
            chunk_id = str((chunk or {}).get("chunk_id") or "")
            source_pdf = str((chunk or {}).get("source_pdf") or "")
            for case in cases:
                if case.case_id in seen:
                    continue
                if chunk_id in case.source_chunk_ids or (not case.source_chunk_ids and source_pdf == case.source_pdf):
                    seen.add(case.case_id)
                    result.append(converter(case))
                    if len(result) >= limit:
                        return result
        return result

    def _images_from_hits(self, hits: list[dict], limit: int = 6) -> list[dict]:
        """从普通 RAG 命中中提取去重图片。"""
        images = []
        seen = set()
        for hit in hits:
            for image in hit.get("images") or []:
                image_id = str(image.get("image_id") or "")
                if not image_id or image_id in seen:
                    continue
                seen.add(image_id)
                images.append(image)
                if len(images) >= limit:
                    return images
        return images

    @staticmethod
    def _rag_image_display_mode(tasks: list[dict[str, Any]]) -> str:
        """只有任务规划明确包含图片检索时，才在回答中提供按需图片入口。"""
        task_types = {str(task.get("type") or "") for task in tasks}
        return "references" if "image_lookup" in task_types else "hidden"

    @staticmethod
    def _report_payload(report_path, *, report_id: str, title: str, report_mode: str = "") -> dict:
        """生成包含稳定 PDF 地址和第一页缩略图地址的持久化报告元数据。"""
        path = Path(report_path)
        payload = {
            "report_id": report_id,
            "title": title,
            "url": f"/api/case-multidim/reports/{path.name}",
            "filename": path.name,
        }
        if report_mode:
            payload["report_mode"] = report_mode
        try:
            thumbnail = ensure_report_thumbnail(path)
            payload["thumbnail_url"] = f"/api/case-multidim/report-thumbnails/{thumbnail.name}"
        except Exception as exc:
            # 缩略图属于展示增强，失败时仍应允许用户查看和下载已生成的 PDF。
            LOGGER.warning("[报告缩略图] 生成失败 report=%s error=%s", path, exc)
        return payload
