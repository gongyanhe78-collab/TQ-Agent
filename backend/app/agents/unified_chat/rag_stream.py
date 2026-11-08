"""统一会话中的结构化查询、文档检索和回答模型流。"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from time import perf_counter
from typing import Any
from uuid import uuid4

from backend.app.agents.intent_analyzer.analyzers import (
    AgentIntentAnalyzer,
)
from backend.app.agents.planner.planner import AgentPlanner
from backend.app.agents.unified_chat.orchestration_models import (
    BudgetTermination,
)
from backend.app.agents.unified_chat.performance import PerformanceTrace
from backend.app.core.config import get_settings
from backend.app.services.feedback_guidance import (
    render_session_guidance,
)

LOGGER = logging.getLogger("uvicorn.error")


class RagStreamMixin:
    """保留主项目文档向量检索和云端流式回答链路。"""

    def _should_upgrade_to_web_search(self) -> bool:
        """判断本地检索为空时是否允许升级为联网搜索。"""
        settings = get_settings()
        if not (
            settings.web_search_enabled
            and settings.rag_empty_upgrade_web_search
        ):
            return False
        llm_client = self.dependencies.llm_client
        try:
            return bool(llm_client.is_available())
        except Exception:
            return False

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
        """执行结构化或文档向量 RAG，并输出统一 SSE。"""
        run_id = f"document_rag_{uuid4().hex}"
        original_question = str(payload.question or "")
        question = str(resolved_question or original_question)
        generation_question = question
        session_guidance_prompt = render_session_guidance(
            session_guidances
        )
        regeneration_instruction = str(
            self._regeneration_context.get(
                "regeneration_instruction"
            )
            or ""
        ).strip()
        if regeneration_instruction:
            generation_question = (
                f"{question}\n\n重新生成要求："
                f"{regeneration_instruction}。"
                "请仍严格依据本轮检索证据回答，不得编造材料外事实。"
            )
        # 跨会话长期记忆只作为背景上下文，不改变检索范围与证据边界。
        long_term_memory = (
            getattr(self, "_long_term_memory_context", "") or ""
        ).strip()
        if long_term_memory:
            generation_question = (
                "用户长期记忆（仅作背景，不得据此编造新事实）：\n"
                f"{long_term_memory}\n\n"
                f"当前问题：\n{generation_question}"
            )
        context_case_ids = list(referenced_case_ids or [])
        rag_tasks = [
            task
            for task in (task_plan or [])
            if str(task.get("route") or "") == "rag"
        ]
        rag_tasks = self._apply_same_scope_feedback_to_tasks(
            rag_tasks,
            session_guidances,
        )
        # 任务描述可增加反馈维度，底层检索仍使用本轮原始问题。
        for task in rag_tasks:
            if task.get("feedback_guidance_ids"):
                task["retrieval_question"] = question
        feedback_task_applied = any(
            task.get("feedback_guidance_ids") for task in rag_tasks
        )
        uses_task_plan = bool(rag_tasks) and (
            len(rag_tasks) > 1
            or any(
                str(task.get("type") or "") == "case_listing"
                for task in rag_tasks
            )
            or feedback_task_applied
        )
        if not context_case_ids:
            context_case_ids = (
                self.dependencies.context_cases.from_messages(
                    history,
                    original_question,
                )
            )
        if session_id:
            self._start_persisted_run(
                run_id,
                session_id,
                "document_rag",
                original_question,
            )
        if rag_strategy == "structured" and not uses_task_plan:
            yield self._progress_metadata(
                "document_rag",
                run_id,
                "structured_search",
                12,
                "正在查询标准化个例",
                performance,
            )
            structured = await asyncio.to_thread(
                self.dependencies.structured_query.answer,
                question,
                context_case_ids=context_case_ids,
            )
            if (
                structured is not None
                and self.dependencies.structured_query.needs_metric_documents(
                    structured
                )
            ):
                structured = (
                    self.dependencies.structured_query.metric_document_result(
                        question,
                        top_k=payload.top_k,
                        top_n=payload.top_n,
                        context_case_ids=context_case_ids,
                    )
                    or structured
                )
            if structured is not None:
                common = self._structured_common(
                    structured,
                    run_id,
                    original_question,
                )
                guidance_audit = {
                    "applied": False,
                    "count": len(session_guidances or []),
                    "renderer": "structured_fallback",
                    "reason": "structured_result_fallback",
                }
                common.setdefault("audit", {})[
                    "session_feedback_guidance"
                ] = guidance_audit
                self._attach_memory_trace(
                    common,
                    original_question,
                    question,
                    context_case_ids,
                    route_trace,
                )
                yield self._metadata(common, status="running")
                async for event in self._stream_structured_answer(
                    common=common,
                    structured=structured,
                    run_id=run_id,
                    session_id=session_id,
                    generation_question=generation_question,
                    session_guidance_prompt=session_guidance_prompt,
                    guidance_audit=guidance_audit,
                    performance=performance,
                ):
                    yield event
                return

        yield self._progress_metadata(
            "document_rag",
            run_id,
            "retrieval",
            12,
            "正在检索本地知识库",
            performance,
        )
        performance.start("retrieval_ms")
        task_context_blocks: list[str] = []
        task_results: list[dict[str, Any]] = []
        planned_evidence_cases: list[dict[str, Any]] = []
        if uses_task_plan:
            (
                retrieval_mode,
                hits,
                task_context_blocks,
                task_results,
                planned_evidence_cases,
            ) = self._retrieve_rag_task_plan(
                rag_tasks,
                context_case_ids,
                top_k=payload.top_k,
                top_n=payload.top_n,
            )
        else:
            # 单任务继续沿用原检索路径。
            retrieval_mode, hits = (
                self.dependencies.document_qa.retrieve(
                    question,
                    top_k=payload.top_k,
                    top_n=payload.top_n,
                    context_case_ids=context_case_ids,
                )
            )
        performance.stop("retrieval_ms")
        yield self._progress_metadata(
            "document_rag",
            run_id,
            "retrieved",
            35,
            f"已找到 {len(hits)} 个相关证据片段",
            performance,
        )
        if (
            not hits
            and not task_context_blocks
            and self._should_upgrade_to_web_search()
        ):
            # 本地知识库没有证据，升级为联网搜索回答。
            if session_id:
                try:
                    self._finish_persisted_run(
                        run_id,
                        status="completed",
                        answer="",
                        metadata=self._runtime_metadata(),
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "[RAG升级] 检索运行记录收口失败 run_id=%s error=%s",
                        run_id,
                        exc,
                    )
            yield self._progress_metadata(
                "general_chat",
                run_id,
                "web_search",
                40,
                "正在联网搜索最新信息",
                performance,
            )
            async for event in self._stream_general_chat(
                question,
                session_id,
                run_id=f"general_chat_{uuid4().hex}",
                performance=performance,
            ):
                yield event
            return
        analysis = AgentIntentAnalyzer().analyze(question)
        plan = AgentPlanner().plan(question, analysis)
        hit_payloads = [
            self.dependencies.presenter.document_hit_to_response(hit)
            for hit in hits
        ]
        images = self._images_from_hits(hit_payloads)
        # 图片作为可追溯证据持久化，是否展示由任务计划单独声明。
        image_display_mode = self._rag_image_display_mode(rag_tasks)
        evidence_cases = self._context_evidence_cases(
            context_case_ids
        )
        known_case_ids = {
            str(item.get("case_id") or "")
            for item in evidence_cases
        }
        for case in planned_evidence_cases:
            case_id = str(case.get("case_id") or "")
            if case_id and case_id not in known_case_ids:
                evidence_cases.append(case)
                known_case_ids.add(case_id)
        if not evidence_cases:
            evidence_cases = (
                self._evidence_cases_from_document_hits(hit_payloads)
            )
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
            "audit": {
                "agent_runs": [
                    {
                        "agent": "document_retriever",
                        "status": "ok",
                    }
                ]
            },
        }
        common["audit"]["session_feedback_guidance"] = {
            "applied": bool(session_guidance_prompt),
            "count": len(session_guidances or []),
        }
        if uses_task_plan:
            # 任务只保存状态和证据 ID，完整正文仍由统一字段持久化。
            common["task_plan"] = task_results
            common["audit"]["agent_runs"].extend(
                {
                    "agent": "task_evidence_retriever",
                    "task_id": item.get("id"),
                    "task_type": item.get("type"),
                    "status": item.get("status", "ok"),
                    "evidence_chunk_ids": item.get(
                        "evidence_chunk_ids",
                        [],
                    ),
                }
                for item in task_results
            )
        self._attach_memory_trace(
            common,
            original_question,
            question,
            context_case_ids,
            route_trace,
        )
        yield self._metadata(common, status="running")
        async for event in self._stream_retrieved_answer(
            common=common,
            question=question,
            generation_question=generation_question,
            hits=hits,
            task_context_blocks=task_context_blocks,
            session_guidance_prompt=session_guidance_prompt,
            run_id=run_id,
            performance=performance,
        ):
            yield event
        yield self._progress_metadata(
            "document_rag",
            run_id,
            "persist",
            98,
            "正在保存回答和证据",
            performance,
        )
        if session_id:
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

    async def _stream_structured_answer(
        self,
        *,
        common: dict,
        structured: dict,
        run_id: str,
        session_id: str,
        generation_question: str,
        session_guidance_prompt: str,
        guidance_audit: dict,
        performance: PerformanceTrace,
    ) -> AsyncIterator[dict[str, Any]]:
        """使用核验事实调用回答模型，失败时回退结构化正文。"""
        structured_answer = str(common.get("answer") or "")
        answer_parts: list[str] = []
        llm_used = False
        llm_status = "structured_fallback"
        if self.dependencies.llm_client.is_available():
            render_started = perf_counter()
            performance.start("llm_total_ms")
            first_delta = True
            yield self._progress_metadata(
                "document_rag",
                run_id,
                "generation",
                45,
                "正在整理结构化事实并生成回答",
                performance,
            )
            try:
                render_context = self._structured_render_context(
                    structured,
                    common,
                )
                try:
                    answer_stream = (
                        self.dependencies.llm_client.stream_answer_with_context(
                            generation_question,
                            render_context,
                            session_guidance_prompt=session_guidance_prompt,
                        )
                    )
                except TypeError:
                    # 兼容旧测试客户端和外部适配器。
                    answer_stream = (
                        self.dependencies.llm_client.stream_answer_with_context(
                            generation_question,
                            render_context,
                        )
                    )
                async for stream_kind, delta in self._stream_sync_deltas(
                    answer_stream
                ):
                    if stream_kind == "heartbeat":
                        yield self._progress_metadata(
                            "document_rag",
                            run_id,
                            "generation",
                            50,
                            "正在等待模型返回首段回答",
                            performance,
                        )
                        continue
                    if first_delta:
                        performance.set(
                            "llm_ttft_ms",
                            (perf_counter() - render_started) * 1000,
                        )
                        first_delta = False
                        yield self._progress_metadata(
                            "document_rag",
                            run_id,
                            "streaming",
                            60,
                            "模型已开始整理回答",
                            performance,
                        )
                    for visible_delta in self._split_stream_delta(delta):
                        answer_parts.append(visible_delta)
                        yield {
                            "type": "delta",
                            "text": visible_delta,
                        }
                        await asyncio.sleep(0)
                performance.stop("llm_total_ms")
                if not answer_parts:
                    raise RuntimeError("结构化回答渲染模型返回空答案")
                llm_used = True
                llm_status = "called_with_structured_facts"
                guidance_audit.update(
                    {
                        "applied": bool(session_guidance_prompt),
                        "renderer": "llm",
                        "reason": (
                            "structured_facts_rendered_by_answer_model"
                        ),
                    }
                )
            except BudgetTermination:
                # 运行时终止由统一编排器转成结构化 SSE，不能被模型兜底吞掉。
                raise
            except Exception as exc:
                performance.stop("llm_total_ms")
                # 首字前失败回退核验答案；已有增量时保留已显示内容。
                llm_status = f"structured_render_failed: {exc}"
                guidance_audit.update(
                    {
                        "applied": False,
                        "renderer": "structured_fallback",
                        "reason": "structured_render_failed",
                    }
                )
                if not answer_parts:
                    answer_parts.append(structured_answer)
                    yield {
                        "type": "delta",
                        "text": structured_answer,
                    }
        else:
            answer_parts.append(structured_answer)
            yield {"type": "delta", "text": structured_answer}
            guidance_audit.update(
                {
                    "applied": False,
                    "renderer": "structured_fallback",
                    "reason": "answer_model_unavailable",
                }
            )
        common["answer"] = "".join(answer_parts)
        common["llm_used"] = llm_used
        common["llm_status"] = llm_status
        common["answer_renderer"] = (
            "llm" if llm_used else "structured_fallback"
        )
        yield self._progress_metadata(
            "document_rag",
            run_id,
            "persist",
            98,
            "正在保存回答和证据",
            performance,
        )
        if session_id:
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

    async def _stream_retrieved_answer(
        self,
        *,
        common: dict,
        question: str,
        generation_question: str,
        hits: list[Any],
        task_context_blocks: list[str],
        session_guidance_prompt: str,
        run_id: str,
        performance: PerformanceTrace,
    ) -> AsyncIterator[dict[str, Any]]:
        """基于文档证据流式生成回答，失败时沿用原兜底答案。"""
        answer_parts: list[str] = []
        llm_used = False
        llm_status = "skipped_no_hits"
        has_evidence = bool(hits or task_context_blocks)
        if not has_evidence:
            answer_parts.append("未检索到相关证据片段。")
            yield {"type": "delta", "text": answer_parts[-1]}
        elif self.dependencies.llm_client.is_available():
            context_blocks = (
                task_context_blocks
                or self.dependencies.document_qa.context_blocks(hits)
            )
            try:
                llm_started = perf_counter()
                performance.start("llm_total_ms")
                first_delta = True
                yield self._progress_metadata(
                    "document_rag",
                    run_id,
                    "generation",
                    45,
                    "正在基于证据生成回答",
                    performance,
                )
                try:
                    answer_stream = (
                        self.dependencies.llm_client.stream_answer_with_context(
                            generation_question,
                            context_blocks,
                            session_guidance_prompt=session_guidance_prompt,
                        )
                    )
                except TypeError:
                    # 兼容旧测试客户端和外部适配器。
                    answer_stream = (
                        self.dependencies.llm_client.stream_answer_with_context(
                            generation_question,
                            context_blocks,
                        )
                    )
                async for stream_kind, delta in self._stream_sync_deltas(
                    answer_stream
                ):
                    if stream_kind == "heartbeat":
                        yield self._progress_metadata(
                            "document_rag",
                            run_id,
                            "generation",
                            50,
                            "正在等待模型返回首段回答",
                            performance,
                        )
                        continue
                    if first_delta:
                        performance.set(
                            "llm_ttft_ms",
                            (perf_counter() - llm_started) * 1000,
                        )
                        first_delta = False
                        yield self._progress_metadata(
                            "document_rag",
                            run_id,
                            "streaming",
                            60,
                            "模型已开始生成回答",
                            performance,
                        )
                    for visible_delta in self._split_stream_delta(delta):
                        answer_parts.append(visible_delta)
                        yield {
                            "type": "delta",
                            "text": visible_delta,
                        }
                        await asyncio.sleep(0)
                performance.stop("llm_total_ms")
                if not answer_parts:
                    raise RuntimeError("云端模型返回空答案")
                llm_used = True
                llm_status = "called_with_local_evidence"
            except BudgetTermination:
                # 取消、超时和预算异常必须穿透回答兜底。
                raise
            except Exception as exc:
                # 首字前失败使用原兜底；已有部分内容时不再重复拼接。
                fallback = (
                    ""
                    if answer_parts
                    else (
                        self.dependencies.document_qa.fallback_answer(
                            question,
                            hits,
                        )
                        if hits
                        else self._task_context_fallback(
                            task_context_blocks
                        )
                    )
                )
                if fallback:
                    answer_parts.append(fallback)
                llm_status = f"failed: {exc}"
                if fallback:
                    yield {"type": "delta", "text": fallback}
        else:
            fallback = (
                self.dependencies.document_qa.fallback_answer(
                    question,
                    hits,
                )
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
        common["audit"]["session_feedback_guidance"].update(
            {
                "renderer": "llm" if llm_used else "fallback",
                "applied": bool(
                    session_guidance_prompt and llm_used
                ),
            }
        )
