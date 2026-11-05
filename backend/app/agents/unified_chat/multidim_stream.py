"""统一会话中的多维检索提案、确认和 PDF 报告流。"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from backend.app.agents.case_multidim_search.core import (
    progress as multidim_progress,
)
from backend.app.agents.case_multidim_search.core.natural_query import (
    NaturalCaseQueryParser,
)
from backend.app.agents.case_multidim_search.integrations.non_thinking_llm import (
    ensure_non_thinking_client,
)
from backend.app.agents.unified_chat.performance import PerformanceTrace
from backend.app.agents.unified_chat.result_adapter import (
    adapt_multidim_result,
    confirmation_answer,
)


class MultidimStreamMixin:
    """在主会话中复用多维 Agent 的条件确认与分析流程。"""

    async def _stream_multidim(
        self,
        question: str,
        session_id: str,
        pending: dict | None,
        performance: PerformanceTrace,
    ) -> AsyncIterator[dict[str, Any]]:
        """完成条件提案、确认和既有多维分析流程。"""
        if not session_id:
            yield {
                "type": "error",
                "message": "多维检索必须先创建主页面会话",
            }
            return
        if pending and self._is_cancellation(question):
            self.multidim_store.reset(session_id)
            answer = (
                "已取消待确认的多维检索条件，"
                "请重新描述需要检索的时间、地区或灾种。"
            )
            run_id = f"case_multidim_cancel_{uuid4().hex}"
            common = self._empty_result(
                "case_multidim_search",
                run_id,
                question,
                answer,
            )
            self._persist_complete(
                session_id,
                common,
                status="cancelled",
                performance=performance,
            )
            yield self._metadata(common, status="cancelled")
            async for event in self._stream_text_chunks(answer):
                yield event
            yield self._done(common, status="cancelled")
            self._schedule_memory_refresh(session_id)
            return

        if pending and self._is_confirmation(question):
            async for event in self._execute_multidim(
                session_id,
                pending,
                performance,
            ):
                yield event
            return

        parser = NaturalCaseQueryParser(
            ensure_non_thinking_client(self.dependencies.llm_client),
        )
        performance.start("multidim_parse_ms")
        parse_task = asyncio.create_task(
            asyncio.to_thread(
                self.multidim_store.propose,
                session_id,
                question,
                parser,
            )
        )
        while True:
            try:
                proposal = await asyncio.wait_for(
                    asyncio.shield(parse_task),
                    timeout=0.5,
                )
                break
            except asyncio.TimeoutError:
                yield self._progress_metadata(
                    "case_multidim_search",
                    "",
                    "condition_parse",
                    12,
                    "正在解析报告检索条件",
                    performance,
                )
        performance.stop("multidim_parse_ms")
        run_id = (
            proposal.proposal_id
            or f"case_multidim_proposal_{uuid4().hex}"
        )
        answer = confirmation_answer(proposal)
        common = self._empty_result(
            "case_multidim_search",
            run_id,
            question,
            answer,
        )
        common.update(
            {
                "retrieval_mode": (
                    "agent:case_multidim_confirmation"
                ),
                "query_conditions": (
                    proposal.parsed_request.model_dump()
                ),
                "proposal_id": proposal.proposal_id,
                "proposal_operation": proposal.operation,
                "proposal_question": question,
                "warnings": list(proposal.warnings or []),
            }
        )
        status = (
            "awaiting_confirmation"
            if proposal.can_confirm
            else "needs_clarification"
        )
        self._persist_complete(
            session_id,
            common,
            status=status,
            performance=performance,
        )
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
        """确认条件后调用多维 Agent，并按需生成 PDF 报告。"""
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
            "progress": {
                "stage": "search",
                "percent": 5,
                "message": "正在执行多维检索与统计分析",
            },
        }
        multidim_progress.start(run_id)
        try:
            performance.start("multidim_analysis_ms")
            analysis_task = asyncio.create_task(
                asyncio.to_thread(
                    self.multidim_agent.analyze_structured,
                    request,
                )
            )
            last_progress = None
            while not analysis_task.done():
                progress = multidim_progress.get(run_id)
                signature = self._progress_signature(progress)
                if progress and signature != last_progress:
                    last_progress = signature
                    yield self._progress_metadata(
                        "case_multidim_search",
                        run_id,
                        str(progress.get("stage") or "analysis"),
                        int(progress.get("percent") or 0),
                        str(
                            progress.get("message")
                            or "正在分析个例"
                        ),
                        performance,
                        extra=progress,
                    )
                await asyncio.sleep(0.5)
            result = await analysis_task
            performance.stop("multidim_analysis_ms")
            # 报告链路确认后始终生成可预览 PDF。
            multidim_progress.update(
                run_id,
                stage="pdf_export",
                message="正在准备 PDF 报告",
                percent=5,
            )

            def update_pdf_progress(
                percent: int,
                message: str,
            ) -> None:
                """把 PDF 构建线程的阶段进度写入统一进度存储。"""
                multidim_progress.update(
                    run_id,
                    stage="pdf_export",
                    message=message,
                    percent=percent,
                )

            performance.start("pdf_total_ms")
            pdf_task = asyncio.create_task(
                asyncio.to_thread(
                    self.multidim_agent.export_pdf_from_answer,
                    result.answer_id,
                    filename=f"{run_id}.pdf",
                    report_title=result.title,
                    progress_callback=update_pdf_progress,
                )
            )
            last_progress = None
            while not pdf_task.done():
                progress = multidim_progress.get(run_id)
                signature = self._progress_signature(progress)
                if progress and signature != last_progress:
                    last_progress = signature
                    yield self._progress_metadata(
                        "case_multidim_search",
                        run_id,
                        "pdf_export",
                        int(progress.get("percent") or 0),
                        str(
                            progress.get("message")
                            or "正在生成 PDF 报告"
                        ),
                        performance,
                        extra=progress,
                    )
                await asyncio.sleep(0.5)
            report_path = await pdf_task
            performance.stop("pdf_total_ms")
            reports = [
                self._report_payload(
                    report_path,
                    report_id=result.answer_id,
                    title=result.title,
                )
            ]
            common = adapt_multidim_result(
                result,
                self.document_store,
                reports,
            )
            common["run_id"] = run_id
            common["question"] = original_message
            common["answer_id"] = result.answer_id
            multidim_progress.complete(run_id)
            # 报告和结构化结果先展示，文本按片段刷新，最后持久化。
            yield self._metadata(common, status="running")
            async for event in self._stream_text_chunks(
                common["answer"]
            ):
                yield event
            yield self._progress_metadata(
                "case_multidim_search",
                run_id,
                "persist",
                98,
                "正在保存报告和会话记录",
                performance,
            )
            self._persist_complete(
                session_id,
                common,
                status="completed",
                already_started=True,
                performance=performance,
            )
            yield self._completion_metadata(
                common,
                status="completed",
            )
            yield self._done(common, status="completed")
            self._schedule_memory_refresh(session_id)
        except Exception as exc:
            multidim_progress.fail(run_id, f"检索失败：{exc}")
            self.session_store.finish_agent_run(
                run_id,
                status="failed",
                metadata={"error": str(exc)},
            )
            raise
