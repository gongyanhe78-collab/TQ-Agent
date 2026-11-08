"""承接式报告与意图澄清的统一消息流。"""
from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from backend.app.agents.unified_chat.orchestration_models import (
    BudgetTermination,
)
from backend.app.agents.unified_chat.performance import PerformanceTrace


class ReportStreamMixin:
    """把上一轮分析结果复用为简易报告，并处理澄清消息。"""

    async def _stream_clarification(
        self,
        question: str,
        session_id: str,
        answer: str,
        performance: PerformanceTrace,
    ) -> AsyncIterator[dict[str, Any]]:
        """把意图冲突追问作为普通聊天消息返回并写入主会话。"""
        text = answer or (
            "请确认本轮要生成历史个例报告，"
            "还是分析当前过程并匹配相似历史个例。"
        )
        common = self._empty_result(
            "query_intent",
            f"query_intent_{uuid4().hex}",
            question,
            text,
        )
        common["retrieval_mode"] = "agent:query_intent_clarification"
        if session_id:
            self._persist_complete(
                session_id,
                common,
                status="needs_clarification",
                performance=performance,
            )
        yield self._metadata(common, status="needs_clarification")
        async for event in self._stream_text_chunks(text):
            yield event
        yield self._done(common, status="needs_clarification")
        self._schedule_memory_refresh(session_id)

    async def _stream_previous_result_report(
        self,
        question: str,
        session_id: str,
        previous_result: dict[str, Any],
        performance: PerformanceTrace,
    ) -> AsyncIterator[dict[str, Any]]:
        """只把上一轮结果排版为 PDF，不重新检索或调用分析模型。"""
        run_id = f"conversation_report_{uuid4().hex}"
        report_title = self._conversation_report_title(
            previous_result
        )
        source_answer = str(
            previous_result.get("answer") or ""
        ).strip()
        common = {
            "agent_type": "conversation_report",
            "run_id": run_id,
            "question": question,
            "answer": (
                "已复用上一轮个例分析结果，"
                "正在整理为简易 PDF 报告。"
            ),
            "retrieval_mode": "conversation_reuse",
            "hit_count": len(
                previous_result.get("evidence_cases") or []
            ),
            "hits": [],
            "evidence_cases": list(
                previous_result.get("evidence_cases") or []
            ),
            "evidence_chunks": list(
                previous_result.get("evidence_chunks") or []
            ),
            # 报告后处理只返回 PDF，不在当前消息再次铺开图片。
            "images": [],
            "visuals": [],
            "reports": [],
            "analysis": dict(
                previous_result.get("analysis") or {}
            ),
            "task_plan": list(
                previous_result.get("task_plan") or []
            ),
            "agent_result": dict(
                previous_result.get("agent_result") or {}
            ),
            "report_mode": "from_previous_result",
            "source_message_id": previous_result.get(
                "source_message_id"
            ),
            "source_run_id": previous_result.get("run_id", ""),
            "source_content_sha256": hashlib.sha256(
                source_answer.encode("utf-8")
            ).hexdigest(),
            "report_title": report_title,
            "llm_used": False,
            "llm_status": "reused_previous_result",
        }
        if not session_id:
            yield {
                "type": "error",
                "message": "简易报告必须先创建主页面会话",
            }
            return
        self._start_persisted_run(
            run_id,
            session_id,
            "conversation_report",
            question,
        )
        yield self._metadata(common, status="running")
        # 先发送可见反馈，再在线程中执行本地 PDF 排版。
        async for event in self._stream_text_chunks(
            common["answer"],
            chunk_size=36,
        ):
            yield event
        try:
            report_path = await asyncio.to_thread(
                self.multidim_agent.export_pdf_from_previous_result,
                previous_result,
                filename=f"{run_id}.pdf",
                report_title=report_title,
            )
            common["reports"] = [
                self._report_payload(
                    report_path,
                    report_id=run_id,
                    title=report_title,
                    report_mode="from_previous_result",
                )
            ]
            common["answer"] = (
                "已根据上一轮分析结果生成简易 PDF 汇总报告。"
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
        except BudgetTermination as exc:
            self._finish_active_run(
                status="failed",
                error=str(exc),
            )
            raise
        except Exception as exc:
            self._finish_active_run(
                status="failed",
                error=str(exc),
            )
            yield {
                "type": "error",
                "message": f"简易 PDF 生成失败：{exc}",
                "run_id": run_id,
                **self._runtime_fields(),
            }
