"""统一会话中的 Smart 相似个例匹配流。"""
from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from uuid import uuid4

from backend.app.agents.smart_case_match.schemas import (
    NaturalLanguageMatchRequest,
)
from backend.app.agents.unified_chat.performance import PerformanceTrace
from backend.app.agents.unified_chat.result_adapter import (
    adapt_smart_result,
)


class SmartStreamMixin:
    """完整消费 Smart 原流式流程，仅转换事件名称和持久化位置。"""

    async def _stream_smart(
        self,
        question: str,
        session_id: str,
        is_disconnected: Callable[[], Awaitable[bool]] | None,
        performance: PerformanceTrace,
    ) -> AsyncIterator[dict[str, Any]]:
        """把 Smart 事件转换为统一 SSE 协议。"""
        run_id = ""
        started = False
        request = NaturalLanguageMatchRequest(message=question)
        performance.start("smart_total_ms")
        async for event, data in self.smart_agent.stream_natural_match(
            request,
            is_disconnected,
        ):
            if event == "accepted":
                run_id = str(
                    data.get("run_id")
                    or f"smart_case_{uuid4().hex}"
                )
                if session_id:
                    self._start_persisted_run(
                        run_id,
                        session_id,
                        "smart_case_match",
                        question,
                    )
                    started = True
                yield {
                    "type": "metadata",
                    "agent_type": "smart_case_match",
                    "run_id": run_id,
                    "status": "running",
                    "progress": data,
                    **self._runtime_fields(),
                }
                continue
            if event in {
                "stage",
                "heartbeat",
                "parsed_query",
                "matched_cases",
                "forecast",
            }:
                yield {
                    "type": "metadata",
                    "agent_type": "smart_case_match",
                    "run_id": run_id,
                    "status": "running",
                    "progress": data,
                    **self._runtime_fields(),
                }
                continue
            if event == "error":
                if session_id and started:
                    self._finish_active_run(
                        status="failed",
                        error=str(data.get("message", "")),
                    )
                yield {
                    "type": "error",
                    "message": data.get(
                        "message",
                        "相似个例匹配失败",
                    ),
                    "run_id": run_id,
                    **self._runtime_fields(),
                }
                return
            if event == "completed":
                performance.stop("smart_total_ms")
                common = adapt_smart_result(
                    data,
                    self.document_store,
                )
                common["question"] = question
                # 结构化结果先展示，文字由主页面隐藏重复副本。
                yield self._metadata(common, status="running")
                async for delta_event in self._stream_text_chunks(
                    common["answer"]
                ):
                    yield delta_event
                status = str(data.get("status") or "completed")
                yield self._progress_metadata(
                    "smart_case_match",
                    str(common.get("run_id") or run_id),
                    "persist",
                    98,
                    "正在保存匹配结果和证据",
                    performance,
                )
                if session_id:
                    self._persist_complete(
                        session_id,
                        common,
                        status=status,
                        already_started=started,
                        performance=performance,
                    )
                yield self._completion_metadata(
                    common,
                    status=status,
                )
                yield self._done(common, status=status)
                self._schedule_memory_refresh(session_id)
                return
