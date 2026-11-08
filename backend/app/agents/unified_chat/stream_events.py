"""统一 SSE 事件构造与文本流拆分工具。"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from backend.app.agents.case_multidim_search.reporting.report_preview import (
    ensure_report_thumbnail,
)
from backend.app.agents.unified_chat.orchestration_models import (
    BudgetTermination,
)
from backend.app.agents.unified_chat.performance import PerformanceTrace

LOGGER = logging.getLogger("uvicorn.error")


class StreamEventMixin:
    """为统一编排器提供不依赖具体业务 Agent 的事件与流工具。"""

    def _runtime_snapshot(self) -> dict[str, Any]:
        """读取当前请求运行时快照，缺失运行时保持旧事件兼容。"""
        runtime = getattr(self, "runtime", None)
        if runtime is None:
            return {}
        return runtime.snapshot()

    def _runtime_fields(self) -> dict[str, Any]:
        """只向现有 SSE 事件追加四个稳定运行时字段。"""
        snapshot = self._runtime_snapshot()
        return {
            "request_id": snapshot.get("request_id", ""),
            "trace_id": snapshot.get("trace_id", ""),
            "termination_reason": snapshot.get(
                "termination_reason",
                "",
            ),
            "subtask_status": list(
                snapshot.get("subtask_status") or []
            ),
        }

    def _runtime_metadata(self) -> dict[str, Any]:
        """提取需要写入 Agent 运行表的完整预算和 Token 快照。"""
        snapshot = self._runtime_snapshot()
        if not snapshot:
            return {}
        return {
            "request_id": snapshot.get("request_id", ""),
            "trace_id": snapshot.get("trace_id", ""),
            "termination_reason": snapshot.get(
                "termination_reason",
                "",
            ),
            "steps": int(snapshot.get("steps") or 0),
            "model_calls": int(snapshot.get("model_calls") or 0),
            "subtask_status": list(
                snapshot.get("subtask_status") or []
            ),
            "budget": dict(snapshot.get("budget") or {}),
            "token_stats": dict(
                snapshot.get("token_stats") or {}
            ),
            "token_usage_source": snapshot.get(
                "token_usage_source",
                "",
            ),
        }

    def _merge_runtime_metadata(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """保留业务元数据，并让当前运行时快照覆盖同名字段。"""
        merged = dict(metadata or {})
        merged.update(self._runtime_metadata())
        return merged

    def _runtime_failure_metadata(
        self,
        error: str,
    ) -> dict[str, Any]:
        """把普通业务异常收口为内部错误并合并运行时快照。"""
        runtime = getattr(self, "runtime", None)
        if runtime is not None:
            runtime.mark_terminated("internal_error")
        return self._merge_runtime_metadata({"error": str(error)})

    def _finalize_runtime_status(self, status: str) -> None:
        """根据最终事件状态收口运行生命周期。"""
        runtime = getattr(self, "runtime", None)
        if runtime is None:
            return
        normalized = str(status or "").strip()
        if normalized in {
            "completed",
            "awaiting_confirmation",
            "needs_clarification",
        }:
            runtime.mark_completed()
        elif normalized == "cancelled":
            runtime.mark_terminated("cancelled_by_client")
        elif normalized == "failed":
            runtime.mark_terminated("internal_error")

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

    def _progress_metadata(
        self,
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
        progress = {
            **dict(extra or {}),
            "stage": stage,
            "percent": percent,
            "message": message,
        }
        return {
            "type": "metadata",
            "agent_type": agent_type,
            "run_id": run_id,
            "status": "running",
            "progress": progress,
            "performance": performance.snapshot(),
            **self._runtime_fields(),
        }

    def _metadata(self, common: dict, *, status: str) -> dict:
        """生成统一 metadata 事件。"""
        self._finalize_runtime_status(status)
        return {
            "type": "metadata",
            "status": status,
            **{key: value for key, value in common.items() if key != "answer"},
            **self._runtime_fields(),
        }

    def _done(self, common: dict, *, status: str) -> dict:
        """生成统一 done 事件，包含运行号和报告地址。"""
        self._finalize_runtime_status(status)
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
            "regenerated_from_message_id": common.get(
                "regenerated_from_message_id"
            ),
            "regeneration_instruction": common.get(
                "regeneration_instruction", ""
            ),
            "source_user_message_id": common.get("source_user_message_id"),
            "version_group_id": common.get("version_group_id", ""),
            "version": int(common.get("version") or 1),
            "performance": dict(common.get("performance") or {}),
            **self._runtime_fields(),
        }

    def _completion_metadata(self, common: dict, *, status: str) -> dict:
        """正文流结束后发送轻量后处理结果，不重复传输大块证据。"""
        self._finalize_runtime_status(status)
        return {
            "type": "metadata",
            "status": status,
            "agent_type": common.get("agent_type", ""),
            "run_id": common.get("run_id", ""),
            "message_id": common.get("message_id"),
            "reports": list(common.get("reports") or []),
            "follow_up_suggestions": list(common.get("follow_up_suggestions") or []),
            "follow_up_status": common.get("follow_up_status", "none"),
            "regenerated_from_message_id": common.get(
                "regenerated_from_message_id"
            ),
            "regeneration_instruction": common.get(
                "regeneration_instruction", ""
            ),
            "source_user_message_id": common.get("source_user_message_id"),
            "version_group_id": common.get("version_group_id", ""),
            "version": int(common.get("version") or 1),
            **self._runtime_fields(),
        }

    async def _stream_sync_deltas(
        self,
        iterator,
    ) -> AsyncIterator[tuple[str, str]]:
        """在线程中消费同步生成器，并在模型等待期间持续发出心跳。"""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        stop_event = threading.Event()
        runtime = getattr(self, "runtime", None)
        if runtime is not None:
            # 启动模型流之前先确认取消、截止时间和调用预算。
            runtime.check()
            runtime.count_model_call()

        def produce() -> None:
            """阻塞读取云端流，并通过线程安全队列通知事件循环。"""
            try:
                for value in iterator:
                    if stop_event.is_set():
                        break
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        ("delta", value),
                    )
            except BudgetTermination as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    ("error", exc),
                )
            except Exception as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    ("error", exc),
                )
            finally:
                try:
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        ("done", None),
                    )
                except RuntimeError:
                    # 请求已终止时事件循环可能先关闭，线程无需再上报完成。
                    pass

        producer = asyncio.create_task(asyncio.to_thread(produce))
        try:
            while True:
                if runtime is not None:
                    runtime.check()
                try:
                    kind, value = await asyncio.wait_for(
                        queue.get(),
                        timeout=0.75,
                    )
                except TimeoutError:
                    if runtime is not None:
                        # 长模型请求不能依赖首字返回才执行取消和超时检查。
                        runtime.check()
                    yield "heartbeat", ""
                    continue
                if kind == "done":
                    break
                if kind == "error":
                    raise value
                if value:
                    if runtime is not None:
                        runtime.count_tokens(str(value))
                    yield "delta", str(value)
        finally:
            stop_event.set()
            if not producer.done():
                producer.cancel()
            await asyncio.gather(
                producer,
                return_exceptions=True,
            )

    async def _stream_text_chunks(
        self,
        text: str,
        chunk_size: int = 48,
    ) -> AsyncIterator[dict[str, str]]:
        """把已生成答案按可读片段发送，保持拼接后的正文逐字完全一致。"""
        value = str(text or "")
        start = 0
        punctuation = "。！？!?；;\n"
        while start < len(value):
            runtime = getattr(self, "runtime", None)
            if runtime is not None:
                runtime.check()
            end = min(len(value), start + chunk_size)
            if end < len(value):
                # 在片段后半段优先寻找自然边界，避免页面刷新时停在半句话中。
                boundary = max(
                    (
                        value.rfind(mark, start + chunk_size // 2, end + 1)
                        for mark in punctuation
                    ),
                    default=-1,
                )
                if boundary >= start:
                    end = boundary + 1
            yield {"type": "delta", "text": value[start:end]}
            start = end
            await asyncio.sleep(0)

    @staticmethod
    def _split_stream_delta(text: str, chunk_size: int = 32) -> list[str]:
        """将云端单次返回的大块文本拆成稳定的小增量，不改变内容顺序。"""
        value = str(text or "")
        return [
            value[index:index + chunk_size]
            for index in range(0, len(value), chunk_size)
        ] or [""]

    @staticmethod
    def _empty_result(
        agent_type: str,
        run_id: str,
        question: str,
        answer: str,
    ) -> dict:
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

    @staticmethod
    def _report_payload(
        report_path,
        *,
        report_id: str,
        title: str,
        report_mode: str = "",
    ) -> dict:
        """生成包含稳定 PDF 地址和第一页缩略图地址的报告元数据。"""
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
            payload["thumbnail_url"] = (
                f"/api/case-multidim/report-thumbnails/{thumbnail.name}"
            )
        except Exception as exc:
            # 缩略图属于展示增强，失败时仍允许查看和下载已生成的 PDF。
            LOGGER.warning("[报告缩略图] 生成失败 report=%s error=%s", path, exc)
        return payload

    @staticmethod
    def _rag_image_display_mode(tasks: list[dict[str, Any]]) -> str:
        """只有任务规划包含图片检索时，才提供按需图片入口。"""
        task_types = {str(task.get("type") or "") for task in tasks}
        return "references" if "image_lookup" in task_types else "hidden"
