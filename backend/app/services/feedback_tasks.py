"""会话反馈的异步归一化任务管理。"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from backend.app.services.feedback_guidance import guidance_fingerprint, normalize_with_model

LOGGER = logging.getLogger("uvicorn.error")


class FeedbackTaskService:
    """在应用生命周期内托管反馈归一化任务，避免后台任务被回收。"""

    def __init__(self, *, session_store: Any, llm_client: Any) -> None:
        self.session_store = session_store
        self.llm_client = llm_client
        self.tasks: set[asyncio.Task[None]] = set()

    def schedule(
        self,
        guidance_id: str,
        original_question: str,
        visible_answer: str,
        raw_feedback: str,
    ) -> None:
        """登记反馈归一化任务，并让请求不必等待模型调用。"""
        task = asyncio.create_task(
            self._normalize(
                guidance_id,
                original_question,
                visible_answer,
                raw_feedback,
            )
        )
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _normalize(
        self,
        guidance_id: str,
        original_question: str,
        visible_answer: str,
        raw_feedback: str,
    ) -> None:
        """后台归一化反馈，模型异常只更新状态，不影响主聊天流。"""
        try:
            result = await asyncio.to_thread(
                normalize_with_model,
                self.llm_client,
                original_question,
                visible_answer,
                raw_feedback,
            )
            stored = self.session_store.update_feedback_guidance(guidance_id, result)
            if stored and stored.get("status") == "active":
                self._deduplicate_guidance(guidance_id, stored)
        except Exception as exc:  # pragma: no cover - 仅记录部署环境异常
            LOGGER.warning(
                "[会话反馈] 异步归一化失败 guidance_id=%s error=%s",
                guidance_id,
                exc,
            )
            self._record_failure(guidance_id, exc)

    def _deduplicate_guidance(self, guidance_id: str, stored: dict[str, Any]) -> None:
        """合并内容相同的临时要求，并限制会话内有效反馈数量。"""
        session_id = str(stored.get("session_id") or "")
        current_key = guidance_fingerprint(str(stored.get("guidance_prompt") or ""))
        duplicate = next(
            (
                item
                for item in self.session_store.list_feedback_guidances(
                    session_id,
                    statuses=("active",),
                    limit=5,
                )
                if str(item.get("guidance_id") or "") != guidance_id
                and current_key
                and guidance_fingerprint(str(item.get("guidance_prompt") or "")) == current_key
            ),
            None,
        )
        if duplicate:
            self.session_store.set_feedback_guidance_status(guidance_id, "duplicate")
            return
        self.session_store.expire_old_feedback_guidances(session_id, keep=5)

    def _record_failure(self, guidance_id: str, exc: Exception) -> None:
        """尽量把失败状态写回数据库，避免前端长期停留在 pending。"""
        try:
            self.session_store.update_feedback_guidance(
                guidance_id,
                {"status": "failed", "error": str(exc)[:240]},
            )
        except Exception:
            LOGGER.exception("[会话反馈] 归一化失败状态写入失败 guidance_id=%s", guidance_id)

    def close(self) -> None:
        """取消尚未完成的后台任务；应用正常退出时通常已经没有待处理任务。"""
        for task in self.tasks:
            task.cancel()
        self.tasks.clear()
