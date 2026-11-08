"""开放域问题直接调用支持联网搜索的普通聊天模型。"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from backend.app.agents.unified_chat.orchestration_models import (
    BudgetTermination,
)
from backend.app.agents.unified_chat.performance import PerformanceTrace
from backend.app.core.config import get_settings


class GeneralChatStreamMixin:
    """把开放域问题流式转发给支持联网搜索的普通聊天模型。"""

    async def _stream_general_chat(
        self,
        question: str,
        session_id: str,
        *,
        run_id: str = "",
        performance: PerformanceTrace,
    ) -> AsyncIterator[dict[str, Any]]:
        """开放域问题直接调用支持联网搜索的普通聊天模型并写回主会话。"""
        settings = get_settings()
        text = str(question or "").strip()
        run_id = run_id or f"general_chat_{uuid4().hex}"
        common = self._empty_result("general_chat", run_id, text, "")
        common["retrieval_mode"] = "agent:general_chat_web_search"
        common["web_search"] = bool(settings.web_search_enabled)
        common["llm_used"] = False
        common["llm_status"] = "skipped_no_api_key"
        common["answer_renderer"] = "fallback"
        llm_client = self.dependencies.llm_client
        fallback = (
            "抱歉，当前无法回答这个问题。您可以询问山西省内的天气历史、"
            "个例分析、统计或预警信息。"
        )
        yield self._metadata(common, status="running")
        if settings.web_search_enabled and llm_client.is_available():
            answer_parts: list[str] = []
            try:
                if hasattr(llm_client, "stream_general_chat"):
                    answer_stream = llm_client.stream_general_chat(
                        text,
                        enable_search=True,
                        forced_search=settings.web_search_forced,
                    )
                else:
                    # 兼容旧测试客户端和外部适配器。
                    try:
                        answer_stream = llm_client.stream_answer_with_context(
                            text,
                            [],
                            enable_search=True,
                            forced_search=settings.web_search_forced,
                        )
                    except TypeError:
                        answer_stream = llm_client.stream_answer_with_context(text, [])
                async for stream_kind, delta in self._stream_sync_deltas(
                    answer_stream
                ):
                    if stream_kind == "heartbeat":
                        yield self._progress_metadata(
                            "general_chat",
                            run_id,
                            "streaming",
                            60,
                            "正在等待联网搜索返回结果",
                            performance,
                        )
                        continue
                    for visible_delta in self._split_stream_delta(delta):
                        answer_parts.append(visible_delta)
                        yield {"type": "delta", "text": visible_delta}
                        await asyncio.sleep(0)
                if not answer_parts:
                    raise RuntimeError("联网搜索模型返回空答案")
                common["answer"] = "".join(answer_parts)
                common["llm_used"] = True
                common["llm_status"] = "called_with_web_search"
                common["answer_renderer"] = "llm"
            except BudgetTermination:
                # 取消、超时和预算异常必须穿透回答兜底。
                raise
            except Exception as exc:
                # 联网搜索失败时回退中文兜底，不阻塞主链路。
                common["llm_status"] = f"web_search_failed: {exc}"
                common["answer"] = fallback
                yield {"type": "delta", "text": fallback}
        else:
            common["answer"] = fallback
            yield {"type": "delta", "text": fallback}
        if session_id:
            self._persist_complete(
                session_id,
                common,
                status="completed",
                performance=performance,
            )
        yield self._completion_metadata(common, status="completed")
        yield self._done(common, status="completed")
        self._schedule_memory_refresh(session_id)
