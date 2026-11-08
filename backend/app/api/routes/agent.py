"""统一查询兼容接口，全部通过应用服务容器执行业务。"""
from __future__ import annotations

import asyncio
import inspect
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from backend.app.agents.unified_chat.orchestration_models import (
    ExecutionContext,
)
from backend.app.api.dependencies import ChatServiceDep
from backend.app.api.schemas import AgentQueryRequest, QueryRequest, StreamQueryRequest
from backend.app.observability.context import (
    current_context,
    new_request_id,
    new_trace_id,
)
from backend.app.services.query.idempotency import (
    duplicate_event,
    get_query_idempotency,
)

router = APIRouter()


def _execution_context_parameter(stream_method) -> str:
    """返回旧流接口兼容的上下文传参方式。"""
    try:
        parameters = list(
            inspect.signature(stream_method).parameters.values()
        )
    except (TypeError, ValueError):
        return ""
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]
    if len(positional) >= 3:
        return "positional"
    if any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in parameters
    ):
        return "positional"
    if any(
        parameter.name == "execution_context"
        and parameter.kind == inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters
    ):
        return "keyword"
    return ""


@router.post("/api/query")
def query_cases(payload: QueryRequest, chat_service: ChatServiceDep):
    """为旧客户端执行非流式结构化优先文档查询。"""
    return chat_service.query_document(payload)


@router.post("/api/agent/query")
def agent_query(payload: AgentQueryRequest, chat_service: ChatServiceDep):
    """面向其他智能体项目保留原响应协议的结构化查询。"""
    return chat_service.query_agent(payload)


def _sse_headers() -> dict[str, str]:
    """返回统一 SSE 响应头，重复请求与主链路保持一致。"""
    return {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
        "Keep-Alive": "timeout=60",
    }


@router.post("/api/query/stream")
async def unified_stream_query_cases(
    payload: StreamQueryRequest,
    request: Request,
    chat_service: ChatServiceDep,
):
    """主页面统一流式入口，原样转发当前编排事件协议。"""
    request_context = current_context()
    user_id = str(request_context.user_id or "anonymous")
    cancel_event = asyncio.Event()
    execution_context = ExecutionContext(
        request_id=request_context.request_id or new_request_id(),
        trace_id=request_context.trace_id or new_trace_id(),
        user_id=request_context.user_id,
        session_id=str(payload.session_id or ""),
        cancel_event=cancel_event,
    )
    context_parameter = _execution_context_parameter(
        chat_service.stream
    )
    # 幂等键去重：同一键的重复请求直接返回首次结果，不重复写入消息与运行。
    idempotency_key = str(payload.idempotency_key or "").strip()
    registry = get_query_idempotency() if idempotency_key else None
    if registry is not None:
        duplicate = duplicate_event(registry, idempotency_key, user_id)
        if duplicate is not None:

            async def duplicate_stream():
                """只输出一条去重提示事件，不触发编排器。"""
                yield (
                    "data: "
                    + json.dumps(duplicate, ensure_ascii=False)
                    + "\n\n"
                )

            return StreamingResponse(
                duplicate_stream(),
                media_type="text/event-stream; charset=utf-8",
                headers=_sse_headers(),
            )
        registry.claim(idempotency_key, user_id, run_id="")

    async def event_stream():
        """把编排器事件编码为现有 SSE 数据帧。"""
        last_run_id = ""
        last_status = "cancelled"
        try:
            if context_parameter == "positional":
                stream = chat_service.stream(
                    payload,
                    request.is_disconnected,
                    execution_context,
                )
            elif context_parameter == "keyword":
                stream = chat_service.stream(
                    payload,
                    request.is_disconnected,
                    execution_context=execution_context,
                )
            else:
                stream = chat_service.stream(
                    payload,
                    request.is_disconnected,
                )
            async for event in stream:
                if isinstance(event, dict):
                    run_id = str(event.get("run_id") or "")
                    if run_id:
                        last_run_id = run_id
                    status = str(event.get("status") or "")
                    if status:
                        last_status = status
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        finally:
            # 客户端断开或流结束时统一触发运行时取消信号。
            cancel_event.set()
            if registry is not None:
                registry.update(
                    idempotency_key,
                    user_id,
                    last_run_id,
                    last_status,
                )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers=_sse_headers(),
    )
