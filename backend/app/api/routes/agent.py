"""统一查询兼容接口，全部通过应用服务容器执行业务。"""
from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from backend.app.api.dependencies import ServicesDep
from backend.app.api.schemas import AgentQueryRequest, QueryRequest, StreamQueryRequest


router = APIRouter()


@router.post("/api/query")
def query_cases(payload: QueryRequest, services: ServicesDep):
    """为旧客户端执行非流式结构化优先文档查询。"""
    return services.chat_service.query_document(payload)


@router.post("/api/agent/query")
def agent_query(payload: AgentQueryRequest, services: ServicesDep):
    """面向其他智能体项目保留原响应协议的结构化查询。"""
    return services.chat_service.query_agent(payload)


@router.post("/api/query/stream")
async def unified_stream_query_cases(
    payload: StreamQueryRequest,
    request: Request,
    services: ServicesDep,
):
    """主页面统一流式入口，原样转发当前编排事件协议。"""

    async def event_stream():
        """把编排器事件编码为现有 SSE 数据帧。"""
        async for event in services.chat_service.stream(
            payload,
            request.is_disconnected,
        ):
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Keep-Alive": "timeout=60",
        },
    )
