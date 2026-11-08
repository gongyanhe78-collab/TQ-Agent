"""相似个例智能匹配体的 HTTP 路由。"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from backend.app.agents.smart_case_match.infrastructure.stream_events import encode_sse
from backend.app.agents.smart_case_match.schemas import (
    NaturalLanguageMatchRequest,
    SmartCaseMatchRequest,
    SmartCaseMatchResponse,
)
from backend.app.agents.smart_case_match.workflow import progress
from backend.app.api.dependencies import CurrentUserDep, ServicesDep


router = APIRouter(prefix="/api/smart-case-match", tags=["smart-case-match"])
SMART_AGENT_DIR = Path(__file__).resolve().parents[2] / "agents" / "smart_case_match"


@router.get("/health")
def health(services: ServicesDep):
    """检查本地资料、Embedding、LLM 和 LangGraph 状态。"""
    try:
        return services.smart_agent.health()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"相似个例智能体健康检查失败：{exc}",
        ) from exc


@router.get("/chat")
def chat_page():
    """返回独立的相似个例匹配工作台。"""
    return FileResponse(
        SMART_AGENT_DIR / "pages" / "chat_page.html",
        media_type="text/html; charset=utf-8",
    )


@router.get("/conversation")
def conversation_page():
    """返回自然语言输入的流式相似个例聊天页面。"""
    return FileResponse(
        SMART_AGENT_DIR / "pages" / "conversation_page.html",
        media_type="text/html; charset=utf-8",
    )


@router.get("/progress/{progress_id}")
def match_progress(progress_id: str, user: CurrentUserDep):
    """返回轻量任务进度，不包含业务正文和模型输入。"""
    state = progress.get(progress_id, owner_id=str(user["user_id"]))
    if state is None:
        raise HTTPException(status_code=404, detail="匹配进度不存在")
    return state


@router.post("/search", response_model=SmartCaseMatchResponse)
def search(
    payload: SmartCaseMatchRequest,
    services: ServicesDep,
    user: CurrentUserDep,
):
    """执行召回、语义补充、重排、逐例提炼和综合提示生成。"""
    try:
        return services.smart_agent.run(payload, owner_id=str(user["user_id"]))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"相似个例匹配失败：{exc}") from exc


@router.post("/conversation/stream")
async def stream_conversation(
    payload: NaturalLanguageMatchRequest,
    request: Request,
    services: ServicesDep,
    user: CurrentUserDep,
):
    """按 SSE 事件返回自然语言解析、匹配个例和综合研判。"""

    async def generate():
        """编码相似个例 Agent 的阶段事件。"""
        async for event, data in services.smart_agent.stream_natural_match(
            payload,
            request.is_disconnected,
            owner_id=str(user["user_id"]),
        ):
            yield encode_sse(event, data)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/assets/chunks/{chunk_id}")
def chunk_asset(
    chunk_id: str,
    services: ServicesDep,
    user: CurrentUserDep,
):
    """按 chunk_id 返回文字证据正文。"""
    try:
        return services.smart_agent.chunk_detail(chunk_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/assets/images/{image_id}")
def image_asset(
    image_id: str,
    services: ServicesDep,
    user: CurrentUserDep,
):
    """按 image_id 读取本地图片。"""
    try:
        return FileResponse(services.smart_agent.image_path(image_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
