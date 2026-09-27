"""意图识别智能体的 API 与自动分发聊天页面。"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from backend.app.config import settings

from .agent import QueryIntentAgent
from .schemas import IntentRouteRequest, IntentRouteResponse


router = APIRouter(prefix="/api", tags=["query-intent-agent"])
_agent = QueryIntentAgent(data_dir=settings.smart_case_data_dir)


@router.get("/natural-chat")
def natural_chat_page():
    """返回自动识别任务类型的统一聊天页面。"""
    return FileResponse(Path(__file__).parent / "pages" / "natural_chat_page.html", media_type="text/html; charset=utf-8")


@router.post("/query-intent/route", response_model=IntentRouteResponse)
def route_intent(payload: IntentRouteRequest) -> IntentRouteResponse:
    """识别本轮任务并返回原业务接口的分发信息。"""
    return _agent.route(payload)


@router.get("/query-intent/knowledge-range")
def knowledge_range():
    """返回页面角落展示的本地知识库覆盖时间。"""
    return _agent.knowledge_range()
