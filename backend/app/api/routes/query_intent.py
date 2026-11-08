"""意图识别智能体的 API 与自动分发聊天页面。"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from backend.app.agents.query_intent.schemas import (
    IntentRouteRequest,
    IntentRouteResponse,
)
from backend.app.api.dependencies import CurrentUserDep, ServicesDep


router = APIRouter(prefix="/api", tags=["query-intent"])
QUERY_INTENT_DIR = Path(__file__).resolve().parents[2] / "agents" / "query_intent"


@router.get("/natural-chat")
def natural_chat_page():
    """返回自动识别任务类型的统一聊天页面。"""
    return FileResponse(
        QUERY_INTENT_DIR / "pages" / "natural_chat_page.html",
        media_type="text/html; charset=utf-8",
    )


@router.post("/query-intent/route", response_model=IntentRouteResponse)
def route_intent(
    payload: IntentRouteRequest,
    services: ServicesDep,
    user: CurrentUserDep,
) -> IntentRouteResponse:
    """识别本轮任务并返回原业务接口的分发信息。"""
    return services.intent_agent.route(payload)


@router.get("/query-intent/knowledge-range")
def knowledge_range(services: ServicesDep, user: CurrentUserDep):
    """返回页面角落展示的本地知识库覆盖时间。"""
    return services.intent_agent.knowledge_range()
