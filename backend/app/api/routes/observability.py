"""本地 Prometheus 文本指标端点。"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from backend.app.observability.metrics import render_prometheus

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> PlainTextResponse:
    """直接渲染本地指标，不构造任何业务服务。"""
    return PlainTextResponse(
        render_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
