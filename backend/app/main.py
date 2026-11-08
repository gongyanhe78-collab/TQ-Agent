"""FastAPI 应用工厂，只负责中间件、生命周期和路由装配。"""
from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import (
    agent,
    auth,
    case_multidim,
    memories,
    observability,
    query_intent,
    sessions,
    smart_case_match,
    system,
    vectors,
)
from backend.app.core.lifecycle import lifespan
from backend.app.observability.middleware import ObservabilityMiddleware
from backend.app.observability.tracing import shutdown_tracing
from backend.app.security.rate_limit import RateLimitMiddleware

if TYPE_CHECKING:
    from backend.app.services.container import AppServices


ROUTERS = (
    auth.router,
    agent.router,
    memories.router,
    sessions.router,
    case_multidim.router,
    smart_case_match.router,
    query_intent.router,
    system.router,
    vectors.router,
    observability.router,
)


def _cors_origins() -> list[str]:
    """读取显式前端来源，凭据模式下禁止通配来源。"""
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173",
    )
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return [item for item in origins if item != "*"] or [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]


def register_routes(app: FastAPI) -> None:
    """按稳定的路由顺序注册全部 API。"""
    for router in ROUTERS:
        app.include_router(router)


def create_app(
    services_factory: Callable[[], AppServices] | None = None,
) -> FastAPI:
    """创建应用；服务仅在真实生命周期启动时构造。"""

    @asynccontextmanager
    async def app_lifespan(application: FastAPI):
        """把可选测试服务工厂传入统一生命周期。"""
        try:
            async with lifespan(
                application,
                services_factory=services_factory,
            ):
                yield
        finally:
            shutdown_tracing()

    app = FastAPI(
        title="Weather Case RAG System",
        version="0.2.0",
        lifespan=app_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(ObservabilityMiddleware)
    # 登录与查询限流放在最外层，超限时在进入业务前直接返回 429。
    app.add_middleware(RateLimitMiddleware)
    register_routes(app)
    return app


app = create_app()
