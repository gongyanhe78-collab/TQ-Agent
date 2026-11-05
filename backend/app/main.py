"""FastAPI 应用工厂，只负责中间件、生命周期和路由装配。"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import (
    agent,
    case_multidim,
    query_intent,
    sessions,
    smart_case_match,
    system,
    vectors,
)
from backend.app.core.lifecycle import lifespan

if TYPE_CHECKING:
    from backend.app.services.container import AppServices


ROUTERS = (
    agent.router,
    sessions.router,
    case_multidim.router,
    smart_case_match.router,
    query_intent.router,
    system.router,
    vectors.router,
)


def register_routes(app: FastAPI) -> None:
    """按稳定的路由顺序注册全部 API。"""
    for router in ROUTERS:
        app.include_router(router)


def create_app(
    services_factory: Callable[[], "AppServices"] | None = None,
) -> FastAPI:
    """创建应用；服务仅在真实生命周期启动时构造。"""

    @asynccontextmanager
    async def app_lifespan(application: FastAPI):
        """把可选测试服务工厂传入统一生命周期。"""
        async with lifespan(
            application,
            services_factory=services_factory,
        ):
            yield

    app = FastAPI(
        title="Weather Case RAG System",
        version="0.2.0",
        lifespan=app_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_routes(app)
    return app


app = create_app()
