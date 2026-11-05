"""FastAPI 应用生命周期与配置初始化。"""
from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator

from backend.app.core.config import get_settings

if TYPE_CHECKING:
    from fastapi import FastAPI

    from backend.app.core.config import Settings
    from backend.app.services.container import AppServices


def initialize_settings(settings: "Settings") -> None:
    """准备配置要求的目录与 Numba 缓存环境变量。"""
    settings.ensure_directories()
    os.environ.setdefault("NUMBA_CACHE_DIR", str(settings.numba_cache_dir))


@asynccontextmanager
async def lifespan(
    app: "FastAPI",
    *,
    services_factory: Callable[[], "AppServices"] | None = None,
) -> AsyncIterator[None]:
    """创建应用级服务容器，并在退出时释放可关闭资源。"""
    from backend.app.services.container import build_services

    settings = get_settings()
    initialize_settings(settings)
    services = services_factory() if services_factory else build_services(settings)
    app.state.services = services
    try:
        yield
    finally:
        services.close()
