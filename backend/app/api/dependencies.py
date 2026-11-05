"""FastAPI 路由依赖。"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request, HTTPException

from backend.app.services.container import AppServices


def get_services(request: Request) -> AppServices:
    """返回当前应用生命周期创建的服务容器。"""
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(status_code=503, detail="Application services are not ready")
    return services


ServicesDep = Annotated[AppServices, Depends(get_services)]
