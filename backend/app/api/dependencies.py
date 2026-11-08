"""FastAPI 认证、用户存储和路由依赖。"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Request, HTTPException

from backend.app.services.container import AppServices
from backend.app.services.auth_service import InvalidCredentials


def get_services(request: Request) -> AppServices:
    """返回当前应用生命周期创建的服务容器。"""
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(status_code=503, detail="Application services are not ready")
    return services


ServicesDep = Annotated[AppServices, Depends(get_services)]


def _bearer_token(request: Request) -> str:
    """从标准 Authorization 头提取访问令牌。"""
    value = str(request.headers.get("authorization") or "").strip()
    scheme, separator, token = value.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token.strip():
        return ""
    return token.strip()


def get_current_user(request: Request, services: ServicesDep) -> dict[str, Any]:
    """校验访问令牌并返回当前用户，所有受保护接口共用同一 401 契约。"""
    token = _bearer_token(request)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="需要登录",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return services.auth_service.get_current_user(token)
    except InvalidCredentials as exc:
        raise HTTPException(
            status_code=401,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


CurrentUserDep = Annotated[dict[str, Any], Depends(get_current_user)]


def get_optional_user(
    request: Request,
    services: ServicesDep,
) -> dict[str, Any] | None:
    """读取可选认证用户，仅用于明确允许匿名的公开接口。"""
    if not _bearer_token(request):
        return None
    return get_current_user(request, services)


OptionalUserDep = Annotated[dict[str, Any] | None, Depends(get_optional_user)]


def require_admin(user: CurrentUserDep) -> dict[str, Any]:
    """要求当前用户具有管理员角色。"""
    if str(user.get("role") or "") != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


AdminUserDep = Annotated[dict[str, Any], Depends(require_admin)]


def get_session_store(services: ServicesDep, user: CurrentUserDep) -> Any:
    """构造绑定当前用户的会话仓储，禁止请求体指定用户身份。"""
    factory = getattr(services, "session_store_for", None)
    if factory is None:
        raise HTTPException(status_code=503, detail="会话存储尚未就绪")
    return factory(str(user["user_id"]))


SessionStoreDep = Annotated[Any, Depends(get_session_store)]


def get_chat_service(services: ServicesDep, user: CurrentUserDep) -> Any:
    """构造绑定当前用户的统一查询服务。"""
    factory = getattr(services, "chat_service_for", None)
    if factory is None:
        raise HTTPException(status_code=503, detail="查询服务尚未就绪")
    return factory(str(user["user_id"]))


ChatServiceDep = Annotated[Any, Depends(get_chat_service)]


def get_multidim_page_store(
    services: ServicesDep,
    user: CurrentUserDep,
) -> Any:
    """构造只属于当前用户的多维自然语言会话仓储。"""
    factory = getattr(services, "multidim_page_store_for", None)
    if factory is None:
        raise HTTPException(status_code=503, detail="多维会话存储尚未就绪")
    return factory(str(user["user_id"]))


MultidimPageStoreDep = Annotated[Any, Depends(get_multidim_page_store)]


def get_multidim_agent(services: ServicesDep, user: CurrentUserDep) -> Any:
    """构造绑定当前用户输出目录的多维检索 Agent。"""
    factory = getattr(services, "multidim_agent_for", None)
    if factory is None:
        raise HTTPException(status_code=503, detail="多维检索服务尚未就绪")
    return factory(str(user["user_id"]))


MultidimAgentDep = Annotated[Any, Depends(get_multidim_agent)]


def get_memory_service(services: ServicesDep, user: CurrentUserDep) -> Any:
    """构造绑定当前用户的长期记忆服务。"""
    factory = getattr(services, "memory_service_for", None)
    if factory is None:
        raise HTTPException(status_code=503, detail="长期记忆服务尚未就绪")
    return factory(str(user["user_id"]))


MemoryServiceDep = Annotated[Any, Depends(get_memory_service)]
