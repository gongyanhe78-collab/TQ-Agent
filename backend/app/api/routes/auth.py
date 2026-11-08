"""注册、登录、刷新、登出和当前用户接口。"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status

from backend.app.api.dependencies import CurrentUserDep, ServicesDep
from backend.app.api.schemas import LoginRequest, RegisterRequest
from backend.app.services.auth_service import (
    AuthTokens,
    InvalidCredentials,
    InvalidRegistration,
    UserAlreadyExists,
)


LOGGER = logging.getLogger("uvicorn.error")
router = APIRouter(prefix="/api/auth", tags=["auth"])


def _auth_response(tokens: AuthTokens, response: Response, services: Any) -> dict:
    """设置刷新 Cookie，并返回只含短期访问令牌的响应正文。"""
    settings = services.settings
    response.set_cookie(
        key=settings.auth_refresh_cookie_name,
        value=tokens.refresh_token,
        max_age=int(services.auth_service.refresh_ttl.total_seconds()),
        httponly=True,
        secure=bool(settings.auth_cookie_secure),
        samesite=settings.auth_cookie_samesite,
        path="/api/auth",
    )
    return {
        "access_token": tokens.access_token,
        "token_type": "bearer",
        "expires_in": tokens.expires_in,
        "user": tokens.user,
    }


def _read_refresh_token(request: Request, services: Any) -> str:
    """只从 HttpOnly Cookie 读取刷新令牌。"""
    name = services.settings.auth_refresh_cookie_name
    return str(request.cookies.get(name) or "")


def _clear_refresh_cookie(response: Response, services: Any) -> None:
    """按原 Path 清除浏览器刷新 Cookie。"""
    settings = services.settings
    response.delete_cookie(
        key=settings.auth_refresh_cookie_name,
        path="/api/auth",
        secure=bool(settings.auth_cookie_secure),
        httponly=True,
        samesite=settings.auth_cookie_samesite,
    )


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    response: Response,
    services: ServicesDep,
) -> dict:
    """注册普通用户并直接签发首组令牌。"""
    try:
        services.auth_service.register(payload.username, payload.password)
        tokens = services.auth_service.login(payload.username, payload.password)
    except UserAlreadyExists as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidRegistration as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    LOGGER.info("[认证] 注册成功 username=%s", tokens.user["username"])
    return _auth_response(tokens, response, services)


@router.post("/login")
def login(
    payload: LoginRequest,
    response: Response,
    services: ServicesDep,
) -> dict:
    """校验账号密码并轮换出一组新令牌。"""
    try:
        tokens = services.auth_service.login(payload.username, payload.password)
    except InvalidCredentials as exc:
        LOGGER.warning("[认证] 登录失败 username=%s", payload.username)
        raise HTTPException(
            status_code=401,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    LOGGER.info("[认证] 登录成功 username=%s", tokens.user["username"])
    return _auth_response(tokens, response, services)


@router.post("/refresh")
def refresh(
    request: Request,
    response: Response,
    services: ServicesDep,
) -> dict:
    """使用刷新 Cookie 轮换令牌；失败时清除失效 Cookie。"""
    refresh_token = _read_refresh_token(request, services)
    if not refresh_token:
        _clear_refresh_cookie(response, services)
        raise HTTPException(status_code=401, detail="刷新令牌无效")
    try:
        tokens = services.auth_service.refresh(refresh_token)
    except InvalidCredentials as exc:
        LOGGER.warning("[认证] 刷新失败")
        _clear_refresh_cookie(response, services)
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return _auth_response(tokens, response, services)


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    services: ServicesDep,
) -> dict:
    """撤销刷新令牌族并清除 Cookie。"""
    refresh_token = _read_refresh_token(request, services)
    if refresh_token:
        services.auth_service.logout(refresh_token)
    _clear_refresh_cookie(response, services)
    LOGGER.info("[认证] 退出完成")
    return {"logged_out": True}


@router.get("/me")
def me(user: CurrentUserDep) -> dict:
    """返回当前访问令牌对应的安全用户信息。"""
    return user
