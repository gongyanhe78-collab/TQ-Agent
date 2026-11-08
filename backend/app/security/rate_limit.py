"""固定窗口限流：按用户/IP 与路由类别限制登录和查询频率。"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from starlette.types import ASGIApp, Receive, Scope, Send

from backend.app.core.config import get_settings

LOGGER = logging.getLogger("uvicorn.error")

# 路由类别：登录类与查询类分别使用独立配额。
LOGIN_PATHS = {"/api/auth/login", "/api/auth/register"}
QUERY_PATHS = {
    "/api/query",
    "/api/query/stream",
    "/api/agent/query",
}


def route_category(path: str) -> str:
    """按请求路径返回限流类别；未命中返回空字符串表示不限流。"""
    if path in LOGIN_PATHS:
        return "login"
    if path in QUERY_PATHS:
        return "query"
    return ""


class FixedWindowRateLimiter:
    """Redis 优先、进程内存回退的固定窗口限流器。"""

    def __init__(
        self,
        *,
        url: str = "",
        prefix: str = "risk",
        enabled: bool = True,
        client: Any | None = None,
        clock: Any | None = None,
    ) -> None:
        self.url = url
        self.prefix = prefix
        self.enabled = bool(enabled)
        self._client = client
        self._clock = clock or time.monotonic
        self._memory: dict[str, list[float]] = {}
        self._memory_lock = threading.Lock()

    def allow(
        self,
        key: str,
        *,
        limit: int,
        window_seconds: int,
    ) -> bool:
        """固定窗口内计数是否未超限；超限返回 False。"""
        now = self._clock()
        normalized_key = self._normalize_key(key)
        client = self._get_client()
        if client is not None:
            try:
                bucket = int(now // window_seconds)
                redis_key = f"{normalized_key}:{bucket}"
                count = client.incr(redis_key)
                if count == 1:
                    client.expire(redis_key, window_seconds + 1)
                return int(count) <= max(1, int(limit))
            except Exception as exc:
                self._degrade("incr", exc)
        with self._memory_lock:
            bucket = int(now // window_seconds)
            stamp = now
            counts = self._memory.get(normalized_key)
            if counts is None or int(stamp // window_seconds) != int(
                (counts[-1] if counts else stamp) // window_seconds
            ):
                counts = []
                self._memory[normalized_key] = counts
            counts.append(stamp)
            return len(counts) <= max(1, int(limit))

    def _normalize_key(self, key: str) -> str:
        """生成稳定限流键，避免把完整身份信息写入日志。"""
        return f"{self.prefix}:rl:{key}"

    def _get_client(self) -> Any | None:
        """按需创建 Redis 客户端，导入与启动阶段不强制连接。"""
        if not self.enabled:
            return None
        if self._client is None:
            try:
                from redis import Redis

                self._client = Redis.from_url(
                    self.url,
                    decode_responses=True,
                    socket_connect_timeout=1,
                    socket_timeout=1,
                    health_check_interval=30,
                )
            except Exception as exc:
                self.enabled = False
                self._degrade("connect", exc)
        return self._client

    def _degrade(self, operation: str, exc: Exception) -> None:
        """记录 Redis 降级但不抛出，保证限流判断不阻塞请求。"""
        LOGGER.warning(
            "[限流] Redis %s 失败，降级内存计数 error=%s",
            operation,
            exc,
        )


_LIMITER: FixedWindowRateLimiter | None = None
_LIMITER_LOCK = threading.Lock()


def get_rate_limiter() -> FixedWindowRateLimiter:
    """返回进程级限流器单例，首次访问按当前配置创建。"""
    global _LIMITER
    with _LIMITER_LOCK:
        if _LIMITER is None:
            settings = get_settings()
            _LIMITER = FixedWindowRateLimiter(
                url=settings.redis_url,
                prefix=settings.redis_key_prefix,
                enabled=settings.rate_limit_enabled,
            )
        return _LIMITER


class RateLimitMiddleware:
    """对登录与查询路由按类别限流，超限返回 429 并带 Retry-After 头。"""

    def __init__(
        self,
        app: ASGIApp,
        *,
        limiter: FixedWindowRateLimiter | None = None,
    ) -> None:
        self.app = app
        self._limiter = limiter

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """按路径类别检查配额，未超限则继续处理请求。"""
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        category = route_category(str(scope.get("path") or ""))
        if not category:
            await self.app(scope, receive, send)
            return
        limiter = self._limiter or get_rate_limiter()
        settings = get_settings()
        client_ip = self._client_ip(scope)
        if category == "login":
            limit = settings.rate_limit_login_max
            window = settings.rate_limit_login_window_seconds
            key = f"login:{client_ip}"
        else:
            limit = settings.rate_limit_query_max
            window = settings.rate_limit_query_window_seconds
            user_id = self._user_id(scope)
            key = f"query:{user_id or client_ip}"
        if not limiter.allow(key, limit=limit, window_seconds=window):
            await self._send_429(send)
            return
        await self.app(scope, receive, send)

    @staticmethod
    def _client_ip(scope: Scope) -> str:
        """读取客户端 IP；代理场景回退为固定标识。"""
        client = scope.get("client")
        return str(client[0]) if client and client[0] else "unknown"

    @staticmethod
    def _user_id(scope: Scope) -> str:
        """读取当前请求用户 ID；未认证时为空字符串。"""
        from backend.app.observability.context import current_context

        return str(current_context().user_id or "")

    @staticmethod
    async def _send_429(send: Send) -> None:
        """发送 429 响应并附 Retry-After 头。"""
        body = '{"detail":"请求过于频繁，请稍后再试"}'.encode()
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"retry-after", b"60"),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
            }
        )
