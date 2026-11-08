"""查询幂等注册表：Redis 优先、进程内存自动回退。"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from backend.app.core.config import get_settings

LOGGER = logging.getLogger("uvicorn.error")


class QueryIdempotencyRegistry:
    """以 user_id + idempotency_key 为键登记查询运行，避免重复写入。

    Redis 异常时自动降级到进程内内存表，行为与 RedisContextCache 一致；
    幂等判断失败不会阻塞主查询，只会放行当前请求。
    """

    def __init__(
        self,
        *,
        url: str = "",
        prefix: str = "risk",
        ttl_seconds: int = 1800,
        enabled: bool = True,
        client: Any | None = None,
    ) -> None:
        self.url = url
        self.prefix = prefix
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.enabled = bool(enabled)
        self._client = client
        self._memory: dict[str, tuple[dict[str, str], float]] = {}
        self._memory_lock = threading.Lock()

    def claim(self, key: str, user_id: str, run_id: str = "") -> bool:
        """原子占位幂等键；键已存在时返回 False，避免并发重复处理。"""
        normalized_key = self._normalize_key(key, user_id)
        entry = json.dumps(
            {"run_id": run_id, "status": "running"},
            ensure_ascii=False,
        )
        client = self._get_client()
        if client is not None:
            try:
                if client.set(
                    normalized_key,
                    entry,
                    nx=True,
                    ex=self.ttl_seconds,
                ):
                    return True
                return False
            except Exception as exc:
                self._degrade("claim", exc)
        with self._memory_lock:
            now = time.monotonic()
            existing = self._memory.get(normalized_key)
            if existing is not None and existing[1] > now:
                return False
            self._memory[normalized_key] = (
                {"run_id": run_id, "status": "running"},
                now + self.ttl_seconds,
            )
            return True

    def update(
        self,
        key: str,
        user_id: str,
        run_id: str,
        status: str,
    ) -> None:
        """把键对应的运行编号与终态写回，供重复请求读取。"""
        normalized_key = self._normalize_key(key, user_id)
        entry = json.dumps(
            {"run_id": run_id, "status": status},
            ensure_ascii=False,
        )
        client = self._get_client()
        if client is not None:
            try:
                client.setex(normalized_key, self.ttl_seconds, entry)
                return
            except Exception as exc:
                self._degrade("update", exc)
        with self._memory_lock:
            self._memory[normalized_key] = (
                {"run_id": run_id, "status": status},
                time.monotonic() + self.ttl_seconds,
            )

    def lookup(self, key: str, user_id: str) -> dict[str, str] | None:
        """返回已登记的运行信息；未登记或已过期返回 None。"""
        normalized_key = self._normalize_key(key, user_id)
        client = self._get_client()
        if client is not None:
            try:
                raw = client.get(normalized_key)
                if not raw:
                    return None
                value = json.loads(raw)
                return value if isinstance(value, dict) else None
            except Exception as exc:
                self._degrade("lookup", exc)
        with self._memory_lock:
            now = time.monotonic()
            existing = self._memory.get(normalized_key)
            if existing is None or existing[1] <= now:
                self._memory.pop(normalized_key, None)
                return None
            return dict(existing[0])

    def _normalize_key(self, key: str, user_id: str) -> str:
        """生成带用户隔离的稳定缓存键。"""
        return f"{self.prefix}:idem:{user_id}:{key}"

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
        """记录 Redis 降级但不抛出，保证幂等判断不阻塞主查询。"""
        LOGGER.warning(
            "[幂等] Redis %s 失败，降级内存存储 error=%s",
            operation,
            exc,
        )


_REGISTRY: QueryIdempotencyRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_query_idempotency() -> QueryIdempotencyRegistry:
    """返回进程级幂等注册表单例，首次访问按当前配置创建。"""
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            settings = get_settings()
            _REGISTRY = QueryIdempotencyRegistry(
                url=settings.redis_url,
                prefix=settings.redis_key_prefix,
                ttl_seconds=settings.idempotency_ttl_seconds,
                enabled=settings.redis_cache_enabled,
            )
        return _REGISTRY


def duplicate_event(
    registry: QueryIdempotencyRegistry,
    key: str,
    user_id: str,
) -> dict[str, str] | None:
    """幂等键已登记时返回去重事件；未登记返回 None 放行新请求。"""
    if not key:
        return None
    existing = registry.lookup(key, user_id)
    if existing is None:
        return None
    status = str(existing.get("status") or "running")
    message = (
        "该请求正在处理中，已通过幂等键去重"
        if status == "running"
        else "该请求已处理，未重复执行"
    )
    return {
        "type": "duplicate",
        "run_id": str(existing.get("run_id") or ""),
        "status": status,
        "message": message,
    }
