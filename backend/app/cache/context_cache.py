"""Redis 优先、PostgreSQL 回退的会话上下文缓存。"""
from __future__ import annotations

import json
import logging
from typing import Any

LOGGER = logging.getLogger("uvicorn.error")


class RedisContextCache:
    """缓存近期消息增量；Redis 异常始终降级到数据库。"""

    def __init__(
        self,
        *,
        url: str,
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
        self.degraded_count = 0
        self._dirty_sessions: set[tuple[str, str]] = set()

    def get_messages_after(
        self,
        user_id: str,
        session_id: str,
        message_id: int,
        *,
        version: str | None = None,
    ) -> list[dict] | None:
        """读取按版本隔离的消息增量。"""
        client = self._get_client()
        if client is None:
            return None
        if not self._ensure_clean(user_id, session_id):
            return None
        try:
            resolved_version = (
                version
                if version is not None
                else self._get_version(user_id, session_id, client)
            )
            if resolved_version is None:
                return None
            key = self._messages_key(
                user_id,
                session_id,
                message_id,
                resolved_version,
            )
            raw = client.get(key)
            if not raw:
                return None
            value = json.loads(raw)
            return value if isinstance(value, list) else None
        except Exception as exc:
            self._degrade("read", exc)
            return None

    def set_messages_after(
        self,
        user_id: str,
        session_id: str,
        message_id: int,
        messages: list[dict],
        *,
        version: str | None = None,
    ) -> None:
        """缓存数据库读取结果，键内版本避免旧数据覆盖新消息。"""
        client = self._get_client()
        if client is None or (user_id, session_id) in self._dirty_sessions:
            return
        try:
            resolved_version = (
                version
                if version is not None
                else self._get_version(user_id, session_id, client)
            )
            if resolved_version is None:
                return
            key = self._messages_key(
                user_id,
                session_id,
                message_id,
                resolved_version,
            )
            client.setex(
                key,
                self.ttl_seconds,
                json.dumps(messages, ensure_ascii=False),
            )
        except Exception as exc:
            self._degrade("write", exc)

    def invalidate_session(self, user_id: str, session_id: str) -> bool:
        """通过会话版本自增使所有旧缓存立即失效。"""
        client = self._get_client()
        if client is None:
            self._dirty_sessions.add((user_id, session_id))
            return False
        try:
            client.incr(self._version_key(user_id, session_id))
            client.expire(
                self._version_key(user_id, session_id),
                self.ttl_seconds,
            )
            self._dirty_sessions.discard((user_id, session_id))
            return True
        except Exception as exc:
            self._dirty_sessions.add((user_id, session_id))
            self._degrade("invalidate", exc)
            return False

    def get_version(self, user_id: str, session_id: str) -> str | None:
        """在数据库读取前读取一次版本，回填时复用该快照。"""
        client = self._get_client()
        if client is None:
            return None
        if not self._ensure_clean(user_id, session_id):
            return None
        return self._get_version(user_id, session_id, client)

    def get_memory_context(self, user_id: str) -> dict | None:
        """读取当前用户短期记忆上下文缓存。"""
        client = self._get_client()
        if client is None:
            return None
        try:
            raw = client.get(self._memory_context_key(user_id))
            if not raw:
                return None
            value = json.loads(raw)
            return value if isinstance(value, dict) else None
        except Exception as exc:
            self._degrade("memory_read", exc)
            return None

    def set_memory_context(
        self,
        user_id: str,
        payload: dict,
        ttl_seconds: int = 1800,
    ) -> None:
        """缓存当前用户记忆上下文，TTL 默认不超过 30 分钟。"""
        client = self._get_client()
        if client is None:
            return
        try:
            client.setex(
                self._memory_context_key(user_id),
                min(1800, max(60, int(ttl_seconds))),
                json.dumps(payload, ensure_ascii=False),
            )
        except Exception as exc:
            self._degrade("memory_write", exc)

    def invalidate_user(self, user_id: str) -> bool:
        """删除当前用户的记忆上下文缓存，供记忆删除后清理。"""
        client = self._get_client()
        if client is None:
            return False
        try:
            client.delete(self._memory_context_key(user_id))
            return True
        except Exception as exc:
            self._degrade("memory_invalidate", exc)
            return False

    def ping(self) -> bool:
        """返回 Redis 当前是否可响应，健康检查不应抛异常。"""
        client = self._get_client()
        if client is None:
            return False
        try:
            return bool(client.ping())
        except Exception as exc:
            self._degrade("ping", exc)
            return False

    def _get_client(self) -> Any | None:
        """按需创建 Redis 客户端，导入和启动阶段不强制连接。"""
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
                self._degrade("client", exc)
                return None
        return self._client

    def _memory_context_key(self, user_id: str) -> str:
        """构造当前用户记忆上下文缓存键。"""
        return f"{self.prefix}:memory-context:{str(user_id)}"

    def _messages_key(
        self,
        user_id: str,
        session_id: str,
        message_id: int,
        version: str,
    ) -> str:
        """构造带会话版本的只读消息缓存键。"""
        return (
            f"{self.prefix}:messages:{user_id}:{session_id}:"
            f"v{version}:after:{max(0, int(message_id))}"
        )

    def _get_version(
        self,
        user_id: str,
        session_id: str,
        client: Any,
    ) -> str | None:
        """读取当前版本；失败时返回空并触发数据库降级。"""
        try:
            return str(client.get(self._version_key(user_id, session_id)) or "0")
        except Exception as exc:
            self._degrade("version", exc)
            return None

    def _ensure_clean(self, user_id: str, session_id: str) -> bool:
        """失效失败的会话保持脏标记，避免旧缓存继续被命中。"""
        if (user_id, session_id) not in self._dirty_sessions:
            return True
        return self.invalidate_session(user_id, session_id)

    def _version_key(self, user_id: str, session_id: str) -> str:
        """返回用户会话缓存版本键。"""
        return f"{self.prefix}:session-version:{user_id}:{session_id}"

    def _degrade(self, operation: str, exc: Exception) -> None:
        """记录降级次数，失败不影响 PostgreSQL 权威数据。"""
        self.degraded_count += 1
        LOGGER.warning("[缓存] Redis %s失败，已回退数据库: %s", operation, exc)


class CachedSessionStore:
    """在不改变旧 SessionStore 接口的前提下增加上下文缓存。"""

    def __init__(
        self,
        delegate: Any,
        *,
        cache: RedisContextCache,
        user_id: str,
    ) -> None:
        self.delegate = delegate
        self.cache = cache
        self.user_id = user_id

    def list_messages_after(self, session_id: str, message_id: int = 0) -> list[dict]:
        """优先读取 Redis，未命中或异常时从 PostgreSQL 回填。"""
        version = self.cache.get_version(self.user_id, session_id)
        cached = self.cache.get_messages_after(
            self.user_id,
            session_id,
            int(message_id),
            version=version,
        )
        if cached is not None:
            return cached
        messages = self.delegate.list_messages_after(session_id, int(message_id))
        self.cache.set_messages_after(
            self.user_id,
            session_id,
            int(message_id),
            messages,
            version=version,
        )
        return messages

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> dict:
        """消息落库后使缓存失效，并在下一次读取时从权威库回填。"""
        message = self.delegate.add_message(session_id, role, content, metadata)
        self.cache.invalidate_session(self.user_id, session_id)
        return message

    def update_message_metadata(
        self,
        session_id: str,
        message_id: int,
        patch: dict,
    ) -> dict:
        """元数据落库后使会话上下文缓存失效。"""
        message = self.delegate.update_message_metadata(session_id, message_id, patch)
        self.cache.invalidate_session(self.user_id, session_id)
        return message

    def delete_session(self, session_id: str) -> bool:
        """删除会话后清理其派生缓存。"""
        deleted = self.delegate.delete_session(session_id)
        self.cache.invalidate_session(self.user_id, session_id)
        return deleted

    def update_session_title(self, session_id: str, title: str) -> dict:
        """标题更新后清理上下文版本。"""
        session = self.delegate.update_session_title(session_id, title)
        self.cache.invalidate_session(self.user_id, session_id)
        return session

    def __getattr__(self, name: str) -> Any:
        """其余只读与业务方法直接委托权威仓储。"""
        return getattr(self.delegate, name)
