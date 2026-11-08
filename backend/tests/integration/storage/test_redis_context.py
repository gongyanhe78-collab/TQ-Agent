"""Redis 会话上下文缓存的版本隔离和降级测试。"""
from __future__ import annotations

import unittest

from backend.app.cache.context_cache import CachedSessionStore, RedisContextCache


class _FakeRedis:
    """只实现缓存代码使用的 Redis 命令，模拟真实版本键语义。"""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expires: dict[str, int] = {}
        self.fail_commands: set[str] = set()

    def get(self, key: str) -> str | None:
        self._raise_if_needed("get")
        return self.values.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._raise_if_needed("setex")
        self.values[key] = value
        self.expires[key] = ttl

    def incr(self, key: str) -> int:
        self._raise_if_needed("incr")
        value = int(self.values.get(key) or 0) + 1
        self.values[key] = str(value)
        return value

    def expire(self, key: str, ttl: int) -> bool:
        self._raise_if_needed("expire")
        self.expires[key] = ttl
        return key in self.values

    def ping(self) -> bool:
        self._raise_if_needed("ping")
        return True

    def clear(self) -> None:
        self.values.clear()
        self.expires.clear()

    def _raise_if_needed(self, command: str) -> None:
        if command in self.fail_commands:
            raise ConnectionError(f"模拟 Redis {command} 失败")


class _MemoryDelegate:
    """提供与 SQLAlchemy 会话仓储相同的增量读取接口。"""

    def __init__(self) -> None:
        self.messages: dict[str, list[dict]] = {}
        self.read_count = 0
        self.after_read = None

    def list_messages_after(self, session_id: str, message_id: int = 0) -> list[dict]:
        self.read_count += 1
        result = [
            dict(item)
            for item in self.messages.get(session_id, [])
            if int(item["message_id"]) > int(message_id)
        ]
        if self.after_read is not None:
            callback = self.after_read
            self.after_read = None
            callback()
        return result

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> dict:
        message_id = len(self.messages.setdefault(session_id, [])) + 1
        message = {
            "message_id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "metadata": dict(metadata or {}),
        }
        self.messages[session_id].append(message)
        return dict(message)


class RedisContextCacheTests(unittest.TestCase):
    """验证 Redis 只作为可丢弃缓存，版本必须跟随权威消息变化。"""

    def setUp(self) -> None:
        self.redis = _FakeRedis()
        self.cache = RedisContextCache(
            url="redis://unused",
            ttl_seconds=1800,
            client=self.redis,
        )
        self.delegate = _MemoryDelegate()
        self.store = CachedSessionStore(
            self.delegate,
            cache=self.cache,
            user_id="user-a",
        )
        self.delegate.messages["session-a"] = [
            {
                "message_id": 1,
                "session_id": "session-a",
                "role": "user",
                "content": "第一轮问题",
                "metadata": {},
            }
        ]

    def test_cache_hit_avoids_database_read(self) -> None:
        """同一会话版本第二次读取应直接使用 Redis 结果。"""
        first = self.store.list_messages_after("session-a")
        second = self.store.list_messages_after("session-a")

        self.assertEqual(first, second)
        self.assertEqual(self.delegate.read_count, 1)

    def test_cleared_redis_restores_context_from_database(self) -> None:
        """Redis 数据丢失后应从数据库回填相同上下文。"""
        first = self.store.list_messages_after("session-a")
        self.redis.clear()
        restored = self.store.list_messages_after("session-a")

        self.assertEqual(restored, first)
        self.assertEqual(self.delegate.read_count, 2)

    def test_redis_errors_return_database_messages_and_increment_metric(self) -> None:
        """Redis 命令异常时业务继续，并记录可观测的降级次数。"""
        self.redis.fail_commands.update({"get", "setex", "incr", "expire"})

        messages = self.store.list_messages_after("session-a")

        self.assertEqual([item["content"] for item in messages], ["第一轮问题"])
        self.assertGreaterEqual(self.cache.degraded_count, 1)

    def test_new_message_does_not_hit_stale_version(self) -> None:
        """消息提交后版本递增，旧缓存结果不能再次返回。"""
        self.store.list_messages_after("session-a")
        self.store.add_message("session-a", "assistant", "第一轮回答")

        messages = self.store.list_messages_after("session-a")

        self.assertEqual(
            [item["content"] for item in messages],
            ["第一轮问题", "第一轮回答"],
        )

    def test_backfill_uses_the_version_observed_before_database_read(self) -> None:
        """数据库读取期间出现新版本时，旧结果只能写入旧版本键。"""
        version_key = "risk:session-version:user-a:session-a"
        self.redis.values[version_key] = "3"
        self.delegate.after_read = lambda: self.redis.values.__setitem__(
            version_key,
            "4",
        )

        self.store.list_messages_after("session-a")

        self.assertIn(
            "risk:messages:user-a:session-a:v3:after:0",
            self.redis.values,
        )
        self.assertNotIn(
            "risk:messages:user-a:session-a:v4:after:0",
            self.redis.values,
        )

    def test_failed_invalidation_marks_session_dirty_until_redis_recovers(self) -> None:
        """写入成功后失效失败时，即使旧缓存仍在也不能返回旧上下文。"""
        self.store.list_messages_after("session-a")
        self.redis.fail_commands.update({"incr", "expire"})
        self.store.add_message("session-a", "assistant", "第一轮回答")

        stale_retry = self.store.list_messages_after("session-a")
        self.redis.fail_commands.clear()
        recovered = self.store.list_messages_after("session-a")

        self.assertEqual(
            [item["content"] for item in stale_retry],
            ["第一轮问题", "第一轮回答"],
        )
        self.assertEqual(
            [item["content"] for item in recovered],
            ["第一轮问题", "第一轮回答"],
        )
        self.assertEqual(self.delegate.read_count, 3)


if __name__ == "__main__":
    unittest.main()
