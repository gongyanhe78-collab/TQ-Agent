"""登录与查询限流的固定窗口实现单元测试。"""
from __future__ import annotations

import unittest

from backend.app.security.rate_limit import (
    FixedWindowRateLimiter,
    route_category,
)


class _FakeClock:
    """可手动推进的单调时钟，用于验证窗口滚动。"""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeRedis:
    """记录 incr 与 expire 调用的内存 Redis 替身。"""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.expired: list[str] = []

    def incr(self, key: str) -> int:
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def expire(self, key: str, seconds: int) -> None:
        self.expired.append(key)


class RouteCategoryTests(unittest.TestCase):
    """验证登录与查询路由类别划分。"""

    def test_login_paths_are_categorized_as_login(self) -> None:
        self.assertEqual(route_category("/api/auth/login"), "login")
        self.assertEqual(route_category("/api/auth/register"), "login")

    def test_query_paths_are_categorized_as_query(self) -> None:
        self.assertEqual(route_category("/api/query"), "query")
        self.assertEqual(route_category("/api/query/stream"), "query")
        self.assertEqual(route_category("/api/agent/query"), "query")

    def test_other_paths_are_not_limited(self) -> None:
        self.assertEqual(route_category("/api/sessions"), "")
        self.assertEqual(route_category("/health"), "")


class FixedWindowRateLimiterTests(unittest.TestCase):
    """验证内存回退与 Redis 优先的固定窗口计数。"""

    def setUp(self) -> None:
        self.clock = _FakeClock()

    def test_memory_limiter_allows_until_limit_then_rejects(self) -> None:
        limiter = FixedWindowRateLimiter(
            url="redis://unused",
            enabled=True,
            clock=self.clock,
        )
        key = "query:user-1"
        for _ in range(3):
            self.assertTrue(limiter.allow(key, limit=3, window_seconds=60))
        self.assertFalse(limiter.allow(key, limit=3, window_seconds=60))

    def test_memory_limiter_resets_when_window_rolls(self) -> None:
        limiter = FixedWindowRateLimiter(
            url="redis://unused",
            enabled=True,
            clock=self.clock,
        )
        key = "login:127.0.0.1"
        self.assertTrue(limiter.allow(key, limit=1, window_seconds=60))
        self.assertFalse(limiter.allow(key, limit=1, window_seconds=60))
        self.clock.advance(61)
        self.assertTrue(limiter.allow(key, limit=1, window_seconds=60))

    def test_redis_limiter_uses_incr_and_expire(self) -> None:
        redis = _FakeRedis()
        limiter = FixedWindowRateLimiter(
            url="redis://unused",
            enabled=True,
            client=redis,
            clock=self.clock,
        )
        key = "query:user-2"
        self.assertTrue(limiter.allow(key, limit=2, window_seconds=60))
        self.assertTrue(limiter.allow(key, limit=2, window_seconds=60))
        self.assertFalse(limiter.allow(key, limit=2, window_seconds=60))
        self.assertEqual(len(redis.expired), 1)

    def test_redis_failure_falls_back_to_memory(self) -> None:
        class _BrokenRedis:
            def incr(self, _key: str) -> int:
                raise RuntimeError("redis down")

            def expire(self, _key: str, _seconds: int) -> None:
                raise RuntimeError("redis down")

        limiter = FixedWindowRateLimiter(
            url="redis://unused",
            enabled=True,
            client=_BrokenRedis(),
            clock=self.clock,
        )
        key = "login:127.0.0.1"
        self.assertTrue(limiter.allow(key, limit=1, window_seconds=60))
        self.assertFalse(limiter.allow(key, limit=1, window_seconds=60))

    def test_disabled_limiter_falls_back_to_memory(self) -> None:
        limiter = FixedWindowRateLimiter(
            url="redis://unused",
            enabled=False,
            clock=self.clock,
        )
        key = "query:user-3"
        self.assertTrue(limiter.allow(key, limit=1, window_seconds=60))
        self.assertFalse(limiter.allow(key, limit=1, window_seconds=60))

class RateLimitMiddlewareTests(unittest.TestCase):
    """验证中间件在超限时返回 429 并带 Retry-After 头。"""

    def test_over_limit_query_returns_429(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from backend.app.core.config import get_settings
        from backend.app.security.rate_limit import RateLimitMiddleware

        settings = get_settings()
        original = (
            settings.rate_limit_query_max,
            settings.rate_limit_query_window_seconds,
        )
        settings.rate_limit_query_max = 1
        settings.rate_limit_query_window_seconds = 60
        try:
            app = FastAPI()

            @app.get("/api/query")
            def ok():
                return {"ok": True}

            app.add_middleware(
                RateLimitMiddleware,
                limiter=FixedWindowRateLimiter(
                    url="redis://unused",
                    enabled=True,
                    clock=self._clock(),
                ),
            )
            client = TestClient(app)
            self.assertEqual(client.get("/api/query").status_code, 200)
            response = client.get("/api/query")
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.headers.get("retry-after"), "60")
        finally:
            (
                settings.rate_limit_query_max,
                settings.rate_limit_query_window_seconds,
            ) = original

    @staticmethod
    def _clock():
        return _FakeClock()


if __name__ == "__main__":
    unittest.main()
