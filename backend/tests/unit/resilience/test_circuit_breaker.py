"""熔断器状态机单元测试。"""
from __future__ import annotations

import time
import unittest

from backend.app.core.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    get_circuit_breaker,
    reset_circuit_breakers,
)


class CircuitBreakerStateTests(unittest.TestCase):
    """验证 closed/open/half_open 三态转换与冷却窗口。"""

    def test_closed_breaker_allows_requests(self) -> None:
        breaker = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=30.0)
        self.assertTrue(breaker.allow())
        self.assertEqual(breaker.snapshot()["state"], "closed")

    def test_continuous_failures_open_breaker(self) -> None:
        breaker = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=30.0)
        for _ in range(3):
            breaker.record_failure()
        self.assertEqual(breaker.snapshot()["state"], "open")
        self.assertFalse(breaker.allow())

    def test_open_breaker_denies_within_cooldown(self) -> None:
        breaker = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=60.0)
        breaker.record_failure()
        self.assertFalse(breaker.allow())
        # 冷却窗口未结束前持续拒绝。
        self.assertFalse(breaker.allow())
        self.assertFalse(breaker.allow())

    def test_cooldown_transitions_to_half_open_and_recovers(self) -> None:
        breaker = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=0.01)
        breaker.record_failure()
        self.assertEqual(breaker.snapshot()["state"], "open")
        time.sleep(0.02)
        self.assertTrue(breaker.allow())
        self.assertEqual(breaker.snapshot()["state"], "half_open")
        breaker.record_success()
        self.assertEqual(breaker.snapshot()["state"], "closed")
        self.assertTrue(breaker.allow())

    def test_half_open_failure_reopens_breaker(self) -> None:
        breaker = CircuitBreaker(
            "test",
            failure_threshold=3,
            cooldown_seconds=0.01,
            recovery_trials=1,
        )
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        time.sleep(0.02)
        self.assertTrue(breaker.allow())
        breaker.record_failure()
        self.assertEqual(breaker.snapshot()["state"], "open")

    def test_half_open_limits_recovery_trials(self) -> None:
        breaker = CircuitBreaker(
            "test",
            failure_threshold=1,
            cooldown_seconds=0.01,
            recovery_trials=1,
        )
        breaker.record_failure()
        time.sleep(0.02)
        self.assertTrue(breaker.allow())
        # 半开状态只放行一次探测，后续请求被拒绝。
        self.assertFalse(breaker.allow())

    def test_success_resets_failure_count(self) -> None:
        breaker = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=30.0)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        self.assertEqual(breaker.snapshot()["failure_count"], 0)
        self.assertEqual(breaker.snapshot()["state"], "closed")


class CircuitBreakerRegistryTests(unittest.TestCase):
    """验证进程级熔断器注册表的单例与重置行为。"""

    def setUp(self) -> None:
        reset_circuit_breakers()

    def test_registry_returns_same_instance_for_name(self) -> None:
        first = get_circuit_breaker("chat", failure_threshold=2, cooldown_seconds=5.0)
        second = get_circuit_breaker("chat", failure_threshold=9, cooldown_seconds=9.0)
        self.assertIs(first, second)
        # 已存在的实例保持首次参数，避免重复请求覆盖配置。
        self.assertEqual(first.failure_threshold, 2)
        self.assertEqual(first.cooldown_seconds, 5.0)

    def test_reset_clears_registry(self) -> None:
        get_circuit_breaker("chat")
        reset_circuit_breakers()
        breaker = get_circuit_breaker("chat", failure_threshold=1, cooldown_seconds=1.0)
        self.assertEqual(breaker.failure_threshold, 1)


class CircuitOpenErrorTests(unittest.TestCase):
    """验证熔断异常可作为降级信号被上层捕获。"""

    def test_error_is_runtime_error_subclass(self) -> None:
        error = CircuitOpenError("服务 chat 熔断打开")
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("chat", str(error))
