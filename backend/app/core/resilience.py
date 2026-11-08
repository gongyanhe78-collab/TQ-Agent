"""外部服务熔断与降级基础设施。"""
from __future__ import annotations

import threading
import time


class CircuitOpenError(RuntimeError):
    """熔断打开时抛出的快速失败异常，由上层按降级策略承接。"""


class CircuitBreaker:
    """为单个外部服务维护 closed/open/half_open 三态熔断器。

    连续失败达到阈值后进入 open 状态并拒绝调用；冷却窗口结束后
    进入 half_open 状态放行有限探测请求，成功后复位，失败则重新打开。
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        recovery_trials: int = 1,
    ) -> None:
        self.name = str(name)
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.recovery_trials = max(1, int(recovery_trials))
        self._state = "closed"
        self._failure_count = 0
        self._opened_at = 0.0
        self._recovery_count = 0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """判断本次调用是否放行，并在半开状态消耗一次探测配额。"""
        with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                if time.monotonic() - self._opened_at >= self.cooldown_seconds:
                    self._state = "half_open"
                    # 进入半开状态的首个请求即消耗一次探测配额，避免多放行一个请求。
                    self._recovery_count = 1
                    return True
                return False
            if self._recovery_count < self.recovery_trials:
                self._recovery_count += 1
                return True
            return False

    def record_success(self) -> None:
        """成功调用后复位为 closed 并清空失败计数。"""
        with self._lock:
            self._state = "closed"
            self._failure_count = 0
            self._recovery_count = 0
            self._opened_at = 0.0

    def record_failure(self) -> None:
        """失败调用累计；半开探测失败或达到阈值时打开熔断。"""
        with self._lock:
            if self._state == "half_open":
                self._open()
                return
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._open()

    def _open(self) -> None:
        """进入 open 状态并记录打开时刻。"""
        self._state = "open"
        self._opened_at = time.monotonic()
        self._recovery_count = 0

    def snapshot(self) -> dict[str, object]:
        """返回观测与测试使用的状态快照。"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state,
                "failure_count": self._failure_count,
                "opened_at": self._opened_at,
            }


_BREAKERS: dict[str, CircuitBreaker] = {}
_BREAKERS_LOCK = threading.Lock()


def get_circuit_breaker(
    name: str,
    *,
    failure_threshold: int = 5,
    cooldown_seconds: float = 30.0,
    recovery_trials: int = 1,
) -> CircuitBreaker:
    """返回指定服务的进程级熔断器单例，首次访问时按参数创建。"""
    with _BREAKERS_LOCK:
        breaker = _BREAKERS.get(name)
        if breaker is None:
            breaker = CircuitBreaker(
                name,
                failure_threshold=failure_threshold,
                cooldown_seconds=cooldown_seconds,
                recovery_trials=recovery_trials,
            )
            _BREAKERS[name] = breaker
        return breaker


def reset_circuit_breakers() -> None:
    """清空熔断器注册表，仅用于测试隔离。"""
    with _BREAKERS_LOCK:
        _BREAKERS.clear()

