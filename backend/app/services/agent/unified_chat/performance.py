"""统一聊天请求的轻量性能计时。"""
from __future__ import annotations

from time import perf_counter


class PerformanceTrace:
    """记录统一入口各阶段耗时，不参与任何业务判断。"""

    def __init__(self) -> None:
        self._request_started = perf_counter()
        self._stage_started: dict[str, float] = {}
        self._durations: dict[str, float] = {}

    def start(self, name: str) -> None:
        """开始记录指定阶段。"""
        self._stage_started[name] = perf_counter()

    def stop(self, name: str) -> float:
        """结束指定阶段并返回毫秒耗时。"""
        started = self._stage_started.pop(name, None)
        if started is None:
            return float(self._durations.get(name, 0.0))
        elapsed = (perf_counter() - started) * 1000
        self._durations[name] = round(elapsed, 2)
        return elapsed

    def set(self, name: str, milliseconds: float) -> None:
        """写入由其他线程测得的阶段耗时。"""
        self._durations[name] = round(max(0.0, float(milliseconds)), 2)

    def snapshot(self) -> dict[str, float]:
        """返回适合写入 SSE 和运行审计的耗时快照。"""
        return {
            **self._durations,
            "total_ms": round((perf_counter() - self._request_started) * 1000, 2),
        }
