"""无外部依赖且标签有界的本地 Prometheus 指标注册表。"""
from __future__ import annotations

import math
import threading
from collections import Counter
from typing import Any


_LOCK = threading.RLock()
_REQUESTS: Counter[tuple[str, str, str]] = Counter()
_ERRORS: Counter[tuple[str, str, str]] = Counter()
_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_DURATIONS: dict[str, dict[str, Any]] = {}
_ROUTES: Counter[str] = Counter()
_RETRIEVAL_HITS: Counter[str] = Counter()
_MODEL_CALLS: Counter[str] = Counter()
_MODEL_TOKENS: Counter[tuple[str, str]] = Counter()
_FALLBACKS: Counter[tuple[str, str, str]] = Counter()
_BUDGET_TERMINATIONS: Counter[str] = Counter()

_ROUTE_VALUES = {
    "query_document",
    "agent_query",
    "query_stream",
    "rag",
    "vector",
    "case_multidim",
    "smart_match",
    "metrics",
    "health",
    "auth",
    "sessions",
    "other",
}
_ROUTE_ALIASES = {
    "/api/query": "query_document",
    "/api/agent/query": "agent_query",
    "/api/query/stream": "query_stream",
    "/metrics": "metrics",
    "/api/health": "health",
    "/api/knowledge/status": "health",
}
_METHOD_VALUES = {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD", "OTHER"}
_BACKEND_VALUES = {
    "chroma",
    "json",
    "postgres",
    "redis",
    "milvus",
    "sqlite",
    "memory",
    "local",
    "unknown",
    "other",
}
_MODEL_KIND_VALUES = {"chat", "embedding", "rerank", "intent", "other"}
_REASON_VALUES = {
    "backend_error",
    "connection_error",
    "dimension_mismatch",
    "embedding_unavailable",
    "timeout",
    "redis_unavailable",
    "vector_unavailable",
    "cache_miss",
    "token_budget_exceeded",
    "cost_budget_exceeded",
    "disabled",
    "other",
}


def _normalize_route(value: object) -> str:
    """把路径或路由名归入固定指标枚举。"""
    normalized = str(value or "").strip()
    normalized = _ROUTE_ALIASES.get(normalized, normalized)
    if normalized in _ROUTE_VALUES:
        return normalized
    if normalized.startswith("/api/sessions"):
        return "sessions"
    if normalized.startswith("/api/auth"):
        return "auth"
    if normalized.startswith("/api/vectors"):
        return "vector"
    if normalized.startswith("/api/case-multidim"):
        return "case_multidim"
    if normalized.startswith("/api/smart-case-match"):
        return "smart_match"
    return "other"


def _normalize_method(value: object) -> str:
    """把 HTTP 方法压缩为固定枚举。"""
    method = str(value or "").upper()
    return method if method in _METHOD_VALUES else "OTHER"


def _normalize_status(status_code: object) -> str:
    """把状态码仅保留为固定状态分类。"""
    try:
        status = int(status_code)
    except (TypeError, ValueError):
        return "unknown"
    if 100 <= status <= 599:
        return f"{status // 100}xx"
    return "unknown"


def _normalize_backend(value: object) -> str:
    """把存储或模型后端归入固定枚举。"""
    backend = str(value or "").lower()
    return backend if backend in _BACKEND_VALUES else "other"


def _normalize_kind(value: object) -> str:
    """把模型调用类型归入固定枚举。"""
    kind = str(value or "").lower()
    return kind if kind in _MODEL_KIND_VALUES else "other"


def _normalize_reason(value: object) -> str:
    """把回退或终止原因归入固定枚举。"""
    reason = str(value or "").lower()
    return reason if reason in _REASON_VALUES else "other"


def _safe_count(value: object) -> int:
    """把计数转为非负整数。"""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _counter_lines(
    name: str,
    help_text: str,
    labels: tuple[str, ...],
    values: Counter[Any],
) -> list[str]:
    """渲染固定标签计数器，无样本时仍输出元数据。"""
    lines = [f"# HELP {name} {help_text}", f"# TYPE {name} counter"]
    for key, value in sorted(values.items(), key=lambda item: str(item[0])):
        key_values = key if isinstance(key, tuple) else (key,)
        label_text = ",".join(
            f'{label}="{key_value}"'
            for label, key_value in zip(labels, key_values)
        )
        lines.append(f"{name}{{{label_text}}} {float(value)}")
    return lines


def record_request(
    method: str,
    route: str,
    status_code: int,
    duration: float,
) -> None:
    """记录一次 HTTP 请求的数量、错误和耗时。"""
    method_label = _normalize_method(method)
    route_label = _normalize_route(route)
    status_label = _normalize_status(status_code)
    try:
        duration_value = float(duration)
    except (TypeError, ValueError):
        duration_value = 0.0
    if not math.isfinite(duration_value) or duration_value < 0:
        duration_value = 0.0
    with _LOCK:
        _REQUESTS[(method_label, route_label, status_label)] += 1
        try:
            numeric_status = int(status_code)
        except (TypeError, ValueError):
            numeric_status = 500
        if numeric_status >= 400:
            _ERRORS[(method_label, route_label, status_label)] += 1
        duration_stats = _DURATIONS.setdefault(
            route_label,
            {
                "count": 0,
                "sum": 0.0,
                "buckets": [0 for _ in _DURATION_BUCKETS],
            },
        )
        duration_stats["count"] += 1
        duration_stats["sum"] += duration_value
        for index, boundary in enumerate(_DURATION_BUCKETS):
            if duration_value <= boundary:
                duration_stats["buckets"][index] += 1


def record_route(route: str) -> None:
    """记录一次 Agent 路由选择。"""
    with _LOCK:
        _ROUTES[_normalize_route(route)] += 1


def record_retrieval_hit(backend: str, count: int = 1) -> None:
    """记录检索后端返回的命中数量。"""
    hit_count = _safe_count(count)
    if hit_count == 0:
        return
    with _LOCK:
        _RETRIEVAL_HITS[_normalize_backend(backend)] += hit_count


def record_model_call(
    kind: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """记录模型调用次数以及可选输入输出令牌数。"""
    kind_label = _normalize_kind(kind)
    with _LOCK:
        _MODEL_CALLS[kind_label] += 1
        if input_tokens:
            _MODEL_TOKENS[(kind_label, "input")] += _safe_count(input_tokens)
        if output_tokens:
            _MODEL_TOKENS[(kind_label, "output")] += _safe_count(output_tokens)


def record_fallback(
    kind: str,
    backend: str = "unknown",
    reason: str = "other",
) -> None:
    """记录缓存或向量链路的安全回退。"""
    fallback_kind = "cache" if str(kind).lower() == "cache" else "vector"
    with _LOCK:
        _FALLBACKS[
            (
                fallback_kind,
                _normalize_backend(backend),
                _normalize_reason(reason),
            )
        ] += 1


def record_budget_termination(reason: str) -> None:
    """记录 Agent 预算终止原因。"""
    with _LOCK:
        _BUDGET_TERMINATIONS[_normalize_reason(reason)] += 1


def _render_duration(duration_values: dict[str, dict[str, Any]]) -> list[str]:
    """渲染固定桶边界的请求耗时直方图。"""
    lines = [
        "# HELP app_request_duration_seconds 请求处理耗时",
        "# TYPE app_request_duration_seconds histogram",
    ]
    for route, stats in sorted(duration_values.items()):
        for boundary, count in zip(_DURATION_BUCKETS, stats["buckets"]):
            lines.append(
                "app_request_duration_seconds_bucket"
                f'{{route="{route}",le="{boundary}"}} {count}'
            )
        lines.append(
            "app_request_duration_seconds_bucket"
            f'{{route="{route}",le="+Inf"}} {stats["count"]}'
        )
        lines.append(
            "app_request_duration_seconds_sum"
            f'{{route="{route}"}} {stats["sum"]}'
        )
        lines.append(
            "app_request_duration_seconds_count"
            f'{{route="{route}"}} {stats["count"]}'
        )
    return lines


def _render_fallbacks(
    fallback_values: Counter[tuple[str, str, str]],
) -> tuple[list[str], list[str]]:
    """把统一回退计数拆分为缓存与向量两个固定指标。"""
    cache_values: Counter[tuple[str, str]] = Counter()
    vector_values: Counter[tuple[str, str]] = Counter()
    for (kind, backend, reason), value in fallback_values.items():
        if kind == "cache":
            cache_values[(backend, reason)] = value
        else:
            vector_values[(backend, reason)] = value
    return (
        _counter_lines(
            "cache_fallbacks_total",
            "缓存回退次数",
            ("backend", "reason"),
            cache_values,
        ),
        _counter_lines(
            "vector_fallbacks_total",
            "向量链路回退次数",
            ("backend", "reason"),
            vector_values,
        ),
    )


def render_prometheus() -> str:
    """以 Prometheus 文本格式导出当前本地指标快照。"""
    with _LOCK:
        requests = Counter(_REQUESTS)
        errors = Counter(_ERRORS)
        routes = Counter(_ROUTES)
        retrieval_hits = Counter(_RETRIEVAL_HITS)
        model_calls = Counter(_MODEL_CALLS)
        model_tokens = Counter(_MODEL_TOKENS)
        duration_values = {
            route: {
                "count": stats["count"],
                "sum": stats["sum"],
                "buckets": list(stats["buckets"]),
            }
            for route, stats in _DURATIONS.items()
        }
        fallback_values = Counter(_FALLBACKS)
        budget_terminations = Counter(_BUDGET_TERMINATIONS)
    lines: list[str] = []
    lines.extend(
        _counter_lines(
            "app_requests_total",
            "HTTP 请求总数",
            ("method", "route", "status"),
            requests,
        )
    )
    lines.extend(
        _counter_lines(
            "app_request_errors_total",
            "HTTP 错误请求总数",
            ("method", "route", "status"),
            errors,
        )
    )
    lines.extend(_render_duration(duration_values))
    lines.extend(
        _counter_lines(
            "agent_routes_total",
            "Agent 路由选择总数",
            ("route",),
            routes,
        )
    )
    lines.extend(
        _counter_lines(
            "retrieval_hits_total",
            "检索命中总数",
            ("backend",),
            retrieval_hits,
        )
    )
    lines.extend(
        _counter_lines(
            "model_calls_total",
            "模型调用总数",
            ("kind",),
            model_calls,
        )
    )
    lines.extend(
        _counter_lines(
            "model_tokens_total",
            "模型令牌总数",
            ("kind", "type"),
            model_tokens,
        )
    )
    cache_lines, vector_lines = _render_fallbacks(fallback_values)
    lines.extend(cache_lines)
    lines.extend(vector_lines)
    lines.extend(
        _counter_lines(
            "agent_budget_terminations_total",
            "Agent 预算终止总数",
            ("reason",),
            budget_terminations,
        )
    )
    return "\n".join(lines) + "\n"


def reset_metrics() -> None:
    """清空本地指标注册表，仅供应用启动和隔离测试使用。"""
    with _LOCK:
        _REQUESTS.clear()
        _ERRORS.clear()
        _DURATIONS.clear()
        _ROUTES.clear()
        _RETRIEVAL_HITS.clear()
        _MODEL_CALLS.clear()
        _MODEL_TOKENS.clear()
        _FALLBACKS.clear()
        _BUDGET_TERMINATIONS.clear()
