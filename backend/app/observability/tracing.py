"""延迟初始化且失败安全的 OpenTelemetry 追踪门面。"""
from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from backend.app.observability.context import current_context, is_valid_trace_id

_LOCK = threading.RLock()
_INITIALIZED = False
_PROVIDER: Any = None
_TRACER: Any = None
_SAMPLE_RATE = 1.0

_SPAN_NAMES = {
    "http.request",
    "auth.verify",
    "session.load",
    "memory.load",
    "intent.route",
    "document.retrieve",
    "case.multidim",
    "case.smart_match",
    "embedding.call",
    "rerank.call",
    "chat.call",
    "postgres.query",
    "redis.command",
    "vector.query",
    "sse.stream",
    "app.operation",
}
_METHOD_VALUES = {
    "GET",
    "POST",
    "PUT",
    "PATCH",
    "DELETE",
    "OPTIONS",
    "HEAD",
    "OTHER",
}
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
_BACKEND_VALUES = {
    "chroma",
    "postgres",
    "redis",
    "milvus",
    "sqlite",
    "memory",
    "local",
    "unknown",
    "other",
}
_STATUS_VALUES = {
    "unset",
    "ok",
    "error",
    "1xx",
    "2xx",
    "3xx",
    "4xx",
    "5xx",
    "unknown",
}
_KIND_VALUES = {"chat", "embedding", "rerank", "intent", "other"}
_MODEL_VALUES = {
    "openai",
    "dashscope",
    "deepseek",
    "qwen",
    "qwen-plus",
    "qwen-max",
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "local",
    "unknown",
    "other",
}
_EXCEPTION_TYPE_VALUES = {
    "BaseException",
    "Exception",
    "ValueError",
    "TypeError",
    "RuntimeError",
    "KeyError",
    "LookupError",
    "ArithmeticError",
    "ConnectionError",
    "TimeoutError",
    "OSError",
    "PermissionError",
    "FileNotFoundError",
    "HTTPException",
    "ClientDisconnect",
}
_NUMERIC_ATTRIBUTES = {
    "count",
    "input_tokens",
    "output_tokens",
    "token_count",
}
_BOOLEAN_ATTRIBUTES = {"success"}


class _NullSpan:
    """外部追踪不可用时保持调用方控制流不变的空 span。"""

    def set_attribute(self, _key: str, _value: object) -> None:
        """安全忽略 span 属性。"""

    def set_status(self, _status: object) -> None:
        """安全忽略 span 状态。"""


class _SafeSpan:
    """仅允许有界安全属性进入 OTel span。"""

    def __init__(self, span: Any) -> None:
        self._span = span

    def set_attribute(self, key: str, value: object) -> None:
        """过滤未登记字段和非安全值。"""
        normalized_key = str(key)
        normalized_value = _safe_attribute_value(normalized_key, value)
        if normalized_value is None:
            return
        try:
            self._span.set_attribute(normalized_key, normalized_value)
        except Exception:
            return

    def set_status(self, status: object) -> None:
        """只传递固定状态码，重建并丢弃任意描述。"""
        try:
            from opentelemetry.trace import Status, StatusCode

            status_code = getattr(status, "status_code", StatusCode.UNSET)
            if status_code not in (
                StatusCode.UNSET,
                StatusCode.OK,
                StatusCode.ERROR,
            ):
                return
            self._span.set_status(Status(status_code))
        except Exception:
            return


def _normalize_method(value: object) -> str:
    """把 HTTP 方法限制为固定枚举。"""
    method = str(value or "").strip().upper()
    return method if method in _METHOD_VALUES else "OTHER"


def _normalize_route(value: object) -> str:
    """把路由限制为固定低基数枚举。"""
    route = str(value or "").strip()
    route = _ROUTE_ALIASES.get(route, route)
    return route if route in _ROUTE_VALUES else "other"


def _normalize_backend(value: object) -> str:
    """把存储或模型后端限制为固定枚举。"""
    backend = str(value or "").strip().lower()
    return backend if backend in _BACKEND_VALUES else "other"


def _normalize_status(value: object) -> str:
    """把状态文本或数字限制为固定状态分类。"""
    status = str(value or "").strip().lower()
    if status in _STATUS_VALUES:
        return status
    try:
        numeric_status = int(status)
    except (TypeError, ValueError):
        return "unknown"
    if 100 <= numeric_status <= 599:
        return f"{numeric_status // 100}xx"
    return "unknown"


def _normalize_kind(value: object) -> str:
    """把模型调用类型限制为固定枚举。"""
    kind = str(value or "").strip().lower()
    return kind if kind in _KIND_VALUES else "other"


def _normalize_stage(value: object) -> str:
    """把阶段名称限制为已登记 span 名称。"""
    stage = str(value or "").strip()
    return stage if stage in _SPAN_NAMES else "app.operation"


def _normalize_exception_type(value: object) -> str:
    """异常属性只允许固定类型名。"""
    exception_type = str(value or "").strip()
    return (
        exception_type
        if exception_type in _EXCEPTION_TYPE_VALUES
        else "Exception"
    )


def _normalize_model(value: object) -> str:
    """模型属性只允许固定公开枚举。"""
    model = str(value or "").strip().lower()
    return model if model in _MODEL_VALUES else "other"


def _safe_attribute_value(key: str, value: object) -> object | None:
    """把 span 属性限制为固定类型和固定枚举。"""
    if key in _BOOLEAN_ATTRIBUTES:
        return bool(value)
    if key in _NUMERIC_ATTRIBUTES:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return None
        return max(0, numeric)
    if key == "status_code":
        try:
            status_code = int(value)
        except (TypeError, ValueError):
            return None
        return status_code if 100 <= status_code <= 599 else None
    normalizers = {
        "backend": _normalize_backend,
        "exception_type": _normalize_exception_type,
        "kind": _normalize_kind,
        "method": _normalize_method,
        "model": _normalize_model,
        "route": _normalize_route,
        "stage": _normalize_stage,
        "status": _normalize_status,
    }
    normalizer = normalizers.get(key)
    return normalizer(value) if normalizer is not None else None


def _safe_attributes(attributes: dict[str, object]) -> dict[str, object]:
    """批量过滤 span 属性，避免正文或密钥进入追踪后端。"""
    safe: dict[str, object] = {}
    for key, value in attributes.items():
        normalized = _safe_attribute_value(str(key), value)
        if normalized is not None:
            safe[str(key)] = normalized
    return safe


def _bounded_sample_rate(settings: Any) -> float:
    """把采样率限制在 0 到 1，并在生产环境强制不超过 0.1。"""
    try:
        rate = float(getattr(settings, "tracing_sample_rate", 1.0))
    except (TypeError, ValueError):
        rate = 1.0
    rate = min(1.0, max(0.0, rate))
    environment = (
        os.getenv("APP_ENV")
        or os.getenv("ENVIRONMENT")
        or os.getenv("DEPLOYMENT_ENV")
        or ""
    ).strip().lower()
    if environment in {"prod", "production"}:
        rate = min(rate, 0.1)
    return rate


def _trace_is_sampled(trace_id: str) -> bool:
    """按追踪 ID 稳定计算本地采样结果。"""
    if _SAMPLE_RATE >= 1.0:
        return True
    if _SAMPLE_RATE <= 0.0 or not is_valid_trace_id(trace_id):
        return False
    return (int(trace_id, 16) / float(1 << 128)) < _SAMPLE_RATE


def configure_tracing(settings: Any = None) -> None:
    """首次调用时延迟创建 OTel Provider；任何异常均安静降级。"""
    global _INITIALIZED, _PROVIDER, _TRACER, _SAMPLE_RATE
    with _LOCK:
        if _INITIALIZED:
            return
        _INITIALIZED = True
        if settings is None or not bool(getattr(settings, "tracing_enabled", True)):
            return
        _SAMPLE_RATE = _bounded_sample_rate(settings)
        try:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.sampling import ParentBasedTraceIdRatio

            provider = TracerProvider(
                resource=Resource.create(
                    {
                        "service.name": str(
                            getattr(settings, "otel_service_name", "risk-master")
                        )
                    }
                ),
                sampler=ParentBasedTraceIdRatio(_SAMPLE_RATE),
            )
            endpoint = str(getattr(settings, "otlp_endpoint", "") or "").strip()
            if endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                        OTLPSpanExporter,
                    )
                    from opentelemetry.sdk.trace.export import BatchSpanProcessor

                    provider.add_span_processor(
                        BatchSpanProcessor(
                            OTLPSpanExporter(endpoint=endpoint),
                        )
                    )
                except Exception:
                    pass
            _PROVIDER = provider
            _TRACER = provider.get_tracer("backend.app.observability")
        except Exception:
            _PROVIDER = None
            _TRACER = None


def _parent_context() -> Any:
    """把当前请求 trace ID 转换为 OTel 父上下文。"""
    from opentelemetry import trace

    request_context = current_context()
    if not is_valid_trace_id(request_context.trace_id):
        return None
    trace_flags = (
        trace.TraceFlags.SAMPLED
        if _trace_is_sampled(request_context.trace_id)
        else trace.TraceFlags(0)
    )
    span_context = trace.SpanContext(
        trace_id=int(request_context.trace_id, 16),
        span_id=1,
        is_remote=True,
        trace_flags=trace_flags,
    )
    return trace.set_span_in_context(trace.NonRecordingSpan(span_context))


@contextmanager
def start_span(name: str, **attributes: object) -> Iterator[_SafeSpan | _NullSpan]:
    """开启安全 span；未初始化或后端异常时返回空实现。"""
    normalized_name = name if name in _SPAN_NAMES else "app.operation"
    span: _SafeSpan | _NullSpan = _NullSpan()
    manager: Any = None
    try:
        if _TRACER is not None:
            manager = _TRACER.start_as_current_span(
                normalized_name,
                context=_parent_context(),
                attributes=_safe_attributes(attributes),
            )
            span = _SafeSpan(manager.__enter__())
    except Exception:
        manager = None
        span = _NullSpan()
    try:
        yield span
    except Exception as exc:
        try:
            from opentelemetry.trace import Status, StatusCode

            span.set_attribute("exception_type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR))
        except Exception:
            pass
        raise
    finally:
        if manager is not None:
            try:
                manager.__exit__(None, None, None)
            except Exception:
                pass


def shutdown_tracing() -> None:
    """关闭 OTel Provider，重复调用保持幂等。"""
    global _INITIALIZED, _PROVIDER, _TRACER, _SAMPLE_RATE
    with _LOCK:
        provider = _PROVIDER
        _PROVIDER = None
        _TRACER = None
        _INITIALIZED = False
        _SAMPLE_RATE = 1.0
    if provider is not None:
        try:
            provider.shutdown()
        except Exception:
            pass
