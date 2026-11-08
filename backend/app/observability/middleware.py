"""纯 ASGI 请求关联、追踪与本地指标中间件。"""
from __future__ import annotations

import time
from typing import Any

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.app.core.config import get_settings
from backend.app.observability.context import (
    bind_request_context,
    is_valid_request_id,
    is_valid_trace_id,
    new_request_id,
    new_trace_id,
    trace_id_from_traceparent,
)
from backend.app.observability.logging import configure_logging
from backend.app.observability.metrics import record_request
from backend.app.observability.tracing import (
    configure_tracing,
    start_span,
)

_OBSERVABILITY_HEADERS = {b"x-request-id", b"x-trace-id"}


def _route_label(scope: Scope) -> str:
    """从路由作用域生成固定低基数标签，未知路径统一归为 other。"""
    route = scope.get("route")
    path = str(
        getattr(route, "path", "") or scope.get("path", "")
    )
    if path == "/api/query":
        return "query_document"
    if path == "/api/agent/query":
        return "agent_query"
    if path == "/api/query/stream":
        return "query_stream"
    if path == "/metrics":
        return "metrics"
    if path in {"/api/health", "/api/knowledge/status"}:
        return "health"
    if path.startswith("/api/auth"):
        return "auth"
    if path.startswith("/api/sessions"):
        return "sessions"
    if path.startswith("/api/vectors"):
        return "vector"
    if path.startswith("/api/case-multidim"):
        return "case_multidim"
    if path.startswith("/api/smart-case-match"):
        return "smart_match"
    return "other"


def _with_observability_headers(
    message: Message,
    request_id: str,
    trace_id: str,
) -> Message:
    """复制响应起始消息并替换关联头，保留其他安全响应头。"""
    headers = [
        (key, value)
        for key, value in message.get("headers", [])
        if key.lower() not in _OBSERVABILITY_HEADERS
    ]
    headers.extend(
        (
            (b"x-request-id", request_id.encode("ascii")),
            (b"x-trace-id", trace_id.encode("ascii")),
        )
    )
    return {**message, "headers": headers}


class ObservabilityMiddleware:
    """完整覆盖普通与流式 ASGI 响应生命周期的观测中间件。"""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """绑定请求上下文，直到响应完成或异常时才提交观测结果。"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started_at = time.perf_counter()
        request_headers = Headers(scope=scope)
        traceparent_trace_id = trace_id_from_traceparent(
            request_headers.get("traceparent")
        )
        header_trace_id = request_headers.get("X-Trace-ID", "")
        trace_id = (
            traceparent_trace_id
            or (
                header_trace_id
                if is_valid_trace_id(header_trace_id)
                else new_trace_id()
            )
        )
        header_request_id = request_headers.get("X-Request-ID", "")
        request_id = (
            header_request_id
            if is_valid_request_id(header_request_id)
            else new_request_id()
        )
        metrics_enabled = True

        with bind_request_context(
            request_id=request_id,
            trace_id=trace_id,
        ) as request_context:
            settings: Any = None
            try:
                settings = get_settings()
                metrics_enabled = bool(
                    getattr(settings, "metrics_enabled", True)
                )
            except Exception:
                settings = None

            if settings is not None:
                try:
                    configure_logging(settings)
                except Exception:
                    pass
                try:
                    configure_tracing(settings)
                except Exception:
                    pass

            route_label = _route_label(scope)
            status_code = 500
            finalized = False

            with start_span(
                "http.request",
                method=scope.get("method", "OTHER"),
                route=route_label,
                success=True,
            ) as span:

                def finalize(final_status: int) -> None:
                    """确保一次请求只提交一次 Span 和指标结果。"""
                    nonlocal finalized
                    if finalized:
                        return
                    finalized = True
                    duration = time.perf_counter() - started_at
                    final_route = _route_label(scope)
                    try:
                        span.set_attribute("status_code", final_status)
                        span.set_attribute("status", final_status)
                        span.set_attribute("success", final_status < 500)
                        if final_status >= 500:
                            from opentelemetry.trace import Status, StatusCode

                            span.set_status(Status(StatusCode.ERROR))
                    except Exception:
                        pass
                    if not metrics_enabled:
                        return
                    try:
                        record_request(
                            scope.get("method", "OTHER"),
                            final_route,
                            final_status,
                            duration,
                        )
                    except Exception:
                        pass

                async def wrapped_send(message: Message) -> None:
                    """在响应头和最终响应体发送边界维护观测状态。"""
                    nonlocal response_started, status_code
                    message_type = message.get("type")
                    if message_type == "http.response.start":
                        try:
                            status_code = int(message.get("status", 500))
                        except (TypeError, ValueError):
                            status_code = 500
                        safe_message = _with_observability_headers(
                            message,
                            request_context.request_id,
                            request_context.trace_id,
                        )
                        await send(safe_message)
                        response_started = True
                        return
                    if message_type == "http.response.body":
                        await send(message)
                        if not bool(message.get("more_body", False)):
                            finalize(status_code)
                        return
                    await send(message)

                response_started = False
                try:
                    await self.app(scope, receive, wrapped_send)
                except Exception:
                    response_was_started = response_started
                    if not response_was_started:
                        status_code = 500
                        try:
                            await wrapped_send(
                                {
                                    "type": "http.response.start",
                                    "status": 500,
                                    "headers": [
                                        (
                                            b"content-type",
                                            b"text/plain; charset=utf-8",
                                        ),
                                        (
                                            b"content-length",
                                            b"21",
                                        ),
                                    ],
                                }
                            )
                            await wrapped_send(
                                {
                                    "type": "http.response.body",
                                    "body": b"Internal Server Error",
                                    "more_body": False,
                                }
                            )
                        except BaseException:
                            pass
                    finalize(500)
                    raise
                finally:
                    if not finalized:
                        finalize(status_code)
