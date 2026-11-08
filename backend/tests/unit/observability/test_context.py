"""验证请求上下文、日志关联、固定指标与追踪降级契约。"""
from __future__ import annotations

import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest


TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def test_request_context_is_isolated_and_uses_w3c_trace_id() -> None:
    """绑定期间应生成合法追踪 ID，退出后不得残留请求身份。"""
    from backend.app.observability.context import (
        bind_request_context,
        current_context,
    )

    with bind_request_context(user_id="u1"):
        request_context = current_context()
        assert len(request_context.trace_id) == 32
        assert TRACE_ID_PATTERN.fullmatch(request_context.trace_id)
        assert request_context.trace_id != "0" * 32
        assert request_context.request_id
        assert request_context.user_id == "u1"

    assert current_context().user_id == ""
    assert current_context().request_id == ""
    assert current_context().trace_id == ""


def test_nested_context_binding_preserves_parent_ids() -> None:
    """子阶段只更新显式字段，并应保留同一请求的关联 ID。"""
    from backend.app.observability.context import (
        bind_request_context,
        current_context,
    )

    with bind_request_context(request_id="request-1", trace_id="1" * 32):
        with bind_request_context(run_id="run-1", session_id="session-1"):
            nested = current_context()
            assert nested.request_id == "request-1"
            assert nested.trace_id == "1" * 32
            assert nested.run_id == "run-1"
            assert nested.session_id == "session-1"

        restored = current_context()
        assert restored.request_id == "request-1"
        assert restored.run_id == ""


def test_concurrent_request_contexts_do_not_leak_between_threads() -> None:
    """两个并发请求必须各自看到自己的上下文。"""
    from backend.app.observability.context import (
        bind_request_context,
        current_context,
    )

    barrier = Barrier(2)

    def read_user(user_id: str) -> str:
        with bind_request_context(user_id=user_id, run_id=f"run-{user_id}"):
            barrier.wait()
            observed = current_context()
            barrier.wait()
            return f"{observed.user_id}:{observed.run_id}"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(read_user, ("user-a", "user-b")))

    assert sorted(results) == ["user-a:run-user-a", "user-b:run-user-b"]


@pytest.mark.parametrize(
    "invalid_trace_id",
    (
        "",
        "0" * 32,
        "A" * 32,
        "1" * 31,
        "z" * 32,
    ),
)
def test_invalid_trace_id_is_replaced(invalid_trace_id: str) -> None:
    """非法 W3C 追踪 ID 必须替换为新的合法小写十六进制 ID。"""
    from backend.app.observability.context import (
        bind_request_context,
        current_context,
    )

    with bind_request_context(trace_id=invalid_trace_id):
        trace_id = current_context().trace_id

    assert TRACE_ID_PATTERN.fullmatch(trace_id)
    assert trace_id != "0" * 32


def test_new_ids_follow_first_party_contracts() -> None:
    """新请求 ID 非空且不重复，新追踪 ID 满足 W3C 格式。"""
    from backend.app.observability.context import new_request_id, new_trace_id

    request_ids = {new_request_id() for _ in range(20)}
    trace_ids = {new_trace_id() for _ in range(20)}

    assert len(request_ids) == 20
    assert all(request_id for request_id in request_ids)
    assert len(trace_ids) == 20
    assert all(TRACE_ID_PATTERN.fullmatch(trace_id) for trace_id in trace_ids)
    assert "0" * 32 not in trace_ids


def test_logging_filter_injects_context_and_json_sanitizes_secrets() -> None:
    """日志字段应与上下文关联，JSON 日志保持单行且不泄露密钥。"""
    from backend.app.observability.context import bind_request_context
    from backend.app.observability.logging import (
        ContextLogFilter,
        JsonContextFormatter,
    )

    with bind_request_context(
        request_id="request-log",
        trace_id="2" * 32,
        user_id="user-log",
        run_id="run-log",
    ):
        record = logging.LogRecord(
            name="risk.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="请求失败 Authorization: Bearer top-secret api_key=sk-hidden",
            args=(),
            exc_info=None,
        )
        ContextLogFilter().filter(record)
        output = JsonContextFormatter().format(record)

    payload = json.loads(output)
    assert "\n" not in output
    assert payload["request_id"] == "request-log"
    assert payload["trace_id"] == "2" * 32
    assert payload["user_id"] == "user-log"
    assert payload["run_id"] == "run-log"
    assert "top-secret" not in output
    assert "sk-hidden" not in output


def test_text_error_log_only_keeps_type_stage_and_sanitized_summary() -> None:
    """异常文本日志不得包含堆栈或未脱敏异常正文。"""
    from backend.app.observability.logging import TextContextFormatter

    try:
        raise ValueError(
            "Authorization: Bearer exception-secret "
            "问题正文不应进入日志"
        )
    except ValueError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="risk.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="模型调用失败",
        args=(),
        exc_info=exc_info,
    )
    record.stage = "chat.call"
    output = TextContextFormatter().format(record)

    assert "ValueError" in output
    assert "chat.call" in output
    assert "Traceback" not in output
    assert "exception-secret" not in output
    assert "问题正文" not in output


def test_metrics_render_required_names_and_normalize_labels() -> None:
    """本地文本指标应包含固定名称，且任意文本不能进入标签。"""
    from backend.app.observability.metrics import (
        record_budget_termination,
        record_fallback,
        record_model_call,
        record_request,
        record_retrieval_hit,
        record_route,
        render_prometheus,
        reset_metrics,
    )

    reset_metrics()
    record_request("POST", "/api/query/stream", 200, 0.25)
    record_route("rag")
    record_retrieval_hit("chroma", count=2)
    record_model_call("chat", input_tokens=12, output_tokens=5)
    record_fallback("vector", backend="milvus", reason="connection_error")
    record_fallback("cache", reason="redis_unavailable")
    record_budget_termination("token_budget_exceeded")
    record_route("private-question-content")
    record_fallback("vector", reason="Authorization Bearer hidden-token")

    text = render_prometheus()
    required_names = (
        "app_requests_total",
        "app_request_errors_total",
        "app_request_duration_seconds",
        "agent_routes_total",
        "retrieval_hits_total",
        "model_calls_total",
        "model_tokens_total",
        "cache_fallbacks_total",
        "vector_fallbacks_total",
        "agent_budget_terminations_total",
    )
    for name in required_names:
        assert name in text

    assert 'route="query_stream"' in text
    assert 'route="rag"' in text
    assert 'backend="chroma"' in text
    assert 'reason="connection_error"' in text
    assert "private-question-content" not in text
    assert "hidden-token" not in text


def test_tracing_lazily_degrades_when_export_backend_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """缺少 OTLP exporter 时起点 span 不能抛出异常或改变业务控制流。"""
    from backend.app.observability import tracing

    settings = SimpleNamespace(
        otel_service_name="risk-test",
        otlp_endpoint="http://127.0.0.1:4318",
        tracing_enabled=True,
        tracing_sample_rate=1.0,
    )
    tracing.shutdown_tracing()
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        None,
    )

    tracing.configure_tracing(settings)
    with tracing.start_span("chat.call", model="test-model", success=True) as span:
        span.set_attribute("token_count", 12)

    assert "opentelemetry.exporter.otlp.proto.http.trace_exporter" in sys.modules
    tracing.shutdown_tracing()
    tracing.shutdown_tracing()


def test_safe_span_uses_fixed_enums_and_strips_status_description() -> None:
    """恶意自由文本不得进入 exporter 可见属性和 Status 描述。"""
    from opentelemetry.trace import Status, StatusCode

    from backend.app.observability.tracing import _SafeSpan

    class _RecordingSpan:
        """记录 OTel 包装层最终提交给导出后端的数据。"""

        def __init__(self) -> None:
            self.attributes: dict[str, object] = {}
            self.status: Status | None = None

        def set_attribute(self, key: str, value: object) -> None:
            """保存最终属性。"""
            self.attributes[key] = value

        def set_status(self, status: Status) -> None:
            """保存最终状态。"""
            self.status = status

    recording_span = _RecordingSpan()
    safe_span = _SafeSpan(recording_span)
    safe_span.set_attribute("method", "sk-live-api-key-123")
    safe_span.set_attribute("route", "privatequestiontext")
    safe_span.set_attribute("backend", "privateusername")
    safe_span.set_attribute("status", "englisherrordescription")
    safe_span.set_attribute("kind", "api_key_secret")
    safe_span.set_attribute("stage", "privatequestionsegment")
    safe_span.set_attribute("exception_type", "privateexceptiontext")
    safe_span.set_attribute("model", "private-model-name")
    safe_span.set_attribute("status_code", 9999)
    safe_span.set_attribute("unregistered", "authorization-bearer-secret")
    safe_span.set_status(
        Status(
            StatusCode.ERROR,
            "Authorization: Bearer status-description-secret",
        )
    )

    assert recording_span.attributes == {
        "method": "OTHER",
        "route": "other",
        "backend": "other",
        "status": "unknown",
        "kind": "other",
        "stage": "app.operation",
        "exception_type": "Exception",
        "model": "other",
    }
    assert recording_span.status is not None
    assert recording_span.status.status_code is StatusCode.ERROR
    assert recording_span.status.description is None
    assert "status-description-secret" not in str(recording_span.status)
