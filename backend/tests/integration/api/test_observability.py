"""验证 FastAPI 可观测性中间件、响应头与本地指标端点。"""
from __future__ import annotations

import asyncio
import re
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import StreamingResponse

from backend.app.api.routes import observability
from backend.app.main import create_app
from backend.app.observability.middleware import ObservabilityMiddleware


TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
REQUIRED_METRICS = (
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


def _metric_total(metrics_text: str, metric_name: str) -> float:
    """汇总指定指标的全部样本值，元数据行不计入。"""
    pattern = re.compile(
        rf"^{re.escape(metric_name)}(?:\{{[^\n]*\}})? ([0-9.eE+-]+)$",
        re.MULTILINE,
    )
    return sum(float(value) for value in pattern.findall(metrics_text))


class _FakeChatService:
    """提供不触碰检索和模型链路的稳定查询响应。"""

    def query_document(self, payload):
        return {"source": "document", "question": payload.question}

    def query_agent(self, payload):
        return {"source": "agent", "question": payload.question}

    async def stream(self, payload, _is_disconnected):
        yield {"type": "delta", "content": "统一流响应"}


class _FakeDocumentStore:
    """为健康接口提供固定且不打开真实向量库的状态。"""

    def collection_info(self) -> dict:
        return {"count": 0, "dimension": 0, "source": "test"}


class _AvailableClient:
    """模拟无需联网即可报告可用的模型客户端。"""

    def is_available(self) -> bool:
        return True


class _FakeServices:
    """只暴露认证与查询路由所需门面。"""

    def __init__(self) -> None:
        self.chat_service = _FakeChatService()
        self.auth_service = self
        self.document_store = _FakeDocumentStore()
        self.embedding_client = _AvailableClient()
        self.llm_client = _AvailableClient()
        self.close_count = 0

    def get_current_user(self, token: str) -> dict:
        return {"user_id": "observability-user", "role": "user", "token": token}

    def chat_service_for(self, user_id: str) -> _FakeChatService:
        assert user_id == "observability-user"
        return self.chat_service

    def close(self) -> None:
        self.close_count += 1


@pytest.fixture
def client() -> TestClient:
    """复用应用工厂但不启动真实业务服务。"""
    app = create_app()
    app.state.services = _FakeServices()
    test_client = TestClient(
        app,
        headers={"Authorization": "Bearer observability-token"},
    )
    try:
        yield test_client
    finally:
        test_client.close()


def test_stream_response_returns_request_and_trace_headers(client: TestClient) -> None:
    """流式响应必须带关联响应头且保留原 SSE 事件内容。"""
    response = client.post("/api/query/stream", json={"question": "test"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    assert TRACE_ID_PATTERN.fullmatch(response.headers["X-Trace-ID"])
    assert response.text.endswith('data: {"type": "delta", "content": "统一流响应"}\n\n')


@pytest.mark.parametrize(
    ("path", "payload", "source"),
    (
        ("/api/query", {"question": "test"}, "document"),
        ("/api/agent/query", {"question": "test"}, "agent"),
    ),
)
def test_compatibility_query_routes_keep_body_and_add_headers(
    client: TestClient,
    path: str,
    payload: dict,
    source: str,
) -> None:
    """两个旧查询接口响应体保持兼容，同时返回关联响应头。"""
    response = client.post(path, json=payload)

    assert response.status_code == 200
    assert response.json()["source"] == source
    assert response.headers["X-Request-ID"]
    assert TRACE_ID_PATTERN.fullmatch(response.headers["X-Trace-ID"])


def test_client_request_id_and_valid_trace_id_are_preserved(client: TestClient) -> None:
    """合法入站关联 ID 应稳定写入响应头。"""
    trace_id = "3" * 32
    response = client.get(
        "/api/health",
        headers={
            "X-Request-ID": "client-request-01",
            "X-Trace-ID": trace_id,
        },
    )

    if response.status_code == 200:
        assert response.headers["X-Request-ID"] == "client-request-01"
        assert response.headers["X-Trace-ID"] == trace_id
    else:
        assert response.headers["X-Request-ID"] == "client-request-01"
        assert response.headers["X-Trace-ID"] == trace_id


def test_invalid_trace_id_and_uppercase_id_are_replaced(client: TestClient) -> None:
    """非法或非小写入站追踪 ID 必须替换为新 ID。"""
    response = client.post(
        "/api/query",
        json={"question": "test"},
        headers={"X-Trace-ID": "4" * 31, "X-Request-ID": "bad trace request"},
    )

    assert response.status_code == 200
    assert TRACE_ID_PATTERN.fullmatch(response.headers["X-Trace-ID"])
    assert response.headers["X-Trace-ID"] != "4" * 31
    assert response.headers["X-Request-ID"] != "bad trace request"


def test_traceparent_header_controls_response_trace_id(client: TestClient) -> None:
    """合法 W3C traceparent 应提供同一请求的追踪 ID。"""
    trace_id = "5" * 32
    traceparent = f"00-{trace_id}-{('6' * 16)}-01"
    response = client.post(
        "/api/query/stream",
        json={"question": "test"},
        headers={"traceparent": traceparent},
    )

    assert response.status_code == 200
    assert response.headers["X-Trace-ID"] == trace_id


def test_metrics_endpoint_returns_prometheus_without_constructing_services() -> None:
    """指标端点不得依赖业务服务，并应包含全部固定指标名称。"""
    app = create_app()
    local_client = TestClient(app)
    response = local_client.get("/metrics")
    local_client.close()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.headers["X-Request-ID"]
    assert TRACE_ID_PATTERN.fullmatch(response.headers["X-Trace-ID"])
    for metric_name in REQUIRED_METRICS:
        assert metric_name in response.text


def test_request_metrics_record_bounded_labels(client: TestClient) -> None:
    """完成查询后本地指标只包含固定路由标签，不包含问题正文。"""
    from backend.app.observability.metrics import reset_metrics

    reset_metrics()
    response = client.post(
        "/api/query/stream",
        json={"question": "private-question-should-not-be-a-label"},
    )
    metrics_response = client.get("/metrics")

    assert response.status_code == 200
    metrics_text = metrics_response.text
    assert 'route="query_stream"' in metrics_text
    assert "private-question-should-not-be-a-label" not in metrics_text
    assert "observability-token" not in metrics_text


def test_observability_failure_does_not_change_business_response(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """追踪初始化异常时仍应返回成功响应，并确保故障发生在实际调用路径。"""
    from backend.app.observability import middleware

    configure_calls: list[str] = []

    def fail_configure(_settings=None):
        configure_calls.append("called")
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(middleware, "configure_tracing", fail_configure)
    response = client.post("/api/query", json={"question": "test"})

    assert configure_calls == ["called"]
    assert response.status_code == 200
    assert response.json()["source"] == "document"
    assert response.headers["X-Request-ID"]
    assert response.headers["X-Trace-ID"]


class _RecordingOtSpan:
    """记录中间件最终提交给追踪后端的属性与状态。"""

    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.status: object | None = None

    def set_attribute(self, key: str, value: object) -> None:
        """保存最终属性。"""
        self.attributes[key] = value

    def set_status(self, status: object) -> None:
        """保存最终状态。"""
        self.status = status


class _RecordingSpanContext:
    """提供 OTel span 上下文管理器协议。"""

    def __init__(self, span: _RecordingOtSpan) -> None:
        self.span = span

    def __enter__(self) -> _RecordingOtSpan:
        """进入记录 span。"""
        return self.span

    def __exit__(self, *_args: object) -> None:
        """退出记录 span。"""


class _RecordingTracer:
    """捕获 tracing 门面实际传给追踪后端的有界属性。"""

    def __init__(self) -> None:
        self.spans: list[_RecordingOtSpan] = []

    def start_as_current_span(
        self,
        _name: str,
        *,
        context: object,
        attributes: dict[str, object],
    ) -> _RecordingSpanContext:
        """创建可检查的记录 span。"""
        del context
        span = _RecordingOtSpan()
        span.attributes.update(attributes)
        self.spans.append(span)
        return _RecordingSpanContext(span)


def test_http_span_normalizes_malicious_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP method 自由文本不得进入 exporter 可观察的 Span 属性。"""
    from backend.app.observability import tracing

    tracer = _RecordingTracer()
    tracing.shutdown_tracing()
    monkeypatch.setattr(tracing, "_TRACER", tracer)
    monkeypatch.setattr(tracing, "_INITIALIZED", True)

    async def probe_app(scope, receive, send) -> None:
        del scope, receive
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"ok",
                "more_body": False,
            }
        )

    client = TestClient(ObservabilityMiddleware(probe_app))
    response = client.request("sk-live-api-key-123", "/probe")
    client.close()

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    assert tracer.spans
    assert tracer.spans[0].attributes["method"] == "OTHER"
    assert "sk-live-api-key-123" not in str(tracer.spans[0].attributes)


def _create_lifecycle_probe_app() -> FastAPI:
    """创建覆盖慢流、流中异常和未处理异常的离线探针应用。"""
    app = FastAPI()

    @app.get("/probe/slow-stream")
    async def slow_stream() -> StreamingResponse:
        async def body():
            yield b"first"
            await asyncio.sleep(0.35)
            yield b"second"

        return StreamingResponse(body(), media_type="text/plain")

    @app.get("/probe/stream-error")
    async def stream_error() -> StreamingResponse:
        async def body():
            yield b"first"
            await asyncio.sleep(0.01)
            raise RuntimeError("Authorization: Bearer stream-error-secret")

        return StreamingResponse(body(), media_type="text/plain")

    @app.get("/probe/unhandled-error")
    async def unhandled_error() -> None:
        raise RuntimeError("Authorization: Bearer unhandled-error-secret")

    app.include_router(observability.router)
    app.add_middleware(ObservabilityMiddleware)
    return app


def test_slow_stream_records_full_response_duration() -> None:
    """慢速流应在最后一条响应体发送后才记录完整耗时。"""
    from backend.app.observability.metrics import reset_metrics

    reset_metrics()
    started_at = time.perf_counter()
    with TestClient(
        _create_lifecycle_probe_app(),
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/probe/slow-stream")
        elapsed = time.perf_counter() - started_at
        metrics_text = client.get("/metrics").text

    assert response.status_code == 200
    assert response.text == "firstsecond"
    assert response.headers["X-Request-ID"]
    assert response.headers["X-Trace-ID"]
    assert elapsed >= 0.30
    assert _metric_total(
        metrics_text,
        "app_request_duration_seconds_sum",
    ) >= 0.30


def test_stream_error_propagates_after_response_started_and_records_once() -> None:
    """流体异常应向 ASGI 服务器传播，并只计入一次错误指标。"""
    from backend.app.observability.metrics import reset_metrics

    reset_metrics()
    with TestClient(
        _create_lifecycle_probe_app(),
    ) as client:
        with pytest.raises(Exception):
            client.get("/probe/stream-error")
        metrics_text = client.get("/metrics").text

    assert _metric_total(metrics_text, "app_request_errors_total") == 1
    assert "stream-error-secret" not in metrics_text


def test_unhandled_exception_returns_standard_500_with_headers() -> None:
    """未处理异常应返回标准 500 文本、关联头和一次错误指标。"""
    from backend.app.observability.metrics import reset_metrics

    reset_metrics()
    with TestClient(
        _create_lifecycle_probe_app(),
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/probe/unhandled-error")
        metrics_text = client.get("/metrics").text

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert response.headers["X-Request-ID"]
    assert response.headers["X-Trace-ID"]
    assert _metric_total(metrics_text, "app_request_errors_total") == 1
    assert "unhandled-error-secret" not in response.text
    assert "unhandled-error-secret" not in metrics_text


def test_http_requests_do_not_increment_agent_routes_total(
    client: TestClient,
) -> None:
    """HTTP 请求和指标抓取不得伪造 Agent 路由决策。"""
    from backend.app.observability.metrics import reset_metrics

    reset_metrics()
    response = client.post("/api/query", json={"question": "test"})
    metrics_text = client.get("/metrics").text

    assert response.status_code == 200
    assert "# HELP agent_routes_total Agent 路由选择总数" in metrics_text
    assert _metric_total(metrics_text, "agent_routes_total") == 0
