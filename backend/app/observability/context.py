"""基于 contextvars 的请求关联上下文。"""
from __future__ import annotations

import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Iterator


_TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_TRACEPARENT_PATTERN = re.compile(
    r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$"
)
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._~:/+-]{1,128}$")
_ZERO_TRACE_ID = "0" * 32


@dataclass(frozen=True, slots=True)
class RequestContext:
    """描述一次请求或其子阶段的可传播关联身份。"""

    request_id: str = ""
    trace_id: str = ""
    user_id: str = ""
    session_id: str = ""
    run_id: str = ""


_EMPTY_CONTEXT = RequestContext()
_REQUEST_CONTEXT: ContextVar[RequestContext] = ContextVar(
    "risk_request_context",
    default=_EMPTY_CONTEXT,
)


def current_context() -> RequestContext:
    """返回当前并发上下文中的不可变请求身份。"""
    return _REQUEST_CONTEXT.get()


def new_request_id() -> str:
    """生成适合写入响应头和日志的随机请求 ID。"""
    return uuid.uuid4().hex


def new_trace_id() -> str:
    """生成满足 W3C Trace Context 约束的 32 位追踪 ID。"""
    while True:
        trace_id = uuid.uuid4().hex
        if trace_id != _ZERO_TRACE_ID:
            return trace_id


def is_valid_trace_id(value: object) -> bool:
    """判断值是否为非全零的小写 W3C 追踪 ID。"""
    normalized = str(value or "")
    return bool(
        _TRACE_ID_PATTERN.fullmatch(normalized)
        and normalized != _ZERO_TRACE_ID
    )


def is_valid_request_id(value: object) -> bool:
    """判断入站请求 ID 是否仅包含安全字符且长度受限。"""
    return bool(_REQUEST_ID_PATTERN.fullmatch(str(value or "")))


def trace_id_from_traceparent(value: object) -> str:
    """从合法 W3C traceparent 提取追踪 ID，非法输入返回空值。"""
    matched = _TRACEPARENT_PATTERN.fullmatch(str(value or "").strip())
    if not matched:
        return ""
    trace_id, span_id, _flags = matched.groups()
    if trace_id == _ZERO_TRACE_ID or span_id == "0" * 16:
        return ""
    return trace_id


@contextmanager
def bind_request_context(
    *,
    request_id: str | None = None,
    trace_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
) -> Iterator[RequestContext]:
    """绑定当前阶段身份，退出时恢复父上下文。"""
    parent = current_context()
    if request_id is None:
        resolved_request_id = parent.request_id or new_request_id()
    else:
        resolved_request_id = (
            str(request_id).strip()
            if is_valid_request_id(request_id)
            else new_request_id()
        )

    if trace_id is None:
        resolved_trace_id = (
            parent.trace_id
            if is_valid_trace_id(parent.trace_id)
            else new_trace_id()
        )
    else:
        resolved_trace_id = (
            str(trace_id)
            if is_valid_trace_id(trace_id)
            else new_trace_id()
        )

    child = replace(
        parent,
        request_id=resolved_request_id,
        trace_id=resolved_trace_id,
        user_id=parent.user_id if user_id is None else str(user_id).strip(),
        session_id=(
            parent.session_id
            if session_id is None
            else str(session_id).strip()
        ),
        run_id=parent.run_id if run_id is None else str(run_id).strip(),
    )
    token = _REQUEST_CONTEXT.set(child)
    try:
        yield child
    finally:
        _REQUEST_CONTEXT.reset(token)
