"""请求上下文日志过滤器与本地安全格式化器。"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from backend.app.observability.context import current_context


_SECRET_PATTERNS = (
    re.compile(
        r"(?i)(authorization\s*:\s*(?:bearer|basic)\s+)([^\s,;]+)"
    ),
    re.compile(
        r"(?i)((?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
        r"cookie|set-cookie|authorization)\s*[=:]\s*)([^\s,;]+)"
    ),
    re.compile(r"(?i)(bearer\s+)([^\s,;]+)"),
)
_LOG_LOGGER_NAME = "backend.app"


def sanitize_text(value: Any) -> str:
    """把日志文本压缩为单行并脱敏常见凭据字段。"""
    text = str(value if value is not None else "")
    text = text.replace("\r", " ").replace("\n", " ")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(r"\1[已脱敏]", text)
    return re.sub(r" +", " ", text).strip()


def _exception_payload(record: logging.LogRecord) -> dict[str, str] | None:
    """仅保留异常类型、阶段和脱敏摘要，避免输出堆栈正文。"""
    if not isinstance(record.exc_info, tuple) or len(record.exc_info) != 3:
        return None
    exception_type = record.exc_info[0]
    return {
        "type": getattr(exception_type, "__name__", "Exception"),
        "stage": sanitize_text(getattr(record, "stage", "")),
        "summary": "[已脱敏]",
    }


class ContextLogFilter(logging.Filter):
    """把当前请求身份注入日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:
        """注入固定关联字段并允许日志继续传递。"""
        context = current_context()
        record.request_id = context.request_id
        record.trace_id = context.trace_id
        record.user_id = context.user_id
        record.session_id = context.session_id
        record.run_id = context.run_id
        if not hasattr(record, "stage"):
            record.stage = ""
        return True


class JsonContextFormatter(logging.Formatter):
    """生成适合容器采集的单行 JSON 日志。"""

    def format(self, record: logging.LogRecord) -> str:
        """格式化并脱敏日志正文与异常摘要。"""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_text(record.getMessage()),
            "request_id": sanitize_text(
                getattr(record, "request_id", current_context().request_id)
            ),
            "trace_id": sanitize_text(
                getattr(record, "trace_id", current_context().trace_id)
            ),
            "user_id": sanitize_text(
                getattr(record, "user_id", current_context().user_id)
            ),
            "session_id": sanitize_text(
                getattr(record, "session_id", current_context().session_id)
            ),
            "run_id": sanitize_text(
                getattr(record, "run_id", current_context().run_id)
            ),
            "stage": sanitize_text(getattr(record, "stage", "")),
        }
        exception_payload = _exception_payload(record)
        if exception_payload is not None:
            payload["exception"] = exception_payload
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )


class TextContextFormatter(logging.Formatter):
    """生成本地可读且不含异常堆栈的文本日志。"""

    def format(self, record: logging.LogRecord) -> str:
        """格式化固定关联字段，异常只输出类型和脱敏摘要。"""
        context = current_context()
        fields = (
            f"{record.levelname} "
            f"request_id={sanitize_text(getattr(record, 'request_id', context.request_id))} "
            f"trace_id={sanitize_text(getattr(record, 'trace_id', context.trace_id))} "
            f"user_id={sanitize_text(getattr(record, 'user_id', context.user_id))} "
            f"run_id={sanitize_text(getattr(record, 'run_id', context.run_id))} "
            f"stage={sanitize_text(getattr(record, 'stage', ''))} "
            f"message={sanitize_text(record.getMessage())}"
        )
        exception_payload = _exception_payload(record)
        if exception_payload is not None:
            fields += (
                f" error_type={exception_payload['type']}"
                f" error_stage={exception_payload['stage']}"
                f" error_summary={exception_payload['summary']}"
            )
        return fields


def configure_logging(settings: Any) -> None:
    """为应用日志安装一次带上下文字段的本地处理器。"""
    logger = logging.getLogger(_LOG_LOGGER_NAME)
    if any(
        getattr(handler, "_risk_context_handler", False)
        for handler in logger.handlers
    ):
        return
    handler = logging.StreamHandler()
    handler._risk_context_handler = True
    handler.setFormatter(
        JsonContextFormatter()
        if bool(getattr(settings, "log_json", False))
        else TextContextFormatter()
    )
    handler.addFilter(ContextLogFilter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
