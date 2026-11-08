"""日志脱敏与超长正文截断的单元测试。"""
from __future__ import annotations

import logging
import unittest

from backend.app.observability.logging import (
    JsonContextFormatter,
    sanitize_text,
)


class SanitizeTextTests(unittest.TestCase):
    """验证密钥脱敏、单行化和超长内容截断。"""

    def test_masks_authorization_bearer_token(self) -> None:
        text = "Authorization: Bearer abc.def.ghi call=/api/query"
        sanitized = sanitize_text(text)
        self.assertNotIn("abc.def.ghi", sanitized)
        self.assertIn("[已脱敏]", sanitized)

    def test_masks_api_key_style_fields(self) -> None:
        text = "api_key=sk-123456 refresh_token=secret-value"
        sanitized = sanitize_text(text)
        self.assertNotIn("sk-123456", sanitized)
        self.assertNotIn("secret-value", sanitized)

    def test_flattens_newlines_to_single_line(self) -> None:
        sanitized = sanitize_text("第一行\n第二行\n第三行")
        self.assertNotIn("\n", sanitized)

    def test_truncates_overlong_text_without_full_question_leak(self) -> None:
        long_text = "用户问题" + "很长的内容" * 100
        sanitized = sanitize_text(long_text)
        self.assertLessEqual(len(sanitized), 210)
        self.assertIn("[已截断]", sanitized)
        self.assertNotIn("很长的内容" * 100, sanitized)


class JsonContextFormatterTests(unittest.TestCase):
    """验证 JSON 格式化器不输出异常堆栈正文。"""

    def test_exception_payload_omits_stack_trace(self) -> None:
        formatter = JsonContextFormatter()
        record = logging.LogRecord(
            name="backend.app.test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="模型调用失败",
            args=(),
            exc_info=(ValueError, ValueError("boom"), None),
        )
        output = formatter.format(record)
        self.assertIn("模型调用失败", output)
        self.assertNotIn("Traceback", output)
        self.assertNotIn("boom", output)


if __name__ == "__main__":
    unittest.main()
