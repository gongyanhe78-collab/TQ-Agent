"""承接式会话 PDF 的内容边界回归测试。"""
from __future__ import annotations

import unittest
import unicodedata
from pathlib import Path
from tempfile import TemporaryDirectory

from pypdf import PdfReader

from backend.app.services.agent.case_multidim_search.reporting.conversation_report import ConversationReportBuilder


class ConversationReportBuilderTests(unittest.TestCase):
    """确保简易报告只排版页面可见回答且不重复扩写。"""

    def test_pdf_contains_visible_answer_once_without_formal_report_sections(self) -> None:
        """页面正文应只出现一次，隐藏证据不会进入专用排版器。"""
        answer = (
            "## 统计结论\n\n"
            "2025年5月至7月共记录13次主要气象灾害过程。\n\n"
            "| 时段 | 过程 |\n"
            "| --- | --- |\n"
            "| 5月2日 | 雷暴大风天气过程 |\n\n"
            "1. 重点关注短时强降水。"
        )
        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "conversation.pdf"
            ConversationReportBuilder().build(
                answer,
                output,
                "2025年5月至7月气象灾害分析报告",
            )
            reader = PdfReader(str(output))
            # Chromium 的 CJK 字体可能返回兼容表意码位，归一化后再校验真实文本。
            text = unicodedata.normalize(
                "NFKC",
                "\n".join(page.extract_text() or "" for page in reader.pages),
            )
            page_width = float(reader.pages[0].mediabox.width)
            page_height = float(reader.pages[0].mediabox.height)

        self.assertIn("2025年5月至7月气象灾害分析报告", text)
        self.assertEqual(text.count("2025年5月至7月共记录13次主要气象灾害过程"), 1)
        self.assertEqual(text.count("雷暴大风天气过程"), 1)
        self.assertNotIn("代表个例分析", text)
        self.assertNotIn("综合结论与建议", text)
        self.assertNotIn("<!doctype html>", text.lower())
        # 消息级报告固定使用 A4，防止浏览器把 HTML 源码或默认 Letter 页面打印出来。
        self.assertAlmostEqual(page_width, 595.28, delta=2.0)
        self.assertAlmostEqual(page_height, 841.89, delta=2.0)

    def test_parser_keeps_all_rows_when_table_exceeds_one_page_chunk(self) -> None:
        """长表格分块前必须保留全部行，不能沿用正式文档的十四行截断。"""
        rows = "\n".join(f"| {index} | 个例{index} |" for index in range(1, 18))
        blocks = ConversationReportBuilder._parse_markdown(
            f"| 编号 | 名称 |\n| --- | --- |\n{rows}"
        )
        table = next(block for block in blocks if block["type"] == "table")
        self.assertEqual(len(table["rows"]), 17)


if __name__ == "__main__":
    unittest.main()
