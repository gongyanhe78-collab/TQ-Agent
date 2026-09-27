"""验证图片展示策略和 PDF 持久化缩略图的专项回归测试。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter

from backend.app.services.agent.case_multidim_search.reporting.report_preview import (
    ensure_report_thumbnail,
)
from backend.app.services.agent.unified_chat.orchestrator import UnifiedChatOrchestrator


class EvidenceAndReportPreviewTest(unittest.TestCase):
    """覆盖本次六项界面优化中依赖后端协议的部分。"""

    def test_rag_images_are_hidden_without_image_task(self) -> None:
        """普通分析命中图片时也不应默认展示。"""
        tasks = [{"type": "case_analysis", "route": "rag"}]
        self.assertEqual(UnifiedChatOrchestrator._rag_image_display_mode(tasks), "hidden")

    def test_rag_images_become_references_for_image_lookup(self) -> None:
        """意图模型规划图片子任务后应提供按需查看入口。"""
        tasks = [
            {"type": "case_analysis", "route": "rag"},
            {"type": "image_lookup", "route": "rag", "depends_on": ["task_1"]},
        ]
        self.assertEqual(UnifiedChatOrchestrator._rag_image_display_mode(tasks), "references")

    def test_real_pdf_generates_reusable_thumbnail(self) -> None:
        """真实 PDF 第一页应生成稳定 PNG，并在第二次调用时直接复用。"""
        with tempfile.TemporaryDirectory() as temporary_dir:
            pdf_path = Path(temporary_dir) / "sample.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=595, height=842)
            with pdf_path.open("wb") as stream:
                writer.write(stream)

            thumbnail = ensure_report_thumbnail(pdf_path)
            first_mtime = thumbnail.stat().st_mtime_ns
            reused = ensure_report_thumbnail(pdf_path)

            self.assertEqual(reused, thumbnail)
            self.assertEqual(reused.stat().st_mtime_ns, first_mtime)
            self.assertGreater(reused.stat().st_size, 100)
            self.assertEqual(reused.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
