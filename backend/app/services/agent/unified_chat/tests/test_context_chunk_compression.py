"""主 RAG 已知个例证据压缩测试。"""
from __future__ import annotations

import unittest

from backend.app.main import _compress_context_chunk_group
from backend.app.models import DocumentChunk


def _chunk(number: int, content: str) -> DocumentChunk:
    """构造保持固定原文顺序的测试片段。"""
    return DocumentChunk(
        source_pdf="case.pdf",
        chunk_id=f"chunk-{number:02d}",
        chunk_no=number,
        content=content,
    )


class ContextChunkCompressionTests(unittest.TestCase):
    """长个例应减量，同时保留四类核心业务证据。"""

    def test_long_case_preserves_section_coverage_and_order(self) -> None:
        chunks = [_chunk(index, f"普通衔接内容 {index}") for index in range(1, 17)]
        chunks[2].content = "天气实况显示累计降水和最高气温均达到过程极值。"
        chunks[5].content = "环流形势与天气成因分析包含500hPa高压脊和850hPa水汽。"
        chunks[8].content = "过程影响农业、交通和电力服务，需关注人体健康风险。"
        chunks[11].content = "数值预报与预报效果复盘显示EC模式存在局地漏报。"

        selected = _compress_context_chunk_group("详细分析实况、成因、影响和预报效果", chunks, limit=12)
        selected_ids = [item.chunk_id for item in selected]

        self.assertLessEqual(len(selected), 12)
        self.assertEqual(selected_ids, sorted(selected_ids))
        self.assertIn("chunk-01", selected_ids)
        self.assertIn("chunk-03", selected_ids)
        self.assertIn("chunk-06", selected_ids)
        self.assertIn("chunk-09", selected_ids)
        self.assertIn("chunk-12", selected_ids)
        self.assertIn("chunk-16", selected_ids)

    def test_short_case_keeps_all_chunks(self) -> None:
        chunks = [_chunk(2, "第二段"), _chunk(1, "第一段")]

        selected = _compress_context_chunk_group("分析这个个例", chunks, limit=12)

        self.assertEqual([item.chunk_id for item in selected], ["chunk-01", "chunk-02"])


if __name__ == "__main__":
    unittest.main()
