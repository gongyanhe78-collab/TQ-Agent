"""动态证据窗口与多样化选择的确定性测试。"""

from __future__ import annotations

from backend.app.models import DocumentChunk, DocumentRetrievalHit
from backend.app.services.query.evidence_window import (
    diversify_evidence_by_case,
    dynamic_evidence_window,
)


def _命中(chunk_id: str) -> DocumentRetrievalHit:
    return DocumentRetrievalHit(
        chunk=DocumentChunk(
            source_pdf=f"{chunk_id}.pdf",
            chunk_id=chunk_id,
            chunk_no=1,
            content="证据正文",
        ),
        score=1.0,
    )


def test_dynamic_window_expands_for_aggregate_and_comparison_questions() -> None:
    """统计、列举和比较问题需要更宽的证据窗口。"""

    assert dynamic_evidence_window("统计2025年5月共有几个过程", 3) == 8
    assert dynamic_evidence_window("列出全部天气过程", 3) == 8
    assert dynamic_evidence_window("比较两个过程的主要差异", 3) == 8
    assert dynamic_evidence_window("五台山最大阵风是多少", 5) == 5


def test_dynamic_window_uses_matched_case_count_and_caps_at_ten() -> None:
    """明确命中多个个例时按个例数扩展，但最多保留十条证据。"""

    assert dynamic_evidence_window("继续分析这些结果", 3, matched_case_count=2) == 5
    assert dynamic_evidence_window("继续分析这些结果", 3, matched_case_count=6) == 10
    assert dynamic_evidence_window("单点事实", 12) == 10


def test_diversify_evidence_prefers_one_chunk_per_case_before_repeats() -> None:
    """多案例问题应先覆盖不同个例，再补足同个例的后续切片。"""

    hits = [_命中("a1"), _命中("a2"), _命中("a3"), _命中("b1"), _命中("b2"), _命中("c1")]
    mapping = {
        "a1": "case-a",
        "a2": "case-a",
        "a3": "case-a",
        "b1": "case-b",
        "b2": "case-b",
        "c1": "case-c",
    }

    ordered = diversify_evidence_by_case(hits, mapping, limit=5)

    assert [item.chunk.chunk_id for item in ordered] == ["a1", "b1", "c1", "a2", "b2"]
