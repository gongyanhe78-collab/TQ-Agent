"""结构化事实抽取与回答上下文注入的确定性测试。"""

from __future__ import annotations

from types import SimpleNamespace

from backend.app.models import DocumentChunk, DocumentRetrievalHit, StandardCase
from backend.app.services.model_client.chat import DashScopeChatClient
from backend.app.services.query.document_answer import DocumentQuestionAnswerer
from backend.app.services.query.structured_facts import (
    extract_structured_facts,
    render_structured_fact_block,
)
from backend.evals.runners.real_model import 真实模型执行器


class _图片展示器:
    """为上下文测试提供无图片的展示器替身。"""

    def images_for_document_hit(self, _hit):
        return []


class _文档存储:
    """按 chunk_id 返回测试切片。"""

    def __init__(self, chunks: dict[str, DocumentChunk]) -> None:
        self.chunks = chunks

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        return self.chunks.get(chunk_id)


def _命中切片() -> DocumentRetrievalHit:
    return DocumentRetrievalHit(
        chunk=DocumentChunk(
            source_pdf="FST2025-4.pdf",
            chunk_id="FST2025-4-chunk-003",
            chunk_no=3,
            content=(
                "过程概况如下。国家站最大阵风风速达 39 m/s（13级），"
                "出现在五台山国家基本气象站[53588]，发生于 2025年4月13日07:41；"
                "全省共33个国家站最大阵风突破4月历史极值。"
            ),
        ),
        score=1.0,
    )


def test_extract_structured_facts_keeps_exact_numeric_evidence() -> None:
    """抽取结果必须保留原文中的数值、单位、站点和发生时间。"""

    facts = extract_structured_facts(
        _命中切片().chunk.content,
        evidence_id="FST2025-4-chunk-003",
        limit=8,
    )

    texts = [fact.text for fact in facts]
    assert any("39 m/s" in text and "五台山国家基本气象站" in text for text in texts)
    assert any("2025年4月13日07:41" in text for text in texts)
    assert all(fact.evidence_id == "FST2025-4-chunk-003" for fact in facts)


def test_rendered_fact_block_is_explicit_and_traceable() -> None:
    """事实块应明确标注来源，避免模型把抽取内容当成无依据摘要。"""

    facts = extract_structured_facts(
        _命中切片().chunk.content,
        evidence_id="FST2025-4-chunk-003",
        limit=8,
    )

    block = render_structured_fact_block(facts)

    assert "可核验结构化事实" in block
    assert "不得省略、替换或四舍五入" in block
    assert "FST2025-4-chunk-003" in block
    assert "2025年4月13日07:41" in block


def test_document_answerer_injects_fact_block_into_model_context() -> None:
    """生产文档问答上下文应包含从命中切片抽取的结构化事实。"""

    hit = _命中切片()
    answerer = DocumentQuestionAnswerer(
        retriever=SimpleNamespace(),
        presenter=_图片展示器(),
        llm_client=SimpleNamespace(),
    )

    blocks = answerer.context_blocks([hit])

    assert any("可核验结构化事实" in block for block in blocks)
    assert any("五台山国家基本气象站" in block for block in blocks)


def test_real_runner_injects_case_metadata_and_source_facts() -> None:
    """真实评测执行器应在个例证据中补充元数据和原文事实。"""

    chunk = _命中切片().chunk
    runner = object.__new__(真实模型执行器)
    runner.standard_cases = [
        StandardCase(
            case_id="FST2025-4-std-case-001",
            title="4月11-13日大风过程",
            date_range="2025年4月11-13日",
            disaster_types=["大风"],
            affected_areas=["全省"],
            source_chunk_ids=[chunk.chunk_id],
        )
    ]
    runner.document_store = _文档存储({chunk.chunk_id: chunk})

    blocks = runner._证据上下文(
        [chunk.chunk_id],
        ["FST2025-4-std-case-001"],
        route="rag",
    )

    combined = "\n".join(blocks)
    assert "个例ID：FST2025-4-std-case-001" in combined
    assert "可核验结构化事实" in combined
    assert "2025年4月13日07:41" in combined


def test_answer_prompt_requires_every_verified_fact() -> None:
    """回答提示词必须要求逐项保留数值、日期、站点和数量。"""

    client = DashScopeChatClient(
        api_key="test-key",
        base_url="https://example.invalid",
        model="qwen3.7-flash",
    )

    messages = client._answer_messages(  # noqa: SLF001
        "五台山最大阵风是多少？",
        ["可核验结构化事实：国家站最大阵风为39米每秒。"],
    )

    system_prompt = messages[0]["content"]
    assert "逐项" in system_prompt
    assert "不得省略、替换或四舍五入" in system_prompt
    assert "数值、日期、站点和数量" in system_prompt
    assert "关键事实核对" in system_prompt
    assert "回答开头必须明确写出总数" in system_prompt
    assert "必须同时说明相同点和差异" in system_prompt

