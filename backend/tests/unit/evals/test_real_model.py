"""真实模型执行器的纯函数与指标归一化测试。"""

from __future__ import annotations

from backend.evals.metrics.answer import (
    forbidden_claim_count,
    required_fact_coverage,
)
from backend.app.agents.case_multidim_search.schemas import CaseSearchQuery
from backend.evals.runners.real_model import (
    真实模型执行器,
    _判断拒答,
    _提取引用证据,
    _构建记忆上下文,
    _窗口超出知识范围,
    _是否模型降级演练,
)


def test_build_memory_context_from_conversation_setup() -> None:
    """多轮预设应转换为意图 Agent 可识别的历史轮次。"""

    memory = _构建记忆上下文(
        [
            {"role": "user", "content": "列出2025年5月的全部标准天气过程。"},
            {"role": "assistant", "content": "已整理5个标准个例。"},
        ]
    )
    assert memory["recent_turns"][0]["role"] == "user"
    assert memory["previous_turn"]["content"] == "已整理5个标准个例。"


def test_build_memory_context_handles_empty_setup() -> None:
    """没有历史轮次时返回空上下文，不影响单轮评测。"""

    assert _构建记忆上下文([]) == {}


def test_extract_cited_evidence_keeps_answer_order_and_dedupes() -> None:
    """只提取回答中真实出现的证据 ID，并保持首次出现顺序。"""

    answer = "依据 FST2025-3-chunk-002 与 FST2025-3-chunk-001，再次引用 FST2025-3-chunk-002。"
    cited = _提取引用证据(
        answer,
        ["FST2025-3-chunk-001", "FST2025-3-chunk-002", "FST2025-3-chunk-009"],
    )
    assert cited == ["FST2025-3-chunk-001", "FST2025-3-chunk-002"]


def test_abstention_requires_clarify_or_explicit_refusal() -> None:
    """澄清路由和明确拒答才计为拒答，正常回答不误判。"""

    assert _判断拒答("未检索到相关证据。", "rag", True) is True
    assert _判断拒答("这是完整的分析结论。", "rag", False) is False
    assert _判断拒答("请补充具体日期。", "clarify", False) is True


def test_answer_generation_uses_production_token_budget() -> None:
    """真实评测回答不能使用低于生产默认值的 800 token 截断预算。"""

    class RecordingChat:
        """记录回答调用参数，不访问外部模型。"""

        def __init__(self) -> None:
            self.max_tokens = None

        def answer_with_context(self, _question, _blocks, *, max_tokens):
            self.max_tokens = max_tokens
            return "完整回答"

    recorder = RecordingChat()
    runner = object.__new__(真实模型执行器)
    runner.llm_client = recorder
    runner._证据上下文 = lambda chunk_ids, case_ids, route: ["证据块"]
    runner._记忆上下文块 = lambda memory: ""

    answer = runner._生成回答(
        "测试问题",
        chunk_ids=["chunk-1"],
        case_ids=[],
        route="rag",
    )

    assert answer == "完整回答"
    assert recorder.max_tokens == 2048


def test_required_fact_coverage_normalizes_temperature_units() -> None:
    """℃ 与“摄氏度”应视为同一事实，空白不影响覆盖判断。"""

    answer = "2025年3月全省平均气温为6.0℃，较常年同期偏高0.9℃。"
    facts = ["平均气温为6.0摄氏度", "较常年同期偏高0.9摄氏度"]
    assert required_fact_coverage(answer, facts) == 1.0


def test_forbidden_claim_count_ignores_whitespace_differences() -> None:
    """禁止结论的匹配同样忽略空白差异，避免漏判。"""

    assert forbidden_claim_count("一定 会出现特大暴雨", ["一定会出现特大暴雨"]) == 1


def test_explicit_query_window_detects_out_of_knowledge_range() -> None:
    """2025年8月查询必须被识别为超出当前知识库截止日期。"""

    assert _窗口超出知识范围(
        CaseSearchQuery(years=[2025], months=[8]),
        "2025-01-11",
        "2025-07-31",
    ) is True
    assert _窗口超出知识范围(
        CaseSearchQuery(years=[2025], months=[7]),
        "2025-01-11",
        "2025-07-31",
    ) is False


def test_model_degradation_drill_detection() -> None:
    """只有明确的模型不可用演练才走确定性证据摘要。"""

    assert _是否模型降级演练("在聊天模型不可用时，仍返回已有证据摘要。") is True
    assert _是否模型降级演练("检索2025年8月山西暴雪标准个例。") is False
