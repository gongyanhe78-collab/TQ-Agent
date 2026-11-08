"""检索指标与回答指标的确定性行为测试。"""

from __future__ import annotations

from backend.evals.metrics.answer import (
    abstention_accuracy,
    behavior_satisfaction_rate,
    citation_support_rate,
    forbidden_claim_count,
    required_fact_coverage,
)
from backend.evals.metrics.retrieval import (
    evidence_id_hit_rate,
    hit_at_k,
    mrr,
    no_result_accuracy,
    recall_at_k,
    route_accuracy,
)
from backend.evals.schemas.dataset import BehaviorCheck, ExpectedBehavior


def test_hit_at_k_and_mrr_use_stable_ranked_evidence() -> None:
    """排名必须按输入顺序稳定处理，不依赖无序集合的迭代顺序。"""

    assert hit_at_k(["a", "b", "c"], ["c"], 3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["a", "d"], 5) == 0.5
    assert mrr(["a", "b", "c"], ["c"]) == 1 / 3


def test_route_accuracy_requires_exact_match() -> None:
    """路由只有完全一致才计为正确。"""

    assert route_accuracy(["rag", "clarify"], ["rag", "multidim_search"]) == 0.5
    assert route_accuracy([], ["rag"]) == 0.0


def test_empty_evidence_only_counts_when_expected_is_empty() -> None:
    """空证据集合只有在期望也为空时才计为正确。"""

    assert hit_at_k([], ["item"], 3) == 0.0
    assert hit_at_k([], [], 3) == 1.0
    assert hit_at_k(["item"], [], 3) == 0.0
    assert recall_at_k([], ["a"], 5) == 0.0
    assert recall_at_k([], [], 5) == 1.0
    assert mrr([], ["a"]) == 0.0
    assert mrr([], []) == 1.0
    assert evidence_id_hit_rate([], ["a"]) == 0.0
    assert evidence_id_hit_rate([], []) == 1.0
    assert evidence_id_hit_rate(["a"], []) == 0.0


def test_no_result_accuracy_matches_expected_flag() -> None:
    """无结果识别只有预测和期望一致才计为正确。"""

    assert no_result_accuracy(True, True) == 1.0
    assert no_result_accuracy(False, False) == 1.0
    assert no_result_accuracy(True, False) == 0.0
    assert no_result_accuracy(False, True) == 0.0


def test_required_fact_coverage_supports_substring_and_alternatives() -> None:
    """必含事实支持确定性子串和以竖线分隔的任选其一表达式。"""

    assert required_fact_coverage("今日香蕉产量为两吨", ["香蕉"]) == 1.0
    assert required_fact_coverage("今日香蕉产量为两吨", ["苹果|香蕉"]) == 1.0
    assert required_fact_coverage("今日桃子产量为三吨", ["苹果|香蕉", "桃子"]) == 0.5
    assert required_fact_coverage("", []) == 1.0


def test_required_fact_coverage_normalizes_rendering_and_common_units() -> None:
    """Markdown 加粗和常见单位不应让正确事实被误判为缺失。"""

    assert required_fact_coverage("**14.1毫米**", ["14.1毫米"]) == 1.0
    assert required_fact_coverage("风速为**39 m/s**", ["风速为39米每秒"]) == 1.0
    assert required_fact_coverage("雨量为 **126.6mm**", ["雨量为126.6毫米"]) == 1.0
    assert required_fact_coverage("时间为7月12-18日", ["时间为7月12日至18日"]) == 1.0
    assert required_fact_coverage("时间为4月11日-13日", ["时间为4月11-13日"]) == 1.0
    assert required_fact_coverage("平均降水量：14.1毫米", ["平均降水量为14.1毫米"]) == 1.0
    assert required_fact_coverage("较常年同期：偏少3.4毫米", ["较常年同期偏少3.4毫米"]) == 1.0


def test_behavior_satisfaction_rate_supports_route_answer_and_evidence_checks() -> None:
    """行为指标应按结构化检查逐条判定，而不是匹配“应……”描述文本。"""

    behaviors = [
        ExpectedBehavior(
            description="应先澄清",
            checks=[BehaviorCheck(kind="route_equals", values=["clarify"])],
        ),
        ExpectedBehavior(
            description="应说明冰雹事实并引用证据",
            checks=[
                BehaviorCheck(kind="answer_contains_all", values=["冰雹", "5月8-9日"]),
                BehaviorCheck(kind="evidence_hit", values=["chunk-1"]),
            ],
        ),
    ]

    assert behavior_satisfaction_rate(
        "5月8-9日出现冰雹",
        actual_route="rag",
        actual_evidence=["chunk-1"],
        no_result=False,
        abstained=False,
        behaviors=behaviors,
    ) == 0.5
    assert behavior_satisfaction_rate(
        "请明确具体日期",
        actual_route="clarify",
        actual_evidence=[],
        no_result=False,
        abstained=True,
        behaviors=behaviors[:1],
    ) == 1.0
    assert behavior_satisfaction_rate(
        "",
        actual_route="rag",
        actual_evidence=[],
        no_result=False,
        abstained=False,
        behaviors=[],
    ) == 1.0


def test_forbidden_claim_count_uses_full_phrase_not_keyword() -> None:
    """禁止结论必须整体出现才计数，避免普通关键词误报。"""

    assert forbidden_claim_count("本月降水量偏少", ["降水量偏多"]) == 0
    assert forbidden_claim_count("本月降水量偏多", ["降水量偏多"]) == 1


def test_citation_support_rate_uses_retrieved_evidence_ids() -> None:
    """引用支持率只统计已检索证据覆盖的引用，空引用不惩罚。"""

    assert citation_support_rate(["c1"], ["c1", "c2"]) == 1.0
    assert citation_support_rate(["c1", "c2", "c1"], ["c1"]) == 0.5
    assert citation_support_rate([], ["c1"]) == 1.0


def test_abstention_accuracy_matches_only_when_both_equal() -> None:
    """拒答准确率要求预测和期望同时一致。"""

    assert abstention_accuracy(True, True) == 1.0
    assert abstention_accuracy(False, False) == 1.0
    assert abstention_accuracy(True, False) == 0.0
    assert abstention_accuracy(False, True) == 0.0
