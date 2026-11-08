"""真实评测多轮记忆上下文构建测试。"""

from __future__ import annotations

from backend.app.agents.query_intent.agent import QueryIntentAgent
from backend.app.models import StandardCase
from backend.evals.runners.real_model import _应沿用上一轮结果, _构建记忆上下文


def _个例(
    case_id: str,
    title: str,
    date_range: str,
    disasters: list[str],
) -> StandardCase:
    """构造多轮上下文测试所需的标准个例。"""

    return StandardCase(
        case_id=case_id,
        title=title,
        date_range=date_range,
        disaster_types=disasters,
        affected_areas=["全省"],
    )


def test_memory_context_carries_previous_aggregate_case_refs() -> None:
    """上一轮聚合查询的个例 ID 和结果摘要必须进入记忆上下文。"""

    cases = [
        _个例("case-5-1", "5月2日雷暴大风过程", "2025年5月2日", ["雷暴大风"]),
        _个例("case-5-2", "5月8-9日强对流过程", "2025年5月8-9日", ["强对流"]),
    ]
    memory = _构建记忆上下文(
        [
            {"role": "user", "content": "列出2025年5月的全部标准天气过程。"},
            {"role": "assistant", "content": "已整理5个标准个例。"},
        ],
        cases,
    )

    assert memory["previous_turn"]["question"] == "列出2025年5月的全部标准天气过程。"
    assert memory["previous_turn"]["answer_summary"] == "已整理5个标准个例。"
    assert [item["case_id"] for item in memory["previous_turn"]["case_refs"]] == [
        "case-5-1",
        "case-5-2",
    ]
    assert [item["case_id"] for item in memory["available_case_refs"]] == [
        "case-5-1",
        "case-5-2",
    ]


def test_memory_context_orders_compared_cases_by_date_mentions() -> None:
    """比较句中的多个日期应按原文出现顺序转换为个例引用。"""

    cases = [
        _个例("case-apr-2", "4月16-18日大风过程", "2025年4月16-18日", ["大风"]),
        _个例("case-apr-1", "4月11-13日大风过程", "2025年4月11-13日", ["大风"]),
    ]
    memory = _构建记忆上下文(
        [
            {
                "role": "user",
                "content": "比较2025年4月11日至13日和4月16日至18日的大风过程。",
            },
            {"role": "assistant", "content": "已得到两个4月大风个例的对比结果。"},
        ],
        cases,
    )

    assert [item["case_id"] for item in memory["previous_turn"]["case_refs"]] == [
        "case-apr-1",
        "case-apr-2",
    ]


def test_memory_context_prefers_exact_answer_date_range() -> None:
    """上一轮回答给出完整日期范围时，不应把范围内的单日个例混入引用。"""

    cases = [
        _个例("case-single", "7月28日强对流过程", "2025年7月28日", ["强对流"]),
        _个例("case-range", "7月28-31日高温过程", "2025年7月28-31日", ["高温"]),
    ]
    memory = _构建记忆上下文(
        [
            {"role": "user", "content": "找出2025年7月的高温过程。"},
            {"role": "assistant", "content": "对应7月28日至31日的高温过程。"},
        ],
        cases,
    )

    assert [item["case_id"] for item in memory["previous_turn"]["case_refs"]] == ["case-range"]

def test_follow_up_question_detection_covers_result_references() -> None:
    """常见承接表达应触发上一轮结果复用。"""

    for question in (
        "把刚才列出的5月个例按灾种分类。",
        "只看其中与太原相关的个例。",
        "把区域从运城改成临汾，再查一次。",
        "基于上一轮结果再补充一条原文证据。",
    ):
        assert _应沿用上一轮结果(question) is True
    assert _应沿用上一轮结果("列出2025年5月的全部标准天气过程。") is False


def test_intent_context_detection_recognizes_memory_phrases() -> None:
    """意图规则应能识别新增的多轮承接表达。"""

    for question in ("刚才提到的最大阵风是多少", "只看其中与太原相关的个例", "在当前结果里只保留暴雨过程", "继续按我一直偏好的时间顺序展示结果"):
        assert QueryIntentAgent._has_context_reference(question) is True

def test_memory_block_contains_previous_result_and_case_refs() -> None:
    """回答阶段必须收到上一轮摘要和个例 ID，才能承接指代。"""

    from backend.evals.runners.real_model import 真实模型执行器

    block = 真实模型执行器._记忆上下文块({
        "previous_turn": {
            "question": "列出2025年6月的两个标准个例。",
            "answer_summary": "已得到6月13日至14日和6月27日至28日两个个例。",
            "case_refs": [
                {"case_id": "case-6-1", "title": "6月13-14日暴雨过程"},
                {"case_id": "case-6-2", "title": "6月27-28日强对流过程"},
            ],
        },
    })
    assert "上一轮问题" in block
    assert "case-6-1" in block
    assert "case-6-2" in block