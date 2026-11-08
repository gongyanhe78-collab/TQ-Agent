"""真实评测执行器的相似个例 Agent 接入测试。"""

from __future__ import annotations

from types import SimpleNamespace

from backend.evals.runners.real_model import 真实模型执行器
from backend.evals.schemas.dataset import EvaluationCase


class _相似匹配Agent:
    """记录请求并返回稳定相似个例列表。"""

    def __init__(self, case_ids: list[str]) -> None:
        self.case_ids = case_ids
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            matched_cases=[
                {
                    "case_id": case_id,
                    "retrieval_score": 0.91 - index * 0.1,
                    "match_reasons": [f"理由{index + 1}"],
                }
                for index, case_id in enumerate(self.case_ids)
            ]
        )


class _失败相似匹配Agent:
    """模拟相似匹配 Agent 故障。"""

    def run(self, _request):
        raise RuntimeError("模型超时")


def _样例外壳() -> EvaluationCase:
    """构造执行检索所需的最小黄金样例。"""

    return EvaluationCase.model_validate(
        {
            "case_id": "golden-v1-similar-001",
            "category": "相似匹配",
            "question": "比较两个历史暴雨过程",
            "conversation_setup": [],
            "expected_route": "similar_case_match",
            "expected_case_ids": ["smart-case-1"],
            "expected_chunk_ids": [],
            "required_facts": [],
            "expected_behaviors": [],
            "forbidden_claims": [],
            "max_latency_ms": 120000,
            "tags": [],
            "knowledge_version": "standard-cases-2025-v1",
            "prompt_version": "enterprise-runtime-v1",
            "model_config": {
                "provider": "dashscope",
                "chat_model": "qwen-plus",
                "embedding_model": "text-embedding-v4",
                "temperature": 0,
                "seed": 42,
            },
        }
    )


def _执行器(agent) -> 真实模型执行器:
    """创建只包含相似匹配测试依赖的执行器。"""

    runner = object.__new__(真实模型执行器)
    runner.top_n = 5
    runner.smart_match_agent = agent
    runner._结构化检索 = lambda _case, _memory: ["fallback-case"]  # type: ignore[method-assign]
    runner._文档检索 = lambda _question, **_kwargs: ["chunk-1"]  # type: ignore[method-assign]
    return runner


def test_similar_route_uses_smart_match_agent_case_ids() -> None:
    """相似匹配路由必须使用真实 Agent 的排序结果，而不是日期筛选结果。"""

    agent = _相似匹配Agent(["smart-case-1", "smart-case-2"])
    runner = _执行器(agent)

    chunk_ids, case_ids, answer_case_ids = runner._执行检索(
        _样例外壳(),
        question="比较两个历史暴雨过程",
        route="similar_case_match",
        memory={},
    )

    assert chunk_ids == ["chunk-1"]
    assert case_ids == ["smart-case-1", "smart-case-2"]
    assert answer_case_ids == case_ids
    assert agent.requests[0].raw_query == "比较两个历史暴雨过程"
    assert agent.requests[0].include_images is False


def test_similar_route_falls_back_to_structured_cases_when_agent_fails() -> None:
    """相似 Agent 故障时保留结构化检索结果，避免整条评测链路失败。"""

    runner = _执行器(_失败相似匹配Agent())

    _chunk_ids, case_ids, answer_case_ids = runner._执行检索(
        _样例外壳(),
        question="比较两个历史暴雨过程",
        route="similar_case_match",
        memory={},
    )

    assert case_ids == ["fallback-case"]
    assert answer_case_ids == case_ids


def test_rag_similarity_question_uses_agent_ids_without_changing_route() -> None:
    """库内比较仍走 RAG，但应使用 Agent 的相似度排序补充个例证据。"""

    agent = _相似匹配Agent(["smart-case-1", "smart-case-2"])
    runner = _执行器(agent)

    _chunk_ids, case_ids, answer_case_ids = runner._执行检索(
        _样例外壳(),
        question="比较两个历史暴雨过程",
        route="rag",
        memory={},
    )

    assert case_ids == ["smart-case-1", "smart-case-2", "fallback-case"]
    assert answer_case_ids == case_ids
    assert agent.requests[0].raw_query == "比较两个历史暴雨过程"


def test_rag_factual_question_does_not_call_similar_agent() -> None:
    """普通事实问答不应为了比较意图之外的样例额外调用相似 Agent。"""

    agent = _相似匹配Agent(["smart-case-1"])
    runner = _执行器(agent)

    _chunk_ids, case_ids, answer_case_ids = runner._执行检索(
        _样例外壳(),
        question="2025年5月平均降水量是多少",
        route="rag",
        memory={},
    )

    assert case_ids == ["fallback-case"]
    assert answer_case_ids == case_ids
    assert agent.requests == []

def test_rag_keeps_multiple_structured_cases_only_in_answer_context() -> None:
    """多候选个例只进入回答上下文，不扰动实际证据排名。"""

    runner = _执行器(_相似匹配Agent([]))
    runner._结构化检索 = lambda _case, _memory: ["case-1", "case-2"]  # type: ignore[method-assign]

    _chunk_ids, case_ids, answer_case_ids = runner._执行检索(
        _样例外壳(),
        question="列出2025年4月的全部大风过程",
        route="rag",
        memory={},
    )

    assert case_ids == []
    assert answer_case_ids == ["case-1", "case-2"]
