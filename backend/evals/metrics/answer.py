"""回答质量类确定性指标。"""

from __future__ import annotations


def _稳定去重(items: list[str]) -> list[str]:
    """按出现顺序去重，保持引用集合稳定。"""

    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def required_fact_coverage(answer: str, required_facts: list[str]) -> float:
    """计算人工审核过的确定性子串或任选其一的覆盖比例。"""

    if not required_facts:
        return 1.0

    covered = 0
    for fact in required_facts:
        if not fact:
            covered += 1
            continue
        alternatives = [item.strip() for item in fact.split("|") if item.strip()]
        if "|" in fact and alternatives:
            satisfied = any(alternative in answer for alternative in alternatives)
        else:
            satisfied = fact in answer
        covered += 1 if satisfied else 0
    return covered / len(required_facts)


def forbidden_claim_count(answer: str, forbidden_claims: list[str]) -> int:
    """统计完整禁止短语出现的条数，不使用简单关键词包含判断。"""

    return sum(1 for claim in forbidden_claims if claim and claim in answer)


def citation_support_rate(citations: list[str], retrieved_ids: list[str]) -> float:
    """计算引用中有检索证据支撑的比例，空引用视为没有无效引用。"""

    unique_citations = _稳定去重(citations)
    if not unique_citations:
        return 1.0
    retrieved = set(retrieved_ids)
    supported = sum(1 for citation in unique_citations if citation in retrieved)
    return supported / len(unique_citations)


def abstention_accuracy(abstained: bool, expected_abstain: bool) -> float:
    """计算拒答预测与期望是否一致。"""

    return 1.0 if bool(abstained) == bool(expected_abstain) else 0.0
