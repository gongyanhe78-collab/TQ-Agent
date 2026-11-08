"""回答质量类确定性指标。"""

from __future__ import annotations

import re
import unicodedata


def _归一化文本(text: str) -> str:
    """统一全角字符、温度单位与空白，避免等价写法被判为事实缺失。"""

    normalized = unicodedata.normalize("NFKC", str(text or ""))
    # NFKC 会把“摄氏度”和“℃”展开成不同形式，这里统一归一为同一种写法。
    normalized = (
        normalized.replace("摄氏度", "摄氏度")
        .replace("攝氏度", "摄氏度")
        .replace("℃", "摄氏度")
        .replace("°C", "摄氏度")
        .replace("°Ｃ", "摄氏度")
    )
    return re.sub(r"\s+", "", normalized)


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

    normalized_answer = _归一化文本(answer)
    covered = 0
    for fact in required_facts:
        if not fact:
            covered += 1
            continue
        alternatives = [item for item in fact.split("|") if item.strip()]
        if "|" in fact and alternatives:
            satisfied = any(
                _归一化文本(alternative) in normalized_answer
                for alternative in alternatives
            )
        else:
            satisfied = _归一化文本(fact) in normalized_answer
        covered += 1 if satisfied else 0
    return covered / len(required_facts)


def forbidden_claim_count(answer: str, forbidden_claims: list[str]) -> int:
    """统计完整禁止短语出现的条数，不使用简单关键词包含判断。"""

    normalized_answer = _归一化文本(answer)
    return sum(
        1
        for claim in forbidden_claims
        if claim and _归一化文本(claim) in normalized_answer
    )


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
