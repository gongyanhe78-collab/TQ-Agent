"""回答质量类确定性指标。"""

from __future__ import annotations

import re
import unicodedata

from backend.evals.schemas.dataset import ExpectedBehavior


def _归一化文本(text: str) -> str:
    """统一渲染标记、全角字符、常见单位和空白，避免等价事实被误判。"""

    normalized = unicodedata.normalize("NFKC", str(text or ""))
    # Markdown 强调符不属于事实内容，去掉后再做子串匹配。
    normalized = (
        normalized.replace("**", "")
        .replace("__", "")
        .replace("`", "")
        .replace("~~", "")
    )
    # NFKC 会把“摄氏度”和“℃”展开成不同形式，这里统一归一为同一种写法。
    normalized = (
        normalized.replace("摄氏度", "摄氏度")
        .replace("攝氏度", "摄氏度")
        .replace("℃", "摄氏度")
        .replace("°C", "摄氏度")
        .replace("°Ｃ", "摄氏度")
    )
    # 常见气象单位统一为中文写法，避免同一事实仅因单位表示不同被误判。
    normalized = re.sub(
        r"(?<=\d)\s*(?:m\s*/\s*s|m·s-1|米\s*/\s*秒)\s*",
        "米每秒",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"(?<=\d)\s*mm\b", "毫米", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"(?<=\d)\s*cm\b", "厘米", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"(?<=\d)\s*km\b", "千米", normalized, flags=re.IGNORECASE)
    # “7月12-18日”“7月12日-18日”和“7月12日至18日”表达同一时间范围。
    normalized = re.sub(
        r"(\d{1,2})月(\d{1,2})日?\s*[-~～—至]\s*(\d{1,2})日?",
        r"\1月\2日至\3日",
        normalized,
    )
    normalized = normalized.replace("天气过程", "过程")
    return re.sub(r"\s+", "", normalized)

def _归一化事实文本(text: str) -> str:
    """在通用归一化基础上忽略字段分隔符和系词，便于核对标签与数值。"""

    normalized = _归一化文本(text)
    return (
        normalized.replace("：", "")
        .replace(":", "")
        .replace("为", "")
        .replace("是", "")
    )


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

    normalized_answer = _归一化事实文本(answer)
    covered = 0
    for fact in required_facts:
        if not fact:
            covered += 1
            continue
        alternatives = [item for item in fact.split("|") if item.strip()]
        if "|" in fact and alternatives:
            satisfied = any(
                _归一化事实文本(alternative) in normalized_answer
                for alternative in alternatives
            )
        else:
            satisfied = _归一化事实文本(fact) in normalized_answer
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
def behavior_satisfaction_rate(
    answer: str,
    *,
    actual_route: str,
    actual_evidence: list[str],
    no_result: bool,
    abstained: bool,
    behaviors: list[ExpectedBehavior],
) -> float:
    """计算结构化行为期望的满足率，每个行为要求其全部检查同时通过。"""

    if not behaviors:
        return 1.0

    evidence = set(actual_evidence)
    answer_text = _归一化文本(answer)

    def _检查通过(behavior: ExpectedBehavior) -> bool:
        for check in behavior.checks:
            values = [item for item in check.values if item]
            if check.kind == "route_equals" and actual_route not in values:
                return False
            if check.kind == "answer_contains_any" and not any(
                _归一化文本(item) in answer_text for item in values
            ):
                return False
            if check.kind == "answer_contains_all" and not all(
                _归一化文本(item) in answer_text for item in values
            ):
                return False
            if check.kind == "evidence_hit" and not (
                any(item in evidence for item in values) if values else bool(evidence)
            ):
                return False
            if check.kind == "no_result":
                expected = not values or values[0].lower() not in {"false", "0", "否"}
                if bool(no_result) != expected:
                    return False
            if check.kind == "abstained":
                expected = not values or values[0].lower() not in {"false", "0", "否"}
                if bool(abstained) != expected:
                    return False
        return True

    return sum(1 for behavior in behaviors if _检查通过(behavior)) / len(behaviors)
