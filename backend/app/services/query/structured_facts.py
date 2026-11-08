"""从已命中证据中抽取可逐字核验的结构化事实。"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class StructuredFact:
    """一条带证据来源的原文事实。"""

    evidence_id: str
    text: str


_事实分隔符 = re.compile(r"(?<=[。！？；])\s*|\r?\n+")
_包含数字 = re.compile(r"\d")
_事实提示词 = (
    "年", "月", "日", "时", "分", "站", "市", "县", "区", "省",
    "毫米", "米每秒", "m/s", "℃", "级", "个", "次", "占", "%",
    "距平", "平均", "最大", "最小", "最高", "最低", "累计",
)


def _清理事实句(text: str) -> str:
    """压缩段落内空白，同时保留原始数值和单位写法。"""
    return re.sub(r"\s+", " ", str(text or "")).strip(" \t。；")


def extract_structured_facts(
    text: str,
    *,
    evidence_id: str = "",
    limit: int = 12,
) -> list[StructuredFact]:
    """按标点切分原文，保留包含数值及业务提示词的短事实句。"""
    max_items = max(int(limit), 0)
    if max_items == 0:
        return []

    facts: list[StructuredFact] = []
    seen: set[str] = set()
    for segment in _事实分隔符.split(str(text or "")):
        sentence = _清理事实句(segment)
        if len(sentence) < 6 or not _包含数字.search(sentence):
            continue
        if not any(term in sentence for term in _事实提示词):
            continue
        if len(sentence) > 360:
            sentence = next(
                (
                    _清理事实句(part)
                    for part in re.split(r"[，,]", sentence)
                    if len(_清理事实句(part)) >= 6 and _包含数字.search(part)
                ),
                sentence[:360],
            )
        normalized = re.sub(r"\s+", "", sentence)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        facts.append(StructuredFact(evidence_id=evidence_id, text=sentence))
        if len(facts) >= max_items:
            break
    return facts


def render_structured_fact_block(
    facts: list[StructuredFact],
    *,
    max_facts: int = 24,
) -> str:
    """把结构化事实渲染成回答模型可逐项复用的证据块。"""
    if not facts:
        return ""
    lines = [
        "可核验结构化事实",
        "以下内容来自已命中证据原文；回答涉及数值、日期、站点和数量时必须逐项保留，"
        "不得省略、替换或四舍五入。",
    ]
    seen: set[tuple[str, str]] = set()
    for fact in facts:
        key = (fact.evidence_id, re.sub(r"\s+", "", fact.text))
        if not fact.text or key in seen:
            continue
        seen.add(key)
        source = f"[{fact.evidence_id}] " if fact.evidence_id else ""
        lines.append(f"- {source}{fact.text}")
        if len(seen) >= max(int(max_facts), 1):
            break
    return "\n".join(lines)
