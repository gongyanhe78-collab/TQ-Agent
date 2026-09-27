"""
答案合成模块
将检索到的证据片段合成为结构化、易读的答案，支持文档片段证据整理、
指标对比分析和缺失数据说明，避免直接返回原始检索片段。
"""
from __future__ import annotations

import re

from backend.app.services.agent.models import EvidenceChunk


class AnswerSynthesizer:
    """
    答案合成器
    围绕用户问题格式化答案，而不是直接返回原始检索片段，
    支持文档证据整理和指标对比分析两种输出模式。
    """

    def document_fallback(self, question: str, chunks: list[EvidenceChunk]) -> str:
        """
        生成文档片段证据整理答案

        Args:
            question: 用户问题文本
            chunks: EvidenceChunk 证据片段列表

        Returns:
            格式化的答案字符串
        """
        topic = _compact_question(question)
        if not chunks:
            lines = [
                "结论：当前没有检索到足够证据，不能直接回答这个问题。",
                f"说明：{_evidence_gap(question) or '当前本地知识库没有足够的相关材料。'}",
            ]
            return "\n".join(lines)

        findings = [_clean_content(chunk.content) for chunk in chunks[:3]]
        lines = [
            f"结论：{_local_conclusion(topic, findings)}",
        ]
        # 只有确实影响结论可靠性的口径限制才提示，普通回答不再机械附加“材料缺口”。
        limitation = _evidence_gap(question)
        if limitation:
            lines.append(f"口径说明：{limitation}")
        lines.append(_source_line(chunks))
        return "\n".join(lines)

    def metric_comparison_from_chunks(self, question: str, chunks: list[EvidenceChunk]) -> str:
        """
        从文档片段中提取指标数值并生成对比分析答案

        Args:
            question: 用户问题文本
            chunks: EvidenceChunk 证据片段列表

        Returns:
            格式化的指标对比答案字符串
        """
        metric = _metric_from_question(question)
        extracted = _extract_metric_values(question, chunks, metric)
        if len(extracted) >= 2:
            first, second = extracted[0], extracted[1]
            diff = second["value"] - first["value"]
            if diff > 0:
                relation = f"{second['month']}月高于{first['month']}月"
            elif diff < 0:
                relation = f"{second['month']}月低于{first['month']}月"
            else:
                relation = f"{second['month']}月与{first['month']}月基本持平"
            unit = second["unit"] or first["unit"]
            diff_text = f"，差值约 {abs(diff):g}{unit}" if diff else ""
            fact_lines = _metric_fact_lines(metric, extracted[:4])
            lines = [
                (
                    f"结论：{first['month']}月{metric}约 {first['value']:g}{first['unit']}，"
                    f"{second['month']}月约 {second['value']:g}{second['unit']}；{relation}{diff_text}。"
                ),
                *fact_lines,
                "口径说明：这里是从相关材料中提取指标数值后做同口径比较；若涉及站点平均或区域平均，仍建议结合正式统计表进一步核验。",
                _source_line(chunks),
            ]
            return "\n".join(lines)
        if chunks:
            lines = [
                f"结论：已检索到{metric}相关材料，但现有片段还不足以同时抽出两个可直接比较月份的数值。",
                "口径说明：需要补齐两个比较月份的数值、单位和统计口径后，才能给出定量差异。",
                _source_line(chunks),
            ]
            return "\n".join(lines)
        return self.document_fallback(question, chunks)


def _compact_question(question: str) -> str:
    text = question.strip().rstrip("？?。.")
    for prefix in ("请", "帮我", "麻烦"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text or "这个问题"


def _clean_content(content: str, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", content).strip()
    if not text:
        return "原文片段为空"
    if len(text) <= limit:
        return text
    return text[:limit].rstrip("，；、。") + "。"


def _local_conclusion(topic: str, findings: list[str]) -> str:
    first_finding = findings[0] if findings else ""
    if not first_finding:
        return "现有证据不足以形成明确结论。"
    if any(term in topic for term in ("哪个", "最", "多少", "几", "排名", "排行")):
        return "现有材料只能提供线索，暂时不足以单独支撑精确排序或统计结论。"
    return f"从现有材料看，{first_finding}"


def _evidence_gap(question: str) -> str:
    if any(term in question for term in ("受灾", "最严重", "灾情", "损失")):
        return "现有标准化个例只能统计天气过程和影响线索，不能据此判断实际经济损失。"
    # 普通知识问答不再固定输出模板化的“材料缺口”段落。
    return ""


def _metric_from_question(question: str) -> str:
    if any(term in question for term in ("平均降水量", "平均雨量", "平均降水")):
        return "平均降水量"
    if any(term in question for term in ("累计降水量", "累计雨量", "总降水量")):
        return "累计降水量"
    if "雨量" in question:
        return "雨量"
    return "降水量"


def _extract_metric_values(question: str, chunks: list[EvidenceChunk], metric: str) -> list[dict]:
    months = _months_from_text(question)
    values = []
    seen = set()
    for chunk in chunks:
        for month in months:
            value = _metric_value_for_month(chunk.content, month, metric)
            if value is None:
                continue
            key = (month, chunk.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            values.append(
                {
                    "month": month,
                    "value": value["value"],
                    "unit": value["unit"],
                    "source_pdf": chunk.source_pdf,
                    "chunk_id": chunk.chunk_id,
                }
            )
    values.sort(key=lambda item: months.index(item["month"]) if item["month"] in months else 999)
    return values


def _metric_value_for_month(text: str, month: int, metric: str) -> dict | None:
    compact = re.sub(r"\s+", "", text)
    metric_terms = [metric]
    if metric == "降水量":
        metric_terms.extend(["平均降水量", "累计降水量", "雨量"])
    for term in metric_terms:
        pattern = rf"{month}月[^。；；\n]{{0,60}}?{re.escape(term)}[^0-9\d]{{0,12}}(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>毫米|mm|MM)"
        match = re.search(pattern, compact)
        if match:
            return {"value": float(match.group("value")), "unit": _normalize_unit(match.group("unit"))}
        pattern_after = rf"{month}月[^。；；\n]{{0,60}}?(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>毫米|mm|MM)[^。；；\n]{{0,20}}?{re.escape(term)}"
        match = re.search(pattern_after, compact)
        if match:
            return {"value": float(match.group("value")), "unit": _normalize_unit(match.group("unit"))}
    return None


def _months_from_text(text: str) -> list[int]:
    months = []
    for value in re.findall(r"(\d{1,2})\s*月", text):
        month = int(value)
        if 1 <= month <= 12 and month not in months:
            months.append(month)
    chinese_months = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "十一": 11,
        "十二": 12,
    }
    for token, month in sorted(chinese_months.items(), key=lambda item: len(item[0]), reverse=True):
        if f"{token}月" in text and month not in months:
            months.append(month)
    return months


def _normalize_unit(unit: str) -> str:
    return "毫米" if unit.lower() == "mm" else unit


def _metric_fact_lines(metric: str, items: list[dict]) -> list[str]:
    # 这里只保留自然语言事实，不暴露内部 chunk 标识。
    lines = []
    for item in items:
        lines.append(f"补充：{item['month']}月{metric}约为{item['value']:g}{item['unit']}。")
    return lines


def _source_line(chunks: list[EvidenceChunk]) -> str:
    # 来源单独放在末尾，避免正文出现工程化痕迹。
    seen = []
    for chunk in chunks:
        if chunk.source_pdf and chunk.source_pdf not in seen:
            seen.append(chunk.source_pdf)
    if not seen:
        return "材料来源于：当前未记录来源文件。"
    return f"材料来源于：{'、'.join(seen)}。"
