"""
智能体意图分析模块
基于规则的确定性意图识别系统，支持多意图并行识别、槽位提取、
工具路由决策和安全风险检测。
"""
from __future__ import annotations

import re

from backend.app.services.agent.models import (
    AgentAnalysis,
    IntentHypothesis,
    IntentType,
    QuerySlots,
    SafetySignal,
    ToolRoute,
)
from backend.app.services.structured_qa import AREA_ALIASES, CITY_TERMS, DISASTER_TERMS, asks_for_image, parse_image_type


class AgentIntentAnalyzer:
    """
    智能体意图分析器
    运行基于规则的确定性意图分析链路，支持多意图并行识别和结果合并，
    提取查询中的结构化槽位并生成工具路由决策。
    """

    def analyze(self, question: str, history: list[dict] | None = None) -> AgentAnalysis:
        """
        分析用户问题的意图和结构化信息

        Args:
            question: 用户问题文本
            history: 对话历史（暂未使用）

        Returns:
            AgentAnalysis 对象，包含槽位、假设、工具需求和安全信号
        """
        slots = QuerySlots(
            months=_parse_months(question),
            disasters=_parse_disasters(question),
            areas=_parse_areas(question),
            metrics=_parse_metrics(question),
            image_type=parse_image_type(question) if asks_for_image(question) else "",
            raw_date=_parse_raw_date(question),
        )
        hypotheses = _rule_hypotheses(question, slots)
        required_tools, forbidden_tools = _tool_needs(hypotheses, slots)
        safety = _safety_signal(question)
        return AgentAnalysis(
            question=question,
            slots=slots,
            hypotheses=hypotheses,
            required_tools=required_tools,
            forbidden_tools=forbidden_tools,
            safety=safety,
        )


def _rule_hypotheses(question: str, slots: QuerySlots) -> list[IntentHypothesis]:
    hypotheses: list[IntentHypothesis] = []
    detail_request = _asks_for_case_detail(question)
    if detail_request:
        hypotheses.append(_hypothesis(IntentType.CASE_REVIEW, 0.96, "命中上下文个例逐个解释表达"))
    if (
        not detail_request
        and any(term in question for term in ("多少", "几次", "几个", "统计", "汇总", "分别", "分布", "最严重", "受灾", "灾情", "主要灾种", "哪个月份", "缺少图片", "缺图", "哪些时间", "共同特征"))
    ):
        hypotheses.append(_hypothesis(IntentType.STATISTICAL_SUMMARY, 0.95, "命中统计/计数表达"))
    if any(term in question for term in ("对比", "比较", "差异", "不同", "相比", "区别")):
        reason = "命中指标对比表达" if slots.metrics else "命中对比分析表达"
        hypotheses.append(_hypothesis(IntentType.COMPARATIVE_ANALYSIS, 0.94 if slots.metrics else 0.92, reason))
    if slots.image_type or any(term in question for term in ("哪些", "清单", "列表", "有没有", "有无")):
        hypotheses.append(_hypothesis(IntentType.EVIDENCE_SEARCH, 0.88, "命中证据清单或图片条件"))
    if (
        not any(hypothesis.intent in {IntentType.STATISTICAL_SUMMARY, IntentType.COMPARATIVE_ANALYSIS, IntentType.CASE_REVIEW} for hypothesis in hypotheses)
        and any(term in question for term in ("复盘", "分析", "过程", "主要特征", "环流背景"))
    ):
        hypotheses.append(_hypothesis(IntentType.CASE_REVIEW, 0.72, "命中单过程分析表达"))
    if any(term in question for term in ("相似", "像哪些", "类似", "历史过程", "历史个例", "参考个例")):
        hypotheses.append(_hypothesis(IntentType.SIMILAR_CASE, 0.9, "命中相似历史个例表达"))
    if any(term in question for term in ("今天", "当前", "现在", "重点关注", "关注什么", "服务提示", "决策")):
        hypotheses.append(_hypothesis(IntentType.DECISION_SUPPORT, 0.88, "命中服务决策或当前过程表达"))
        hypotheses.append(_hypothesis(IntentType.SIMILAR_CASE, 0.78, "决策提示需要参考相似历史个例"))
    if not hypotheses:
        hypotheses.append(_hypothesis(IntentType.GENERAL_RAG, 0.5, "未命中结构化业务意图"))
    return hypotheses


def _hypothesis(intent: IntentType, confidence: float, reason: str) -> IntentHypothesis:
    return IntentHypothesis(intent=intent, confidence=confidence, reason=reason, source="rule")


def _asks_for_case_detail(question: str) -> bool:
    return any(
        term in question
        for term in (
            "分别详细讲",
            "分别详细解释",
            "分别解释",
            "分别说明",
            "分别讲",
            "逐个解释",
            "逐个说明",
            "逐条解释",
            "逐条说明",
            "详细讲讲",
            "详细解释",
            "详细说明",
            "展开讲",
            "特点分别",
            "分别有哪些特点",
            "特点是什么",
        )
    )


def _tool_needs(hypotheses: list[IntentHypothesis], slots: QuerySlots) -> tuple[list[ToolRoute], list[ToolRoute]]:
    intents = {hypothesis.intent for hypothesis in hypotheses}
    required: list[ToolRoute] = []
    forbidden: list[ToolRoute] = []
    if intents & {
        IntentType.STATISTICAL_SUMMARY,
        IntentType.EVIDENCE_SEARCH,
        IntentType.CASE_REVIEW,
        IntentType.COMPARATIVE_ANALYSIS,
    }:
        required.append(ToolRoute.STANDARD_CASES_FILTER)
    if IntentType.COMPARATIVE_ANALYSIS in intents and slots.metrics:
        required.append(ToolRoute.METRIC_DATA_QUERY)
        required.append(ToolRoute.METRIC_COMPARISON)
    if IntentType.STATISTICAL_SUMMARY in intents or (IntentType.COMPARATIVE_ANALYSIS in intents and not slots.metrics):
        required.append(ToolRoute.STANDARD_CASES_AGGREGATE)
    if IntentType.EVIDENCE_SEARCH in intents or slots.image_type or IntentType.DECISION_SUPPORT in intents:
        required.append(ToolRoute.IMAGE_METADATA_SEARCH)
    if IntentType.CASE_REVIEW in intents or IntentType.EVIDENCE_SEARCH in intents:
        required.append(ToolRoute.DOCUMENT_CHUNK_RETRIEVE)
    if IntentType.SIMILAR_CASE in intents or IntentType.DECISION_SUPPORT in intents:
        required.append(ToolRoute.SIMILAR_CASE_MATCH)
    required.append(ToolRoute.ANSWER_SYNTHESIS)
    if IntentType.STATISTICAL_SUMMARY in intents:
        forbidden.append(ToolRoute.TOP_K_CHUNK_RETRIEVE)
    return _dedupe_routes(required), _dedupe_routes(forbidden)


def _safety_signal(question: str) -> SafetySignal:
    risk_notes = []
    if any(term in question for term in ("删除", "重建", "刷新", "入库", "上传", "覆盖")):
        risk_notes.append("query_mentions_write_operation")
    return SafetySignal(read_only=True, risk_notes=risk_notes)


def _parse_months(text: str) -> list[int]:
    months: set[int] = set()
    for start, end in re.findall(r"(\d{1,2})\s*[-~～至到]\s*(\d{1,2})\s*月", text):
        start_month, end_month = int(start), int(end)
        if 1 <= start_month <= end_month <= 12:
            months.update(range(start_month, end_month + 1))
    for month in re.findall(r"(\d{1,2})\s*月", text):
        value = int(month)
        if 1 <= value <= 12:
            months.add(value)
    return sorted(months)


def _parse_raw_date(text: str) -> str:
    range_match = re.search(r"(\d{1,2}\s*月\s*\d{1,2}\s*[-~～至到]\s*\d{1,2}\s*日?)", text)
    if range_match:
        return re.sub(r"\s+", "", range_match.group(1))
    month_day = re.search(r"(\d{1,2}\s*月\s*\d{1,2}\s*日?)", text)
    if month_day:
        return re.sub(r"\s+", "", month_day.group(1))
    month_only = re.search(r"(\d{1,2}\s*月)", text)
    return re.sub(r"\s+", "", month_only.group(1)) if month_only else ""


def _parse_disasters(text: str) -> list[str]:
    return [term for term in DISASTER_TERMS if term in text]


def _parse_metrics(text: str) -> list[str]:
    metrics = []
    if any(term in text for term in ("平均降水量", "平均雨量", "平均降水")):
        metrics.append("平均降水量")
    elif any(term in text for term in ("累计降水量", "累计雨量", "总降水量")):
        metrics.append("累计降水量")
    elif "降水量" in text or "雨量" in text:
        metrics.append("降水量")
    return metrics


def _parse_areas(text: str) -> list[str]:
    areas = [area for area in AREA_ALIASES if area in text]
    areas.extend(city for city in CITY_TERMS if city in text)
    if "山西" in text and not any(area.startswith("山西") for area in areas):
        areas.append("山西")
    return _dedupe_strings(areas)


def _dedupe_routes(values: list[ToolRoute]) -> list[ToolRoute]:
    result: list[ToolRoute] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
