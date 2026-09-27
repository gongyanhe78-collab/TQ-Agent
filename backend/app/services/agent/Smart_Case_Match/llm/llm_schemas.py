"""相似个例各类 LLM 任务的严格输出协议。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr


# 逐例提炼只有这两种来源可进入跨个例综合，节点、模型服务和审计共用同一口径。
VALID_REFERENCE_SOURCES = frozenset({"llm_valid", "llm_repaired"})


class StrictLlmModel(BaseModel):
    """统一拒绝模型临时增加的字段，防止输出协议静默漂移。"""

    model_config = ConfigDict(extra="forbid")


class EnumEvidence(StrictLlmModel):
    """自然语言枚举选择及其原文依据。"""

    value: StrictStr
    evidence: StrictStr = ""


class NaturalQueryOutput(StrictLlmModel):
    """自然语言解析模型的完整输出。"""

    process_name: StrictStr = ""
    date_expression: StrictStr = ""
    date_evidence: StrictStr = ""
    # 保留旧字段用于兼容历史模型输出；最终精确日期始终由规则层重新计算。
    start_date: StrictStr = ""
    end_date: StrictStr = ""
    date: StrictStr = ""
    disaster_types: list[EnumEvidence] = Field(default_factory=list)
    affected_areas: list[EnumEvidence] = Field(default_factory=list)
    observation_description: StrictStr = ""
    circulation_description: StrictStr = ""
    intensity_description: StrictStr = ""
    metric_descriptions: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    raw_query: StrictStr = ""


class CandidateAssessment(StrictLlmModel):
    """候选个例的动态指标兼容度，分数只用于降权。

    mechanism_score 和 intensity_score 保留是为了兼容已有模型和旧页面；新模型优先返回
    dimension_scores，由当前灾种配置决定具体指标，避免把所有灾种都按雨雪判断。
    """

    case_id: StrictStr
    dimension_scores: dict[StrictStr, StrictFloat | StrictInt] = Field(default_factory=dict)
    missing_dimensions: list[StrictStr] = Field(default_factory=list)
    metric_scores: dict[StrictStr, StrictFloat | StrictInt] = Field(default_factory=dict)
    missing_metrics: list[StrictStr] = Field(default_factory=list)
    mechanism_score: StrictFloat | StrictInt | None = Field(default=None, ge=0, le=1)
    intensity_score: StrictFloat | StrictInt | None = Field(default=None, ge=0, le=1)


class CandidateRerankOutput(StrictLlmModel):
    """候选个例 LLM 输出排序及机制、强度两项业务兼容度。"""

    ordered_case_ids: list[StrictStr] = Field(default_factory=list)
    candidate_assessments: list[CandidateAssessment]


class CaseReferencePoint(StrictLlmModel):
    """单条历史参考经验及其文字证据。"""

    text: StrictStr
    evidence_chunk_ids: list[StrictStr] = Field(default_factory=list)


class CaseReferenceOutput(StrictLlmModel):
    """单个历史个例提炼模型的输出。"""

    match_reasons: list[StrictStr] = Field(default_factory=list)
    reference_points: list[CaseReferencePoint] = Field(default_factory=list)
    similarities: list[StrictStr] = Field(default_factory=list)
    differences: list[StrictStr] = Field(default_factory=list)
    warning_references: list[StrictStr] = Field(default_factory=list)


class ForecastSummaryOutput(StrictLlmModel):
    """综合研判摘要。"""

    similarity_assessment: StrictStr
    core_features: list[StrictStr] = Field(default_factory=list)
    main_risk: StrictStr
    confidence: StrictFloat | StrictInt
    support_case_ids: list[StrictStr] = Field(default_factory=list)


class ForecastTipOutput(StrictLlmModel):
    """单条可执行预报提示。"""

    priority: StrictInt
    focus_object: StrictStr
    possible_bias: StrictStr
    suggested_action: StrictStr
    support_case_ids: list[StrictStr] = Field(default_factory=list)
    evidence_chunk_ids: list[StrictStr] = Field(default_factory=list)
    consensus_level: StrictStr
    confidence: StrictFloat | StrictInt


class ForecastSynthesisOutput(StrictLlmModel):
    """跨个例综合模型的完整输出。"""

    forecast_summary: ForecastSummaryOutput
    forecast_tips: list[ForecastTipOutput] = Field(default_factory=list)


def normalize_legacy_enum_items(value: Any) -> list[dict[str, str]]:
    """兼容旧模型的字符串数组，但统一转换后再进入严格协议校验。"""
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, str):
            result.append({"value": item, "evidence": ""})
        elif isinstance(item, dict):
            result.append(item)
    return result
