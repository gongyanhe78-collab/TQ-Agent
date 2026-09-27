"""相似个例智能匹配体的数据协议。"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator


class SmartCaseMatchRequest(BaseModel):
    """页面提交的新灾害过程描述。"""

    process_name: str = Field(default="", max_length=120, description="新过程名称")
    start_date: str = Field(default="", max_length=40, description="过程开始日期")
    end_date: str = Field(default="", max_length=40, description="过程结束日期")
    date: str = Field(default="", max_length=80, description="自然语言日期补充")
    disaster_types: list[str] = Field(default_factory=list, description="灾害类型")
    affected_areas: list[str] = Field(default_factory=list, description="影响地市或区域")
    observation_description: str = Field(default="", max_length=6000, description="实况描述")
    circulation_description: str = Field(default="", max_length=6000, description="环流形势描述")
    intensity_description: str = Field(default="", max_length=3000, description="强度和持续时间描述")
    metric_descriptions: dict[str, str] = Field(default_factory=dict, description="按灾种识别出的指标及原文描述")
    raw_query: str = Field(default="", max_length=6000, description="自然语言补充描述")
    progress_id: str = Field(default="", max_length=80, description="页面进度跟踪编号")
    top_n: int = Field(default=3, ge=3, le=5, description="期望返回数量")
    diversity_mode: Literal["off", "moderate"] = Field(default="moderate", description="结果多样性控制")
    include_images: bool = Field(default=True, description="是否返回证据图片")

    @model_validator(mode="after")
    def validate_content(self) -> "SmartCaseMatchRequest":
        """保证请求至少包含灾种、地区或日期中的可用业务条件。"""
        if not (
            self.disaster_types
            or self.affected_areas
            or self.start_date
            or self.end_date
            or self.date
            or self.raw_query.strip()
        ):
            raise ValueError("请至少填写灾害类型、影响区域或过程日期")
        return self


class NaturalLanguageMatchRequest(BaseModel):
    """聊天页面提交的单轮自然语言过程描述。"""

    message: str = Field(min_length=2, max_length=6000, description="自然语言天气过程描述")
    progress_id: str = Field(default="", max_length=80, description="页面进度跟踪编号")


class SmartCaseMatchResponse(BaseModel):
    """对外返回的稳定结果协议。"""

    run_id: str = Field(description="本次匹配任务的唯一标识，用于进度查询与日志追踪")
    status: Literal["completed", "degraded", "failed"] = Field(
        description="任务状态：completed=正常完成，degraded=降级完成（数据部分缺失），failed=失败"
    )
    query_summary: dict[str, Any] = Field(
        description="用户输入的结构化摘要（灾种、区域、日期等），用于前端展示系统理解的查询条件"
    )
    matched_cases: list[dict[str, Any]] = Field(
        description="匹配到的历史相似个例列表，按相似度从高到低排序"
    )
    forecast_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="基于相似个例推导的预报结论摘要",
    )
    forecast_tips: list[dict[str, Any]] = Field(
        description="预报提示与业务关注点列表，如天气趋势、防御建议等"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="警告信息列表，如数据缺失、匹配质量下降等提示",
    )
    audit: dict[str, Any] = Field(
        default_factory=dict,
        description="审计与调试信息，包含各阶段耗时、模型版本等，便于排查问题",
    )


class SmartCaseState(TypedDict, total=False):
    """LangGraph 节点之间传递的内部状态。

    total=False 表示所有字段均为可选，各节点只读写自己关心的字段。
    这是类型提示用的 TypedDict，运行时就是普通 dict，不做校验。
    """

    run_id: str  # 本次任务唯一标识
    client_progress_id: str  # 客户端可选关联号，只用于兼容进度查询，不作为任务主键
    request: dict[str, Any]  # 原始请求参数（用户输入）
    query: dict[str, Any]  # 结构化查询条件（从 request 中解析/提取出的灾种、区域、日期等）
    all_cases: list[dict[str, Any]]  # 全部标准个例库数据
    all_chunks: list[dict[str, Any]]  # 全部段落级向量块（用于语义检索）
    candidate_cases: list[dict[str, Any]]  # 初筛后的候选个例（粗匹配结果）
    scored_cases: list[dict[str, Any]]  # 打分排序后的个例（带相似度分数）
    selected_cases: list[dict[str, Any]]  # 最终选中的 topN 个例
    candidate_rank_audit: dict[str, Any]  # 候选LLM评估、排序和缺失维度的内部审计快照
    enriched_cases: list[dict[str, Any]]  # 补充了详细信息的个例（如图片、关键数据等）
    case_references: list[dict[str, Any]]  # 个例参考文献/证据片段列表
    forecast_summary: dict[str, Any]  # 基于相似个例推导的预报结论摘要
    forecast_tips: list[dict[str, Any]]  # 预报提示与业务关注点
    final_output: dict[str, Any]  # 最终组装好的输出结果（对应 SmartCaseMatchResponse）
    warnings: list[str]  # 过程中产生的警告信息
    audit: dict[str, Any]  # 审计与调试信息（各阶段耗时、调用次数等）
    # 保留字段用于兼容现有节点调用；正常入口传 None，只统计耗时而不限制链路执行。
    deadline_monotonic: float | None
