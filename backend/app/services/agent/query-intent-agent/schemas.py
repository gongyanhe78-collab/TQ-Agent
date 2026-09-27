"""意图识别智能体的请求与响应协议。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


IntentName = Literal["rag", "multidim_search", "similar_case_match", "clarify"]
TaskType = Literal[
    "knowledge_qa",
    "case_identification",
    "case_listing",
    "case_analysis",
    "forecast_warning_lookup",
    "aggregate_statistics",
    "metric_aggregation",
    "comparison",
    "impact_area_lookup",
    "image_lookup",
    "similar_case_match",
    "report_generation",
    "clarification",
]
TaskRoute = Literal["rag", "multidim_search", "similar_case_match", "clarify"]


class TaskPlanItem(BaseModel):
    """一个可独立检索、执行和审计的子任务。"""

    id: str = Field(description="任务编号，例如 task_1")
    type: TaskType = Field(description="白名单任务类型")
    route: TaskRoute = Field(description="执行该任务的 Agent 路由")
    question: str = Field(min_length=1, max_length=6000, description="该子任务的完整问题")
    depends_on: list[str] = Field(default_factory=list, description="必须先完成的任务编号")
    evidence_scope: str = Field(default="vector", description="该任务的证据范围策略")
    conditions: dict[str, Any] = Field(
        default_factory=dict,
        description="由意图模型归一化并经本地校验的可执行筛选条件",
    )


class IntentRouteRequest(BaseModel):
    """统一聊天页提交给意图识别智能体的单轮消息。"""

    message: str = Field(min_length=1, max_length=6000, description="用户本轮原始消息")
    previous_intent: IntentName | None = Field(
        default=None,
        description="统一页面最近一次实际执行的意图，仅用于理解承接表达",
    )
    memory_context: dict[str, Any] = Field(
        default_factory=dict,
        description="当前 session_id 的分层摘要，不包含其他聊天窗口内容",
    )


class KnowledgeRange(BaseModel):
    """从本地标准化个例动态计算出的知识库时间范围。"""

    start_date: str = Field(default="", description="知识库最早个例日期")
    end_date: str = Field(default="", description="知识库最晚个例日期")
    case_count: int = Field(default=0, ge=0, description="参与计算的标准化个例数量")
    source: str = Field(default="standard_cases.json", description="时间范围的数据来源")


class IntentRouteResponse(BaseModel):
    """经过规则、模型和白名单校验后的最终分发决定。"""

    intent: IntentName = Field(description="最终意图，只允许主 RAG、多维报告、相似匹配或澄清")
    confidence: float = Field(ge=0.0, le=1.0, description="最终分发置信度")
    normalized_message: str = Field(description="完成中文日期归一化后的下游输入")
    original_message: str = Field(default="", description="用户本轮未经改写的原始消息")
    context_related: bool = Field(default=False, description="本轮问题是否承接当前会话历史")
    relation_type: str = Field(default="standalone", description="与历史内容的关联类型")
    reference_type: Literal["none", "knowledge_case", "external_process", "aggregate_result"] = Field(
        default="none",
        description="本轮实际引用的实体来源类型",
    )
    context_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="上下文关联判断置信度")
    referenced_case_ids: list[str] = Field(default_factory=list, description="从会话记忆中解析出的个例 ID")
    knowledge_case_ids: list[str] = Field(default_factory=list, description="经当前标准个例库复核后仍有效的个例 ID")
    reference_clues: list[str] = Field(default_factory=list, description="用于定位历史实体的日期、标题等线索")
    context_message_ids: list[int] = Field(default_factory=list, description="本轮问题引用的历史消息 ID")
    context_resolution_source: str = Field(default="none", description="问题重写来自模型、规则或无上下文")
    conditions: dict[str, Any] = Field(default_factory=dict, description="确定性规则提取的初步条件")
    reason: str = Field(default="", description="最终决定的简短依据")
    need_clarification: bool = Field(default=False, description="是否需要用户选择目标智能体")
    clarification_question: str = Field(default="", description="低置信度时展示的追问")
    suggested_intent: Literal["rag", "multidim_search", "similar_case_match"] | None = Field(
        default=None,
        description="存在倾向但仍需确认时的建议意图",
    )
    target_agent: str = Field(default="", description="实际准备分发到的业务智能体名称")
    target_endpoint: str = Field(default="", description="前端继续调用的原业务入口")
    routing_source: str = Field(default="rule", description="最终决定来自规则、模型或上下文")
    rag_strategy: Literal["structured", "vector"] = Field(
        default="vector",
        description="主 RAG 内部策略：标准个例聚合或文档向量检索",
    )
    tasks: list[TaskPlanItem] = Field(
        default_factory=list,
        description="本轮问题拆解后的任务计划；旧客户端可继续只使用 intent",
    )
    knowledge_range: KnowledgeRange = Field(description="本轮判断采用的知识库时间范围")
    audit_id: str = Field(default="", description="本轮意图审计编号")
