"""
智能体数据模型模块
定义智能体路由系统使用的所有核心数据结构：意图类型、工具路由、
查询槽位、安全信号、意图假设、执行计划和执行图等。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class IntentType(str, Enum):
    """
    智能体路由支持的业务级用户意图类型枚举

    Values:
        EVIDENCE_SEARCH: 证据检索（图片、清单等）
        CASE_REVIEW: 个例复盘（单个过程详情）
        STATISTICAL_SUMMARY: 统计汇总（数量、分布）
        COMPARATIVE_ANALYSIS: 对比分析（指标对比、多过程对比）
        SIMILAR_CASE: 相似个例匹配
        DECISION_SUPPORT: 决策支持（服务提示）
        GENERAL_RAG: 通用文档检索（兜底）
    """

    EVIDENCE_SEARCH = "evidence_search"
    CASE_REVIEW = "case_review"
    STATISTICAL_SUMMARY = "statistical_summary"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    SIMILAR_CASE = "similar_case"
    DECISION_SUPPORT = "decision_support"
    GENERAL_RAG = "general_rag"


class ToolRoute(str, Enum):
    """
    执行计划中的逻辑工具路由枚举

    Values:
        STANDARD_CASES_FILTER: 标准化个例过滤
        STANDARD_CASES_AGGREGATE: 标准化个例聚合统计
        METRIC_DATA_QUERY: 指标数据查询
        METRIC_COMPARISON: 指标对比分析
        IMAGE_METADATA_SEARCH: 图片元数据检索
        DOCUMENT_CHUNK_RETRIEVE: 文档片段检索
        TOP_K_CHUNK_RETRIEVE: Top-K 文档检索（统计查询时禁用）
        SIMILAR_CASE_MATCH: 相似个例匹配
        ANSWER_SYNTHESIS: 答案合成
    """

    STANDARD_CASES_FILTER = "standard_cases_filter"
    STANDARD_CASES_AGGREGATE = "standard_cases_aggregate"
    METRIC_DATA_QUERY = "metric_data_query"
    METRIC_COMPARISON = "metric_comparison"
    IMAGE_METADATA_SEARCH = "image_metadata_search"
    DOCUMENT_CHUNK_RETRIEVE = "document_chunk_retrieve"
    TOP_K_CHUNK_RETRIEVE = "top_k_chunk_retrieve"
    SIMILAR_CASE_MATCH = "similar_case_match"
    ANSWER_SYNTHESIS = "answer_synthesis"


@dataclass
class QuerySlots:
    """
    从用户问题中提取的结构化槽位

    Attributes:
        months: 月份列表（如 [5, 6] 表示 5 月和 6 月）
        disasters: 灾害类型列表（如 ["暴雨", "强对流"]）
        areas: 区域列表（如 ["山西北部", "太原"]）
        metrics: 指标列表（如 ["平均降水量", "累计降水量"]）
        image_type: 图片类型（如 "radar", "satellite"）
        raw_date: 原始日期字符串（如 "5 月 10 日"）
    """

    months: list[int] = field(default_factory=list)
    disasters: list[str] = field(default_factory=list)
    areas: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    image_type: str = ""
    raw_date: str = ""


@dataclass
class SafetySignal:
    """
    查询的安全和权限评估

    Attributes:
        read_only: 是否为只读操作（True 表示安全）
        risk_notes: 风险提示列表（如检测到删除、重建等写操作）
    """

    read_only: bool = True
    risk_notes: list[str] = field(default_factory=list)


@dataclass
class IntentHypothesis:
    """
    单个分析器对某个意图的判断假设

    Attributes:
        intent: 识别出的意图类型
        confidence: 置信度（0-1）
        reason: 判断理由说明
        source: 识别来源（如 "rule" 表示规则匹配）
    """

    intent: IntentType
    confidence: float
    reason: str
    source: str


@dataclass
class AgentAnalysis:
    """
    所有意图分析链路的合并结果

    Attributes:
        question: 用户原始问题
        slots: 提取的结构化槽位
        hypotheses: 意图假设列表（按置信度排序）
        required_tools: 需要调用的工具路由列表
        forbidden_tools: 禁用的工具路由列表
        safety: 安全信号评估结果
    """

    question: str
    slots: QuerySlots
    hypotheses: list[IntentHypothesis] = field(default_factory=list)
    required_tools: list[ToolRoute] = field(default_factory=list)
    forbidden_tools: list[ToolRoute] = field(default_factory=list)
    safety: SafetySignal = field(default_factory=SafetySignal)

    @property
    def intent_types(self) -> list[IntentType]:
        result: list[IntentType] = []
        for hypothesis in sorted(self.hypotheses, key=lambda item: item.confidence, reverse=True):
            if hypothesis.intent not in result:
                result.append(hypothesis.intent)
        return result or [IntentType.GENERAL_RAG]


@dataclass
class PlanStep:
    """
    单个只读执行步骤

    Attributes:
        step_id: 步骤 ID（如 "step-01"）
        tool: 要调用的工具路由
        purpose: 该步骤的执行目的说明
        inputs: 该步骤的输入参数字典
    """

    step_id: str
    tool: ToolRoute
    purpose: str
    inputs: dict = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """
    工具执行前生成的有序执行计划

    Attributes:
        question: 用户原始问题
        intents: 识别出的意图列表
        slots: 提取的结构化槽位
        steps: 执行步骤列表
        forbidden_tools: 禁用的工具列表
        risk_notes: 风险提示列表
        units: 执行单元列表（用于复杂图执行）
        graph: 执行图（包含节点和条件边）
    """

    question: str
    intents: list[IntentType]
    slots: QuerySlots
    steps: list[PlanStep]
    forbidden_tools: list[ToolRoute] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    units: list["ExecutionUnit"] = field(default_factory=list)
    graph: "GraphExecutionPlan | None" = None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "intents": [intent.value for intent in self.intents],
            "slots": {
                "months": self.slots.months,
                "disasters": self.slots.disasters,
                "areas": self.slots.areas,
                "metrics": self.slots.metrics,
                "image_type": self.slots.image_type,
                "raw_date": self.slots.raw_date,
            },
            "steps": [
                {
                    "step_id": step.step_id,
                    "tool": step.tool.value,
                    "purpose": step.purpose,
                    "inputs": step.inputs,
                }
                for step in self.steps
            ],
            "forbidden_tools": [tool.value for tool in self.forbidden_tools],
            "risk_notes": self.risk_notes,
            "units": [unit.to_dict() for unit in self.units],
            "graph": self.graph.to_dict() if self.graph else {},
        }


@dataclass
class ExecutionUnit:
    """用户问题拆解后的一个最小执行单元。"""

    unit_id: str
    intent: IntentType
    objective: str
    tool: ToolRoute
    inputs: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "unit_id": self.unit_id,
            "intent": self.intent.value,
            "objective": self.objective,
            "tool": self.tool.value,
            "inputs": self.inputs,
            "depends_on": self.depends_on,
        }


@dataclass
class GraphNode:
    """LangGraph 风格执行图中的节点。"""

    node_id: str
    node_type: str
    label: str
    unit_id: str = ""

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "unit_id": self.unit_id,
        }


@dataclass
class GraphEdge:
    """LangGraph 风格执行图中的有向边。"""

    source: str
    target: str
    condition: str = "always"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "condition": self.condition,
        }


@dataclass
class GraphExecutionPlan:
    """包含节点和条件边的执行编排。"""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }


@dataclass
class EvidenceChunk:
    """
    答案合成使用的轻量级文档片段证据

    Attributes:
        chunk_id: 文档片段 ID
        source_pdf: 来源 PDF 文件名
        content: 片段内容文本
    """

    chunk_id: str
    source_pdf: str
    content: str
