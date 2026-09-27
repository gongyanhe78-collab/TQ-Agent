"""
智能体执行计划模块
将意图分析结果转换为只读执行计划，包括工具调用顺序、
输入参数、执行单元和有向图边，支持复杂的多步骤查询处理。
"""
from __future__ import annotations

from backend.app.services.agent.models import (
    AgentAnalysis,
    ExecutionPlan,
    ExecutionUnit,
    GraphEdge,
    GraphExecutionPlan,
    GraphNode,
    IntentType,
    PlanStep,
    ToolRoute,
)


class AgentPlanner:
    """
    智能体执行计划器
    将合并后的意图分析结果转换为结构化的只读执行计划，
    确定工具调用顺序、输入参数和执行依赖关系。
    """

    def plan(self, question: str, analysis: AgentAnalysis) -> ExecutionPlan:
        """
        根据意图分析结果生成执行计划

        Args:
            question: 用户问题文本
            analysis: AgentAnalysis 意图分析结果

        Returns:
            ExecutionPlan 执行计划对象
        """
        intents = analysis.intent_types
        units = _execution_units(question, analysis, intents)
        graph = _graph_for_units(units)
        routes = self._ordered_routes(intents, analysis.required_tools)
        steps = [
            PlanStep(
                step_id=f"step-{index:02d}",
                tool=route,
                purpose=_purpose_for(route, intents),
                inputs=_inputs_for(route, analysis),
            )
            for index, route in enumerate(routes, start=1)
        ]
        return ExecutionPlan(
            question=question,
            intents=intents,
            slots=analysis.slots,
            steps=steps,
            forbidden_tools=analysis.forbidden_tools,
            risk_notes=analysis.safety.risk_notes,
            units=units,
            graph=graph,
        )

    def _ordered_routes(self, intents: list[IntentType], required_tools: list[ToolRoute]) -> list[ToolRoute]:
        ordered: list[ToolRoute] = []
        for route in (
            ToolRoute.STANDARD_CASES_FILTER,
            ToolRoute.STANDARD_CASES_AGGREGATE,
            ToolRoute.METRIC_DATA_QUERY,
            ToolRoute.METRIC_COMPARISON,
            ToolRoute.IMAGE_METADATA_SEARCH,
            ToolRoute.SIMILAR_CASE_MATCH,
            ToolRoute.DOCUMENT_CHUNK_RETRIEVE,
            ToolRoute.ANSWER_SYNTHESIS,
        ):
            if route in required_tools and route not in ordered:
                ordered.append(route)
        if not ordered:
            ordered.extend([ToolRoute.DOCUMENT_CHUNK_RETRIEVE, ToolRoute.ANSWER_SYNTHESIS])
        if IntentType.STATISTICAL_SUMMARY in intents and ToolRoute.TOP_K_CHUNK_RETRIEVE in ordered:
            ordered.remove(ToolRoute.TOP_K_CHUNK_RETRIEVE)
        return ordered


def _purpose_for(route: ToolRoute, intents: list[IntentType]) -> str:
    if route == ToolRoute.STANDARD_CASES_FILTER:
        return "按时间、灾种、地区等结构化条件定位标准化个例"
    if route == ToolRoute.STANDARD_CASES_AGGREGATE:
        return "对标准化个例做全量过滤后的统计或对比聚合"
    if route == ToolRoute.METRIC_DATA_QUERY:
        return "按指标、月份和区域查询结构化气象要素数据"
    if route == ToolRoute.METRIC_COMPARISON:
        return "对已取回的指标值做同口径比较，并检查缺失数据"
    if route == ToolRoute.IMAGE_METADATA_SEARCH:
        return "按图片类型、图注和关联个例查找可追溯图片证据"
    if route == ToolRoute.SIMILAR_CASE_MATCH:
        return "综合时空、灾种、区域和图像证据匹配相似历史个例"
    if route == ToolRoute.DOCUMENT_CHUNK_RETRIEVE:
        return "补充原文 chunk 证据，支持复盘和证据说明"
    if route == ToolRoute.ANSWER_SYNTHESIS:
        return "先回答用户问题，再组织证据、口径和风险提示"
    return "执行只读工具步骤"


def _inputs_for(route: ToolRoute, analysis: AgentAnalysis) -> dict:
    if route in {ToolRoute.METRIC_DATA_QUERY, ToolRoute.METRIC_COMPARISON}:
        return {
            "metrics": analysis.slots.metrics,
            "months": analysis.slots.months,
            "areas": analysis.slots.areas,
        }
    if route in {ToolRoute.STANDARD_CASES_FILTER, ToolRoute.STANDARD_CASES_AGGREGATE}:
        return {
            "months": analysis.slots.months,
            "disasters": analysis.slots.disasters,
            "areas": analysis.slots.areas,
        }
    if route == ToolRoute.IMAGE_METADATA_SEARCH:
        return {"image_type": analysis.slots.image_type, "months": analysis.slots.months}
    return {}


def _execution_units(question: str, analysis: AgentAnalysis, intents: list[IntentType]) -> list[ExecutionUnit]:
    if IntentType.COMPARATIVE_ANALYSIS in intents and analysis.slots.metrics:
        metric = analysis.slots.metrics[0]
        months = analysis.slots.months[:2]
        units = [
            ExecutionUnit(
                unit_id="intent_decomposition",
                intent=IntentType.COMPARATIVE_ANALYSIS,
                objective=f"识别比较对象、指标和月份：{metric}",
                tool=ToolRoute.ANSWER_SYNTHESIS,
                inputs={"question": question, "metric": metric, "months": months},
            )
        ]
        for month in months:
            units.append(
                ExecutionUnit(
                    unit_id=f"metric_data_query_{month}",
                    intent=IntentType.COMPARATIVE_ANALYSIS,
                    objective=f"查询{month}月{metric}",
                    tool=ToolRoute.METRIC_DATA_QUERY,
                    inputs={"metric": metric, "month": month},
                    depends_on=["intent_decomposition"],
                )
            )
        units.extend(
            [
                ExecutionUnit(
                    unit_id="metric_data_guard",
                    intent=IntentType.COMPARATIVE_ANALYSIS,
                    objective="判断指标数据是否足以支持同口径比较，不足时只保留内部审计状态",
                    tool=ToolRoute.METRIC_COMPARISON,
                    inputs={"metric": metric, "months": months},
                    depends_on=[f"metric_data_query_{month}" for month in months],
                ),
            ]
        )
        return units
    return [
        ExecutionUnit(
            unit_id=f"unit-{index:02d}",
            intent=intents[0] if intents else IntentType.GENERAL_RAG,
            objective=_purpose_for(route, intents),
            tool=route,
            inputs=_inputs_for(route, analysis),
            depends_on=[] if index == 1 else [f"unit-{index - 1:02d}"],
        )
        for index, route in enumerate(analysis.required_tools or [ToolRoute.DOCUMENT_CHUNK_RETRIEVE, ToolRoute.ANSWER_SYNTHESIS], start=1)
    ]


def _graph_for_units(units: list[ExecutionUnit]) -> GraphExecutionPlan:
    nodes = [
        GraphNode(node_id=unit.unit_id, node_type="tool" if unit.tool != ToolRoute.ANSWER_SYNTHESIS else "llm", label=unit.objective, unit_id=unit.unit_id)
        for unit in units
    ]
    if any(unit.unit_id == "metric_data_guard" for unit in units):
        nodes.append(GraphNode(node_id="answer_synthesis", node_type="llm", label="汇总有证据支持的同口径比较结果"))
    edges = []
    unit_ids = {unit.unit_id for unit in units}
    for unit in units:
        for dependency in unit.depends_on:
            if dependency in unit_ids:
                edges.append(GraphEdge(source=dependency, target=unit.unit_id, condition=_edge_condition(dependency, unit.unit_id)))
    if any(unit.unit_id == "metric_data_guard" for unit in units):
        edges.append(GraphEdge(source="metric_data_guard", target="answer_synthesis", condition="has_metric_data"))
        edges.append(GraphEdge(source="metric_data_guard", target="answer_synthesis", condition="missing_metric_data"))
    return GraphExecutionPlan(nodes=nodes, edges=edges)


def _edge_condition(source: str, target: str) -> str:
    if source.startswith("metric_data_query_") and target == "metric_data_guard":
        return "metric_data_collected"
    if source == "metric_data_guard" and target == "answer_synthesis":
        return "has_metric_data"
    return "always"
