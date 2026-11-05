"""相似个例智能匹配体的 LangGraph 拓扑。"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import SmartCaseGraphNodes
from ..schemas import SmartCaseState


def build_smart_case_graph(nodes: SmartCaseGraphNodes):
    """按"选得准，再用得好"的顺序编译 StateGraph。"""
    builder = StateGraph(SmartCaseState)

    # -- 第一阶段：选得准（召回与排序） --
    # 输入标准化：解析用户提交的灾种、区域、日期等字段，
    # 清洗归一化后写入 state.query，作为后续检索的统一查询条件。
    builder.add_node("normalize_input", nodes.normalize_input)

    # 结构化召回：根据灾种、区域、时间范围等结构化条件，
    # 从标准个例库中筛选出符合条件的候选个例集（粗筛）。
    builder.add_node("structured_recall", nodes.structured_recall)

    # 语义补充召回：对实况/环流等文本描述做向量检索，
    # 从段落级向量库中召回语义相似的个例，并与结构化结果合并去重。
    builder.add_node("semantic_supplement", nodes.semantic_supplement)

    # 排序与筛选：对候选个例做综合打分（结构相似度 + 语义相似度 + 多样性），
    # 选出 top_n 个最相似且有差异的代表性个例。
    builder.add_node("rank_and_select", nodes.rank_and_select)

    # -- 第二阶段：用得好（富化与研判） --
    # 个例富化：为选中的个例补充详细信息，
    # 包括关键数据、图表证据、文档图片等，让结果可直接用于业务分析。
    builder.add_node("enrich_cases", nodes.enrich_cases)

    # 经验提炼：从相似个例的历史报告中提取可复用的经验要点、
    # 预报难点和参考结论，形成 case_references 列表。
    builder.add_node("extract_case_references", nodes.extract_case_references)

    # 综合研判：基于相似个例的共性特征和趋势，
    # 生成针对本次过程的预报提示和业务关注点（forecast_summary + forecast_tips）。
    builder.add_node("synthesize_forecast_tips", nodes.synthesize_forecast_tips)

    # 结果组装：将所有产出整合为 final_output，
    # 对齐 SmartCaseMatchResponse 协议，做最终校验后输出。
    builder.add_node("assemble_result", nodes.assemble_result)

    builder.add_edge(START, "normalize_input")
    builder.add_edge("normalize_input", "structured_recall")
    builder.add_edge("structured_recall", "semantic_supplement")
    builder.add_edge("semantic_supplement", "rank_and_select")
    builder.add_edge("rank_and_select", "enrich_cases")
    builder.add_edge("enrich_cases", "extract_case_references")
    builder.add_edge("extract_case_references", "synthesize_forecast_tips")
    builder.add_edge("synthesize_forecast_tips", "assemble_result")
    builder.add_edge("assemble_result", END)
    return builder.compile()
