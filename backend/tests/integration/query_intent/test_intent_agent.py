"""意图识别智能体的离线回归测试，不调用真实大模型接口。"""
from __future__ import annotations

import importlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


agent_module = importlib.import_module("backend.app.services.agent.query-intent-agent.agent")
schema_module = importlib.import_module("backend.app.services.agent.query-intent-agent.schemas")

JAN_11_ID = "山西省2025年1月天气过程总结-std-case-001"
JAN_14_ID = "山西省2025年1月天气过程总结-std-case-002"
JAN_23_ID = "山西省2025年1月天气过程总结-std-case-003"


class UnavailableLlm:
    """让测试固定走规则分支，避免访问外部模型服务。"""

    def is_available(self) -> bool:
        return False


class ContextAwareLlm:
    """模拟云端会话理解模型，验证模型判断仍需经过知识库实体校验。"""

    def is_available(self) -> bool:
        return True

    def answer_with_context(self, _prompt, _context, max_tokens=0):
        return json.dumps({
            "is_related": True,
            "relation_type": "follow_up_reference",
            "reference_type": "knowledge_case",
            "confidence": 0.98,
            "standalone_message": "详细分析1月23-26日雨雪寒潮大风天气过程的预报提示和灾害情况",
            "referenced_case_ids": [JAN_23_ID],
            "reference_clues": ["1月23日"],
            "reason": "本轮日期和指代共同指向上一轮第三个历史个例",
        }, ensure_ascii=False)


class CombinedCountingLlm:
    """记录合并意图请求次数，确认一次调用同时返回上下文和路由。"""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_context = []

    def is_available(self) -> bool:
        return True

    def answer_with_context(self, _prompt, context, max_tokens=0):
        self.call_count += 1
        self.last_context = context
        return json.dumps({
            "is_related": True,
            "relation_type": "follow_up_reference",
            "reference_type": "knowledge_case",
            "context_confidence": 0.98,
            "standalone_message": "详细分析1月23-26日雨雪寒潮大风天气过程",
            "referenced_case_ids": [JAN_23_ID],
            "reference_clues": ["1月23日"],
            "context_reason": "指向上一轮库内个例",
            "intent": "rag",
            "intent_confidence": 0.99,
            "intent_reason": "库内个例分析属于主RAG",
        }, ensure_ascii=False)


class MultiTaskLlm:
    """模拟意图模型返回三个不同证据需求，验证通用任务计划协议。"""

    def is_available(self) -> bool:
        return True

    def answer_with_context(self, _prompt, _context, max_tokens=0):
        return json.dumps({
            "is_related": False,
            "relation_type": "standalone",
            "reference_type": "none",
            "context_confidence": 1.0,
            "standalone_message": "分析2025年1月23-26日雨雪寒潮大风天气过程、查询预警并给出图片证据",
            "referenced_case_ids": [],
            "reference_clues": ["2025年1月23日"],
            "intent": "rag",
            "intent_confidence": 0.99,
            "intent_reason": "三个子任务均由主RAG完成",
            "tasks": [
                {"id": "task_1", "type": "case_analysis", "question": "分析2025年1月23-26日雨雪寒潮大风天气过程", "depends_on": []},
                {"id": "task_2", "type": "forecast_warning_lookup", "question": "查询该过程预报预警", "depends_on": ["task_1"]},
                {"id": "task_3", "type": "image_lookup", "question": "查询该过程图片证据", "depends_on": ["task_1"]},
            ],
        }, ensure_ascii=False)


class AnnualStatisticsLlm:
    """模拟模型把全年统计与逐项列举拆成共享个例集合的任务图。"""

    def is_available(self) -> bool:
        return True

    def answer_with_context(self, _prompt, _context, max_tokens=0):
        conditions = {
            "years": [2025],
            "disaster_types": ["高温"],
            "time_scope_type": "calendar_year",
            "output_fields": ["count", "title", "date_range"],
        }
        return json.dumps({
            "is_related": False,
            "relation_type": "standalone",
            "reference_type": "none",
            "context_confidence": 1.0,
            "standalone_message": "2025年全年有几次高温天气，分别是什么时候",
            "referenced_case_ids": [],
            "reference_clues": [],
            "intent": "rag",
            "intent_confidence": 0.99,
            "intent_reason": "统计并枚举标准个例",
            "query_scope": conditions,
            "tasks": [
                {"id": "find", "type": "case_identification", "question": "筛选2025年全年高温个例", "depends_on": [], "conditions": conditions},
                {"id": "count", "type": "aggregate_statistics", "question": "统计命中个例数量", "depends_on": ["find"], "conditions": conditions},
                {"id": "list", "type": "case_listing", "question": "列出命中个例名称和时间", "depends_on": ["find"], "conditions": conditions},
            ],
        }, ensure_ascii=False)


class AnnualMisclassifiedLlm(AnnualStatisticsLlm):
    """模拟模型误把枚举附加成比较任务，验证结构校验能够拒绝该任务。"""

    def answer_with_context(self, _prompt, _context, max_tokens=0):
        payload = json.loads(super().answer_with_context(_prompt, _context, max_tokens=max_tokens))
        payload["tasks"] = [
            {
                "id": "count",
                "type": "aggregate_statistics",
                "question": "统计2025年高温个例数量",
                "depends_on": [],
                "conditions": payload["query_scope"],
            },
            {
                "id": "wrong-comparison",
                "type": "comparison",
                "question": "分别是什么时候",
                "depends_on": [],
                "conditions": payload["query_scope"],
            },
        ]
        return json.dumps(payload, ensure_ascii=False)


class FeatureAnalysisLlm:
    """模拟模型把“特征”放在 output_fields 中，验证本地契约会补正文分析任务。"""

    def is_available(self) -> bool:
        return True

    def answer_with_context(self, _prompt, _context, max_tokens=0):
        conditions = {
            "years": [2025],
            "months": [3],
            "output_fields": ["count", "name", "date", "features"],
        }
        return json.dumps({
            "is_related": False,
            "relation_type": "standalone",
            "reference_type": "none",
            "context_confidence": 1.0,
            "standalone_message": "统计2025年3月灾害次数、名称、时间和特征",
            "referenced_case_ids": [],
            "reference_clues": [],
            "intent": "rag",
            "intent_confidence": 0.99,
            "intent_reason": "统计并分析命中的标准个例",
            "query_scope": conditions,
            "tasks": [
                {"id": "find", "type": "case_identification", "question": "筛选2025年3月个例", "depends_on": [], "conditions": conditions},
                {"id": "count", "type": "aggregate_statistics", "question": "统计命中个例数量", "depends_on": ["find"], "conditions": conditions},
                {"id": "list", "type": "case_listing", "question": "列出命中个例名称和时间", "depends_on": ["find"], "conditions": conditions},
            ],
        }, ensure_ascii=False)


class RequestedOutputsLlm:
    """模拟模型只返回回答目标，验证计划校验器能补齐缺失的正文任务。"""

    def is_available(self) -> bool:
        return True

    def answer_with_context(self, _prompt, _context, max_tokens=0):
        conditions = {"years": [2025], "months": [3]}
        return json.dumps({
            "is_related": False,
            "relation_type": "standalone",
            "reference_type": "none",
            "context_confidence": 1.0,
            "standalone_message": "统计2025年3月灾害次数、名称和特征",
            "intent": "rag",
            "intent_confidence": 0.99,
            "intent_reason": "需要统计、列举并分析个例正文",
            "requested_outputs": ["count", "case_list", "case_features"],
            "query_scope": conditions,
            # 故意只返回统计和清单任务，检查本地契约是否补齐 case_analysis。
            "tasks": [
                {"id": "find", "type": "case_identification", "question": "筛选2025年3月个例", "depends_on": [], "conditions": conditions},
                {"id": "count", "type": "aggregate_statistics", "question": "统计命中个例数量", "depends_on": ["find"], "conditions": conditions},
                {"id": "list", "type": "case_listing", "question": "列出命中个例名称和时间", "depends_on": ["find"], "conditions": conditions},
            ],
        }, ensure_ascii=False)


class ExternalProcessSemanticLlm:
    """模拟模型结合当前时间和完整目录识别库外新过程。"""

    def __init__(self, omit_first_intent: bool = False) -> None:
        self.omit_first_intent = omit_first_intent
        self.call_count = 0
        self.inputs: list[dict] = []

    def is_available(self) -> bool:
        return True

    def answer_with_context(self, _prompt, context, max_tokens=0):
        self.call_count += 1
        model_input = json.loads(context[0])
        self.inputs.append(model_input)
        if self.omit_first_intent and self.call_count == 1:
            return json.dumps({
                "is_related": False,
                "relation_type": "standalone",
                "reference_type": "external_process",
                "context_confidence": 0.97,
                "standalone_message": model_input["本轮问题"],
                "referenced_case_ids": [],
                "reference_clues": ["相对于当前业务时间的新近过程"],
                "context_reason": "该过程无法唯一对应标准个例目录",
            }, ensure_ascii=False)
        return json.dumps({
            "is_related": False,
            "relation_type": "standalone",
            "reference_type": "external_process",
            "context_confidence": 0.98,
            "standalone_message": model_input.get("本轮问题", "库外新过程分析预测"),
            "referenced_case_ids": [],
            "reference_clues": ["相对于当前业务时间的新近过程"],
            "context_reason": "日期和实体均不能唯一对应标准个例目录",
            "intent": "similar_case_match",
            "intent_confidence": 0.98,
            "intent_reason": "库外新过程需要历史相似个例支撑分析预测",
            "tasks": [{
                "id": "smart",
                "type": "similar_case_match",
                "question": model_input.get("本轮问题", "库外新过程分析预测"),
                "depends_on": [],
                "conditions": {},
            }],
        }, ensure_ascii=False)


class MetricAggregationLlm:
    """模拟模型把自然语言极值问题转换为通用指标聚合任务。"""

    def is_available(self) -> bool:
        return True

    def answer_with_context(self, _prompt, context, max_tokens=0, **_kwargs):
        model_input = json.loads(context[0])
        scope = {
            "years": [2025], "months": [1, 2, 3, 4, 5, 6, 7],
            "areas": ["山西"], "time_scope_type": "date_range",
            "start_date": "2025-01-01", "end_date": "2025-07-31",
            "requested_outputs": ["metric_extrema", "case_features"],
            "metric": "air_temperature.minimum", "operator": "min",
            "result_dimensions": ["value", "time", "location", "case_id", "chunk_id"],
        }
        return json.dumps({
            "is_related": False,
            "relation_type": "standalone",
            "reference_type": "none",
            "context_confidence": 0.99,
            "standalone_message": model_input["本轮问题"],
            "referenced_case_ids": [],
            "reference_clues": [],
            "context_reason": "独立的跨时段指标极值问题",
            "intent": "rag",
            "intent_confidence": 0.99,
            "intent_reason": "知识库指标聚合后分析获胜个例",
            "requested_outputs": ["metric_extrema", "case_features"],
            "query_scope": scope,
            "tasks": [
                {"id": "find", "type": "case_identification", "question": "定位时间范围", "depends_on": [], "conditions": scope},
                {"id": "metric", "type": "metric_aggregation", "question": "计算最低气温", "depends_on": ["find"], "conditions": scope},
                {"id": "analysis", "type": "case_analysis", "question": "分析极值过程", "depends_on": ["metric"], "conditions": scope},
            ],
        }, ensure_ascii=False)


class RepairableMisroutedMetricLlm:
    """先模拟错误重型路由，再按契约修复成通用指标任务。"""

    def __init__(self) -> None:
        self.call_count = 0

    def is_available(self) -> bool:
        return True

    def answer_with_context(self, _prompt, context, max_tokens=0, **_kwargs):
        self.call_count += 1
        model_input = json.loads(context[0])
        scope = {
            "years": [2025], "months": [1, 2, 3, 4, 5, 6, 7],
            "time_scope_type": "month_set",
        }
        if self.call_count == 1:
            return json.dumps({
                "is_related": False,
                "relation_type": "standalone",
                "reference_type": "none",
                "context_confidence": 1.0,
                "standalone_message": model_input["本轮问题"],
                "intent": "multidim_search",
                "intent_confidence": 1.0,
                "intent_reason": "误把多任务分析当成多维报告",
                "requested_outputs": ["count", "case_list", "case_features"],
                "query_scope": scope,
                "tasks": [
                    {"id": "find", "type": "case_identification", "question": "定位候选个例", "depends_on": [], "conditions": scope},
                    {"id": "analysis", "type": "case_analysis", "question": "分析对应过程", "depends_on": ["find"], "conditions": scope},
                ],
            }, ensure_ascii=False)
        metric_scope = {
            **scope,
            "requested_outputs": ["metric_extrema", "case_features"],
            "metric": "precipitation.maximum",
            "operator": "max",
            "result_dimensions": ["value", "time", "location", "case_id", "chunk_id"],
        }
        return json.dumps({
            "is_related": False,
            "relation_type": "standalone",
            "reference_type": "none",
            "context_confidence": 1.0,
            "standalone_message": model_input["本轮问题"],
            "intent": "rag",
            "intent_confidence": 0.99,
            "intent_reason": "指标极值及其所属过程分析属于主RAG",
            "requested_outputs": ["metric_extrema", "case_features"],
            "query_scope": metric_scope,
            "tasks": [
                {"id": "find", "type": "case_identification", "question": "定位候选个例", "depends_on": [], "conditions": metric_scope},
                {"id": "metric", "type": "metric_aggregation", "question": "计算最大降水量", "depends_on": ["find"], "conditions": metric_scope},
                {"id": "analysis", "type": "case_analysis", "question": "分析极值所属过程", "depends_on": ["metric"], "conditions": metric_scope},
            ],
        }, ensure_ascii=False)


class IntentAgentTest(unittest.TestCase):
    """覆盖白名单分发、时间边界和上一轮意图承接。"""

    def setUp(self) -> None:
        # 测试使用独立审计文件，避免污染运行目录中的审计记录。
        self.temp_dir = TemporaryDirectory()
        self.agent = agent_module.QueryIntentAgent(
            llm_client=UnavailableLlm(),
            audit_path=Path(self.temp_dir.name) / "audit.jsonl",
            case_provider=lambda: [
                {"case_id": JAN_11_ID, "title": "1月11日降雪天气过程", "date_range": "2025年1月11日"},
                {"case_id": JAN_14_ID, "title": "1月14日-15日寒潮天气过程", "date_range": "2025年1月14-15日"},
                {"case_id": JAN_23_ID, "title": "1月23-26日雨雪寒潮大风天气过程", "date_range": "2025年1月23-26日"},
                {"case_id": "case-mar-cold", "title": "3月寒潮天气过程", "date_range": "2025年3月1日"},
            ],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def route(self, message: str, previous_intent=None, memory_context=None):
        request = schema_module.IntentRouteRequest(
            message=message,
            previous_intent=previous_intent,
            memory_context=memory_context or {},
        )
        return self.agent.route(request)

    def test_knowledge_range_is_dynamic_local_data(self) -> None:
        knowledge = self.agent.knowledge_range()
        self.assertEqual(knowledge.start_date, "2025-01-11")
        self.assertEqual(knowledge.end_date, "2025-07-31")
        self.assertGreater(knowledge.case_count, 0)

    def test_historical_query_without_report_uses_structured_rag(self) -> None:
        result = self.route("检索2025年一月至三月暴雪个例")
        self.assertEqual(result.intent, "rag")
        self.assertEqual(result.rag_strategy, "structured")
        self.assertEqual(result.conditions["years"], [2025])
        self.assertEqual(result.conditions["months"], [1, 2, 3])

    def test_explicit_report_routes_multidim(self) -> None:
        result = self.route("帮我生成2025年三月暴雪个例分析报告")
        self.assertEqual(result.intent, "multidim_search")

    def test_current_process_routes_similar(self) -> None:
        result = self.route("当前有强冷空气大风和降雪，找类似过程")
        self.assertEqual(result.intent, "similar_case_match")

    def test_out_of_range_requires_analysis_before_similar(self) -> None:
        self.assertEqual(self.route("2026年8月强对流过程").intent, "rag")
        self.assertEqual(self.route("请分析2026年8月强对流过程的风险").intent, "similar_case_match")

    def test_ordinary_message_falls_back_to_vector_rag(self) -> None:
        result = self.route("帮我看看天气")
        self.assertFalse(result.need_clarification)
        self.assertEqual(result.intent, "rag")
        self.assertEqual(result.rag_strategy, "vector")

    def test_simple_month_count_does_not_call_sub_agent(self) -> None:
        result = self.route("三月份有几个灾害？")
        self.assertEqual(result.intent, "rag")
        self.assertEqual(result.rag_strategy, "structured")

    def test_resolves_two_january_cases_from_previous_turn(self) -> None:
        """“1月份的这两个”必须绑定上一轮两个一月个例，而不是重新全库猜测。"""
        refs = [
            {"case_id": JAN_14_ID, "title": "1月14日-15日寒潮天气过程", "date_range": "2025年1月14-15日"},
            {"case_id": JAN_23_ID, "title": "1月23-26日雨雪寒潮大风天气过程", "date_range": "2025年1月23-26日"},
            {"case_id": "case-mar-cold", "title": "3月寒潮天气过程", "date_range": "2025年3月1日"},
        ]
        memory = {
            "previous_intent": "rag",
            "previous_turn": {
                "user_message_id": 10,
                "assistant_message_id": 11,
                "case_refs": refs,
            },
            "available_case_refs": refs,
        }
        result = self.route("详细分析一下1月份的这两个", memory_context=memory)

        self.assertEqual(result.intent, "rag")
        self.assertEqual(result.rag_strategy, "vector")
        self.assertTrue(result.context_related)
        self.assertEqual(result.referenced_case_ids, [JAN_14_ID, JAN_23_ID])
        self.assertIn("1月14日-15日寒潮天气过程", result.normalized_message)
        self.assertIn("1月23-26日雨雪寒潮大风天气过程", result.normalized_message)
        self.assertEqual(result.context_message_ids, [10, 11])

    def test_single_date_follow_up_routes_to_scoped_vector_rag(self) -> None:
        """上一轮多个一月个例中追问 1 月 23 日时，只能锁定主库中的第三个个例。"""
        refs = [
            {"case_id": JAN_11_ID, "title": "1月11日降雪天气过程", "date_range": "2025年1月11日"},
            {"case_id": JAN_14_ID, "title": "1月14日-15日寒潮天气过程", "date_range": "2025年1月14-15日"},
            {"case_id": JAN_23_ID, "title": "1月23-26日雨雪寒潮大风天气过程", "date_range": "2025年1月23-26日"},
        ]
        memory = {"previous_turn": {"case_refs": refs}, "available_case_refs": refs}

        result = self.route("帮我详细分析一下1月23号的这个灾害情况", memory_context=memory)

        self.assertEqual(result.intent, "rag")
        self.assertEqual(result.rag_strategy, "vector")
        self.assertEqual(result.knowledge_case_ids, [JAN_23_ID])
        self.assertEqual(result.reference_type, "knowledge_case")
        self.assertIn("1月23-26日雨雪寒潮大风天气过程", result.normalized_message)

    def test_model_context_resolution_cannot_be_overridden_by_forecast_word(self) -> None:
        """模型正确锁定库内实体后，重写文本中的“预报提示”不能再把任务送入 Smart。"""
        refs = [
            {"case_id": JAN_11_ID, "title": "1月11日降雪天气过程", "date_range": "2025年1月11日"},
            {"case_id": JAN_23_ID, "title": "1月23-26日雨雪寒潮大风天气过程", "date_range": "2025年1月23-26日"},
        ]
        agent = agent_module.QueryIntentAgent(
            llm_client=ContextAwareLlm(),
            audit_path=Path(self.temp_dir.name) / "model-audit.jsonl",
            case_provider=self.agent.case_provider,
        )
        request = schema_module.IntentRouteRequest(
            message="帮我详细分析一下1月23号的这个灾害情况",
            memory_context={"previous_turn": {"case_refs": refs}, "available_case_refs": refs},
        )

        result = agent.route(request)

        self.assertEqual(result.context_resolution_source, "llm")
        self.assertEqual(result.intent, "rag")
        self.assertEqual(result.knowledge_case_ids, [JAN_23_ID])

    def test_context_resolution_and_routing_share_one_model_call(self) -> None:
        """一轮请求只能调用一次意图模型，并同时得到完整问题和目标路由。"""
        llm = CombinedCountingLlm()
        refs = [{
            "case_id": JAN_23_ID,
            "title": "1月23-26日雨雪寒潮大风天气过程",
            "date_range": "2025年1月23-26日",
        }]
        agent = agent_module.QueryIntentAgent(
            llm_client=llm,
            audit_path=Path(self.temp_dir.name) / "combined-audit.jsonl",
            case_provider=self.agent.case_provider,
        )
        request = schema_module.IntentRouteRequest(
            message="详细分析这个过程",
            memory_context={"previous_turn": {"case_refs": refs}, "available_case_refs": refs},
        )

        first = agent.route(request)
        second = agent.route(request)

        self.assertEqual(first.intent, "rag")
        self.assertEqual(first.referenced_case_ids, [JAN_23_ID])
        self.assertEqual(second.intent, "rag")
        self.assertEqual(llm.call_count, 1)
        sent_memory = json.loads(llm.last_context[0])["分层会话记忆"]
        self.assertLessEqual(len(sent_memory["available_case_refs"]), 20)

    def test_forecast_review_of_knowledge_case_stays_in_rag(self) -> None:
        """历史个例的预报效果属于库内证据分析，裸词“预报”不能触发 Smart。"""
        ref = {"case_id": JAN_23_ID, "title": "1月23-26日雨雪寒潮大风天气过程", "date_range": "2025年1月23-26日"}
        memory = {"previous_turn": {"case_refs": [ref]}, "available_case_refs": [ref]}

        result = self.route("分析这个历史过程的预报效果", memory_context=memory)

        self.assertEqual(result.intent, "rag")
        self.assertEqual(result.rag_strategy, "vector")
        self.assertEqual(result.referenced_case_ids, [JAN_23_ID])

    def test_compound_question_builds_three_dependent_tasks(self) -> None:
        """一个问题包含多个诉求时，应生成任意数量的白名单任务而不是单一意图。"""
        ref = {
            "case_id": JAN_23_ID,
            "title": "1月23-26日雨雪寒潮大风天气过程",
            "date_range": "2025年1月23-26日",
        }
        memory = {"previous_turn": {"case_refs": [ref]}, "available_case_refs": [ref]}

        result = self.route(
            "分析这个历史过程的成因、影响了哪些地区，并说明发生前有没有预报预警",
            memory_context=memory,
        )

        self.assertEqual(result.intent, "rag")
        self.assertEqual(
            [item.type for item in result.tasks],
            ["case_analysis", "forecast_warning_lookup", "impact_area_lookup"],
        )
        self.assertEqual(result.tasks[1].depends_on, ["task_1"])
        self.assertEqual(result.tasks[2].depends_on, ["task_1"])
        self.assertEqual(result.tasks[1].evidence_scope, "same_pdf_related")

    def test_model_compiles_extreme_value_as_metric_dataflow(self) -> None:
        """极值查询应保留指标契约，并让分析任务依赖聚合结果。"""
        agent = agent_module.QueryIntentAgent(
            llm_client=MetricAggregationLlm(),
            audit_path=Path(self.temp_dir.name) / "metric-audit.jsonl",
            case_provider=self.agent.case_provider,
        )
        result = agent.route(schema_module.IntentRouteRequest(
            message="2025年1月到7月山西最低气温是什么时候发生在哪个地方，并进行分析",
        ))
        self.assertEqual(
            [item.type for item in result.tasks],
            ["case_identification", "metric_aggregation", "case_analysis"],
        )
        self.assertEqual(result.tasks[1].conditions["metric"], "air_temperature.minimum")
        self.assertEqual(result.tasks[1].conditions["operator"], "min")
        self.assertEqual(result.tasks[2].depends_on, ["task_2", "task_1"])
        self.assertNotIn("disaster_types", result.tasks[0].conditions)

    def test_inconsistent_heavy_route_is_repaired_to_metric_dataflow(self) -> None:
        """模型把极值分析误判成报告时，应按结构契约修复而非进入多维确认。"""
        llm = RepairableMisroutedMetricLlm()
        agent = agent_module.QueryIntentAgent(
            llm_client=llm,
            audit_path=Path(self.temp_dir.name) / "metric-repair-audit.jsonl",
            case_provider=self.agent.case_provider,
        )

        result = agent.route(schema_module.IntentRouteRequest(
            message="2025年1月至7月最大降水量是多少，发生在什么时间和地点，并分析对应过程",
        ))

        self.assertEqual(llm.call_count, 2)
        self.assertEqual(result.intent, "rag")
        self.assertEqual(
            [item.type for item in result.tasks],
            ["case_identification", "metric_aggregation", "case_analysis"],
        )
        self.assertEqual(result.tasks[1].conditions["metric"], "precipitation.maximum")
        self.assertEqual(result.tasks[1].conditions["operator"], "max")
        self.assertNotIn("report_generation", [item.type for item in result.tasks])

    def test_model_can_return_arbitrary_length_validated_task_plan(self) -> None:
        """模型任务计划应保留数量和依赖，同时由后端覆盖路由与证据范围。"""
        agent = agent_module.QueryIntentAgent(
            llm_client=MultiTaskLlm(),
            audit_path=Path(self.temp_dir.name) / "multi-task-audit.jsonl",
            case_provider=self.agent.case_provider,
        )

        result = agent.route(schema_module.IntentRouteRequest(
            message="分析2025年1月23-26日雨雪寒潮大风天气过程、查询预警并给出图片证据",
        ))

        self.assertEqual(len(result.tasks), 3)
        self.assertEqual([item.route for item in result.tasks], ["rag", "rag", "rag"])
        self.assertEqual(result.tasks[1].depends_on, ["task_1"])
        self.assertEqual(result.tasks[2].evidence_scope, "image_index")

    def test_model_plan_uses_shared_case_set_for_annual_count_and_listing(self) -> None:
        """全年统计中的逐项列举不是比较，且时间范围不得被缩成知识库月份。"""
        agent = agent_module.QueryIntentAgent(
            llm_client=AnnualStatisticsLlm(),
            audit_path=Path(self.temp_dir.name) / "annual-statistics-audit.jsonl",
            case_provider=self.agent.case_provider,
        )

        result = agent.route(schema_module.IntentRouteRequest(
            message="2025年全年有几次高温天气，分别是什么时候",
            memory_context={
                "previous_turn": {"question": "分析7月天气", "answer_summary": "7月出现高温过程"},
            },
        ))

        self.assertFalse(result.context_related)
        self.assertEqual(result.normalized_message, "2025年全年有几次高温天气，分别是什么时候")
        self.assertEqual(result.conditions["years"], [2025])
        self.assertNotIn("months", result.conditions)
        self.assertEqual(
            [item.type for item in result.tasks],
            ["case_identification", "aggregate_statistics", "case_listing"],
        )
        self.assertEqual(result.tasks[1].depends_on, ["task_1"])
        self.assertEqual(result.tasks[2].depends_on, ["task_1"])
        self.assertNotIn("comparison", [item.type for item in result.tasks])

    def test_invalid_comparison_is_rejected_and_compiled_from_output_fields(self) -> None:
        """即使模型误分类，也只按结构化输出要求生成筛选、计数和枚举任务。"""
        agent = agent_module.QueryIntentAgent(
            llm_client=AnnualMisclassifiedLlm(),
            audit_path=Path(self.temp_dir.name) / "annual-misclassified-audit.jsonl",
            case_provider=self.agent.case_provider,
        )

        result = agent.route(schema_module.IntentRouteRequest(
            message="2025年全年有几次高温天气，分别是什么时候",
        ))

        self.assertEqual(
            [item.type for item in result.tasks],
            ["case_identification", "aggregate_statistics", "case_listing"],
        )
        self.assertEqual(result.tasks[1].depends_on, ["task_1"])
        self.assertEqual(result.tasks[2].depends_on, ["task_1"])

    def test_feature_output_field_compiles_case_analysis_with_chunk_scope(self) -> None:
        """模型把 features 声明为输出字段时，必须追加依赖个例集合的正文分析任务。"""
        agent = agent_module.QueryIntentAgent(
            llm_client=FeatureAnalysisLlm(),
            audit_path=Path(self.temp_dir.name) / "feature-analysis-audit.jsonl",
            case_provider=self.agent.case_provider,
        )

        result = agent.route(schema_module.IntentRouteRequest(
            message="统计2025年3月灾害次数、名称、时间和特征",
        ))

        task_by_type = {item.type: item for item in result.tasks}
        self.assertIn("case_identification", task_by_type)
        self.assertIn("aggregate_statistics", task_by_type)
        self.assertIn("case_listing", task_by_type)
        self.assertIn("case_analysis", task_by_type)
        self.assertIn(task_by_type["case_identification"].id, task_by_type["case_analysis"].depends_on)
        self.assertEqual(task_by_type["case_analysis"].evidence_scope, "case_chunks")

    def test_requested_outputs_repair_missing_analysis_task(self) -> None:
        """模型漏写正文任务时，回答目标契约应自动补齐并建立依赖。"""
        agent = agent_module.QueryIntentAgent(
            llm_client=RequestedOutputsLlm(),
            audit_path=Path(self.temp_dir.name) / "requested-output-audit.jsonl",
            case_provider=self.agent.case_provider,
        )

        result = agent.route(schema_module.IntentRouteRequest(
            message="帮我统计2025年3月发生了几次灾害，分别是哪几次，有什么特征",
        ))

        self.assertEqual(result.conditions["requested_outputs"], ["count", "case_list", "case_features"])
        task_by_type = {item.type: item for item in result.tasks}
        self.assertIn("case_analysis", task_by_type)
        self.assertEqual(task_by_type["case_analysis"].evidence_scope, "case_chunks")
        self.assertIn(task_by_type["case_identification"].id, task_by_type["case_analysis"].depends_on)

    def test_explicit_knowledge_case_without_history_stays_in_rag(self) -> None:
        """独立问题完整点名库内个例时，预报问法不得误触发 Smart。"""
        result = self.route("分析2025年1月23-26日雨雪寒潮大风天气过程，发生之前有没有预报预警")

        self.assertEqual(result.intent, "rag")
        self.assertEqual(result.referenced_case_ids, [JAN_23_ID])
        self.assertEqual(result.context_resolution_source, "explicit_knowledge_lookup")
        self.assertEqual([item.type for item in result.tasks], ["case_analysis", "forecast_warning_lookup"])

    def test_invalid_memory_case_ids_require_clarification(self) -> None:
        """旧版本产生的 jan-2、all-* 不得被当成当前知识库实体继续路由。"""
        refs = [
            {"case_id": "jan-2", "title": "旧版一月个例"},
            {"case_id": "all-15", "title": "旧版聚合个例"},
        ]
        memory = {"previous_turn": {"case_refs": refs}, "available_case_refs": refs}

        result = self.route("详细分析这个个例", memory_context=memory)

        self.assertEqual(result.intent, "clarify")
        self.assertEqual(result.referenced_case_ids, [])
        self.assertTrue(result.need_clarification)

    def test_ambiguous_single_reference_requires_clarification(self) -> None:
        """上一轮存在多个库内实体但没有足够定位线索时，不得猜测或调用 Smart。"""
        refs = [
            {"case_id": JAN_14_ID, "title": "1月14日-15日寒潮天气过程", "date_range": "2025年1月14-15日"},
            {"case_id": JAN_23_ID, "title": "1月23-26日雨雪寒潮大风天气过程", "date_range": "2025年1月23-26日"},
        ]
        memory = {"previous_turn": {"case_refs": refs}, "available_case_refs": refs}

        result = self.route("详细分析这个过程", memory_context=memory)

        self.assertEqual(result.intent, "clarify")
        self.assertTrue(result.need_clarification)

    def test_reference_without_history_requires_clarification(self) -> None:
        """当前会话没有历史时不得为“这两个”凭空选择个例。"""
        result = self.route("详细分析一下这两个")
        self.assertEqual(result.intent, "clarify")
        self.assertTrue(result.need_clarification)
        self.assertIn("没有可供解析的历史内容", result.clarification_question)

    def test_model_routes_semantic_external_process_variants_to_smart(self) -> None:
        """不同自然语言表达均由模型语义判断，不依赖生产代码中的相对时间词表。"""
        questions = [
            "过去24小时晋东南出现极端强降水，请帮我分析并作出预测",
            "刚结束的一轮晋城强降水已经突破历史极值，请研判后续风险",
            "截至现在长治累计雨量持续攀升，请找历史参考经验并给出预警建议",
        ]
        for index, question in enumerate(questions):
            with self.subTest(question=question):
                llm = ExternalProcessSemanticLlm()
                agent = agent_module.QueryIntentAgent(
                    llm_client=llm,
                    audit_path=Path(self.temp_dir.name) / f"external-{index}.jsonl",
                    case_provider=self.agent.case_provider,
                )
                result = agent.route(schema_module.IntentRouteRequest(message=question))

                self.assertEqual(result.intent, "similar_case_match")
                self.assertEqual(result.reference_type, "external_process")
                self.assertEqual([item.type for item in result.tasks], ["similar_case_match"])
                self.assertIn("当前业务时间", llm.inputs[0])
                self.assertEqual(len(llm.inputs[0]["标准个例目录"]), 4)

    def test_missing_intent_is_repaired_by_model_instead_of_default_rag(self) -> None:
        """首次语义结果漏填 intent 时应模型修复，不能静默落入默认 RAG。"""
        llm = ExternalProcessSemanticLlm(omit_first_intent=True)
        agent = agent_module.QueryIntentAgent(
            llm_client=llm,
            audit_path=Path(self.temp_dir.name) / "intent-repair.jsonl",
            case_provider=self.agent.case_provider,
        )

        result = agent.route(schema_module.IntentRouteRequest(
            message="过去24小时晋东南出现极端强降水，请帮我分析并作出预测",
        ))

        self.assertEqual(llm.call_count, 2)
        self.assertEqual(result.intent, "similar_case_match")
        self.assertEqual(result.routing_source, "rule_confirmed_by_llm")

    def test_explicit_catalog_case_overrides_incorrect_smart_model_decision(self) -> None:
        """明确日期唯一命中库内个例时仍由主 RAG 硬保护。"""
        july_case = {
            "case_id": "case-july-28",
            "title": "7月28日强对流天气过程",
            "date_range": "2025年7月28日",
            "disaster_types": ["强对流"],
            "affected_areas": ["山西省"],
        }
        llm = ExternalProcessSemanticLlm()
        agent = agent_module.QueryIntentAgent(
            llm_client=llm,
            audit_path=Path(self.temp_dir.name) / "knowledge-guard.jsonl",
            case_provider=lambda: [july_case],
        )

        result = agent.route(schema_module.IntentRouteRequest(
            message="分析2025年7月28日强对流天气过程",
        ))

        self.assertEqual(result.intent, "rag")
        self.assertEqual(result.referenced_case_ids, ["case-july-28"])
        self.assertEqual(result.context_resolution_source, "explicit_knowledge_lookup")


if __name__ == "__main__":
    unittest.main()
