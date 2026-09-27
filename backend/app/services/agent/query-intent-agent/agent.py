"""查询意图识别、置信度校验和目标智能体选择。"""
from __future__ import annotations

import json
import hashlib
import inspect
import logging
import re
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from backend.app.config import settings
from backend.app.services.agent.case_multidim_search.core.natural_query import NaturalCaseQueryParser
from backend.app.services.model_client import get_intent_llm_client

from .audit import IntentAuditWriter
from .knowledge import KnowledgeRangeReader
from .schemas import IntentName, IntentRouteRequest, IntentRouteResponse, KnowledgeRange, TaskPlanItem


LOGGER = logging.getLogger("uvicorn.error")
REPORT_TERMS = ("生成报告", "生成一份报告", "出一份报告", "制作报告", "形成报告", "导出报告", "生成PDF", "导出PDF")
SIMILAR_TERMS = ("相似", "类似", "匹配", "参考经验", "参考个例", "可借鉴", "历史上有没有类似")
CURRENT_TERMS = ("当前", "现在", "目前", "今天", "今日", "明天", "未来", "预计", "即将")
RISK_ANALYSIS_TERMS = ("分析", "研判", "风险", "影响", "建议", "参考经验", "相似", "类似", "匹配")
AGGREGATION_TERMS = ("多少", "几个", "几次", "统计", "汇总", "数量", "分布", "占比", "最多", "最少", "分别", "对比", "比较", "清单", "列出", "有哪些")
STRUCTURED_QUERY_TERMS = ("查询", "检索", "筛选", "查找", "历史个例", "灾害个例")
# 指标任务只接受执行器能够确定性聚合的规范名称，模型不得自由扩展字段值。
METRIC_CONTRACT_VALUES = {
    "air_temperature.minimum",
    "air_temperature.maximum",
    "precipitation.maximum",
    "wind_speed.maximum",
    "visibility.minimum",
}


@dataclass
class CandidateDecision:
    """规则或大模型给出的中间意图判断。"""

    intent: IntentName
    confidence: float
    reason: str
    source: str
    hard: bool = False


@dataclass
class ContextResolution:
    """会话关联判断与指代消解的中间结果。"""

    related: bool
    standalone_message: str
    confidence: float
    relation_type: str = "standalone"
    case_ids: list[str] | None = None
    message_ids: list[int] | None = None
    source: str = "none"
    reason: str = ""
    unresolved_reference: bool = False
    reference_type: str = "none"
    reference_clues: list[str] | None = None
    task_plan: list[dict[str, Any]] | None = None
    query_conditions: dict[str, Any] | None = None


class QueryIntentAgent:
    """只负责识别和分发，不执行两个下游 Agent 的业务逻辑。"""

    def __init__(
        self,
        data_dir: Path | None = None,
        llm_client=None,
        audit_path: Path | None = None,
        case_provider: Callable[[], list[Any]] | None = None,
    ):
        """初始化共享知识库范围读取器、可选模型和审计日志。"""
        package_dir = Path(__file__).resolve().parent
        self.data_dir = Path(data_dir or settings.smart_case_data_dir)
        self.knowledge_reader = KnowledgeRangeReader(self.data_dir)
        self.llm_client = llm_client if llm_client is not None else get_intent_llm_client()
        self.audit_writer = IntentAuditWriter(audit_path or package_dir / "runtime" / "intent_audit.jsonl")
        self.case_provider = case_provider
        # 复用多维检索的确定性条件词表，避免两个入口产生不同的日期和灾种口径。
        self.condition_parser = NaturalCaseQueryParser(llm_client=None)
        # 意图缓存只保存小型 JSON 结果，避免相同问题因重复提交再次等待云端模型。
        self._model_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._model_cache_lock = Lock()
        self._model_cache_limit = 256

    def knowledge_range(self) -> KnowledgeRange:
        """返回页面展示和意图判断共同使用的知识库时间范围。"""
        return self.knowledge_reader.read()

    def route(self, request: IntentRouteRequest) -> IntentRouteResponse:
        """归一化消息，融合规则与模型结果并返回白名单分发决定。"""
        original = str(request.message or "").strip()
        knowledge = self.knowledge_range()
        normalized_original = self.normalize_text(original)
        knowledge_cases = self._knowledge_case_catalog()
        memory_context = self._sanitize_memory_context(request.memory_context, knowledge_cases)
        model_context, model = self._model_understanding(
            normalized_original,
            memory_context,
            knowledge,
            knowledge_cases,
            request.previous_intent or memory_context.get("previous_intent"),
        )
        context = self._resolve_context(
            normalized_original,
            memory_context,
            knowledge,
            knowledge_cases,
            model_context=model_context,
        )
        normalized = self.normalize_text(context.standalone_message or normalized_original)
        # 模型已经完成语义理解时直接采用其规范化条件；规则解析只在模型未提供条件时兜底。
        model_conditions = dict(context.query_conditions or {})
        conditions = model_conditions or self._extract_conditions(normalized)
        explicit_case_ids = self._match_explicit_knowledge_cases(normalized, conditions, knowledge_cases)
        if not context.case_ids and len(explicit_case_ids) == 1:
            # 用户已完整点名库内唯一个例时，不依赖会话历史也必须执行主库保护。
            context.case_ids = explicit_case_ids
            context.reference_type = "knowledge_case"
            context.reference_clues = self._date_clues(normalized)
            context.unresolved_reference = False
            context.source = "explicit_knowledge_lookup"
            context.reason = "根据用户明确日期、灾种或标题在当前标准个例库中唯一定位。"
        previous_intent = request.previous_intent or memory_context.get("previous_intent")
        if context.unresolved_reference:
            rule = CandidateDecision("clarify", 0.99, context.reason or "历史指代无法可靠解析。", "context_rule", True)
            model = None
        else:
            rule = self._rule_decision(normalized, conditions, knowledge, previous_intent, context)
        final, suggested = self._merge_decisions(rule, model)
        # 任务目标契约优先于“是否一句话包含多个诉求”的保守澄清。
        # 只要模型已经把回答目标限定在主 RAG 能完成的范围内，就应执行任务图，
        # 而不是因为多任务本身返回 clarify；真正无法定位实体时仍保留硬澄清结果。
        rag_output_targets = {
            "count", "case_list", "case_features", "intensity", "impact", "impact_area",
            "evolution", "mechanism", "circulation", "weather_facts", "evidence_summary",
            "forecast_warning", "image_evidence", "comparison",
        }
        requested_outputs = set(conditions.get("requested_outputs") or [])
        model_task_types = {
            str(item.get("type") or "")
            for item in (context.task_plan or [])
            if isinstance(item, dict)
        }
        # 重型路由必须由模型的结构化任务目标支持，不能只凭一个高置信 intent 越权升级。
        if (
            final.intent == "multidim_search"
            and rule.intent != "multidim_search"
            and "report" not in requested_outputs
            and "report_generation" not in model_task_types
        ):
            final = CandidateDecision(
                "rag",
                max(rule.confidence, 0.92),
                "任务计划没有报告目标，按主 RAG 执行结构化检索与分析",
                "task_contract",
                True,
            )
            suggested = None
        if (
            final.intent == "clarify"
            and requested_outputs
            and requested_outputs.issubset(rag_output_targets)
            # “它们/各过程”等代词可由前置的统计或清单任务解析；只有没有任何
            # 可生成实体集合的目标时，才把未解析指代升级为必须澄清。
            and (
                not context.unresolved_reference
                or bool(requested_outputs & {"count", "case_list"})
            )
        ):
            final = CandidateDecision(
                "rag",
                max(final.confidence, 0.86),
                "回答目标均可由主 RAG 任务图执行",
                "task_contract",
                True,
            )
            suggested = None
        audit_id = f"intent_{uuid4().hex}"
        need_clarification = final.intent == "clarify"
        target_agent, target_endpoint = self._target(final.intent)
        tasks = self._build_task_plan(normalized, final.intent, context, conditions, context.task_plan)
        response = IntentRouteResponse(
            intent=final.intent,
            confidence=max(0.0, min(1.0, round(final.confidence, 4))),
            normalized_message=normalized,
            original_message=original,
            context_related=context.related,
            relation_type=context.relation_type,
            reference_type=context.reference_type,
            context_confidence=max(0.0, min(1.0, round(context.confidence, 4))),
            referenced_case_ids=list(context.case_ids or []),
            knowledge_case_ids=list(context.case_ids or []),
            reference_clues=list(context.reference_clues or []),
            context_message_ids=list(context.message_ids or []),
            context_resolution_source=context.source,
            conditions=conditions,
            reason=final.reason,
            need_clarification=need_clarification,
            clarification_question=(
                (context.reason or "请明确指出需要继续分析的历史个例。")
                if context.unresolved_reference
                else "请确认你希望按时间、灾种和地区筛选历史个例，还是根据一个天气过程匹配相似历史个例。"
                if need_clarification
                else ""
            ),
            suggested_intent=suggested,
            target_agent=target_agent,
            target_endpoint=target_endpoint,
            routing_source=final.source,
            rag_strategy="vector" if context.case_ids else self._rag_strategy(normalized, conditions),
            knowledge_range=knowledge,
            audit_id=audit_id,
            tasks=tasks,
        )
        self._write_audit(audit_id, original, normalized, conditions, previous_intent, rule, model, response)
        LOGGER.info(
            "[意图分发] audit_id=%s intent=%s confidence=%.2f source=%s target=%s",
            audit_id,
            response.intent,
            response.confidence,
            response.routing_source,
            response.target_agent or "待澄清",
        )
        return response

    def _match_explicit_knowledge_cases(
        self,
        text: str,
        conditions: dict[str, Any],
        catalog: dict[str, dict[str, Any]],
    ) -> list[str]:
        """只根据明确 ID、标题或日期定位库内实体，相似属性不能证明实体相同。"""
        date_clues = self._date_clues(text)
        explicit_ids = [case_id for case_id in catalog if case_id and case_id in text]
        if explicit_ids:
            return explicit_ids
        explicit_titles = [case_id for case_id, item in catalog.items() if item.get("title") and str(item["title"]) in text]
        if explicit_titles:
            return explicit_titles
        if not date_clues:
            return []
        candidates = [item for item in catalog.values() if self._case_ref_matches_date_clues(item, date_clues)]
        disasters = [str(value) for value in conditions.get("disaster_types") or []]
        if disasters:
            matched = [
                item for item in candidates
                if any(
                    wanted in actual or actual in wanted
                    for wanted in disasters
                    for actual in item.get("disaster_types") or []
                )
            ]
            if matched:
                candidates = matched
        return [str(item["case_id"]) for item in candidates]

    def _build_task_plan(
        self,
        text: str,
        intent: IntentName,
        context: ContextResolution,
        conditions: dict[str, Any],
        model_tasks: list[dict[str, Any]] | None = None,
    ) -> list[TaskPlanItem]:
        """把一条自然语言问题拆成可校验的任务计划，避免多个问题共用一个检索范围。"""
        compact = re.sub(r"\s+", "", text)
        tasks: list[TaskPlanItem] = []

        def add(task_type: str, route: str, question: str, scope: str, depends_on: list[str] | None = None) -> None:
            if any(item.type == task_type for item in tasks):
                return
            tasks.append(TaskPlanItem(
                id=f"task_{len(tasks) + 1}",
                type=task_type,
                route=route,
                question=question,
                depends_on=list(depends_on or []),
                evidence_scope=scope,
                conditions=dict(conditions),
            ))

        # 有效模型计划是语义层的唯一来源，本地不再按单个关键词二次追加任务。
        validated_model_tasks = self._validate_model_tasks(
            model_tasks or [], text, context, intent, conditions,
        )
        validated_model_tasks = self._compile_structured_dataflow(
            validated_model_tasks, conditions, text, intent,
            has_context_cases=bool(context.case_ids),
        )
        if validated_model_tasks and self._model_plan_covers_intent(validated_model_tasks, intent):
            return self._normalize_task_plan(validated_model_tasks)

        if intent == "clarify":
            add("clarification", "clarify", text, "none")
            return self._normalize_task_plan(tasks)

        needs_case_analysis = bool(
            context.case_ids or any(term in compact for term in ("分析", "灾害情况", "过程", "成因", "影响"))
        )
        identification_task = next((item for item in tasks if item.type == "case_identification"), None)
        if needs_case_analysis and not context.case_ids:
            add(
                "case_identification",
                "rag",
                f"在标准个例库中定位下列问题所指的历史个例：{text}",
                "structured_cases",
            )
            identification_task = next((item for item in tasks if item.type == "case_identification"), None)
        base_task = next((item for item in tasks if item.type == "case_analysis"), None)
        base_id: str | None = base_task.id if base_task else None
        if needs_case_analysis:
            add(
                "case_analysis",
                "rag",
                f"{text}\n重点检索灾害实况、影响和环流成因。",
                "case_chunks",
                [identification_task.id] if identification_task else [],
            )
            base_task = next((item for item in tasks if item.type == "case_analysis"), None)
            base_id = base_task.id if base_task else None

        # 预报、预警、预报效果等证据往往位于同一 PDF 的月度服务表，不能继承主体个例的窄范围。
        if any(term in compact for term in ("预报", "预警", "预报效果", "预警服务", "有没有报", "是否报")):
            add(
                "forecast_warning_lookup",
                "rag",
                f"{text}\n重点检索过程发生前及过程中的预报预警、发布时间、级别、灾种和区域。",
                "same_pdf_related",
                [base_id] if base_id else [],
            )
        if any(term in compact for term in ("多少", "几个", "几次", "统计", "数量", "分布", "占比", "清单", "列出")):
            add("aggregate_statistics", "rag", f"{text}\n仅完成结构化统计或清单查询。", "structured_cases")
        if any(term in compact for term in ("影响区域", "哪些地区", "影响了哪些", "波及", "影响范围")):
            add("impact_area_lookup", "rag", f"{text}\n重点提取影响区域和灾情范围。", "case_chunks", [base_id] if base_id else [])
        if any(term in compact for term in ("对比", "比较", "差异", "相比", "异同", "哪个更")):
            add("comparison", "rag", f"{text}\n分别提取比较对象的同口径事实后进行对比。", "multi_case_chunks")
        if any(term in compact for term in ("图片", "图表", "图像", "图证据")):
            add("image_lookup", "rag", f"{text}\n重点查找与结论直接对应的图片证据。", "image_index", [base_id] if base_id else [])

        # 重型任务放在依赖任务之后，仍交给原 Agent 执行，不改变其内容生成过程。
        if intent == "similar_case_match":
            add("similar_case_match", "similar_case_match", text, "similar_case_index", [item.id for item in tasks])
        if intent == "multidim_search":
            add("report_generation", "multidim_search", text, "structured_cases", [item.id for item in tasks])

        if not tasks:
            add("knowledge_qa", "rag", text, "vector")
        return self._normalize_task_plan(
            self._compile_structured_dataflow(
                tasks, conditions, text, intent,
                has_context_cases=bool(context.case_ids),
            )
        )

    @staticmethod
    def _normalize_task_plan(tasks: list[TaskPlanItem]) -> list[TaskPlanItem]:
        """保持模型给出的拓扑顺序并重新编号，不擅自补写语义依赖。"""
        old_to_new = {item.id: f"task_{index + 1}" for index, item in enumerate(tasks)}
        normalized: list[TaskPlanItem] = []
        for index, item in enumerate(tasks, start=1):
            dependencies = [old_to_new[value] for value in item.depends_on if value in old_to_new]
            normalized.append(item.model_copy(update={
                "id": f"task_{index}",
                "depends_on": [value for value in dependencies if int(value.split("_")[-1]) < index],
            }))
        return normalized

    def _validate_model_tasks(
        self,
        raw_tasks: list[dict[str, Any]],
        fallback_question: str,
        context: ContextResolution,
        final_intent: IntentName,
        default_conditions: dict[str, Any],
    ) -> list[TaskPlanItem]:
        """校验模型任务白名单、执行范围和依赖顺序，拒绝越权或循环计划。"""
        route_by_type = {
            "knowledge_qa": "rag",
            "case_identification": "rag",
            "case_listing": "rag",
            "case_analysis": "rag",
            "forecast_warning_lookup": "rag",
            "aggregate_statistics": "rag",
            "metric_aggregation": "rag",
            "comparison": "rag",
            "impact_area_lookup": "rag",
            "image_lookup": "rag",
            "similar_case_match": "similar_case_match",
            "report_generation": "multidim_search",
            "clarification": "clarify",
        }
        scope_by_type = {
            "knowledge_qa": "vector",
            "case_identification": "structured_cases",
            "case_listing": "structured_cases",
            "case_analysis": "case_chunks",
            "forecast_warning_lookup": "same_pdf_related",
            "aggregate_statistics": "structured_cases",
            "metric_aggregation": "document_metric_facts",
            "comparison": "multi_case_chunks",
            "impact_area_lookup": "case_chunks",
            "image_lookup": "image_index",
            "similar_case_match": "similar_case_index",
            "report_generation": "structured_cases",
            "clarification": "none",
        }
        result: list[TaskPlanItem] = []
        known_ids: set[str] = set()
        id_mapping: dict[str, str] = {}
        for raw in raw_tasks[:10]:
            if not isinstance(raw, dict):
                continue
            task_type = str(raw.get("type") or "")
            if task_type not in route_by_type:
                continue
            route = route_by_type[task_type]
            if final_intent == "rag" and route != "rag":
                continue
            if final_intent == "clarify" and route != "clarify":
                continue
            if final_intent == "multidim_search" and route == "similar_case_match":
                continue
            if final_intent == "similar_case_match" and route == "multidim_search":
                continue
            if final_intent == "similar_case_match" and route != "similar_case_match":
                # 库外新过程的历史检索、综合研判和预测都由 Smart 自身完成，
                # 不能再附加主 RAG 的个例定位或预警检索任务污染执行范围。
                continue
            # 已定位库内个例时，模型不能用相似匹配绕过主知识库。
            if context.case_ids and route == "similar_case_match":
                task_type, route = "comparison", "rag"
            raw_conditions = raw.get("conditions") if isinstance(raw.get("conditions"), dict) else {}
            task_conditions = {
                **default_conditions,
                **self._validate_query_conditions(raw_conditions),
            }
            if task_type == "comparison":
                # 比较任务必须明确给出至少两个对象和比较维度，枚举结果不能冒充比较。
                targets = task_conditions.get("comparison_targets") or raw.get("comparison_targets") or []
                dimensions = task_conditions.get("comparison_dimensions") or raw.get("comparison_dimensions") or []
                if not isinstance(targets, list) or len(targets) < 2 or not isinstance(dimensions, list) or not dimensions:
                    continue
                task_conditions["comparison_targets"] = [str(value)[:200] for value in targets[:10]]
                task_conditions["comparison_dimensions"] = [str(value)[:100] for value in dimensions[:10]]
            question = str(raw.get("question") or fallback_question).strip()[:6000]
            if not question:
                continue
            task_id = f"task_{len(result) + 1}"
            dependencies = [
                id_mapping[str(item)]
                for item in raw.get("depends_on") or []
                if str(item) in id_mapping and id_mapping[str(item)] in known_ids
            ]
            result.append(TaskPlanItem(
                id=task_id,
                type=task_type,
                route=route,
                question=question,
                depends_on=dependencies,
                evidence_scope=scope_by_type[task_type],
                conditions=task_conditions,
            ))
            known_ids.add(task_id)
            id_mapping[str(raw.get("id") or task_id)] = task_id
        return result

    @staticmethod
    def _model_plan_covers_intent(tasks: list[TaskPlanItem], intent: IntentName) -> bool:
        """确认模型计划包含顶层路由所需的终端任务，避免有效路由被不完整计划吞掉。"""
        required_type = {
            "multidim_search": "report_generation",
            "similar_case_match": "similar_case_match",
            "clarify": "clarification",
        }.get(intent)
        return required_type is None or any(item.type == required_type for item in tasks)

    @staticmethod
    def _compile_structured_dataflow(
        tasks: list[TaskPlanItem],
        conditions: dict[str, Any],
        question: str,
        intent: IntentName,
        has_context_cases: bool = False,
    ) -> list[TaskPlanItem]:
        """根据模型声明的输出字段补齐数据依赖，不读取用户问题中的关键词。"""
        if intent != "rag" or not tasks:
            return tasks
        output_fields = set(conditions.get("output_fields") or [])
        analysis_fields = set(conditions.get("analysis_fields") or [])
        requested_outputs = set(conditions.get("requested_outputs") or [])
        # 输出字段按任务契约分成“个例目录字段”和“必须回到正文核验的分析字段”。
        # 这里校验的是模型输出协议，不依赖用户问题中的某个关键词。
        listing_fields = {
            "case_id", "title", "name", "date", "date_range", "disaster_types",
            "affected_areas", "source_pdf", "count",
        }
        deep_analysis_fields = {
            "features", "case_features", "weather_facts", "intensity_metrics",
            "impact", "impact_area", "evolution", "mechanism", "circulation",
            "forecast", "warning", "evidence_summary",
        }
        needs_analysis = bool(
            analysis_fields & deep_analysis_fields
            or output_fields & deep_analysis_fields
            or any(item.type == "case_analysis" for item in tasks)
            or requested_outputs & {
                "case_features", "intensity", "impact", "impact_area", "evolution",
                "mechanism", "circulation", "weather_facts", "evidence_summary",
            }
        )
        # 聚合统计统一复用一个个例集合，并附带个例清单，避免只给标签数量却不说明具体过程。
        needs_count = bool(
            "count" in output_fields
            or "count" in requested_outputs
            or any(item.type == "aggregate_statistics" for item in tasks)
        )
        needs_listing = needs_count or bool(
            output_fields & {"case_id", "title", "date_range", "disaster_types", "affected_areas"}
            or "case_list" in requested_outputs
        )
        # 即使问题没有要求计数或目录清单，只要模型声明了特征、影响、环流等正文分析字段，
        # 也必须继续编译 case_analysis；否则正文检索会被这个早退条件静默跳过。
        if not needs_count and not needs_listing and not needs_analysis:
            return tasks

        compiled = list(tasks)
        identification = next((item for item in compiled if item.type == "case_identification"), None)
        existing_analysis = next((item for item in compiled if item.type == "case_analysis"), None)
        # 所有需要按个例逐条回答的任务都必须先获得统一的个例集合。
        # 只有上下文已经明确给出库内 case_id 时才跳过识别，避免模型漏写
        # case_identification 导致后续分析任务在空范围上执行。
        case_scoped_types = {
            "case_analysis", "case_listing", "aggregate_statistics", "metric_aggregation", "comparison",
            "forecast_warning_lookup", "impact_area_lookup", "image_lookup",
        }
        need_case_set = bool(
            needs_count or needs_listing or needs_analysis
            or any(item.type in case_scoped_types for item in compiled)
        )
        if identification is None and need_case_set and not has_context_cases:
            identification = TaskPlanItem(
                id="model_case_set",
                type="case_identification",
                route="rag",
                question=question,
                evidence_scope="structured_cases",
                conditions=dict(conditions),
            )
            compiled.insert(0, identification)
        elif identification is not None and compiled[0] is not identification:
            compiled = [identification, *(item for item in compiled if item is not identification)]

        def depend_on_case_set(item: TaskPlanItem) -> TaskPlanItem:
            if identification is None:
                # 上下文已经给出明确库内个例时，任务直接使用该范围，不需要虚构识别依赖。
                return item.model_copy(update={"conditions": dict(item.conditions or conditions)})
            dependencies = list(item.depends_on)
            if identification.id not in dependencies:
                dependencies.append(identification.id)
            return item.model_copy(update={"depends_on": dependencies, "conditions": dict(item.conditions or conditions)})

        # 模型自行生成的预警、影响区域、图片和比较任务也统一挂到个例集合，
        # 保证执行器按“识别 -> 分任务检索”的拓扑顺序运行。
        if identification is not None:
            for index, item in enumerate(compiled):
                if item.type in case_scoped_types:
                    compiled[index] = depend_on_case_set(item)

        metric_task = next((item for item in compiled if item.type == "metric_aggregation"), None)
        if metric_task is not None and identification is not None:
            # 指标极值可能出现在任何灾种正文中，候选集合不能被灾种标签提前截断。
            identification_conditions = dict(identification.conditions or conditions)
            identification_conditions.pop("disaster_types", None)
            compiled[compiled.index(identification)] = identification.model_copy(
                update={"conditions": identification_conditions},
            )
            compiled[compiled.index(metric_task)] = depend_on_case_set(metric_task)

        if needs_count and identification is not None:
            aggregate = next((item for item in compiled if item.type == "aggregate_statistics"), None)
            if aggregate is None:
                compiled.append(TaskPlanItem(
                    id="model_count",
                    type="aggregate_statistics",
                    route="rag",
                    question=question,
                    depends_on=[identification.id],
                    evidence_scope="structured_cases",
                    conditions=dict(conditions),
                ))
            else:
                compiled[compiled.index(aggregate)] = depend_on_case_set(aggregate)
        if needs_listing and identification is not None:
            listing = next((item for item in compiled if item.type == "case_listing"), None)
            if listing is None:
                compiled.append(TaskPlanItem(
                    id="model_listing",
                    type="case_listing",
                    route="rag",
                    question=question,
                    depends_on=[identification.id],
                    evidence_scope="structured_cases",
                    conditions=dict(conditions),
                ))
            else:
                compiled[compiled.index(listing)] = depend_on_case_set(listing)
        if needs_analysis:
            analysis = next((item for item in compiled if item.type == "case_analysis"), None)
            if analysis is None:
                if identification is None:
                    # 理论上仅有深度分析字段也必须先建立可追溯的个例集合。
                    identification = TaskPlanItem(
                        id="model_case_set",
                        type="case_identification",
                        route="rag",
                        question=question,
                        evidence_scope="structured_cases",
                        conditions=dict(conditions),
                    )
                    compiled.insert(0, identification)
                analysis = TaskPlanItem(
                    id="model_analysis",
                    type="case_analysis",
                    route="rag",
                    question=(
                        f"{question}\n请针对上游识别出的每一个个例，依据其关联正文提取可核验的过程特征、"
                        "强度、影响、演变和成因；没有证据的字段明确说明缺失。"
                    ),
                    depends_on=[identification.id],
                    evidence_scope="case_chunks",
                    conditions={
                        **dict(conditions),
                        "analysis_fields": sorted(analysis_fields or deep_analysis_fields),
                    },
                )
                compiled.append(analysis)
            else:
                compiled[compiled.index(analysis)] = depend_on_case_set(analysis).model_copy(update={
                    "evidence_scope": "case_chunks",
                    "conditions": {
                        **dict(analysis.conditions or conditions),
                        "analysis_fields": sorted(analysis_fields or deep_analysis_fields),
                    },
                })

        # 比较任务的对象通常是“前面列出的各过程”，模型不必凭空枚举 ID。
        # 当模型只声明 comparison 输出而未填 targets/dimensions 时，仍创建
        # 一个依赖个例清单的语义任务，由执行阶段从上游结果确定比较对象。
        if "comparison" in requested_outputs and not any(
            item.type == "comparison" for item in compiled
        ):
            dependencies: list[str] = []
            listing = next((item for item in compiled if item.type == "case_listing"), None)
            if listing is not None:
                dependencies.append(listing.id)
            elif identification is not None:
                dependencies.append(identification.id)
            compiled.append(TaskPlanItem(
                id="model_comparison",
                type="comparison",
                route="rag",
                question=(
                    f"{question}\n请使用上游列出的全部个例，按问题要求的维度进行比较；"
                    "比较对象和维度由上游结果与当前问题共同确定。"
                ),
                depends_on=dependencies,
                evidence_scope="multi_case_chunks",
                conditions=dict(conditions),
            ))

        # requested_outputs 是意图模型对用户回答目标的结构化声明。
        # 即使模型漏写了相应任务，计划校验器仍按声明补齐证据任务，而不依赖某一种中文问法。
        if "forecast_warning" in requested_outputs and not any(
            item.type == "forecast_warning_lookup" for item in compiled
        ):
            dependencies = [identification.id] if identification is not None else []
            compiled.append(TaskPlanItem(
                id="model_forecast_warning",
                type="forecast_warning_lookup",
                route="rag",
                question=f"{question}\n查询已识别个例发生前及过程中的预报预警证据。",
                depends_on=dependencies,
                evidence_scope="same_pdf_related",
                conditions=dict(conditions),
            ))
        if "image_evidence" in requested_outputs and not any(
            item.type == "image_lookup" for item in compiled
        ):
            dependencies = [identification.id] if identification is not None else []
            compiled.append(TaskPlanItem(
                id="model_image_evidence",
                type="image_lookup",
                route="rag",
                question=f"{question}\n查找与已识别个例结论直接对应的图片证据。",
                depends_on=dependencies,
                evidence_scope="image_index",
                conditions=dict(conditions),
            ))
        if "impact_area" in requested_outputs and not any(
            item.type == "impact_area_lookup" for item in compiled
        ):
            dependencies = [identification.id] if identification is not None else []
            compiled.append(TaskPlanItem(
                id="model_impact_area",
                type="impact_area_lookup",
                route="rag",
                question=f"{question}\n提取每个已识别个例的影响区域和灾情范围。",
                depends_on=dependencies,
                evidence_scope="case_chunks",
                conditions=dict(conditions),
            ))
        return compiled[:10]

    @staticmethod
    def _validate_requested_outputs(raw: Any) -> list[str]:
        """校验意图模型声明的回答目标，作为任务计划完整性检查的语义依据。"""
        allowed = {
            "count", "case_list", "case_features", "intensity", "impact", "impact_area",
            "evolution", "mechanism", "circulation", "weather_facts", "evidence_summary",
            "forecast_warning", "image_evidence", "comparison", "report", "metric_extrema",
        }
        if not isinstance(raw, list):
            return []
        return list(dict.fromkeys(str(value).strip() for value in raw if str(value).strip() in allowed))[:20]

    @staticmethod
    def _validate_query_conditions(raw: Any) -> dict[str, Any]:
        """校验模型归一化条件的类型和值域，不再从模型生成的中文任务中反向解析。"""
        if not isinstance(raw, dict):
            return {}
        result: dict[str, Any] = {}
        for key in ("start_date", "end_date"):
            value = str(raw.get(key) or "").strip()
            if not value:
                continue
            try:
                date.fromisoformat(value)
            except ValueError:
                continue
            result[key] = value
        years = []
        for value in raw.get("years") or []:
            try:
                year = int(value)
            except (TypeError, ValueError):
                continue
            if 1900 <= year <= 2100 and year not in years:
                years.append(year)
        months = []
        for value in raw.get("months") or []:
            try:
                month = int(value)
            except (TypeError, ValueError):
                continue
            if 1 <= month <= 12 and month not in months:
                months.append(month)
        if years:
            result["years"] = years
        if months:
            result["months"] = months
        for key in (
            "disaster_types", "cities", "areas", "output_fields", "analysis_fields",
            "comparison_targets", "comparison_dimensions", "result_dimensions",
        ):
            values = raw.get(key) or []
            if isinstance(values, list):
                normalized = [str(value).strip()[:200] for value in values if str(value).strip()]
                if normalized:
                    result[key] = list(dict.fromkeys(normalized))[:20]
        metric = str(raw.get("metric") or "").strip()
        if metric:
            result["metric"] = metric[:120]
        operator = str(raw.get("operator") or "").strip().lower()
        if operator in {"min", "max"}:
            result["operator"] = operator
        unit = str(raw.get("unit") or "").strip()
        if unit:
            result["unit"] = unit[:40]
        requested_outputs = QueryIntentAgent._validate_requested_outputs(raw.get("requested_outputs"))
        if requested_outputs:
            result["requested_outputs"] = requested_outputs
        scope_type = str(raw.get("time_scope_type") or "").strip()
        if scope_type in {"calendar_year", "date_range", "month_set", "unbounded"}:
            result["time_scope_type"] = scope_type
        return result

    def _resolve_context(
        self,
        message: str,
        memory: dict[str, Any],
        knowledge: KnowledgeRange,
        knowledge_cases: dict[str, dict[str, Any]],
        *,
        model_context: ContextResolution | None = None,
    ) -> ContextResolution:
        """结合分层记忆判断问题关联性，并生成可独立检索的完整问题。"""
        if not memory or not memory.get("previous_turn"):
            unresolved = self._has_context_reference(message)
            standalone_reference_type = (
                model_context.reference_type
                if model_context and model_context.reference_type == "external_process"
                else "none"
            )
            rule_without_history = ContextResolution(
                related=False,
                standalone_message=message,
                confidence=0.99 if unresolved else 1.0,
                source="rule" if unresolved else "llm" if standalone_reference_type == "external_process" else "none",
                reason=(
                    "当前会话中没有可供解析的历史内容，请明确写出要分析的个例名称。"
                    if unresolved
                    else str(model_context.reason or "") if model_context else ""
                ),
                unresolved_reference=unresolved,
                # 无历史时仍保留模型拆解出的任务计划，但不允许模型凭空创造历史实体。
                task_plan=list(model_context.task_plan or []) if model_context else None,
                query_conditions=dict(model_context.query_conditions or {}) if model_context else None,
                reference_type=standalone_reference_type,
                reference_clues=list(model_context.reference_clues or []) if model_context else [],
            )
            # 没有历史时模型只能负责意图分类，不能自行创造会话关联。
            return rule_without_history

        rule_context = self._rule_context_resolution(message, memory)
        # 云端模型优先负责语义关联和指代理解，确定性规则只补足可验证实体或在模型不可用时兜底。
        if model_context is None:
            return rule_context
        if model_context.related and model_context.case_ids:
            return model_context
        if model_context.related and model_context.reference_type in {"external_process", "aggregate_result"}:
            return model_context
        if model_context.related and rule_context.related and rule_context.case_ids:
            model_context.case_ids = list(rule_context.case_ids)
            model_context.reference_type = "knowledge_case"
            model_context.reference_clues = list(rule_context.reference_clues or [])
            model_context.source = "llm_and_knowledge_rule"
            return model_context
        if model_context.related and self._has_context_reference(message) and not model_context.case_ids:
            model_context.unresolved_reference = True
            model_context.reason = "模型判断本轮承接历史，但未能在当前标准个例库中唯一定位目标，请明确个例名称或日期。"
            return model_context
        return model_context

    def _rule_context_resolution(self, message: str, memory: dict[str, Any]) -> ContextResolution:
        """离线解析明确的承接词、数量和月份，作为云端模型不可用时的可靠兜底。"""
        if not self._has_context_reference(message):
            return ContextResolution(False, message, 0.92, source="rule")
        previous = memory.get("previous_turn") if isinstance(memory.get("previous_turn"), dict) else {}
        refs = [item for item in previous.get("case_refs") or [] if isinstance(item, dict) and item.get("case_id")]
        if not refs:
            refs = [item for item in memory.get("available_case_refs") or [] if isinstance(item, dict) and item.get("case_id")]
        # 上一轮答案若明确点名部分个例，优先采用实际展示给用户的这些实体。
        answer_summary = str(previous.get("answer_summary") or "")
        mentioned_refs = [
            item
            for item in refs
            if str(item.get("case_id") or "") in answer_summary
            or (item.get("title") and str(item.get("title")) in answer_summary)
        ]
        if mentioned_refs:
            refs = mentioned_refs
        months = self._months_in_text(message)
        if months:
            month_refs = [item for item in refs if self._case_ref_matches_month(item, months)]
            if month_refs:
                refs = month_refs
        date_clues = self._date_clues(message)
        if date_clues:
            dated_refs = [item for item in refs if self._case_ref_matches_date_clues(item, date_clues)]
            if dated_refs:
                refs = dated_refs
        expected = self._referenced_count(message)
        if expected is not None:
            if len(refs) < expected:
                return ContextResolution(
                    True,
                    message,
                    0.99,
                    relation_type="follow_up_reference",
                    source="rule",
                    reason=f"本轮指向 {expected} 个历史个例，但当前会话只能可靠定位 {len(refs)} 个，请明确个例名称。",
                    unresolved_reference=True,
                )
            refs = refs[:expected]
        elif self._expects_single_reference(message) and len(refs) > 1:
            return ContextResolution(
                True,
                message,
                0.96,
                relation_type="follow_up_reference",
                source="rule",
                reason="本轮指向单个历史个例，但根据当前会话仍能匹配多个库内个例，请明确个例名称或日期。",
                unresolved_reference=True,
                reference_clues=date_clues,
            )
        if not refs:
            if any(term in message for term in ("个例", "过程", "灾害", "事件")):
                return ContextResolution(
                    True,
                    message,
                    0.98,
                    relation_type="follow_up_reference",
                    source="knowledge_rule",
                    reason="会话中的候选实体不在当前标准个例库中，请明确要分析的有效个例名称或日期。",
                    unresolved_reference=True,
                    reference_clues=date_clues,
                )
            # 没有个例实体的“继续解释”仍可由模型依据上一轮摘要完成一般问题重写。
            return ContextResolution(False, message, 0.45, source="rule")
        titles = [str(item.get("title") or item.get("case_id")) for item in refs]
        message_ids = [
            int(value)
            for value in (previous.get("user_message_id"), previous.get("assistant_message_id"))
            if value
        ]
        standalone = f"针对历史个例{'、'.join(titles)}，{message}"
        return ContextResolution(
            True,
            standalone,
            0.99,
            relation_type="follow_up_reference",
            case_ids=[str(item["case_id"]) for item in refs],
            message_ids=message_ids,
            source="rule",
            reason="依据本轮指代词、数量和月份，从上一轮证据个例中完成指代消解。",
            reference_type="knowledge_case",
            reference_clues=date_clues,
        )

    def _model_understanding(
        self,
        message: str,
        memory: dict[str, Any],
        knowledge: KnowledgeRange,
        knowledge_cases: dict[str, dict[str, Any]],
        previous_intent: IntentName | None,
    ) -> tuple[ContextResolution | None, CandidateDecision | None]:
        """用一次云端调用完成上下文消解和路由，返回结果仍需本地规则复核。"""
        try:
            if not self.llm_client or not self.llm_client.is_available():
                return None, None
        except Exception:
            return None, None
        prompt = (
            "你是气象问答系统的会话理解和意图路由器，不回答业务问题。一次完成历史关联判断、"
            "指代消解、完整问题重写、任务分类和多任务拆解。"
            "先结合当前业务时间理解绝对时间和相对时间的真实指向，再判断本轮是否指代历史对话，"
            "最后判断目标是库内历史个例、外部新过程还是聚合结果。"
            "库内实体身份必须由标准个例目录确认：只有case_id、个例标题或明确日期能唯一对应目录项时，"
            "才可判为knowledge_case；仅灾种、地区、强度或环流形势相似不代表同一个例，只能视作历史相似候选。"
            "如果用户描述的过程相对当前业务时间刚发生、正在发生或即将发生，且无法唯一对应标准个例目录，"
            "必须判为external_process；不得因为目录中存在同地区或同灾种历史记录而改判为库内个例。"
            "外部新过程要求分析、预测、研判、预警或参考经验时，intent必须是similar_case_match，"
            "任务计划必须包含similar_case_match，不能生成用于定位该新过程的RAG个例检索任务。"
            "只能引用available_case_refs中存在且属于knowledge_case的case_id，不得创造个例、日期或天气事实。"
            "intent只能是rag、multidim_search、similar_case_match、clarify。"
            "只有明确要求生成或导出报告才选择multidim_search；"
            "只有描述当前、未来或知识库范围外的新风险过程并要求分析才选择similar_case_match；"
            "简单统计、库内个例分析和普通知识问答全部选择rag。"
            "一个问题可以拆成任意数量但最多10个子任务，不要因包含多个任务而选择clarify。"
            "语义判断以整句和上下文为准，不得根据单个词机械分类。比较任务必须存在至少两个明确比较对象和比较维度；"
            "对一个结果集合逐项列出名称、时间或属性属于枚举，不属于比较。"
            "先用case_identification产生满足条件的case_id集合；计数使用aggregate_statistics，"
            "列出名称、时间或其他字段使用case_listing，二者通过depends_on依赖同一个case_identification。"
            "任务type只能是knowledge_qa、case_identification、case_listing、case_analysis、forecast_warning_lookup、"
            "aggregate_statistics、metric_aggregation、comparison、impact_area_lookup、image_lookup、similar_case_match、"
            "report_generation、clarification。每个任务必须写成独立完整的问题并携带可执行conditions；"
            "后续任务用depends_on引用前面的任务ID，不得把独立任务的case_id范围互相继承。"
            "涉及预报或预警时必须单独拆成forecast_warning_lookup；生成报告和相似匹配必须分别使用专用任务类型。"
            "query_scope和conditions只允许start_date、end_date、years、months、disaster_types、cities、areas、"
            "time_scope_type、output_fields、analysis_fields、requested_outputs、comparison_targets、comparison_dimensions、"
            "metric、operator、unit、result_dimensions；日期必须为ISO格式。"
            "output_fields只填写个例编号、名称、时间、灾种、影响区域、来源和数量等目录字段；"
            "过程特征、实况、强度、影响、演变、环流、成因、预报或预警等需要读取正文的内容，"
            "必须放入analysis_fields并创建case_analysis任务，不能把它们当作case_listing字段。"
            "必须在requested_outputs中完整列出用户要求的所有回答目标，值只能从count、case_list、"
            "case_features、intensity、impact、impact_area、evolution、mechanism、circulation、weather_facts、"
            "evidence_summary、forecast_warning、image_evidence、comparison、report、metric_extrema中选择。"
            "requested_outputs是任务完整性契约：count需要aggregate_statistics，case_list需要case_listing，"
            "case_features及其他正文分析目标需要case_analysis，forecast_warning需要forecast_warning_lookup，"
            "image_evidence需要image_lookup；数值极值需要metric_aggregation，并给出规范metric、min或max算子和返回维度；"
            "metric必须使用执行器支持的规范值：air_temperature.minimum、air_temperature.maximum、"
            "precipitation.maximum、wind_speed.maximum或visibility.minimum，不得自造指标名称；"
            "询问某指标是多少、最大值、最小值及其时间地点属于metric_extrema，不属于count；"
            "count只表示个例次数或数量。即使一句话同时要求极值和过程分析，仍是rag，"
            "必须生成case_identification、metric_aggregation以及依赖极值结果的case_analysis；"
            "这些任务必须依赖产生候选范围的case_identification。"
            "全年表示完整自然年，只填写years和time_scope_type=calendar_year，不得按知识库范围缩短用户请求。"
            "知识库范围只用于后续覆盖提示。理由、线索和每个任务问题必须简洁，不超过80个汉字。"
            "无关的新问题必须is_related=false并原样保留。intent字段为必填项，即使其他字段不确定也不得省略。只输出JSON："
            '{"is_related":bool,"relation_type":"standalone或follow_up_reference或follow_up_topic",'
            '"reference_type":"none或knowledge_case或external_process或aggregate_result",'
            '"context_confidence":0到1,"standalone_message":"完整问题","referenced_case_ids":[],"reference_clues":[],'
            '"context_reason":"关联依据","intent":"任务类型","intent_confidence":0到1,"intent_reason":"路由依据",'
            '"requested_outputs":["回答目标"],'
            '"query_scope":{"years":[],"months":[],"start_date":"","end_date":"","disaster_types":[],'
            '"cities":[],"areas":[],"time_scope_type":"calendar_year或date_range或month_set或unbounded"},'
            '"tasks":[{"id":"task_1","type":"任务类型","question":"独立完整问题","depends_on":[],'
            '"conditions":{},"comparison_targets":[],"comparison_dimensions":[]}]}。'
        )
        compact_memory = self._intent_memory_view(memory)
        current_time = datetime.now().astimezone()
        model_input = {
            "本轮问题": message,
            "分层会话记忆": compact_memory,
            "当前业务时间": {
                "日期": current_time.date().isoformat(),
                # 分钟级时间足以消解近期表达，也避免秒级变化让同一分钟内的意图缓存失效。
                "时间": current_time.isoformat(timespec="minutes"),
                "时区": str(current_time.tzinfo or ""),
            },
            "知识库时间范围": knowledge.model_dump(),
            # 完整精简目录用于实体身份核验，不把正文 chunk 塞入意图请求。
            "标准个例目录": list(knowledge_cases.values()),
            "最近实际意图": previous_intent,
        }
        cache_key = self._model_cache_key(model_input)
        payload = self._read_model_cache(cache_key)
        try:
            if payload is None:
                payload = self._call_model_json(prompt, model_input, max_tokens=800)
                contract_errors = self._model_payload_contract_errors(payload)
                if contract_errors:
                    # 模型输出缺字段或重型路由与任务目标矛盾时，只对异常结果修复一次；
                    # 正常请求仍保持单次意图调用，不增加统一入口延迟。
                    repair_input = {
                        "本轮问题": message,
                        "当前业务时间": model_input["当前业务时间"],
                        "知识库时间范围": model_input["知识库时间范围"],
                        "标准个例目录": model_input["标准个例目录"],
                        "首次语义结果": payload,
                        "契约错误": contract_errors,
                    }
                    repair_prompt = (
                        f"{prompt}"
                        "首次结果违反了结构化契约，请根据契约错误重新输出完整JSON，不得只修补intent。"
                        "multidim_search必须同时声明report目标和report_generation任务；"
                        "某项气象指标的最大值、最小值、数值、时间或地点属于metric_extrema，"
                        "必须使用metric_aggregation，不得改写成个例count。"
                    )
                    repaired = self._call_model_json(repair_prompt, repair_input, max_tokens=800)
                    if not self._model_payload_contract_errors(repaired):
                        payload = repaired
                if payload:
                    self._write_model_cache(cache_key, payload)
        except Exception as exc:
            LOGGER.warning("[意图理解] 云端模型失败，使用确定性规则：%s", exc)
            return None, None
        related = bool(payload.get("is_related"))
        standalone = str(payload.get("standalone_message") or message).strip()
        available = {
            str(item.get("case_id")): item
            for item in memory.get("available_case_refs") or []
            if isinstance(item, dict) and item.get("case_id")
        }
        case_ids = []
        for case_id in payload.get("referenced_case_ids") or []:
            value = str(case_id)
            if value in available and value in knowledge_cases and value not in case_ids:
                case_ids.append(value)
        if not related:
            standalone = message
            case_ids = []
        reference_type = str(payload.get("reference_type") or "none")
        if reference_type not in {"none", "knowledge_case", "external_process", "aggregate_result"}:
            reference_type = "none"
        previous = memory.get("previous_turn") if isinstance(memory.get("previous_turn"), dict) else {}
        message_ids = [
            int(value)
            for value in (previous.get("user_message_id"), previous.get("assistant_message_id"))
            if value
        ] if related else []
        query_conditions = self._validate_query_conditions(payload.get("query_scope"))
        # 根级回答目标属于整个问题，而不是某个子任务；合并进条件供统一计划校验器使用。
        requested_outputs = self._validate_requested_outputs(payload.get("requested_outputs"))
        if requested_outputs:
            query_conditions["requested_outputs"] = requested_outputs
        context = ContextResolution(
            related,
            standalone,
            float(payload.get("context_confidence") or payload.get("confidence") or 0.0),
            relation_type=str(payload.get("relation_type") or ("follow_up_topic" if related else "standalone")),
            case_ids=case_ids,
            message_ids=message_ids,
            source="llm",
            reason=str(payload.get("context_reason") or payload.get("reason") or "云端模型完成会话关联判断")[:300],
            reference_type=("knowledge_case" if case_ids else reference_type),
            reference_clues=[str(item)[:80] for item in (payload.get("reference_clues") or [])[:8]],
            task_plan=[item for item in (payload.get("tasks") or []) if isinstance(item, dict)][:10],
            query_conditions=query_conditions,
        )

        intent = str(payload.get("intent") or "")
        model_decision = None
        if intent in {"rag", "multidim_search", "similar_case_match", "clarify"}:
            model_decision = CandidateDecision(
                intent,
                max(0.0, min(1.0, float(payload.get("intent_confidence") or payload.get("confidence") or 0.0))),
                str(payload.get("intent_reason") or payload.get("reason") or "云端模型完成任务分类")[:200],
                "llm",
            )
        return context, model_decision

    @staticmethod
    def _intent_memory_view(memory: dict[str, Any]) -> dict[str, Any]:
        """仅保留意图理解需要的分层记忆字段，控制云端请求体大小。"""
        if not isinstance(memory, dict):
            return {}
        previous = dict(memory.get("previous_turn") or {})
        if previous:
            previous["question"] = str(previous.get("question") or "")[:500]
            previous["answer_summary"] = str(previous.get("answer_summary") or "")[:500]
            previous["case_refs"] = list(previous.get("case_refs") or [])[:12]
        return {
            "version": memory.get("version", 2),
            "covered_message_id": memory.get("covered_message_id", 0),
            "long_term_summary": str(memory.get("long_term_summary") or "")[-1200:],
            "recent_turns": list(memory.get("recent_turns") or [])[-3:],
            "previous_turn": previous,
            "available_case_refs": list(memory.get("available_case_refs") or [])[:20],
            "previous_intent": memory.get("previous_intent"),
        }

    @staticmethod
    def _model_cache_key(model_input: dict[str, Any]) -> str:
        """用问题、记忆版本和知识范围生成稳定缓存键。"""
        raw = json.dumps(model_input, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _call_model_json(self, prompt: str, model_input: dict[str, Any], max_tokens: int) -> dict[str, Any]:
        """以兼容正式客户端和测试替身的方式调用模型并提取 JSON。"""
        call = self.llm_client.answer_with_context
        kwargs: dict[str, Any] = {"max_tokens": max_tokens}
        parameters = inspect.signature(call).parameters
        if "timeout_seconds" in parameters:
            kwargs["timeout_seconds"] = 15.0
        if "json_mode" in parameters:
            kwargs["json_mode"] = True
        raw = call(prompt, [json.dumps(model_input, ensure_ascii=False)], **kwargs)
        return self._json_object(str(raw or ""))

    def _read_model_cache(self, key: str) -> dict[str, Any] | None:
        """读取内存缓存并刷新最近使用顺序。"""
        with self._model_cache_lock:
            payload = self._model_cache.pop(key, None)
            if payload is None:
                return None
            self._model_cache[key] = payload
            return deepcopy(payload)

    def _write_model_cache(self, key: str, payload: dict[str, Any]) -> None:
        """写入有上限的进程内缓存，避免长期运行无限增长。"""
        with self._model_cache_lock:
            self._model_cache.pop(key, None)
            self._model_cache[key] = deepcopy(payload)
            while len(self._model_cache) > self._model_cache_limit:
                self._model_cache.popitem(last=False)

    @staticmethod
    def _model_payload_contract_errors(payload: dict[str, Any]) -> list[str]:
        """检查意图、回答目标和终端任务是否互相一致。"""
        valid_intents = {"rag", "multidim_search", "similar_case_match", "clarify"}
        intent = str(payload.get("intent") or "")
        outputs = set(QueryIntentAgent._validate_requested_outputs(payload.get("requested_outputs")))
        tasks = [item for item in (payload.get("tasks") or []) if isinstance(item, dict)]
        task_types = {str(item.get("type") or "") for item in tasks}
        errors: list[str] = []
        if intent not in valid_intents:
            errors.append("intent缺失或不在白名单")
        if intent == "multidim_search" and not (
            "report" in outputs and "report_generation" in task_types
        ):
            errors.append("multidim_search缺少report目标或report_generation任务")
        if intent == "similar_case_match" and "similar_case_match" not in task_types:
            errors.append("similar_case_match缺少同名终端任务")
        if "metric_extrema" in outputs:
            metric_tasks = [item for item in tasks if item.get("type") == "metric_aggregation"]
            if not metric_tasks:
                errors.append("metric_extrema缺少metric_aggregation任务")
            else:
                conditions = dict(metric_tasks[0].get("conditions") or {})
                if str(conditions.get("metric") or "") not in METRIC_CONTRACT_VALUES:
                    errors.append("metric_aggregation缺少受支持的规范metric")
                if str(conditions.get("operator") or "") not in {"min", "max"}:
                    errors.append("metric_aggregation缺少min或max算子")
        return errors

    @staticmethod
    def _has_context_reference(text: str) -> bool:
        """识别省略了“个例/过程”名词的常见中文承接表达。"""
        terms = (
            "上述", "上面", "上文", "前面", "这些", "那些", "这两个", "那两个", "这几个", "那几个",
            "这几次", "那几次", "它们", "他们", "分别分析", "继续分析", "接着分析", "进一步分析",
            "这个个例", "这个过程", "该个例", "该过程", "其中一个", "这个灾害", "这次灾害", "这次过程",
        )
        return any(term in text for term in terms) or bool(
            re.search(r"(?:[这那]|第)(?:一|二|两|三|四|五|六|七|八|九|十|\d+)(?:个|次)", text)
            or re.search(r"[这那](?:个|次)?(?:历史)?(?:个例|过程|灾害|事件)", text)
        )

    @staticmethod
    def _referenced_count(text: str) -> int | None:
        """提取“这两个、那3次”等指代数量。"""
        match = re.search(r"[这那]?(一|二|两|三|四|五|六|七|八|九|十|\d+)(?:个|次)", text)
        if not match:
            return None
        value = match.group(1)
        if value.isdigit():
            return int(value)
        return {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}.get(value)

    @staticmethod
    def _months_in_text(text: str) -> list[int]:
        """提取问题中用于筛选历史个例的月份。"""
        return list(dict.fromkeys(int(item) for item in re.findall(r"(\d{1,2})\s*月", text) if 1 <= int(item) <= 12))

    @staticmethod
    def _case_ref_matches_month(case_ref: dict[str, Any], months: list[int]) -> bool:
        """根据标题、时段和来源判断记忆个例是否属于指定月份。"""
        text = " ".join(str(case_ref.get(key) or "") for key in ("title", "date_range", "source_pdf"))
        return any(re.search(rf"(?<!\d){month}\s*月", text) for month in months)

    @staticmethod
    def _date_clues(text: str) -> list[str]:
        """提取用户明确写出的年月日线索，避免仅按月份选中整月所有个例。"""
        clues: list[str] = []
        pattern = r"(?:(20\d{2})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*(?:日|号)?"
        for match in re.finditer(pattern, text):
            year, month, day = match.groups()
            clue = f"{year + '年' if year else ''}{int(month)}月{int(day)}日"
            if clue not in clues:
                clues.append(clue)
        return clues

    @staticmethod
    def _case_ref_matches_date_clues(case_ref: dict[str, Any], clues: list[str]) -> bool:
        """判断日期线索是否落入个例时段，支持“1月23-26日”等范围。"""
        text = " ".join(str(case_ref.get(key) or "") for key in ("title", "date_range"))
        interval_pattern = (
            r"(?:(20\d{2})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*(?:日|号)?"
            r"(?:\s*(?:至|到|[-—~～])\s*(?:(\d{1,2})\s*月\s*)?(\d{1,2})\s*(?:日|号)?)?"
        )
        intervals: list[tuple[int | None, int, int, int, int]] = []
        for match in re.finditer(interval_pattern, text):
            year, start_month, start_day, end_month, end_day = match.groups()
            intervals.append((
                int(year) if year else None,
                int(start_month),
                int(start_day),
                int(end_month or start_month),
                int(end_day or start_day),
            ))
        for clue in clues:
            match = re.fullmatch(r"(?:(20\d{2})年)?(\d{1,2})月(\d{1,2})日", clue)
            if not match:
                continue
            clue_year, clue_month, clue_day = match.groups()
            clue_key = (int(clue_month), int(clue_day))
            for year, start_month, start_day, end_month, end_day in intervals:
                if clue_year and year and int(clue_year) != year:
                    continue
                if (start_month, start_day) <= clue_key <= (end_month, end_day):
                    return True
        return False

    @staticmethod
    def _expects_single_reference(text: str) -> bool:
        """识别语义上要求唯一历史实体的表达，仅用于模型不可用时的消歧兜底。"""
        terms = ("这个", "该个例", "该过程", "其中一个", "这次", "这一", "那个")
        return any(term in text for term in terms)

    def _knowledge_case_catalog(self) -> dict[str, dict[str, Any]]:
        """读取当前主标准个例，作为模型输出和会话实体的唯一有效 ID 集合。"""
        try:
            if self.case_provider is not None:
                records = list(self.case_provider() or [])
            else:
                records = json.loads(self.knowledge_reader.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        catalog: dict[str, dict[str, Any]] = {}
        for record in records:
            if isinstance(record, dict):
                item = record
            else:
                item = {
                    key: getattr(record, key, None)
                    for key in ("case_id", "title", "date_range", "disaster_types", "affected_areas", "source_pdf")
                }
            case_id = str(item.get("case_id") or "").strip()
            if not case_id:
                continue
            catalog[case_id] = {
                "case_id": case_id,
                "title": str(item.get("title") or ""),
                "date_range": str(item.get("date_range") or ""),
                "disaster_types": [str(value) for value in (item.get("disaster_types") or [])[:8]],
                "affected_areas": [str(value) for value in (item.get("affected_areas") or [])[:8]],
                "source_pdf": str(item.get("source_pdf") or ""),
                "reference_type": "knowledge_case",
            }
        return catalog

    @staticmethod
    def _sanitize_memory_context(
        memory: dict[str, Any],
        catalog: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """在提交给模型前移除当前主库不存在的旧 ID，防止历史摘要诱导模型创造实体。"""
        sanitized = deepcopy(memory) if isinstance(memory, dict) else {}

        def valid_refs(items: list[Any]) -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = []
            seen: set[str] = set()
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                case_id = str(item.get("case_id") or "").strip()
                if case_id in catalog and case_id not in seen:
                    seen.add(case_id)
                    result.append(dict(catalog[case_id]))
            return result

        sanitized["available_case_refs"] = valid_refs(sanitized.get("available_case_refs") or [])
        sanitized["long_term_case_refs"] = valid_refs(sanitized.get("long_term_case_refs") or [])
        previous = sanitized.get("previous_turn")
        if isinstance(previous, dict):
            previous["case_refs"] = valid_refs(previous.get("case_refs") or [])
        for turn in sanitized.get("recent_turns") or []:
            if isinstance(turn, dict):
                turn["case_refs"] = valid_refs(turn.get("case_refs") or [])
        return sanitized

    def normalize_text(self, text: str) -> str:
        """只归一化日期中的中文数字和空白，不改写其他业务内容。"""
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()

        def replace_number(match: re.Match[str]) -> str:
            """把紧邻年月日的中文数字转换成阿拉伯数字。"""
            value = self._chinese_number(match.group(1))
            return str(value) if value is not None else match.group(1)

        return re.sub(r"([〇零一二两三四五六七八九十]{1,8})(?=[年月日])", replace_number, normalized)

    def _chinese_number(self, value: str) -> int | None:
        """解析日期常见的中文数字，覆盖年份逐字写法和一至九十九。"""
        digits = {"〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        if value and all(char in digits for char in value):
            return int("".join(str(digits[char]) for char in value))
        if "十" in value:
            left, right = value.split("十", 1)
            tens = digits.get(left, 1) if left else 1
            ones = digits.get(right, 0) if right else 0
            return tens * 10 + ones
        return None

    def _extract_conditions(self, text: str) -> dict[str, Any]:
        """使用既有确定性规则提取条件，模型结果不得覆盖这些事实字段。"""
        extraction = self.condition_parser._rule_extract(text)
        values = extraction.values
        result: dict[str, Any] = {}
        for key in ("start_date", "end_date", "years", "months", "disaster_types", "cities", "areas"):
            value = values.get(key)
            if value:
                result[key] = value
        return result

    def _rule_decision(
        self,
        text: str,
        conditions: dict[str, Any],
        knowledge: KnowledgeRange,
        previous_intent: IntentName | None,
        context: ContextResolution,
    ) -> CandidateDecision:
        """以严格业务门槛区分两个重型 Agent，其余请求全部留在主 RAG。"""
        asks_report = self._asks_for_report(text)
        relation = self._time_relation(conditions, knowledge)
        asks_risk_analysis = any(term in text for term in RISK_ANALYSIS_TERMS)
        has_knowledge_case = bool(context.case_ids)
        describes_new_process = relation in {"after", "before"} or self._describes_current_or_future_process(text)
        if asks_report:
            return CandidateDecision("multidim_search", 0.99, "用户明确要求生成或导出分析报告。", "rule", True)
        if has_knowledge_case:
            return CandidateDecision("rag", 0.99, "本轮已定位到当前标准个例库中的历史个例，限定其正文证据交由主 RAG 分析。", "knowledge_guard", True)
        if context.reference_type == "aggregate_result":
            return CandidateDecision("rag", 0.98, "本轮承接普通聚合结果，继续交由主 RAG 处理。", "context_guard", True)
        if context.reference_type == "external_process" and asks_risk_analysis:
            describes_new_process = True
        if describes_new_process and asks_risk_analysis:
            return CandidateDecision("similar_case_match", 0.98, "用户要求分析知识库范围外或当前未来的风险过程。", "rule", True)
        # 简单计数、库内查询、知识问答和普通分析都属于主 RAG，不能误触发重型 Agent。
        return CandidateDecision("rag", 0.92, "未满足两个子 Agent 的严格调用条件，交由主 RAG 处理。", "rule")

    @staticmethod
    def _describes_current_or_future_process(text: str) -> bool:
        """识别真正的新过程表达，避免把“预报提示、预报效果、预报复盘”等历史材料误判为当前过程。"""
        if any(term in text for term in CURRENT_TERMS):
            return True
        forecast_context = re.sub(r"预报(?:提示|效果|复盘|服务|检验|结论|准确率)", "", text)
        return "预报" in forecast_context and any(term in forecast_context for term in ("将", "会", "可能", "发生", "来袭"))

    def _model_decision(
        self,
        text: str,
        conditions: dict[str, Any],
        knowledge: KnowledgeRange,
        previous_intent: IntentName | None,
        rule: CandidateDecision,
    ) -> CandidateDecision | None:
        """让大模型只判断任务类型，不接受其生成的业务条件。"""
        # 明确的业务动作已经由规则高置信确定，跳过额外模型调用可降低延迟，
        # 同时避免模型误把“检索”和“相似匹配”等明确指令改判。
        if rule.hard or (rule.intent == "rag" and rule.confidence >= 0.9):
            # 简单聚合和普通问答已有稳定规则结论，不再为路由额外等待一次云端模型。
            return None
        try:
            if not self.llm_client or not self.llm_client.is_available():
                return None
        except Exception:
            return None
        instruction = (
            "你是气象个例系统的意图识别器，只判断任务类型，不回答业务问题。"
            "只输出JSON，intent只能是rag、multidim_search、similar_case_match、clarify。"
            "只有用户明确要求生成或导出报告时才选择multidim_search；"
            "只有用户描述当前、未来或知识库范围外的新风险过程并要求分析时才选择similar_case_match；"
            "简单统计、库内查询和其他知识问答全部选择rag；"
            "clarify仅表示目标实体或路由确实无法判断；一句话包含多个任务时必须选择可执行的主路由，"
            "并在tasks中拆分全部子任务，不得因为任务数量多而clarify，不得输出mixed_task。"
            "不能创造用户未提供的日期、灾种、地区和天气事实。"
            "输出格式：{\"intent\":\"...\",\"confidence\":0到1,\"reason\":\"简短依据\"}。"
        )
        context = {
            "用户消息": text,
            "规则提取条件": conditions,
            "知识库时间范围": knowledge.model_dump(),
            "最近实际意图": previous_intent,
            "规则候选": {"intent": rule.intent, "confidence": rule.confidence, "reason": rule.reason},
        }
        try:
            raw = self.llm_client.answer_with_context(
                instruction,
                [json.dumps(context, ensure_ascii=False)],
                max_tokens=320,
            )
            payload = self._json_object(str(raw or ""))
            intent = str(payload.get("intent") or "")
            confidence = float(payload.get("confidence") or 0)
            reason = str(payload.get("reason") or "大模型语义判断")[:200]
        except Exception as exc:
            LOGGER.warning("[意图分发] 大模型判断失败，使用规则候选：%s", exc)
            return None
        if intent not in {"rag", "multidim_search", "similar_case_match", "clarify"}:
            return None
        return CandidateDecision(intent, max(0.0, min(1.0, confidence)), reason, "llm")

    def _merge_decisions(
        self,
        rule: CandidateDecision,
        model: CandidateDecision | None,
    ) -> tuple[CandidateDecision, str | None]:
        """按置信度门槛合并候选，冲突时宁可追问也不错误执行。"""
        if rule.hard:
            source = "rule_confirmed_by_llm" if model and model.intent == rule.intent else rule.source
            return CandidateDecision(rule.intent, rule.confidence, rule.reason, source, True), None
        if model is None:
            if rule.intent != "clarify" and rule.confidence >= 0.85:
                return CandidateDecision(rule.intent, rule.confidence, rule.reason, "rule_fallback"), None
            suggested = rule.intent if rule.intent != "clarify" else None
            return CandidateDecision("clarify", rule.confidence, rule.reason, "rule_fallback"), suggested
        if model.intent == rule.intent and model.intent != "clarify":
            return CandidateDecision(model.intent, max(rule.confidence, model.confidence), model.reason, "rule_and_llm"), None
        if (
            rule.intent == "rag"
            and rule.source == "rule"
            and model.intent in {"multidim_search", "similar_case_match"}
            and model.confidence >= 0.85
        ):
            # 默认 RAG 只是兜底候选；模型结合完整目录确认了重型任务后应采用语义判断。
            # 明确报告、明确库内个例等硬保护已在前面返回，不会被此分支覆盖。
            return CandidateDecision(model.intent, model.confidence, model.reason, "llm_over_default_rag", True), None
        if rule.intent == "clarify" and model.intent != "clarify" and model.confidence >= 0.85:
            return model, None
        if model.intent == "clarify" and rule.intent != "clarify" and rule.confidence >= 0.85:
            return CandidateDecision(rule.intent, rule.confidence, rule.reason, "rule"), None
        suggested = None
        for candidate in sorted((rule, model), key=lambda item: item.confidence, reverse=True):
            if candidate.intent != "clarify":
                suggested = candidate.intent
                break
        reason = "规则与语义判断不一致，需要用户确认本轮任务类型。"
        confidence = max(rule.confidence, model.confidence)
        return CandidateDecision("clarify", min(confidence, 0.84), reason, "conflict"), suggested

    def _time_relation(self, conditions: dict[str, Any], knowledge: KnowledgeRange) -> str:
        """判断用户时间条件与知识库范围的关系。"""
        knowledge_start = self._iso_date(knowledge.start_date)
        knowledge_end = self._iso_date(knowledge.end_date)
        if not knowledge_start or not knowledge_end:
            return "unknown"
        requested = self._condition_interval(conditions)
        if requested is None:
            return "unknown"
        start, end = requested
        if start > knowledge_end:
            return "after"
        if end < knowledge_start:
            return "before"
        return "inside"

    def _condition_interval(self, conditions: dict[str, Any]) -> tuple[date, date] | None:
        """把确定性条件换算成用于边界判断的日期区间。"""
        start = self._iso_date(str(conditions.get("start_date") or ""))
        end = self._iso_date(str(conditions.get("end_date") or ""))
        if start or end:
            return start or end, end or start
        years = sorted({int(value) for value in conditions.get("years") or [] if 1900 <= int(value) <= 2100})
        months = sorted({int(value) for value in conditions.get("months") or [] if 1 <= int(value) <= 12})
        if not years:
            return None
        first_month, last_month = (months[0], months[-1]) if months else (1, 12)
        first = date(years[0], first_month, 1)
        if last_month == 12:
            last = date(years[-1], 12, 31)
        else:
            from calendar import monthrange

            last = date(years[-1], last_month, monthrange(years[-1], last_month)[1])
        return first, last

    def _target(self, intent: IntentName) -> tuple[str, str]:
        """把白名单意图映射到现有业务 Agent 入口。"""
        if intent == "rag":
            return "主 RAG", "/api/query/stream"
        if intent == "multidim_search":
            return "多维检索智能体", "/api/case-multidim/natural/parse"
        if intent == "similar_case_match":
            return "相似个例智能体", "/api/smart-case-match/conversation/stream"
        return "", ""

    def _asks_for_report(self, text: str) -> bool:
        """识别明确的报告生成动作，单独提到“报告”不触发耗时 Agent。"""
        compact = re.sub(r"\s+", "", text)
        if any(term.lower() in compact.lower() for term in REPORT_TERMS):
            return True
        # 年份、月份、灾种等检索条件通常位于动作和报告之间，因此不能使用过短的固定窗口。
        return bool(
            re.search(
                r"(?:帮我|请|需要|要|给我)?(?:生成|制作|导出|形成|整理).{0,80}(?:PDF|报告)",
                compact,
                re.IGNORECASE,
            )
        )

    def _rag_strategy(self, text: str, conditions: dict[str, Any]) -> str:
        """普通 RAG 中，聚合与条件清单优先标准个例，其余问题使用向量检索。"""
        # 明确要求深入解释时必须读取关联 chunk 正文，不能停留在标准个例八字段摘要。
        if any(term in text for term in ("详细分析", "深入分析", "分析一下", "进一步分析", "成因", "环流", "复盘")):
            return "vector"
        if any(term in text for term in AGGREGATION_TERMS):
            return "structured"
        if conditions and any(term in text for term in STRUCTURED_QUERY_TERMS):
            return "structured"
        return "vector"

    def _json_object(self, raw: str) -> dict[str, Any]:
        """过滤思考标签并提取模型返回的第一个完整 JSON 对象。"""
        text = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _iso_date(self, value: str) -> date | None:
        """安全解析 ISO 日期。"""
        try:
            return date.fromisoformat(value) if value else None
        except ValueError:
            return None

    def _write_audit(
        self,
        audit_id: str,
        original: str,
        normalized: str,
        conditions: dict[str, Any],
        previous_intent: IntentName | None,
        rule: CandidateDecision,
        model: CandidateDecision | None,
        response: IntentRouteResponse,
    ) -> None:
        """记录原文、候选、最终分发和目标状态，便于定位误分发。"""
        self.audit_writer.write({
            "audit_id": audit_id,
            "created_at": datetime.now().astimezone().isoformat(),
            "original_message": original,
            "normalized_message": normalized,
            "context_related": response.context_related,
            "relation_type": response.relation_type,
            "reference_type": response.reference_type,
            "context_confidence": response.context_confidence,
            "referenced_case_ids": response.referenced_case_ids,
            "knowledge_case_ids": response.knowledge_case_ids,
            "reference_clues": response.reference_clues,
            "context_message_ids": response.context_message_ids,
            "context_resolution_source": response.context_resolution_source,
            "conditions": conditions,
            "previous_intent": previous_intent,
            "rule_result": {"intent": rule.intent, "confidence": rule.confidence, "reason": rule.reason},
            "llm_result": (
                {"intent": model.intent, "confidence": model.confidence, "reason": model.reason}
                if model
                else None
            ),
            "final_intent": response.intent,
            "final_confidence": response.confidence,
            "routing_source": response.routing_source,
            "target_agent": response.target_agent,
            "need_clarification": response.need_clarification,
            "tasks": [item.model_dump() for item in response.tasks],
            "knowledge_range": response.knowledge_range.model_dump(),
        })
