"""Smart Case Match LangGraph 的单一职责节点集合。"""
from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter, sleep
from typing import Any

from ..infrastructure.data_store import LocalCaseDataStore
from ..matching.dimension_profiles import resolve_dimension_profile
from ..matching.enrichment import enrich_selected_cases
from ..llm.llm_schemas import VALID_REFERENCE_SOURCES
from ..llm.llm_service import SmartCaseLlmService
from ..matching.matching import add_semantic_scores, fuse_scores, normalize_query, structured_recall
from ..matching.selection import apply_llm_tie_break, select_cases
from ..matching.semantic_rerank import rerank_semantic_hits
from . import progress

MAX_PRIMARY_EVIDENCE_IMAGES = 6
MAX_SUPPLEMENTAL_IMAGES = 6
CASE_REFERENCE_MAX_WORKERS = 2
CASE_REFERENCE_FINAL_RETRY_DELAY_SECONDS = 0.8
# 逐例提炼经过原始校验或协议修复后都属于有效LLM结果，不能被误计为降级。
VALID_REFERENCE_LLM_STATUSES = {"called", "called_repaired"}

# 参考经验、文字证据和图片图注共用同一套业务主题，避免仅凭同页或同 chunk 产生伪关联。
EVIDENCE_TOPIC_TERMS = {
    "降水": ("降水", "雨强", "降雨", "降水量", "累计雨量", "大暴雨", "暴雨"),
    "风": ("大风", "阵风", "雷暴大风", "风速", "风力", "风廓线"),
    "温度": ("气温", "降温", "变温", "低温", "高温", "零度线", "0℃线", "冷垫"),
    "雪": ("降雪", "积雪", "雪深", "雨雪", "暴雪", "雨夹雪", "相态"),
    "环流": (
        "500hPa", "700hPa", "850hPa", "副高", "切变", "急流", "水汽", "环流", "气压场",
        "高度场", "温度场", "槽", "脊", "阻塞高压", "冷高压", "冷空气", "气旋", "冷锋",
    ),
    "雷达": ("雷达", "回波", "组合反射率", "短时强降水"),
    "沙尘": ("沙尘", "扬沙", "浮尘", "能见度", "退偏振比"),
    "能见度": ("能见度", "浓雾", "大雾", "雾"),
    "对流": ("强对流", "雷暴", "冰雹", "对流云"),
}

# 当前查询主题允许使用的辅助证据主题，例如雪过程可由温度场和环流配置图共同支撑。
QUERY_TOPIC_COMPATIBILITY = {
    # 雪过程的普通降水量图不能直接说明降雪、积雪或相态，因此不再把“降水”作为默认辅助主题。
    "雪": {"雪", "温度", "环流", "雷达"},
    "降水": {"降水", "环流", "雷达", "对流"},
    "风": {"风", "环流", "对流", "沙尘"},
    "温度": {"温度", "环流", "雪"},
    "环流": {"环流", "温度", "降水", "风", "雪", "对流"},
    "雷达": {"雷达", "降水", "对流", "雪"},
    "沙尘": {"沙尘", "风", "环流"},
    "能见度": {"能见度", "温度", "环流", "风"},
    "对流": {"对流", "雷达", "降水", "风", "环流"},
}



class SmartCaseGraphNodes:
    """保存节点依赖，并为 StateGraph 提供可观测的节点函数。"""

    def __init__(
        self,
        store: LocalCaseDataStore,
        embedding_client: Any,
        rerank_client: Any,
        llm_service: SmartCaseLlmService,
    ):
        self.store = store
        self.embedding_client = embedding_client
        self.rerank_client = rerank_client
        self.llm_service = llm_service

    def normalize_input(self, state: dict[str, Any]) -> dict[str, Any]:
        """标准化页面输入，并加载本地个例列表。"""
        started = perf_counter()
        query = normalize_query(state["request"])
        progress.update(state["run_id"], "normalize_input", "输入已标准化，正在扫描本地个例库", 8)
        return {
            "query": query,
            "all_cases": list(self.store.cases),
            "warnings": list(state.get("warnings") or []),
            "audit": self._audit(state, "normalize_input", started, case_count=len(self.store.cases)),
        }

    def structured_recall(self, state: dict[str, Any]) -> dict[str, Any]:
        """执行地区、灾种和季节主导的结构化召回。"""
        started = perf_counter()
        candidates = structured_recall(state["query"], state["all_cases"])
        progress.update(state["run_id"], "structured_recall", f"已召回 {len(candidates)} 个结构化候选", 22)
        return {
            "candidate_cases": candidates,
            "audit": self._audit(state, "structured_recall", started, candidate_count=len(candidates)),
        }

    def semantic_supplement(self, state: dict[str, Any]) -> dict[str, Any]:
        """在结构化候选所属 chunk 内补充语义分。"""
        started = perf_counter()
        candidates, warnings = add_semantic_scores(
            state["query"],
            state["candidate_cases"],
            self.store,
            self.embedding_client,
        )
        candidates, rerank_status, rerank_warnings = rerank_semantic_hits(
            state["query"],
            candidates,
            self.store,
            self.rerank_client,
        )
        semantic_count = sum(1 for item in candidates if item.get("semantic_available"))
        progress.update(state["run_id"], "semantic_supplement", "向量召回与 Rerank 精排完成", 38)
        return {
            "candidate_cases": candidates,
            "warnings": list(state.get("warnings") or []) + warnings + rerank_warnings,
            "audit": self._audit(
                state,
                "semantic_supplement",
                started,
                semantic_case_count=semantic_count,
                embedding_status="called" if semantic_count else "not_available",
                semantic_rerank_status=rerank_status,
            ),
        }

    def rank_and_select(self, state: dict[str, Any]) -> dict[str, Any]:
        """融合排序、LLM 近分微调并按质量门槛选出最终个例。"""
        started = perf_counter()
        fused = self._attach_semantic_evidence(fuse_scores(state["candidate_cases"]), state["query"])
        requested_count = int(state["query"].get("top_n") or 3)
        diversity_mode = str(state["query"].get("diversity_mode") or "moderate")
        dimension_profile = resolve_dimension_profile(state["query"])
        # 每次固定调用候选重排模型，确保所有请求都经过统一的业务参考价值判断。
        # 客户端与 HTTP 连接池仍由 LLM 服务复用，本次恢复不会撤销底层连接复用优化。
        progress.update(state["run_id"], "rank_and_select", "正在判断候选个例的实际参考价值", 45)
        ordered_ids, assessment_map, llm_status = self.llm_service.rerank(
            state["query"],
            fused[:15],
            deadline=state.get("deadline_monotonic"),
        )
        reranked = apply_llm_tie_break(
            fused,
            ordered_ids,
            assessment_map,
            dimension_profile=dimension_profile,
        )
        selected, warnings = select_cases(
            reranked,
            requested_count,
            diversity_mode,
        )
        if llm_status not in {"called", "called_partial"}:
            warnings.append(f"LLM 候选重排未成功（{llm_status}），已保留融合分排序。")
        # 保存候选阶段的完整决策轨迹，供每次请求结束后的独立 JSON 审计文件使用。
        candidate_rank_audit = {
            "llm_status": llm_status,
            "llm_candidate_ids": [str(item.get("case_id") or "") for item in fused[:15]],
            "llm_ordered_case_ids": list(ordered_ids),
            "fused_scores_before_llm": {
                str(item.get("case_id") or ""): float(item.get("retrieval_score") or 0.0)
                for item in fused
            },
            "assessments": {
                case_id: {
                    "dimension_scores": dict(value.get("dimension_scores") or {}),
                    "missing_dimensions": list(value.get("missing_dimensions") or []),
                    "metric_scores": dict(value.get("metric_scores") or {}),
                    "missing_metrics": list(value.get("missing_metrics") or []),
                    "assessment_source": str(value.get("assessment_source") or ""),
                }
                for case_id, value in assessment_map.items()
            },
        }
        return {
            "scored_cases": reranked,
            "selected_cases": selected,
            "candidate_rank_audit": candidate_rank_audit,
            "warnings": list(state.get("warnings") or []) + warnings,
            "audit": self._audit(
                state,
                "rank_and_select",
                started,
                fused_count=len(fused),
                selected_count=len(selected),
                llm_rerank_status=llm_status,
                candidate_assessment_count=len(assessment_map),
                dimension_profile=dimension_profile.get("profile_id"),
                active_dimensions=[item.get("key") for item in dimension_profile.get("dimensions") or []],
            ),
        }

    def enrich_cases(self, state: dict[str, Any]) -> dict[str, Any]:
        """只做入选个例的正文、过程前预警和图片证据准备。"""
        started = perf_counter()
        enriched = enrich_selected_cases(state["query"], state["selected_cases"], self.store)
        progress.update(state["run_id"], "enrich_cases", "已准备个例正文、图片和过程前预警证据", 65)
        return {
            "enriched_cases": enriched,
            "audit": self._audit(
                state,
                "enrich_cases",
                started,
                enriched_count=len(enriched),
                chunk_count=sum(len(item.get("relevant_chunks") or []) for item in enriched),
            ),
        }

    def extract_case_references(self, state: dict[str, Any]) -> dict[str, Any]:
        """逐个例独立提炼参考点，防止低质量个例污染其他结果。"""
        started = perf_counter()
        references: list[dict[str, Any] | None] = [None] * len(state.get("enriched_cases") or [])
        statuses: Counter[str] = Counter()
        cases = state.get("enriched_cases") or []
        total = max(1, len(cases))
        if cases:
            worker_count = min(CASE_REFERENCE_MAX_WORKERS, len(cases))
            progress.update(state["run_id"], "extract_case_references", f"正在并发提炼 {len(cases)} 个历史个例", 68)
            # 每个个例相互独立，使用两个工作线程减少模型调用的串行等待时间。
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="smart-case-reference") as executor:
                future_indexes = {
                    executor.submit(
                        self._extract_one_reference_with_retry,
                        state,
                        case,
                    ): index
                    for index, case in enumerate(cases)
                }
                completed = 0
                for future in as_completed(future_indexes):
                    index = future_indexes[future]
                    # 并发完成顺序与个例顺序不同，必须通过索引取回本次结果对应的个例。
                    case = cases[index]
                    try:
                        reference, status = future.result()
                    except Exception as exc:
                        # 单个例失败不能污染其他个例；保留该个例并生成可溯源规则摘要。
                        fallback = getattr(self.llm_service, "_fallback_case_reference", None)
                        reference = fallback(state["query"], case) if fallback else {
                            "case_id": str(case.get("case_id") or ""),
                            "reference_points": [],
                            "similarities": [],
                            "differences": ["该个例提炼失败，需人工核验。"],
                            "warning_references": [],
                        }
                        status = "failed"
                    # LLM 只能选择真实 chunk ID；这里再校验经验文字与 chunk 内容是否属于同一业务主题。
                    reference = self._validate_reference_evidence(state["query"], reference, case)
                    if not reference.get("reference_points"):
                        fallback = getattr(self.llm_service, "_fallback_case_reference", None)
                        if fallback:
                            reference = self._validate_reference_evidence(
                                state["query"],
                                fallback(state["query"], case),
                                case,
                            )
                        # 只有模型证据确实全部被过滤且启用了规则兜底，才标记为证据降级；
                        # 有效模型结果必须保持 called，避免成功调用被错误统计为失败。
                        if status in VALID_REFERENCE_LLM_STATUSES:
                            status = "invalid_evidence"
                    reference_source = str(reference.get("reference_source") or "")
                    if status in VALID_REFERENCE_LLM_STATUSES and reference.get("reference_points"):
                        reference_source = reference_source or ("llm_repaired" if status == "called_repaired" else "llm_valid")
                    else:
                        # 规则结果只用于个例卡片兜底，后续综合节点会明确排除，避免形成伪共识。
                        reference_source = "rule_fallback"
                    # 把候选阶段的机制/强度评估随逐例结果传给综合节点，避免综合研判把弱机制个例当成同等依据。
                    reference = {
                        **reference,
                        "reference_source": reference_source,
                        "reference_status": status,
                        "case_title": str(case.get("title") or ""),
                        "date_range": str(case.get("date_range") or ""),
                        "retrieval_score": float(case.get("retrieval_score") or 0.0),
                        "mechanism_score": case.get("mechanism_score"),
                        "intensity_score": case.get("intensity_score"),
                        "dimension_scores": dict(case.get("dimension_scores") or {}),
                        "metric_scores": dict(case.get("metric_scores") or {}),
                        "missing_metrics": list(case.get("missing_metrics") or []),
                        "missing_dimensions": list(case.get("missing_dimensions") or []),
                        "dimension_compatibility": case.get("dimension_compatibility"),
                        "dimension_raw_compatibility": case.get("dimension_raw_compatibility"),
                        "dimension_coverage": case.get("dimension_coverage"),
                        "dimension_profile": case.get("dimension_profile"),
                        "dimension_assessment_source": case.get("dimension_assessment_source"),
                    }
                    # 按原入选顺序写回，避免并发完成顺序改变页面排名。
                    references[index] = reference
                    statuses[status] += 1
                    completed += 1
                    progress.update(
                        state["run_id"],
                        "extract_case_references",
                        f"已完成 {completed}/{len(cases)} 个历史个例提炼",
                        68 + int(14 * completed / total),
                    )
        progress.update(state["run_id"], "extract_case_references", "逐个例参考提炼完成", 82)
        warnings = list(state.get("warnings") or [])
        failed = sum(count for status, count in statuses.items() if status not in VALID_REFERENCE_LLM_STATUSES)
        if failed:
            warnings.append(f"有 {failed} 个历史个例未完成有效 LLM 提炼，已明确降级为可溯源规则摘要。")
        return {
            "case_references": [item for item in references if item is not None],
            "warnings": warnings,
            "audit": self._audit(
                state,
                "extract_case_references",
                started,
                reference_statuses=dict(statuses),
                reference_max_workers=CASE_REFERENCE_MAX_WORKERS,
            ),
        }

    def _extract_one_reference(
        self,
        state: dict[str, Any],
        case: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        """调用逐例提炼，并兼容不支持截止时间参数的本地测试或旧适配器。"""
        try:
            return self.llm_service.extract_case_reference(
                state["query"],
                case,
                deadline=state.get("deadline_monotonic"),
            )
        except TypeError as exc:
            if "deadline" not in str(exc):
                raise
            return self.llm_service.extract_case_reference(state["query"], case)

    def _extract_one_reference_with_retry(
        self,
        state: dict[str, Any],
        case: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        """逐例提炼未通过时等待后再完整重试一次，覆盖偶发模型漂移和证据过滤失败。"""
        reference, status = self._extract_one_reference(state, case)
        validated = self._validate_reference_evidence(state["query"], reference, case)
        needs_retry = status not in VALID_REFERENCE_LLM_STATUSES or not validated.get("reference_points")
        if not needs_retry or status == "not_available":
            return validated, status

        # 最终重试仍在逐例线程池内执行，多个异常个例可以并行恢复；短退避避免连续命中同一瞬时状态。
        sleep(CASE_REFERENCE_FINAL_RETRY_DELAY_SECONDS)
        logger_message = (
            f"逐例提炼最终重试 case_id={case.get('case_id')} previous_status={status}"
        )
        progress.update(state["run_id"], "extract_case_references", logger_message, 76)
        try:
            retried_reference, retried_status = self._extract_one_reference(state, case)
            retried_reference = self._validate_reference_evidence(
                state["query"],
                retried_reference,
                case,
            )
            if retried_status in VALID_REFERENCE_LLM_STATUSES and retried_reference.get("reference_points"):
                return retried_reference, retried_status
            return retried_reference, retried_status
        except Exception:
            # 最终重试异常时保留第一次结果，外层仍会按既有规则安全降级，不让单例击穿整条图。
            return validated, status

    def synthesize_forecast_tips(self, state: dict[str, Any]) -> dict[str, Any]:
        """从多个独立个例参考中生成跨个例预报提示。"""
        started = perf_counter()
        all_references = list(state.get("case_references") or [])
        valid_references = [
            item for item in all_references
            if str(item.get("reference_source") or "") in VALID_REFERENCE_SOURCES
            and item.get("reference_points")
        ]
        # 综合研判只接收有效LLM参考；规则摘要仍可展示，但不能与正式提炼结果等权进入共识。
        synthesis, status = self.llm_service.synthesize(
            state["query"],
            valid_references,
            deadline=state.get("deadline_monotonic"),
        )
        tips = list(synthesis.get("forecast_tips") or []) if isinstance(synthesis, dict) else []
        summary = dict(synthesis.get("forecast_summary") or {}) if isinstance(synthesis, dict) else {}
        progress.update(state["run_id"], "synthesize_forecast_tips", "跨个例综合提示生成完成", 94)
        warnings = list(state.get("warnings") or [])
        excluded_count = len(all_references) - len(valid_references)
        if excluded_count:
            warnings.append(f"综合研判已排除 {excluded_count} 个规则降级参考，未将其计入跨个例共识。")
        if status != "called":
            warnings.append(f"综合预报提示未完成有效 LLM 生成（{status}），已使用保守摘要。")
        return {
            "forecast_summary": summary,
            "forecast_tips": tips,
            "warnings": warnings,
            "audit": self._audit(
                state,
                "synthesize_forecast_tips",
                started,
                synthesis_status=status,
                synthesis_reference_count=len(valid_references),
                excluded_fallback_reference_count=excluded_count,
                tip_count=len(tips),
                summary_available=bool(summary),
            ),
        }

    def _attach_semantic_evidence(
        self,
        candidates: list[dict[str, Any]],
        query: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """同时提供语义代表片段和诊断指标片段，让候选模型能逐项深入比较。"""
        profile = resolve_dimension_profile(query or {})
        diagnostic_terms = list(dict.fromkeys([
            *(str(value) for value in profile.get("signal_terms") or []),
            *(str(value) for value in profile.get("metrics") or []),
            *(str(value) for value in (query or {}).get("disaster_types") or []),
        ]))
        result = []
        for candidate in candidates:
            evidence = []
            used_chunk_ids = set()
            for chunk_id in candidate.get("semantic_chunk_ids") or []:
                chunk = self.store.get_chunk(str(chunk_id))
                if not chunk:
                    continue
                used_chunk_ids.add(str(chunk_id))
                evidence.append({
                    "chunk_id": str(chunk_id),
                    "source_pdf": str(chunk.get("source_pdf") or ""),
                    "content": self._diagnostic_excerpt(str(chunk.get("content") or ""), diagnostic_terms),
                    "evidence_role": "semantic",
                })
            diagnostic_candidates = []
            for chunk_id in candidate.get("source_chunk_ids") or []:
                if str(chunk_id) in used_chunk_ids:
                    continue
                chunk = self.store.get_chunk(str(chunk_id))
                if not chunk:
                    continue
                content = str(chunk.get("content") or "")
                matched_terms = [term for term in diagnostic_terms if term and term in content]
                if not matched_terms:
                    continue
                diagnostic_candidates.append((len(set(matched_terms)), str(chunk_id), chunk, matched_terms))
            diagnostic_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            # 每个候选最多补两段指标证据，控制输入长度的同时覆盖强度、机制和演变信息。
            for _, chunk_id, chunk, matched_terms in diagnostic_candidates[:2]:
                evidence.append({
                    "chunk_id": chunk_id,
                    "source_pdf": str(chunk.get("source_pdf") or ""),
                    "content": self._diagnostic_excerpt(str(chunk.get("content") or ""), diagnostic_terms),
                    "evidence_role": "diagnostic",
                    "matched_terms": list(dict.fromkeys(matched_terms))[:8],
                })
            result.append({**candidate, "semantic_evidence": evidence})
        return result

    @staticmethod
    def _diagnostic_excerpt(content: str, terms: list[str], limit: int = 720) -> str:
        """优先截取包含当前灾种指标的句子，避免简单截断漏掉雨强、阵风或演变信息。"""
        sentences = [value.strip() for value in re.split(r"(?<=[。；！？])", content) if value.strip()]
        matched = [sentence for sentence in sentences if any(term and term in sentence for term in terms)]
        selected = matched[:3] or sentences[:2]
        return "".join(selected)[:limit]

    def assemble_result(self, state: dict[str, Any]) -> dict[str, Any]:
        """清理内部正文和向量字段，组装唯一对外结果。"""
        started = perf_counter()
        matched_cases = self.build_matched_cases(state)
        warnings = list(dict.fromkeys(state.get("warnings") or []))
        completeness_warnings = self._result_completeness_warnings(state, matched_cases)
        warnings = list(dict.fromkeys(warnings + completeness_warnings))
        status = "completed" if not warnings else "degraded"
        final_output = {
            "run_id": state["run_id"],
            "status": status,
            "query_summary": self._query_summary(state["query"]),
            "matched_cases": matched_cases[:5],
            "forecast_summary": dict(state.get("forecast_summary") or {}),
            "forecast_tips": list(state.get("forecast_tips") or [])[:6],
            "warnings": warnings,
            "audit": self._audit(state, "assemble_result", started, output_case_count=len(matched_cases)),
        }
        progress.update(state["run_id"], "assemble_result", "正在校验并组装最终结果", 98)
        return {"final_output": final_output, "audit": final_output["audit"]}

    @staticmethod
    def _result_completeness_warnings(
        state: dict[str, Any],
        matched_cases: list[dict[str, Any]],
    ) -> list[str]:
        """按业务输出契约判断是否真正完整，避免空研判被误标为正常完成。"""
        warnings = []
        requested = int((state.get("query") or {}).get("top_n") or 3)
        if not matched_cases:
            warnings.append("没有找到达到质量门槛的历史个例。")
            return warnings
        if len(matched_cases) < min(3, requested):
            warnings.append(f"有效匹配个例仅 {len(matched_cases)} 个，低于业务期望数量。")
        referenced = sum(1 for item in matched_cases if item.get("key_references"))
        if referenced < len(matched_cases):
            warnings.append(f"有 {len(matched_cases) - referenced} 个匹配个例缺少可核验参考经验。")
        summary = state.get("forecast_summary") or {}
        if not all(summary.get(key) for key in ("similarity_assessment", "core_features", "main_risk")):
            warnings.append("核心结论摘要字段不完整。")
        valid_tips = [
            item for item in state.get("forecast_tips") or []
            if item.get("support_case_ids") and item.get("evidence_chunk_ids") and item.get("text")
        ]
        if len(valid_tips) < 3:
            warnings.append(f"仅生成 {len(valid_tips)} 条证据完整的综合预报提示，低于业务要求的 3 条。")
        return warnings

    def build_matched_cases(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """从当前图状态构造匹配个例，供同步结果和流式阶段共同复用。"""
        reference_map = {str(item.get("case_id")): item for item in state.get("case_references") or []}
        matched_cases = []
        for case in state.get("enriched_cases") or []:
            case_id = str(case.get("case_id") or "")
            reference = reference_map.get(case_id, {})
            # 最终理由优先使用逐例提炼结果；候选重排跳过时仍能得到专业、针对性的说明。
            reasons = list(
                reference.get("match_reasons")
                or case.get("llm_match_reasons")
                or case.get("structured_reasons")
                or []
            )
            key_references, images, supplemental_images = self._organize_case_evidence(
                state.get("query") or {},
                case,
                list(reference.get("reference_points") or []),
                list(case.get("evidence_images") or []),
            )
            matched_cases.append({
                "case_id": case_id,
                "rank": int(case.get("rank") or len(matched_cases) + 1),
                "title": str(case.get("title") or case_id),
                "date_range": str(case.get("date_range") or ""),
                "disaster_types": list(case.get("disaster_types") or []),
                "affected_areas": list(case.get("affected_areas") or []),
                "source_pdf": str(case.get("source_pdf") or ""),
                "retrieval_score": float(case.get("retrieval_score") or 0.0),
                "score_breakdown": {
                    **dict(case.get("score_breakdown") or {}),
                    "semantic": float(case.get("semantic_score") or 0.0),
                    "structured": float(case.get("structured_score") or 0.0),
                },
                "mechanism_score": case.get("mechanism_score"),
                "intensity_score": case.get("intensity_score"),
                "dimension_scores": dict(case.get("dimension_scores") or {}),
                "metric_scores": dict(case.get("metric_scores") or {}),
                "missing_metrics": list(case.get("missing_metrics") or []),
                "missing_dimensions": list(case.get("missing_dimensions") or []),
                "dimension_compatibility": case.get("dimension_compatibility"),
                # 同时透传原始兼容度和证据覆盖度，便于接口调用方解释候选为何被降权。
                "dimension_raw_compatibility": case.get("dimension_raw_compatibility"),
                "dimension_coverage": case.get("dimension_coverage"),
                "dimension_profile": case.get("dimension_profile"),
                "dimension_assessment_source": case.get("dimension_assessment_source"),
                "candidate_assessment_available": bool(case.get("candidate_assessment_available")),
                "reference_source": str(reference.get("reference_source") or ""),
                "reference_status": str(reference.get("reference_status") or ""),
                "match_reasons": reasons[:4],
                "key_references": key_references,
                "key_differences": list(reference.get("differences") or []),
                "similarities": list(reference.get("similarities") or []),
                "historical_warnings": list(case.get("historical_warnings") or []),
                "source_chunk_ids": list(case.get("source_chunk_ids") or []),
                "evidence_images": images,
                "supplemental_images": supplemental_images,
            })
        return matched_cases

    def _number_images(self, images: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """为图片生成稳定图号，便于正文观点引用。"""
        numbered = []
        for index, image in enumerate(images):
            numbered.append({
                **image,
                "display_no": f"图{index + 1}",
                "caption": self._clean_image_caption(image.get("caption")),
            })
        return numbered

    @staticmethod
    def _clean_image_caption(caption: Any) -> str:
        """移除文档原始图号，避免页面出现“图3：图1”这样的双重编号。"""
        text = re.sub(r"\s+", " ", str(caption or "")).strip()
        if not text.startswith("图"):
            return text
        text = re.sub(r"^图\s*", "", text).strip()
        # “图 2025年……”中的 2025 是年份，不应被当作文档图号删除。
        if re.match(r"^20\d{2}\s*年", text):
            return text
        figure_no = r"(?:\d{1,3}|[一二三四五六七八九十百]+)"
        text = re.sub(
            rf"^{figure_no}(?:\s*[-－—~～至、,，]\s*{figure_no})*\s*[：:]?\s*",
            "",
            text,
        )
        return text.strip()

    def _organize_case_evidence(
        self,
        query: dict[str, Any],
        case: dict[str, Any],
        points: list[dict[str, Any]],
        images: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """按参考经验和当前查询组织主证据图片，其余图片作为不编号的补充资料。"""
        prepared = [
            {**image, "caption": self._clean_image_caption(image.get("caption"))}
            for image in images
        ]
        associations: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        referenced_ids: list[str] = []
        referenced_captions: set[str] = set()
        for point in points:
            matched = self._match_images_for_point(query, case, point, prepared)
            associations.append((point, matched))
            for image in matched:
                image_id = str(image.get("image_id") or "")
                caption_key = self._normalize_evidence_text(image.get("caption")) or image_id
                if image_id and image_id not in referenced_ids and caption_key not in referenced_captions and len(referenced_ids) < MAX_PRIMARY_EVIDENCE_IMAGES:
                    referenced_ids.append(image_id)
                    referenced_captions.add(caption_key)

        by_id = {str(image.get("image_id") or ""): image for image in prepared}
        primary_images = self._number_images([by_id[image_id] for image_id in referenced_ids if image_id in by_id])
        primary_by_id = {str(image.get("image_id") or ""): image for image in primary_images}
        primary_by_caption = {self._normalize_evidence_text(image.get("caption")): image for image in primary_images}
        linked_points = []
        for point, matched in associations:
            item = dict(point)
            refs = []
            for image in matched:
                numbered = primary_by_id.get(str(image.get("image_id") or ""))
                if not numbered:
                    numbered = primary_by_caption.get(self._normalize_evidence_text(image.get("caption")))
                if not numbered:
                    continue
                refs.append({
                    "label": str(numbered.get("display_no") or ""),
                    "image_id": str(numbered.get("image_id") or ""),
                    "caption": str(numbered.get("caption") or ""),
                })
            if refs:
                item["evidence_image_refs"] = refs
            linked_points.append(item)

        referenced_set = set(referenced_ids)
        candidates = [
            (str(image.get("extraction_type") or "").lower() not in {"snapshot", "page_snapshot"}, -index, image)
            for index, image in enumerate(prepared)
            if str(image.get("image_id") or "") not in referenced_set
        ]
        candidates.sort(key=lambda item: item[:2], reverse=True)
        supplemental = []
        supplemental_captions = set(referenced_captions)
        for _, _, image in candidates:
            caption_key = self._normalize_evidence_text(image.get("caption")) or str(image.get("image_id") or "")
            if caption_key in supplemental_captions:
                continue
            supplemental_captions.add(caption_key)
            supplemental.append(image)
            if len(supplemental) >= MAX_SUPPLEMENTAL_IMAGES:
                break
        return linked_points, primary_images, supplemental

    def _match_images_for_point(
        self,
        query: dict[str, Any],
        case: dict[str, Any],
        point: dict[str, Any],
        images: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """图片必须先通过观点主题和查询相关性门槛，再使用邻近关系排序。"""
        # 取出经验要点的文字内容
        text = str(point.get("text") or "")
        # 根据要点文字命中的关键词，收集这条经验涉及哪些主题（如 {"降水", "风"}）
        wanted = self._evidence_topics(text)
        query_text = " ".join([
            str(query.get("query_text") or ""),
            " ".join(str(item) for item in query.get("disaster_types") or []),
            str(query.get("circulation_description") or ""),
            str(query.get("observation_description") or ""),
        ])
        query_topics = self._evidence_topics(query_text)
        # 图片必须落在“当前查询 + 历史个例”的共同业务主题内；辅助主题只从共同核心灾种展开。
        # 这样既允许雪过程使用温度场、环流场，也会拒绝只反映普通降水量的弱相关图片。
        allowed_intersection_topics = self._query_case_evidence_topics(query, case)
        if not allowed_intersection_topics:
            allowed_intersection_topics = self._compatible_topics(query_topics)
        # 取出经验要点关联的所有 chunk 段落（证据来源的正文片段）
        chunks = [
            chunk
            for chunk_id in point.get("evidence_chunk_ids") or []
            if (chunk := self.store.get_chunk(str(chunk_id))) is not None
        ]
        # 收集这些 chunk 正文中直接引用的图片 ID —— 这些是最直接的证据，优先级最高
        direct_ids = {
            str(image_id)
            for chunk in chunks
            for image_id in chunk.get("evidence_image_ids") or []
        }
        # 给每张图片打分，结果存入 scored 列表
        scored = []
        for index, image in enumerate(images):
            image_id = str(image.get("image_id") or "")
            # 取图片完整元数据（优先从 store 取，取不到就用传入的 image 本身）
            raw_image = self.store.get_image(image_id) or image
            # 图注文字
            caption = str(image.get("caption") or "")
            caption_topics = self._evidence_topics(caption)
            # 同一 PDF 可包含多个过程，图注日期明确落在其他过程时必须拒绝，防止跨个例串图。
            if not self._image_date_matches_case(caption, case):
                continue
            # 图片附近的正文文字（用于和要点段落计算重叠度）
            nearby_text = str(raw_image.get("nearby_text") or "")

            # 图注是图片内容的事实边界：没有图注主题或与经验主题不相交时，不能作为主证据。
            topic_matches = wanted & caption_topics
            if not wanted or not topic_matches:
                continue
            # 图注必须同时属于当前测试过程和历史个例的共同证据范围，不能只与其中一方相关。
            intersection_matches = topic_matches & allowed_intersection_topics
            if allowed_intersection_topics and not intersection_matches:
                continue
            # 文本重叠度：图片附近的文字与经验要点所在段落的文字相似度（取最高的那个 chunk）
            overlap = max(
                (self._text_overlap(str(chunk.get("content") or ""), nearby_text) for chunk in chunks),
                default=0.0,
            )
            # 是否为 chunk 中直接引用的图片（最强证据信号）
            direct = image_id in direct_ids

            # 现有存量元数据中的 chunk 与图片关联字段可能为空，不能把直接关联设为硬门槛，
            # 否则会误删图注明确匹配的环流图、温度图。直接关联和正文重合度仅用于候选排序，
            # 图片准入仍由“查询、历史个例、参考经验”三方主题交集及过程日期共同控制。

            # embedded=True 表示是独立提取的图表（非整页快照），清晰度更高，略微优先
            embedded = str(image.get("extraction_type") or "").lower() not in {"snapshot", "page_snapshot"}
            # 主题一致是主要分值；chunk 邻近仅小幅加分，不能再压倒图片真实内容。
            score = (
                45.0 * len(intersection_matches or topic_matches)
                + 25.0 * overlap
                + (10.0 if direct else 0.0)
                + (5.0 if embedded else 0.0)
            )
            # 元组存排序键：(分数, 是否独立图, 是否直接命中, -原序号, 图片对象)
            # -index 用于同分情况下原列表靠前的优先
            scored.append((score, embedded, direct, -index, image))

        # 同一页面同时存在独立图和整页快照时，主证据优先保留可直接查看的独立图。
        # 只要候选里有独立图，就过滤掉"既不是独立图也不是直接命中"的整页快照
        if any(embedded for _, embedded, _, _, _ in scored):
            scored = [item for item in scored if item[1] or item[2]]

        # 按 (分数, 是否独立图, 是否直接命中, -原序号) 从高到低排序
        scored.sort(key=lambda item: item[:4], reverse=True)

        # 从排序结果中只选最相关的一张主证据图，避免弱相关图片稀释证据链。
        selected = []
        seen_captions = set()
        for _, _, _, _, image in scored:
            # 归一化图注作为去重 key（去掉空格标点，不区分大小写）
            caption_key = self._normalize_evidence_text(image.get("caption")) or str(image.get("image_id") or "")
            # 图注相同的图只留一张（可能是不同版本的同一张图）
            if caption_key in seen_captions:
                continue
            seen_captions.add(caption_key)
            selected.append(image)
            # 每条经验最多配 1 张图；没有通过主题门槛的图片时允许为空。
            if len(selected) >= 1:
                break
        return selected

    def _validate_reference_evidence(
        self,
        query: dict[str, Any],
        reference: dict[str, Any],
        case: dict[str, Any],
    ) -> dict[str, Any]:
        """校验查询、个例、参考经验与文字证据的交集，删除弱相关或伪关联引用。"""
        allowed_ids = {str(item.get("chunk_id") or "") for item in case.get("relevant_chunks") or []}
        allowed_topics = self._query_case_evidence_topics(query, case)
        points = []
        for point in reference.get("reference_points") or []:
            text = str(point.get("text") or "").strip()
            if not text:
                continue
            point_topics = self._evidence_topics(text)
            # 参考经验本身必须服务于当前测试过程和匹配个例的共同特征，不能仅仅是历史个例中的任意事实。
            point_intersection_topics = point_topics & allowed_topics if allowed_topics else point_topics
            if allowed_topics and not point_intersection_topics:
                continue
            point_numbers = set(re.findall(r"\d+(?:\.\d+)?", text))
            evidence_ids = []
            for chunk_id in point.get("evidence_chunk_ids") or []:
                chunk_id = str(chunk_id)
                if chunk_id not in allowed_ids:
                    continue
                chunk = self.store.get_chunk(chunk_id)
                content = str((chunk or {}).get("content") or "")
                chunk_topics = self._evidence_topics(content)
                # chunk 正文必须能支撑“共同主题中的这条经验”，而不是仅在长正文其他位置提到同类灾害。
                topic_ok = (
                    bool(point_intersection_topics & chunk_topics)
                    if point_intersection_topics
                    else self._text_overlap(text, content, size=5) >= 0.04
                )
                number_ok = not point_numbers or bool(point_numbers & set(re.findall(r"\d+(?:\.\d+)?", content)))
                if topic_ok and number_ok:
                    evidence_ids.append(chunk_id)
            if evidence_ids:
                points.append({**point, "evidence_chunk_ids": list(dict.fromkeys(evidence_ids))[:3]})
        return {**reference, "reference_points": points[:4]}

    def _query_case_evidence_topics(
        self,
        query: dict[str, Any],
        case: dict[str, Any],
    ) -> set[str]:
        """计算当前查询与历史个例共同允许的核心及成因证据主题。"""
        query_disaster_text = " ".join(str(item) for item in query.get("disaster_types") or [])
        query_core = self._evidence_topics(query_disaster_text)
        if not query_core:
            # 非结构化输入缺少灾种时，才从实况和原始描述中恢复核心主题，避免环流描述反客为主。
            query_core = self._evidence_topics(" ".join([
                str(query.get("observation_description") or ""),
                str(query.get("raw_query") or ""),
            ]))

        case_core = self._evidence_topics(" ".join([
            " ".join(str(item) for item in case.get("disaster_types") or []),
            str(case.get("title") or ""),
        ]))
        common_core = query_core & case_core
        if not common_core:
            return set()

        # 共同核心灾种决定证据边界；动态配置再补充该灾种真正有诊断意义的主题。
        # 这样新增灾种时不需要把雨雪的辅助主题硬编码到所有查询中。
        profile_topics = set(resolve_dimension_profile(query).get("evidence_topics") or [])
        return common_core | (profile_topics & self._compatible_topics(common_core | profile_topics))

    @staticmethod
    def _evidence_topics(value: Any) -> set[str]:
        """从经验、查询或图注中提取统一业务主题。"""
        text = str(value or "")
        return {
            name
            for name, terms in EVIDENCE_TOPIC_TERMS.items()
            if any(term.lower() in text.lower() for term in terms)
        }

    @staticmethod
    def _compatible_topics(topics: set[str]) -> set[str]:
        """展开当前查询可接受的辅助证据主题。"""
        result = set(topics)
        for topic in topics:
            result.update(QUERY_TOPIC_COMPATIBILITY.get(topic, {topic}))
        return result

    @staticmethod
    def _image_date_matches_case(caption: str, case: dict[str, Any]) -> bool:
        """图注明确包含日期时，要求至少一个日期落在当前历史个例的日期范围内。"""
        case_text = " ".join([str(case.get("date_range") or ""), str(case.get("title") or "")])
        case_range = re.search(
            r"(?:(?:20\d{2})\s*年\s*)?(?P<start_month>\d{1,2})\s*月\s*"
            r"(?P<start_day>\d{1,2})\s*(?:日|号)?\s*[-~～至到]\s*"
            r"(?:(?:20\d{2})\s*年\s*)?(?:(?P<end_month>\d{1,2})\s*月\s*)?"
            r"(?P<end_day>\d{1,2})\s*(?:日|号)?",
            case_text,
        )
        if not case_range:
            return True
        start_month = int(case_range.group("start_month"))
        start_day = int(case_range.group("start_day"))
        end_month = int(case_range.group("end_month") or start_month)
        end_day = int(case_range.group("end_day"))
        # 用月日序号比较时，结束月日小于开始月日即表示跨年，例如 12 月 29 日至 1 月 3 日。
        crosses_year = (end_month, end_day) < (start_month, start_day)
        caption_dates = []
        current_month = start_month
        for match in re.finditer(
            r"(?:(?:20\d{2})\s*年\s*)?(?:(?P<month>\d{1,2})\s*月\s*)?"
            r"(?P<day>\d{1,2})\s*(?:日|号)",
            str(caption or ""),
        ):
            if match.group("month"):
                current_month = int(match.group("month"))
            day = int(match.group("day"))
            if 1 <= current_month <= 12 and 1 <= day <= 31:
                caption_dates.append((current_month, day))
        # 图注没有完整的月日信息时不做武断过滤；有日期但全部越界时才拒绝。
        if not caption_dates:
            return True
        for month, day in caption_dates:
            if not crosses_year:
                if (start_month, start_day) <= (month, day) <= (end_month, end_day):
                    return True
                continue
            # 跨年范围拆成“开始年尾段 + 次年初段”比较，覆盖 12 月到 1 月的图注日期。
            if (month, day) >= (start_month, start_day) or (month, day) <= (end_month, end_day):
                return True
        return False

    @staticmethod
    def _normalize_evidence_text(value: Any) -> str:
        """移除空白和标点，供正文重合度及重复图注判断使用。"""
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").lower())

    @classmethod
    def _text_overlap(cls, source: str, target: str, size: int = 8) -> float:
        """通过中文字符片段估算 chunk 正文与图片附近文本是否来自同一页面。"""
        source_text = cls._normalize_evidence_text(source)
        target_text = cls._normalize_evidence_text(target)
        if len(source_text) < size or len(target_text) < size:
            return 0.0
        # 两侧使用相同采样步长，避免只因 source 采样稀疏而产生方向性重合度偏差。
        sample_step = 4
        source_parts = {source_text[index:index + size] for index in range(0, len(source_text) - size + 1, sample_step)}
        target_parts = {target_text[index:index + size] for index in range(0, len(target_text) - size + 1, sample_step)}
        if not source_parts:
            return 0.0
        return len(source_parts & target_parts) / len(source_parts)

    def _audit(self, state: dict[str, Any], node: str, started: float, **details: Any) -> dict[str, Any]:
        """累计节点耗时和关键计数，不保存 API Key 或完整正文。"""
        audit = dict(state.get("audit") or {})
        timings = dict(audit.get("timings_ms") or {})
        timings[node] = round((perf_counter() - started) * 1000, 2)
        audit["timings_ms"] = timings
        # 保留每个节点的完整明细，避免同名计数键在后续节点中相互覆盖。
        node_details = dict(audit.get("node_details") or {})
        node_details[node] = dict(details)
        audit["node_details"] = node_details
        # 继续平铺旧字段，兼容已有审计页面和调用方；完整历史以 node_details 为准。
        audit.update(details)
        return audit

    def _query_summary(self, query: dict[str, Any]) -> dict[str, Any]:
        """返回页面确认用的查询摘要，隐藏内部拼接提示文本。

        内部 query 字典包含一些给 LLM 用的中间字段（如拼接好的提示词、
        归一化中间态等），不应该暴露给前端。此函数作为"白名单过滤器"，
        只挑出前端展示查询条件需要的字段，形成干净的摘要返回。
        """
        return {
            "process_name": query.get("process_name"),              # 用户输入的过程名称
            "start_date": query.get("start_date"),                  # 过程开始日期
            "end_date": query.get("end_date"),                      # 过程结束日期
            "date_text": query.get("date_text"),                    # 自然语言描述的日期（如"去年七月"）
            "disaster_types": query.get("disaster_types"),          # 灾害类型列表
            "affected_areas": query.get("affected_areas"),          # 影响区域列表
            "observation_description": query.get("observation_description"),  # 实况描述
            "circulation_description": query.get("circulation_description"),  # 环流形势描述
            "requested_count": query.get("top_n"),                  # 用户期望返回的个例数量
        }
