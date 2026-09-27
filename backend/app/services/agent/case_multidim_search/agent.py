"""个例多维检索智能体主入口。"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from uuid import uuid4
from urllib.parse import quote

from backend.app.models import DocumentChunk, StandardCase
from backend.app.services.agent.case_multidim_search.analysis.aggregator import CaseAggregator
from backend.app.services.agent.case_multidim_search.analysis.case_metric_pipeline import CaseMetricFacts, CaseMetricPipeline
from backend.app.services.agent.case_multidim_search.analysis.intensity import IntensityExtractor
from backend.app.services.agent.case_multidim_search.retrieval.case_chunk_analyzer import CaseChunkAnalyzer
from backend.app.services.agent.case_multidim_search.retrieval.case_evidence import CaseEvidenceBuilder
from backend.app.services.agent.case_multidim_search.reporting.chart_tool import ChartTool
from backend.app.services.agent.case_multidim_search.analysis.disaster_profile import DISASTER_PROFILES
from backend.app.services.agent.case_multidim_search.reporting.exporter import CaseExporter
from backend.app.services.agent.case_multidim_search.reporting.conversation_report import ConversationReportBuilder
from backend.app.services.agent.case_multidim_search.analysis.llm_intensity_extractor import LlmIntensityExtractor
from backend.app.services.agent.case_multidim_search.integrations.non_thinking_llm import ensure_non_thinking_client
from backend.app.services.agent.case_multidim_search.reporting.pdf_report import PdfReportBuilder
from backend.app.services.agent.case_multidim_search.core import progress as search_progress
from backend.app.services.agent.case_multidim_search.analysis.report_analyzer import ReportAnalyzer
from backend.app.services.agent.case_multidim_search.analysis.report_llm_enhancer import ReportLlmEnhancer
from backend.app.services.agent.case_multidim_search.schemas import (
    CaseAnalysisResponse,
    CaseAnalysisSection,
    CaseAnalysisVisual,
    CaseSearchHit,
    CaseSearchQuery,
    CaseSearchResponse,
    StructuredCaseSearchRequest,
)
from backend.app.services.agent.case_multidim_search.retrieval.structured_retriever import COMPOUND_DISASTER_TERMS, StructuredCaseRetriever


@dataclass
class AgentExecution:
    """一次执行中供查询、导出和报告共同使用的数据。"""

    response: CaseSearchResponse
    cases: list[StandardCase]


@dataclass
class PreparedCase:
    """逐例流水线中已经完成正文和元数据准备、等待模型分析的任务。"""

    index: int
    item: object
    chunks: list[DocumentChunk]
    case_text: str
    evidence: object
    metric_facts: CaseMetricFacts
    selected_chunks: list[DocumentChunk]
    candidate_images: list[dict]
    matched_disasters: list[str]
    focus_disaster: str
    focused_query: CaseSearchQuery
    prepare_seconds: float


class CaseMultidimSearchAgent:
    """组合结构化表单检索、分析、统计和报告能力。"""

    def __init__(
        self,
        standard_case_store,
        document_store,
        image_store,
        output_dir: Path,
        embedding_client=None,
        llm_client=None,
    ):
        """初始化检索、分析、绘图、导出和报告组件。"""
        self.standard_case_store = standard_case_store
        self.document_store = document_store
        self.embedding_client = embedding_client
        self.image_store = image_store
        # 记录本次检索已经解析成功的图片路径，避免展示阶段再次依赖远程元数据刷新。
        self._resolved_image_paths: dict[str, Path] = {}
        self.llm_client = ensure_non_thinking_client(llm_client)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.structured_retriever = StructuredCaseRetriever()
        self.llm_intensity_extractor = LlmIntensityExtractor(
            llm_client=self.llm_client,
            cache_dir=self.output_dir / "intensity_metric_cache",
        )
        self.evidence_builder = CaseEvidenceBuilder(image_store)
        self.aggregator = CaseAggregator()
        self.report_analyzer = ReportAnalyzer()
        self.report_llm_enhancer = ReportLlmEnhancer(
            self.llm_client,
            self.output_dir / "report_llm_cache",
        )
        self.chart_tool = ChartTool()
        self.exporter = CaseExporter()
        self.pdf_builder = PdfReportBuilder(self.chart_tool)
        self.conversation_pdf_builder = ConversationReportBuilder(self.chart_tool)
        self.case_chunk_analyzer = CaseChunkAnalyzer(
            document_store=document_store,
            llm_client=self.llm_client,
            cache_dir=self.output_dir / "case_analysis_cache",
            metric_extractor=self.llm_intensity_extractor,
            standard_case_store=self.standard_case_store,
        )
        # 强度事实库独立扫描全部关联 chunk，不再依赖逐例正文调用顺手返回指标。
        self.case_metric_pipeline = CaseMetricPipeline(
            IntensityExtractor(),
            self.llm_intensity_extractor,
            self.output_dir / "case_metric_fact_cache",
        )
        self._pdf_cache_dir = self.output_dir / "pdf_report_cache"
        self._pdf_cache_dir.mkdir(parents=True, exist_ok=True)
        self._pdf_cache_lock = Lock()

    def export_pdf_from_answer(
        self,
        answer_id: str,
        filename: str = "",
        report_title: str = "",
        progress_callback=None,
    ) -> Path:
        """读取已确认的分析结果并按统一版式导出 PDF。"""
        analysis = self._load_analysis_response(answer_id)
        title = report_title or "山西省气象灾害个例多维分析报告"
        safe_name = self._safe_pdf_name(filename or f"{answer_id}.pdf")
        # 分析阶段已经生成的图表按响应顺序传给 PDF，确保网页预览与下载文件复用同一资产。
        chart_paths = [
            Path(visual.path)
            for visual in analysis.charts
            if visual.path and Path(visual.path).is_file()
        ]
        output_path = self.output_dir / safe_name
        cache_path = self._pdf_cache_path(analysis, title)
        with self._pdf_cache_lock:
            if cache_path.is_file():
                if callable(progress_callback):
                    progress_callback(88, "正在复用已生成的同内容报告")
                shutil.copyfile(cache_path, output_path)
                if callable(progress_callback):
                    progress_callback(96, "正在保存 PDF 文件")
                return output_path
            result = self.pdf_builder.build(
                analysis.search_response,
                output_path,
                report_title=title,
                progress_callback=progress_callback,
                chart_paths=chart_paths,
            )
            # 缓存只复用完全相同的分析内容，写缓存失败不影响当前报告返回。
            try:
                temporary = cache_path.with_suffix(f".{uuid4().hex}.tmp")
                shutil.copyfile(result, temporary)
                temporary.replace(cache_path)
            except OSError:
                pass
            return result

    def export_pdf_from_previous_result(
        self,
        previous_result: dict,
        filename: str = "",
        report_title: str = "",
        progress_callback=None,
    ) -> Path:
        """只排版上一轮页面回答，不重新检索、分析或拼接隐藏证据。"""
        result = previous_result if isinstance(previous_result, dict) else {}
        safe_name = self._safe_pdf_name(filename or "conversation-summary-report.pdf")
        output_path = self.output_dir / safe_name
        return self.conversation_pdf_builder.build(
            str(result.get("answer") or ""),
            output_path,
            report_title=report_title or "气象灾害分析报告",
            progress_callback=progress_callback,
        )

    def _pdf_cache_path(self, analysis: CaseAnalysisResponse, title: str) -> Path:
        """依据报告实际内容生成缓存路径，运行号和输出文件名不参与复用。"""
        search_payload = analysis.search_response.model_dump(mode="json")
        chart_payload = []
        for visual in analysis.charts:
            item = visual.model_dump(mode="json")
            item.pop("path", None)
            chart_payload.append(item)
        raw = json.dumps(
            {"title": title, "search_response": search_payload, "charts": chart_payload},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return self._pdf_cache_dir / f"{digest}.pdf"

    def analyze_structured(self, request: StructuredCaseSearchRequest) -> CaseAnalysisResponse:
        """按结构化条件执行全量个例检索，并生成可审阅的中间分析结果。"""
        execution = self._execute_structured(request)
        answer_id = f"case_answer_{uuid4().hex}"
        title = request.report_title or self._default_structured_title(request, execution.response)
        search_progress.update(
            request.progress_id,
            stage="rendering",
            message="正在渲染统计图表",
            percent=94,
        )
        charts = self._render_answer_charts(answer_id, execution.response)
        images = self._analysis_images(execution.response)
        sections = self._analysis_sections(execution.response)
        evidence = self._analysis_evidence(execution.response)
        response = CaseAnalysisResponse(
            answer_id=answer_id,
            title=title,
            question=execution.response.question,
            answer=execution.response.answer,
            sections=sections,
            images=images,
            charts=charts,
            evidence=evidence,
            search_response=execution.response,
            can_export_pdf=True,
            warnings=execution.response.warnings,
            audit={
                **execution.response.audit,
                "answer_id": answer_id,
                "image_count": len(images),
                "chart_count": len(charts),
                "cache_path": str(self._answer_cache_path(answer_id)),
            },
        )
        search_progress.update(
            request.progress_id,
            stage="saving",
            message="正在保存检索结果",
            percent=98,
        )
        self._save_analysis_response(response)
        return response

    def export_structured_csv(self, request: StructuredCaseSearchRequest) -> Path:
        """只按结构化条件筛选 JSON 并导出 CSV，不重复触发逐例大模型分析。"""
        cases = self._cases_for_export(request)
        return self.exporter.export_csv(cases, self.output_dir / f"case-search-{uuid4().hex}.csv")


    def _cases_for_export(self, request: StructuredCaseSearchRequest) -> list[StandardCase]:
        """为表格导出复用 JSON 筛选；当前仅按时间、灾种和地区条件检索。"""
        query = request.to_query()
        matches = self.structured_retriever.search(self.standard_case_store.list_cases(), query)
        return [item.case for item in matches]

    def health(self) -> dict:
        """检查本地知识库个例、文档片段、图片和中文字体状态。"""
        status = "ok"
        try:
            standard_case_count = len(self.standard_case_store.list_cases())
            standard_case_error = ""
        except Exception as exc:
            status = "degraded"
            standard_case_count = None
            standard_case_error = str(exc)
        try:
            if hasattr(self.document_store, "collection_info"):
                chunk_count = int(self.document_store.collection_info().get("count") or 0)
            elif hasattr(self.document_store, "list_chunk_ids"):
                chunk_count = len(self.document_store.list_chunk_ids())
            else:
                chunk_count = 0
        except Exception as exc:
            status = "degraded"
            chunk_count = None
            chunk_error = str(exc)
        else:
            chunk_error = ""
        try:
            metadata_available = (
                bool(self.image_store.metadata_available())
                if hasattr(self.image_store, "metadata_available")
                else bool(getattr(self.image_store, "metadata_path", Path()).exists())
            )
            image_error = ""
        except Exception as exc:
            status = "degraded"
            metadata_available = False
            image_error = str(exc)
        return {
            "status": status,
            "standard_case_count": standard_case_count,
            "standard_case_error": standard_case_error,
            "document_chunk_count": chunk_count,
            "document_chunk_error": chunk_error,
            "image_metadata_available": metadata_available,
            "image_metadata_error": image_error,
            "chinese_font_available": self.chart_tool._font_path() is not None,
        }
    def _execute_structured(self, request: StructuredCaseSearchRequest) -> AgentExecution:
        """先用标准化个例 JSON 完成筛选统计，再逐个加载关联 chunk 做深度分析。"""
        execution_started = time.monotonic()
        query = request.to_query()
        all_cases = self.standard_case_store.list_cases()
        structured = self.structured_retriever.search(all_cases, query)
        search_progress.update(
            request.progress_id,
            stage="json_statistics",
            message=f"已筛选到 {len(structured)} 个候选个例，正在生成概览统计",
            percent=10,
            completed_cases=0,
            total_cases=len(structured),
        )

        # 第一阶段只使用标准化个例 JSON 生成概览统计，不加载任何文档 chunk。
        json_cases = [item.case for item in structured]
        preliminary_aggregations = self.aggregator.aggregate(json_cases, {})
        preliminary_analysis = self.report_analyzer.analyze(
            json_cases,
            preliminary_aggregations,
            {},
            0,
            query=query,
            displayed_case_count=len(json_cases),
            evidence_by_case={},
        )
        intensity_by_case: dict[str, list] = {}
        # 当前结构化检索仅保留时间、灾种和地区条件；强度只用于后续报告统计，不参与筛选。
        filtered_matches = list(structured)

        display_matches = (
            filtered_matches
            if request.report_include_all_cases
            else self._select_representative_matches(filtered_matches, request.display_limit)
        )
        display_matches = display_matches[
            : request.display_limit if not request.report_include_all_cases else len(display_matches)
        ]
        cases = [item.case for item in filtered_matches]
        search_progress.update(
            request.progress_id,
            stage="case_analysis",
            message=f"准备分析 {len(display_matches)} 个个例",
            percent=15,
            completed_cases=0,
            total_cases=len(display_matches),
        )
        llm_available = self._case_llm_available(request.use_llm_case_analysis)
        hits: list[CaseSearchHit] = []
        image_count = 0
        image_limit = 10_000 if request.include_all_images else request.image_limit_per_case
        analysis_statuses: Counter[str] = Counter()
        selected_chunk_counts: list[int] = []
        evidence_by_case = {}
        case_text_by_case: dict[str, str] = {}
        prepare_seconds: list[float] = []
        case_analysis_seconds: list[float] = []
        metric_fact_audit_by_case: dict[str, dict] = {}

        result_slots: list[tuple[CaseSearchHit, object, str, int, dict] | None] = [None] * len(display_matches)
        # 本地 chunk、边界、证据和图片元数据以 12 路并行准备；规则指标和两类 API
        # 在各自模块内分别限流，避免提高本地并发后无意放大外部请求。
        prepare_workers = min(12, max(1, len(display_matches)))
        llm_workers = min(4, max(1, int(request.case_analysis_concurrency)))
        # 生成池最多四路逐例分析，实际 API 请求还受生成客户端的全局四路闸门控制。
        with ThreadPoolExecutor(max_workers=prepare_workers, thread_name_prefix="case-prepare") as prepare_pool, ThreadPoolExecutor(
            max_workers=llm_workers,
            thread_name_prefix="case-llm",
        ) as llm_pool:
            prepare_futures = {
                prepare_pool.submit(self._prepare_case_task, index, item, query, request, image_limit, llm_available): index
                for index, item in enumerate(display_matches)
            }
            llm_futures = {}
            for future in as_completed(prepare_futures):
                prepared = future.result()
                prepare_seconds.append(prepared.prepare_seconds)
                selected_chunk_counts.append(len(prepared.selected_chunks))
                llm_future = llm_pool.submit(self._analyze_prepared_case, prepared, request, llm_available)
                llm_futures[llm_future] = prepared.index

            completed_count = 0
            for future in as_completed(llm_futures):
                hit, evidence, case_text, candidate_count, metric_audit, analysis_status, elapsed = future.result()
                index = llm_futures[future]
                result_slots[index] = (hit, evidence, case_text, candidate_count, metric_audit)
                analysis_statuses[analysis_status] += 1
                case_analysis_seconds.append(elapsed)
                completed_count += 1
                total_count = len(display_matches)
                percent = 90 if total_count == 0 else 15 + int(75 * completed_count / total_count)
                search_progress.update(
                    request.progress_id,
                    stage="case_analysis",
                    message=f"已完成 {completed_count}/{total_count} 个个例分析",
                    percent=percent,
                    completed_cases=completed_count,
                    total_cases=total_count,
                )

        for result in result_slots:
            if result is None:
                continue
            hit, evidence, case_text, candidate_count, metric_audit = result
            hits.append(hit)
            case_id = str(hit.case.get("case_id") or "")
            intensity_by_case[case_id] = list(hit.intensity_metrics)
            metric_fact_audit_by_case[case_id] = metric_audit
            case_text_by_case[case_id] = case_text
            if evidence is not None:
                evidence_by_case[case_id] = evidence
            image_count += candidate_count

        # 网页和 PDF 共用同一个逐例顺序，并在保存响应前统一改写图片编号和正文引用。
        hits = self._group_hits_for_display(hits)
        self._renumber_case_images(hits)

        # 完成逐例分析后，再统一计算聚合统计和报告级结论。
        aggregations = self.aggregator.aggregate(cases, intensity_by_case)
        analysis = self.report_analyzer.analyze(
            cases,
            aggregations,
            intensity_by_case,
            image_count,
            query=query,
            displayed_case_count=len(hits),
            evidence_by_case=evidence_by_case,
            case_text_by_case=case_text_by_case,
        )
        charts = self.chart_tool.select_specs(aggregations, query=query)
        report_llm_status = "not_requested"
        chart_llm_status = "not_requested"
        report_pipeline_seconds = 0.0
        if request.use_llm_case_analysis and llm_available and hits:
            search_progress.update(
                request.progress_id,
                stage="report_llm",
                message="正在并行生成报告综合分析和图表启示",
                percent=90,
                completed_cases=len(display_matches),
                total_cases=len(display_matches),
            )
            # 两个任务只读取同一份规则统计事实，互不依赖；统一等待后再合并最终报告。
            report_started = time.monotonic()
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="report-llm") as report_pool:
                report_future = report_pool.submit(
                    self.report_llm_enhancer.enhance,
                    analysis,
                    cases,
                    aggregations,
                    hits,
                    query,
                    2200,
                )
                chart_future = (
                    report_pool.submit(
                        self.report_llm_enhancer.enhance_chart_insights,
                        analysis,
                        charts,
                        query,
                        # 图表越多，JSON 中需要容纳的启示越多；在 480～640 token 内按数量给预算。
                        min(640, max(480, 120 * len(charts))),
                    )
                    if charts
                    else None
                )
                analysis, report_llm_status = report_future.result()
                if chart_future is not None:
                    charts, chart_llm_status = chart_future.result()
            report_pipeline_seconds = time.monotonic() - report_started
        elif request.use_llm_case_analysis and not llm_available:
            report_llm_status = "not_available"
            chart_llm_status = "not_available"
        # 保留第一阶段仅基于 JSON 得到的概览，便于核对两阶段结果。
        analysis["json_overview"] = preliminary_analysis.get("executive_summary", "")
        analysis["report_llm_status"] = report_llm_status
        search_progress.update(
            request.progress_id,
            stage="rendering",
            message="正在生成统计图表和最终结果",
            percent=92,
            completed_cases=len(display_matches),
            total_cases=len(display_matches),
        )
        analysis["chart_llm_status"] = chart_llm_status
        warnings = []
        if not filtered_matches:
            warnings.append("没有找到同时满足当前结构化条件的标准化个例。")
        if request.use_llm_case_analysis and not llm_available and filtered_matches:
            warnings.append("逐个个例大模型不可用，已自动使用规则分析结果。")
        failed_count = analysis_statuses["llm_failed"] + analysis_statuses["llm_empty"] + analysis_statuses["llm_incomplete"]
        if failed_count:
            warnings.append(f"有 {failed_count} 个个例的大模型分析失败，已分别使用规则结果降级。")
        if report_llm_status in {"llm_failed", "invalid_json"}:
            warnings.append("报告级大模型综合失败，已保留规则统计分析结果。")
        response = CaseSearchResponse(
            question=self._structured_question(request),
            parsed_query=query,
            retrieval_mode="structured_json_then_case_chunks",
            result_count=len(filtered_matches),
            displayed_result_count=len(hits),
            answer=analysis["executive_summary"],
            analysis=analysis,
            results=hits,
            aggregations=aggregations,
            charts=charts,
            warnings=warnings,
            audit={
                "structured_candidates": len(structured),
                "json_statistics_case_count": preliminary_aggregations.get("case_count", 0),
                "fused_results": len(filtered_matches),
                "analysis_case_count": len(cases),
                "displayed_results": len(hits),
                "resolved_images": image_count,
                "all_case_ids": [case.case_id for case in cases],
                "displayed_case_ids": [hit.case.get("case_id", "") for hit in hits],
                "report_include_all_cases": request.report_include_all_cases,
                "chunk_loading_mode": "shared_parallel_prefetch_pipeline",
                "case_chunk_limit": request.case_chunk_limit,
                "case_context_char_limit": request.case_context_char_limit,
                "case_analysis_concurrency": llm_workers,
                "llm_thinking_enabled": False,
                # 正文保持一次调用；指标按每批六个 chunk 的独立事实库调用，二者不再耦合。
                "case_analysis_llm_calls_per_case": 1,
                "metric_llm_batch_size": 6,
                "metric_fact_batches_total": sum(
                    int(item.get("batch_count") or 0) for item in metric_fact_audit_by_case.values()
                ),
                "metric_llm_calls_per_case_upper_bound": max(
                    (int(item.get("batch_count") or 0) for item in metric_fact_audit_by_case.values()),
                    default=0,
                ) if llm_available else 0,
                "case_max_output_tokens": max(1200, min(1536, int(request.case_max_output_tokens))),
                "case_max_output_tokens_requested": request.case_max_output_tokens,
                "selected_chunk_count_total": sum(selected_chunk_counts),
                "selected_chunk_count_max": max(selected_chunk_counts, default=0),
                "case_llm_requested": request.use_llm_case_analysis,
                "case_llm_available": llm_available,
                "case_llm_calls": analysis_statuses["llm_generated"],
                "case_llm_attempts": (
                    analysis_statuses["llm_generated"]
                    + analysis_statuses["llm_failed"]
                    + analysis_statuses["llm_empty"]
                    + analysis_statuses["llm_incomplete"]
                ),
                "case_llm_cache_hits": analysis_statuses["cache_hit"],
                "case_analysis_statuses": dict(analysis_statuses),
                "metric_fact_audit_by_case": metric_fact_audit_by_case,
                "case_prepare_seconds_total": round(sum(prepare_seconds), 3),
                "case_analysis_seconds_total": round(sum(case_analysis_seconds), 3),
                "report_pipeline_seconds": round(report_pipeline_seconds, 3),
                "pipeline_wall_seconds": round(time.monotonic() - execution_started, 3),
                "report_llm_status": report_llm_status,
                "report_llm_used": report_llm_status in {"llm_generated", "cache_hit"},
            },
        )
        return AgentExecution(response=response, cases=cases)


    def _prepare_case_task(
        self,
        index: int,
        item,
        query: CaseSearchQuery,
        request: StructuredCaseSearchRequest,
        image_limit: int,
        llm_available: bool,
    ) -> PreparedCase:
        """并发准备单个个例的全部正文、数字证据、精选片段和图片元数据。"""
        started = time.monotonic()
        case = item.case
        chunks = self.case_chunk_analyzer.load_case_chunks(case)
        case_text = "\n".join(
            str(chunk.content or "") for chunk in chunks if str(chunk.content or "").strip()
        )
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        evidence = self.evidence_builder.build([case], chunk_map).get(case.case_id)
        metric_facts = self.case_metric_pipeline.extract(case, chunks, llm_enabled=llm_available)
        matched_disasters = self._case_matched_disasters(case, query)
        focus_disaster = self._case_focus_disaster(case, query)
        focused_query = self._focused_case_query(query, focus_disaster)
        selected_chunks = self.case_chunk_analyzer.select_relevant_chunks(
            case,
            chunks,
            focused_query,
            request.case_chunk_limit,
            request.case_context_char_limit,
        )
        images = self._images_for_case(case, image_limit, resolve_paths=False) if request.include_images else []
        return PreparedCase(
            index=index,
            item=item,
            chunks=chunks,
            case_text=case_text,
            evidence=evidence,
            metric_facts=metric_facts,
            selected_chunks=selected_chunks,
            candidate_images=self._dedupe_candidate_images(images),
            matched_disasters=matched_disasters,
            focus_disaster=focus_disaster,
            focused_query=focused_query,
            prepare_seconds=time.monotonic() - started,
        )

    def _analyze_prepared_case(
        self,
        prepared: PreparedCase,
        request: StructuredCaseSearchRequest,
        llm_available: bool,
    ) -> tuple[CaseSearchHit, object, str, int, dict, str, float]:
        """对已准备个例生成正文，并复用已完整扫描的指标事实库。"""
        started = time.monotonic()
        case = prepared.item.case
        fallback_analysis = self._case_result_analysis(
            case,
            [],
            prepared.evidence,
            len(prepared.candidate_images),
            prepared.focus_disaster,
        )
        case_analysis, status, cited_image_ids, _legacy_metrics = self.case_chunk_analyzer.analyze_case_with_metrics(
            case,
            prepared.selected_chunks,
            prepared.focused_query,
            fallback_analysis,
            llm_available,
            request.case_context_char_limit,
            # 联合输出包含正文、指标 JSON 和图片引用，先在 1200～1536 token 内控制预算。
            max(1200, min(1536, int(request.case_max_output_tokens))),
            displayed_images=prepared.candidate_images,
            # 正文仅使用精选 chunk；所有强度数据已在独立事实库中完成全 chunk 分批扫描。
            extract_metrics=False,
            # 已核验指标和来源句进入正文上下文，模型不再承担强度发现或极值判断。
            verified_metrics=prepared.metric_facts.metrics,
        )
        case_analysis, aligned_metadata = self._align_case_analysis_images(
            case_analysis,
            prepared.candidate_images,
            max_count=len(prepared.candidate_images),
            cited_image_ids=cited_image_ids,
        )
        aligned_images = self._resolve_selected_images(aligned_metadata)
        hit = CaseSearchHit(
            case=self._case_dict_with_focus(case, prepared.focus_disaster, prepared.matched_disasters),
            score=prepared.item.score,
            matched_fields=prepared.item.matched_fields,
            analysis=case_analysis,
            intensity_metrics=prepared.metric_facts.metrics,
            evidence_images=aligned_images,
        )
        return (
            hit,
            prepared.evidence,
            prepared.case_text,
            len(aligned_images),
            prepared.metric_facts.audit(),
            status,
            time.monotonic() - started,
        )

    def _resolve_selected_images(self, images: list[dict]) -> list[dict]:
        """模型完成引用选择后才下载图片，失败图片不进入网页和 PDF。"""
        resolved: list[dict] = []
        for image in images:
            image_id = str(image.get("image_id") or "").strip()
            record = self.image_store.get_image(image_id) if image_id else None
            if record is None:
                continue
            path = self._resolve_image_path(record)
            if path is None:
                continue
            self._resolved_image_paths[image_id] = Path(path)
            data = dict(image)
            data["image_path"] = str(path)
            data["path_resolved"] = True
            resolved.append(data)
        return resolved

    def _image_metric_chunks(self, case: StandardCase) -> list[DocumentChunk]:
        """把当前个例图片的图注和附近文字转换为临时证据 chunk。"""
        try:
            records = list(self.image_store.list_by_image_ids(case.evidence_image_ids or []))
        except Exception:
            return []
        chunks: list[DocumentChunk] = []
        seen: set[str] = set()
        for record in records:
            image_id = str(getattr(record, "image_id", "") or "").strip()
            if not image_id or image_id in seen:
                continue
            nearby_text = str(getattr(record, "nearby_text", "") or "").strip()
            caption = str(getattr(record, "caption", "") or "").strip()
            content = "\n".join(item for item in (caption, nearby_text) if item)
            if not content:
                continue
            seen.add(image_id)
            chunks.append(
                DocumentChunk(
                    source_pdf=str(getattr(record, "source_pdf", "") or ""),
                    chunk_id=f"image-nearby::{image_id}",
                    chunk_no=int(getattr(record, "image_no", 0) or 0),
                    content=content,
                )
            )
        return chunks

    def _llm_available(self) -> bool:
        """检查大模型服务是否可用；强度补抽可独立于逐例正文开关执行。"""
        if self.llm_client is None:
            return False
        try:
            return bool(self.llm_client.is_available())
        except Exception:
            return False

    def _case_llm_available(self, requested: bool) -> bool:
        """只在用户启用逐例分析时检查一次大模型，避免每个个例重复探测服务。"""
        if not requested or self.llm_client is None:
            return False
        try:
            return bool(self.llm_client.is_available())
        except Exception:
            return False

    def _case_matched_disasters(self, case: StandardCase, query: CaseSearchQuery) -> list[str]:
        """按检索条件顺序返回该个例实际命中的全部灾种，不用标题首词替代检索命中。"""
        wanted = [str(item or "").strip() for item in (query.disaster_types or []) if str(item or "").strip()]
        if not wanted:
            return []
        values = [str(item or "").strip() for item in (case.disaster_types or []) if str(item or "").strip()]
        text = " ".join([
            str(case.title or ""),
            str(case.date_range or ""),
            str(case.summary or ""),
            str(case.weather_facts or ""),
            str(case.forecast_focus or ""),
            " ".join(values),
        ])
        return [term for term in wanted if self._term_matches_case_text(term, values, text)]

    def _title_matched_disaster(self, title: str, matched: list[str]) -> str:
        """只在已命中的检索灾种中按标题出现位置选择主导灾种。"""
        candidates: list[tuple[int, int, str]] = []
        for term in matched:
            position = str(title or "").find(term)
            if position >= 0:
                candidates.append((position, -len(term), term))
                continue
            atomics = COMPOUND_DISASTER_TERMS.get(term, ())
            if atomics and all(atomic in str(title or "") for atomic in atomics):
                candidates.append((min(str(title or "").find(atomic) for atomic in atomics), -len(term), term))
        candidates.sort()
        return candidates[0][2] if candidates else ""

    def _case_focus_disaster(self, case: StandardCase, query: CaseSearchQuery) -> str:
        """主导灾种保持单值：先看标题中的检索命中灾种，再按检索顺序兜底。"""
        matched = self._case_matched_disasters(case, query)
        if matched:
            return self._title_matched_disaster(str(case.title or ""), matched) or matched[0]
        values = [str(item or "").strip() for item in (case.disaster_types or []) if str(item or "").strip()]
        if not query.disaster_types:
            return self._automatic_case_focus_disaster(str(case.title or ""), values)
        return str(query.disaster_types[0] or "").strip()
    def _automatic_case_focus_disaster(self, title: str, values: list[str]) -> str:
        """未指定检索灾种时，按标题优先、并发灾种兜底自动确定归属灾种。"""
        standard_terms = list(
            dict.fromkeys(
                name
                for profile in DISASTER_PROFILES
                for name in profile.names
            )
        )
        title_match = self._title_focus_disaster(title, standard_terms)
        if title_match:
            return title_match
        # 标题没有明确灾种时，优先沿用原始灾种列表的顺序，保证个例分组可解释。
        for value in values:
            if any(
                value == term or value in term or term in value
                for term in standard_terms
            ):
                return value
        return values[0] if values else ""

    def _title_process_focus_disaster(self, title: str, values: list[str]) -> str:
        """标题明确涉及降水过程时，按降水类灾种的具体程度确定归属。"""
        title_text = str(title or "")
        if not any(token in title_text for token in ("\u964d\u6c34", "\u964d\u96e8", "\u96e8")):
            return ""
        priority = ("\u77ed\u65f6\u5f3a\u964d\u6c34", "\u5f3a\u964d\u6c34", "\u5927\u66b4\u96e8", "\u66b4\u96e8", "\u96e8\u96ea", "\u964d\u96ea", "\u66b4\u96ea")
        for term in priority:
            if any(term == value or term in value or value in term for value in values):
                return term
        return ""

    def _title_focus_disaster(self, title: str, wanted: list[str]) -> str:
        """优先根据标题确定归属灾种；多个灾种命中时按标题出现顺序和词长择优。"""
        title = str(title or "")
        candidates: list[tuple[int, int, str]] = []
        for term in wanted:
            if not term:
                continue
            position = title.find(term)
            if position >= 0:
                # 同一位置命中时优先选择更具体的长词，例如“雷暴大风”优先于“大风”。
                candidates.append((position, -len(term), term))
                continue
            if term == "大暴雨" and "暴雨" in title:
                # 检索词是“大暴雨”但标题只写“暴雨”时，仍按标题灾种归属为“暴雨”。
                candidates.append((title.find("暴雨"), -len("暴雨"), "暴雨"))
            elif term == "雷暴大风" and "雷暴" in title and "大风" in title:
                candidates.append((min(title.find("雷暴"), title.find("大风")), -len(term), term))
        if not candidates:
            return ""
        candidates.sort()
        return candidates[0][2]
    def _term_matches_case_text(self, term: str, values: list[str], text: str) -> bool:
        """判断灾种词是否命中个例文本，允许“雷暴大风”与雷暴、大风拆分标签互相兼容。"""
        if not term:
            return False
        if term in text or any(term == value or term in value for value in values):
            return True
        if term == "雷暴大风":
            return all(atomic in text or any(atomic == value or atomic in value for value in values) for atomic in ("雷暴", "大风"))
        return False

    def _focused_case_query(self, query: CaseSearchQuery, focus_disaster: str) -> CaseSearchQuery:
        """为单个个例构造单灾种视角查询，只影响 chunk 精排和逐例大模型提示词。"""
        if not focus_disaster:
            return query
        return query.model_copy(update={"disaster_types": [focus_disaster]})

    def _case_dict_with_focus(
        self,
        case: StandardCase,
        focus_disaster: str,
        matched_disasters: list[str] | None = None,
    ) -> dict:
        """在返回字典中同时保留分析视角和全部检索命中灾种。"""
        data = case.to_dict()
        matched = [str(item or "").strip() for item in (matched_disasters or []) if str(item or "").strip()]
        if matched:
            data["matched_disasters"] = matched
            data["matched_disaster"] = focus_disaster or matched[0]
        elif focus_disaster:
            data["matched_disasters"] = [focus_disaster]
            data["matched_disaster"] = focus_disaster
        if focus_disaster:
            data["analysis_focus_disaster"] = focus_disaster
        return data
    def _case_result_analysis(self, case: StandardCase, metrics: list, evidence, image_count: int, focus_disaster: str = "") -> str:
        """生成单个命中个例的业务化分析，不暴露 chunk 和 image 等底层编号。"""
        date_text = case.date_range or "时间未明确"
        disasters = "、".join(case.disaster_types) or "灾种未标注"
        areas = "、".join(case.city_tags or case.affected_areas) or "影响区域未明确"
        focus_text = f"本次检索将该个例归入{focus_disaster}视角，分析优先围绕{focus_disaster}展开。" if focus_disaster else ""
        parts = [f"{date_text}，{areas}出现以{disasters}为主的天气过程。{focus_text}"]
        evidence_text = evidence.public_text(limit=2) if evidence else ""
        if evidence_text:
            parts.append(evidence_text)
        if metrics:
            metric_text = "；".join(
                f"{metric.metric_name}{metric.relation or '为'}{metric.value:g}{metric.unit}"
                for metric in metrics[:4]
            )
            parts.append(f"强度证据显示，{metric_text}。")
        else:
            parts.append("当前原文中尚未提取到统一可比的关键强度指标数值。")
        if image_count:
            parts.append(f"配套的 {image_count} 张实况或诊断图件可用于核对过程形态、影响落区及演变依据。")
        review_disaster = focus_disaster or (case.disaster_types[0] if case.disaster_types else "本次过程")
        parts.append(
            f"业务复盘应围绕{review_disaster}的实况跃增、影响系统演变和重点落区建立对应关系，"
            "并结合吕梁山、五台山、太行山迎风坡、晋中盆地或城市敏感区复核短临触发信号、预警提前量及服务衔接。"
        )
        return "\n\n".join(parts)
    def _structured_question(self, request: StructuredCaseSearchRequest) -> str:
        """把结构化条件压缩为用于展示和报告标题生成的可读查询描述。"""
        parts = []
        if request.start_date or request.end_date:
            parts.append(f"时间：{request.start_date or '不限'}至{request.end_date or '不限'}")
        if request.years:
            parts.append("年份：" + "、".join(str(value) for value in request.years))
        if request.months:
            parts.append("月份：" + "、".join(f"{value}月" for value in request.months))
        if request.disaster_types:
            parts.append("灾种：" + "、".join(request.disaster_types))
        if request.cities:
            parts.append("地市：" + "、".join(request.cities))
        if request.areas:
            parts.append("区域：" + "、".join(request.areas))
        return "；".join(parts) or "结构化全库检索"

    def _default_structured_title(self, request: StructuredCaseSearchRequest, response: CaseSearchResponse | None = None) -> str:
        """默认使用综合性报告标题，检索条件交给副标题承载。"""
        return "山西省气象灾害个例多维分析报告"

    def image_asset_path(self, image_id: str) -> Path:
        """按图片编号读取图片，优先复用本次运行缓存并支持缓存目录兜底。"""
        normalized_id = str(image_id or "").strip()
        cached = self._resolved_image_paths.get(normalized_id)
        if cached is not None and cached.is_file():
            return cached

        record = self.image_store.get_image(normalized_id)
        if record is not None:
            path = self._resolve_image_path(record)
            if path is not None and path.is_file():
                self._resolved_image_paths[normalized_id] = path
                return path

        # 元数据服务短暂不可用时，仍尝试读取已经下载过的远程图片缓存。
        cache_dir = getattr(self.image_store, "cache_dir", None)
        client = getattr(self.image_store, "client", None)
        safe_name = normalized_id
        safe_name_fn = getattr(client, "_safe_filename", None)
        if callable(safe_name_fn):
            safe_name = safe_name_fn(normalized_id)
        if cache_dir and safe_name:
            image_dir = Path(cache_dir) / "images"
            for suffix in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
                candidate = image_dir / f"{safe_name}{suffix}"
                if candidate.is_file():
                    self._resolved_image_paths[normalized_id] = candidate
                    return candidate
        raise FileNotFoundError(f"未找到可展示的图片文件：{normalized_id}")


    def _analysis_sections(self, response: CaseSearchResponse) -> list[CaseAnalysisSection]:
        """把内部分析字典转换为前端可直接展示的章节数组。"""
        sections = response.analysis.get("sections", {}) if response.analysis else {}
        headings = [
            ("temporal", "一、时间分布分析"),
            ("disaster", "二、灾种结构分析"),
            ("spatial", "三、影响区域分析"),
            ("intensity", "四、关键强度指标分析"),
        ]
        return [
            CaseAnalysisSection(heading=heading, content=str(sections[key]))
            for key, heading in headings
            if sections.get(key)
        ]

    def _analysis_images(self, response: CaseSearchResponse) -> list[CaseAnalysisVisual]:
        """从代表个例中抽取数据库原图，返回可预览的资源信息。"""
        visuals: list[CaseAnalysisVisual] = []
        seen: set[str] = set()
        for hit in response.results:
            case = hit.case
            for image in hit.evidence_images:
                image_id = str(image.get("image_id") or "")
                if not image_id or image_id in seen:
                    continue
                seen.add(image_id)
                visuals.append(
                    CaseAnalysisVisual(
                        visual_id=image_id,
                        visual_type="database_image",
                        title=str(image.get("display_caption") or image.get("caption") or image_id),
                        url=f"/api/case-multidim/assets/images/{quote(image_id, safe='')}",
                        path=str(image.get("image_path") or ""),
                        source=str(image.get("source_pdf") or case.get("source_pdf") or ""),
                    )
                )
        return visuals

    def _render_answer_charts(self, answer_id: str, response: CaseSearchResponse) -> list[CaseAnalysisVisual]:
        """把统计图表提前渲染成图片，方便用户在生成 PDF 前预览。"""
        chart_dir = self.output_dir / "charts"
        visuals: list[CaseAnalysisVisual] = []
        for index, spec in enumerate(response.charts, start=1):
            chart_id = f"{answer_id}-chart-{index}"
            path = self.chart_tool.render(spec, chart_dir / f"{chart_id}.png")
            visuals.append(
                CaseAnalysisVisual(
                    visual_id=chart_id,
                    visual_type="generated_chart",
                    title=spec.title,
                    chart_key=getattr(spec, "chart_key", ""),
                    url=f"/api/case-multidim/assets/{path.name}",
                    path=str(path),
                    source=spec.interpretation or "智能体统计绘图工具",
                )
            )
        return visuals

    def _analysis_evidence(self, response: CaseSearchResponse) -> list[dict]:
        """生成命中证据摘要，解释每个代表个例为什么被纳入分析。"""
        evidence = []
        for hit in response.results:
            case = hit.case
            evidence.append(
                {
                    "case_id": case.get("case_id", ""),
                    "title": case.get("title", ""),
                    "date_range": case.get("date_range", ""),
                    "source_pdf": case.get("source_pdf", ""),
                    "matched_fields": hit.matched_fields,
                    "reason": "、".join(hit.matched_fields) or "结构化条件命中",
                }
            )
        return evidence

    def _save_analysis_response(self, response: CaseAnalysisResponse) -> None:
        """把分析结果保存为 JSON，供后续 PDF 导出接口按 answer_id 读取。"""
        path = self._answer_cache_path(response.answer_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.model_dump_json(indent=2), encoding="utf-8")

    def _load_analysis_response(self, answer_id: str) -> CaseAnalysisResponse:
        """按回答编号读取已缓存的分析结果。"""
        path = self._answer_cache_path(answer_id)
        if not path.is_file():
            raise FileNotFoundError(f"未找到分析结果：{answer_id}")
        return CaseAnalysisResponse.model_validate_json(path.read_text(encoding="utf-8"))

    def _answer_cache_path(self, answer_id: str) -> Path:
        """把回答编号映射为受控缓存路径，避免路径穿越。"""
        safe_id = "".join(char for char in str(answer_id) if char.isalnum() or char in {"_", "-"})
        if not safe_id:
            raise ValueError("answer_id 不能为空")
        return self.output_dir / "answers" / f"{safe_id}.json"

    def _safe_pdf_name(self, filename: str) -> str:
        """规范化 PDF 下载文件名，只保留文件名并强制使用 pdf 后缀。"""
        name = Path(str(filename or "case-search-report.pdf")).name
        if not name.lower().endswith(".pdf"):
            name = f"{Path(name).stem or 'case-search-report'}.pdf"
        return name

    def _select_representative_matches(self, matches: list, limit: int) -> list:
        """先覆盖不同月份，再按相关度和证据完整度补足代表个例。"""
        if len(matches) <= limit:
            return list(matches)
        ranked = sorted(matches, key=self._representative_rank, reverse=True)
        month_groups: dict[int, list] = {}
        for item in ranked:
            month_groups.setdefault(self._representative_month(item.case), []).append(item)
        selected = []
        for group in month_groups.values():
            if len(selected) >= limit:
                break
            selected.append(group[0])
        selected_ids = {item.case.case_id for item in selected}
        for item in ranked:
            if len(selected) >= limit:
                break
            if item.case.case_id not in selected_ids:
                selected.append(item)
                selected_ids.add(item.case.case_id)
        return sorted(selected, key=self._representative_rank, reverse=True)

    def _representative_rank(self, item) -> tuple:
        """计算代表个例的相关度和证据完整度排序键。"""
        case = item.case
        return (
            item.score,
            bool(case.evidence_image_ids),
            bool(case.source_chunk_ids),
            case.case_id,
        )

    def _representative_month(self, case: StandardCase) -> int:
        """读取代表性分组使用的首个月份，缺失时归入零值组。"""
        months = self.aggregator.case_months(case)
        return months[0] if months else 0

    def _dedupe_candidate_images(self, images: list[dict]) -> list[dict]:
        """按图注去重候选图，避免同一图被 PDF 内部多个图片对象重复交给模型。"""
        selected = []
        seen = set()
        for image in images:
            caption = self._normalize_image_ref_text(str(image.get("caption") or ""))
            key = caption or str(image.get("image_id") or image.get("image_path") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            selected.append(image)
        return selected

    def _align_case_analysis_images(
        self,
        analysis: str,
        images: list[dict],
        max_count: int = 8,
        cited_image_ids: list[str] | None = None,
    ) -> tuple[str, list[dict]]:
        """结合正文图号和模型返回的图片ID筛图，并按正文引用位置稳定排序。"""
        if not analysis or max_count <= 0:
            return analysis, []
        # 模型偶尔用“图N...”代替完整图文关系；先清理开放式引用，再进行候选图号匹配。
        complete_reference_analysis = self._remove_open_ended_image_references(analysis)
        cleaned_analysis = self._remove_unmatched_image_reference_sentences(complete_reference_analysis, images)
        if not images:
            return cleaned_analysis, []

        by_id = {str(image.get("image_id") or ""): image for image in images if str(image.get("image_id") or "")}
        selected: list[dict] = []
        seen: set[str] = set()
        # 图片ID用于消除同一图号包含多个子图时的歧义，正文图号仍用于确定最终展示顺序。
        for image_id in cited_image_ids or []:
            image = by_id.get(str(image_id))
            if image is None or str(image_id) in seen:
                continue
            if not self._analysis_references_image_or_base(cleaned_analysis, image):
                continue
            seen.add(str(image_id))
            selected.append(image)
        for image in self._images_referenced_by_analysis(cleaned_analysis, images, max_count=max_count):
            image_key = str(image.get("image_id") or image.get("image_path") or "")
            if image_key in seen:
                continue
            seen.add(image_key)
            selected.append(image)

        citation_order = {
            str(image.get("image_id") or image.get("image_path") or index): index
            for index, image in enumerate(selected)
        }
        normalized_analysis = self._normalize_image_ref_text(cleaned_analysis)
        selected.sort(
            key=lambda image: (
                self._image_reference_position(normalized_analysis, image)
                if self._image_reference_position(normalized_analysis, image) >= 0
                else 10**9,
                citation_order.get(str(image.get("image_id") or image.get("image_path") or ""), 10**9),
            )
        )
        return cleaned_analysis, selected[:max_count]

    def _group_hits_for_display(self, hits: list[CaseSearchHit]) -> list[CaseSearchHit]:
        """按归属灾种稳定分组，网页和 PDF 均直接使用同一结果顺序。"""
        groups: dict[str, list[CaseSearchHit]] = {}
        for hit in hits:
            case_data = hit.case or {}
            disaster = str(
                case_data.get("matched_disaster")
                or case_data.get("analysis_focus_disaster")
                or "未归属灾种"
            )
            groups.setdefault(disaster, []).append(hit)
        return [hit for group in groups.values() for hit in group]

    def _renumber_case_images(self, hits: list[CaseSearchHit]) -> None:
        """按最终逐例顺序统一图片编号，并同步改写分析正文中的旧图号引用。"""
        next_number = 1
        for hit in hits:
            numbered_images: list[dict] = []
            for image in hit.evidence_images:
                data = dict(image)
                original_caption = str(data.get("original_caption") or data.get("caption") or "原始证据图").strip()
                data["original_caption"] = original_caption
                data["display_number"] = next_number
                caption_body = self._caption_without_figure_number(original_caption)
                data["display_caption"] = f"图 {next_number} {caption_body}".strip()
                numbered_images.append(data)
                next_number += 1
            hit.analysis = self._rewrite_analysis_figure_references(hit.analysis, numbered_images)
            hit.evidence_images = numbered_images

    def _caption_without_figure_number(self, caption: str) -> str:
        """移除原始图号但保留图题正文，供统一编号后的网页和 PDF 复用。"""
        text = re.sub(r"\s+", " ", str(caption or "")).strip()
        cleaned = re.sub(
            r"^图\s*(?:[0-9０-９]+|[一二三四五六七八九十]+)(?:\s*[（(][^）)]*[）)]|[a-zA-Z])?\s*",
            "",
            text,
        ).strip()
        return cleaned or "原始证据图"

    def _rewrite_analysis_figure_references(self, analysis: str, images: list[dict]) -> str:
        """用占位符两阶段替换旧图号，避免新图号再次被后续规则误替换。"""
        text = str(analysis or "")
        replacements: list[tuple[str, str]] = []
        base_counts: Counter[str] = Counter()
        image_tokens: list[tuple[dict, list[str]]] = []
        for image in images:
            tokens = self._figure_tokens_from_caption(str(image.get("original_caption") or ""))
            specific = [token for token in tokens if token]
            image_tokens.append((image, specific))
            for token in specific:
                base_counts[re.sub(r"\([^)]*\)$", "", token)] += 1
        for image, tokens in image_tokens:
            replacement = f"图 {int(image.get('display_number') or 0)}"
            for token in tokens:
                replacements.append((token, replacement))
                base = re.sub(r"\([^)]*\)$", "", token)
                if base_counts[base] == 1:
                    replacements.append((base, replacement))

        placeholders: dict[str, str] = {}
        for index, (token, replacement) in enumerate(sorted(dict.fromkeys(replacements), key=lambda item: -len(item[0]))):
            match = re.fullmatch(r"图([0-9]+)(?:\(([^)]+)\)|([a-zA-Z]))?", token)
            if not match:
                continue
            number, suffix = match.group(1), match.group(2) or match.group(3)
            if suffix:
                pattern = rf"图\s*{re.escape(number)}\s*(?:[（(]\s*{re.escape(suffix)}\s*[）)]|{re.escape(suffix)})(?![0-9a-zA-Z])"
            else:
                pattern = rf"图\s*{re.escape(number)}(?![0-9])"
            placeholder = f"__CASE_FIGURE_{index}__"
            updated, count = re.subn(pattern, placeholder, text, flags=re.IGNORECASE)
            if count:
                text = updated
                placeholders[placeholder] = replacement
        for placeholder, replacement in placeholders.items():
            text = text.replace(placeholder, replacement)
        return text

    def _remove_open_ended_image_references(self, analysis: str) -> str:
        """清理带省略号的图片占位引用，避免报告出现无法核对的“图N...”。"""
        text = str(analysis or "")
        figure = r"图\s*[0-9]+(?:\s*[（(][^）)]*[）)]|[a-zA-Z])?"
        ellipsis = r"(?:\.{3,}|…+)"

        # “可参考图N...”没有说明图片支撑什么判断，整段引用前缀都不能保留。
        reference_prefix = rf"(?:可|可以|可供)?\s*参考\s*{figure}(?:\s*[、,，]\s*{figure})*\s*{ellipsis}\s*[。；;，,：:]?"
        text = re.sub(reference_prefix, "", text, flags=re.IGNORECASE)

        # “图1、图2、...”只删除未知的开放尾项，保留前面能够逐一匹配的明确图号。
        explicit_list = rf"({figure}(?:\s*[、,，]\s*{figure})+)\s*[、,，]\s*{ellipsis}"
        text = re.sub(explicit_list, r"\1", text, flags=re.IGNORECASE)

        # 其余“图N...”同样不是完整图文关系；删除占位图号，保留后续业务判断。
        text = re.sub(rf"{figure}\s*{ellipsis}\s*[。；;，,：:]?", "", text, flags=re.IGNORECASE)
        return text

    def _remove_unmatched_image_reference_sentences(self, analysis: str, images: list[dict]) -> str:
        """逐段删除候选列表外的图片引用句，并保留原有自然段边界。"""
        candidate_tokens = {
            token
            for image in images
            for token in self._figure_tokens_from_caption(str(image.get("caption") or ""))
        }
        # 允许正文使用“图22（左上）”等方位描述匹配图22的候选子图。
        candidate_tokens.update(re.sub(r"\([^)]*\)$", "", token) for token in list(candidate_tokens))
        sentence_pattern = re.compile(r"[^。！？]*[。！？]|[^。！？]+$")
        reference_pattern = re.compile(r"图\s*([0-9]+)(?:\s*[（(]\s*([a-zA-Z])\s*[)）])?")
        cleaned_paragraphs: list[str] = []
        for paragraph in re.split(r"\n\s*\n+", str(analysis or "")):
            kept: list[str] = []
            for sentence_match in sentence_pattern.finditer(paragraph):
                sentence = sentence_match.group(0)
                references = []
                for reference in reference_pattern.finditer(sentence):
                    number = reference.group(1)
                    suffix = reference.group(2)
                    token = f"图{number}"
                    if suffix:
                        token += f"({suffix.lower()})"
                    references.append(self._normalize_image_ref_text(token))
                # 只有引用候选列表外图号的句子才移除，普通正文和段落结构完整保留。
                if references and any(token not in candidate_tokens for token in references):
                    continue
                kept.append(sentence)
            cleaned = "".join(kept).strip()
            if cleaned:
                cleaned_paragraphs.append(cleaned)
        return "\n\n".join(cleaned_paragraphs)
    def _analysis_references_image_or_base(self, analysis: str, image: dict) -> bool:
        """核对模型返回的图片ID在正文中确有图号引用，并允许方位型子图描述。"""
        normalized_analysis = self._normalize_image_ref_text(analysis)
        caption = str(image.get("caption") or "").strip()
        tokens = self._figure_tokens_from_caption(caption)
        for token in tokens:
            if self._analysis_image_token_position(normalized_analysis, token) >= 0:
                return True
            base = re.sub(r"\([^)]*\)$", "", token)
            # 图片ID已消除子图歧义，正文使用“左上/右下”等方位描述时只需核对主图号。
            if re.search(re.escape(base) + r"(?![0-9])", normalized_analysis, flags=re.IGNORECASE):
                return True
        return self._normalize_image_ref_text(caption) in normalized_analysis
    def _analysis_references_image(self, analysis: str, image: dict) -> bool:
        """核对单张候选图的完整图题或图号是否确实出现在正文。"""
        normalized_analysis = self._normalize_image_ref_text(analysis)
        caption = str(image.get("caption") or "").strip()
        tokens = [self._normalize_image_ref_text(caption)]
        tokens.extend(self._figure_tokens_from_caption(caption))
        return any(token and token in normalized_analysis for token in tokens)

    def _images_referenced_by_analysis(
        self,
        analysis: str,
        images: list[dict],
        max_count: int = 8,
    ) -> list[dict]:
        """从正文图号或完整图题反查图片，严格按正文出现顺序返回。"""
        if not analysis or max_count <= 0:
            return []
        normalized_analysis = self._normalize_image_ref_text(analysis)
        references: list[tuple[int, int, dict]] = []
        seen: set[str] = set()
        for index, image in enumerate(images, start=1):
            image_key = str(image.get("image_id") or image.get("image_path") or index)
            if image_key in seen:
                continue
            position = self._image_reference_position(normalized_analysis, image)
            if position < 0:
                continue
            seen.add(image_key)
            references.append((position, index, image))
        references.sort(key=lambda item: (item[0], item[1]))
        return [image for _, _, image in references[:max_count]]

    def _image_reference_position(self, normalized_analysis: str, image: dict) -> int:
        """返回正文中对应图片的首次引用位置；未引用时返回 -1。"""
        caption = str(image.get("caption") or "").strip()
        tokens = [self._normalize_image_ref_text(caption)]
        tokens.extend(self._figure_tokens_from_caption(caption))
        positions: list[int] = []
        for token in dict.fromkeys(item for item in tokens if item):
            position = self._analysis_image_token_position(normalized_analysis, token)
            if position >= 0:
                positions.append(position)
        # 正文可用“图13-14”或“图15(a)至(b)”合并引用，展开后仍只展示该范围内的图。
        range_tokens = self._figure_range_tokens_from_analysis(normalized_analysis)
        if not positions and any(token in range_tokens for token in tokens):
            figure_match = re.search(chr(0x56FE) + r"[0-9]+", normalized_analysis)
            if figure_match:
                positions.append(figure_match.start())
        return min(positions) if positions else -1

    def _analysis_image_token_position(self, normalized_analysis: str, token: str) -> int:
        """匹配独立图号，避免“图1”误命中“图10”。"""
        token = self._normalize_image_ref_text(token)
        if not token:
            return -1
        if re.fullmatch(r"图[0-9]+(?:\([a-zA-Z]+\)|[a-zA-Z])?", token):
            match = re.search(re.escape(token) + r"(?![0-9a-zA-Z(])", normalized_analysis, flags=re.IGNORECASE)
            return match.start() if match else -1
        return normalized_analysis.find(token)

    def _figure_range_tokens_from_analysis(self, normalized_analysis: str) -> set[str]:
        """从正文中的范围和并列图号引用中展开每个子图编号。"""
        tokens: set[str] = set()
        # 模型常写“图22(a)至(g)”，不展开会只命中前一两张。
        figure = chr(0x56FE)
        zhi = chr(0x81F3)
        dao = chr(0x5230)
        dash = chr(0x2014)
        full_tilde = chr(0xFF5E)
        patterns = (
            rf"{figure}([0-9]+)\(([a-zA-Z])\)(?:{zhi}|{dao}|[-{dash}~{full_tilde}])\(([a-zA-Z])\)",
            rf"{figure}([0-9]+)([a-zA-Z])(?:{zhi}|{dao}|[-{dash}~{full_tilde}])([a-zA-Z])",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, normalized_analysis):
                figure_no = match.group(1)
                start = ord(match.group(2).lower())
                end = ord(match.group(3).lower())
                if start > end or end - start > 12:
                    continue
                for code in range(start, end + 1):
                    suffix = chr(code)
                    tokens.add(f"{figure}{figure_no}({suffix})")
                    tokens.add(f"{figure}{figure_no}{suffix}")

        # 处理“图15(a)、图15(b)”和“图15(a)和(b)”这类并列引用，避免正文提到但不展示。
        list_separators = r"(?:、|,|，|和|及|与)"
        paired_patterns = (
            rf"{figure}([0-9]+)\(([a-zA-Z])\){list_separators}{figure}?\1?\(([a-zA-Z])\)",
            rf"{figure}([0-9]+)([a-zA-Z]){list_separators}{figure}?\1?([a-zA-Z])",
            rf"{figure}([0-9]+)\(([a-zA-Z])\){list_separators}\(([a-zA-Z])\)",
        )
        for pattern in paired_patterns:
            for match in re.finditer(pattern, normalized_analysis):
                figure_no = match.group(1)
                for suffix in (match.group(2).lower(), match.group(3).lower()):
                    tokens.add(f"{figure}{figure_no}({suffix})")
                    tokens.add(f"{figure}{figure_no}{suffix}")
        # 处理图号为数字范围或并列引用，例如图13-14、图13至图14。
        numeric_list_separators = "(?:" + "|".join((chr(0x3001), ",", chr(0xFF0C), chr(0x548C), chr(0x53CA), chr(0x4E0E))) + ")"
        numeric_patterns = (
            rf"{figure}([0-9]+)(?:{zhi}|{dao}|[-{dash}~{full_tilde}]){figure}?([0-9]+)",
            rf"{figure}([0-9]+){numeric_list_separators}{figure}?([0-9]+)",
        )
        for pattern in numeric_patterns:
            for match in re.finditer(pattern, normalized_analysis):
                start_no, end_no = int(match.group(1)), int(match.group(2))
                if start_no > end_no or end_no - start_no > 20:
                    continue
                for figure_no in range(start_no, end_no + 1):
                    token = self._normalize_image_ref_text(f"{figure}{figure_no}")
                    tokens.add(token)
                    tokens.update(self._figure_number_variants(token))

        return tokens
    def _image_reference_tokens(self, index: int, image: dict, figure_token_counts: Counter[str]) -> list[str]:
        """构造图片匹配词；重复图号不做短匹配，避免“图2”误匹配多张图。"""
        caption = str(image.get("caption") or "").strip()
        image_id = str(image.get("image_id") or "").strip()
        raw_tokens = [
            f"图片{index}",
            f"候选图片{index}",
            f"图像{index}",
            caption,
            image_id,
        ]
        for token in self._figure_tokens_from_caption(caption):
            # 同一图号存在多个子图或重复图注时，保留全部图号候选，最终由正文边界匹配和 max_count 控制展示数量。
            raw_tokens.append(token)
        return [self._normalize_image_ref_text(token) for token in raw_tokens if token]

    def _analysis_contains_image_token(self, normalized_analysis: str, token: str) -> bool:
        """判断正文是否独立引用某个图号，避免“图2”误命中“图20”。"""
        token = self._normalize_image_ref_text(token)
        if not token:
            return False
        if token.startswith("?") and re.fullmatch(r"?\d+(?:\([^)]*\))?", token):
            return bool(re.search(rf"{re.escape(token)}(?![0-9])", normalized_analysis))
        return token in normalized_analysis

    def _figure_tokens_from_caption(self, caption: str) -> list[str]:
        """从图题开头提取图号，并处理“图10”后紧跟月份或年份的 PDF 粘连文本。"""
        text = str(caption or "").strip()
        match = re.search(r"图\s*([0-9０-９]+|[一二三四五六七八九十]+)(?:\s*[（(]([^）)]+)[）)]|([a-zA-Z]))?", text)
        if not match:
            return []
        number_text = match.group(1)
        suffix = match.group(2) or match.group(3)
        if re.fullmatch(r"[0-9０-９]+", number_text):
            digits = number_text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
            digits = self._figure_digits_from_joined_date(text, match, digits)
            token = f"图{digits}"
        else:
            token = f"图{number_text}"
        if suffix:
            token += f"({suffix})"
        normalized = self._normalize_image_ref_text(token)
        return list(dict.fromkeys([normalized, *self._figure_number_variants(normalized)]))

    def _figure_digits_from_joined_date(self, text: str, match, digits: str) -> str:
        """从“图104月”、“图22025年”这类粘连片段中还原真正图号。"""
        figure_start = match.start(1)
        # 年份是 4 位，例如“图22025年”应解析为“图2”而不是“图22025”。
        year_index = text.find("年", figure_start)
        if year_index >= 0:
            numeric_prefix = re.sub(r"\D", "", text[figure_start:year_index])
            if len(numeric_prefix) > 4:
                year = numeric_prefix[-4:]
                figure_digits = numeric_prefix[:-4]
                if figure_digits and 1900 <= int(year) <= 2099:
                    return figure_digits

        # 月/日/时最多两位，优先保留更长的图号：“图104月”解析为图10，“图64月”解析为图6。
        for marker, max_value in (("月", 12), ("日", 31), ("时", 24)):
            marker_index = text.find(marker, figure_start)
            if marker_index < 0:
                continue
            numeric_prefix = re.sub(r"\D", "", text[figure_start:marker_index])
            candidates: list[str] = []
            for date_len in (1, 2):
                if len(numeric_prefix) <= date_len:
                    continue
                figure_digits = numeric_prefix[:-date_len]
                date_digits = numeric_prefix[-date_len:]
                try:
                    figure_value = int(figure_digits)
                    date_value = int(date_digits)
                except ValueError:
                    continue
                if 1 <= figure_value <= 99 and 1 <= date_value <= max_value:
                    candidates.append(figure_digits)
            if candidates:
                return max(candidates, key=len)

        # 没有粘连日期时，直接使用抽取到的数字；过长则做保守截断。
        return digits[:2] if len(digits) > 3 else digits

    def _figure_number_variants(self, token: str) -> list[str]:
        """为“图4/图四”和“图15(e)/图15e”生成互通匹配词，避免图文错位。"""
        normalized = self._normalize_image_ref_text(token)
        variants: list[str] = []
        compact_suffix = re.sub(r"\(([^)]+)\)", r"\1", normalized)
        if compact_suffix != normalized:
            # 中文报告里常把“图15（e）”写成“图15e”，这里补齐这种常见写法。
            variants.append(compact_suffix)
        digit_match = re.match(r"图([0-9０-９]+)(.*)", normalized)
        if digit_match:
            digits = digit_match.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789"))
            suffix = digit_match.group(2)
            try:
                variants.append(f"图{self._chinese_number(int(digits))}{suffix}")
            except ValueError:
                pass
            return list(dict.fromkeys(variants))
        chinese_match = re.match(r"图([一二三四五六七八九十]+)(.*)", normalized)
        if chinese_match:
            number = self._parse_chinese_number(chinese_match.group(1))
            suffix = chinese_match.group(2)
            if number is not None:
                variants.append(f"图{number}{suffix}")
        return list(dict.fromkeys(variants))

    def _parse_chinese_number(self, text: str) -> int | None:
        """解析 1 到 99 的中文图号，供“图四”匹配“图4”。"""
        digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        if text in digits:
            return digits[text]
        if text == "十":
            return 10
        if "十" in text:
            left, _, right = text.partition("十")
            tens = digits.get(left, 1 if left == "" else 0)
            ones = digits.get(right, 0) if right else 0
            return tens * 10 + ones if tens else None
        return None

    def _chinese_number(self, value: int) -> str:
        """把 1 到 99 的数字转成常见中文图号写法。"""
        digits = "零一二三四五六七八九"
        if value <= 0 or value >= 100:
            return str(value)
        if value < 10:
            return digits[value]
        tens, ones = divmod(value, 10)
        prefix = "十" if tens == 1 else digits[tens] + "十"
        return prefix if ones == 0 else prefix + digits[ones]

    def _normalize_image_ref_text(self, text: str) -> str:
        """归一化图片引用文本，去掉空白和常见标点差异，提高图题匹配稳定性。"""
        text = str(text or "")
        text = re.sub(r"\s+", "", text)
        return text.replace("（", "(").replace("）", ")").replace("：", ":")

    def _images_for_case(self, case: StandardCase, limit: int, resolve_paths: bool = True) -> list[dict]:
        """读取并筛选图片元数据，可按需延迟到模型选图后再下载文件。"""
        if limit <= 0:
            return []
        records = [record for record in self.image_store.list_by_image_ids(case.evidence_image_ids) if self._is_report_image_record(record)]
        if len(records) < limit:
            seen = {record.image_id for record in records}
            for chunk_id in case.source_chunk_ids:
                for record in self.image_store.list_by_chunk_id(chunk_id, limit=limit):
                    if record.image_id not in seen and self._is_report_image_record(record):
                        seen.add(record.image_id)
                        records.append(record)
        records.sort(
            key=lambda record: (
                record.extraction_type == "page_snapshot",
                not bool(record.caption),
                -(record.width or 0) * (record.height or 0),
            )
        )
        images = []
        for record in records:
            data = record.to_dict()
            data["caption"] = self._clean_image_caption(str(data.get("caption") or ""))
            image_type, data_category = self.image_store.classify(record)
            resolved_path = None
            if resolve_paths:
                resolved_path = self._resolve_image_path(record)
                if resolved_path is None:
                    continue
                self._resolved_image_paths[str(record.image_id)] = Path(resolved_path)
            data.update(
                {
                    "image_path": str(resolved_path) if resolved_path is not None else str(record.image_path or ""),
                    "path_resolved": resolved_path is not None,
                    "image_type": image_type,
                    "data_category": data_category,
                }
            )
            images.append(data)
            if len(images) >= limit:
                break
        return images

    def _clean_image_caption(self, caption: str) -> str:
        """规范展示用图注，把“图22025年”转成“图 2 2025年”。"""
        text = re.sub(r"\s+", " ", str(caption or "")).strip()
        match = re.match(r"图\s*([0-9０-９]+)(.*)", text)
        if not match:
            return text
        raw_number = match.group(1)
        digits = raw_number.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
        figure_digits = self._figure_digits_from_joined_date(text, match, digits)
        tail = raw_number[len(figure_digits):] + match.group(2)
        tail = re.sub(r"\s+", " ", tail).strip()
        return f"图 {figure_digits} {tail}".strip() if tail else f"图 {figure_digits}"

    def _is_report_image_record(self, record) -> bool:
        """过滤不适合进入报告和预览区的页面快照类图片。"""
        extraction_type = str(getattr(record, "extraction_type", "") or "").lower()
        if extraction_type == "page_snapshot":
            return False
        image_path = str(getattr(record, "image_path", "") or "").lower()
        if "snapshot" in image_path or "page_snapshot" in image_path:
            return False
        return True

    def _resolve_image_path(self, record) -> Path | None:
        """解析图片本地路径；公司接口图片会在首次使用时下载到运行时缓存。"""
        resolver = getattr(self.image_store, "resolve_image_path", None)
        if callable(resolver):
            resolved = resolver(record)
            if resolved is not None and Path(resolved).is_file():
                return Path(resolved)
        raw_path = Path(record.image_path)
        if raw_path.is_file():
            return raw_path
        cache_dir = getattr(self.image_store, "cache_dir", None)
        client = getattr(self.image_store, "client", None)
        safe_name = str(getattr(record, "image_id", "") or "")
        safe_name_fn = getattr(client, "_safe_filename", None)
        if callable(safe_name_fn):
            safe_name = safe_name_fn(safe_name)
        if cache_dir and safe_name:
            image_dir = Path(cache_dir) / "images"
            for suffix in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
                candidate = image_dir / f"{safe_name}{suffix}"
                if candidate.is_file():
                    return candidate
        metadata_path = Path(getattr(self.image_store, "metadata_path", ""))
        image_root = metadata_path.parent / "document_images"
        candidates = [
            image_root / raw_path.parent.name / raw_path.name,
            image_root / Path(record.source_pdf).stem / raw_path.name,
            image_root / raw_path.name,
        ]
        return next((path for path in candidates if path.is_file()), None)

