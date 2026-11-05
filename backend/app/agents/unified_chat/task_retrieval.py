"""按统一意图任务图执行检索、依赖传递和证据排序。"""
from __future__ import annotations

import re
from typing import Any

from backend.app.agents.unified_chat.metric_aggregation import (
    aggregate_metric_facts,
    metric_result_text,
)
from backend.app.models import DocumentRetrievalHit


class TaskRetrievalMixin:
    """执行 RAG 子任务图，保持任务间显式依赖和证据范围。"""

    def _retrieve_rag_task_plan(
        self,
        tasks: list[dict[str, Any]],
        context_case_ids: list[str],
        *,
        top_k: int,
        top_n: int,
    ) -> tuple[
        str,
        list[Any],
        list[str],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        """按显式任务依赖传递个例集合，独立任务不共享隐式范围。"""
        all_hits: dict[str, Any] = {}
        context_blocks: list[str] = []
        task_results: list[dict[str, Any]] = []
        evidence_cases: list[dict[str, Any]] = []
        case_catalog = {
            str(case.case_id): case for case in self._standard_cases()
        }
        initial_case_ids = [
            case_id
            for case_id in context_case_ids
            if case_id in case_catalog
        ]
        task_case_ids: dict[str, list[str]] = {}
        task_types: dict[str, str] = {}
        for task in tasks:
            task_id = str(
                task.get("id") or f"task_{len(task_results) + 1}"
            )
            task_type = str(task.get("type") or "knowledge_qa")
            question = str(task.get("question") or "")
            retrieval_question = str(
                task.get("retrieval_question") or question
            )
            scope = str(task.get("evidence_scope") or "vector")
            conditions = dict(task.get("conditions") or {})
            dependency_ids = [
                str(value) for value in task.get("depends_on") or []
            ]
            # 筛选型依赖已经收窄候选集合，下游必须优先消费其结果。
            narrowing_dependencies = [
                dependency_id
                for dependency_id in dependency_ids
                if task_types.get(dependency_id) == "metric_aggregation"
            ]
            active_dependencies = narrowing_dependencies or dependency_ids
            dependency_scope_locked = bool(narrowing_dependencies)
            dependency_case_ids: list[str] = []
            for dependency_id in active_dependencies:
                for case_id in task_case_ids.get(str(dependency_id), []):
                    if case_id not in dependency_case_ids:
                        dependency_case_ids.append(case_id)
            scoped_case_ids = (
                dependency_case_ids
                if dependency_ids
                else list(initial_case_ids)
            )
            if (
                conditions
                and not scoped_case_ids
                and not dependency_scope_locked
                and scope
                in {
                    "case_chunks",
                    "multi_case_chunks",
                    "image_index",
                    "same_pdf_related",
                }
            ):
                scoped_case_ids = [
                    case.case_id
                    for case in self._structured_cases_for_task(conditions)
                ]
            scoped_cases = [
                case_catalog[case_id]
                for case_id in scoped_case_ids
                if case_id in case_catalog
            ]
            source_pdfs = {
                str(case.source_pdf)
                for case in scoped_cases
                if getattr(case, "source_pdf", "")
            }
            structured_found = False
            resolved_case_ids = list(scoped_case_ids)
            structured_result: dict[str, Any] = {}
            if dependency_scope_locked and not scoped_case_ids:
                # 上游筛选无结果时，下游保持无结果，禁止污染答案。
                hits = []
                retrieval_mode = "unresolved_dependency_scope"
            elif task_type == "metric_aggregation":
                # 指标聚合读取候选个例全部关联正文，并把极值作为依赖结果。
                candidate_cases = (
                    scoped_cases
                    or self._structured_cases_for_task(conditions)
                )
                candidate_chunk_ids = {
                    str(chunk_id)
                    for case in candidate_cases
                    for chunk_id in getattr(
                        case,
                        "source_chunk_ids",
                        [],
                    )
                }
                candidate_sources = {
                    str(getattr(case, "source_pdf", "") or "")
                    for case in candidate_cases
                }
                metric_chunks = [
                    chunk
                    for chunk in self.document_store.list_chunks()
                    if str(getattr(chunk, "chunk_id", ""))
                    in candidate_chunk_ids
                    or str(getattr(chunk, "source_pdf", ""))
                    in candidate_sources
                ]
                chunk_case_ids: dict[str, list[str]] = {}
                for case in candidate_cases:
                    for chunk_id in getattr(
                        case,
                        "source_chunk_ids",
                        [],
                    ):
                        chunk_case_ids.setdefault(
                            str(chunk_id),
                            [],
                        ).append(str(case.case_id))
                structured_result = aggregate_metric_facts(
                    metric_chunks,
                    metric=str(conditions.get("metric") or ""),
                    operator=str(conditions.get("operator") or ""),
                    chunk_case_ids=chunk_case_ids,
                )
                resolved_case_ids = list(
                    structured_result.get("selected_case_ids") or []
                )
                hits = []
                retrieval_mode = "document_metric_facts"
                structured_found = (
                    structured_result.get("status") == "ok"
                )
                if structured_found:
                    context_blocks.append(
                        f"检索目标：{question}\n结构化查询结果："
                        f"{metric_result_text(structured_result)}"
                    )
            elif scope == "structured_cases" and conditions:
                matched_cases = self._structured_cases_for_task(
                    conditions,
                    scoped_case_ids or None,
                )
                resolved_case_ids = [
                    str(case.case_id) for case in matched_cases
                ]
                hits = []
                retrieval_mode = f"structured_task:{task_type}"
                structured_found = bool(matched_cases)
                evidence_text = self._structured_task_evidence(
                    task_type,
                    matched_cases,
                    conditions,
                )
                if evidence_text and matched_cases:
                    context_blocks.append(
                        f"检索目标：{question}\n"
                        f"结构化查询结果：{evidence_text}"
                    )
                evidence_cases.extend(
                    case.to_dict() for case in matched_cases
                )
            elif task_type == "case_analysis" and scoped_case_ids:
                # 分析任务稍后按个例逐一检索，避免串案例证据。
                hits = []
                retrieval_mode = "document_case_analysis_pending"
            elif scope in {"case_chunks", "image_index"} and scoped_case_ids:
                retrieval_mode, hits = (
                    self.dependencies.document_qa.retrieve(
                        retrieval_question,
                        top_k=top_k,
                        top_n=top_n,
                        context_case_ids=scoped_case_ids,
                    )
                )
            elif scope == "multi_case_chunks":
                if len(scoped_case_ids) >= 2:
                    retrieval_mode, hits = (
                        self.dependencies.document_qa.retrieve(
                            retrieval_question,
                            top_k=top_k,
                            top_n=top_n,
                            context_case_ids=scoped_case_ids,
                        )
                    )
                else:
                    # 比较对象未解析时，不允许回退为无范围检索。
                    retrieval_mode, hits = (
                        "unresolved_comparison_scope",
                        [],
                    )
            elif scope == "same_pdf_related" and source_pdfs:
                candidates = [
                    chunk
                    for chunk in self.document_store.list_chunks()
                    if str(chunk.source_pdf) in source_pdfs
                ]
                # 预警任务优先保留目标日期、预报或预警词，并扩大候选数量。
                hits = self._rank_related_evidence_chunks(
                    question,
                    candidates,
                    max(top_n, 8),
                )
                retrieval_mode = "document_same_pdf_related"
            elif scope == "structured_cases":
                hits = []
                retrieval_mode = "structured_cases"
                structured = (
                    self.dependencies.structured_query.answer(
                        question,
                        context_case_ids=scoped_case_ids or None,
                    )
                )
                if structured:
                    structured_found = True
                    answer = str(structured.get("answer") or "")
                    structured_cases = list(
                        structured.get("evidence_cases") or []
                    )
                    if answer and structured_cases:
                        context_blocks.append(
                            f"检索目标：{question}\n"
                            f"结构化查询结果：{answer}"
                        )
                    for case in structured.get("evidence_cases") or []:
                        if isinstance(case, dict):
                            evidence_cases.append(case)
                            case_id = str(case.get("case_id") or "")
                            if (
                                case_id
                                and case_id in case_catalog
                                and case_id not in resolved_case_ids
                            ):
                                resolved_case_ids.append(case_id)
            else:
                retrieval_mode, hits = (
                    self.dependencies.document_qa.retrieve(
                        question,
                        top_k=top_k,
                        top_n=top_n,
                        context_case_ids=None,
                    )
                )
            # case_analysis 必须按个例分别取证，避免某个个例没有任何 chunk。
            case_evidence: list[dict[str, Any]] = []
            if task_type == "case_analysis" and scoped_case_ids:
                per_case_hits: list[Any] = []
                for case_id in scoped_case_ids:
                    case_mode, case_hits = (
                        self.dependencies.document_qa.retrieve(
                            retrieval_question,
                            top_k=top_k,
                            top_n=top_n,
                            context_case_ids=[case_id],
                        )
                    )
                    if case_hits:
                        per_case_hits.extend(case_hits)
                    case_evidence.append(
                        {
                            "case_id": case_id,
                            "retrieval_mode": case_mode,
                            "evidence_chunk_ids": [
                                str(hit.chunk.chunk_id)
                                for hit in case_hits
                            ],
                        }
                    )
                hits = per_case_hits
                retrieval_mode = "document_case_analysis_by_case"

            hit_ids = []
            for hit in hits:
                chunk_id = str(hit.chunk.chunk_id)
                all_hits.setdefault(chunk_id, hit)
                hit_ids.append(chunk_id)
                source_blocks = self.dependencies.document_qa.context_blocks(
                    [hit]
                )
                for source_block in source_blocks:
                    case_label = ""
                    if task_type == "case_analysis":
                        matched_case = next(
                            (
                                item["case_id"]
                                for item in case_evidence
                                if chunk_id
                                in item.get("evidence_chunk_ids", [])
                            ),
                            "",
                        )
                        case_label = (
                            f"\n对应个例：{matched_case}"
                            if matched_case
                            else ""
                        )
                    block = (
                        f"检索目标：{question}{case_label}\n"
                        f"{source_block}"
                    )
                    if block not in context_blocks:
                        context_blocks.append(block)
            task_case_ids[task_id] = resolved_case_ids
            task_types[task_id] = task_type
            task_results.append(
                {
                    "id": task_id,
                    "type": task_type,
                    "route": "rag",
                    "question": question,
                    "depends_on": list(task.get("depends_on") or []),
                    "evidence_scope": scope,
                    "conditions": conditions,
                    "retrieval_mode": retrieval_mode,
                    "status": (
                        "ok" if hits or structured_found else "no_hits"
                    ),
                    "evidence_chunk_ids": hit_ids,
                    "resolved_case_ids": resolved_case_ids,
                    "case_evidence": case_evidence,
                    "structured_result": structured_result,
                }
            )
        for case_id in initial_case_ids:
            evidence_cases.append(case_catalog[case_id].to_dict())
        ordered_hits = list(all_hits.values())
        unique_evidence_cases: list[dict[str, Any]] = []
        seen_case_ids: set[str] = set()
        for case in evidence_cases:
            case_id = str(case.get("case_id") or "")
            if not case_id or case_id in seen_case_ids:
                continue
            seen_case_ids.add(case_id)
            unique_evidence_cases.append(case)
        return (
            "document_task_plan",
            ordered_hits,
            context_blocks,
            task_results,
            unique_evidence_cases,
        )

    @staticmethod
    def _rank_related_evidence_chunks(
        question: str,
        chunks: list[Any],
        limit: int,
    ) -> list[Any]:
        """按日期与证据类型定位同源关联材料。"""
        normalized_question = re.sub(r"\s+", "", question)
        date_terms: list[str] = []
        for year, month, day in re.findall(
            r"(?:(20\d{2})年)?(\d{1,2})月(\d{1,2})日?",
            normalized_question,
        ):
            date_terms.extend(
                [
                    f"{int(month)}月{int(day)}日",
                    f"{int(month):02d}月{int(day):02d}日",
                    f"{int(month):02d}月{int(day):02d}",
                ]
            )
            if year:
                date_terms.append(
                    f"{year}年{int(month)}月{int(day)}日"
                )
        evidence_terms = [
            term
            for term in (
                "预报预警服务",
                "预警服务",
                "预报",
                "预警",
                "重要气象信息",
                "发布时间",
                "灾害实况",
                "灾情",
                "影响区域",
                "环流形势",
                "天气成因",
                "图",
                "图片",
            )
            if term in normalized_question
        ]
        scored: list[tuple[int, int, Any]] = []
        dated_evidence_indices: list[int] = []
        for index, chunk in enumerate(chunks):
            content = re.sub(
                r"\s+",
                "",
                str(chunk.content or ""),
            )
            date_score = sum(
                12 for term in date_terms if term in content
            )
            evidence_score = sum(
                max(2, len(term))
                for term in evidence_terms
                if term in content
            )
            score = date_score + evidence_score
            if date_score > 0 and evidence_score > 0:
                dated_evidence_indices.append(index)
            scored.append((score, index, chunk))
        # 日期型预报预警查询优先返回完整表格区间。
        if dated_evidence_indices and any(
            term in evidence_terms
            for term in (
                "预报",
                "预警",
                "预警服务",
                "预报预警服务",
            )
        ):
            last_anchor = max(dated_evidence_indices)
            headings = [
                index
                for index in range(last_anchor, -1, -1)
                if any(
                    term in str(chunks[index].content or "")
                    for term in (
                        "预警服务情况",
                        "预报预警服务",
                    )
                )
            ]
            start = (
                headings[0]
                if headings
                else max(0, min(dated_evidence_indices) - 1)
            )
            relevant_anchors = [
                index
                for index in dated_evidence_indices
                if index >= start
            ]
            end = min(
                len(chunks),
                max(relevant_anchors or dated_evidence_indices) + 2,
            )
            if end - start <= max(limit, 12):
                score_by_index = {
                    index: score for score, index, _chunk in scored
                }
                return [
                    DocumentRetrievalHit(
                        chunk=chunks[index],
                        score=float(score_by_index.get(index, 0)),
                    )
                    for index in range(start, end)
                ]
        positive = sorted(
            (item for item in scored if item[0] > 0),
            key=lambda item: (-item[0], item[1]),
        )
        selected_indices: set[int] = set()
        for _score, index, _chunk in positive:
            selected_indices.add(index)
            # 表格字段经常跨越前后两个 chunk，必须成组保留。
            if index > 0:
                selected_indices.add(index - 1)
            if index + 1 < len(chunks):
                selected_indices.add(index + 1)
            if len(selected_indices) >= limit:
                break
        if not selected_indices:
            selected_indices.update(range(min(limit, len(chunks))))
        ordered = sorted(selected_indices)[:limit]
        score_by_index = {
            index: score for score, index, _chunk in scored
        }
        return [
            DocumentRetrievalHit(
                chunk=chunks[index],
                score=float(score_by_index.get(index, 0)),
            )
            for index in ordered
        ]
