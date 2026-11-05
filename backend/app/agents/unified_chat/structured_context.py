"""结构化事实到回答上下文的转换与任务证据描述。"""
from __future__ import annotations

import json
import re
from typing import Any

from backend.app.agents.case_multidim_search.schemas import CaseSearchQuery


class StructuredContextMixin:
    """整理结构化查询结果，供回答模型和任务计划消费。"""

    @staticmethod
    def _structured_render_context(
        structured: dict[str, Any],
        common: dict[str, Any],
    ) -> list[str]:
        """把结构化事实整理成回答模型上下文。"""
        result = structured if isinstance(structured, dict) else {}
        blocks: list[str] = []
        answer = str(result.get("answer") or "").strip()
        if answer:
            blocks.append(
                "已核验的结构化查询结果"
                "（只能整理和润色，不得改动其中的数量、时间、名称或范围）：\n"
                + answer[:12000]
            )

        # 结构化字段比已经排版的答案更适合生成表格、分组和逐例说明。
        safe_cases: list[dict[str, Any]] = []
        for item in list(common.get("evidence_cases") or [])[:40]:
            if not isinstance(item, dict):
                continue
            safe_cases.append(
                {
                    "title": str(item.get("title") or ""),
                    "date_range": str(item.get("date_range") or ""),
                    "disaster_types": list(
                        item.get("disaster_types") or []
                    )[:12],
                    "affected_areas": list(
                        item.get("affected_areas") or []
                    )[:12],
                    "summary": str(item.get("summary") or "")[:1200],
                    "weather_facts": str(
                        item.get("weather_facts") or ""
                    )[:1600],
                    "forecast_focus": str(
                        item.get("forecast_focus") or ""
                    )[:800],
                    "source_pdf": str(item.get("source_pdf") or ""),
                }
            )
        if safe_cases:
            blocks.append(
                "已核验的标准个例字段（可按用户要求整理成表格或分组）：\n"
                + json.dumps(safe_cases, ensure_ascii=False)
            )

        safe_chunks: list[str] = []
        for item in list(common.get("evidence_chunks") or [])[:24]:
            if not isinstance(item, dict):
                continue
            chunk = (
                item.get("chunk")
                if isinstance(item.get("chunk"), dict)
                else item
            )
            content = str(chunk.get("content") or "").strip()
            if content:
                safe_chunks.append(
                    f"来源：{str(chunk.get('source_pdf') or '')}\n"
                    f"{content[:2400]}"
                )
        if safe_chunks:
            blocks.append(
                "可追溯证据片段（仅用于支持事实，不足时不得补写）：\n"
                + "\n\n".join(safe_chunks)
            )

        return blocks or [
            "本轮结构化检索没有返回可供回答模型使用的事实内容。"
        ]

    @staticmethod
    def _apply_same_scope_feedback_to_tasks(
        tasks: list[dict[str, Any]],
        session_guidances: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """把需要补证据的反馈作为同范围任务约束。"""
        normalized_tasks = [dict(task) for task in tasks]
        dimensions: list[str] = []
        guidance_ids: list[str] = []
        for guidance in session_guidances or []:
            if not guidance.get("needs_same_scope_retrieval"):
                continue
            for value in guidance.get("requested_dimensions") or []:
                text = re.sub(r"\s+", " ", str(value or "")).strip()[:60]
                if text and text not in dimensions:
                    dimensions.append(text)
            if guidance.get("guidance_id"):
                guidance_ids.append(str(guidance["guidance_id"]))
        if not dimensions:
            return normalized_tasks

        # 只扩展已有 RAG 任务，避免反馈越权改变路由。
        target = next(
            (
                task
                for task in normalized_tasks
                if str(task.get("type") or "") == "case_analysis"
            ),
            None,
        )
        if target is None:
            return normalized_tasks
        dimension_text = "、".join(dimensions[:5])
        target["retrieval_question"] = str(
            target.get("question") or ""
        )[:6000]
        target["question"] = (
            f"{str(target.get('question') or '').rstrip()}\n"
            "在原有时间、地点、灾种和证据范围内，补充用户反馈要求的维度："
            f"{dimension_text}。"
        )[:6000]
        target["feedback_guidance_ids"] = guidance_ids[:5]
        return normalized_tasks

    def _structured_cases_for_task(
        self,
        conditions: dict[str, Any],
        scoped_case_ids: list[str] | None = None,
    ) -> list[Any]:
        """直接使用模型归一化条件查询标准个例。"""
        allowed = {
            key: conditions.get(key)
            for key in (
                "start_date",
                "end_date",
                "years",
                "months",
                "disaster_types",
                "cities",
                "areas",
            )
            if conditions.get(key)
        }
        query = CaseSearchQuery(**allowed, limit=500)
        candidates = self._standard_cases()
        if scoped_case_ids:
            wanted = set(scoped_case_ids)
            candidates = [
                case
                for case in candidates
                if str(case.case_id) in wanted
            ]
        original_order = {
            str(case.case_id): index
            for index, case in enumerate(candidates)
        }
        matched_cases = [
            match.case
            for match in self.structured_case_retriever.search(
                candidates,
                query,
            )
        ]
        return sorted(
            matched_cases,
            key=lambda case: original_order.get(
                str(case.case_id),
                len(original_order),
            ),
        )

    def _structured_task_evidence(
        self,
        task_type: str,
        cases: list[Any],
        conditions: dict[str, Any],
    ) -> str:
        """把同一命中集合投影成计数或清单。"""
        if not cases:
            return "当前标准化个例库中没有命中给定结构化条件的个例。"
        if task_type == "aggregate_statistics":
            return (
                f"命中个例总数：{len(cases)}。"
                "计数对象为上游筛选出的同一组标准化个例。"
            )
        lines = [
            f"- {case.case_id}｜{case.date_range}｜{case.title}｜"
            f"灾种：{'、'.join(case.disaster_types) or '未标注'}｜"
            f"来源：{case.source_pdf}"
            for case in cases
        ]
        if task_type == "case_identification":
            heading = (
                f"已定位 {len(cases)} 个符合条件的标准化个例，"
                "以下 case_id 作为依赖任务的输入："
            )
        elif task_type == "case_listing":
            heading = (
                f"按用户要求列出 {len(cases)} 个个例的名称和发生时间："
            )
        else:
            heading = f"结构化检索共命中 {len(cases)} 个个例："
        coverage = self._structured_coverage_note(conditions)
        return "\n".join([heading, *lines, *([coverage] if coverage else [])])

    def _structured_coverage_note(
        self,
        conditions: dict[str, Any],
    ) -> str:
        """区分用户请求范围和当前库内记录跨度。"""
        if (
            conditions.get("time_scope_type") != "calendar_year"
            or not conditions.get("years")
        ):
            return ""
        knowledge = self.intent_agent.knowledge_range()
        years = "、".join(
            str(value) for value in conditions.get("years") or []
        )
        return (
            f"范围说明：用户请求的是 {years} 完整自然年；"
            "当前标准个例记录跨度为 "
            f"{knowledge.start_date or '未知'} 至 "
            f"{knowledge.end_date or '未知'}。"
            "上述数量只代表当前库内已收录个例，不能据此宣称未覆盖时段没有相关过程。"
        )

    @staticmethod
    def _task_context_fallback(context_blocks: list[str]) -> str:
        """模型不可用时仅返回各结构化子任务的已核验结果。"""
        answers = [
            block.split("结构化查询结果：", 1)[1].strip()
            for block in context_blocks
            if "结构化查询结果：" in block
        ]
        return "\n\n".join(answers) or (
            "已找到相关证据，但当前无法生成综合回答。"
        )

    def _structured_common(
        self,
        result: dict,
        run_id: str,
        question: str,
    ) -> dict:
        """把标准个例结构化回答转换成统一聊天结果。"""
        return {
            "agent_type": "document_rag",
            "run_id": run_id,
            "question": question,
            "answer": result.get("answer", ""),
            "retrieval_mode": result.get(
                "retrieval_mode",
                "structured_cases",
            ),
            "hit_count": result.get("hit_count", 0),
            "hits": result.get("hits", []),
            "evidence_cases": result.get("evidence_cases", []),
            "evidence_chunks": result.get("evidence_chunks", []),
            "images": result.get("images", []),
            "image_display_mode": result.get(
                "image_display_mode",
                "hidden",
            ),
            "visuals": result.get("visuals", []),
            "reports": [],
            "analysis": result.get("analysis", {}),
            "query_conditions": result.get("query_conditions", {}),
            "intent_trace": result.get("intent_trace", {}),
            "execution_plan": result.get("execution_plan", {}),
            "audit": result.get("audit", {}),
            "llm_used": bool(result.get("llm_used", False)),
        }
