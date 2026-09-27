"""
智能体编排器模块
协调整个只读智能体的执行流程：意图分析、执行规划、工具调用、
答案合成和最终审查，支持多条件分支和自动降级策略。
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from backend.app.models import DocumentRetrievalHit, StandardCase
from backend.app.services.agent.intent_analyzer.analyzers import AgentIntentAnalyzer
from backend.app.services.agent.synthesizer.answer_synthesizer import AnswerSynthesizer
from backend.app.services.agent.models import EvidenceChunk, IntentType, ToolRoute
from backend.app.services.agent.planner.planner import AgentPlanner
from backend.app.services.similar_case_matcher import SimilarCaseQuery
from backend.app.services.structured_qa import StructuredQuestionAnswerer


class AgentOrchestrator:
    """
    智能体编排器
    协调整个只读智能体链路：意图分析 → 执行规划 → 结构化个例检索 →
    相似个例匹配 → 文档向量检索 → 答案合成 → 最终审查。
    """

    def __init__(
        self,
        *,
        cases_provider: Callable[[], list[StandardCase]],
        structured_answerer: StructuredQuestionAnswerer,
        document_retriever: Callable[[str, int, int], tuple[str, list[DocumentRetrievalHit]]],
        document_hit_to_response: Callable[[DocumentRetrievalHit], dict],
        similar_matcher: Any | None = None,
        standard_case_to_response: Callable[[StandardCase], dict] | None = None,
    ):
        self.cases_provider = cases_provider
        self.structured_answerer = structured_answerer
        self.document_retriever = document_retriever
        self.document_hit_to_response = document_hit_to_response
        self.similar_matcher = similar_matcher
        self.standard_case_to_response = standard_case_to_response or (lambda case: case.to_dict())
        self.intent_analyzer = AgentIntentAnalyzer()
        self.planner = AgentPlanner()
        self.answer_synthesizer = AnswerSynthesizer()

    def answer(self, question: str, *, top_k: int = 5, top_n: int = 3) -> dict:
        """
        执行完整的只读智能体问答流程

        Args:
            question: 用户问题文本
            top_k: 向量检索取回的数量
            top_n: 返回结果数量

        Returns:
            完整的答案结果字典，包含回答、证据、执行链路等信息
        """
        agent_runs = []
        failures = []
        analysis = self.intent_analyzer.analyze(question)
        agent_runs.append(
            {
                "agent": "intent_analyzer",
                "status": "ok",
                "intents": [intent.value for intent in analysis.intent_types],
                "reason": "并行意图链路合并完成",
            }
        )
        plan = self.planner.plan(question, analysis)
        agent_runs.append(
            {
                "agent": "planner",
                "status": "ok",
                "tools": [step.tool.value for step in plan.steps],
                "reason": "已生成只读执行计划",
            }
        )
        cases = self.cases_provider()
        structured_metric_gap = None
        try:
            structured = self.structured_answerer.answer(question, cases)
            agent_runs.append(
                {
                    "agent": "structured_answerer",
                    "status": "ok" if structured is not None else "skipped",
                    "reason": "结构化个例链路已执行" if structured is not None else "问题不适合结构化个例链路",
                }
            )
        except Exception as exc:
            structured = None
            failure = f"structured_answerer: {exc}"
            failures.append(failure)
            agent_runs.append({"agent": "structured_answerer", "status": "failed", "reason": str(exc)})
        if structured is not None:
            if _structured_answer_needs_document_metric_summary(structured):
                structured_metric_gap = structured
                agent_runs.append(
                    {
                        "agent": "metric_data_guard",
                        "status": "missing_structured_metric_data",
                        "reason": "结构化指标缺失，沿 missing_metric_data 条件边转入文档证据检索",
                    }
                )
            else:
                evidence_cases = [self.standard_case_to_response(item.case) for item in structured.cases]
                evidence_images = self.structured_answerer.response_images(structured)
                return self._result(
                    question=question,
                    answer=structured.answer,
                    retrieval_mode=f"agent:{structured.intent.intent}",
                    plan=plan,
                    evidence_cases=evidence_cases,
                    evidence_images=evidence_images,
                    visuals=structured.visuals,
                    agent_runs=agent_runs,
                    failures=failures,
                    llm_status="structured_answer",
                )
        if IntentType.SIMILAR_CASE in plan.intents and self.similar_matcher is not None:
            try:
                matches = self.similar_matcher.match(
                    SimilarCaseQuery(
                        q=question,
                        date=plan.slots.raw_date,
                        disaster_type=plan.slots.disasters[0] if plan.slots.disasters else "",
                        area=plan.slots.areas[0] if plan.slots.areas else "",
                        image_type=plan.slots.image_type,
                        top_n=top_n,
                    ),
                    cases,
                )
                agent_runs.append(
                    {
                        "agent": "similar_case_matcher",
                        "status": "ok" if matches else "skipped",
                        "reason": "相似个例链路已执行" if matches else "未找到相似个例",
                    }
                )
            except Exception as exc:
                matches = []
                failures.append(f"similar_case_matcher: {exc}")
                agent_runs.append({"agent": "similar_case_matcher", "status": "failed", "reason": str(exc)})
            if matches:
                answer = self._similar_answer(question, matches)
                evidence_cases = [self.standard_case_to_response(match.case) for match in matches]
                evidence_images = [
                    image
                    for match in matches
                    for image in getattr(match, "evidence_images", [])
                ][:6]
                return self._result(
                    question=question,
                    answer=answer,
                    retrieval_mode="agent:similar_case",
                    plan=plan,
                    evidence_cases=evidence_cases,
                    evidence_images=evidence_images,
                    agent_runs=agent_runs,
                    failures=failures,
                    llm_status="similar_case_answer",
                )
        retrieval_mode, hits = self.document_retriever(question, top_k, top_n)
        agent_runs.append(
            {
                "agent": "document_retriever",
                "status": "ok",
                "reason": f"文档证据检索返回 {len(hits)} 条",
            }
        )
        chunk_evidence = [
            EvidenceChunk(
                chunk_id=hit.chunk.chunk_id,
                source_pdf=hit.chunk.source_pdf,
                content=hit.chunk.content,
            )
            for hit in hits
        ]
        if structured_metric_gap is not None and not hits:
            evidence_cases = [self.standard_case_to_response(item.case) for item in structured_metric_gap.cases]
            return self._result(
                question=question,
                answer=structured_metric_gap.answer,
                retrieval_mode=f"agent:{structured_metric_gap.intent.intent}",
                plan=plan,
                evidence_cases=evidence_cases,
                visuals=structured_metric_gap.visuals,
                agent_runs=agent_runs,
                failures=failures,
                llm_status="structured_metric_gap",
            )
        answer = (
            self.answer_synthesizer.metric_comparison_from_chunks(question, chunk_evidence)
            if plan.slots.metrics and IntentType.COMPARATIVE_ANALYSIS in plan.intents
            else self.answer_synthesizer.document_fallback(question, chunk_evidence)
        )
        return self._result(
            question=question,
            answer=answer,
            retrieval_mode=f"agent:{retrieval_mode}",
            plan=plan,
            hit_payloads=[self.document_hit_to_response(hit) for hit in hits],
            agent_runs=agent_runs,
            failures=failures,
            llm_status="document_fallback",
        )

    def _result(
        self,
        *,
        question: str,
        answer: str,
        retrieval_mode: str,
        plan,
        evidence_cases: list[dict] | None = None,
        evidence_images: list[Any] | None = None,
        hit_payloads: list[dict] | None = None,
        visuals: list[dict] | None = None,
        agent_runs: list[dict] | None = None,
        failures: list[str] | None = None,
        llm_status: str,
    ) -> dict:
        hits = hit_payloads or []
        images = [_image_to_dict(image) for image in (evidence_images or [])]
        return {
            "question": question,
            "answer": answer,
            "retrieval_mode": retrieval_mode,
            "llm_used": False,
            "llm_status": llm_status,
            "hit_count": len(evidence_cases or hits),
            "hit_payloads": hits,
            "evidence_cases": evidence_cases or [],
            "evidence_chunks": hits,
            "evidence_images": images,
            "visuals": visuals or _metric_document_visuals(plan, hits),
            "intent_trace": {
                "intents": [intent.value for intent in plan.intents],
                "required_tools": [step.tool.value for step in plan.steps],
                "forbidden_tools": [tool.value for tool in plan.forbidden_tools],
                "risk_notes": plan.risk_notes,
            },
            "execution_plan": plan.to_dict(),
            "audit": {
                "agent_runs": agent_runs or [],
                "executed_tools": [step.tool.value for step in plan.steps],
                "skipped_tools": [tool.value for tool in plan.forbidden_tools],
                "failures": failures or [],
            },
        }

    def _similar_answer(self, question: str, matches: list[Any]) -> str:
        topic = question.strip().rstrip("？?")
        if any(term in question for term in ("重点关注", "关注什么", "服务提示", "今天", "当前", "决策")):
            lines = [f"{topic}：", "建议重点关注："]
            lines.extend(_decision_focus_lines(question, matches))
            lines.append("参考个例：")
        else:
            lines = [f"{topic}：最值得优先参考的是以下历史过程。"]
        for index, match in enumerate(matches[:3], start=1):
            reasons = "；".join(match.reasons[:2]) if match.reasons else "存在时空和灾种相似性"
            lines.append(f"{index}. {match.case.date_range} {match.case.title}：相似度 {match.score}，{reasons}。")
        lines.append("提示：这里是基于历史标准化个例的经验参考，不替代实时监测和正式预报结论。")
        return "\n".join(lines)


def _image_to_dict(image: Any) -> dict:
    if isinstance(image, dict):
        return image
    if hasattr(image, "to_dict"):
        return image.to_dict()
    return {
        "image_id": getattr(image, "image_id", ""),
        "source_pdf": getattr(image, "source_pdf", ""),
        "page_no": getattr(image, "page_no", 0),
        "caption": getattr(image, "caption", ""),
    }


def _structured_answer_needs_document_metric_summary(structured: Any) -> bool:
    intent = getattr(structured, "intent", None)
    metrics = getattr(intent, "metrics", []) if intent else []
    answer = getattr(structured, "answer", "")
    return bool(metrics) and "没有可用" in answer and "结构化数据" in answer


def _metric_document_visuals(plan: Any, hits: list[dict]) -> list[dict]:
    slots = getattr(plan, "slots", None)
    if not getattr(slots, "metrics", []):
        return []
    if not hits:
        return []
    rows = []
    for hit in hits[:6]:
        chunk = hit.get("chunk", {}) if isinstance(hit, dict) else {}
        rows.append(
            {
                "source": chunk.get("source_pdf", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "content": re.sub(r"\s+", " ", str(chunk.get("content", ""))).strip()[:120],
            }
        )
    return [
        {
            "type": "table",
            "title": "文档指标片段",
            "columns": [
                {"key": "source", "label": "来源"},
                {"key": "chunk_id", "label": "片段"},
                {"key": "content", "label": "内容摘要"},
            ],
            "rows": rows,
        }
    ]


def _decision_focus_lines(question: str, matches: list[Any]) -> list[str]:
    lines = []
    if any(term in question for term in ("强对流", "雷暴", "大风", "冰雹")):
        lines.append("- 短临监测：盯紧雷达回波快速发展、移向移速、短时大风和冰雹信号。")
        lines.append("- 落区风险：重点看山西北部、地形抬升区和前期已受影响区域的重复影响。")
    if any(term in question for term in ("暴雨", "强降水", "降水", "6月")):
        lines.append("- 降水风险：关注累计雨量、短时雨强、山洪地质灾害和城市内涝风险。")
    for match in matches[:2]:
        tips = getattr(match, "forecast_tips", []) or []
        for tip in tips[:1]:
            lines.append(f"- 历史经验：{_compact_text(tip, limit=110)}")
    return lines or ["- 先确认灾种、落区、强度和持续时间，再对照相似历史个例校验风险点。"]


def _compact_text(text: str, limit: int = 120) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip("，；、。") + "。"
