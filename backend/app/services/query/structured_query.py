"""标准化个例驱动的结构化问答。"""
from __future__ import annotations

import re
from typing import Any

from backend.app.agents.intent_analyzer.analyzers import AgentIntentAnalyzer
from backend.app.agents.planner.planner import AgentPlanner
from backend.app.agents.synthesizer.answer_synthesizer import AnswerSynthesizer
from backend.app.models import DocumentRetrievalHit
from backend.app.services.query.context_cases import ContextCaseResolver
from backend.app.services.query.document_qa import DocumentQA
from backend.app.services.query.result_presenters import QueryResultPresenter


class StructuredQueryService:
    """优先使用标准化个例回答统计、证据清单和对比问题。"""

    def __init__(
        self,
        *,
        standard_case_store: Any,
        answerer: Any,
        presenter: QueryResultPresenter,
        document_qa: DocumentQA,
    ) -> None:
        self.standard_case_store = standard_case_store
        self.answerer = answerer
        self.presenter = presenter
        self.document_qa = document_qa

    def answer(self, question: str, context_case_ids: list[str] | None = None) -> dict[str, Any] | None:
        """返回结构化答案；不属于结构化问题时返回空值。"""
        if (
            ContextCaseResolver.question_uses_context_reference(question)
            and not context_case_ids
        ):
            return self.missing_context_answer(question)
        cases = ContextCaseResolver.cases_scoped_by_context(
            self.standard_case_store.list_cases(),
            context_case_ids,
        )
        answer = self.answerer.answer(question, cases)
        if answer is None:
            return None
        analysis = AgentIntentAnalyzer().analyze(question)
        plan = AgentPlanner().plan(question, analysis)
        evidence_cases = [
            self.presenter.standard_case_to_response(item.case) for item in answer.cases
        ]
        return {
            "question": question,
            "answer": answer.answer,
            "retrieval_mode": f"structured_cases:{answer.intent.intent}",
            "llm_used": False,
            "llm_status": "structured_answer",
            "hit_count": len(answer.cases),
            "hits": [],
            "evidence_cases": evidence_cases,
            "evidence_chunks": [],
            "images": self.answerer.response_images(answer),
            "visuals": answer.visuals,
            "intent": answer.intent.intent,
            "context_case_ids": context_case_ids or [],
            "intent_trace": {
                "intents": [intent.value for intent in plan.intents],
                "required_tools": [step.tool.value for step in plan.steps],
                "forbidden_tools": [tool.value for tool in plan.forbidden_tools],
                "risk_notes": plan.risk_notes,
            },
            "execution_plan": plan.to_dict(),
            "audit": {
                "agent_runs": [
                    {
                        "agent": "intent_analyzer",
                        "status": "ok",
                        "intents": [intent.value for intent in plan.intents],
                        "reason": "结构化问答已完成意图识别",
                    },
                    {
                        "agent": "structured_answerer",
                        "status": "ok",
                        "reason": f"标准化个例链路返回 {len(answer.cases)} 个证据个例",
                    },
                ],
                "executed_tools": [step.tool.value for step in plan.steps],
                "skipped_tools": [tool.value for tool in plan.forbidden_tools],
                "failures": [],
            },
        }

    @staticmethod
    def needs_metric_documents(result: dict[str, Any] | None) -> bool:
        """判断结构化指标缺失后是否需要转入文档证据检索。"""
        if not result:
            return False
        plan = result.get("execution_plan") or {}
        slots = plan.get("slots") or {}
        answer = str(result.get("answer") or "")
        return bool(slots.get("metrics")) and "没有可用" in answer and "结构化数据" in answer

    def metric_document_result(
        self,
        question: str,
        *,
        top_k: int = 5,
        top_n: int = 3,
        context_case_ids: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """用文档片段补齐结构化数据中缺失的指标。"""
        retrieval_mode, hits = self.document_qa.retrieve(
            question,
            top_k=top_k,
            top_n=top_n,
            context_case_ids=context_case_ids,
        )
        if not hits:
            return None
        analysis = AgentIntentAnalyzer().analyze(question)
        plan = AgentPlanner().plan(question, analysis)
        hit_payloads = [self.presenter.document_hit_to_response(hit) for hit in hits]
        answer = AnswerSynthesizer().metric_comparison_from_chunks(
            question,
            self.document_qa.answerer.evidence_chunks(hits),
        )
        images = self.presenter.images_from_document_hit_payloads(hit_payloads)
        return {
            "question": question,
            "answer": answer,
            "retrieval_mode": f"document_metric:{retrieval_mode}",
            "llm_used": False,
            "llm_status": "document_metric_summary",
            "hit_count": len(hits),
            "hits": hit_payloads,
            "hit_payloads": hit_payloads,
            "evidence_cases": [],
            "evidence_chunks": hit_payloads,
            "evidence_images": images,
            "images": images,
            "visuals": self._metric_document_visuals(hits),
            "context_case_ids": context_case_ids or [],
            "intent_trace": {
                "intents": [intent.value for intent in plan.intents],
                "required_tools": [step.tool.value for step in plan.steps],
                "forbidden_tools": [tool.value for tool in plan.forbidden_tools],
                "risk_notes": plan.risk_notes,
            },
            "execution_plan": plan.to_dict(),
            "audit": {
                "agent_runs": [
                    {
                        "agent": "intent_analyzer",
                        "status": "ok",
                        "intents": [intent.value for intent in analysis.intent_types],
                        "reason": "指标型问题已完成意图识别",
                    },
                    {
                        "agent": "metric_data_guard",
                        "status": "missing_structured_metric_data",
                        "reason": "结构化指标缺失，沿 missing_metric_data 条件边转入文档证据检索",
                    },
                    {
                        "agent": "document_retriever",
                        "status": "ok",
                        "reason": f"文档证据检索返回 {len(hits)} 条",
                    },
                    {
                        "agent": "answer_synthesizer",
                        "status": "ok",
                        "reason": "已从零散文档片段抽取指标值并汇总比较",
                    },
                ],
                "executed_tools": [step.tool.value for step in plan.steps],
                "skipped_tools": [],
                "failures": [],
            },
        }

    @staticmethod
    def missing_context_answer(question: str) -> dict[str, Any]:
        """在执行结构化查询前明确拒绝缺少个例范围的指代问题。"""
        analysis = AgentIntentAnalyzer().analyze(question)
        plan = AgentPlanner().plan(question, analysis)
        answer = "\n".join(
            [
                "结论：我没有从上文解析出明确的个例范围，不能直接替你展开。",
                "说明：这类追问需要上一轮回答中有可追溯的个例表或明确的个例标题。",
                "建议：请重新指定时间、数量或个例名称，例如“详细讲讲1月这三个个例”。",
            ]
        )
        return {
            "question": question,
            "answer": answer,
            "retrieval_mode": "structured_cases:context_missing",
            "llm_used": False,
            "llm_status": "context_missing",
            "hit_count": 0,
            "hits": [],
            "evidence_cases": [],
            "evidence_chunks": [],
            "images": [],
            "visuals": [],
            "intent": "context_missing",
            "context_case_ids": [],
            "intent_trace": {
                "intents": [intent.value for intent in plan.intents],
                "required_tools": [step.tool.value for step in plan.steps],
                "forbidden_tools": [tool.value for tool in plan.forbidden_tools],
                "risk_notes": plan.risk_notes,
            },
            "execution_plan": plan.to_dict(),
            "audit": {
                "agent_runs": [
                    {
                        "agent": "context_resolver",
                        "status": "failed",
                        "reason": "问题包含上文指代，但没有解析到明确个例范围",
                    }
                ],
                "executed_tools": [],
                "skipped_tools": [tool.value for tool in plan.forbidden_tools],
                "failures": ["context_reference_unresolved"],
            },
        }

    @staticmethod
    def _metric_document_visuals(hits: list[DocumentRetrievalHit]) -> list[dict[str, Any]]:
        if not hits:
            return []
        return [
            {
                "type": "table",
                "title": "文档指标片段",
                "columns": [
                    {"key": "source", "label": "来源"},
                    {"key": "chunk_id", "label": "片段"},
                    {"key": "content", "label": "内容摘要"},
                ],
                "rows": [
                    {
                        "source": hit.chunk.source_pdf,
                        "chunk_id": hit.chunk.chunk_id,
                        "content": re.sub(r"\s+", " ", hit.chunk.content).strip()[:120],
                    }
                    for hit in hits[:6]
                ],
            }
        ]
