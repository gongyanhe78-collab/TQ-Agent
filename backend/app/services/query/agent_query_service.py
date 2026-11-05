"""旧智能体查询协议的兼容业务服务。"""
from __future__ import annotations

from typing import Any

from backend.app.agents.intent_analyzer.analyzers import AgentIntentAnalyzer
from backend.app.agents.models import IntentType
from backend.app.agents.orchestrator.orchestrator import AgentOrchestrator
from backend.app.agents.planner.planner import AgentPlanner
from backend.app.agents.reviewer.final_reviewer import AgentFinalReviewer
from backend.app.services.query.context_cases import ContextCaseResolver
from backend.app.services.query.document_qa import DocumentQA
from backend.app.services.query.result_presenters import QueryResultPresenter
from backend.app.services.query.structured_query import StructuredQueryService


class AgentQueryService:
    """执行兼容旧智能体协议的结构化查询与最终审查。"""

    def __init__(
        self,
        *,
        standard_case_store: Any,
        similar_case_matcher: Any,
        presenter: QueryResultPresenter,
        document_qa: DocumentQA,
        structured_query: StructuredQueryService,
        answerer: Any,
        llm_client: Any,
    ) -> None:
        self.standard_case_store = standard_case_store
        self.similar_case_matcher = similar_case_matcher
        self.presenter = presenter
        self.document_qa = document_qa
        self.structured_query = structured_query
        self.answerer = answerer
        self.llm_client = llm_client

    def answer(
        self,
        question: str,
        top_k: int = 5,
        top_n: int = 3,
        context_case_ids: list[str] | None = None,
        conversation_history: list[dict] | None = None,
    ) -> dict[str, Any]:
        """回答一次智能体查询并执行最终一致性审查。"""
        result = self._answer_once(
            question,
            top_k=top_k,
            top_n=top_n,
            context_case_ids=context_case_ids,
        )
        reviewer = AgentFinalReviewer(llm_client=self.llm_client)
        review = reviewer.review(question, result, conversation_history or [])
        result = self._apply_final_review(result, review)
        if (
            review.get("should_retry")
            and review.get("normalized_question")
            and review["normalized_question"] != question
        ):
            retry_question = str(review["normalized_question"])
            retried = self._answer_once(
                retry_question,
                top_k=top_k,
                top_n=top_n,
                context_case_ids=context_case_ids,
            )
            retry_review = reviewer.review(question, retried, conversation_history or [])
            result = self._apply_final_review(
                retried,
                {
                    **retry_review,
                    "status": "retry",
                    "reason": f"已使用规范化问题重新执行：{retry_question}",
                    "normalized_question": retry_question,
                },
            )
            result["question"] = question
        return result

    def _answer_once(
        self,
        question: str,
        *,
        top_k: int,
        top_n: int,
        context_case_ids: list[str] | None,
    ) -> dict[str, Any]:
        analysis = AgentIntentAnalyzer().analyze(question)
        if analysis.intent_types == [IntentType.GENERAL_RAG]:
            plan = AgentPlanner().plan(question, analysis)
            result = self.document_qa.answer(
                question,
                top_k=top_k,
                top_n=top_n,
                context_case_ids=context_case_ids,
            )
            hit_payloads = [
                self.presenter.document_hit_to_response(hit)
                for hit in result.get("hits", [])
            ]
            images = self.presenter.images_from_document_hit_payloads(hit_payloads)
            return {
                **result,
                "hit_count": len(result.get("hits", [])),
                "hit_payloads": hit_payloads,
                "evidence_cases": [],
                "evidence_chunks": hit_payloads,
                "evidence_images": images,
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
                            "reason": "普通问答未命中结构化业务意图",
                        },
                        {
                            "agent": "document_retriever",
                            "status": "ok",
                            "reason": f"文档证据检索返回 {len(result.get('hits', []))} 条",
                        },
                    ],
                    "executed_tools": [step.tool.value for step in plan.steps],
                    "skipped_tools": [tool.value for tool in plan.forbidden_tools],
                    "failures": [],
                },
            }
        cases_provider = lambda: ContextCaseResolver.cases_scoped_by_context(
            self.standard_case_store.list_cases(), context_case_ids
        )
        agent = AgentOrchestrator(
            cases_provider=cases_provider,
            structured_answerer=self.answerer,
            document_retriever=lambda query, k, n: self.document_qa.retrieve(
                query,
                top_k=k,
                top_n=n,
                context_case_ids=context_case_ids,
            ),
            document_hit_to_response=self.presenter.document_hit_to_response,
            similar_matcher=self.similar_case_matcher,
            standard_case_to_response=self.presenter.standard_case_to_response,
        )
        result = agent.answer(question, top_k=top_k, top_n=top_n)
        result["context_case_ids"] = context_case_ids or []
        return result

    @staticmethod
    def _apply_final_review(result: dict, review: dict) -> dict:
        reviewed = dict(result)
        audit = dict(reviewed.get("audit") or {})
        agent_runs = list(audit.get("agent_runs") or [])
        agent_runs.append(
            {
                "agent": "final_reviewer",
                "status": review.get("status", "ok"),
                "reason": review.get("reason", ""),
                "normalized_question": review.get(
                    "normalized_question", reviewed.get("question", "")
                ),
                "llm_used": bool(review.get("llm_used", False)),
            }
        )
        audit["agent_runs"] = agent_runs
        reviewed["audit"] = audit
        final_answer = str(review.get("final_answer") or "")
        if final_answer:
            reviewed["answer"] = final_answer
            reviewed["llm_used"] = True
            reviewed["llm_status"] = "final_review"
        return reviewed
