"""三个查询入口共享的统一业务服务。"""
from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from typing import Any

from backend.app.api.schemas import AgentQueryRequest, QueryRequest, StreamQueryRequest
from backend.app.agents.unified_chat.orchestration_models import (
    AgentBudget,
    ExecutionContext,
)
from backend.app.agents.unified_chat.runtime import AgentRuntime
from backend.app.services.query.agent_query_service import AgentQueryService
from backend.app.services.query.context_cases import ContextCaseResolver
from backend.app.services.query.document_qa import DocumentQA
from backend.app.services.query.result_presenters import QueryResultPresenter
from backend.app.services.query.structured_query import StructuredQueryService


class ChatService:
    """三个查询入口共享的查询业务门面。"""

    def __init__(
        self,
        *,
        settings: Any,
        session_store: Any,
        document_qa: DocumentQA,
        structured_query: StructuredQueryService,
        context_cases: ContextCaseResolver,
        agent_query: AgentQueryService,
        presenter: QueryResultPresenter,
        orchestrator_factory: Callable[..., Any],
    ) -> None:
        self.settings = settings
        self.session_store = session_store
        self.document_qa = document_qa
        self.structured_query = structured_query
        self.context_cases = context_cases
        self.agent_query = agent_query
        self.presenter = presenter
        self.orchestrator_factory = orchestrator_factory

    def query_document(self, request: QueryRequest) -> dict[str, Any]:
        """执行结构化优先的非流式文档查询。"""
        structured = self.structured_query.answer(request.question)
        if structured is not None:
            if self.structured_query.needs_metric_documents(structured):
                metric_document = self.structured_query.metric_document_result(
                    request.question,
                    top_k=request.top_k,
                    top_n=request.top_n,
                )
                if metric_document is not None:
                    return metric_document
            return structured
        result = self.document_qa.answer(
            request.question,
            top_k=request.top_k,
            top_n=request.top_n,
        )
        return {
            "question": result["question"],
            "answer": result["answer"],
            "retrieval_mode": result["retrieval_mode"],
            "llm_used": result["llm_used"],
            "llm_status": result["llm_status"],
            "hit_count": len(result["hits"]),
            "hits": [self.presenter.document_hit_to_response(hit) for hit in result["hits"]],
        }

    def query_agent(self, request: AgentQueryRequest) -> dict[str, Any]:
        """执行兼容旧智能体协议的结构化查询。"""
        if request.session_id and self.session_store.get_session(request.session_id) is None:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Session not found")
        context_case_ids: list[str] = []
        conversation_history: list[dict] = []
        if request.session_id:
            conversation_history = self.session_store.list_messages(request.session_id)
            context_case_ids = self.context_cases.from_messages(
                conversation_history,
                request.question,
            )
            self.session_store.add_message(request.session_id, "user", request.question)
        result = self.agent_query.answer(
            request.question,
            top_k=request.top_k,
            top_n=request.top_n,
            context_case_ids=context_case_ids,
            conversation_history=conversation_history,
        )
        hits = result.get("hit_payloads", [])
        if request.session_id:
            self._save_agent_answer(request, result, hits, context_case_ids)
        return {
            "question": result["question"],
            "answer": result["answer"],
            "session_id": request.session_id,
            "hit_count": result.get("hit_count", len(hits)),
            "hits": hits if request.return_context else [],
            "intent_trace": result.get("intent_trace", {}),
            "execution_plan": result.get("execution_plan", {}),
            "evidence_cases": result.get("evidence_cases", []),
            "evidence_chunks": result.get("evidence_chunks", []),
            "evidence_images": result.get("evidence_images", []),
            "visuals": result.get("visuals", []),
            "audit": result.get("audit", {}),
            "retrieval": {
                "mode": result["retrieval_mode"],
                "embedding_model": self.settings.embedding_model,
                "rerank_model": self.settings.rerank_model,
                "top_k": request.top_k,
                "top_n": request.top_n,
            },
            "llm": {
                "model": self.settings.chat_model,
                "used": result["llm_used"],
                "status": result["llm_status"],
            },
            "metadata": request.metadata or {},
        }

    async def stream(
        self,
        request: StreamQueryRequest,
        is_disconnected: Callable[[], Awaitable[bool]],
        execution_context: ExecutionContext | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """执行统一流式查询主链路并原样转发编排事件。"""
        context = execution_context or ExecutionContext()
        if not context.session_id and request.session_id:
            context = replace(context, session_id=request.session_id)
        runtime = AgentRuntime(
            AgentBudget.from_settings(self.settings),
            context,
        )
        if self._factory_accepts_runtime():
            orchestrator = self.orchestrator_factory(runtime)
        else:
            orchestrator = self.orchestrator_factory()
        async for event in orchestrator.stream(request, is_disconnected):
            yield event

    def _factory_accepts_runtime(self) -> bool:
        """判断旧工厂是否需要兼容，只有显式参数才注入请求运行时。"""
        try:
            parameters = inspect.signature(
                self.orchestrator_factory
            ).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            }
            for parameter in parameters
        )

    def _save_agent_answer(
        self,
        request: AgentQueryRequest,
        result: dict[str, Any],
        hits: list[dict],
        context_case_ids: list[str],
    ) -> None:
        """保存智能体兼容接口返回的助手消息和统一记忆元数据。"""
        answer_images = result.get("images") or self.presenter.images_from_document_hit_payloads(hits)
        memory = self.assistant_memory_metadata(
            question=request.question,
            answer=result["answer"],
            retrieval_mode=result["retrieval_mode"],
            context_case_ids=result.get("context_case_ids", context_case_ids),
            hit_payloads=hits,
            evidence_cases=result.get("evidence_cases", []),
            evidence_chunks=result.get("evidence_chunks", hits),
            images=answer_images[:6],
            visuals=result.get("visuals", [])[:6],
            intent_trace=result.get("intent_trace", {}),
            execution_plan=result.get("execution_plan", {}),
            audit=result.get("audit", {}),
        )
        self.session_store.add_message(
            request.session_id,
            "assistant",
            result["answer"],
            metadata=memory,
        )

    @staticmethod
    def assistant_memory_metadata(
        *,
        question: str,
        answer: str,
        retrieval_mode: str,
        context_case_ids: list[str],
        hit_payloads: list[dict] | None = None,
        evidence_cases: list[dict] | None = None,
        evidence_chunks: list[dict] | None = None,
        images: list[dict] | None = None,
        visuals: list[dict] | None = None,
        intent_trace: dict | None = None,
        execution_plan: dict | None = None,
        audit: dict | None = None,
    ) -> dict[str, Any]:
        """构建会话隔离的统一记忆包，并保留顶层兼容字段。"""
        chunk_memory = evidence_chunks if evidence_chunks is not None else (hit_payloads or [])
        memory = {
            "version": 1,
            "question": question,
            "answer": answer,
            "retrieval_mode": retrieval_mode,
            "context_case_ids": context_case_ids,
            "evidence_cases": evidence_cases or [],
            "evidence_chunks": chunk_memory,
            "images": images or [],
            "visuals": visuals or [],
            "intent_trace": intent_trace or {},
            "execution_plan": execution_plan or {},
            "audit": audit or {},
        }
        return {
            "memory": memory,
            "images": memory["images"],
            "visuals": memory["visuals"],
            "evidence_cases": memory["evidence_cases"],
            "evidence_chunks": memory["evidence_chunks"],
            "context_case_ids": memory["context_case_ids"],
            "intent_trace": memory["intent_trace"],
            "execution_plan": memory["execution_plan"],
            "audit": memory["audit"],
            "retrieval_mode": memory["retrieval_mode"],
        }
