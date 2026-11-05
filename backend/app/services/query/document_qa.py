"""文档检索与问答服务门面。"""
from __future__ import annotations

from typing import Any

from backend.app.models import DocumentRetrievalHit
from backend.app.services.query.document_answer import DocumentQuestionAnswerer
from backend.app.services.query.document_retrieval import DocumentRetriever


class DocumentQA:
    """向编排器和兼容接口提供统一的文档问答能力。"""

    def __init__(self, *, retriever: DocumentRetriever, answerer: DocumentQuestionAnswerer) -> None:
        self.retriever = retriever
        self.answerer = answerer
        self.document_store = retriever.document_store
        self.llm_client = answerer.llm_client

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        top_n: int = 3,
        context_case_ids: list[str] | None = None,
    ) -> tuple[str, list[DocumentRetrievalHit]]:
        """召回文档证据。"""
        return self.retriever.retrieve(
            question,
            top_k=top_k,
            top_n=top_n,
            context_case_ids=context_case_ids,
        )

    def answer(
        self,
        question: str,
        top_k: int = 5,
        top_n: int = 3,
        context_case_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """执行非流式文档问答。"""
        return self.answerer.answer(
            question,
            top_k=top_k,
            top_n=top_n,
            context_case_ids=context_case_ids,
        )

    def context_blocks(self, hits: list[DocumentRetrievalHit]) -> list[str]:
        """构造文档证据上下文。"""
        return self.answerer.context_blocks(hits)

    def fallback_answer(self, question: str, hits: list[DocumentRetrievalHit]) -> str:
        """生成保留证据的本地兜底回答。"""
        return self.answerer.fallback_answer(question, hits)

    def fast_answer(self, question: str, hits: list[DocumentRetrievalHit]) -> str:
        """生成首屏本地回答。"""
        return self.answerer.fast_answer(question, hits)
