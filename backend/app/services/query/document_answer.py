"""基于文档证据的非流式回答与本地兜底。"""
from __future__ import annotations

from typing import Any

from backend.app.agents.synthesizer.answer_synthesizer import AnswerSynthesizer
from backend.app.models import DocumentRetrievalHit
from backend.app.services.query.document_retrieval import DocumentRetriever
from backend.app.services.query.result_presenters import QueryResultPresenter


class DocumentQuestionAnswerer:
    """执行检索并向兼容接口返回稳定问答结构。"""

    def __init__(
        self,
        *,
        retriever: DocumentRetriever,
        presenter: QueryResultPresenter,
        llm_client: Any,
    ) -> None:
        self.retriever = retriever
        self.presenter = presenter
        self.llm_client = llm_client

    def answer(
        self,
        question: str,
        top_k: int = 5,
        top_n: int = 3,
        context_case_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """基于文档片段执行非流式 RAG。"""
        retrieval_mode, hits = self.retriever.retrieve(
            question,
            top_k=top_k,
            top_n=top_n,
            context_case_ids=context_case_ids,
        )
        if not hits:
            return {
                "question": question,
                "answer": "未检索到相关证据片段。",
                "retrieval_mode": retrieval_mode,
                "llm_used": False,
                "llm_status": "skipped_no_hits",
                "hits": hits,
            }
        if self.llm_client.is_available():
            try:
                answer = self.llm_client.answer_with_context(
                    question,
                    self.context_blocks(hits),
                )
                return {
                    "question": question,
                    "answer": answer,
                    "retrieval_mode": retrieval_mode,
                    "llm_used": True,
                    "llm_status": "called",
                    "hits": hits,
                }
            except Exception as exc:
                return {
                    "question": question,
                    "answer": self.fallback_answer(question, hits),
                    "retrieval_mode": retrieval_mode,
                    "llm_used": False,
                    "llm_status": f"failed: {exc}",
                    "hits": hits,
                }
        return {
            "question": question,
            "answer": self.fallback_answer(question, hits),
            "retrieval_mode": retrieval_mode,
            "llm_used": False,
            "llm_status": "skipped_no_api_key",
            "hits": hits,
        }

    def context_blocks(self, hits: list[DocumentRetrievalHit]) -> list[str]:
        """从文档片段命中构建模型证据上下文。"""
        blocks = []
        for hit in hits:
            images = self.presenter.images_for_document_hit(hit)
            image_lines = [
                (
                    f"- {image.caption or f'第{image.page_no}页图片'}"
                    f"（图片ID：{image.image_id}，来源：{image.source_pdf} 第{image.page_no}页）"
                )
                for image in images
            ]
            image_block = "\n相关图像证据：\n" + "\n".join(image_lines) if image_lines else ""
            blocks.append(
                f"片段ID：{hit.chunk.chunk_id}\n"
                f"来源：{hit.chunk.source_pdf}\n"
                f"片段序号：{hit.chunk.chunk_no}\n"
                f"相似度：{hit.score}\n"
                f"正文：{hit.chunk.content}"
                f"{image_block}"
            )
        return blocks

    def fallback_answer(self, question: str, hits: list[DocumentRetrievalHit]) -> str:
        """用文档命中结果生成本地兜底答案。"""
        evidence = self.evidence_chunks(hits)
        if self.question_needs_metric_summary(question):
            return AnswerSynthesizer().metric_comparison_from_chunks(question, evidence)
        return AnswerSynthesizer().document_fallback(question, evidence)

    def fast_answer(self, question: str, hits: list[DocumentRetrievalHit]) -> str:
        """生成可立即返回的本地首屏回答。"""
        if not hits:
            return "未检索到相关证据片段。"
        lines = [self.fallback_answer(question, hits)]
        if self.llm_client.is_available():
            lines.append("模型综合分析会继续补充。")
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def evidence_chunks(hits: list[DocumentRetrievalHit]) -> list[Any]:
        """把文档命中转换为回答合成器使用的证据片段。"""
        from backend.app.agents.models import EvidenceChunk

        return [
            EvidenceChunk(
                chunk_id=hit.chunk.chunk_id,
                source_pdf=hit.chunk.source_pdf,
                content=hit.chunk.content,
            )
            for hit in hits
        ]

    @staticmethod
    def question_needs_metric_summary(question: str) -> bool:
        """判断问题是否需要降水指标对比摘要。"""
        return any(
            term in question for term in ("降水量", "雨量", "平均降水", "累计降水")
        ) and any(
            term in question
            for term in ("对比", "相比", "比较", "差异", "高于", "低于")
        )
