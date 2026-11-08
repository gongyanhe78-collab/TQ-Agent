"""知识库重建服务。"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class KnowledgeService:
    """执行自然段向量、图片证据和标准个例的全量重建。"""

    def __init__(self, *, settings: Any) -> None:
        self.settings = settings

    def rebuild_documents(self, pdf_paths: list[Path] | None = None) -> dict[str, Any]:
        """重建统一知识库，并返回现有接口使用的统计字段。"""
        from backend.app.services.knowledge_builder.build_paragraph_vectors import (
            build as build_vectors,
        )
        from backend.app.services.knowledge_builder.build_standard_cases import build as build_cases

        requested_files = [Path(path).name for path in (pdf_paths or [])]
        vector_result = build_vectors(resource_dir=self.settings.resource_dir)
        case_result = build_cases(use_llm=False)
        return {
            "processed_pdfs": requested_files
            or [path.name for path in sorted(self.settings.resource_dir.glob("*.pdf"))],
            "indexed": vector_result["chunk_count"],
            "chunk_count": vector_result["chunk_count"],
            "image_count": vector_result["image_count"],
            "case_count": case_result["case_count"],
            "vector_dimension": vector_result["document_collection"].get("dimension"),
            "vector_count": vector_result["document_collection"].get("count"),
            "vector_source": vector_result["document_collection"].get("source"),
        }
