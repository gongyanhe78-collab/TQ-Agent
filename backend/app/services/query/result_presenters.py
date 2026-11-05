"""查询结果中的文档、图片和标准个例响应转换。"""
from __future__ import annotations

from typing import Any

from backend.app.models import DocumentChunk, DocumentRetrievalHit


class QueryResultPresenter:
    """把领域对象转换为现有 API 和会话使用的稳定结构。"""

    def __init__(self, image_store: Any) -> None:
        self.image_store = image_store

    def document_chunk_to_response(self, chunk: DocumentChunk) -> dict[str, Any]:
        """将文档片段转换为前端证据响应。"""
        return {
            "source_pdf": chunk.source_pdf,
            "chunk_id": chunk.chunk_id,
            "chunk_no": chunk.chunk_no,
            "content": chunk.content,
            "file_path": chunk.file_path,
        }

    def images_for_document_hit(
        self,
        hit: DocumentRetrievalHit,
        limit: int = 3,
    ) -> list[Any]:
        """返回片段关联的可展示图片证据。"""
        return self.image_store.list_by_chunk_id(hit.chunk.chunk_id, limit=limit)

    def image_evidence_to_response(self, image: Any) -> dict[str, Any]:
        """将图片证据元数据转换为前端响应。"""
        image_type, data_category = self.image_store.classify(image)
        return {
            "image_id": image.image_id,
            "source_pdf": image.source_pdf,
            "page_no": image.page_no,
            "image_no": image.image_no,
            "extraction_type": image.extraction_type,
            "image_type": image_type,
            "data_category": data_category,
            "caption": image.caption,
            "nearby_text": image.nearby_text,
            "related_chunk_ids": image.related_chunk_ids,
            "width": image.width,
            "height": image.height,
            "url": f"/api/image-evidence/{image.image_id}",
        }

    def document_hit_to_response(self, hit: DocumentRetrievalHit) -> dict[str, Any]:
        """将文档检索命中转换为 API 响应。"""
        return {
            "score": hit.score,
            "chunk": self.document_chunk_to_response(hit.chunk),
            "images": [
                self.image_evidence_to_response(image)
                for image in self.images_for_document_hit(hit)
            ],
        }

    def standard_case_to_response(
        self,
        case: Any,
        *,
        image_type: str | None = None,
        data_category: str | None = None,
    ) -> dict[str, Any]:
        """将标准化个例转换为包含图片证据的 API 响应。"""
        payload = case.to_dict()
        images = self.image_store.list_by_image_ids(case.evidence_image_ids)
        if image_type or data_category:
            images = [
                image
                for image in images
                if self.image_matches_filter(
                    image,
                    image_type=image_type,
                    data_category=data_category,
                )
            ]
        payload["evidence_images"] = [
            self.image_evidence_to_response(image)
            for image in images
        ]
        return payload

    def image_matches_filter(
        self,
        image: Any,
        *,
        image_type: str | None = None,
        data_category: str | None = None,
    ) -> bool:
        """判断图片是否符合调用方指定的证据筛选条件。"""
        current_type, current_category = self.image_store.classify(image)
        if image_type and current_type != image_type:
            return False
        if data_category and data_category not in current_category and data_category not in image.caption:
            return False
        return True

    @staticmethod
    def images_from_document_hit_payloads(
        hit_payloads: list[dict[str, Any]],
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        """从检索命中响应中抽取去重后的图片证据。"""
        images: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in hit_payloads:
            for image in hit.get("images", []):
                image_id = image.get("image_id")
                if not image_id or image_id in seen:
                    continue
                seen.add(image_id)
                images.append(image)
                if len(images) >= limit:
                    return images
        return images
