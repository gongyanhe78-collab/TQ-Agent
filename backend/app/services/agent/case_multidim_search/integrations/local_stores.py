"""本地共享知识库到多维个例检索数据接口的适配层。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.config import settings
from backend.app.models import DocumentChunk, StandardCase
from backend.app.services.agent.Smart_Case_Match.infrastructure.data_store import LocalCaseDataStore
from backend.app.services.document_store import ChromaDocumentChunkStore
from backend.app.services.image_extraction import BROWSER_IMAGE_EXTENSIONS, IMAGE_TYPE_KEYWORDS, ImageEvidence


# 多维检索与主 RAG 连接同一个集合，不能在主目录里隐式创建第二个空集合。
DOCUMENT_COLLECTION_NAME = settings.document_collection_name
REGION_CITY_MAP = {
    "晋北": ("大同", "朔州", "忻州"),
    "山西北部": ("大同", "朔州", "忻州"),
    "晋中": ("太原", "阳泉", "晋中", "吕梁"),
    "山西中部": ("太原", "阳泉", "晋中", "吕梁"),
    "晋南": ("长治", "晋城", "临汾", "运城"),
    "山西南部": ("长治", "晋城", "临汾", "运城"),
}


class LocalStandardCaseStore:
    """从共享本地资料加载标准化个例，保持原有检索接口不变。"""

    def __init__(self, data_store: LocalCaseDataStore):
        """复用同一个本地数据存储，避免三类资料被重复读取。"""
        self.data_store = data_store

    def list_cases(self) -> list[StandardCase]:
        """返回可直接供结构化检索器使用的全部标准化个例。"""
        return [StandardCase.from_dict(self._normalize_case(item)) for item in self.data_store.cases]

    def get_case(self, case_id: str) -> StandardCase | None:
        """按个例编号返回标准化个例。"""
        item = self.data_store.get_case(case_id)
        return StandardCase.from_dict(self._normalize_case(item)) if item is not None else None

    def _normalize_case(self, item: dict[str, Any]) -> dict[str, Any]:
        """为本地资料补齐区域标签，兼容原有区域硬过滤语义。"""
        data = dict(item)
        areas = [str(value) for value in data.get("affected_areas") or [] if str(value)]
        cities = set(areas) | {str(value) for value in data.get("city_tags") or [] if str(value)}
        for region, region_cities in REGION_CITY_MAP.items():
            # 全省过程覆盖每个业务区域；其他过程只在涉及该区域地市时补充区域标签。
            if "全省" in areas or cities.intersection(region_cities):
                areas.append(region)
        data["affected_areas"] = list(dict.fromkeys(areas))
        return data


class LocalDocumentChunkStore:
    """从共享本地资料按 chunk ID 精确读取正文。"""

    def __init__(self, data_store: LocalCaseDataStore):
        """连接共享目录中已有的 Chroma 向量库，并保留其 JSON 元数据读取能力。"""
        self.data_store = data_store
        self.vector_store = ChromaDocumentChunkStore(
            data_store.data_dir / "document_index",
            DOCUMENT_COLLECTION_NAME,
        )

    def get_chunks(self, chunk_ids: list[str]) -> list[DocumentChunk]:
        """按传入编号顺序从本地 Chroma 关联的元数据中返回文档分块。"""
        return [
            chunk
            for chunk_id in chunk_ids or []
            if (chunk := self.get_chunk(str(chunk_id))) is not None
        ]

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """按编号读取 Chroma 向量库的原文元数据，缺失时回退共享 JSON。"""
        chunk = self.vector_store.get_chunk(str(chunk_id))
        if chunk is not None:
            return chunk
        item = self.data_store.get_chunk(chunk_id)
        return self._to_document_chunk(item) if item is not None else None

    def collection_info(self) -> dict[str, Any]:
        """返回共享 Chroma 向量库状态，兼容既有健康检查协议。"""
        info = self.vector_store.collection_info()
        if int(info.get("count") or 0) or not self.data_store.chunks:
            return info
        # 新建或损坏的 Chroma 索引尚未写入向量时，健康检查仍准确反映可读取的 JSON 元数据。
        return {"count": len(self.data_store.chunks), "dimension": None, "source": "json_fallback"}

    def _to_document_chunk(self, item: dict[str, Any]) -> DocumentChunk:
        """把共享 JSON 记录转换为多维检索使用的数据模型。"""
        return DocumentChunk(
            source_pdf=str(item.get("source_pdf") or ""),
            chunk_id=str(item.get("chunk_id") or ""),
            chunk_no=self._int_value(item.get("chunk_no")),
            content=str(item.get("content") or item.get("text") or ""),
            file_path=str(item.get("file_path") or "") or None,
            embedding=None,
        )

    def _int_value(self, value: Any) -> int:
        """安全转换分块序号，异常数据回退为零。"""
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0


class LocalImageEvidenceStore:
    """从共享本地资料读取图片元数据和图片文件。"""

    def __init__(self, data_store: LocalCaseDataStore):
        """读取元数据文件位置，供健康检查和图片路径解析复用。"""
        self.data_store = data_store
        self.metadata_path = data_store.data_dir / "image_metadata.json"
        self._records_by_id = {
            record.image_id: record
            for item in data_store.images
            if (record := self._to_image_evidence(item)) is not None
        }

    def list_by_chunk_id(self, chunk_id: str, limit: int = 6) -> list[ImageEvidence]:
        """返回与指定正文分块关联的可展示图片。"""
        records = [
            record
            for record in self._records_by_id.values()
            if str(chunk_id) in record.related_chunk_ids and self._is_browser_displayable(record)
        ]
        records.sort(key=self._sort_key)
        return records[:limit]

    def get_image(self, image_id: str) -> ImageEvidence | None:
        """按图片编号返回元数据。"""
        return self._records_by_id.get(str(image_id))

    def list_by_image_ids(self, image_ids: list[str]) -> list[ImageEvidence]:
        """按传入编号顺序返回可展示图片。"""
        return [
            record
            for image_id in image_ids or []
            if (record := self.get_image(str(image_id))) is not None and self._is_browser_displayable(record)
        ]

    def classify(self, record: ImageEvidence) -> tuple[str, str]:
        """沿用既有图片类型关键词，确保报告分类规则不变。"""
        text = record.caption.strip() or record.nearby_text
        for image_type, data_category, keywords in IMAGE_TYPE_KEYWORDS:
            if any(keyword in text for keyword in keywords):
                return image_type, data_category
        if record.extraction_type == "page_snapshot":
            return "page_snapshot", "整页快照"
        return "figure", "图片"

    def metadata_available(self) -> bool:
        """确认图片元数据文件存在且至少包含一条有效记录。"""
        return self.metadata_path.is_file() and bool(self._records_by_id)

    def resolve_image_path(self, record: ImageEvidence) -> Path | None:
        """将历史绝对路径回退解析到共享资料目录下的图片文件。"""
        item = self.data_store.get_image(record.image_id)
        resolved = Path(str(item.get("resolved_path") or "")) if item else Path()
        return resolved if resolved.is_file() else None

    def _to_image_evidence(self, item: dict[str, Any]) -> ImageEvidence | None:
        """把共享 JSON 图片记录转换为多维检索的图片证据模型。"""
        image_id = str(item.get("image_id") or "").strip()
        if not image_id:
            return None
        return ImageEvidence(
            image_id=image_id,
            source_pdf=str(item.get("source_pdf") or ""),
            page_no=self._int_value(item.get("page_no")),
            image_no=self._int_value(item.get("image_no")),
            image_path=str(item.get("image_path") or ""),
            extraction_type=str(item.get("extraction_type") or "embedded"),
            nearby_text=str(item.get("nearby_text") or ""),
            caption=str(item.get("caption") or ""),
            related_chunk_ids=[str(value) for value in item.get("related_chunk_ids") or [] if str(value)],
            width=self._optional_int(item.get("width")),
            height=self._optional_int(item.get("height")),
        )

    def _is_browser_displayable(self, record: ImageEvidence) -> bool:
        """过滤浏览器和 PDF 无法直接展示的图片格式。"""
        return Path(record.image_path).suffix.lower() in BROWSER_IMAGE_EXTENSIONS

    def _sort_key(self, record: ImageEvidence) -> tuple:
        """保持报告中优先展示非快照且图注完整图片的排序规则。"""
        return (record.extraction_type == "page_snapshot", not bool(record.caption), record.page_no, record.image_no)

    def _int_value(self, value: Any) -> int:
        """安全转换页码和图号。"""
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _optional_int(self, value: Any) -> int | None:
        """安全转换可为空的图片尺寸。"""
        try:
            return int(value) if value not in {None, ""} else None
        except (TypeError, ValueError):
            return None
