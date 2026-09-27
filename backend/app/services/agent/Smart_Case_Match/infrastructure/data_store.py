"""Smart Case Match 的本地数据读取和证据路径解析。"""
from __future__ import annotations

import json
import logging
from pathlib import Path, PureWindowsPath
from typing import Any

from backend.app.config import settings


logger = logging.getLogger("uvicorn.error")


class LocalCaseDataStore:
    """只读加载本地标准个例、chunk 和图片元数据。"""

    def __init__(self, data_dir: Path | None = None):
        """初始化数据目录，并建立 chunk 到个例的多值反向索引。"""
        # 默认读取主项目 data，避免替换目录后重新产生 Smart 专用知识库。
        self.data_dir = Path(data_dir or settings.smart_case_data_dir)
        self.document_images_dir = self.data_dir / "document_images"
        self._load_errors: list[str] = []
        self.cases = self._load_json("standard_cases1.json")
        self.chunks = self._load_json("document_index/chunks.json")
        self.images = self._load_json("image_metadata.json")
        self._chunks_by_id = {str(item.get("chunk_id")): item for item in self.chunks}
        self._images_by_id = {str(item.get("image_id")): item for item in self.images}
        self._case_by_id = {str(item.get("case_id")): item for item in self.cases}
        self._case_ids_by_chunk: dict[str, list[str]] = {}
        for case in self.cases:
            for chunk_id in case.get("source_chunk_ids") or []:
                self._case_ids_by_chunk.setdefault(str(chunk_id), []).append(str(case.get("case_id")))

    def _load_json(self, relative_path: str) -> list[dict[str, Any]]:
        """按 UTF-8 读取数组型 JSON，并把格式异常记录到健康检查而不是静默吞掉。"""
        path = self.data_dir / relative_path
        if not path.is_file():
            return []
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            message = f"{relative_path}: JSON读取失败：{exc}"
            self._load_errors.append(message)
            logger.error("[SmartCaseMatch][Data] %s", message)
            return []
        if not isinstance(value, list):
            message = f"{relative_path}: 顶层必须是数组，实际为 {type(value).__name__}"
            self._load_errors.append(message)
            logger.error("[SmartCaseMatch][Data] %s", message)
            return []
        invalid_count = sum(1 for item in value if not isinstance(item, dict))
        if invalid_count:
            message = f"{relative_path}: 有 {invalid_count} 个元素不是对象，已忽略"
            self._load_errors.append(message)
            logger.error("[SmartCaseMatch][Data] %s", message)
            return [item for item in value if isinstance(item, dict)]
        return value

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        """按 ID 获取标准个例。"""
        return self._case_by_id.get(str(case_id))

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """按 ID 获取 chunk。"""
        return self._chunks_by_id.get(str(chunk_id))

    def get_case_chunks(self, case: dict[str, Any]) -> list[dict[str, Any]]:
        """按个例的正式关联顺序读取正文，允许边界 chunk 被多个个例共享。"""
        return [
            chunk
            for chunk_id in case.get("source_chunk_ids") or []
            if (chunk := self.get_chunk(str(chunk_id))) is not None
        ]

    def case_ids_for_chunk(self, chunk_id: str) -> list[str]:
        """返回一个 chunk 所属的全部个例 ID。"""
        return list(self._case_ids_by_chunk.get(str(chunk_id), []))

    def get_image(self, image_id: str) -> dict[str, Any] | None:
        """读取图片元数据，并把旧目录路径解析到当前本地资料目录。"""
        raw = self._images_by_id.get(str(image_id))
        if raw is None:
            return None
        item = dict(raw)
        resolved = self.resolve_image_path(item)
        item["resolved_path"] = str(resolved) if resolved else ""
        return item

    def get_case_images(self, case: dict[str, Any], limit: int = 6) -> list[dict[str, Any]]:
        """按个例图片 ID 返回可展示证据，过滤不存在或不可访问的图片。"""
        result = []
        for image_id in case.get("evidence_image_ids") or []:
            image = self.get_image(str(image_id))
            if image and image.get("resolved_path"):
                result.append(image)
            if len(result) >= limit:
                break
        return result

    def resolve_image_path(self, image: dict[str, Any]) -> Path | None:
        """优先使用元数据路径，不可用时按来源 PDF 目录和文件名定位图片。"""
        raw_value = str(image.get("image_path") or "")
        raw_path = Path(raw_value)
        if raw_path.is_file():
            return raw_path

        # 图片元数据由 Windows 生成时包含反斜杠，Linux Path 无法正确取得文件名。
        filename = PureWindowsPath(raw_value).name
        source_value = str(image.get("source_pdf") or "")
        source_stem = PureWindowsPath(source_value).stem
        if filename and source_stem:
            candidate = self.document_images_dir / source_stem / filename
            if candidate.is_file():
                return candidate
        if filename:
            matches = list(self.document_images_dir.rglob(filename))
            if matches:
                return matches[0]
        return None

    def health(self) -> dict[str, Any]:
        """返回本地资料数量和关键关联完整性，供页面健康检查使用。"""
        missing_chunk_refs = 0
        missing_image_refs = 0
        for case in self.cases:
            missing_chunk_refs += sum(1 for item in case.get("source_chunk_ids") or [] if str(item) not in self._chunks_by_id)
            missing_image_refs += sum(1 for item in case.get("evidence_image_ids") or [] if str(item) not in self._images_by_id)
        return {
            "status": "ok" if self.cases and self.chunks and not self._load_errors else "degraded",
            "data_source": "local",
            "case_count": len(self.cases),
            "chunk_count": len(self.chunks),
            "image_count": len(self.images),
            "missing_chunk_refs": missing_chunk_refs,
            "missing_image_refs": missing_image_refs,
            "shared_chunk_count": sum(1 for ids in self._case_ids_by_chunk.values() if len(ids) > 1),
            "data_load_errors": list(self._load_errors),
        }
