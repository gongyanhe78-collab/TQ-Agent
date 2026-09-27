"""规范化迁入主目录的 Smart 数据，不重新调用 Embedding API。"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from backend.app.config import settings
from backend.app.models import DocumentChunk
from backend.app.services.document_store import ChromaDocumentChunkStore


STANDARD_CASE_FIELDS = (
    "case_id",
    "title",
    "date_range",
    "disaster_types",
    "affected_areas",
    "source_pdf",
    "source_chunk_ids",
    "evidence_image_ids",
)


def normalize_main_data() -> dict:
    """修正路径和图片关联，并用已有向量重建主 Chroma 集合。"""
    chunks_data = _read_array(settings.document_index_dir / "chunks.json")
    cases_data = _read_array(settings.standard_cases_path)
    images_data = _read_array(settings.image_metadata_path)

    chunks_by_pdf: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks_data:
        source_pdf = str(chunk.get("source_pdf") or "")
        chunk["file_path"] = str(Path("resource") / source_pdf)
        chunks_by_pdf[source_pdf].append(chunk)

    linked_image_count = 0
    for image in images_data:
        source_pdf = str(image.get("source_pdf") or "")
        filename = Path(str(image.get("image_path") or "")).name
        image["image_path"] = str(Path("document_images") / Path(source_pdf).stem / filename)
        related = _related_chunk_ids(image, chunks_by_pdf.get(source_pdf, []))
        image["related_chunk_ids"] = related
        if related:
            linked_image_count += 1

    minimal_cases = [
        {field: case.get(field, [] if field.endswith("_ids") or field in {"disaster_types", "affected_areas"} else "") for field in STANDARD_CASE_FIELDS}
        for case in cases_data
    ]
    _write_array(settings.document_index_dir / "chunks.json", chunks_data)
    _write_array(settings.standard_cases_path, minimal_cases)
    _write_array(settings.image_metadata_path, images_data)

    # 复用 JSON 中已有的云端向量，只重建集合名称和元数据，不产生 API 费用。
    store = ChromaDocumentChunkStore(settings.document_index_dir, settings.document_collection_name)
    store.replace_chunks([DocumentChunk.from_dict(item) for item in chunks_data])
    manifest = _build_manifest(chunks_data, minimal_cases, images_data, linked_image_count)
    manifest_path = settings.data_dir / "knowledge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _related_chunk_ids(image: dict, chunks: list[dict], limit: int = 4) -> list[str]:
    """利用图片附近原文和段落正文的包含关系恢复 chunk 关联。"""
    nearby = _compact(str(image.get("nearby_text") or image.get("caption") or ""))
    if not nearby:
        return []
    exact = []
    scored = []
    nearby_grams = _ngrams(nearby)
    for chunk in chunks:
        content = _compact(str(chunk.get("content") or ""))
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id or len(content) < 6:
            continue
        probe = content[: min(48, len(content))]
        if probe in nearby or (len(nearby) < len(content) and nearby in content):
            exact.append(chunk_id)
            continue
        grams = _ngrams(content)
        denominator = max(1, min(len(grams), len(nearby_grams)))
        score = len(grams.intersection(nearby_grams)) / denominator
        if score >= 0.18:
            scored.append((score, chunk_id))
    if exact:
        return list(dict.fromkeys(exact))[:limit]
    scored.sort(reverse=True)
    return [chunk_id for _, chunk_id in scored[:limit]]


def _compact(text: str) -> str:
    """去掉版面空白，使 PDF 换行不影响文本匹配。"""
    return re.sub(r"\s+", "", text or "")


def _ngrams(text: str, size: int = 3) -> set[str]:
    """生成轻量中文字符片段，用于没有精确包含关系时的兜底匹配。"""
    if len(text) <= size:
        return {text} if text else set()
    return {text[index:index + size] for index in range(len(text) - size + 1)}


def _build_manifest(chunks: list[dict], cases: list[dict], images: list[dict], linked_images: int) -> dict:
    """记录主知识库来源和模型信息，后续可判断向量是否需要更新。"""
    pdfs = []
    for pdf_path in sorted(settings.resource_dir.glob("*.pdf")):
        pdfs.append({"name": pdf_path.name, "sha256": _sha256(pdf_path)})
    dimension = len(chunks[0].get("embedding") or []) if chunks else 0
    return {
        "build_id": f"smart-main-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": "Smart_Case_Match natural-paragraph index",
        "collection_name": settings.document_collection_name,
        "embedding_model": settings.dashscope_embedding_model,
        "embedding_dimension": dimension,
        "chunk_count": len(chunks),
        "standard_case_count": len(cases),
        "image_count": len(images),
        "linked_image_count": linked_images,
        "standard_case_fields": list(STANDARD_CASE_FIELDS),
        "pdfs": pdfs,
    }


def _sha256(path: Path) -> str:
    """分块计算 PDF 摘要，避免一次读入大文件。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_array(path: Path) -> list[dict]:
    """读取并校验数组型 JSON。"""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"数据文件必须是对象数组：{path}")
    return value


def _write_array(path: Path, value: list[dict]) -> None:
    """使用 UTF-8 和稳定缩进写回业务数据。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(normalize_main_data(), ensure_ascii=False, indent=2))
