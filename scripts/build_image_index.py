from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import settings
from backend.app.services.image_extraction import PdfImageEvidenceExtractor


def clear_document_images() -> None:
    """清空配置中的图片证据目录，保留目录本身。"""
    target = settings.document_images_dir.resolve()
    expected = (settings.data_dir / "document_images").resolve()
    if target != expected:
        raise RuntimeError(f"Unexpected document_images path: {target}")

    target.mkdir(parents=True, exist_ok=True)
    for child in target.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def find_resource_pdfs() -> list[Path]:
    pdfs = sorted(settings.resource_dir.rglob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDF files found in {settings.resource_dir}")
    return pdfs


def load_document_chunks() -> list[dict]:
    chunks_path = settings.document_index_dir / "chunks.json"
    if not chunks_path.exists():
        return []
    return json.loads(chunks_path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    return "".join(str(text).split())


def make_chunk_lookup(chunks: list[dict], page_text_reader):
    chunks_by_pdf: dict[str, list[dict]] = {}
    for chunk in chunks:
        chunks_by_pdf.setdefault(chunk.get("source_pdf", ""), []).append(chunk)

    page_text_cache: dict[str, dict[int, str]] = {}

    def lookup(source_pdf: str, page_no: int) -> list[str]:
        pdf_chunks = chunks_by_pdf.get(source_pdf, [])
        if not pdf_chunks:
            return []
        if source_pdf not in page_text_cache:
            page_text_cache[source_pdf] = page_text_reader(settings.resource_dir / source_pdf)
        page_text = normalize_text(page_text_cache[source_pdf].get(page_no, ""))
        if not page_text:
            return []
        related = []
        for chunk in pdf_chunks:
            content = normalize_text(chunk.get("content", ""))
            if not content:
                continue
            probe = content[:80]
            if probe and probe in page_text:
                related.append(chunk.get("chunk_id", ""))
        return [chunk_id for chunk_id in related if chunk_id]

    return lookup


def build_image_index() -> dict:
    clear_document_images()
    extractor = PdfImageEvidenceExtractor(
        output_dir=settings.document_images_dir,
        metadata_path=settings.image_metadata_path,
    )
    chunks = load_document_chunks()
    extractor.chunk_lookup = make_chunk_lookup(chunks, extractor.page_text_reader)
    records = extractor.build_index(find_resource_pdfs())
    counts = Counter(record.extraction_type for record in records)
    return {
        "image_count": len(records),
        "embedded_count": counts.get("embedded", 0),
        "page_snapshot_count": counts.get("page_snapshot", 0),
        "metadata_path": str(settings.image_metadata_path),
        "image_dir": str(settings.document_images_dir),
        "pdf_count": len({record.source_pdf for record in records}),
        "related_count": sum(1 for record in records if record.related_chunk_ids),
    }


def main() -> None:
    info = build_image_index()
    print("Image evidence index build complete.")
    for key, value in info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
