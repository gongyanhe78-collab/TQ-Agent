"""验证轻量 PDF 文档化路径，避免知识库重建重新引入大型推理依赖。"""
from __future__ import annotations

import sys
from pathlib import Path

from backend.app.services.knowledge_builder import build_paragraph_vectors


def _create_two_page_pdf() -> bytes:
    """生成不依赖第三方写库的最小双页 PDF，供真实读取行为测试使用。"""
    first_page_text = (
        b"BT\n/F1 12 Tf\n72 720 Td\n"
        b"(Page one title) Tj\n0 -20 Td\n(First page detail) Tj\nET"
    )
    second_page_text = (
        b"BT\n/F1 12 Tf\n72 720 Td\n"
        b"(Page two title) Tj\n0 -20 Td\n(Second page detail) Tj\nET"
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 7 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 8 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            f"<< /Length {len(first_page_text)} >>\nstream\n".encode()
            + first_page_text
            + b"\nendstream"
        ),
        (
            f"<< /Length {len(second_page_text)} >>\nstream\n".encode()
            + second_page_text
            + b"\nendstream"
        ),
    ]

    parts = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    offsets: list[int] = []
    current_offset = 0
    for part in parts:
        current_offset += len(part)
    for index, body in enumerate(objects, start=1):
        offsets.append(current_offset)
        object_bytes = f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
        parts.append(object_bytes)
        current_offset += len(object_bytes)

    xref_offset = current_offset
    parts.append(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    parts.extend(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    parts.append(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return b"".join(parts)


def test_load_pdf_elements_reads_pages_without_unstructured(monkeypatch, tmp_path: Path) -> None:
    """知识库重建应直接使用轻量读取器，并在禁用 Unstructured 时仍能保留页码。"""
    pdf_path = tmp_path / "lightweight.pdf"
    pdf_path.write_bytes(_create_two_page_pdf())
    monkeypatch.setitem(sys.modules, "unstructured.partition.pdf", None)

    elements = build_paragraph_vectors._load_pdf_elements(pdf_path)

    assert {element.page_no for element in elements} == {1, 2}
    assert any(
        "Page one title" in element.text and element.page_no == 1
        for element in elements
    )
    assert any(
        "Second page detail" in element.text and element.page_no == 2
        for element in elements
    )
