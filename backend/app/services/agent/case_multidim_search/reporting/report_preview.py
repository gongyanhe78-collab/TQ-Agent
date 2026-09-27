"""为 PDF 报告生成可持久化的第一页缩略图。"""
from __future__ import annotations

from pathlib import Path


def thumbnail_path_for(pdf_path: Path) -> Path:
    """返回与 PDF 同目录、同名的缩略图路径。"""
    return Path(pdf_path).with_suffix(".thumbnail.png")


def ensure_report_thumbnail(pdf_path: Path, *, scale: float = 1.25) -> Path:
    """按需渲染 PDF 第一页，已有且未过期时直接复用。"""
    source = Path(pdf_path)
    if not source.is_file():
        raise FileNotFoundError(f"PDF 报告不存在：{source}")
    target = thumbnail_path_for(source)
    if target.is_file() and target.stat().st_mtime >= source.stat().st_mtime:
        return target

    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(source))
    if len(document) < 1:
        raise ValueError("PDF 报告没有可渲染页面")
    page = document[0]
    bitmap = page.render(scale=scale)
    image = bitmap.to_pil()
    temporary = target.with_suffix(".tmp.png")
    try:
        # 先写临时文件再替换，避免前端读取到尚未写完的缩略图。
        image.save(temporary, format="PNG", optimize=True)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
        image.close()
        page.close()
        document.close()
    return target
