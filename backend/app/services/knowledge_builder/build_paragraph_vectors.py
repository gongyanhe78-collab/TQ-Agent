"""按 PDF 文档化结果重建多维个例检索使用的段落 chunk、图片证据和 Chroma 向量库。"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable



def _bootstrap_project_root() -> Path:
    """把项目根目录加入 sys.path，保证脚本用绝对路径直接执行时也能导入 backend 包。"""
    for parent in Path(__file__).resolve().parents:
        if (parent / "backend").is_dir() and (parent / "resource").is_dir():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return parent
    raise RuntimeError("未找到包含 backend 和 resource 的项目根目录")


PROJECT_ROOT = _bootstrap_project_root()

from backend.app.config import settings
from backend.app.models import DocumentChunk
from backend.app.services.document_store import ChromaDocumentChunkStore
from backend.app.services.model_client.embedding import get_embedding_client
from backend.app.services.image_extraction import ImageEvidence, PdfImageEvidenceExtractor


DATA_DIR = settings.data_dir
DOCUMENT_COLLECTION_NAME = settings.document_collection_name
DOCUMENT_INDEX_DIR = settings.document_index_dir
IMAGE_DIR = settings.document_images_dir
IMAGE_METADATA_PATH = settings.image_metadata_path
CHUNKS_PATH = DOCUMENT_INDEX_DIR / "chunks.json"
MAX_CHUNK_CHARS = 500
PARAGRAPH_TARGET_CHARS = 420
SHORT_ELEMENT_CHARS = 90
TITLE_CATEGORIES = {"Title", "Header"}

NUMBERED_HEADING_RE = re.compile(r"^(?:第[一二三四五六七八九十百千万0-9]+[章节篇]|[一二三四五六七八九十0-9]+[、.．）)])")
CASE_TITLE_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百千万0-9]+[章节篇]\s*)?"
    r"(?:[一二三四五六七八九十0-9]+[、.．）)]\s*)?"
    r"(?:\d{1,2}\s*月\s*)?\d{1,2}\s*(?:日|号)"
    r"(?:\s*[-~～—至到]\s*(?:\d{1,2}\s*月\s*)?\d{1,2}\s*(?:日|号)?)?"
    r".{0,40}(?:天气过程|过程|暴雨|大暴雨|强降水|雷暴|雷雨|大风|雨雪|降雪|寒潮|低温|霜冻|高温|沙尘|冰雹)"
)
SECTION_TITLE_RE = re.compile(
    r"^(?:[一二三四五六七八九十0-9]+[、.．）)]\s*)?"
    r"(?:天气实况|环流形势|影响系统|预报服务|服务情况|灾情|过程概况|总结|成因分析|预警情况|降水实况|大风实况)$"
)
SENTENCE_END_RE = re.compile(r"[。！？；;]$")


@dataclass
class PdfElementText:
    """保存 partition_pdf 返回元素的正文、页码和类型，便于后续段落重组。"""

    text: str
    page_no: int
    category: str = ""


@dataclass
class ParagraphItem:
    """段落级中间结果，标题会单独标记，避免被正文盲目合并。"""

    text: str
    page_no: int
    is_title: bool = False


def find_project_root() -> Path:
    """从脚本位置向上寻找项目根目录，默认 PDF 资源目录由这里推导。"""
    for parent in Path(__file__).resolve().parents:
        if (parent / "resource").is_dir() and (parent / "backend").is_dir():
            return parent
    raise RuntimeError("未找到包含 resource 和 backend 的项目根目录")


def chunk_id_for(pdf_path: Path, index: int) -> str:
    """生成原 PDF 文件名加顺序号形式的 chunk 编号。"""
    return f"{Path(pdf_path).stem}-chunk-{index:03d}"


def split_paragraphs(text: str) -> list[str]:
    """给测试和临时文本使用的纯文本段落拆分入口，空行仍视为明确段落边界。"""
    blocks = [_clean_text(block) for block in re.split(r"\n\s*\n+", text or "") if block.strip()]
    paragraphs = [ParagraphItem(text=block, page_no=1, is_title=False) for block in blocks if block]
    return [item.text for item in _split_oversized_paragraphs(paragraphs)]


def _prepare_unstructured_environment() -> None:
    """设置 Unstructured 运行缓存目录，避免 numba 把缓存写到不可控位置。"""
    settings.numba_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(settings.numba_cache_dir))


def _load_pdf_elements(pdf_path: Path) -> list[PdfElementText]:
    """使用 partition_pdf 完成 PDF 文档化，并保留元素顺序、页码和分类。"""
    _prepare_unstructured_environment()
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as exc:
        raise RuntimeError("缺少 PDF 文档化依赖，请安装 unstructured[pdf]。") from exc

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="fast",
        infer_table_structure=False,
        extract_images_in_pdf=False,
    )
    records: list[PdfElementText] = []
    for element in elements:
        text = _clean_text(str(element))
        if not text:
            continue
        metadata = getattr(element, "metadata", None)
        page_no = int(getattr(metadata, "page_number", None) or 0)
        category = str(getattr(element, "category", "") or type(element).__name__)
        records.append(PdfElementText(text=text, page_no=page_no, category=category))
    if not records:
        raise RuntimeError(f"PDF 文档化未产生有效文本：{pdf_path.name}")
    return records


def _clean_text(text: str) -> str:
    """清理 PDF 抽取中的多余空白，同时保留英文、数字和单位之间必要的空格。"""
    value = re.sub(r"\r\n?", "\n", text or "")
    value = re.sub(r"\s*\n\s*", " ", value)
    value = re.sub(r"[ \t\f\v]+", " ", value).strip()
    value = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", value)
    value = re.sub(r"(?<=\d)\s+(?=[年月日号])", "", value)
    value = re.sub(r"(?<=[年月日号])\s+(?=\d)", "", value)
    return value


def _is_title_element(element: PdfElementText) -> bool:
    """识别应该单独保留的标题元素，尤其是带日期的天气过程标题。"""
    compact = re.sub(r"\s+", "", element.text.strip())
    # 有些 PDF 会把几乎每一行都标成 Title，所以这里不能单纯相信 category。
    if CASE_TITLE_RE.search(compact):
        return True
    if SECTION_TITLE_RE.search(compact):
        return True
    return False


def _join_wrapped_text(left: str, right: str) -> str:
    """合并被 PDF 断开的相邻短行，中文边界直接拼接，英文边界保留一个空格。"""
    if not left:
        return right
    if not right:
        return left
    if re.search(r"[A-Za-z0-9]$", left) and re.search(r"^[A-Za-z0-9]", right):
        return f"{left} {right}"
    return f"{left}{right}"


def _should_flush_before_next(buffer_text: str, next_text: str, next_page: int, buffer_page: int) -> bool:
    """判断当前自然段是否应在加入下一个元素前结束。"""
    if next_page and buffer_page and next_page != buffer_page:
        return True
    if len(buffer_text) >= PARAGRAPH_TARGET_CHARS:
        return True
    if len(next_text) > MAX_CHUNK_CHARS:
        return True
    if SENTENCE_END_RE.search(buffer_text) and len(buffer_text) >= 180 and len(next_text) > SHORT_ELEMENT_CHARS:
        return True
    return False


def _merge_adjacent_elements(elements: list[PdfElementText]) -> list[ParagraphItem]:
    """在 partition_pdf 之后合并相邻短元素，把 PDF 行级碎片重组成自然段。"""
    paragraphs: list[ParagraphItem] = []
    buffer_text = ""
    buffer_page = 0

    def flush_buffer() -> None:
        nonlocal buffer_text, buffer_page
        text = buffer_text.strip()
        if text:
            paragraphs.append(ParagraphItem(text=text, page_no=buffer_page, is_title=False))
        buffer_text = ""
        buffer_page = 0

    for element in elements:
        text = element.text.strip()
        if not text:
            continue
        if _is_title_element(element):
            flush_buffer()
            paragraphs.append(ParagraphItem(text=text, page_no=element.page_no, is_title=True))
            continue
        if buffer_text and _should_flush_before_next(buffer_text, text, element.page_no, buffer_page):
            flush_buffer()
        if not buffer_text:
            buffer_text = text
            buffer_page = element.page_no
        else:
            buffer_text = _join_wrapped_text(buffer_text, text)
    flush_buffer()
    return paragraphs


def _split_long_text(text: str) -> list[str]:
    """用 500 字上限切分超长自然段；没有 LangChain 时使用内置兜底切分。"""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return [text[start:start + MAX_CHUNK_CHARS] for start in range(0, len(text), MAX_CHUNK_CHARS)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_CHARS,
        chunk_overlap=0,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""],
        keep_separator=True,
    )
    return [part.strip() for part in splitter.split_text(text) if part.strip()]


def _split_oversized_paragraphs(paragraphs: list[ParagraphItem]) -> list[ParagraphItem]:
    """标题直接保留，正文自然段超过 500 字时再进入上限切分。"""
    chunks: list[ParagraphItem] = []
    for paragraph in paragraphs:
        if paragraph.is_title or len(paragraph.text) <= MAX_CHUNK_CHARS:
            chunks.append(paragraph)
            continue
        for part in _split_long_text(paragraph.text):
            chunks.append(ParagraphItem(text=part, page_no=paragraph.page_no, is_title=False))
    return chunks


def _paragraph_items_for_pdf(pdf_path: Path) -> list[ParagraphItem]:
    """完成单个 PDF 的文档化、短元素合并和 500 字上限切分。"""
    elements = _load_pdf_elements(pdf_path)
    paragraphs = _merge_adjacent_elements(elements)
    return _split_oversized_paragraphs(paragraphs)


def build_chunk_records(pdf_paths: Iterable[Path]) -> tuple[list[dict], dict[tuple[str, int], list[str]]]:
    """按 PDF 自然段生成未向量化 chunk，并建立页码到 chunk 的图片关联索引。"""
    records: list[dict] = []
    page_chunk_ids: dict[tuple[str, int], list[str]] = {}
    for pdf_path in sorted(Path(path) for path in pdf_paths):
        chunk_no = 0
        for paragraph in _paragraph_items_for_pdf(pdf_path):
            chunk_no += 1
            chunk_id = chunk_id_for(pdf_path, chunk_no)
            records.append(
                {
                    "source_pdf": pdf_path.name,
                    "chunk_id": chunk_id,
                    "chunk_no": chunk_no,
                    "page_no": paragraph.page_no,
                    "content": paragraph.text,
                    "is_title": paragraph.is_title,
                    "file_path": str(pdf_path),
                    "embedding": [],
                    "evidence_image_ids": [],
                    "evidence_image_paths": [],
                }
            )
            if paragraph.page_no:
                page_chunk_ids.setdefault((pdf_path.name, paragraph.page_no), []).append(chunk_id)
    return records, page_chunk_ids


def embed_chunks(records: list[dict]) -> None:
    """调用项目配置中的向量模型，为每个自然段 chunk 写入 embedding。"""
    client = get_embedding_client()
    texts = [record["content"] for record in records]
    embeddings = client.embed_documents(texts) if texts else []
    if len(embeddings) != len(records):
        raise RuntimeError(f"向量数量不匹配：文本 {len(records)} 条，向量 {len(embeddings)} 条")
    for record, embedding in zip(records, embeddings):
        record["embedding"] = embedding


def _layout_figure_groups(page) -> list[tuple[float, float, float, float, str]]:
    """按图注位置将并排、网格和重叠蒙版图片归并为完整总图。"""
    images = list(page.images or [])
    if not images:
        return []
    lines: dict[float, list[dict]] = {}
    for word in page.extract_words(keep_blank_chars=False, use_text_flow=True) or []:
        lines.setdefault(round(float(word.get("top") or 0), 1), []).append(word)
    captions = []
    for line_words in lines.values():
        ordered = sorted(line_words, key=lambda item: float(item.get("x0") or 0))
        text = " ".join(str(item.get("text") or "") for item in ordered).strip()
        if re.match(r"^\u56fe\s*\d+", text):
            captions.append((min(float(item.get("top") or 0) for item in ordered), text))
    groups = []
    previous_caption_top = 0.0
    for caption_top, caption in sorted(captions):
        candidates = [
            image for image in images
            if previous_caption_top <= float(image.get("top") or 0) < caption_top
            and float(image.get("bottom") or 0) <= caption_top + 3.0
        ]
        if candidates:
            groups.append((
                min(float(image.get("x0") or 0) for image in candidates),
                min(float(image.get("top") or 0) for image in candidates),
                max(float(image.get("x1") or 0) for image in candidates),
                max(float(image.get("bottom") or 0) for image in candidates),
                caption,
            ))
        previous_caption_top = caption_top
    return groups


def _is_nearly_blank(image) -> bool:
    """过滤白色占位图、透明蒙版和信息量极低的小图片。"""
    from PIL import ImageStat

    if image.width < 80 or image.height < 60:
        return True
    sample = image.resize((64, 64)).convert("RGB")
    stat = ImageStat.Stat(sample)
    white_ratio = sum(1 for pixel in sample.getdata() if min(pixel) >= 250) / (64 * 64)
    return white_ratio > 0.985 or max(stat.stddev) < 2.0


def read_layout_figures(pdf_path: Path) -> list[dict]:
    """从 PDF 渲染页中按版面裁出完整总图，避免并排子图丢失或图注串联。"""
    import pdfplumber
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf_path))
    payloads = []
    with pdfplumber.open(pdf_path) as layout_pdf:
        for page_no, layout_page in enumerate(layout_pdf.pages, start=1):
            groups = _layout_figure_groups(layout_page)
            if not groups:
                continue
            rendered = document[page_no - 1].render(scale=2.5).to_pil().convert("RGB")
            scale_x = rendered.width / float(layout_page.width)
            scale_y = rendered.height / float(layout_page.height)
            for image_no, (x0, top, x1, bottom, caption) in enumerate(groups, start=1):
                padding = 5.0
                box = (
                    max(0, int((x0 - padding) * scale_x)),
                    max(0, int((top - padding) * scale_y)),
                    min(rendered.width, int((x1 + padding) * scale_x)),
                    min(rendered.height, int((bottom + padding) * scale_y)),
                )
                cropped = rendered.crop(box)
                if _is_nearly_blank(cropped):
                    continue
                buffer = io.BytesIO()
                cropped.save(buffer, format="PNG")
                payloads.append({
                    "page_no": page_no,
                    "image_no": image_no,
                    "extension": ".png",
                    "data": buffer.getvalue(),
                    "width": cropped.width,
                    "height": cropped.height,
                    "caption": caption,
                })
    return payloads


class LayoutPdfImageEvidenceExtractor(PdfImageEvidenceExtractor):
    """仅供多维检索数据构建使用的版面级图片提取器。"""

    def extract_pdf(self, pdf_path: Path) -> list[ImageEvidence]:
        pdf_path = Path(pdf_path)
        page_text = self.page_text_reader(pdf_path)
        layout_payloads = read_layout_figures(pdf_path)
        layout_pages = {int(payload.get("page_no") or 0) for payload in layout_payloads}
        fallback_payloads = [
            payload for payload in self.embedded_image_reader(pdf_path)
            if int(payload.get("page_no") or 0) not in layout_pages
        ]
        records = self._records_from_layout_payloads(pdf_path, [*layout_payloads, *fallback_payloads], page_text)
        records.extend(self._records_from_payloads(pdf_path, "page_snapshot", self.page_snapshot_renderer(pdf_path), page_text))
        return records

    def _records_from_layout_payloads(self, pdf_path: Path, payloads: list[dict], page_text: dict[int, str]) -> list[ImageEvidence]:
        records = self._records_from_payloads(pdf_path, "embedded", payloads, page_text)
        captions = {(int(payload.get("page_no") or 0), int(payload.get("image_no") or 0)): str(payload.get("caption") or "") for payload in payloads}
        for record in records:
            caption = captions.get((record.page_no, record.image_no))
            if caption:
                record.caption = caption
        return records


def build_image_metadata(
    pdf_paths: Iterable[Path],
    page_chunk_ids: dict[tuple[str, int], list[str]],
    include_page_snapshots: bool = True,
) -> list[ImageEvidence]:
    """抽取 PDF 图片证据，并用页码把图片关联到同页的自然段 chunk。"""
    def lookup(source_pdf: str, page_no: int) -> list[str]:
        return page_chunk_ids.get((source_pdf, page_no), [])

    renderer = None if include_page_snapshots else (lambda _pdf_path: [])
    extractor = LayoutPdfImageEvidenceExtractor(
        output_dir=IMAGE_DIR,
        metadata_path=IMAGE_METADATA_PATH,
        page_snapshot_renderer=renderer,
        chunk_lookup=lookup,
    )
    if IMAGE_DIR.exists():
        # 全量重建时清空旧图片目录，避免旧月份或旧 chunk 关联残留。
        shutil.rmtree(IMAGE_DIR)
    records: list[ImageEvidence] = []
    for pdf_path in sorted(Path(path) for path in pdf_paths):
        try:
            records.extend(extractor.extract_pdf(pdf_path))
        except Exception as exc:
            print(f"图片抽取跳过：{pdf_path.name}，原因：{exc}")
    IMAGE_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    IMAGE_METADATA_PATH.write_text(
        json.dumps([record.to_dict() for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return records


def attach_images_to_chunks(chunks: list[dict], images: list[ImageEvidence]) -> None:
    """把图片编号和图片地址回填到关联 chunk，方便后续排查证据链。"""
    by_chunk: dict[str, list[ImageEvidence]] = {}
    for image in images:
        for chunk_id in image.related_chunk_ids:
            by_chunk.setdefault(chunk_id, []).append(image)
    for chunk in chunks:
        related = by_chunk.get(chunk["chunk_id"], [])
        chunk["evidence_image_ids"] = [image.image_id for image in related]
        chunk["evidence_image_paths"] = [image.image_path for image in related]


def to_document_chunks(records: list[dict]) -> list[DocumentChunk]:
    """转换为项目现有 ChromaDocumentChunkStore 接受的文档块结构。"""
    return [
        DocumentChunk(
            source_pdf=str(record.get("source_pdf") or ""),
            chunk_id=str(record.get("chunk_id") or ""),
            chunk_no=int(record.get("chunk_no") or 0),
            content=str(record.get("content") or ""),
            file_path=str(record.get("file_path") or ""),
            embedding=record.get("embedding") or [],
        )
        for record in records
    ]


def write_chunks(chunks: list[DocumentChunk]) -> dict:
    """将 chunk 同时写入 ChromaDB 和 JSON 备份，并返回集合状态。"""
    store = ChromaDocumentChunkStore(DOCUMENT_INDEX_DIR, DOCUMENT_COLLECTION_NAME)
    store.replace_chunks(chunks)
    return store.collection_info()


def build(
    resource_dir: Path | None = None,
    include_images: bool = True,
    include_page_snapshots: bool = True,
) -> dict:
    """执行完整的文档化、段落切分、向量化、图片证据和 Chroma 入库流程。"""
    source_dir = Path(resource_dir) if resource_dir else find_project_root() / "resource"
    pdf_paths = sorted(source_dir.glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"未找到 PDF 文件：{source_dir}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCUMENT_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks, page_chunk_ids = build_chunk_records(pdf_paths)
    embed_chunks(chunks)
    images = build_image_metadata(pdf_paths, page_chunk_ids, include_page_snapshots) if include_images else []
    if include_images:
        attach_images_to_chunks(chunks, images)
    collection_info = write_chunks(to_document_chunks(chunks))
    return {
        "pdf_count": len(pdf_paths),
        "chunk_count": len(chunks),
        "image_count": len(images),
        "chunks_path": str(CHUNKS_PATH),
        "document_index_dir": str(DOCUMENT_INDEX_DIR),
        "document_collection_name": DOCUMENT_COLLECTION_NAME,
        "document_collection": collection_info,
        "image_metadata_path": str(IMAGE_METADATA_PATH),
        "image_dir": str(IMAGE_DIR),
    }


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="按自然段重建多维个例检索的 Chroma 文档向量库。")
    parser.add_argument("--resource-dir", type=Path, default=None, help="PDF 资源目录，默认读取项目 resource 目录。")
    parser.add_argument("--skip-images", action="store_true", help="只构建文本 chunk 和向量库，不抽取图片证据。")
    parser.add_argument("--no-page-snapshots", action="store_true", help="只抽取 PDF 内嵌图片，不渲染整页快照。")
    args = parser.parse_args()
    result = build(
        resource_dir=args.resource_dir,
        include_images=not args.skip_images,
        include_page_snapshots=not args.no_page_snapshots,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
