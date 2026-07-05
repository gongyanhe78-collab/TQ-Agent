from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, TypedDict


BROWSER_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
IMAGE_TYPE_KEYWORDS = (
    ("sounding", "探空图", ("探空", "TlnP", "TInP", "温湿廓线")),
    ("radar", "雷达图", ("雷达", "回波", "组合反射率", "风雷")),
    ("satellite", "卫星图", ("卫星", "云图", "红外", "可见光", "水汽")),
    ("precipitation", "降水图", ("降水", "雨量", "累计降水", "短时强降水", "降雨")),
    ("wind", "大风图", ("大风", "阵风", "风速", "8级", "极大风")),
    ("synoptic", "环流形势图", ("环流", "形势", "500", "700", "850", "海平面气压", "位势高度")),
    ("temperature", "温度图", ("高温", "气温", "温度", "等温线")),
    ("warning", "预警图", ("预警", "风险", "服务")),
)


class ImagePayload(TypedDict, total=False):
    page_no: int
    image_no: int
    extension: str
    data: bytes
    width: int | None
    height: int | None


@dataclass
class ImageEvidence:
    """PDF 图片证据元数据。"""

    image_id: str
    source_pdf: str
    page_no: int
    image_no: int
    image_path: str
    extraction_type: str
    nearby_text: str = ""
    caption: str = ""
    related_chunk_ids: list[str] = field(default_factory=list)
    width: int | None = None
    height: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class PdfImageEvidenceExtractor:
    """从 PDF 中双轨提取嵌入图片和整页快照，并生成可追溯元数据。"""

    def __init__(
        self,
        output_dir: Path,
        metadata_path: Path | None = None,
        embedded_image_reader: Callable[[Path], Iterable[ImagePayload]] | None = None,
        page_snapshot_renderer: Callable[[Path], Iterable[ImagePayload]] | None = None,
        page_text_reader: Callable[[Path], dict[int, str]] | None = None,
        chunk_lookup: Callable[[str, int], list[str]] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.metadata_path = Path(metadata_path) if metadata_path else self.output_dir.parent / "image_metadata.json"
        self.embedded_image_reader = embedded_image_reader or self._read_embedded_images
        self.page_snapshot_renderer = page_snapshot_renderer or self._render_page_snapshots
        self.page_text_reader = page_text_reader or self._read_page_text
        self.chunk_lookup = chunk_lookup or (lambda source_pdf, page_no: [])

    def build_index(self, pdf_paths: list[Path]) -> list[ImageEvidence]:
        """为 PDF 列表重建图片证据元数据。"""
        records: list[ImageEvidence] = []
        for pdf_path in pdf_paths:
            records.extend(self.extract_pdf(Path(pdf_path)))
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(
            json.dumps([record.to_dict() for record in records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return records

    def extract_pdf(self, pdf_path: Path) -> list[ImageEvidence]:
        """对单个 PDF 执行双轨图片提取。"""
        pdf_path = Path(pdf_path)
        page_text = self.page_text_reader(pdf_path)
        records: list[ImageEvidence] = []
        records.extend(self._records_from_payloads(pdf_path, "embedded", self.embedded_image_reader(pdf_path), page_text))
        records.extend(
            self._records_from_payloads(
                pdf_path,
                "page_snapshot",
                self.page_snapshot_renderer(pdf_path),
                page_text,
            )
        )
        return records

    def _records_from_payloads(
        self,
        pdf_path: Path,
        extraction_type: str,
        payloads: Iterable[ImagePayload],
        page_text: dict[int, str],
    ) -> list[ImageEvidence]:
        records: list[ImageEvidence] = []
        source_pdf = pdf_path.name
        stem = pdf_path.stem
        for fallback_no, payload in enumerate(payloads, start=1):
            page_no = int(payload.get("page_no") or 0)
            image_no = int(payload.get("image_no") or fallback_no)
            extension = self._normalize_extension(payload.get("extension") or ".png")
            if extraction_type == "page_snapshot":
                image_id = f"{stem}-page-{page_no:03d}-snapshot"
            else:
                image_id = f"{stem}-page-{page_no:03d}-image-{image_no:03d}"
            image_path = self.output_dir / stem / f"{image_id}{extension}"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(payload.get("data") or b"")
            nearby_text = self._nearby_text(page_text.get(page_no, ""))
            records.append(
                ImageEvidence(
                    image_id=image_id,
                    source_pdf=source_pdf,
                    page_no=page_no,
                    image_no=image_no,
                    image_path=str(image_path),
                    extraction_type=extraction_type,
                    nearby_text=nearby_text,
                    caption=self._caption_from_text(nearby_text),
                    related_chunk_ids=self.chunk_lookup(source_pdf, page_no),
                    width=payload.get("width"),
                    height=payload.get("height"),
                )
            )
        return records

    def _read_embedded_images(self, pdf_path: Path) -> list[ImagePayload]:
        """用 pypdf 抽取 PDF 内嵌图片。"""
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        payloads: list[ImagePayload] = []
        for page_index, page in enumerate(reader.pages, start=1):
            for image_index, image_file in enumerate(getattr(page, "images", []), start=1):
                name = getattr(image_file, "name", "") or ""
                extension = Path(name).suffix or ".png"
                image = getattr(image_file, "image", None)
                payloads.append(
                    {
                        "page_no": page_index,
                        "image_no": image_index,
                        "extension": extension,
                        "data": getattr(image_file, "data", b""),
                        "width": getattr(image, "width", None),
                        "height": getattr(image, "height", None),
                    }
                )
        return payloads

    def _render_page_snapshots(self, pdf_path: Path) -> list[ImagePayload]:
        """用 pypdfium2 将每页渲染为 PNG 快照。"""
        import io

        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(pdf_path))
        payloads: list[ImagePayload] = []
        for page_index in range(len(document)):
            page = document[page_index]
            bitmap = page.render(scale=2)
            image = bitmap.to_pil()
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            payloads.append(
                {
                    "page_no": page_index + 1,
                    "image_no": 1,
                    "extension": ".png",
                    "data": buffer.getvalue(),
                    "width": image.width,
                    "height": image.height,
                }
            )
        return payloads

    def _read_page_text(self, pdf_path: Path) -> dict[int, str]:
        """用 pypdf 读取每页文本，供图题和附近文本提取。"""
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return {
            page_index: (page.extract_text() or "").strip()
            for page_index, page in enumerate(reader.pages, start=1)
        }

    def _caption_from_text(self, text: str) -> str:
        """从附近文本中提取第一条图题。"""
        for line in text.splitlines():
            value = line.strip()
            if re.match(r"^图\s*\d+", value):
                return value
        return ""

    def _nearby_text(self, text: str, limit: int = 1200) -> str:
        """第 0 期先保存页级附近文本，后续可替换为版面级图文邻近。"""
        return text[:limit]

    def _normalize_extension(self, extension: str) -> str:
        value = extension.lower()
        if not value.startswith("."):
            value = f".{value}"
        return value


class ImageEvidenceStore:
    """读取图片证据元数据，并按 chunk 或 image_id 查询。"""

    def __init__(self, metadata_path: Path):
        self.metadata_path = Path(metadata_path)

    def list_by_chunk_id(self, chunk_id: str, limit: int = 6) -> list[ImageEvidence]:
        """返回与指定 chunk 关联的图片证据，优先返回整页快照和带图题图片。"""
        records = [
            record
            for record in self._load_records()
            if chunk_id in record.related_chunk_ids
            and self._is_browser_displayable(record)
        ]
        records.sort(key=self._sort_key)
        return self._dedupe(records)[:limit]

    def get_image(self, image_id: str) -> ImageEvidence | None:
        """按 image_id 返回单条图片证据。"""
        for record in self._load_records():
            if record.image_id == image_id:
                return record
        return None

    def classify(self, record: ImageEvidence) -> tuple[str, str]:
        """根据图注和附近文本给图片打轻量类型标签。"""
        text = record.caption.strip() or record.nearby_text
        for image_type, data_category, keywords in IMAGE_TYPE_KEYWORDS:
            if any(keyword in text for keyword in keywords):
                return image_type, data_category
        if record.extraction_type == "page_snapshot":
            return "page_snapshot", "整页快照"
        return "figure", "图片"

    def list_by_image_ids(self, image_ids: list[str]) -> list[ImageEvidence]:
        """按 image_id 列表返回可展示图片证据，并保持入参顺序。"""
        wanted = set(image_ids)
        records = {
            record.image_id: record
            for record in self._load_records()
            if record.image_id in wanted and self._is_browser_displayable(record)
        }
        return [records[image_id] for image_id in image_ids if image_id in records]

    def _load_records(self) -> list[ImageEvidence]:
        if not self.metadata_path.exists():
            return []
        data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return [ImageEvidence(**item) for item in data]

    def _is_browser_displayable(self, record: ImageEvidence) -> bool:
        return Path(record.image_path).suffix.lower() in BROWSER_IMAGE_EXTENSIONS

    def _sort_key(self, record: ImageEvidence) -> tuple:
        return (
            record.extraction_type != "page_snapshot",
            not bool(record.caption),
            record.page_no,
            record.image_no,
        )

    def _dedupe(self, records: list[ImageEvidence]) -> list[ImageEvidence]:
        selected: list[ImageEvidence] = []
        seen: set[tuple] = set()
        for record in records:
            caption_key = re.sub(r"\s+", "", record.caption or "")
            if record.extraction_type == "page_snapshot":
                key = (record.source_pdf, record.page_no, "page_snapshot")
            elif caption_key:
                key = (record.source_pdf, record.page_no, caption_key)
            else:
                key = (record.source_pdf, record.page_no, record.image_no)
            if key in seen:
                continue
            seen.add(key)
            selected.append(record)
        return selected
