"""公司统一知识库到多维个例检索数据接口的适配层。"""
from __future__ import annotations

import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from typing import Any

from backend.app.models import DocumentChunk, StandardCase
from backend.app.services.image_extraction import BROWSER_IMAGE_EXTENSIONS, IMAGE_TYPE_KEYWORDS, ImageEvidence

from backend.app.services.agent.case_multidim_search.integrations.company_kb_client import CompanyKbClient


SHANXI_CITIES = ("太原", "大同", "朔州", "忻州", "阳泉", "晋中", "吕梁", "长治", "晋城", "临汾", "运城")
# 不同检索请求共享有界网络线程池，避免每次创建 Agent 都遗留一组 chunk 读取线程。
_SHARED_CHUNK_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="case-chunk")


class CompanyStandardCaseStore:
    """通过公司接口读取标准化个例，供本地结构化筛选继续使用。"""

    def __init__(self, client: CompanyKbClient, cache_ttl: float = 300):
        """保存 HTTP 客户端和轻量缓存配置。"""
        self.client = client
        self.cache_ttl = float(cache_ttl)
        self._cases_cache: list[StandardCase] | None = None
        self._cases_loaded_at = 0.0

    def list_cases(self) -> list[StandardCase]:
        """返回全部标准化个例；多选 OR 等业务筛选仍由当前模块本地完成。"""
        now = time.monotonic()
        if self._cases_cache is not None and now - self._cases_loaded_at < self.cache_ttl:
            return list(self._cases_cache)
        raw_cases = self.client.list_standard_cases()
        cases = [StandardCase.from_dict(self._normalize_case(item)) for item in raw_cases]
        self._cases_cache = cases
        self._cases_loaded_at = now
        return list(cases)

    def get_case(self, case_id: str) -> StandardCase | None:
        """按 case_id 读取单个标准化个例。"""
        for case in self.list_cases():
            if case.case_id == case_id:
                return case
        return None

    def search_cases(
        self,
        *,
        date: str | None = None,
        disaster_type: str | None = None,
        area: str | None = None,
        source_pdf: str | None = None,
    ) -> list[StandardCase]:
        """兼容旧 store 的轻量查询接口，主流程仍使用 StructuredCaseRetriever。"""
        cases = self.list_cases()
        if date:
            cases = [case for case in cases if date in case.date_range or date in case.title]
        if disaster_type:
            cases = [
                case for case in cases
                if disaster_type in case.title
                or disaster_type in case.summary
                or any(disaster_type in item for item in case.disaster_types)
            ]
        if area:
            cases = [
                case for case in cases
                if area in case.summary
                or area in case.weather_facts
                or any(area in item for item in case.affected_areas + case.city_tags)
            ]
        if source_pdf:
            cases = [case for case in cases if case.source_pdf == source_pdf]
        return cases

    def _normalize_case(self, item: dict[str, Any]) -> dict[str, Any]:
        """补齐当前运行时依赖的老字段，避免远程标准个例字段较少导致筛选失败。"""
        data = dict(item or {})
        data["case_id"] = str(data.get("case_id") or data.get("id") or self._fallback_case_id(data))
        data["title"] = str(data.get("title") or data.get("case_title") or data["case_id"])
        data["date_range"] = str(data.get("date_range") or data.get("time_range") or "")
        data["disaster_types"] = self._list_value(data.get("disaster_types") or data.get("disaster_type"))
        data["affected_areas"] = self._list_value(data.get("affected_areas") or data.get("affected_area"))
        data["source_pdf"] = str(data.get("source_pdf") or data.get("source_file") or "")
        data["source_chunk_ids"] = self._list_value(data.get("source_chunk_ids") or data.get("chunk_ids"))
        data["evidence_image_ids"] = self._list_value(data.get("evidence_image_ids") or data.get("image_ids"))
        data["summary"] = str(data.get("summary") or "")
        data["weather_facts"] = str(data.get("weather_facts") or "")
        data["forecast_focus"] = str(data.get("forecast_focus") or "")
        data["city_tags"] = self._list_value(data.get("city_tags")) or self._infer_city_tags(data)
        data["months"] = self._months(data)
        data["year"] = self._year(data)
        data["start_date"] = str(data.get("start_date") or "")
        data["end_date"] = str(data.get("end_date") or "")
        return data

    def _fallback_case_id(self, data: dict[str, Any]) -> str:
        """远程缺少 case_id 时用来源文件和标题生成稳定兜底 ID。"""
        text = f"{data.get('source_pdf') or data.get('source_file') or 'case'}-{data.get('title') or data.get('case_title') or ''}"
        return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "-", text).strip("-") or "remote-case"

    def _list_value(self, value: Any) -> list[str]:
        """把接口中的字符串、数组或空值统一转成字符串列表。"""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, tuple | set):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        if not text:
            return []
        return [item for item in re.split(r"[、,，;；|\s]+", text) if item]

    def _infer_city_tags(self, data: dict[str, Any]) -> list[str]:
        """从标题、日期、摘要和影响区域中回填山西地市标签。"""
        text = " ".join(
            [
                str(data.get("title") or ""),
                str(data.get("date_range") or ""),
                str(data.get("summary") or ""),
                " ".join(self._list_value(data.get("affected_areas"))),
            ]
        )
        return [city for city in SHANXI_CITIES if city in text]

    def _months(self, data: dict[str, Any]) -> list[int]:
        """从远程 month/months 字段和文本中提取月份。"""
        values: list[int] = []
        for raw in self._list_value(data.get("months") or data.get("month")):
            if raw.isdigit() and 1 <= int(raw) <= 12:
                values.append(int(raw))
        for text in (data.get("start_date"), data.get("end_date"), data.get("date_range"), data.get("title"), data.get("source_pdf")):
            for value in re.findall(r"(?<!\d)(\d{1,2})\s*月", str(text or "")):
                month = int(value)
                if 1 <= month <= 12:
                    values.append(month)
        return list(dict.fromkeys(values))

    def _year(self, data: dict[str, Any]) -> int | None:
        """从远程 year 字段或文本中提取年份。"""
        year = data.get("year")
        if str(year or "").isdigit():
            return int(year)
        for text in (data.get("start_date"), data.get("end_date"), data.get("date_range"), data.get("source_pdf")):
            match = re.search(r"20\d{2}", str(text or ""))
            if match:
                return int(match.group(0))
        return None


class CompanyDocumentChunkStore:
    """通过公司接口按 chunk_id 精确读取个例关联正文。"""

    def __init__(self, client: CompanyKbClient):
        """保存 HTTP 客户端、共享缓存和有界 chunk 请求池。"""
        self.client = client
        self._chunk_cache: dict[str, DocumentChunk] = {}
        self._missing_chunks: set[str] = set()
        self._chunk_lock = RLock()
        self._chunk_inflight: dict[str, Future] = {}

    def get_chunks(self, chunk_ids: list[str]) -> list[DocumentChunk]:
        """并发读取 chunk，并按个例原始顺序返回，缺失片段自动跳过。"""
        unique_ids = list(dict.fromkeys(str(item or "").strip() for item in chunk_ids or [] if str(item or "").strip()))
        futures = [self._shared_chunk_future(chunk_id) for chunk_id in unique_ids]
        chunks: list[DocumentChunk] = []
        for chunk, chunk_id in zip((future.result() for future in futures), unique_ids):
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """按 chunk_id 读取单个原文片段，并复用正在执行的共享请求。"""
        chunk_id = str(chunk_id or "").strip()
        if not chunk_id:
            return None
        return self._shared_chunk_future(chunk_id).result()

    def _shared_chunk_future(self, chunk_id: str) -> Future:
        """返回缓存或共享中的 chunk Future，避免并发重复请求同一正文。"""
        with self._chunk_lock:
            cached = self._chunk_cache.get(chunk_id)
            if cached is not None:
                future: Future = Future()
                future.set_result(cached)
                return future
            if chunk_id in self._missing_chunks:
                future = Future()
                future.set_result(None)
                return future
            pending = self._chunk_inflight.get(chunk_id)
            if pending is not None:
                return pending
            pending = _SHARED_CHUNK_EXECUTOR.submit(self._fetch_chunk, chunk_id)
            self._chunk_inflight[chunk_id] = pending
            pending.add_done_callback(lambda done, value=chunk_id: self._finish_chunk(value, done))
            return pending

    def _fetch_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """执行一次远程 chunk 请求并写入共享缓存。"""
        raw = self.client.get_chunk(chunk_id)
        if raw is None:
            return None
        return self._to_document_chunk(chunk_id, raw)

    def _finish_chunk(self, chunk_id: str, future: Future) -> None:
        """把共享请求结果落入缓存，后续个例直接复用。"""
        try:
            chunk = future.result()
        except Exception:
            chunk = None
        with self._chunk_lock:
            self._chunk_inflight.pop(chunk_id, None)
            if chunk is None:
                self._missing_chunks.add(chunk_id)
            else:
                self._chunk_cache[chunk.chunk_id] = chunk

    def collection_info(self) -> dict[str, Any]:
        """返回远程知识库 chunk 统计信息，兼容 agent.health。"""
        status = self.client.status()
        count = status.get("total_chunks") or status.get("chroma_count") or status.get("chunk_count") or 0
        return {"count": count, "source": "company_api", "status": status}

    def _to_document_chunk(self, requested_id: str, raw: dict[str, Any]) -> DocumentChunk:
        """把公司接口 chunk JSON 转成当前运行时 DocumentChunk。"""
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        chunk_id = str(raw.get("chunk_id") or raw.get("id") or requested_id)
        chunk_no = raw.get("chunk_no") or raw.get("chunk_index") or metadata.get("chunk_no") or metadata.get("chunk_index") or 0
        try:
            chunk_no = int(chunk_no)
        except (TypeError, ValueError):
            chunk_no = 0
        return DocumentChunk(
            source_pdf=str(raw.get("source_pdf") or raw.get("source_file") or metadata.get("source_pdf") or metadata.get("source_file") or ""),
            chunk_id=chunk_id,
            chunk_no=chunk_no,
            content=str(raw.get("text") or raw.get("content") or raw.get("document") or ""),
            file_path=str(metadata.get("file_path") or raw.get("file_path") or "") or None,
            embedding=None,
        )


class CompanyImageEvidenceStore:
    """通过公司接口读取图片元数据，并把远程图片按需缓存为本地运行时文件。"""

    def __init__(self, client: CompanyKbClient, cache_dir: Path, cache_ttl: float = 300):
        """保存 HTTP 客户端、图片缓存目录和元数据缓存配置。"""
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.cache_ttl = float(cache_ttl)
        self.metadata_path = self.cache_dir / "remote_image_metadata.cache"
        self._records_cache: list[ImageEvidence] | None = None
        self._records_loaded_at = 0.0
        self._records_lock = RLock()

    def list_by_chunk_id(self, chunk_id: str, limit: int = 6) -> list[ImageEvidence]:
        """返回与指定 chunk 关联的图片证据。"""
        records = [
            record
            for record in self._load_records()
            if chunk_id in record.related_chunk_ids and self._is_browser_displayable(record)
        ]
        records.sort(key=self._sort_key)
        return self._dedupe(records)[:limit]

    def get_image(self, image_id: str) -> ImageEvidence | None:
        """按 image_id 返回单条图片证据。"""
        for record in self._load_records():
            if record.image_id == image_id:
                return record
        return None

    def list_by_image_ids(self, image_ids: list[str]) -> list[ImageEvidence]:
        """按 image_id 列表返回图片证据，并保持入参顺序。"""
        wanted = [str(image_id).strip() for image_id in image_ids or [] if str(image_id).strip()]
        if not wanted:
            return []
        wanted_set = set(wanted)
        records = {
            record.image_id: record
            for record in self._load_records()
            if record.image_id in wanted_set and self._is_browser_displayable(record)
        }
        return [records[image_id] for image_id in wanted if image_id in records]

    def classify(self, record: ImageEvidence) -> tuple[str, str]:
        """复用本地图片分类关键词，保持报告原逻辑不变。"""
        text = record.caption.strip() or record.nearby_text
        for image_type, data_category, keywords in IMAGE_TYPE_KEYWORDS:
            if any(keyword in text for keyword in keywords):
                return image_type, data_category
        if record.extraction_type == "page_snapshot":
            return "page_snapshot", "整页快照"
        return "figure", "图片"

    def metadata_available(self) -> bool:
        """判断公司接口是否返回了可用图片元数据。"""
        return bool(self._load_records())

    def resolve_image_path(self, record: ImageEvidence) -> Path | None:
        """按需下载单张远程图片到运行时缓存，避免启动时下载整个图像库。"""
        raw_path = str(record.image_path or "").strip()
        if not raw_path:
            return None
        local_path = Path(raw_path)
        if local_path.is_file():
            return local_path
        if not re.match(r"^https?://", raw_path, flags=re.IGNORECASE):
            return None
        try:
            return self.client.download_asset(raw_path, self.cache_dir / "images", filename_hint=record.image_id)
        except Exception:
            return None

    def _load_records(self) -> list[ImageEvidence]:
        """读取并缓存远程图片元数据。"""
        now = time.monotonic()
        with self._records_lock:
            if self._records_cache is not None and now - self._records_loaded_at < self.cache_ttl:
                return list(self._records_cache)
            raw_images = self.client.list_image_metadata()
            records = [self._to_image_evidence(item) for item in raw_images]
            self._records_cache = records
            self._records_loaded_at = now
            return list(records)

    def _to_image_evidence(self, item: dict[str, Any]) -> ImageEvidence:
        """把公司接口图片 JSON 转成当前运行时 ImageEvidence。"""
        image_id = str(item.get("image_id") or item.get("id") or self._fallback_image_id(item))
        image_url = str(item.get("image_url") or item.get("url") or item.get("image_path") or "")
        remote_url = self.client.absolute_url(image_url)
        page_no = self._int_value(item.get("page_no") or item.get("page_number"))
        image_no = self._int_value(item.get("image_no") or item.get("figure_no") or 0)
        return ImageEvidence(
            image_id=image_id,
            source_pdf=str(item.get("source_pdf") or item.get("source_file") or ""),
            page_no=page_no,
            image_no=image_no,
            image_path=remote_url,
            extraction_type=str(item.get("extraction_type") or item.get("type") or "remote_figure"),
            nearby_text=str(item.get("nearby_text") or item.get("context") or ""),
            caption=str(item.get("caption") or item.get("title") or ""),
            related_chunk_ids=self._list_value(item.get("related_chunk_ids") or item.get("chunk_ids")),
            width=self._optional_int(item.get("width")),
            height=self._optional_int(item.get("height")),
        )

    def _fallback_image_id(self, item: dict[str, Any]) -> str:
        """远程缺少 image_id 时生成稳定兜底 ID。"""
        text = f"{item.get('source_pdf') or item.get('source_file') or 'image'}-{item.get('page_no') or item.get('page_number') or 0}-{item.get('caption') or item.get('title') or ''}"
        return self._safe_name(text)

    def _list_value(self, value: Any) -> list[str]:
        """把接口中的字符串、数组或空值统一转成字符串列表。"""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, tuple | set):
            return [str(item).strip() for item in value if str(item).strip()]
        return [item for item in re.split(r"[、,，;；|\s]+", str(value).strip()) if item]

    def _is_browser_displayable(self, record: ImageEvidence) -> bool:
        """判断图片后缀是否适合浏览器和 PDF 读取。"""
        return Path(str(record.image_path).split("?", 1)[0]).suffix.lower() in BROWSER_IMAGE_EXTENSIONS

    def _sort_key(self, record: ImageEvidence) -> tuple:
        """保持原图片排序倾向：有图题、非快照、页码靠前优先。"""
        return (
            record.extraction_type == "page_snapshot",
            not bool(record.caption),
            record.page_no,
            record.image_no,
        )

    def _dedupe(self, records: list[ImageEvidence]) -> list[ImageEvidence]:
        """按图注和页码去重，避免同一远程图片重复展示。"""
        selected: list[ImageEvidence] = []
        seen: set[tuple] = set()
        for record in records:
            caption_key = re.sub(r"\s+", "", record.caption or "")
            key = (record.source_pdf, record.page_no, caption_key or record.image_id)
            if key in seen:
                continue
            seen.add(key)
            selected.append(record)
        return selected

    def _int_value(self, value: Any) -> int:
        """安全转换整数，失败时返回 0。"""
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _optional_int(self, value: Any) -> int | None:
        """安全转换可空整数。"""
        try:
            return int(value) if value not in {None, ""} else None
        except (TypeError, ValueError):
            return None

    def _safe_name(self, value: str) -> str:
        """生成安全文件名片段。"""
        return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_.-]+", "_", str(value or "remote-image")).strip("._-") or "remote-image"
