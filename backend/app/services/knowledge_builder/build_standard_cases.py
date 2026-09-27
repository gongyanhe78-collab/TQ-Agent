"""从自然段 chunk 构建多维检索使用的精简标准化气象个例。"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any



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

from backend.app.models import DocumentChunk
from backend.app.config import settings
from backend.app.services.image_extraction import ImageEvidenceStore
from backend.app.services.model_client.chat import get_llm_client


DOCUMENT_INDEX_DIR = settings.document_index_dir
IMAGE_METADATA_PATH = settings.image_metadata_path
STANDARD_CASES_PATH = settings.standard_cases_path
CHUNKS_PATH = DOCUMENT_INDEX_DIR / "chunks.json"
CASE_FIELDS = (
    "case_id",
    "title",
    "date_range",
    "disaster_types",
    "affected_areas",
    "source_pdf",
    "source_chunk_ids",
    "evidence_image_ids",
)
DISASTER_TERMS = (
    "强对流",
    "暴雨",
    "大暴雨",
    "强降水",
    "短时强降水",
    "雷暴大风",
    "雷暴",
    "雷雨",
    "大风",
    "冰雹",
    "雨雪",
    "降雪",
    "暴雪",
    "寒潮",
    "低温",
    "霜冻",
    "高温",
    "沙尘",
    "雾",
)
AREA_TERMS = (
    "山西省",
    "全省",
    "太原",
    "大同",
    "朔州",
    "忻州",
    "阳泉",
    "晋中",
    "吕梁",
    "长治",
    "晋城",
    "临汾",
    "运城",
)
DATE_TITLE_PART = (
    r"(?:(?:\d{1,2}\s*月\s*)?\d{1,2}\s*(?:日|号)?"
    r"\s*[-~～—至到]\s*(?:\d{1,2}\s*月\s*)?\d{1,2}\s*(?:日|号)?"
    r"|(?:\d{1,2}\s*月\s*)?\d{1,2}\s*(?:日|号))"
)
CASE_HEADING_RE = re.compile(
    r"(?P<marker>[一二三四五六七八九十]+、)\s*"
    rf"(?P<title>{DATE_TITLE_PART}[^。\n]{{0,50}}?(?:天气过程|过程|天气))"
)
MAJOR_HEADING_RE = re.compile(r"[一二三四五六七八九十]+、[^。\n]{1,90}")
RANGE_DATE_RE = re.compile(
    r"(?:(?P<month>\d{1,2})\s*月\s*)?"
    r"(?P<start>\d{1,2})\s*(?:日|号)?"
    r"\s*[-~～—至到]\s*"
    r"(?:(?P<end_month>\d{1,2})\s*月\s*)?"
    r"(?P<end>\d{1,2})\s*(?:日|号)?"
)
SINGLE_DATE_RE = re.compile(r"(?:(?P<month>\d{1,2})\s*月\s*)?(?P<day>\d{1,2})\s*(?:日|号)")
PDF_MONTH_RE = re.compile(r"(?:FST)?20\d{2}[-_年]?(?P<month>\d{1,2})")
PDF_YEAR_RE = re.compile(r"(?P<year>20\d{2})")


@dataclass
class CaseCandidate:
    """从连续 chunk 中组装出的个例候选。"""

    source_pdf: str
    title: str
    chunk_ids: list[str]
    text: str


@dataclass
class ChunkSpan:
    """记录拼接文本中每个 chunk 的起止位置，用于把个例反关联到 chunk_id。"""

    chunk: DocumentChunk
    start: int
    end: int


def compact_case_dict(data: dict[str, Any]) -> dict[str, Any]:
    """只保留用户要求的八个标准化个例字段。"""
    normalized: dict[str, Any] = {
        "case_id": "",
        "title": "",
        "date_range": "",
        "disaster_types": [],
        "affected_areas": [],
        "source_pdf": "",
        "source_chunk_ids": [],
        "evidence_image_ids": [],
    }
    for key in CASE_FIELDS:
        value = data.get(key, normalized[key])
        if key in {"disaster_types", "affected_areas", "source_chunk_ids", "evidence_image_ids"}:
            normalized[key] = _list_of_strings(value)
        else:
            normalized[key] = str(value or "").strip()
    return {key: normalized[key] for key in CASE_FIELDS}


def load_chunks(path: Path = CHUNKS_PATH) -> list[DocumentChunk]:
    """读取向量化脚本生成的 chunk JSON，并按 chunk_id 去重。"""
    if not path.is_file():
        raise FileNotFoundError(f"未找到 chunk 文件：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[DocumentChunk] = []
    seen_ids: set[str] = set()
    for item in data:
        chunk_id = str(item.get("chunk_id") or "").strip()
        if not chunk_id or chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        chunks.append(
            DocumentChunk(
                source_pdf=str(item.get("source_pdf") or ""),
                chunk_id=chunk_id,
                chunk_no=int(item.get("chunk_no") or 0),
                content=str(item.get("content") or ""),
                file_path=item.get("file_path"),
                embedding=item.get("embedding") or [],
            )
        )
    return chunks


def _clean_title(text: str) -> str:
    """去掉标题里的编号和多余空格，保留真正的天气过程名称。"""
    value = re.sub(r"\s+", "", text or "")
    value = re.sub(r"^[一二三四五六七八九十]+、", "", value)
    value = re.sub(r"(?:1[.．、]?\s*)?(?:天气实况|过程特征|实况特征|环流形势).*$", "", value)
    return value.strip("：:，,。；; ")


def _join_pdf_text(chunks: list[DocumentChunk]) -> tuple[str, list[ChunkSpan]]:
    """把同一 PDF 的 chunk 顺序拼回全文，并保留 chunk 在全文中的位置。"""
    parts: list[str] = []
    spans: list[ChunkSpan] = []
    offset = 0
    for chunk in chunks:
        text = chunk.content or ""
        parts.append(text)
        spans.append(ChunkSpan(chunk=chunk, start=offset, end=offset + len(text)))
        offset += len(text) + 1
    return "\n".join(parts), spans


def _next_major_heading_start(text: str, start: int, default_end: int) -> int:
    """找到当前个例后面的下一个顶层标题，预报服务和小结也会作为结束边界。"""
    for match in MAJOR_HEADING_RE.finditer(text, start + 1):
        return match.start()
    return default_end


def _chunk_ids_for_span(spans: list[ChunkSpan], start: int, end: int) -> list[str]:
    """把个例正文跨度映射回覆盖它的 chunk_id，避免一个个例关联过多无关 chunk。"""
    chunk_ids: list[str] = []
    for span in spans:
        if span.end < start or span.start > end:
            continue
        chunk_ids.append(span.chunk.chunk_id)
    return chunk_ids


def _candidates_from_chunks(chunks: list[DocumentChunk]) -> list[CaseCandidate]:
    """按 PDF 顶层天气过程标题，把 chunk 串成标准化个例候选。"""
    by_pdf: dict[str, list[DocumentChunk]] = defaultdict(list)
    for chunk in chunks:
        by_pdf[chunk.source_pdf].append(chunk)

    candidates: list[CaseCandidate] = []
    for source_pdf, pdf_chunks in sorted(by_pdf.items()):
        ordered = sorted(pdf_chunks, key=lambda item: item.chunk_no)
        pdf_text, spans = _join_pdf_text(ordered)
        case_matches = list(CASE_HEADING_RE.finditer(pdf_text))
        for match in case_matches:
            title = _clean_title(match.group("title"))
            if not title:
                continue
            section_start = match.start()
            section_end = _next_major_heading_start(pdf_text, section_start, len(pdf_text))
            chunk_ids = _chunk_ids_for_span(spans, section_start, section_end)
            if not chunk_ids:
                continue
            candidates.append(
                CaseCandidate(
                    source_pdf=source_pdf,
                    title=title,
                    chunk_ids=chunk_ids,
                    text=pdf_text[section_start:section_end].strip(),
                )
            )
    return candidates


def _month_from_source(source_pdf: str) -> str:
    """从 PDF 文件名中推断月份，用于标题只有日期没有月份的情况。"""
    match = PDF_MONTH_RE.search(source_pdf)
    return match.group("month") if match else ""


def _year_from_source(source_pdf: str) -> str:
    """从 PDF 文件名中推断年份，避免前端或模型按当前年份乱补。"""
    match = PDF_YEAR_RE.search(source_pdf or "")
    return match.group("year") if match else ""


def _with_year(year: str, value: str) -> str:
    """给中文日期范围补上文件名年份；无法识别年份时保持原样。"""
    if not value or not year or re.search(r"20\d{2}\s*年", value):
        return value
    return f"{year}年{value}"


def _extract_date_range(title: str, source_pdf: str) -> str:
    """从标题抽取日期范围，并用 PDF 文件名补齐年份和月份。"""
    year = _year_from_source(source_pdf)
    range_match = RANGE_DATE_RE.search(title or "")
    if range_match:
        month = range_match.group("month") or _month_from_source(source_pdf)
        end_month = range_match.group("end_month") or month
        start = range_match.group("start")
        end = range_match.group("end")
        if month and end_month != month:
            return _with_year(year, f"{month}月{start}日-{end_month}月{end}日")
        if month:
            return _with_year(year, f"{month}月{start}-{end}日")
        return _with_year(year, f"{start}-{end}日")
    single_match = SINGLE_DATE_RE.search(title or "")
    if not single_match:
        return ""
    month = single_match.group("month") or _month_from_source(source_pdf)
    day = single_match.group("day")
    value = f"{month}月{day}日" if month else f"{day}日"
    return _with_year(year, value)

def _extract_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    """按固定词表抽取灾种或地区，并按词表顺序去重输出。"""
    return [term for term in terms if term in text]


def _list_of_strings(value: Any) -> list[str]:
    """把 LLM 或规则产出的字段统一整理成字符串列表。"""
    if isinstance(value, list):
        result = [str(item).strip() for item in value if str(item).strip()]
    elif isinstance(value, str) and value.strip():
        result = [part.strip() for part in re.split(r"[、,，;；]", value) if part.strip()]
    else:
        result = []
    deduped: list[str] = []
    for item in result:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _merge_text_terms(*term_lists: list[str]) -> list[str]:
    """合并多个术语列表并去重，保持原始发现顺序。"""
    merged: list[str] = []
    for terms in term_lists:
        for term in terms or []:
            text = str(term or "").strip()
            if text and text not in merged:
                merged.append(text)
    return merged


def _extract_disaster_types(title: str, text: str) -> list[str]:
    """抽取灾种，并优先补全标题中明确出现的灾种。"""
    title_terms = _extract_terms(title, DISASTER_TERMS)
    body_terms = _extract_terms(text, DISASTER_TERMS)
    return _merge_text_terms(title_terms, body_terms)


def _evidence_image_ids(image_store: ImageEvidenceStore, chunk_ids: list[str]) -> list[str]:
    """根据候选关联的 chunk 查找图片证据，结果去重并保持发现顺序。"""
    seen: set[str] = set()
    image_ids: list[str] = []
    for chunk_id in chunk_ids:
        for image in image_store.list_by_chunk_id(chunk_id, limit=3):
            if image.image_id in seen:
                continue
            seen.add(image.image_id)
            image_ids.append(image.image_id)
    return image_ids


def _rule_standardize(candidate: CaseCandidate, index: int, image_store: ImageEvidenceStore) -> dict[str, Any]:
    """规则方式生成标准化个例，是没有 LLM 或 LLM 失败时的稳定兜底。"""
    text = f"{candidate.title}\n{candidate.text}"
    return compact_case_dict(
        {
            "case_id": f"{Path(candidate.source_pdf).stem}-std-case-{index:03d}",
            "title": candidate.title,
            "date_range": _extract_date_range(candidate.title, candidate.source_pdf),
            "disaster_types": _extract_terms(text, DISASTER_TERMS),
            "affected_areas": _extract_terms(text, AREA_TERMS),
            "source_pdf": candidate.source_pdf,
            "source_chunk_ids": candidate.chunk_ids,
            "evidence_image_ids": _evidence_image_ids(image_store, candidate.chunk_ids),
        }
    )


def _llm_standardize(candidate: CaseCandidate, index: int, image_store: ImageEvidenceStore, llm_client: Any) -> dict[str, Any] | None:
    """可选 LLM 标准化，只要求模型返回本脚本需要的八个字段。"""
    if not llm_client or not llm_client.is_available():
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None

    prompt = {
        "source_pdf": candidate.source_pdf,
        "case_id": f"{Path(candidate.source_pdf).stem}-std-case-{index:03d}",
        "title_hint": candidate.title,
        "source_chunk_ids": candidate.chunk_ids,
        "text": candidate.text[:5000],
    }
    try:
        client = OpenAI(api_key=llm_client.api_key, base_url=llm_client.base_url)
        completion = client.chat.completions.create(
            model=llm_client.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是气象灾害个例标准化抽取助手。只根据给定材料抽取一条个例，"
                        "不要编造材料中没有的信息，输出 JSON。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "请只输出这些字段：case_id、title、date_range、disaster_types、"
                        "affected_areas、source_pdf、source_chunk_ids、evidence_image_ids。\n"
                        + json.dumps(prompt, ensure_ascii=False)
                    ),
                },
            ],
        )
        data = json.loads(completion.choices[0].message.content or "{}")
    except Exception:
        return None

    fallback = _rule_standardize(candidate, index, image_store)
    merged = {**fallback, **data}
    merged["case_id"] = fallback["case_id"]
    # 日期以规则抽取为准，确保年份始终来自 source_pdf，避免 LLM 省略或猜错年份。
    merged["date_range"] = fallback["date_range"]
    merged["source_pdf"] = candidate.source_pdf
    merged["source_chunk_ids"] = candidate.chunk_ids
    merged["evidence_image_ids"] = fallback["evidence_image_ids"]
    # LLM 结果可能漏掉标题中的灾种，这里再次用标题和原文补全，保证前端结构化筛选稳定命中。
    merged["disaster_types"] = _merge_text_terms(
        merged.get("disaster_types") or [],
        _extract_disaster_types(candidate.title, f"{candidate.title}\n{candidate.text}"),
    )
    return compact_case_dict(merged)


def build(use_llm: bool = False, chunks_path: Path = CHUNKS_PATH, output_path: Path = STANDARD_CASES_PATH) -> dict[str, Any]:
    """执行标准化个例抽取，并把精简 JSON 写入模块 data 目录。"""
    chunks = load_chunks(chunks_path)
    image_store = ImageEvidenceStore(IMAGE_METADATA_PATH)
    llm_client = get_llm_client() if use_llm else None
    cases: list[dict[str, Any]] = []
    source_counts: dict[str, int] = defaultdict(int)
    for candidate in _candidates_from_chunks(chunks):
        source_counts[candidate.source_pdf] += 1
        index = source_counts[candidate.source_pdf]
        case = _llm_standardize(candidate, index, image_store, llm_client) if use_llm else None
        cases.append(case or _rule_standardize(candidate, index, image_store))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "chunk_count": len(chunks),
        "case_count": len(cases),
        "standard_cases_path": str(output_path),
        "llm_used": bool(use_llm and llm_client and llm_client.is_available()),
    }


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="从自然段 chunk 抽取精简标准化气象个例。")
    parser.add_argument("--use-llm", action="store_true", help="启用 .env 中配置的大模型辅助抽取。")
    parser.add_argument("--chunks-path", type=Path, default=CHUNKS_PATH, help="chunk JSON 路径，默认读取模块内 document_index/chunks.json。")
    parser.add_argument("--output-path", type=Path, default=STANDARD_CASES_PATH, help="标准化个例 JSON 输出路径。")
    args = parser.parse_args()
    result = build(use_llm=args.use_llm, chunks_path=args.chunks_path, output_path=args.output_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

