"""从主库自然段索引重建经过源 PDF 校验的精简标准个例。"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader


def _find_project_root() -> Path:
    """向上查找同时包含 data、resource 和 backend 的项目根目录。"""
    for parent in Path(__file__).resolve().parents:
        if all((parent / name).is_dir() for name in ("data", "resource", "backend")):
            return parent
    raise RuntimeError("未找到项目根目录")


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CHUNKS_PATH = DATA_DIR / "document_index" / "chunks.json"
DEFAULT_IMAGE_METADATA_PATH = DATA_DIR / "image_metadata.json"
DEFAULT_RESOURCE_DIR = PROJECT_ROOT / "resource"
DEFAULT_OUTPUT_PATH = DATA_DIR / "standard_cases1.json"
PROTECTED_OUTPUT_PATH = DATA_DIR / "standard_cases.json"

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

# 顺序与 Smart Agent 的标准灾种枚举保持一致，便于后续结构化筛选。
DISASTER_TYPES = (
    "强对流",
    "暴雨",
    "大暴雨",
    "强降水",
    "短时强降水",
    "雷暴大风",
    "大风",
    "冰雹",
    "雨雪",
    "降雪",
    "暴雪",
    "寒潮",
    "低温",
    "高温",
    "沙尘",
    "霜冻",
    "雾",
)
SHANXI_CITIES = (
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
AREA_NAMES = (
    "全省",
    "山西北部",
    "山西中部",
    "山西南部",
    "晋北",
    "晋南",
    "东北部",
    "西北部",
    "东南部",
    "西南部",
    "北部",
    "中部",
    "南部",
    *SHANXI_CITIES,
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
    r"(?P<start>\d{1,2})\s*(?:日|号)?\s*[-~～—至到]\s*"
    r"(?:(?P<end_month>\d{1,2})\s*月\s*)?(?P<end>\d{1,2})\s*(?:日|号)?"
)
SINGLE_DATE_RE = re.compile(
    r"(?:(?P<month>\d{1,2})\s*月\s*)?(?P<day>\d{1,2})\s*(?:日|号)"
)
PDF_MONTH_RE = re.compile(r"(?:FST)?20\d{2}[-_年]?(?P<month>\d{1,2})")
PDF_YEAR_RE = re.compile(r"(?P<year>20\d{2})")

# 只取实况段；环流分析、模式检验、预报服务和月度小结不能参与灾种标注。
OBSERVATION_END_PATTERNS = (
    re.compile(
        r"(?<!\d)2\s*[.．、]?\s*(?:环流|影响系统|天气成因|成因分析|模式检验|数值预报|预报及服务|预报服务)"
    ),
    re.compile(r"(?:预报服务情况|服务情况|总结与思考|月度小结|(?<!过程)小结)"),
)


@dataclass(frozen=True)
class Chunk:
    """建库所需的精简 chunk 结构。"""

    source_pdf: str
    chunk_id: str
    chunk_no: int
    content: str


@dataclass(frozen=True)
class ChunkSpan:
    """记录 chunk 在同一 PDF 拼接文本中的字符范围。"""

    chunk: Chunk
    start: int
    end: int


@dataclass(frozen=True)
class CaseCandidate:
    """从一个 PDF 顶层天气过程标题切出的候选个例。"""

    source_pdf: str
    title: str
    chunk_ids: list[str]
    text: str


def _read_json(path: Path) -> Any:
    """以 UTF-8 读取 JSON，并给出明确的缺文件错误。"""
    if not path.is_file():
        raise FileNotFoundError(f"未找到输入文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _dedupe(values: list[str]) -> list[str]:
    """在保持原顺序的前提下去除空值和重复值。"""
    result: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def load_chunks(path: Path) -> list[Chunk]:
    """读取 chunk 索引，拒绝重复或缺失的 chunk_id。"""
    records = _read_json(path)
    if not isinstance(records, list):
        raise ValueError(f"chunk 文件顶层必须是数组：{path}")
    chunks: list[Chunk] = []
    seen_ids: set[str] = set()
    for record in records:
        chunk_id = str(record.get("chunk_id") or "").strip()
        source_pdf = str(record.get("source_pdf") or "").strip()
        if not chunk_id or not source_pdf:
            raise ValueError("chunk 记录缺少 chunk_id 或 source_pdf")
        if chunk_id in seen_ids:
            raise ValueError(f"发现重复 chunk_id：{chunk_id}")
        seen_ids.add(chunk_id)
        chunks.append(
            Chunk(
                source_pdf=source_pdf,
                chunk_id=chunk_id,
                chunk_no=int(record.get("chunk_no") or 0),
                content=str(record.get("content") or ""),
            )
        )
    return chunks


def _clean_title(text: str) -> str:
    """清理标题编号和空格，不改写 PDF 中的过程名称。"""
    value = re.sub(r"\s+", "", text or "")
    value = re.sub(r"^[一二三四五六七八九十]+、", "", value)
    value = re.sub(
        r"(?:1[.．、]?\s*)?(?:天气实况|过程特征|实况特征|环流形势).*$",
        "",
        value,
    )
    return value.strip("：:，,。；; ")


def _join_pdf_chunks(chunks: list[Chunk]) -> tuple[str, list[ChunkSpan]]:
    """按 chunk_no 拼回 PDF 文本，并保留位置映射。"""
    parts: list[str] = []
    spans: list[ChunkSpan] = []
    offset = 0
    for chunk in sorted(chunks, key=lambda item: item.chunk_no):
        parts.append(chunk.content)
        spans.append(ChunkSpan(chunk=chunk, start=offset, end=offset + len(chunk.content)))
        offset += len(chunk.content) + 1
    return "\n".join(parts), spans


def _chunk_ids_for_range(spans: list[ChunkSpan], start: int, end: int) -> list[str]:
    """返回与个例字符范围实际相交的 chunk_id。"""
    return [
        span.chunk.chunk_id
        for span in spans
        if span.start < end and span.end > start
    ]


def _case_candidates(chunks: list[Chunk]) -> list[CaseCandidate]:
    """用顶层天气过程标题切分个例，并在下一个顶层标题处立即结束。"""
    chunks_by_pdf: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_pdf[chunk.source_pdf].append(chunk)

    candidates: list[CaseCandidate] = []
    for source_pdf, pdf_chunks in sorted(chunks_by_pdf.items()):
        pdf_text, spans = _join_pdf_chunks(pdf_chunks)
        for heading in CASE_HEADING_RE.finditer(pdf_text):
            title = _clean_title(heading.group("title"))
            if not title:
                continue
            next_heading = MAJOR_HEADING_RE.search(pdf_text, heading.start() + 1)
            end = next_heading.start() if next_heading else len(pdf_text)
            chunk_ids = _chunk_ids_for_range(spans, heading.start(), end)
            if chunk_ids:
                candidates.append(
                    CaseCandidate(
                        source_pdf=source_pdf,
                        title=title,
                        chunk_ids=chunk_ids,
                        text=pdf_text[heading.start():end].strip(),
                    )
                )
    return candidates


def _observation_text(candidate: CaseCandidate) -> str:
    """截取当前个例的实况部分，防止预报表和月度总结污染标签。"""
    end = len(candidate.text)
    for pattern in OBSERVATION_END_PATTERNS:
        match = pattern.search(candidate.text)
        if match:
            end = min(end, match.start())
    return candidate.text[:end]


def _contains(text: str, term: str) -> bool:
    """判断材料是否直接出现指定业务术语。"""
    return term in text


def _is_observed_high_temperature(title: str, observation: str) -> bool:
    """高温必须有标题证据，或同时具备实况表述与 35℃以上阈值证据。"""
    if "高温" in title:
        return True
    has_observed_event = bool(
        re.search(r"(?:出现|达到|发生|持续|大范围)[^。；]{0,30}高温天气", observation)
        or re.search(r"高温天气[^。；]{0,30}(?:出现|达到|站)", observation)
    )
    has_threshold = bool(
        re.search(r"(?:最高气温[^。；]{0,50})?(?:3[5-9]|4\d)(?:\.\d+)?\s*(?:℃|摄氏度)", observation)
    )
    return has_observed_event and has_threshold


def _extract_disaster_types(candidate: CaseCandidate) -> list[str]:
    """依据标题和实况事实生成灾种，禁止从整篇过程报告无条件抓关键词。"""
    title = candidate.title
    observation = _observation_text(candidate)
    selected: set[str] = set()

    # 标题是过程归类的最高优先级证据。
    for disaster in DISASTER_TYPES:
        if disaster != "高温" and _contains(title, disaster):
            selected.add(disaster)

    # 这些术语只有在实况段直接出现时才作为辅助灾种加入。
    for disaster in (
        "暴雨",
        "大暴雨",
        "短时强降水",
        "雷暴大风",
        "大风",
        "冰雹",
        "雨雪",
        "降雪",
        "暴雪",
        "寒潮",
        "沙尘",
        "霜冻",
        "雾",
    ):
        if _contains(observation, disaster):
            selected.add(disaster)

    # 图题中的“强对流监测”只是产品名称，不能单独证明当前过程属于强对流。
    if "强对流" in title or re.search(
        r"(?:出现|发生|伴随)[^。；]{0,50}强对流(?:天气|过程)", observation
    ):
        selected.add("强对流")
    # 排除“最强降水时段”中的字面重叠，只接受独立的“强降水”术语。
    if re.search(r"(?<!最)强降水", observation):
        selected.add("强降水")

    # “低温”容易被普通气温描述误触发，只接受明确的低温天气或低温持续事实。
    if re.search(r"低温(?:天气|过程|持续|导致|影响)", f"{title}\n{observation}"):
        selected.add("低温")
    if _is_observed_high_temperature(title, observation):
        selected.add("高温")

    # 补齐灾种的业务包含关系，使上位灾种查询能稳定命中真实个例。
    if selected & {"短时强降水", "雷暴大风", "冰雹"}:
        selected.add("强对流")
    if "雷暴大风" in selected:
        selected.add("大风")
    if selected & {"暴雨", "大暴雨", "短时强降水"}:
        selected.add("强降水")
    if "大暴雨" in selected:
        selected.add("暴雨")
    if selected & {"雨雪", "暴雪"}:
        selected.add("降雪")
    if "暴雪" in selected:
        selected.add("雨雪")

    return [disaster for disaster in DISASTER_TYPES if disaster in selected]


def _extract_affected_areas(candidate: CaseCandidate) -> list[str]:
    """只从实况段抽取影响区域，不读取预警服务表中的无关地区。"""
    observation = _observation_text(candidate)
    areas = [area for area in AREA_NAMES if area in observation]
    # “全省范围/全省各地”已覆盖全部地市，保留一个总括标签即可减少冗余。
    if re.search(r"全省(?:范围|各地|均|所有)", observation):
        return ["全省"]
    areas = _dedupe(areas)
    # 原文只写“山西大部分地区”时保留省级范围，不据此虚构具体地市。
    if not areas and re.search(r"(?:山西|我省)", observation):
        return ["山西省"]
    return areas


def _extract_date_range(title: str, source_pdf: str) -> str:
    """从原始标题抽取日期，并仅使用源文件名补齐年份和月份。"""
    year_match = PDF_YEAR_RE.search(source_pdf)
    year = year_match.group("year") if year_match else ""
    month_match = PDF_MONTH_RE.search(source_pdf)
    source_month = month_match.group("month") if month_match else ""
    range_match = RANGE_DATE_RE.search(title)
    if range_match:
        month = range_match.group("month") or source_month
        end_month = range_match.group("end_month") or month
        start = range_match.group("start")
        end = range_match.group("end")
        if month and end_month != month:
            value = f"{month}月{start}日-{end_month}月{end}日"
        elif month:
            value = f"{month}月{start}-{end}日"
        else:
            value = f"{start}-{end}日"
        return f"{year}年{value}" if year else value
    single_match = SINGLE_DATE_RE.search(title)
    if not single_match:
        return ""
    month = single_match.group("month") or source_month
    day = single_match.group("day")
    value = f"{month}月{day}日" if month else f"{day}日"
    return f"{year}年{value}" if year else value


def _image_ids_by_chunk(path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """把图片元数据反向映射到 chunk，并记录每张图片的源 PDF。"""
    records = _read_json(path)
    if not isinstance(records, list):
        raise ValueError(f"图片元数据顶层必须是数组：{path}")
    result: dict[str, list[str]] = defaultdict(list)
    image_sources: dict[str, str] = {}
    for record in records:
        image_id = str(record.get("image_id") or "").strip()
        if not image_id:
            continue
        image_sources[image_id] = str(record.get("source_pdf") or "").strip()
        for chunk_id in record.get("related_chunk_ids") or []:
            result[str(chunk_id)].append(image_id)
    return result, image_sources


def _candidate_image_ids(candidate: CaseCandidate, images_by_chunk: dict[str, list[str]]) -> list[str]:
    """收集当前个例关联 chunk 的图片，保持图片元数据中的原始顺序。"""
    image_ids: list[str] = []
    for chunk_id in candidate.chunk_ids:
        image_ids.extend(images_by_chunk.get(chunk_id, []))
    return _dedupe(image_ids)


def _normalize_for_pdf_match(text: str) -> str:
    """统一空白和日期连接符，仅用于核对标题是否真实存在于 PDF。"""
    return re.sub(r"[\s~～—－-]+", "", text or "")


def _read_pdf_text(path: Path) -> str:
    """逐页抽取 PDF 文本供写入前核验，不把抽取文本写入项目目录。"""
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _validate_cases(
    cases: list[dict[str, Any]],
    chunks: list[Chunk],
    image_sources: dict[str, str],
    resource_dir: Path,
) -> dict[str, int]:
    """在落盘前验证结构、引用、源 PDF 标题和高温标签门禁。"""
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    case_ids: set[str] = set()
    source_counts: dict[str, int] = defaultdict(int)
    pdf_text_cache: dict[str, str] = {}

    for case in cases:
        if tuple(case.keys()) != CASE_FIELDS:
            raise ValueError(f"个例字段不符合约定：{case.get('case_id', '<unknown>')}")
        case_id = case["case_id"]
        if not case_id or case_id in case_ids:
            raise ValueError(f"case_id 缺失或重复：{case_id}")
        case_ids.add(case_id)
        source_counts[case["source_pdf"]] += 1

        missing_chunks = set(case["source_chunk_ids"]) - chunk_ids
        if missing_chunks:
            raise ValueError(f"{case_id} 引用了不存在的 chunk：{sorted(missing_chunks)}")
        missing_images = set(case["evidence_image_ids"]) - set(image_sources)
        if missing_images:
            raise ValueError(f"{case_id} 引用了不存在的图片：{sorted(missing_images)}")
        wrong_source_images = [
            image_id
            for image_id in case["evidence_image_ids"]
            if image_sources[image_id] != case["source_pdf"]
        ]
        if wrong_source_images:
            raise ValueError(f"{case_id} 引用了其他 PDF 的图片：{wrong_source_images}")

        pdf_path = resource_dir / case["source_pdf"]
        if not pdf_path.is_file():
            raise FileNotFoundError(f"{case_id} 的源 PDF 不存在：{pdf_path}")
        if case["source_pdf"] not in pdf_text_cache:
            pdf_text_cache[case["source_pdf"]] = _read_pdf_text(pdf_path)
        normalized_pdf = _normalize_for_pdf_match(pdf_text_cache[case["source_pdf"]])
        normalized_title = _normalize_for_pdf_match(case["title"])
        if normalized_title not in normalized_pdf:
            raise ValueError(f"{case_id} 的标题未在源 PDF 中找到：{case['title']}")

        if "高温" in case["disaster_types"] and "高温" not in case["title"]:
            raise ValueError(f"{case_id} 的高温标签缺少标题级证据")

    # 逐个 PDF 比较顶层过程标题集合，既防止凭空生成，也防止漏掉源文中的过程。
    for source_pdf, expected_count in source_counts.items():
        pdf_text = pdf_text_cache[source_pdf]
        pdf_titles = {
            _normalize_for_pdf_match(_clean_title(match.group("title")))
            for match in CASE_HEADING_RE.finditer(pdf_text)
        }
        case_titles = {
            _normalize_for_pdf_match(case["title"])
            for case in cases
            if case["source_pdf"] == source_pdf
        }
        if pdf_titles != case_titles or len(case_titles) != expected_count:
            raise ValueError(
                f"{source_pdf} 的过程标题与源 PDF 不一致："
                f"PDF={sorted(pdf_titles)}，JSON={sorted(case_titles)}"
            )

    return {
        "case_count": len(cases),
        "source_pdf_count": len(source_counts),
        "validated_chunk_count": len(chunk_ids),
        "validated_image_count": len(image_sources),
    }


def build(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    image_metadata_path: Path = DEFAULT_IMAGE_METADATA_PATH,
    resource_dir: Path = DEFAULT_RESOURCE_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """构建、核验并写入 standard_cases1.json。"""
    output_path = output_path.resolve()
    if output_path == PROTECTED_OUTPUT_PATH.resolve() or output_path.name == "standard_cases.json":
        raise ValueError("拒绝覆盖原 standard_cases.json，请输出为 standard_cases1.json")

    chunks = load_chunks(chunks_path)
    images_by_chunk, image_sources = _image_ids_by_chunk(image_metadata_path)
    source_indexes: dict[str, int] = defaultdict(int)
    cases: list[dict[str, Any]] = []

    for candidate in _case_candidates(chunks):
        source_indexes[candidate.source_pdf] += 1
        index = source_indexes[candidate.source_pdf]
        case = {
            "case_id": f"{Path(candidate.source_pdf).stem}-std-case-{index:03d}",
            "title": candidate.title,
            "date_range": _extract_date_range(candidate.title, candidate.source_pdf),
            "disaster_types": _extract_disaster_types(candidate),
            "affected_areas": _extract_affected_areas(candidate),
            "source_pdf": candidate.source_pdf,
            "source_chunk_ids": candidate.chunk_ids,
            "evidence_image_ids": _candidate_image_ids(candidate, images_by_chunk),
        }
        cases.append(case)

    validation = _validate_cases(cases, chunks, image_sources, resource_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        **validation,
        "output_path": str(output_path),
        "protected_file_untouched": str(PROTECTED_OUTPUT_PATH),
    }


def main() -> None:
    """命令行入口，参数只用于核验不同副本，默认始终生成主 data 下的新文件。"""
    parser = argparse.ArgumentParser(description="重建并核验精简标准个例 standard_cases1.json")
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--image-metadata-path", type=Path, default=DEFAULT_IMAGE_METADATA_PATH)
    parser.add_argument("--resource-dir", type=Path, default=DEFAULT_RESOURCE_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    result = build(
        chunks_path=args.chunks_path,
        image_metadata_path=args.image_metadata_path,
        resource_dir=args.resource_dir,
        output_path=args.output_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
