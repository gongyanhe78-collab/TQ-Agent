from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass

from backend.app.models import DocumentChunk, StandardCase
from backend.app.services.image_extraction import ImageEvidenceStore


DISASTER_TERMS = (
    "暴雨",
    "大暴雨",
    "强降水",
    "雷暴",
    "雷雨",
    "大风",
    "强对流",
    "冰雹",
    "暴雪",
    "雨雪",
    "降雪",
    "寒潮",
    "低温",
    "霜冻",
    "高温",
    "沙尘",
    "雾",
)
AREA_TERMS = (
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
    "山西",
    "沈阳",
    "大连",
    "鞍山",
    "抚顺",
    "本溪",
    "丹东",
    "锦州",
    "营口",
    "阜新",
    "辽阳",
    "盘锦",
    "铁岭",
    "朝阳",
    "葫芦岛",
    "辽宁",
)
DATE_TITLE_PREFIX = r"(?:(?:\d{1,2}\s*月\s*)?\d{1,2}\s*日|\d{1,2}\s*月\s*\d{1,2})(?:\s*[-~～至]\s*\d{1,2}\s*(?:日)?)?"
TITLE_PATTERN = re.compile(
    rf"(?P<title>{DATE_TITLE_PREFIX}.{{0,16}}?"
    r"(?:天气过程|过程|暴雨|大暴雨|强降水|雷暴|雷雨|大风|强对流|冰雹|暴雪|雨雪|降雪|寒潮|低温|霜冻|高温|沙尘|雾))"
)
PROCESS_TITLE_PATTERN = re.compile(
    rf"(?P<title>{DATE_TITLE_PREFIX}.{{0,24}}?(?:天气过程|过程))"
)
CHAPTER_PROCESS_TITLE_PATTERN = re.compile(
    r"(?:^|[。\n\r])\s*[一二三四五六七八九十]+[、.．]\s*"
    rf"(?P<title>{DATE_TITLE_PREFIX}.{{0,30}}?"
    r"(?:天气过程|过程|天气))"
)
CHAPTER_TITLE_PATTERN = re.compile(
    r"(?:^|[。\n\r])\s*[一二三四五六七八九十]+[、.．]\s*"
    rf"(?P<title>{DATE_TITLE_PREFIX}.{{0,30}}?"
    r"(?:天气过程|过程|天气|暴雨|大暴雨|强降水|雷暴|雷雨|大风|强对流|冰雹|暴雪|雨雪|降雪|寒潮|低温|霜冻|高温|沙尘|雾))"
)
STANDALONE_TITLE_PATTERN = re.compile(
    rf"^(?P<title>{DATE_TITLE_PREFIX}.{{0,24}}?"
    r"(?:天气过程|过程|天气|暴雨|大暴雨|强降水|雷暴|雷雨|大风|强对流|冰雹|暴雪|雨雪|降雪|寒潮|低温|霜冻|高温|沙尘|雾))$"
)
DATE_PATTERN = re.compile(r"(?:(\d{1,2})\s*月\s*)?(\d{1,2})\s*(?:日)?(?:\s*[-~～至]\s*(\d{1,2})\s*(?:日)?)?")


@dataclass
class StandardCaseCandidate:
    source_pdf: str
    title: str
    chunk_ids: list[str]
    text: str


class StandardCaseBuilder:
    """从文档 chunk 和图片证据构建第一期标准化个例。"""

    def __init__(self, image_store: ImageEvidenceStore, llm_client=None, use_llm: bool = True):
        self.image_store = image_store
        self.llm_client = llm_client
        self.use_llm = use_llm

    def build(self, chunks: list[DocumentChunk]) -> list[StandardCase]:
        candidates = self._candidates_from_chunks(chunks)
        cases = []
        source_counts: dict[str, int] = defaultdict(int)
        for candidate in candidates:
            source_counts[candidate.source_pdf] += 1
            index = source_counts[candidate.source_pdf]
            case = self._case_from_candidate(candidate, index)
            if case is not None:
                cases.append(case)
        return cases

    def _candidates_from_chunks(self, chunks: list[DocumentChunk]) -> list[StandardCaseCandidate]:
        by_pdf: dict[str, list[DocumentChunk]] = defaultdict(list)
        for chunk in chunks:
            by_pdf[chunk.source_pdf].append(chunk)

        candidates: list[StandardCaseCandidate] = []
        for source_pdf, pdf_chunks in sorted(by_pdf.items()):
            ordered = sorted(pdf_chunks, key=lambda item: item.chunk_no)
            current_title = ""
            current_chunks: list[DocumentChunk] = []
            for chunk in ordered:
                title = self._extract_title(chunk.content)
                if title and current_chunks:
                    candidates.append(self._candidate(source_pdf, current_title, current_chunks))
                    current_chunks = []
                if title:
                    current_title = title
                if current_title:
                    current_chunks.append(chunk)
            if current_chunks:
                candidates.append(self._candidate(source_pdf, current_title, current_chunks))
        return [candidate for candidate in candidates if candidate.title]

    def _candidate(
        self,
        source_pdf: str,
        title: str,
        chunks: list[DocumentChunk],
    ) -> StandardCaseCandidate:
        return StandardCaseCandidate(
            source_pdf=source_pdf,
            title=title,
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            text="\n".join(chunk.content for chunk in chunks),
        )

    def _case_from_candidate(self, candidate: StandardCaseCandidate, index: int) -> StandardCase | None:
        llm_case = self._llm_standardize(candidate, index)
        if llm_case is not None:
            return llm_case
        return self._rule_standardize(candidate, index)

    def _llm_standardize(self, candidate: StandardCaseCandidate, index: int) -> StandardCase | None:
        if not self.use_llm or not self.llm_client or not self.llm_client.is_available():
            return None
        try:
            from openai import OpenAI
        except ImportError:
            return None

        prompt = {
            "source_pdf": candidate.source_pdf,
            "title_hint": candidate.title,
            "source_chunk_ids": candidate.chunk_ids,
            "text": candidate.text[:5000],
        }
        try:
            client = OpenAI(api_key=self.llm_client.api_key, base_url=self.llm_client.base_url)
            completion = client.chat.completions.create(
                model=self.llm_client.model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是气象灾害个例标准化抽取助手。"
                            "只根据给定材料抽取一条灾害个例记录，输出 JSON。"
                            "不要编造材料中没有的事实；无法确定的字段用空字符串或空数组。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "请输出 JSON："
                            '{"title": str, "date_range": str, "disaster_types": list[str], '
                            '"affected_areas": list[str], "summary": str, "weather_facts": str, '
                            '"forecast_focus": str, "confidence": float}\n'
                            + json.dumps(prompt, ensure_ascii=False)
                        ),
                    },
                ],
            )
            data = json.loads(completion.choices[0].message.content or "{}")
        except Exception:
            return None

        return self._case_from_data(candidate, index, data, default_confidence=0.75)

    def _rule_standardize(self, candidate: StandardCaseCandidate, index: int) -> StandardCase:
        data = {
            "title": candidate.title,
            "date_range": self._extract_date_range(candidate.title),
            "disaster_types": self._extract_terms(candidate.text, DISASTER_TERMS),
            "affected_areas": self._extract_terms(candidate.text, AREA_TERMS),
            "summary": self._first_sentences(candidate.text, 2),
            "weather_facts": self._section_like(candidate.text, ("天气实况", "实况", "降水实况", "灾情")),
            "forecast_focus": self._section_like(candidate.text, ("预报", "服务", "提示", "影响")),
            "confidence": 0.45,
        }
        return self._case_from_data(candidate, index, data, default_confidence=0.45)

    def _case_from_data(
        self,
        candidate: StandardCaseCandidate,
        index: int,
        data: dict,
        default_confidence: float,
    ) -> StandardCase:
        title = str(data.get("title") or candidate.title).strip()
        date_range = str(data.get("date_range") or self._extract_date_range(title)).strip()
        evidence_image_ids = self._evidence_image_ids(candidate.chunk_ids)
        return StandardCase(
            case_id=f"{candidate.source_pdf.rsplit('.', 1)[0]}-std-case-{index:03d}",
            title=title,
            date_range=date_range,
            disaster_types=self._list_of_strings(data.get("disaster_types")),
            affected_areas=self._list_of_strings(data.get("affected_areas")),
            source_pdf=candidate.source_pdf,
            source_chunk_ids=candidate.chunk_ids,
            summary=str(data.get("summary") or "").strip(),
            weather_facts=str(data.get("weather_facts") or "").strip(),
            forecast_focus=str(data.get("forecast_focus") or "").strip(),
            evidence_image_ids=evidence_image_ids,
            confidence=float(data.get("confidence") or default_confidence),
        )

    def _extract_title(self, text: str) -> str:
        heading_match = CHAPTER_PROCESS_TITLE_PATTERN.search(text) or CHAPTER_TITLE_PATTERN.search(text)
        if heading_match:
            return self._clean_title(heading_match.group("title"))
        return ""

    def _clean_title(self, title: str) -> str:
        return re.sub(r"\s+", "", title).strip("：:，,。；;")

    def _extract_date_range(self, text: str) -> str:
        match = DATE_PATTERN.search(text)
        if not match:
            return ""
        month, start, end = match.groups()
        if month and end:
            return f"{month}月{start}-{end}日"
        if month:
            return f"{month}月{start}日"
        if end:
            return f"{start}-{end}日"
        return f"{start}日"

    def _extract_terms(self, text: str, terms: tuple[str, ...]) -> list[str]:
        return [term for term in terms if term in text]

    def _first_sentences(self, text: str, limit: int) -> str:
        parts = [part.strip() for part in re.split(r"[。！？\n]", text) if part.strip()]
        return "。".join(parts[:limit])[:500]

    def _section_like(self, text: str, labels: tuple[str, ...]) -> str:
        for label in labels:
            index = text.find(label)
            if index >= 0:
                return text[index:index + 500].strip()
        return ""

    def _evidence_image_ids(self, chunk_ids: list[str]) -> list[str]:
        seen = set()
        image_ids = []
        for chunk_id in chunk_ids:
            for image in self.image_store.list_by_chunk_id(chunk_id, limit=3):
                if image.image_id in seen:
                    continue
                seen.add(image.image_id)
                image_ids.append(image.image_id)
        return image_ids

    def _list_of_strings(self, value) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []
