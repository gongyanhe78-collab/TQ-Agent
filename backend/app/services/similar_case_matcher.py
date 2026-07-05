from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from backend.app.models import StandardCase
from backend.app.services.image_extraction import ImageEvidence, ImageEvidenceStore


AREA_ALIASES = {
    "山西北部": ("大同", "朔州", "忻州", "北部", "山西北部"),
    "北部": ("大同", "朔州", "忻州", "北部", "山西北部"),
    "山西中部": ("太原", "阳泉", "晋中", "吕梁", "中部", "山西中部"),
    "中部": ("太原", "阳泉", "晋中", "吕梁", "中部", "山西中部"),
    "山西南部": ("长治", "晋城", "临汾", "运城", "南部", "山西南部"),
    "南部": ("长治", "晋城", "临汾", "运城", "南部", "山西南部"),
}


DISASTER_TERMS = (
    "雷暴大风",
    "强对流",
    "大暴雨",
    "暴雨",
    "强降水",
    "短时强降水",
    "雷暴",
    "大风",
    "冰雹",
    "暴雪",
    "雨雪",
    "降雪",
    "寒潮",
    "高温",
    "低温",
    "沙尘",
    "雾",
)


IMAGE_KEYWORDS = {
    "radar": ("雷达", "回波", "组合反射率", "风雷"),
    "satellite": ("卫星", "云图", "红外", "可见光"),
    "precipitation": ("降水图", "雨量图", "累计降水", "降水"),
    "wind": ("大风图", "风速", "阵风"),
    "sounding": ("探空", "TlnP", "TInP"),
    "synoptic": ("环流", "形势图", "海平面气压"),
    "temperature": ("高温图", "气温图", "温度图"),
    "warning": ("预警", "风险图"),
}


@dataclass
class SimilarCaseQuery:
    """相似个例匹配输入。"""

    q: str = ""
    date: str = ""
    disaster_type: str = ""
    area: str = ""
    image_type: str = ""
    data_category: str = ""
    source_pdf: str = ""
    top_n: int = 5


@dataclass
class SimilarCaseMatch:
    """单条相似个例匹配结果。"""

    case: StandardCase
    score: float
    score_breakdown: dict[str, float]
    reasons: list[str] = field(default_factory=list)
    forecast_tips: list[str] = field(default_factory=list)
    evidence_images: list[ImageEvidence] = field(default_factory=list)


class SimilarCaseMatcher:
    """按结构化字段、图像证据和文本线索综合匹配历史标准个例。"""

    def __init__(
        self,
        image_store: ImageEvidenceStore,
        image_filter: Callable[[ImageEvidence, str | None, str | None], bool],
    ):
        self.image_store = image_store
        self.image_filter = image_filter

    def match(self, query: SimilarCaseQuery, cases: list[StandardCase]) -> list[SimilarCaseMatch]:
        """返回相似度最高的 3-5 个历史个例。"""
        normalized = self._normalize_query(query)
        matches = [self._score_case(normalized, case) for case in cases]
        if normalized.image_type or normalized.data_category:
            matches = [match for match in matches if match.score_breakdown["image"] > 0]
        matches = [match for match in matches if match.score > 0]
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[: max(1, min(normalized.top_n, 5))]

    def _normalize_query(self, query: SimilarCaseQuery) -> SimilarCaseQuery:
        parsed = parse_case_query(query.q)
        return SimilarCaseQuery(
            q=query.q.strip(),
            date=(query.date or parsed.get("date", "")).strip(),
            disaster_type=(query.disaster_type or parsed.get("disaster_type", "")).strip(),
            area=(query.area or parsed.get("area", "")).strip(),
            image_type=(query.image_type or parsed.get("image_type", "")).strip(),
            data_category=(query.data_category or parsed.get("data_category", "")).strip(),
            source_pdf=query.source_pdf.strip(),
            top_n=query.top_n,
        )

    def _score_case(self, query: SimilarCaseQuery, case: StandardCase) -> SimilarCaseMatch:
        images = self.image_store.list_by_image_ids(case.evidence_image_ids)
        matched_images = self._matched_images(query, images)
        breakdown = {
            "disaster": self._disaster_score(query.disaster_type, case),
            "temporal": self._temporal_score(query.date, case.date_range, case.title),
            "spatial": self._spatial_score(query.area, case),
            "image": self._image_score(query, images, matched_images),
            "text": self._text_score(query.q, case),
            "source": self._source_score(query.source_pdf, case.source_pdf),
        }
        score = (
            breakdown["disaster"] * 0.28
            + breakdown["temporal"] * 0.18
            + breakdown["spatial"] * 0.20
            + breakdown["image"] * 0.16
            + breakdown["text"] * 0.14
            + breakdown["source"] * 0.04
        )
        return SimilarCaseMatch(
            case=case,
            score=round(score, 4),
            score_breakdown={key: round(value, 3) for key, value in breakdown.items()},
            reasons=self._reasons(query, case, breakdown, matched_images),
            forecast_tips=self._forecast_tips(case, matched_images),
            evidence_images=(matched_images or images)[:4],
        )

    def _matched_images(self, query: SimilarCaseQuery, images: list[ImageEvidence]) -> list[ImageEvidence]:
        if not query.image_type and not query.data_category:
            return images[:4]
        return [
            image
            for image in images
            if self.image_filter(image, query.image_type or None, query.data_category or None)
        ]

    def _disaster_score(self, disaster_type: str, case: StandardCase) -> float:
        terms = disaster_terms(disaster_type)
        if not terms:
            return 0.0
        haystack = case_text(case)
        matched = sum(1 for term in terms if term in haystack)
        return matched / len(terms)

    def _temporal_score(self, date: str, case_date: str, case_title: str) -> float:
        query_month, query_days = month_days(date)
        case_month, case_days = month_days(f"{case_date} {case_title}")
        if query_month is None:
            return 0.0
        if query_month != case_month:
            if case_month is None:
                return 0.0
            return 0.35 if abs(query_month - case_month) == 1 else 0.0
        if query_days and case_days:
            return 1.0 if set(query_days) & set(case_days) else 0.72
        return 0.82

    def _spatial_score(self, area: str, case: StandardCase) -> float:
        if not area:
            return 0.0
        haystack = case_text(case)
        precise = AREA_ALIASES.get(area, (area,))
        if any(item in haystack for item in precise):
            return 1.0
        if "山西" in haystack and ("山西" in area or area in AREA_ALIASES):
            return 0.55
        return 0.0

    def _image_score(
        self,
        query: SimilarCaseQuery,
        images: list[ImageEvidence],
        matched_images: list[ImageEvidence],
    ) -> float:
        if query.image_type or query.data_category:
            return 1.0 if matched_images else 0.0
        return 0.45 if images else 0.0

    def _text_score(self, query_text: str, case: StandardCase) -> float:
        tokens = query_tokens(query_text)
        if not tokens:
            return 0.0
        haystack = case_text(case)
        matched = sum(1 for token in tokens if token in haystack)
        return min(1.0, matched / max(4, len(tokens)))

    def _source_score(self, source_pdf: str, case_source: str) -> float:
        if not source_pdf:
            return 0.0
        return 1.0 if source_pdf == case_source else 0.0

    def _reasons(
        self,
        query: SimilarCaseQuery,
        case: StandardCase,
        breakdown: dict[str, float],
        matched_images: list[ImageEvidence],
    ) -> list[str]:
        reasons = []
        if breakdown["disaster"] > 0:
            reasons.append(f"灾种相似：{query.disaster_type} 与历史个例灾种/标题匹配")
        if breakdown["temporal"] >= 0.7:
            reasons.append(f"时段接近：查询时段 {query.date} 与 {case.date_range} 同月或同日")
        elif breakdown["temporal"] > 0:
            reasons.append(f"季节接近：查询时段 {query.date} 与 {case.date_range} 相邻月份")
        if breakdown["spatial"] >= 1:
            reasons.append(f"区域相似：{query.area} 与历史影响区域匹配")
        elif breakdown["spatial"] > 0:
            reasons.append(f"区域泛化匹配：{query.area} 与省级/区域线索有关")
        if matched_images:
            captions = "；".join((image.caption or f"第{image.page_no}页图片") for image in matched_images[:2])
            reasons.append(f"图像证据相似：可参考 {captions}")
        if not reasons:
            reasons.append("文本描述与历史个例存在一定重合")
        return reasons

    def _forecast_tips(self, case: StandardCase, matched_images: list[ImageEvidence]) -> list[str]:
        tips = []
        if case.forecast_focus:
            tips.append(case.forecast_focus[:180])
        elif case.summary:
            tips.append(f"参考历史过程：{case.summary[:160]}")
        if matched_images:
            tips.append("结合相似图像证据，重点核对雷达回波、卫星云图、实况落区与预报落区的偏差。")
        if any(term in case.disaster_types for term in ("雷暴", "大风", "强对流", "冰雹")):
            tips.append("强对流过程需关注短临触发条件、移动方向、阵风和冰雹等局地灾害风险。")
        if any(term in case.disaster_types for term in ("暴雨", "强降水", "大暴雨")):
            tips.append("降水过程需关注低层水汽输送、地形增幅和短时雨强落区。")
        return tips[:3]


def parse_case_query(query: str) -> dict[str, str]:
    """从自然语言中提取相似匹配条件。"""
    parsed: dict[str, str] = {}
    month_day = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日?", query)
    month_only = re.search(r"(?:\d{4}\s*年\s*)?(\d{1,2})\s*月", query)
    if month_day:
        parsed["date"] = f"{month_day.group(1)}月{month_day.group(2)}日"
    elif month_only:
        parsed["date"] = f"{month_only.group(1)}月"

    for term in DISASTER_TERMS:
        if term in query:
            parsed["disaster_type"] = term
            break

    for area_term in (
        "山西北部",
        "山西中部",
        "山西南部",
        "北部",
        "中部",
        "南部",
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
    ):
        if area_term in query:
            parsed["area"] = area_term
            break

    for image_type, keywords in IMAGE_KEYWORDS.items():
        if any(keyword in query for keyword in keywords):
            parsed["image_type"] = image_type
            break
    if "图" in query and "image_type" not in parsed:
        parsed["data_category"] = "图片"
    return parsed


def disaster_terms(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []
    if value == "雷暴大风":
        return ["雷暴", "大风"]
    if value == "雨雪":
        return ["雨", "雪"]
    return [value]


def month_days(value: str) -> tuple[int | None, list[int]]:
    month_match = re.search(r"(\d{1,2})\s*月", value)
    if not month_match:
        return None, []
    month = int(month_match.group(1))
    days = [int(item) for item in re.findall(r"(\d{1,2})\s*日", value)]
    range_match = re.search(r"(\d{1,2})\s*[-~～至]\s*(\d{1,2})\s*日", value)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        if start <= end:
            days.extend(range(start, end + 1))
    return month, sorted(set(days))


def query_tokens(query: str) -> list[str]:
    tokens = [term for term in DISASTER_TERMS if term in query]
    tokens.extend(area for area, aliases in AREA_ALIASES.items() if area in query or any(alias in query for alias in aliases))
    tokens.extend(keyword for keywords in IMAGE_KEYWORDS.values() for keyword in keywords if keyword in query)
    tokens.extend(part for part in re.split(r"[\s，。！？、；：,.!?;:（）()]+", query) if len(part) >= 2)
    deduped = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return deduped


def case_text(case: StandardCase) -> str:
    return " ".join(
        [
            case.title,
            case.date_range,
            case.summary,
            case.weather_facts,
            case.forecast_focus,
            " ".join(case.disaster_types),
            " ".join(case.affected_areas),
            case.source_pdf,
        ]
    )
