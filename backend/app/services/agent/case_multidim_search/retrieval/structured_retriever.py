"""标准化个例的确定性多条件检索。"""
from __future__ import annotations

import re
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date

from backend.app.models import StandardCase
from backend.app.services.agent.case_multidim_search.schemas import CaseSearchQuery


COMPOUND_DISASTER_TERMS = {
    "雷暴大风": ("雷暴", "大风"),
}

# 省级范围使用行政语义匹配，避免“山西”和标准个例中的“全省”因字面不同而互相漏检。
SHANXI_PROVINCE_TERMS = {"山西", "山西省", "全省", "全省各地", "我省"}


@dataclass
class StructuredMatch:
    """结构化筛选命中及其可解释字段。"""

    case: StandardCase
    score: float
    matched_fields: list[str] = field(default_factory=list)


class StructuredCaseRetriever:
    """对显式条件执行硬过滤，不满足任一维度的个例都会被排除。"""

    def search(self, cases: list[StandardCase], query: CaseSearchQuery) -> list[StructuredMatch]:
        """执行当前检索器定义的查询和结果整理流程。"""
        matches = []
        for case in cases:
            matched_fields = self._matched_fields(case, query)
            if matched_fields is None:
                continue
            active_count = max(1, self._active_filter_count(query))
            matches.append(
                StructuredMatch(
                    case=case,
                    score=round(len(matched_fields) / active_count, 4),
                    matched_fields=matched_fields,
                )
            )
        matches.sort(key=lambda item: (item.score, item.case.start_date, item.case.case_id), reverse=True)
        return matches

    def _matched_fields(self, case: StandardCase, query: CaseSearchQuery) -> list[str] | None:
        """检查个例是否满足所有已启用过滤条件。"""
        case_text = self._case_text(case)
        region_terms = list(dict.fromkeys([*(query.cities or []), *(query.areas or [])]))
        region_values = list(dict.fromkeys([*(case.city_tags or []), *(case.affected_areas or [])]))
        checks = [
            ("date", bool(query.start_date or query.end_date or query.years or query.months), self._matches_date(case, query)),
            ("disaster", bool(query.disaster_types), self._matches_disasters(query.disaster_types, case.disaster_types, case_text)),
            # 城市和区域同属“影响区域”维度：维度内部多选按 OR 命中，避免要求太原、大同、阳泉等全部同时满足。
            ("region", bool(region_terms), self._matches_terms(region_terms, region_values, case_text)),
        ]
        if any(active and not matched for _, active, matched in checks):
            return None
        return [name for name, active, matched in checks if active and matched]

    def _matches_date(self, case: StandardCase, query: CaseSearchQuery) -> bool:
        """判断个例日期是否与查询范围重叠，并在标准字段缺失时回退解析文本日期。"""
        case_years = self._case_years(case)
        case_months = self._case_months(case)
        if query.years and not set(query.years).intersection(case_years):
            return False
        if query.months and not set(query.months).intersection(case_months):
            return False
        if not query.start_date and not query.end_date:
            return True
        case_start = self._iso_date(case.start_date)
        case_end = self._iso_date(case.end_date) or case_start
        if case_start is None:
            # 旧标准个例的日期可能只保存在 date_range/title 中；把已解析的年月
            # 转为整月区间参与重叠判断，避免 ISO 起止查询把这些个例全部过滤掉。
            query_start = self._iso_date(query.start_date) or date.min
            query_end = self._iso_date(query.end_date) or date.max
            inferred_years = case_years or set(range(query_start.year, query_end.year + 1))
            inferred_months = case_months or set(range(1, 13))
            return any(
                date(year, month, 1) <= query_end
                and date(year, month, monthrange(year, month)[1]) >= query_start
                for year in inferred_years
                for month in inferred_months
                if 1 <= month <= 12
            )
        query_start = self._iso_date(query.start_date) or date.min
        query_end = self._iso_date(query.end_date) or date.max
        return case_start <= query_end and case_end >= query_start

    def _case_years(self, case: StandardCase) -> set[int]:
        """从标准字段、日期描述和来源 PDF 中回退提取个例年份。"""
        years = {case.year} if case.year else set()
        for text in (case.start_date, case.end_date, case.date_range, case.title, case.source_pdf):
            years.update(int(value) for value in re.findall(r"20\d{2}", text or ""))
        return years

    def _case_months(self, case: StandardCase) -> set[int]:
        """从标准字段、日期描述和来源 PDF 中回退提取个例月份。"""
        months = {month for month in case.months if 1 <= month <= 12}
        for value in (case.start_date, case.end_date):
            parsed = self._iso_date(value)
            if parsed:
                months.add(parsed.month)
        for text in (case.date_range, case.title):
            months.update(
                month
                for month in (int(value) for value in re.findall(r"(\d{1,2})\s*月", text or ""))
                if 1 <= month <= 12
            )
        source_match = re.search(r"(?:20\d{2})[-_年]?(\d{1,2})(?:\D|$)", case.source_pdf or "")
        if source_match:
            month = int(source_match.group(1))
            if 1 <= month <= 12:
                months.add(month)
        return months

    def _matches_disasters(self, wanted: list[str], values: list[str], text: str) -> bool:
        """灾种多选按 OR 语义匹配：命中任一勾选灾种即可进入结果。"""
        return any(self._matches_disaster_term(term, values, text) for term in wanted)

    def _matches_disaster_term(self, term: str, values: list[str], text: str) -> bool:
        """判断单个灾种是否命中，并兼容“雷暴大风”等复合灾种的历史标注。"""
        term = str(term or "").strip()
        if not term:
            return False
        # 有些历史标准化个例的标题包含“强对流”，但 disaster_types 漏标；这里合并列表和文本判断。
        if term in text or any(term == value or term in value for value in values):
            return True
        atomics = COMPOUND_DISASTER_TERMS.get(term, ())
        if not atomics:
            return False
        # 对单个复合灾种筛选，允许原始标签拆成多个原子灾种时仍能命中。
        return all(
            atomic in text or any(atomic == value or atomic in value for value in values)
            for atomic in atomics
        )

    def _matches_terms(self, wanted: list[str], values: list[str], text: str) -> bool:
        """判断目标词是否命中当前个例。"""
        if any(term in SHANXI_PROVINCE_TERMS for term in wanted):
            if any(value in SHANXI_PROVINCE_TERMS for value in values):
                return True
            # 本系统知识库均属于山西省，市县级标签也从属于省级查询范围。
            if values:
                return True
        return any(term in text or any(term in value or value in term for value in values) for term in wanted)

    def _case_text(self, case: StandardCase) -> str:
        """拼接可用于确定性匹配的个例文本。"""
        return " ".join(
            [
                case.title,
                case.date_range,
                case.summary,
                case.weather_facts,
                case.forecast_focus,
                " ".join(case.disaster_types),
                " ".join(case.affected_areas),
                " ".join(case.city_tags),
            ]
        )

    def _active_filter_count(self, query: CaseSearchQuery) -> int:
        """统计当前查询启用的过滤维度数量。"""
        return sum(
            bool(value)
            for value in (
                query.start_date or query.end_date or query.years or query.months,
                query.disaster_types,
                query.cities or query.areas,
            )
        )

    def _iso_date(self, value: str) -> date | None:
        """安全解析标准日期并处理非法值。"""
        try:
            return date.fromisoformat(value) if value else None
        except ValueError:
            return None



