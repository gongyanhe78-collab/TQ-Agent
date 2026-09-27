"""把检索结果汇总成报告和图表可复用的统计数据。"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from itertools import combinations

from backend.app.models import StandardCase
from backend.app.services.agent.case_multidim_search.schemas import IntensityMetric


class CaseAggregator:
    """把个例和强度证据聚合成报告与图表可复用的数据。"""

    def aggregate(
        self,
        cases: list[StandardCase],
        intensity_metrics: dict[str, list[IntensityMetric]] | None = None,
    ) -> dict:
        """统计个例的月份、灾种、地市、强度分布和灾种共现关系。"""
        intensity_metrics = intensity_metrics or {}
        month_counts: Counter[int] = Counter()
        disaster_counts: Counter[str] = Counter()
        city_counts: Counter[str] = Counter()
        intensity_values: dict[str, list[float]] = defaultdict(list)
        intensity_case_values: dict[str, list[dict]] = defaultdict(list)
        disaster_pairs: Counter[tuple[str, str]] = Counter()

        for case in cases:
            for month in self.case_months(case):
                month_counts[month] += 1

            disasters = self._clean_text_list(case.disaster_types)
            for disaster in disasters:
                disaster_counts[disaster] += 1

            # 只统计同一个个例内部真正同时出现的灾种组合，避免重复灌水。
            unique_disasters = list(dict.fromkeys(disasters))
            for left, right in combinations(sorted(unique_disasters), 2):
                disaster_pairs[(left, right)] += 1

            for city in self._case_cities(case):
                city_counts[city] += 1

            # 数据类别不再作为检索或报告统计维度；证据图片和原文片段仍在个例详情中保留用于溯源。

            for metric in intensity_metrics.get(case.case_id, []):
                intensity_values[metric.metric_name].append(metric.value)
                # 保留指标对应的个例来源，供灾种特性图按个例对比使用。
                intensity_case_values[metric.metric_name].append(
                    {
                        "case_id": case.case_id,
                        "title": case.title or case.case_id,
                        "value": metric.value,
                        "unit": metric.unit,
                        "location": metric.location,
                    }
                )

        disaster_cooccurrence = self._build_disaster_cooccurrence(disaster_counts, disaster_pairs)

        return {
            "case_count": len(cases),
            "month_counts": dict(sorted(month_counts.items(), key=lambda item: item[0])),
            "disaster_counts": dict(disaster_counts.most_common()),
            "disaster_cooccurrence": disaster_cooccurrence,
            "city_counts": dict(city_counts.most_common()),
            "intensity_values": dict(intensity_values),
            "intensity_case_values": dict(intensity_case_values),
        }

    def _build_disaster_cooccurrence(
        self,
        disaster_counts: Counter[str],
        disaster_pairs: Counter[tuple[str, str]],
        limit: int = 8,
    ) -> dict:
        """把高频灾种整理成共现热力图所需的矩阵数据。"""
        labels = [name for name, _ in disaster_counts.most_common(limit)]
        if len(labels) < 2:
            return {"labels": labels, "matrix": []}

        matrix: list[list[float]] = []
        for row in labels:
            row_values: list[float] = []
            for col in labels:
                if row == col:
                    row_values.append(float(disaster_counts[row]))
                else:
                    key = tuple(sorted((row, col)))
                    row_values.append(float(disaster_pairs.get(key, 0)))
            matrix.append(row_values)
        return {"labels": labels, "matrix": matrix}

    def _clean_text_list(self, values: list[str] | None) -> list[str]:
        """去掉空白项并保留原始顺序，便于后续统计。"""
        cleaned = []
        for value in values or []:
            item = str(value or "").strip()
            if item:
                cleaned.append(item)
        return cleaned

    def _case_cities(self, case: StandardCase) -> list[str]:
        """优先读取地市标签，缺失时再从影响区域回退，并剔除省级名称。"""
        values = case.city_tags or [
            area for area in case.affected_areas
            if area not in {"山西", "山西省", "全省"}
        ]
        return list(dict.fromkeys(self._clean_text_list(values)))

    def case_months(self, case: StandardCase) -> list[int]:
        """优先读取月份标签，缺失时从日期文本中回退提取。"""
        if case.months:
            return sorted({month for month in case.months if 1 <= month <= 12})
        for text, pattern in (
            (case.start_date, r"^\d{4}-(\d{2})-\d{2}$"),
            (case.date_range, r"(\d{1,2})\s*月"),
            (case.title, r"(\d{1,2})\s*月"),
            (case.source_pdf, r"(?:20\d{2})[-_年](\d{1,2})"),
        ):
            match = re.search(pattern, text or "")
            if match:
                month = int(match.group(1))
                if 1 <= month <= 12:
                    return [month]
        return []



