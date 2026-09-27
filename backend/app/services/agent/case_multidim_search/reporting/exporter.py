"""个例检索结果的 CSV 导出。"""
from __future__ import annotations

import csv
import json
import re
from datetime import date
from pathlib import Path

from backend.app.models import StandardCase


class CaseExporter:
    """导出内容直接来自标准化个例，不通过大模型重写。"""

    # CSV 只导出结构化检索需要核对的字段，summary 当前为空且不适合作为表格列。
    FIELDS = ("case_id", "title", "start_date", "end_date", "disaster_types", "city_tags", "source_pdf")
    SHANXI_CITIES = ("太原", "大同", "朔州", "忻州", "阳泉", "晋中", "吕梁", "长治", "晋城", "临汾", "运城")

    def export_csv(self, cases: list[StandardCase], output_path: Path) -> Path:
        """执行检索并将命中个例导出为表格文件。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.FIELDS))
            writer.writeheader()
            for case in cases:
                writer.writerow(self._row(case))
        return output_path

    def _row(self, case: StandardCase) -> dict[str, str]:
        """把标准化个例转换为统一的导出行，并补齐旧数据中缺失的派生字段。"""
        data = case.to_dict()
        start_date, end_date = self._date_range(case)
        data["start_date"] = case.start_date or start_date
        data["end_date"] = case.end_date or end_date
        data["city_tags"] = case.city_tags or self._city_tags(case)
        return {
            field: json.dumps(data[field], ensure_ascii=False) if isinstance(data[field], list) else str(data[field] or "")
            for field in self.FIELDS
        }

    def _city_tags(self, case: StandardCase) -> list[str]:
        """从影响区域里回填山西地市，避免新标准化 JSON 未写 city_tags 时导出空列。"""
        text = " ".join([case.title, case.date_range, " ".join(case.affected_areas)])
        return [city for city in self.SHANXI_CITIES if city in text]

    def _date_range(self, case: StandardCase) -> tuple[str, str]:
        """从 date_range、标题或 PDF 文件名中推断 ISO 日期，导出时只做兜底不改原始 JSON。"""
        text = " ".join([case.date_range, case.title, case.source_pdf])
        year = self._year(case, text)
        if not year:
            return "", ""

        normalized = re.sub(r"\s+", "", text)
        span = re.search(
            r"(?:(20\d{2})年)?(\d{1,2})月(\d{1,2})(?:日)?[-~～—至到]+(?:(\d{1,2})月)?(\d{1,2})(?:日)?",
            normalized,
        )
        if span:
            start_month = int(span.group(2))
            start_day = int(span.group(3))
            end_month = int(span.group(4) or start_month)
            end_day = int(span.group(5))
            return self._iso(year, start_month, start_day), self._iso(year, end_month, end_day)

        single = re.search(r"(?:(20\d{2})年)?(\d{1,2})月(\d{1,2})日", normalized)
        if single:
            value = self._iso(year, int(single.group(2)), int(single.group(3)))
            return value, value
        return "", ""

    def _year(self, case: StandardCase, text: str) -> int | None:
        """优先使用标准字段年份，缺失时从日期描述或来源文件名中提取。"""
        if case.year:
            return case.year
        match = re.search(r"20\d{2}", text)
        return int(match.group(0)) if match else None

    def _iso(self, year: int, month: int, day: int) -> str:
        """校验并格式化日期，异常日期不强行导出。"""
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return ""
