"""共享本地知识库时间范围的读取与缓存。"""
from __future__ import annotations

import calendar
import json
import re
from datetime import date
from pathlib import Path
from threading import Lock
from typing import Any

from .schemas import KnowledgeRange


class KnowledgeRangeReader:
    """按文件修改时间缓存知识库起止日期，避免每轮重复读取大文件。"""

    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "standard_cases1.json"
        self._lock = Lock()
        self._cached_mtime_ns = -1
        self._cached = KnowledgeRange()

    def read(self) -> KnowledgeRange:
        """返回当前标准化个例库覆盖范围，文件更新后自动重新计算。"""
        try:
            mtime_ns = self.path.stat().st_mtime_ns
        except OSError:
            return KnowledgeRange(source=str(self.path))
        with self._lock:
            if mtime_ns == self._cached_mtime_ns:
                return self._cached.model_copy(deep=True)
            value = self._calculate()
            self._cached_mtime_ns = mtime_ns
            self._cached = value
            return value.model_copy(deep=True)

    def _calculate(self) -> KnowledgeRange:
        """扫描日期字段并计算所有有效个例的最小、最大日期。"""
        try:
            records = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return KnowledgeRange(source=str(self.path))
        if not isinstance(records, list):
            return KnowledgeRange(source=str(self.path))
        dates: list[date] = []
        valid_records = [item for item in records if isinstance(item, dict)]
        for item in valid_records:
            dates.extend(self._record_dates(item))
        return KnowledgeRange(
            start_date=min(dates).isoformat() if dates else "",
            end_date=max(dates).isoformat() if dates else "",
            case_count=len(valid_records),
            source=str(self.path),
        )

    def _record_dates(self, item: dict[str, Any]) -> list[date]:
        """优先读取标准日期，缺失时解析中文日期范围。"""
        result: list[date] = []
        for key in ("start_date", "end_date"):
            parsed = self._iso_date(str(item.get(key) or ""))
            if parsed:
                result.append(parsed)
        for key in ("date_range", "title"):
            result.extend(self._dates_from_text(str(item.get(key) or "")))
        return result

    def _dates_from_text(self, text: str) -> list[date]:
        """兼容“2025年7月28-31日”等标准化个例日期写法。"""
        result: list[date] = []
        for match in re.finditer(r"(20\d{2})[-年](\d{1,2})[-月](\d{1,2})日?", text):
            parsed = self._safe_date(*map(int, match.groups()))
            if parsed:
                result.append(parsed)
        range_match = re.search(
            r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?\s*"
            r"(?:至|到|[-—~～])\s*(?:(\d{1,2})\s*月)?\s*(\d{1,2})\s*日",
            text,
        )
        if range_match:
            year, start_month, start_day = map(int, range_match.group(1, 2, 3))
            end_month = int(range_match.group(4) or start_month)
            end_day = int(range_match.group(5))
            for month, day in ((start_month, start_day), (end_month, end_day)):
                parsed = self._safe_date(year, month, day)
                if parsed:
                    result.append(parsed)
        if not result:
            month_match = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月", text)
            if month_match:
                year, month = map(int, month_match.groups())
                first = self._safe_date(year, month, 1)
                last = self._safe_date(year, month, calendar.monthrange(year, month)[1]) if first else None
                result.extend(value for value in (first, last) if value)
        return result

    def _iso_date(self, value: str) -> date | None:
        """安全解析 ISO 日期。"""
        try:
            return date.fromisoformat(value) if value else None
        except ValueError:
            return None

    def _safe_date(self, year: int, month: int, day: int) -> date | None:
        """过滤资料中的非法日期而不中断范围计算。"""
        try:
            return date(year, month, day)
        except ValueError:
            return None
