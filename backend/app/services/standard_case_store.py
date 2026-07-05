from __future__ import annotations

import json
from pathlib import Path

from backend.app.models import StandardCase


class JsonStandardCaseStore:
    """标准化个例 JSON 存储。"""

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def replace_cases(self, cases: list[StandardCase]) -> None:
        """完整替换标准化个例库。"""
        data = [case.to_dict() for case in sorted(cases, key=lambda item: item.case_id)]
        self.storage_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_cases(self) -> list[StandardCase]:
        """返回全部标准化个例。"""
        if not self.storage_path.exists():
            return []
        data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        return [StandardCase.from_dict(item) for item in data]

    def get_case(self, case_id: str) -> StandardCase | None:
        """按 case_id 返回单条标准化个例。"""
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
        """按第一期需要的几个结构化字段做轻量过滤。"""
        cases = self.list_cases()
        if date:
            cases = [case for case in cases if date in case.date_range or date in case.title]
        if disaster_type:
            cases = [
                case for case in cases
                if disaster_type in case.title
                or disaster_type in case.summary
                or disaster_type in case.disaster_types
            ]
        if area:
            cases = [
                case for case in cases
                if area in case.summary
                or area in case.weather_facts
                or area in case.affected_areas
            ]
        if source_pdf:
            cases = [case for case in cases if case.source_pdf == source_pdf]
        return cases
