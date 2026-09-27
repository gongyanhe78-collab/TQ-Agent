"""
标准化个例存储模块
使用 JSON 文件持久化存储标准化气象灾害个例，支持全量替换、
列表查询、按 ID 查询和结构化字段过滤检索。
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from backend.app.models import StandardCase


class JsonStandardCaseStore:
    """
    标准化个例 JSON 存储器
    提供标准化个例的持久化存储、检索和查询功能，支持按时间、灾种、
    区域等结构化字段进行轻量过滤。
    """

    def __init__(self, storage_path: Path):
        """
        初始化标准化个例存储器

        Args:
            storage_path: JSON 存储文件路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_lock = Lock()
        self._cached_signature: tuple[int, int] | None = None
        self._cached_cases: list[StandardCase] = []

    def replace_cases(self, cases: list[StandardCase]) -> None:
        """完整替换标准化个例库。"""
        data = [case.to_dict() for case in sorted(cases, key=lambda item: item.case_id)]
        self.storage_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with self._cache_lock:
            self._cached_signature = None
            self._cached_cases = []

    def list_cases(self) -> list[StandardCase]:
        """返回全部标准化个例。"""
        if not self.storage_path.exists():
            return []
        stat = self.storage_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        with self._cache_lock:
            if signature != self._cached_signature:
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
                self._cached_cases = [StandardCase.from_dict(item) for item in data]
                self._cached_signature = signature
            # 标准个例在问答链路中只读，返回新列表避免调用方改变缓存容器。
            return list(self._cached_cases)

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
