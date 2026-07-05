import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.models import ExtractedCase
from backend.app.services.vector_store import JsonCaseStore


class JsonCaseStoreTests(unittest.TestCase):
    def test_store_round_trip_and_query(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonCaseStore(Path(tmp_dir) / "cases.json")
            cases = [
                ExtractedCase(
                    source_pdf="FST2025-3.pdf",
                    case_id="FST2025-3-case-01",
                    case_no=1,
                    title="1-3日雨雪天气过程",
                    date_range="1-3日",
                    content="全省出现雨雪天气过程。",
                    embedding=[0.9, 0.1],
                ),
                ExtractedCase(
                    source_pdf="FST2025-3.pdf",
                    case_id="FST2025-3-case-02",
                    case_no=2,
                    title="14-15日暴雪天气过程",
                    date_range="14-15日",
                    content="北部出现暴雪天气过程。",
                    embedding=[0.1, 0.9],
                ),
            ]

            store.upsert_cases(cases)

            ids = store.list_case_ids()
            hits = store.query([0.0, 1.0], top_k=1)

            self.assertEqual(["FST2025-3-case-01", "FST2025-3-case-02"], ids)
            self.assertEqual("FST2025-3-case-02", hits[0].case.case_id)

    def test_delete_cases_removes_requested_ids_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonCaseStore(Path(tmp_dir) / "cases.json")
            cases = [
                ExtractedCase(
                    source_pdf="FST2025-3.pdf",
                    case_id="FST2025-3-case-01",
                    case_no=1,
                    title="case one",
                    date_range="1-3日",
                    content="content one",
                    embedding=[0.9, 0.1],
                ),
                ExtractedCase(
                    source_pdf="FST2025-4.pdf",
                    case_id="FST2025-4-case-01",
                    case_no=1,
                    title="case two",
                    date_range="4-5日",
                    content="content two",
                    embedding=[0.1, 0.9],
                ),
            ]
            store.upsert_cases(cases)

            result = store.delete_cases(["FST2025-3-case-01", "missing-case"])

            self.assertEqual(["FST2025-3-case-01"], result["deleted_keys"])
            self.assertEqual(["missing-case"], result["missing_keys"])
            self.assertEqual(["FST2025-4-case-01"], store.list_case_ids())


if __name__ == "__main__":
    unittest.main()
