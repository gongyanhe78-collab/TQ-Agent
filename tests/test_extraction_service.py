import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.models import ExtractedCase
from backend.app.services.extraction_service import CaseExtractionService
from backend.app.services.llm_client import LlmRefinementResult


class FakePdfReader:
    def read_text(self, pdf_path):
        return "pdf text"


class FakeSplitter:
    def split_cases(self, text, source_pdf):
        return [
            ExtractedCase(
                source_pdf=source_pdf,
                case_id="FST2025-5-case-01",
                case_no=1,
                title="5月2日雷暴大风天气过程",
                date_range="5月2日",
                content="山西出现雷暴大风天气。",
            )
        ]


class RejectingLlmClient:
    def refine_case(self, candidate):
        return LlmRefinementResult(
            is_case=False,
            title="",
            date_range="",
            content="",
            reason="model rejected valid candidate",
        )


class ExtractionServiceTests(unittest.TestCase):
    def test_rule_candidates_are_not_removed_when_llm_rejects_them(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "FST2025-5.pdf"
            pdf_path.write_text("placeholder", encoding="utf-8")
            service = CaseExtractionService(
                pdf_reader=FakePdfReader(),
                splitter=FakeSplitter(),
                llm_client=RejectingLlmClient(),
            )

            cases = service.extract_cases(pdf_path)

        self.assertEqual(1, len(cases))
        self.assertEqual("FST2025-5-case-01", cases[0].case_id)
        self.assertEqual("5月2日雷暴大风天气过程", cases[0].title)


if __name__ == "__main__":
    unittest.main()
