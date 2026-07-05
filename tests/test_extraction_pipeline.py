import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.models import ExtractedCase
from backend.app.services.extraction_pipeline import ExtractionPipeline


class FakeCaseExtractor:
    def extract_cases(self, pdf_path):
        return [
            ExtractedCase(
                source_pdf=pdf_path.name,
                case_id="FST2025-3-case-01",
                case_no=1,
                title="1-3日雨雪天气过程",
                date_range="1-3日",
                content="3月1日至3日全省出现雨雪天气过程。",
            )
        ]


class ExtractionPipelineTests(unittest.TestCase):
    def test_pipeline_writes_txt_with_lightweight_header(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "FST2025-3.pdf"
            pdf_path.write_text("placeholder", encoding="utf-8")
            output_dir = Path(tmp_dir) / "cases"

            pipeline = ExtractionPipeline(
                extractor=FakeCaseExtractor(),
                output_dir=output_dir,
            )

            written_files = pipeline.process_pdf(pdf_path)

            self.assertEqual(1, len(written_files))
            written_text = written_files[0].read_text(encoding="utf-8")
            self.assertIn("source_pdf: FST2025-3.pdf", written_text)
            self.assertIn("case_id: FST2025-3-case-01", written_text)
            self.assertIn("title: 1-3日雨雪天气过程", written_text)
            self.assertIn("date_range: 1-3日", written_text)
            self.assertTrue(written_text.rstrip().endswith("3月1日至3日全省出现雨雪天气过程。"))


if __name__ == "__main__":
    unittest.main()
