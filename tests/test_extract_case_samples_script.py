import tempfile
import unittest
from pathlib import Path

from scripts.extract_case_samples import ExtractionReportItem, add_qa_supplements


class ExtractCaseSamplesScriptTests(unittest.TestCase):
    def test_adds_only_qa_samples_exceeding_pdf_case_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "samples"
            output_dir.mkdir()
            qa_path = Path(tmp_dir) / "weather_qa_results.json"
            qa_path.write_text(
                """
[
  {
    "pdf_filename": "FST2025-3.pdf",
    "samples": [
      {"question": "q1", "answer": "a1"},
      {"question": "q2", "answer": "a2"},
      {"question": "q3", "answer": "a3"},
      {"question": "summary q", "answer": "summary a"}
    ]
  }
]
""".strip(),
                encoding="utf-8",
            )
            case_report = [
                ExtractionReportItem("FST2025-3.pdf", "case 1", "case-01", "case"),
                ExtractionReportItem("FST2025-3.pdf", "case 2", "case-02", "case"),
                ExtractionReportItem("FST2025-3.pdf", "case 3", "case-03", "case"),
            ]

            supplements = add_qa_supplements(qa_path, case_report, output_dir)

            self.assertEqual(1, len(supplements))
            self.assertEqual("FST2025-3-qa-summary-01", supplements[0].sample_id)
            written = (output_dir / "FST2025-3-qa-summary-01.txt").read_text(encoding="utf-8")
            self.assertIn("sample_type: qa_supplement", written)
            self.assertIn("question: summary q", written)


if __name__ == "__main__":
    unittest.main()
