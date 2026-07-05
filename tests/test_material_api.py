import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.config import settings


class MaterialApiTests(unittest.TestCase):
    def test_index_reads_extracted_samples_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_samples_dir = settings.samples_dir
            settings.samples_dir = Path(tmp_dir)
            sample_path = settings.samples_dir / "FST2025-3-qa-summary-01.txt"
            sample_path.write_text(
                "\n".join(
                    [
                        "sample_type: qa_supplement",
                        "source_pdf: FST2025-3.pdf",
                        "case_id: FST2025-3-qa-summary-01",
                        "title: 综合问答样例",
                        "date_range: ",
                        "",
                        "question: q",
                        "",
                        "answer: a",
                    ]
                ),
                encoding="utf-8",
            )
            try:
                from backend.app.main import _case_from_txt

                case = _case_from_txt(sample_path)
            finally:
                settings.samples_dir = original_samples_dir

        self.assertEqual("FST2025-3-qa-summary-01", case.case_id)
        self.assertEqual(1, case.case_no)
        self.assertEqual("综合问答样例", case.title)

    def test_upload_material_saves_pdf_and_indexes_document_chunk_library(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_resource_dir = settings.resource_dir
            settings.resource_dir = Path(tmp_dir)
            try:
                with patch(
                    "backend.app.main._index_document_pdfs",
                    return_value={"processed_pdfs": ["FST2025-3.pdf"], "indexed": 1, "chunk_ids": ["FST2025-3-chunk-001"]},
                ):
                    client = TestClient(app)
                    response = client.post(
                        "/api/materials",
                        content=b"%PDF-1.4 demo",
                        headers={"X-Filename": "FST2025-3.pdf"},
                    )

                self.assertEqual(200, response.status_code)
                self.assertEqual("FST2025-3.pdf", response.json()["uploaded_file"])
                self.assertEqual(b"%PDF-1.4 demo", (Path(tmp_dir) / "FST2025-3.pdf").read_bytes())
            finally:
                settings.resource_dir = original_resource_dir

    def test_upload_material_rejects_non_pdf(self):
        client = TestClient(app)
        response = client.post(
            "/api/materials",
            content=b"not pdf",
            headers={"X-Filename": "notes.txt"},
        )

        self.assertEqual(400, response.status_code)


if __name__ == "__main__":
    unittest.main()
