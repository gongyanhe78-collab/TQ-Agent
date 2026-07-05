import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.image_extraction import ImageEvidenceStore, PdfImageEvidenceExtractor


class PdfImageEvidenceExtractorTests(unittest.TestCase):
    def test_dual_track_extraction_writes_metadata_and_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pdf_path = root / "FST2025-3.pdf"
            pdf_path.write_bytes(b"%PDF placeholder")
            output_dir = root / "images"

            extractor = PdfImageEvidenceExtractor(
                output_dir=output_dir,
                embedded_image_reader=lambda path: [
                    {
                        "page_no": 2,
                        "image_no": 1,
                        "extension": ".png",
                        "data": b"embedded-image",
                        "width": 640,
                        "height": 480,
                    }
                ],
                page_snapshot_renderer=lambda path: [
                    {
                        "page_no": 1,
                        "extension": ".png",
                        "data": b"page-snapshot",
                        "width": 1200,
                        "height": 1600,
                    }
                ],
                page_text_reader=lambda path: {
                    1: "图1 3月1日降水量分布图\n正文说明",
                    2: "图2 3月1日雷达回波图\n正文说明",
                },
                chunk_lookup=lambda source_pdf, page_no: ["FST2025-3-chunk-001"] if page_no == 1 else [],
            )

            records = extractor.extract_pdf(pdf_path)

            self.assertEqual(2, len(records))
            self.assertEqual({"embedded", "page_snapshot"}, {record.extraction_type for record in records})
            self.assertTrue(all(Path(record.image_path).exists() for record in records))
            snapshot = next(record for record in records if record.extraction_type == "page_snapshot")
            self.assertEqual("图1 3月1日降水量分布图", snapshot.caption)
            self.assertEqual(["FST2025-3-chunk-001"], snapshot.related_chunk_ids)
            embedded = next(record for record in records if record.extraction_type == "embedded")
            self.assertEqual("FST2025-3.pdf", embedded.source_pdf)
            self.assertEqual(2, embedded.page_no)

    def test_build_index_persists_metadata_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            resource_dir = root / "resource"
            resource_dir.mkdir()
            (resource_dir / "demo.pdf").write_bytes(b"%PDF placeholder")
            metadata_path = root / "image_metadata.json"

            extractor = PdfImageEvidenceExtractor(
                output_dir=root / "images",
                metadata_path=metadata_path,
                embedded_image_reader=lambda path: [],
                page_snapshot_renderer=lambda path: [
                    {
                        "page_no": 1,
                        "extension": ".png",
                        "data": b"snapshot",
                        "width": 800,
                        "height": 1000,
                    }
                ],
                page_text_reader=lambda path: {1: "图1 测试图"},
            )

            records = extractor.build_index([resource_dir / "demo.pdf"])

            self.assertEqual(1, len(records))
            self.assertTrue(metadata_path.exists())
            self.assertIn("demo-page-001-snapshot", metadata_path.read_text(encoding="utf-8"))

    def test_document_hit_response_includes_related_image_evidence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            from backend.app import main
            from backend.app.models import DocumentChunk, DocumentRetrievalHit
            from backend.app.services.image_extraction import ImageEvidenceStore

            metadata_path = Path(tmp_dir) / "image_metadata.json"
            metadata_path.write_text(
                """
[
  {
    "image_id": "demo-page-001-image-001",
    "source_pdf": "demo.pdf",
    "page_no": 1,
    "image_no": 1,
    "image_path": "D:/company/risk-master/data/document_images/demo/demo-page-001-image-001.png",
    "extraction_type": "embedded",
    "nearby_text": "图1 测试图",
    "caption": "图1 测试图",
    "related_chunk_ids": ["demo-chunk-001"],
    "width": 640,
    "height": 480
  }
]
""".strip(),
                encoding="utf-8",
            )
            hit = DocumentRetrievalHit(
                chunk=DocumentChunk(
                    source_pdf="demo.pdf",
                    chunk_id="demo-chunk-001",
                    chunk_no=1,
                    content="测试片段",
                ),
                score=0.8,
            )

            with patch.object(main, "image_evidence_store", ImageEvidenceStore(metadata_path)):
                response = main._document_hit_to_response(hit)

        self.assertEqual("demo-chunk-001", response["chunk"]["chunk_id"])
        self.assertEqual(1, len(response["images"]))
        self.assertEqual("图1 测试图", response["images"][0]["caption"])
        self.assertEqual("/api/image-evidence/demo-page-001-image-001", response["images"][0]["url"])

    def test_image_evidence_endpoint_serves_registered_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            from backend.app import main
            from backend.app.config import settings
            from backend.app.services.image_extraction import ImageEvidenceStore

            root = Path(tmp_dir)
            image_dir = root / "document_images"
            image_dir.mkdir()
            image_path = image_dir / "demo.png"
            image_path.write_bytes(b"fake-image")
            metadata_path = root / "image_metadata.json"
            image_path_value = str(image_path).replace("\\", "/")
            metadata_path.write_text(
                f"""
[
  {{
    "image_id": "demo-image",
    "source_pdf": "demo.pdf",
    "page_no": 1,
    "image_no": 1,
    "image_path": "{image_path_value}",
    "extraction_type": "embedded",
    "nearby_text": "图1 测试图",
    "caption": "图1 测试图",
    "related_chunk_ids": ["demo-chunk-001"],
    "width": 640,
    "height": 480
  }}
]
""".strip(),
                encoding="utf-8",
            )

            with (
                patch.object(main, "image_evidence_store", ImageEvidenceStore(metadata_path)),
                patch.object(settings, "document_images_dir", image_dir),
            ):
                response = TestClient(main.app).get("/api/image-evidence/demo-image")

            self.assertEqual(200, response.status_code)
            self.assertEqual(b"fake-image", response.content)

    def test_image_store_filters_non_browser_formats(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path = Path(tmp_dir) / "image_metadata.json"
            metadata_path.write_text(
                """
[
  {
    "image_id": "demo-page-001-image-001",
    "source_pdf": "demo.pdf",
    "page_no": 1,
    "image_no": 1,
    "image_path": "D:/company/risk-master/data/document_images/demo/demo-page-001-image-001.jp2",
    "extraction_type": "embedded",
    "nearby_text": "图1 测试图",
    "caption": "图1 测试图",
    "related_chunk_ids": ["demo-chunk-001"],
    "width": 640,
    "height": 480
  },
  {
    "image_id": "demo-page-001-snapshot",
    "source_pdf": "demo.pdf",
    "page_no": 1,
    "image_no": 1,
    "image_path": "D:/company/risk-master/data/document_images/demo/demo-page-001-snapshot.png",
    "extraction_type": "page_snapshot",
    "nearby_text": "图1 测试图",
    "caption": "图1 测试图",
    "related_chunk_ids": ["demo-chunk-001"],
    "width": 1200,
    "height": 1600
  }
]
""".strip(),
                encoding="utf-8",
            )

            images = ImageEvidenceStore(metadata_path).list_by_chunk_id("demo-chunk-001")

        self.assertEqual(["demo-page-001-snapshot"], [image.image_id for image in images])

    def test_document_context_includes_image_captions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            from backend.app import main
            from backend.app.models import DocumentChunk, DocumentRetrievalHit

            metadata_path = Path(tmp_dir) / "image_metadata.json"
            metadata_path.write_text(
                """
[
  {
    "image_id": "demo-page-001-snapshot",
    "source_pdf": "demo.pdf",
    "page_no": 1,
    "image_no": 1,
    "image_path": "D:/company/risk-master/data/document_images/demo/demo-page-001-snapshot.png",
    "extraction_type": "page_snapshot",
    "nearby_text": "图1 雷达回波图",
    "caption": "图1 雷达回波图",
    "related_chunk_ids": ["demo-chunk-001"],
    "width": 1200,
    "height": 1600
  }
]
""".strip(),
                encoding="utf-8",
            )
            hit = DocumentRetrievalHit(
                chunk=DocumentChunk(
                    source_pdf="demo.pdf",
                    chunk_id="demo-chunk-001",
                    chunk_no=1,
                    content="测试片段",
                ),
                score=0.8,
            )

            with patch.object(main, "image_evidence_store", ImageEvidenceStore(metadata_path)):
                blocks = main._context_blocks_from_document_hits([hit])

        self.assertIn("相关图像证据", blocks[0])
        self.assertIn("图1 雷达回波图", blocks[0])


if __name__ == "__main__":
    unittest.main()
