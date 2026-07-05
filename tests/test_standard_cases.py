import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.models import DocumentChunk, StandardCase
from backend.app.services.image_extraction import ImageEvidenceStore
from backend.app.services.standard_case_builder import StandardCaseBuilder
from backend.app.services.standard_case_store import JsonStandardCaseStore


class StandardCaseLayerTests(unittest.TestCase):
    def test_rule_builder_creates_standard_case_with_images(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
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
    "nearby_text": "图1 雷暴大风分布图",
    "caption": "图1 雷暴大风分布图",
    "related_chunk_ids": ["demo-chunk-001"],
    "width": 1200,
    "height": 1600
  }
]
""".strip(),
                encoding="utf-8",
            )
            chunks = [
                DocumentChunk(
                    source_pdf="demo.pdf",
                    chunk_id="demo-chunk-001",
                    chunk_no=1,
                    content=(
                        "一、5月2日雷暴大风天气过程\n"
                        "天气实况：沈阳、大连出现雷暴大风和短时强降水。"
                        "预报提示：关注强对流落区和8级以上大风。"
                    ),
                )
            ]

            builder = StandardCaseBuilder(
                image_store=ImageEvidenceStore(metadata_path),
                llm_client=None,
                use_llm=False,
            )
            cases = builder.build(chunks)

        self.assertEqual(1, len(cases))
        case = cases[0]
        self.assertEqual("demo-std-case-001", case.case_id)
        self.assertEqual("5月2日雷暴大风天气过程", case.title)
        self.assertEqual("5月2日", case.date_range)
        self.assertIn("雷暴", case.disaster_types)
        self.assertIn("大风", case.disaster_types)
        self.assertIn("沈阳", case.affected_areas)
        self.assertEqual(["demo-page-001-snapshot"], case.evidence_image_ids)
        self.assertEqual(["demo-chunk-001"], case.source_chunk_ids)

    def test_store_searches_standard_cases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonStandardCaseStore(Path(tmp_dir) / "standard_cases.json")
            store.replace_cases(
                [
                    StandardCase(
                        case_id="case-001",
                        title="5月2日雷暴大风天气过程",
                        date_range="5月2日",
                        disaster_types=["雷暴", "大风"],
                        affected_areas=["沈阳"],
                        source_pdf="demo.pdf",
                    ),
                    StandardCase(
                        case_id="case-002",
                        title="6月1日暴雨天气过程",
                        date_range="6月1日",
                        disaster_types=["暴雨"],
                        affected_areas=["大连"],
                        source_pdf="demo.pdf",
                    ),
                ]
            )

            results = store.search_cases(date="5月2日", disaster_type="大风", area="沈阳")

        self.assertEqual(["case-001"], [case.case_id for case in results])

    def test_standard_case_api_lists_and_gets_cases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            from backend.app import main

            metadata_path = Path(tmp_dir) / "image_metadata.json"
            metadata_path.write_text(
                """
[
  {
    "image_id": "radar-image",
    "source_pdf": "demo.pdf",
    "page_no": 1,
    "image_no": 1,
    "image_path": "D:/company/risk-master/data/document_images/demo/radar-image.png",
    "extraction_type": "embedded",
    "nearby_text": "图1 雷达回波图",
    "caption": "图1 雷达回波图",
    "related_chunk_ids": ["demo-chunk-001"],
    "width": 640,
    "height": 480
  }
]
""".strip(),
                encoding="utf-8",
            )
            store = JsonStandardCaseStore(Path(tmp_dir) / "standard_cases.json")
            store.replace_cases(
                [
                    StandardCase(
                        case_id="case-001",
                        title="5月2日雷暴大风天气过程",
                        date_range="5月2日",
                        disaster_types=["雷暴", "大风"],
                        affected_areas=["山西北部"],
                        source_pdf="demo.pdf",
                        evidence_image_ids=["radar-image"],
                    )
                ]
            )

            with (
                patch.object(main, "standard_case_store", store),
                patch.object(main, "image_evidence_store", ImageEvidenceStore(metadata_path)),
            ):
                client = TestClient(main.app)
                list_response = client.get("/api/standard-cases?disaster_type=大风")
                detail_response = client.get("/api/standard-cases/case-001")
                nl_response = client.get(
                    "/api/standard-cases",
                    params={"q": "2025年5月山西北部雷暴大风，有雷达图的个例"},
                )

        self.assertEqual(200, list_response.status_code)
        self.assertEqual(1, list_response.json()["count"])
        self.assertEqual("case-001", list_response.json()["cases"][0]["case_id"])
        self.assertEqual(200, detail_response.status_code)
        self.assertEqual("5月2日雷暴大风天气过程", detail_response.json()["title"])
        self.assertEqual(200, nl_response.status_code)
        self.assertEqual(1, nl_response.json()["count"])
        image = nl_response.json()["cases"][0]["evidence_images"][0]
        self.assertEqual("radar", image["image_type"])
        self.assertEqual("雷达图", image["data_category"])

    def test_similar_case_api_scores_structured_fields_and_images(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            from backend.app import main

            metadata_path = Path(tmp_dir) / "image_metadata.json"
            metadata_path.write_text(
                """
[
  {
    "image_id": "radar-image",
    "source_pdf": "demo.pdf",
    "page_no": 1,
    "image_no": 1,
    "image_path": "D:/company/risk-master/data/document_images/demo/radar-image.png",
    "extraction_type": "embedded",
    "nearby_text": "图1 雷达回波图",
    "caption": "图1 雷达回波图",
    "related_chunk_ids": ["demo-chunk-001"],
    "width": 640,
    "height": 480
  }
]
""".strip(),
                encoding="utf-8",
            )
            store = JsonStandardCaseStore(Path(tmp_dir) / "standard_cases.json")
            store.replace_cases(
                [
                    StandardCase(
                        case_id="case-001",
                        title="5月2日雷暴大风天气过程",
                        date_range="5月2日",
                        disaster_types=["雷暴", "大风", "强对流"],
                        affected_areas=["山西北部", "大同"],
                        source_pdf="demo.pdf",
                        summary="山西北部出现雷暴大风。",
                        forecast_focus="关注雷达回波发展和8级以上阵风。",
                        evidence_image_ids=["radar-image"],
                    ),
                    StandardCase(
                        case_id="case-002",
                        title="8月1日高温天气过程",
                        date_range="8月1日",
                        disaster_types=["高温"],
                        affected_areas=["山西南部"],
                        source_pdf="demo.pdf",
                    ),
                ]
            )

            with (
                patch.object(main, "standard_case_store", store),
                patch.object(main, "image_evidence_store", ImageEvidenceStore(metadata_path)),
            ):
                main.similar_case_matcher.image_store = main.image_evidence_store
                client = TestClient(main.app)
                response = client.get(
                    "/api/standard-cases/similar",
                    params={"q": "2025年5月山西北部雷暴大风，有雷达图", "top_n": 3},
                )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertGreaterEqual(payload["count"], 1)
        match = payload["matches"][0]
        self.assertEqual("case-001", match["case_id"])
        self.assertGreater(match["similarity_score"], 0.7)
        self.assertIn("disaster", match["score_breakdown"])
        self.assertTrue(match["match_reasons"])
        self.assertTrue(match["forecast_tips"])
        self.assertEqual("radar", match["evidence_images"][0]["image_type"])


if __name__ == "__main__":
    unittest.main()
