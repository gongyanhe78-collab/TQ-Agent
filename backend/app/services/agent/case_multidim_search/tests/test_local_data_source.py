"""共享本地知识库适配层的回归测试。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.app.services.agent.Smart_Case_Match.infrastructure.data_store import LocalCaseDataStore
from backend.app.services.agent.case_multidim_search.integrations.local_stores import (
    LocalDocumentChunkStore,
    LocalImageEvidenceStore,
    LocalStandardCaseStore,
)
from backend.app.services.agent.case_multidim_search.retrieval.structured_retriever import StructuredCaseRetriever
from backend.app.services.agent.case_multidim_search.schemas import CaseSearchQuery


class LocalKnowledgeBaseStoreTests(unittest.TestCase):
    """验证多维检索通过既有 Store 接口读取共享本地资料。"""

    def test_local_stores_keep_case_chunk_and_image_links(self):
        """个例筛选、原文回查和历史图片路径回退均使用本地资料。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            image_path = data_dir / "document_images" / "weather-report" / "figure-001.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"png")
            (data_dir / "document_index").mkdir()
            # 三个 Agent 已统一使用主库 standard_cases1.json。
            (data_dir / "standard_cases1.json").write_text(
                json.dumps(
                    [
                        {
                            "case_id": "case-001",
                            "title": "5月太原暴雨过程",
                            "date_range": "2025年5月16日",
                            "disaster_types": ["暴雨"],
                            "affected_areas": ["大同"],
                            "source_pdf": "weather-report.pdf",
                            "source_chunk_ids": ["chunk-001"],
                            "evidence_image_ids": ["figure-001"],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (data_dir / "document_index" / "chunks.json").write_text(
                json.dumps(
                    [
                        {
                            "chunk_id": "chunk-001",
                            "chunk_no": 1,
                            "source_pdf": "weather-report.pdf",
                            "content": "大同出现暴雨，过程雨量达到 80 毫米。",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (data_dir / "image_metadata.json").write_text(
                json.dumps(
                    [
                        {
                            "image_id": "figure-001",
                            "source_pdf": "weather-report.pdf",
                            "page_no": 1,
                            "image_no": 1,
                            "image_path": "C:/legacy/document_images/weather-report/figure-001.png",
                            "extraction_type": "embedded",
                            "caption": "图1 大同降水实况",
                            "related_chunk_ids": ["chunk-001"],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            # 三类适配器必须共享同一份本地数据，不能退回到任何 HTTP 数据源。
            data_store = LocalCaseDataStore(data_dir)
            case_store = LocalStandardCaseStore(data_store)
            chunk_store = LocalDocumentChunkStore(data_store)
            image_store = LocalImageEvidenceStore(data_store)

            matches = StructuredCaseRetriever().search(
                case_store.list_cases(),
                CaseSearchQuery(months=[5], disaster_types=["暴雨"], areas=["晋北"]),
            )
            self.assertEqual(["case-001"], [item.case.case_id for item in matches])
            self.assertEqual("chunk-001", chunk_store.get_chunks(["chunk-001"])[0].chunk_id)
            image = image_store.get_image("figure-001")
            self.assertIsNotNone(image)
            self.assertEqual(image_path, image_store.resolve_image_path(image))
            # 临时资料只验证 JSON 元数据回退，避免 Windows 下 Chroma 的 SQLite 句柄阻止临时目录清理。


if __name__ == "__main__":
    unittest.main()
