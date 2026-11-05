"""系统健康检查与统一知识库状态接口测试。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.routes import system


class _AvailableClient:
    """模拟已配置的云端模型客户端。"""

    def is_available(self) -> bool:
        return True


class _DocumentStore:
    """提供固定向量库状态，避免测试打开真实 ChromaDB。"""

    def collection_info(self) -> dict:
        return {"count": 317, "dimension": 1024, "source": "chroma"}


class _StandardCaseStore:
    """提供标准个例列表，用于验证个例数不再错误采用 chunk 数。"""

    def list_cases(self) -> list[dict]:
        return [{"case_id": f"case-{index:03d}"} for index in range(24)]


class SystemRouteTests(unittest.TestCase):
    """通过 HTTP 层验证两个系统状态接口。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.main = SimpleNamespace(
            document_store=_DocumentStore(),
            standard_case_store=_StandardCaseStore(),
            embedding_client=_AvailableClient(),
            llm_client=_AvailableClient(),
        )
        app = FastAPI()
        app.include_router(system.router)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_health_returns_vector_and_cloud_model_status(self) -> None:
        """健康接口应返回主向量库和云端模型可用性。"""
        with patch.object(system, "_main", return_value=self.main):
            response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["vector_keys"], 317)
        self.assertEqual(response.json()["vector_dimension"], 1024)
        self.assertTrue(response.json()["embedding_available"])
        self.assertTrue(response.json()["llm_available"])

    def test_knowledge_status_reads_manifest_and_counts_cases(self) -> None:
        """知识库状态应读取构建清单，并按标准个例数量统计。"""
        manifest = {"build_id": "build-001", "chunk_count": 317, "standard_case_count": 24}
        (self.data_dir / "knowledge_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        with (
            patch.object(system, "_main", return_value=self.main),
            patch.object(system.settings, "data_dir", self.data_dir),
            patch.object(system, "get_rerank_client", return_value=_AvailableClient()),
        ):
            response = self.client.get("/api/knowledge/status")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["case_count"], 24)
        self.assertEqual(payload["vector_count"], 317)
        self.assertTrue(payload["rerank_available"])
        self.assertEqual(payload["build"]["status"], "ready")
        self.assertEqual(payload["build"]["build_id"], "build-001")

    def test_knowledge_status_reports_missing_manifest(self) -> None:
        """构建清单缺失时接口应返回诊断信息而不是抛出异常。"""
        with (
            patch.object(system, "_main", return_value=self.main),
            patch.object(system.settings, "data_dir", self.data_dir),
            patch.object(system, "get_rerank_client", return_value=_AvailableClient()),
        ):
            response = self.client.get("/api/knowledge/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["build"]["status"], "missing")


if __name__ == "__main__":
    unittest.main()
