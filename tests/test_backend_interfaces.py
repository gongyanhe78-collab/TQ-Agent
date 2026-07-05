import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.main import app
from backend.app.models import DocumentChunk, DocumentRetrievalHit, ExtractedCase, RagAnswer


class BackendInterfaceTests(unittest.TestCase):
    def test_vector_keys_endpoint_reads_keys_without_ingest(self):
        class FakeCaseStore:
            def list_case_ids(self):
                return ["FST2025-5-case-01", "FST2025-5-case-02"]

            def collection_info(self):
                return {"count": 2, "dimension": 1024, "source": "fake"}

        from backend.app import main

        original_case_store = main.case_store
        main.case_store = FakeCaseStore()
        try:
            response = TestClient(app).get("/api/vectors/keys")
        finally:
            main.case_store = original_case_store

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(["FST2025-5-case-01", "FST2025-5-case-02"], payload["case_ids"])
        self.assertEqual(2, payload["count"])
        self.assertEqual(1024, payload["dimension"])

    def test_delete_vectors_accepts_batch_keys(self):
        class FakeCaseStore:
            def delete_cases(self, keys):
                return {
                    "requested_keys": keys,
                    "deleted_keys": ["FST2025-3-case-01"],
                    "missing_keys": ["missing-case"],
                    "deleted": 1,
                }

            def collection_info(self):
                return {"count": 23, "dimension": 1024, "source": "fake"}

        from backend.app import main

        original_case_store = main.case_store
        main.case_store = FakeCaseStore()
        try:
            response = TestClient(app).post(
                "/api/vectors/delete",
                json={"keys": ["FST2025-3-case-01", "missing-case"]},
            )
        finally:
            main.case_store = original_case_store

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(["FST2025-3-case-01"], payload["deleted_keys"])
        self.assertEqual(["missing-case"], payload["missing_keys"])
        self.assertEqual(23, payload["vector_count"])

    def test_batch_material_upload_saves_multiple_pdfs_and_runs_document_indexing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_resource_dir = settings.resource_dir
            settings.resource_dir = Path(tmp_dir)
            try:
                with patch(
                    "backend.app.main._index_document_pdfs",
                    return_value={
                        "processed_pdfs": ["FST2025-3.pdf", "FST2025-4.pdf"],
                        "indexed": 2,
                        "chunk_ids": ["FST2025-3-chunk-001", "FST2025-4-chunk-001"],
                    },
                ) as index_documents:
                    response = TestClient(app).post(
                        "/api/materials/batch",
                        files=[
                            ("files", ("FST2025-3.pdf", b"%PDF first", "application/pdf")),
                            ("files", ("FST2025-4.pdf", b"%PDF second", "application/pdf")),
                        ],
                    )

                self.assertEqual(200, response.status_code)
                self.assertEqual(["FST2025-3.pdf", "FST2025-4.pdf"], response.json()["uploaded_files"])
                self.assertEqual(b"%PDF first", (Path(tmp_dir) / "FST2025-3.pdf").read_bytes())
                self.assertEqual(b"%PDF second", (Path(tmp_dir) / "FST2025-4.pdf").read_bytes())
                self.assertEqual(1, index_documents.call_count)
                indexed_paths = [Path(path).name for path in index_documents.call_args.args[0]]
                self.assertEqual(["FST2025-3.pdf", "FST2025-4.pdf"], indexed_paths)
            finally:
                settings.resource_dir = original_resource_dir

    def test_refresh_processes_only_unprocessed_resource_pdfs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_resource_dir = settings.resource_dir
            settings.resource_dir = Path(tmp_dir)
            processed = Path(tmp_dir) / "old.pdf"
            fresh = Path(tmp_dir) / "new.pdf"
            processed.write_bytes(b"%PDF old")
            fresh.write_bytes(b"%PDF new")
            try:
                with patch("backend.app.main.processed_file_store.is_processed") as is_processed:
                    is_processed.side_effect = lambda path: Path(path).name == "old.pdf"
                    with patch(
                        "backend.app.main._process_pdf_paths_incrementally",
                        return_value={"processed_pdfs": ["new.pdf"], "indexed": 1},
                    ) as ingest:
                        response = TestClient(app).post("/api/refresh")

                self.assertEqual(200, response.status_code)
                self.assertEqual(["new.pdf"], response.json()["processed_pdfs"])
                processed_paths = [Path(path).name for path in ingest.call_args.args[0]]
                self.assertEqual(["new.pdf"], processed_paths)
            finally:
                settings.resource_dir = original_resource_dir

    def test_knowledge_status_reports_models_and_build_state(self):
        with patch("backend.app.main.build_status_store.get_status", return_value={"status": "idle", "last_error": None}):
            response = TestClient(app).get("/api/knowledge/status")

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(settings.embedding_model, payload["embedding_model"])
        self.assertEqual(settings.rerank_model, payload["rerank_model"])
        self.assertIn("vector_count", payload)
        self.assertEqual("idle", payload["build"]["status"])

    def test_agent_query_returns_structured_answer_and_records_session_messages(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            from backend.app import main
            from backend.app.services.session_store import SessionStore

            original_session_store = main.session_store
            main.session_store = SessionStore(Path(tmp_dir) / "sessions.sqlite3")
            try:
                session_response = TestClient(app).post("/api/sessions", json={"title": "demo"})
                session_id = session_response.json()["session_id"]

                with patch(
                    "backend.app.main._ask_document_chunks",
                    return_value={
                        "question": "question",
                        "answer": "answer",
                        "retrieval_mode": "document_vector",
                        "llm_used": True,
                        "llm_status": "called",
                        "hits": [
                            DocumentRetrievalHit(
                                chunk=DocumentChunk(
                                    source_pdf="FST2025-3.pdf",
                                    chunk_id="FST2025-3-chunk-001",
                                    chunk_no=1,
                                    content="chunk evidence",
                                ),
                                score=0.9,
                            )
                        ],
                    },
                ):
                    response = TestClient(app).post(
                        "/api/agent/query",
                        json={"question": "question", "session_id": session_id, "return_context": True},
                    )

                self.assertEqual(200, response.status_code)
                payload = response.json()
                self.assertEqual("answer", payload["answer"])
                self.assertEqual(session_id, payload["session_id"])
                self.assertEqual(settings.rerank_model, payload["retrieval"]["rerank_model"])
                self.assertEqual("FST2025-3-chunk-001", payload["hits"][0]["chunk"]["chunk_id"])

                messages_response = TestClient(app).get(f"/api/sessions/{session_id}/messages")
                roles = [item["role"] for item in messages_response.json()["messages"]]
                self.assertEqual(["user", "assistant"], roles)
            finally:
                main.session_store = original_session_store


if __name__ == "__main__":
    unittest.main()
