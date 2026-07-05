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
from backend.app.models import DocumentChunk
from backend.app.services.document_indexing import DocumentChunkIndexer
from backend.app.services.document_store import JsonDocumentChunkStore


class FakeEmbeddingClient:
    def embed_documents(self, texts):
        return [[float(index), 1.0] for index, _ in enumerate(texts, start=1)]


class DocumentChunkIndexerTests(unittest.TestCase):
    def test_builds_file_based_chunk_keys_and_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "FST2025-3.pdf"
            pdf_path.write_bytes(b"%PDF demo")

            indexer = DocumentChunkIndexer(
                embedding_client=FakeEmbeddingClient(),
                document_loader=lambda path: "第一段天气过程。\n\n第二段暴雪过程。",
                text_splitter=lambda text: ["第一段天气过程。", "第二段暴雪过程。"],
            )

            chunks = indexer.index_pdf(pdf_path)

        self.assertEqual(["FST2025-3-chunk-001", "FST2025-3-chunk-002"], [chunk.chunk_id for chunk in chunks])
        self.assertEqual("FST2025-3.pdf", chunks[0].source_pdf)
        self.assertEqual(1, chunks[0].chunk_no)
        self.assertEqual([1.0, 1.0], chunks[0].embedding)


class JsonDocumentChunkStoreTests(unittest.TestCase):
    def test_store_round_trip_lists_document_chunk_keys(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonDocumentChunkStore(Path(tmp_dir) / "chunks.json")
            store.upsert_chunks(
                [
                    DocumentChunk(
                        source_pdf="FST2025-3.pdf",
                        chunk_id="FST2025-3-chunk-001",
                        chunk_no=1,
                        content="chunk content",
                        embedding=[0.1, 0.2],
                    )
                ]
            )

            self.assertEqual(["FST2025-3-chunk-001"], store.list_chunk_ids())
            self.assertEqual("chunk content", store.get_chunk("FST2025-3-chunk-001").content)

    def test_store_queries_document_chunks_by_cosine_similarity(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonDocumentChunkStore(Path(tmp_dir) / "chunks.json")
            store.upsert_chunks(
                [
                    DocumentChunk(
                        source_pdf="FST2025-3.pdf",
                        chunk_id="FST2025-3-chunk-001",
                        chunk_no=1,
                        content="雨雪过程",
                        embedding=[1.0, 0.0],
                    ),
                    DocumentChunk(
                        source_pdf="FST2025-4.pdf",
                        chunk_id="FST2025-4-chunk-001",
                        chunk_no=1,
                        content="暴雪过程",
                        embedding=[0.0, 1.0],
                    ),
                ]
            )

            hits = store.query([0.0, 1.0], top_k=1)

        self.assertEqual("FST2025-4-chunk-001", hits[0].chunk.chunk_id)
        self.assertGreater(hits[0].score, 0.99)


class DocumentIndexApiTests(unittest.TestCase):
    def test_document_keys_endpoint_reads_independent_document_store(self):
        class FakeDocumentStore:
            def list_chunk_ids(self):
                return ["FST2025-3-chunk-001"]

            def collection_info(self):
                return {"count": 1, "dimension": 1024, "source": "fake_document_store"}

        from backend.app import main

        original_store = main.document_store
        main.document_store = FakeDocumentStore()
        try:
            response = TestClient(app).get("/api/documents/keys")
        finally:
            main.document_store = original_store

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(["FST2025-3-chunk-001"], payload["chunk_ids"])
        self.assertEqual(1, payload["count"])
        self.assertEqual("fake_document_store", payload["source"])

    def test_document_chunk_detail_endpoint_returns_chunk_content(self):
        class FakeDocumentStore:
            def get_chunk(self, chunk_id):
                if chunk_id != "FST2025-3-chunk-001":
                    return None
                return DocumentChunk(
                    source_pdf="FST2025-3.pdf",
                    chunk_id=chunk_id,
                    chunk_no=1,
                    content="证据片段正文",
                    file_path="resource/FST2025-3.pdf",
                    embedding=[0.1, 0.2],
                )

        from backend.app import main

        original_store = main.document_store
        main.document_store = FakeDocumentStore()
        try:
            response = TestClient(app).get("/api/documents/chunks/FST2025-3-chunk-001")
        finally:
            main.document_store = original_store

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("FST2025-3-chunk-001", payload["chunk_id"])
        self.assertEqual("证据片段正文", payload["content"])

    def test_document_index_endpoint_indexes_resource_pdfs_into_document_store(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            from backend.app import main
            from backend.app.config import settings

            original_resource_dir = settings.resource_dir
            settings.resource_dir = Path(tmp_dir)
            (settings.resource_dir / "FST2025-3.pdf").write_bytes(b"%PDF demo")
            try:
                with patch(
                    "backend.app.main._index_document_pdfs",
                    return_value={
                        "processed_pdfs": ["FST2025-3.pdf"],
                        "indexed": 2,
                        "chunk_ids": ["FST2025-3-chunk-001", "FST2025-3-chunk-002"],
                        "vector_count": 2,
                    },
                ) as index_documents:
                    response = TestClient(app).post("/api/documents/index")
            finally:
                settings.resource_dir = original_resource_dir

        self.assertEqual(200, response.status_code)
        self.assertEqual(1, index_documents.call_count)
        self.assertEqual(2, response.json()["indexed"])


if __name__ == "__main__":
    unittest.main()
