import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from backend.app.main import app, embedding_client, llm_client, rag_service
from backend.app.services.session_store import SessionStore


class StreamApiTests(unittest.TestCase):
    def test_session_title_preserves_unicode(self):
        from backend.app import main

        original_session_store = main.session_store
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                main.session_store = SessionStore(Path(tmp_dir) / "sessions.sqlite3")
                client = TestClient(app)

                response = client.post("/api/sessions", json={"title": "请分析1-3日雨雪"})

            self.assertEqual(200, response.status_code)
            self.assertEqual("请分析1-3日雨雪", response.json()["title"])
        finally:
            main.session_store = original_session_store

    def test_session_title_can_be_updated(self):
        from backend.app import main

        original_session_store = main.session_store
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                main.session_store = SessionStore(Path(tmp_dir) / "sessions.sqlite3")
                client = TestClient(app)
                session = client.post("/api/sessions", json={"title": "新会话"}).json()

                response = client.patch(
                    f"/api/sessions/{session['session_id']}",
                    json={"title": "14-15日暴雪"},
                )

            self.assertEqual(200, response.status_code)
            self.assertEqual("14-15日暴雪", response.json()["title"])
        finally:
            main.session_store = original_session_store

    def test_stream_query_returns_metadata_and_delta_events(self):
        original_embedding_key = embedding_client.api_key
        original_llm_key = llm_client.api_key
        original_answer_llm_key = rag_service.answer_generator.llm_client.api_key
        try:
            embedding_client.api_key = ""
            llm_client.api_key = ""
            rag_service.answer_generator.llm_client.api_key = ""
            client = TestClient(app)

            with client.stream(
                "POST",
                "/api/query/stream",
                json={"question": "请分析14-15日暴雪天气过程", "top_k": 5, "top_n": 3},
            ) as response:
                body = response.read().decode("utf-8")

            self.assertEqual(200, response.status_code)
            self.assertIn('"type": "metadata"', body)
            self.assertIn('"type": "delta"', body)
            self.assertIn('"type": "done"', body)
        finally:
            embedding_client.api_key = original_embedding_key
            llm_client.api_key = original_llm_key
            rag_service.answer_generator.llm_client.api_key = original_answer_llm_key

    def test_stream_query_records_messages_when_session_id_is_provided(self):
        from backend.app import main

        original_embedding_key = embedding_client.api_key
        original_llm_key = llm_client.api_key
        original_answer_llm_key = rag_service.answer_generator.llm_client.api_key
        original_session_store = main.session_store
        try:
            embedding_client.api_key = ""
            llm_client.api_key = ""
            rag_service.answer_generator.llm_client.api_key = ""
            with tempfile.TemporaryDirectory() as tmp_dir:
                main.session_store = SessionStore(Path(tmp_dir) / "sessions.sqlite3")
                client = TestClient(app)
                session = client.post("/api/sessions", json={"title": "stream demo"}).json()

                with client.stream(
                    "POST",
                    "/api/query/stream",
                    json={
                        "question": "stream question",
                        "top_k": 5,
                        "top_n": 3,
                        "session_id": session["session_id"],
                    },
                ) as response:
                    response.read()

                messages = client.get(f"/api/sessions/{session['session_id']}/messages").json()["messages"]

            self.assertEqual(200, response.status_code)
            self.assertEqual(["user", "assistant"], [message["role"] for message in messages])
            self.assertEqual("stream question", messages[0]["content"])
            self.assertTrue(messages[1]["content"])
        finally:
            embedding_client.api_key = original_embedding_key
            llm_client.api_key = original_llm_key
            rag_service.answer_generator.llm_client.api_key = original_answer_llm_key
            main.session_store = original_session_store


if __name__ == "__main__":
    unittest.main()
