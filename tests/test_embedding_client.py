import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.embedding_client import DashScopeEmbeddingClient


class FakeEmbeddingResponse:
    def __init__(self, count):
        self.data = [
            types.SimpleNamespace(embedding=[float(index)])
            for index in range(count)
        ]


class FakeEmbeddings:
    calls = []

    def create(self, **kwargs):
        self.calls.append(list(kwargs["input"]))
        return FakeEmbeddingResponse(len(kwargs["input"]))


class FakeOpenAI:
    def __init__(self, api_key, base_url):
        self.embeddings = FakeEmbeddings()


class EmbeddingClientTests(unittest.TestCase):
    def test_embed_documents_batches_external_requests(self):
        FakeEmbeddings.calls = []
        fake_openai_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
        with patch.dict(sys.modules, {"openai": fake_openai_module}):
            client = DashScopeEmbeddingClient(api_key="test-key", batch_size=3)

            embeddings = client.embed_documents([f"text-{index}" for index in range(8)])

        self.assertEqual(8, len(embeddings))
        self.assertEqual(
            [["text-0", "text-1", "text-2"], ["text-3", "text-4", "text-5"], ["text-6", "text-7"]],
            FakeEmbeddings.calls,
        )


if __name__ == "__main__":
    unittest.main()
