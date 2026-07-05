import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.llm_client import DashScopeChatClient


class FakeCompletions:
    def create(self, **kwargs):
        return iter(
            [
                types.SimpleNamespace(choices=[]),
                types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            delta=types.SimpleNamespace(content="第一段")
                        )
                    ]
                ),
                types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            delta=types.SimpleNamespace(content=None)
                        )
                    ]
                ),
                types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            delta=types.SimpleNamespace(content="第二段")
                        )
                    ]
                ),
            ]
        )


class FakeOpenAI:
    def __init__(self, api_key, base_url):
        self.chat = types.SimpleNamespace(completions=FakeCompletions())


class LlmClientTests(unittest.TestCase):
    def test_stream_answer_skips_empty_choice_chunks(self):
        fake_openai_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
        with patch.dict(sys.modules, {"openai": fake_openai_module}):
            client = DashScopeChatClient(api_key="test-key")

            chunks = list(client.stream_answer_with_context("问题", ["材料"]))

        self.assertEqual(["第一段", "第二段"], chunks)


if __name__ == "__main__":
    unittest.main()
