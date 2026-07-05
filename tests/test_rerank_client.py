import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.models import ExtractedCase, RetrievalHit
from backend.app.services.rerank_client import DashScopeRerankClient, ModelFirstReranker
from backend.app.services.retrieval import KeywordReranker


def hit(case_id, title, score):
    return RetrievalHit(
        case=ExtractedCase(
            source_pdf="FST2025-3.pdf",
            case_id=case_id,
            case_no=1,
            title=title,
            date_range="1-3日",
            content=f"{title} 正文材料",
        ),
        score=score,
    )


class FakeHttpResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeHttpClient:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def post(self, url, headers, json, timeout):
        self.requests.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return FakeHttpResponse(self.payload)


class FailingRerankClient:
    def rerank(self, question, hits, top_n):
        raise RuntimeError("rerank unavailable")


class RerankClientTests(unittest.TestCase):
    def test_dashscope_rerank_orders_hits_by_model_scores(self):
        hits = [
            hit("case-a", "雨雪天气过程", 0.91),
            hit("case-b", "暴雪天气过程", 0.72),
            hit("case-c", "沙尘寒潮过程", 0.65),
        ]
        http_client = FakeHttpClient(
            {
                "output": {
                    "results": [
                        {"index": 2, "relevance_score": 0.88},
                        {"index": 0, "relevance_score": 0.44},
                    ]
                }
            }
        )
        reranker = DashScopeRerankClient(api_key="test-key", http_client=http_client)

        reranked = reranker.rerank("哪次过程最相似？", hits, top_n=2)

        self.assertEqual(["case-c", "case-a"], [item.case.case_id for item in reranked])
        self.assertEqual("qwen3-rerank", http_client.requests[0]["json"]["model"])
        self.assertEqual(2, http_client.requests[0]["json"]["parameters"]["top_n"])
        self.assertEqual(3, len(http_client.requests[0]["json"]["input"]["documents"]))

    def test_model_first_reranker_falls_back_to_keyword_reranker(self):
        hits = [
            hit("case-a", "雨雪天气过程", 0.91),
            hit("case-b", "暴雪天气过程", 0.72),
        ]
        reranker = ModelFirstReranker(
            model_reranker=FailingRerankClient(),
            fallback_reranker=KeywordReranker(),
        )

        reranked = reranker.rerank("暴雪", hits, top_n=1)

        self.assertEqual(["case-b"], [item.case.case_id for item in reranked])


if __name__ == "__main__":
    unittest.main()
