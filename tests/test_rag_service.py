import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.models import ExtractedCase, RetrievalHit
from backend.app.services.retrieval import RagService


class FakeEmbeddingClient:
    def __init__(self):
        self.queries = []

    def is_available(self):
        return True

    def embed_query(self, text):
        self.queries.append(text)
        if "暴雪" in text:
            return [0.0, 1.0]
        return [1.0, 0.0]


class FakeStore:
    def __init__(self):
        self.query_calls = []

    def query(self, embedding, top_k):
        self.query_calls.append((embedding, top_k))
        return [
            RetrievalHit(
                case=ExtractedCase(
                    source_pdf="FST2025-3.pdf",
                    case_id="FST2025-3-case-01",
                    case_no=1,
                    title="1-3日雨雪天气过程",
                    date_range="1-3日",
                    content="全省出现雨雪天气过程。",
                ),
                score=0.55,
            ),
            RetrievalHit(
                case=ExtractedCase(
                    source_pdf="FST2025-3.pdf",
                    case_id="FST2025-3-case-02",
                    case_no=2,
                    title="14-15日暴雪天气过程",
                    date_range="14-15日",
                    content="北部出现暴雪天气过程。",
                ),
                score=0.54,
            ),
        ]

    def list_cases(self):
        return [hit.case for hit in self.query([], 10)]


class WeatherProcessStore:
    def list_cases(self):
        return [
            ExtractedCase(
                source_pdf="FST2025-3.pdf",
                case_id="FST2025-3-case-01",
                case_no=1,
                title="1-3日雨雪天气过程",
                date_range="1-3日",
                content="3月1-3日出现雨雪天气。",
            ),
            ExtractedCase(
                source_pdf="FST2025-4.pdf",
                case_id="FST2025-4-case-02",
                case_no=2,
                title="4月16-18日大风、沙尘、霜冻和降水过程",
                date_range="16-18日",
                content="4月16日08时至4月18日08时，山西出现大风、沙尘、霜冻和降水天气。",
            ),
        ]


class FakeReranker:
    def rerank(self, question, hits, top_n):
        ordered = sorted(
            hits,
            key=lambda hit: ("暴雪" in hit.case.title, hit.score),
            reverse=True,
        )
        return ordered[:top_n]


class FakeAnswerGenerator:
    def answer(self, question, hits):
        return f"命中 {hits[0].case.case_id}"


class EmbeddingUnavailableClient:
    def is_available(self):
        return False

    def embed_query(self, text):
        raise AssertionError("Should not request embeddings when unavailable")


class DimensionAwareEmbeddingClient(FakeEmbeddingClient):
    def expected_dimension(self):
        return 1024


class MismatchedVectorStore(FakeStore):
    def collection_info(self):
        return {"dimension": 32, "count": 2, "source": "chroma"}


class RagServiceTests(unittest.TestCase):
    def test_rag_service_uses_reranked_top_hit(self):
        embedding_client = FakeEmbeddingClient()
        case_store = FakeStore()
        service = RagService(
            embedding_client=embedding_client,
            case_store=case_store,
            reranker=FakeReranker(),
            answer_generator=FakeAnswerGenerator(),
        )

        result = service.ask("请分析14-15日暴雪天气过程", top_k=5, top_n=2)

        self.assertEqual("FST2025-3-case-02", result.hits[0].case.case_id)
        self.assertIn("FST2025-3-case-02", result.answer)
        self.assertEqual("vector", result.retrieval_mode)
        self.assertEqual("not_requested", result.llm_status)
        self.assertEqual(["请分析14-15日暴雪天气过程"], embedding_client.queries)
        self.assertEqual([([0.0, 1.0], 5)], case_store.query_calls)

    def test_rag_service_falls_back_to_lexical_match_when_embeddings_unavailable(self):
        service = RagService(
            embedding_client=EmbeddingUnavailableClient(),
            case_store=FakeStore(),
            reranker=FakeReranker(),
            answer_generator=FakeAnswerGenerator(),
        )

        result = service.ask("请分析14-15日暴雪天气过程", top_k=5, top_n=2)

        self.assertEqual("FST2025-3-case-02", result.hits[0].case.case_id)
        self.assertIn("FST2025-3-case-02", result.answer)
        self.assertTrue(result.retrieval_mode.startswith("vector_error_fallback"))
        self.assertFalse(result.llm_used)

    def test_lexical_recall_prioritizes_date_and_disaster_type(self):
        service = RagService(
            embedding_client=EmbeddingUnavailableClient(),
            case_store=FakeStore(),
            reranker=FakeReranker(),
            answer_generator=FakeAnswerGenerator(),
        )

        hits = service._lexical_recall("请分析14-15日暴雪天气过程", top_k=2)

        self.assertEqual("FST2025-3-case-02", hits[0].case.case_id)

    def test_lexical_recall_normalizes_date_range_variants(self):
        service = RagService(
            embedding_client=EmbeddingUnavailableClient(),
            case_store=WeatherProcessStore(),
            reranker=FakeReranker(),
            answer_generator=FakeAnswerGenerator(),
        )

        hits = service._lexical_recall("2025年4月16~18日山西省天气过程的主要特征及环流背景是什么？", top_k=2)

        self.assertEqual("FST2025-4-case-02", hits[0].case.case_id)
        self.assertGreater(hits[0].score, hits[1].score)

    def test_rag_service_falls_back_when_vector_dimension_mismatches(self):
        service = RagService(
            embedding_client=DimensionAwareEmbeddingClient(),
            case_store=MismatchedVectorStore(),
            reranker=FakeReranker(),
            answer_generator=FakeAnswerGenerator(),
        )

        result = service.ask("请分析14-15日暴雪天气过程", top_k=5, top_n=2)

        self.assertTrue(result.retrieval_mode.startswith("vector_error_fallback"))
        self.assertIn("dimension mismatch", result.retrieval_mode)
        self.assertEqual("FST2025-3-case-02", result.hits[0].case.case_id)


if __name__ == "__main__":
    unittest.main()
