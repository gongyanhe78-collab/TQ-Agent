"""从同一黄金数据集对比 Chroma 与 Milvus 检索指标并输出报告。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.evals.loader import GoldenDataset
from backend.evals.metrics.retrieval import (
    hit_at_k,
    mrr,
    recall_at_k,
)
from backend.app.services.model_client import get_embedding_client


def _load_cases(dataset_path: Path) -> list:
    """加载黄金数据集并返回评测样例。"""
    root = Path(__file__).resolve().parents[1]
    dataset = GoldenDataset.load(
        dataset_path,
        root / "data" / "standard_cases1.json",
        root / "data" / "document_index" / "chunks.json",
    )
    return dataset.cases


def _chroma_store(settings: Settings):
    """构造 Chroma 文档向量存储。"""
    from backend.app.services.vector.chroma_document_store import (
        ChromaDocumentVectorStore,
    )

    return ChromaDocumentVectorStore(
        authority_path=settings.document_index_dir / "chunks.json",
        persist_dir=settings.document_index_dir,
        collection_name=settings.document_collection_name,
        schema_version=settings.vector_schema_version,
    )


def _milvus_store(settings: Settings):
    """构造 Milvus 文档向量存储。"""
    from backend.app.services.vector.milvus_document_store import (
        MilvusDocumentStore,
    )
    from backend.app.services.vector.rebuild import (
        read_authority_chunks_from,
    )

    chunks = read_authority_chunks_from(settings)
    dimension = 0
    for chunk in chunks:
        if chunk.embedding:
            dimension = len(chunk.embedding)
            break
    return MilvusDocumentStore(
        uri=settings.milvus_uri,
        collection_name=settings.milvus_collection_name,
        authority_path=settings.document_index_dir / "chunks.json",
        dimension=dimension,
        token=settings.milvus_token,
        database=settings.milvus_database,
        schema_version=settings.vector_schema_version,
        dense_top_k=settings.milvus_dense_top_k,
        sparse_top_k=settings.milvus_sparse_top_k,
        rrf_k=settings.milvus_rrf_k,
        timeout_seconds=settings.milvus_timeout_seconds,
        standard_cases_path=settings.standard_cases_path,
        embedding_model=settings.embedding_model,
        document_version="v1",
    )


def _run_backend(
    store,
    cases: list,
    embedding_client,
) -> dict:
    """对给定后端运行同一组问题并计算确定性检索指标。"""
    started = time.time()
    ranked_by_case: list[dict] = []
    hit3_values: list[float] = []
    recall5_values: list[float] = []
    mrr_values: list[float] = []
    failures: list[dict] = []
    for case in cases:
        expected_ids = list(case.expected_chunk_ids or [])
        if not expected_ids:
            continue
        try:
            query_embedding = embedding_client.embed_query(case.question)
            hits = store.query(
                query_embedding,
                top_k=5,
                query=None,
            )
            ranked_ids = [hit.chunk_id for hit in hits]
        except Exception as exc:
            failures.append(
                {
                    "case_id": case.case_id,
                    "error": str(exc),
                }
            )
            continue
        hit3_values.append(hit_at_k(ranked_ids, expected_ids, 3))
        recall5_values.append(recall_at_k(ranked_ids, expected_ids, 5))
        mrr_values.append(mrr(ranked_ids, expected_ids))
        ranked_by_case.append(
            {
                "case_id": case.case_id,
                "expected_ids": expected_ids,
                "ranked_ids": ranked_ids,
            }
        )

    def _average(values: list[float]) -> float:
        """计算平均值；空样本返回 0。"""
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "backend": store.backend,
        "collection": getattr(store, "collection_name", ""),
        "sample_count": len(hit3_values),
        "hit_at_3": _average(hit3_values),
        "recall_at_5": _average(recall5_values),
        "mrr": _average(mrr_values),
        "elapsed_seconds": round(time.time() - started, 3),
        "failure_count": len(failures),
        "failures": failures,
        "ranked_by_case": ranked_by_case,
    }


def _write_reports(output_dir: Path, chroma: dict, milvus: dict) -> None:
    """从同一对比对象生成 JSON 与 Markdown 报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backends": [chroma, milvus],
    }
    json_path = output_dir / "vector_backend_compare.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown_lines = [
        "# 向量后端对比报告",
        "",
        f"- 生成时间：{payload['generated_at']}",
        "",
        "| 指标 | Chroma | Milvus |",
        "|---|---|---|",
    ]
    for key, label in (
        ("hit_at_3", "Hit@3"),
        ("recall_at_5", "Recall@5"),
        ("mrr", "MRR"),
        ("sample_count", "有效样本数"),
        ("failure_count", "失败样本数"),
    ):
        markdown_lines.append(
            f"| {label} | {chroma.get(key, 0)} | {milvus.get(key, 0)} |"
        )
    markdown_lines.extend(
        [
            "",
            "## 结论",
            "",
            "对比报告只提供证据，不自动切换默认后端；",
            "是否切换默认后端由评测基线和运维决策决定。",
        ]
    )
    markdown_path = output_dir / "vector_backend_compare.md"
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    print(f"JSON 报告：{json_path}")
    print(f"Markdown 报告：{markdown_path}")


def main() -> int:
    """解析参数并运行 Chroma 与 Milvus 对比。"""
    parser = argparse.ArgumentParser(description="对比 Chroma 与 Milvus 检索")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("backend/evals/datasets/golden_v1.json"),
        help="黄金数据集路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".tmp/vector-backend-compare"),
        help="报告输出目录",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="最多运行的样例数，0 表示全部",
    )
    args = parser.parse_args()

    settings = Settings()
    embedding_client = get_embedding_client()
    if not embedding_client.is_available():
        print("未配置 DashScope API Key，无法进行真实向量对比。", file=sys.stderr)
        return 2

    cases = _load_cases(args.dataset)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    chroma_store = _chroma_store(settings)
    milvus_store = _milvus_store(settings)
    try:
        chroma_result = _run_backend(chroma_store, cases, embedding_client)
        milvus_result = _run_backend(milvus_store, cases, embedding_client)
        _write_reports(args.output_dir, chroma_result, milvus_result)
    finally:
        chroma_store.close()
        milvus_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
