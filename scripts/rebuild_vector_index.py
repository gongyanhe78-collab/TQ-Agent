"""按权威数据重建文档或记忆向量索引，支持 dry-run 与差异对账。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.app.services.vector.factory import build_document_vector_store
from backend.app.services.vector.rebuild import (
    rebuild_document_store,
    rebuild_memory_store,
)


def _document_store(settings: Settings):
    """构造文档向量存储，Milvus 不可用时按固定链降级。"""
    from backend.app.services.model_client import get_embedding_client

    return build_document_vector_store(
        settings,
        embedding_client=get_embedding_client(),
    )


def _memory_store(settings: Settings):
    """构造记忆向量存储；Milvus 不可用时使用空实现并记录降级。"""
    from backend.app.services.vector.contracts import EmptyMemoryVectorStore
    from backend.app.services.vector.milvus_memory_store import (
        MilvusMemoryStore,
    )

    store = MilvusMemoryStore(
        uri=settings.milvus_uri,
        collection_name=settings.milvus_memory_collection_name,
        dimension=0,
        token=settings.milvus_token,
        database=settings.milvus_database,
        dense_top_k=settings.milvus_dense_top_k,
        sparse_top_k=settings.milvus_sparse_top_k,
        rrf_k=settings.milvus_rrf_k,
        timeout_seconds=settings.milvus_timeout_seconds,
    )
    health = store.health()
    if str(health.get("health_status") or "") in {"healthy", "ready", "ok"}:
        return store, []
    return EmptyMemoryVectorStore(), [
        {
            "backend": "milvus",
            "reason": health.get("detail") or "Milvus 记忆集合不可用，使用空实现",
        }
    ]


def _active_memories(settings: Settings):
    """读取 PostgreSQL 中的 active 长期记忆；依赖未就绪时返回空。"""
    try:
        from backend.app.repositories.memory_repository import (
            MemoryRepository,
        )
    except ImportError as exc:
        return [], f"长期记忆仓储未就绪：{exc}"
    try:
        from backend.app.db.session import build_engine

        engine = build_engine(settings)
        repository = MemoryRepository(engine)
        memories = repository.rebuildable_active()
        return memories, ""
    except Exception as exc:
        return [], f"读取 active 记忆失败：{exc}"


def _write_output(output_dir: Path, results: list[dict]) -> None:
    """把重建结果写入 JSON 文件并打印摘要。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rebuild_report.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for result in results:
        collection = result.get("collection")
        print(
            f"[{collection}] 期望 {result['expected_count']} 实际 "
            f"{result['actual_count']} 缺失 {result['missing_count']} "
            f"多余 {result['extra_count']} 耗时 {result['elapsed_seconds']}s"
        )
        for failure in result.get("failures", []):
            print(f"[{collection}] 失败 {failure.get('phase')}: {failure.get('error')}")
    print(f"报告已写入：{output_path}")


def main() -> int:
    """解析参数并执行文档与记忆集合重建。"""
    parser = argparse.ArgumentParser(description="重建向量索引")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只对账不写库",
    )
    parser.add_argument(
        "--collection",
        choices=["documents", "memories", "all"],
        default="all",
        help="重建范围",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".tmp/vector-rebuild"),
        help="报告输出目录",
    )
    args = parser.parse_args()

    settings = Settings()
    results: list[dict] = []
    want_documents = args.collection in {"documents", "all"}
    want_memories = args.collection in {"memories", "all"}

    if want_documents:
        store = _document_store(settings)
        chunks = []
        try:
            from backend.app.services.vector.rebuild import (
                read_authority_chunks_from,
            )

            chunks = read_authority_chunks_from(settings)
        except Exception as exc:
            results.append(
                {
                    "collection": "documents",
                    "dry_run": args.dry_run,
                    "expected_count": 0,
                    "actual_count": 0,
                    "missing_count": 0,
                    "extra_count": 0,
                    "elapsed_seconds": 0.0,
                    "failures": [{"phase": "read", "action": "read_authority", "error": str(exc)}],
                }
            )
        else:
            results.append(
                rebuild_document_store(
                    store,
                    chunks,
                    dry_run=args.dry_run,
                )
            )
        store.close()

    if want_memories:
        memory_store, fallbacks = _memory_store(settings)
        memories, error = _active_memories(settings)
        result = rebuild_memory_store(
            memory_store,
            memories,
            dry_run=args.dry_run,
        )
        if fallbacks:
            result["fallbacks"] = fallbacks
        if error:
            result["failures"].append({"phase": "read", "action": "memory_repository", "error": error})
        results.append(result)
        memory_store.close()

    _write_output(args.output_dir, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
