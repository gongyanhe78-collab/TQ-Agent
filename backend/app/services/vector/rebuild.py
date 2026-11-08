"""向量索引重建与 ID 一致性对账逻辑，全部只从权威数据构建。"""
from __future__ import annotations

import time
from typing import Any

from backend.app.models import DocumentChunk
from backend.app.services.vector.json_document_store import (
    read_authority_chunks,
)


def compare_id_sets(
    expected: set[str],
    actual: set[str],
) -> dict[str, Any]:
    """比较期望与实际的 ID 集合，返回数量与差异。"""
    normalized_expected = {str(item) for item in expected if str(item)}
    normalized_actual = {str(item) for item in actual if str(item)}
    return {
        "expected_count": len(normalized_expected),
        "actual_count": len(normalized_actual),
        "missing_ids": sorted(normalized_expected - normalized_actual),
        "extra_ids": sorted(normalized_actual - normalized_expected),
        "common_ids": sorted(normalized_expected & normalized_actual),
    }


def rebuild_document_store(
    store: Any,
    chunks: list[DocumentChunk],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """按权威 chunk 重建文档集合，返回数量、差异、耗时和失败明细。"""
    started = time.time()
    expected_ids = {chunk.chunk_id for chunk in chunks if chunk.chunk_id}
    failures: list[dict[str, str]] = []
    try:
        actual_before = set(store.list_chunk_ids())
    except Exception as exc:
        actual_before = set()
        failures.append(
            {
                "phase": "before",
                "action": "list_chunk_ids",
                "error": str(exc),
            }
        )
    comparison = compare_id_sets(expected_ids, actual_before)

    if not dry_run:
        try:
            store.rebuild()
        except Exception as exc:
            failures.append(
                {
                    "phase": "rebuild",
                    "action": "rebuild",
                    "error": str(exc),
                }
            )
        try:
            actual_after = set(store.list_chunk_ids())
        except Exception as exc:
            actual_after = set()
            failures.append(
                {
                    "phase": "after",
                    "action": "list_chunk_ids",
                    "error": str(exc),
                }
            )
        comparison = compare_id_sets(expected_ids, actual_after)

    return {
        "collection": "documents",
        "dry_run": bool(dry_run),
        "expected_count": comparison["expected_count"],
        "actual_count": comparison["actual_count"],
        "missing_count": len(comparison["missing_ids"]),
        "extra_count": len(comparison["extra_ids"]),
        "missing_ids": comparison["missing_ids"][:100],
        "extra_ids": comparison["extra_ids"][:100],
        "elapsed_seconds": round(time.time() - started, 3),
        "failures": failures,
    }


def rebuild_memory_store(
    store: Any,
    memories: list[dict[str, Any]],
    *,
    user_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """按 PostgreSQL active 记忆重建记忆集合，返回数量、差异和失败明细。"""
    started = time.time()
    expected_ids = {
        str(memory.get("memory_id") or "")
        for memory in memories
        if str(memory.get("memory_id") or "")
    }
    failures: list[dict[str, str]] = []
    try:
        if user_id:
            actual = _list_memory_ids(store, user_id)
        else:
            actual = set()
    except Exception as exc:
        actual = set()
        failures.append(
            {
                "phase": "before",
                "action": "list_memory_ids",
                "error": str(exc),
            }
        )
    comparison = compare_id_sets(expected_ids, actual)

    if not dry_run and memories:
        try:
            store.upsert_memories(memories)
        except Exception as exc:
            failures.append(
                {
                    "phase": "rebuild",
                    "action": "upsert_memories",
                    "error": str(exc),
                }
            )

    return {
        "collection": "memories",
        "user_id": user_id or "",
        "dry_run": bool(dry_run),
        "expected_count": comparison["expected_count"],
        "actual_count": comparison["actual_count"],
        "missing_count": len(comparison["missing_ids"]),
        "extra_count": len(comparison["extra_ids"]),
        "missing_ids": comparison["missing_ids"][:100],
        "extra_ids": comparison["extra_ids"][:100],
        "elapsed_seconds": round(time.time() - started, 3),
        "failures": failures,
    }


def _list_memory_ids(store: Any, user_id: str) -> set[str]:
    """列出指定用户的记忆索引标识；后端不支持时返回空集合。"""
    method = getattr(store, "list_memory_ids", None)
    if method is None:
        return set()
    return {
        str(memory_id)
        for memory_id in method(user_id=user_id)
        if str(memory_id)
    }


def read_authority_chunks_from(settings: Any) -> list[DocumentChunk]:
    """从配置的权威 chunk JSON 读取全部正文块。"""
    return read_authority_chunks(settings.document_index_dir / "chunks.json")
