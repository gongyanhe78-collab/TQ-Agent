"""系统健康状态与知识库状态接口。"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from backend.app.api.dependencies import ServicesDep
from backend.app.core.config import Settings

router = APIRouter()


def _knowledge_manifest(settings: Settings) -> dict:
    """读取统一知识库构建清单；缺失或损坏时返回可诊断状态。"""
    manifest_path = settings.data_dir / "knowledge_manifest.json"
    if not manifest_path.exists():
        return {"status": "missing", "path": str(manifest_path)}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "path": str(manifest_path), "error": str(exc)}
    return {"status": "ready", **manifest}


def _dependency_readiness(services: Any) -> dict[str, Any]:
    """探测关键依赖可用性；缺少数值一律标记为 unknown，不抛异常。"""
    # PostgreSQL：执行一次 SELECT 1，避免把连接池故障误报为进程存活。
    postgres = "unknown"
    engine = getattr(services, "engine", None)
    if engine is not None:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            postgres = "ok"
        except Exception:
            postgres = "down"

    # Redis：通过短期缓存客户端 ping，失败时仍返回，不影响进程存活判断。
    redis = "unknown"
    context_cache = getattr(services, "context_cache", None)
    if context_cache is not None:
        try:
            redis = "ok" if context_cache.ping() else "down"
        except Exception:
            redis = "down"

    # 向量库：读取集合信息，异常时标记 down 并返回可诊断的空信息。
    vector = "unknown"
    vector_info: dict[str, Any] = {
        "count": 0,
        "dimension": 0,
        "source": "unknown",
    }
    document_store = getattr(services, "document_store", None)
    if document_store is not None:
        try:
            vector_info = document_store.collection_info()
            vector = "ok"
        except Exception:
            vector = "down"
    return {
        "postgres": postgres,
        "redis": redis,
        "vector": vector,
        "vector_keys": vector_info.get("count", 0),
        "vector_dimension": vector_info.get("dimension", 0),
        "vector_source": vector_info.get("source", "unknown"),
    }


@router.get("/api/health")
def health(services: ServicesDep):
    """存活检查：返回系统状态与关键依赖就绪情况，不因依赖故障返回 5xx。"""
    dependencies = _dependency_readiness(services)
    return {
        "status": "ok",
        **dependencies,
        "embedding_available": services.embedding_client.is_available(),
        "llm_available": services.llm_client.is_available(),
    }


@router.get("/api/ready")
def readiness(services: ServicesDep):
    """就绪检查：关键依赖全部可用返回 200，任一关键依赖不可用返回 503。"""
    dependencies = _dependency_readiness(services)
    ready = (
        dependencies["postgres"] == "ok"
        and dependencies["redis"] == "ok"
        and dependencies["vector"] == "ok"
    )
    status_code = 200 if ready else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if ready else "not_ready",
            **dependencies,
        },
    )


@router.get("/api/knowledge/status")
def knowledge_status(services: ServicesDep):
    """返回知识库索引、模型和构建状态。"""
    vector_info = services.document_store.collection_info()
    settings = services.settings
    return {
        "status": "ok",
        "case_count": len(services.standard_case_store.list_cases()),
        "vector_count": vector_info["count"],
        "vector_dimension": vector_info["dimension"],
        "vector_source": vector_info["source"],
        "embedding_model": settings.embedding_model,
        "rerank_model": settings.rerank_model,
        "chat_model": settings.chat_model,
        "embedding_available": services.embedding_client.is_available(),
        "rerank_available": services.rerank_client.is_available(),
        "llm_available": services.llm_client.is_available(),
        "build": _knowledge_manifest(settings),
    }
