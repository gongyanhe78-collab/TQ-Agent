"""系统健康状态与知识库状态接口。"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

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


@router.get("/api/health")
def health(services: ServicesDep):
    """返回系统状态、向量库信息和模型可用性。"""
    vector_info = services.document_store.collection_info()
    return {
        "status": "ok",
        "vector_keys": vector_info["count"],
        "vector_dimension": vector_info["dimension"],
        "vector_source": vector_info["source"],
        "embedding_available": services.embedding_client.is_available(),
        "llm_available": services.llm_client.is_available(),
    }


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
