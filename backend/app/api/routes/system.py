from __future__ import annotations

import json

from fastapi import APIRouter

from backend.app.config import settings
from backend.app.services.model_client.rerank import get_rerank_client


router = APIRouter()


def _main():
    """请求执行时获取 main 模块中的服务实例，兼容测试里的 monkeypatch。"""
    from backend.app import main

    return main


def _knowledge_manifest() -> dict:
    """读取统一知识库构建清单；清单缺失或损坏时返回可诊断状态。"""
    manifest_path = settings.data_dir / "knowledge_manifest.json"
    if not manifest_path.exists():
        return {"status": "missing", "path": str(manifest_path)}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "path": str(manifest_path), "error": str(exc)}
    return {"status": "ready", **manifest}


@router.get("/api/health")
def health():
    """健康检查接口，返回系统状态、向量库信息和模型可用性。"""
    main = _main()
    vector_info = main.document_store.collection_info()
    return {
        "status": "ok",
        "vector_keys": vector_info["count"],
        "vector_dimension": vector_info["dimension"],
        "vector_source": vector_info["source"],
        "embedding_available": main.embedding_client.is_available(),
        "llm_available": main.llm_client.is_available(),
    }


@router.get("/api/knowledge/status")
def knowledge_status():
    """返回知识库当前索引、模型和构建状态。"""
    main = _main()
    vector_info = main.document_store.collection_info()
    return {
        "status": "ok",
        "case_count": len(main.standard_case_store.list_cases()),
        "vector_count": vector_info["count"],
        "vector_dimension": vector_info["dimension"],
        "vector_source": vector_info["source"],
        "embedding_model": settings.embedding_model,
        "rerank_model": settings.rerank_model,
        "chat_model": settings.chat_model,
        "embedding_available": main.embedding_client.is_available(),
        "rerank_available": get_rerank_client().is_available(),
        "llm_available": main.llm_client.is_available(),
        "build": _knowledge_manifest(),
    }
