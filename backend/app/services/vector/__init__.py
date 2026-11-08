"""文档与长期记忆共用的统一向量存储契约。"""
from __future__ import annotations

from backend.app.services.vector.contracts import (
    CollectionInfo,
    DocumentVectorStore,
    EmptyMemoryVectorStore,
    MemoryVectorStore,
    VectorHit,
    VectorQuery,
)

__all__ = [
    "CollectionInfo",
    "DocumentVectorStore",
    "EmptyMemoryVectorStore",
    "MemoryVectorStore",
    "VectorHit",
    "VectorQuery",
]
