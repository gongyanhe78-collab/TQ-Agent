"""当前用户长期记忆的读取与删除接口。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.api.dependencies import CurrentUserDep, MemoryServiceDep

router = APIRouter()


@router.get("/api/memories")
def list_memories(
    memory_service: MemoryServiceDep,
    user: CurrentUserDep,
):
    """返回当前用户可见的长期记忆，不暴露内部向量。"""
    context = memory_service.context_for_user(str(user["user_id"]))
    return {
        "user_id": context["user_id"],
        "memories": context["memories"],
        "fallback_structured": context["fallback_structured"],
    }


@router.delete("/api/memories/{memory_id}")
def delete_memory(
    memory_id: str,
    memory_service: MemoryServiceDep,
    user: CurrentUserDep,
):
    """软删除当前用户的一条长期记忆。"""
    result = memory_service.delete(
        str(user["user_id"]),
        str(memory_id),
    )
    if result is None or result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="记忆不存在")
    return result
