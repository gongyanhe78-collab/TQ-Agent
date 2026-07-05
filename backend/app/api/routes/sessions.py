from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.api.schemas import CreateSessionRequest


router = APIRouter()


def _main():
    """请求执行时获取 main 模块中的会话存储实例。"""
    from backend.app import main

    return main


@router.post("/api/sessions")
def create_session(payload: CreateSessionRequest):
    """创建一个聊天会话。"""
    return _main().session_store.create_session(payload.title)


@router.get("/api/sessions")
def list_sessions():
    """列出所有聊天会话。"""
    return {"sessions": _main().session_store.list_sessions()}


@router.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    """获取单个聊天会话。"""
    session = _main().session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.patch("/api/sessions/{session_id}")
def update_session(session_id: str, payload: CreateSessionRequest):
    """更新聊天会话标题。"""
    try:
        return _main().session_store.update_session_title(session_id, payload.title or "新会话")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    """删除一个聊天会话及其所有消息。"""
    if not _main().session_store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True, "session_id": session_id}


@router.get("/api/sessions/{session_id}/messages")
def list_session_messages(session_id: str):
    """列出单个聊天会话中的所有消息。"""
    main = _main()
    if main.session_store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "messages": main.session_store.list_messages(session_id)}
