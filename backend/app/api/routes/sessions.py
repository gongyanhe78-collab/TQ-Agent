"""聊天会话、消息操作、导出与公开分享接口。"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from backend.app.api.dependencies import CurrentUserDep, ServicesDep, SessionStoreDep
from backend.app.api.schemas import (
    CreateSessionRequest,
    MessageFeedbackRequest,
    MessageIssueRequest,
)
from backend.app.services.session_exports import render_shared_message, report_title

router = APIRouter()


@router.post("/api/sessions")
def create_session(payload: CreateSessionRequest, session_store: SessionStoreDep):
    """创建一个聊天会话。"""
    return session_store.create_session(payload.title)


@router.get("/api/sessions")
def list_sessions(session_store: SessionStoreDep):
    """列出当前用户拥有的聊天会话。"""
    return {"sessions": session_store.list_sessions()}


@router.get("/api/sessions/{session_id}")
def get_session(session_id: str, session_store: SessionStoreDep):
    """获取当前用户拥有的单个聊天会话。"""
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.patch("/api/sessions/{session_id}")
def update_session(
    session_id: str,
    payload: CreateSessionRequest,
    session_store: SessionStoreDep,
):
    """更新聊天会话标题。"""
    try:
        return session_store.update_session_title(
            session_id,
            payload.title or "新会话",
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, session_store: SessionStoreDep):
    """删除一个聊天会话及其所有消息。"""
    if not session_store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True, "session_id": session_id}


@router.get("/api/sessions/{session_id}/messages")
def list_session_messages(
    session_id: str,
    session_store: SessionStoreDep,
    client_id: str = "",
):
    """列出单个聊天会话中的所有消息。"""
    if session_store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = session_store.list_messages(session_id)
    if client_id:
        for message in messages:
            if message.get("role") != "assistant":
                continue
            feedback = session_store.get_message_feedback(
                session_id,
                message["message_id"],
                client_id,
            )
            message["feedback"] = feedback.get("rating", "none")
            message["feedback_reason"] = feedback.get("reason", "")
    return {"session_id": session_id, "messages": messages}


@router.get("/api/sessions/{session_id}/messages/{message_id}")
def get_session_message(
    session_id: str,
    message_id: int,
    session_store: SessionStoreDep,
):
    """读取单条消息，供前端刷新异步增强字段。"""
    message = session_store.get_message(session_id, message_id)
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    return message


@router.get("/api/sessions/{session_id}/agent-runs")
def list_session_agent_runs(session_id: str, session_store: SessionStoreDep):
    """列出主会话关联的全部智能体运行记录。"""
    if session_store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "runs": session_store.list_agent_runs(session_id),
    }


def _assistant_message(
    session_store,
    session_id: str,
    message_id: int,
) -> dict:
    """读取并校验会话内助手消息，作为消息级操作的权限边界。"""
    message = session_store.get_message(session_id, message_id)
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    if str(message.get("role") or "") != "assistant":
        raise HTTPException(
            status_code=400,
            detail="Only assistant messages support this operation",
        )
    return message


@router.post("/api/sessions/{session_id}/messages/{message_id}/export-pdf")
def export_message_pdf(
    session_id: str,
    message_id: int,
    services: ServicesDep,
    user: CurrentUserDep,
    session_store: SessionStoreDep,
):
    """仅把指定助手消息的可见正文排版为 PDF。"""
    message = _assistant_message(session_store, session_id, message_id)
    source_user = session_store.previous_user_message(session_id, message_id)
    answer = str(message.get("content") or "").strip()
    if not answer:
        raise HTTPException(
            status_code=400,
            detail="Assistant message has no visible content",
        )
    question = str((source_user or {}).get("content") or "")
    title = report_title(question, answer)
    report_id = f"message_report_{uuid4().hex}"
    try:
        agent = services.multidim_agent_for(str(user["user_id"]))
        report_path = agent.export_pdf_from_previous_result(
            {"answer": answer},
            filename=f"{report_id}.pdf",
            report_title=title,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF 导出失败：{exc}") from exc
    report = {
        "report_id": report_id,
        "title": title,
        "filename": report_path.name,
        "url": f"/api/case-multidim/reports/{report_path.name}",
        "download_only": True,
        "report_mode": "message_export",
        "source_message_id": int(message_id),
    }
    metadata = dict(message.get("metadata") or {})
    reports = [
        item
        for item in (metadata.get("reports") or [])
        if isinstance(item, dict) and item.get("report_id") != report_id
    ]
    reports.append(report)
    session_store.update_message_metadata(
        session_id,
        message_id,
        {"reports": reports},
    )
    return report


@router.post("/api/sessions/{session_id}/messages/{message_id}/feedback")
async def save_message_feedback(
    session_id: str,
    message_id: int,
    payload: MessageFeedbackRequest,
    services: ServicesDep,
    user: CurrentUserDep,
    session_store: SessionStoreDep,
):
    """保存互斥反馈，并异步生成当前会话的临时回答要求。"""
    message = _assistant_message(session_store, session_id, message_id)
    result = session_store.set_message_feedback(
        session_id,
        message_id,
        payload.client_id,
        payload.rating,
        payload.reason,
    )
    if payload.rating == "needs_improvement":
        source_user = session_store.previous_user_message(session_id, message_id) or {}
        guidance = session_store.create_feedback_guidance(
            session_id,
            message_id,
            payload.client_id,
            payload.reason,
        )
        services.feedback_tasks_for(str(user["user_id"])).schedule(
            str(guidance.get("guidance_id") or ""),
            str(source_user.get("content") or ""),
            str(message.get("content") or ""),
            payload.reason,
        )
        result.update({
            "guidance_id": guidance.get("guidance_id"),
            "guidance_status": "pending",
        })
    else:
        result.update({"guidance_status": "cleared"})
    return result


@router.post("/api/sessions/{session_id}/messages/{message_id}/report")
def report_message_issue(
    session_id: str,
    message_id: int,
    payload: MessageIssueRequest,
    session_store: SessionStoreDep,
):
    """记录问题分类和用户说明，不混入聊天上下文。"""
    _assistant_message(session_store, session_id, message_id)
    return session_store.add_message_issue(
        session_id,
        message_id,
        payload.client_id,
        payload.category,
        payload.description,
    )


@router.post("/api/sessions/{session_id}/messages/{message_id}/share")
def share_message(
    session_id: str,
    message_id: int,
    request: Request,
    services: ServicesDep,
    session_store: SessionStoreDep,
):
    """创建只含当前问题和可见回答的不可变共享快照。"""
    message = _assistant_message(session_store, session_id, message_id)
    source_user = session_store.previous_user_message(session_id, message_id)
    question = str((source_user or {}).get("content") or "")
    answer = str(message.get("content") or "")
    title = report_title(question, answer).removesuffix("报告")
    expires_at = (
        datetime.now(UTC)
        + timedelta(days=max(1, int(services.settings.share_ttl_days)))
    )
    snapshot = session_store.create_shared_snapshot(
        session_id,
        message_id,
        question,
        answer,
        title,
        expires_at=expires_at,
    )
    snapshot["expires_at"] = expires_at.isoformat()
    snapshot["share_url"] = (
        f"{str(request.base_url).rstrip('/')}/share/{snapshot['token']}"
    )
    return snapshot


@router.delete("/api/sessions/{session_id}/messages/{message_id}/share")
def revoke_message_share(
    session_id: str,
    message_id: int,
    session_store: SessionStoreDep,
):
    """撤销当前会话内某条消息的公开分享，只允许消息所属会话操作。"""
    revoked = session_store.revoke_shared_snapshot(
        session_id,
        int(message_id),
    )
    if not revoked:
        raise HTTPException(status_code=404, detail="Shared message not found")
    return {"revoked": True}


@router.get("/api/shared-messages/{token}")
def get_shared_message(token: str, services: ServicesDep):
    """返回公开快照，不包含原消息元数据和会话内其他内容。"""
    meta = services.shared_store.shared_snapshot_meta(token)
    if meta is None:
        raise HTTPException(status_code=404, detail="Shared message not found")
    snapshot = services.shared_store.get_shared_snapshot(token)
    if snapshot is None:
        raise HTTPException(status_code=403, detail="Shared message expired or revoked")
    return snapshot


@router.get("/share/{token}", response_class=HTMLResponse)
def shared_message_page(token: str, services: ServicesDep):
    """提供最小只读共享页，所有用户文本均先做 HTML 转义。"""
    meta = services.shared_store.shared_snapshot_meta(token)
    if meta is None:
        raise HTTPException(status_code=404, detail="Shared message not found")
    snapshot = services.shared_store.get_shared_snapshot(token)
    if snapshot is None:
        raise HTTPException(status_code=403, detail="Shared message expired or revoked")
    return HTMLResponse(render_shared_message(snapshot))
