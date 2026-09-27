"""
会话管理 API 路由模块
提供聊天会话的创建、查询、更新、删除和消息列表查询功能，
支持多轮会话持久化存储和上下文记忆。
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from backend.app.api.schemas import CreateSessionRequest, MessageFeedbackRequest, MessageIssueRequest
from backend.app.services.feedback_guidance import guidance_fingerprint, normalize_with_model


# API 路由实例
router = APIRouter()
LOGGER = logging.getLogger("uvicorn.error")


def _main():
    """
    延迟加载 main 模块中的服务实例
    避免循环导入问题，在请求实际执行时才获取会话存储服务。
    """
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
def list_session_messages(session_id: str, client_id: str = ""):
    """列出单个聊天会话中的所有消息。"""
    main = _main()
    if main.session_store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = main.session_store.list_messages(session_id)
    if client_id:
        for message in messages:
            if message.get("role") != "assistant":
                continue
            feedback = main.session_store.get_message_feedback(session_id, message["message_id"], client_id)
            message["feedback"] = feedback.get("rating", "none")
            message["feedback_reason"] = feedback.get("reason", "")
    return {"session_id": session_id, "messages": messages}


@router.get("/api/sessions/{session_id}/messages/{message_id}")
def get_session_message(session_id: str, message_id: int):
    """读取单条消息，供前端在正文完成后刷新异步增强字段。"""
    message = _main().session_store.get_message(session_id, message_id)
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    return message


@router.get("/api/sessions/{session_id}/agent-runs")
def list_session_agent_runs(session_id: str):
    """列出主会话关联的 RAG、多维检索和 Smart 全部运行记录。"""
    main = _main()
    if main.session_store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "runs": main.session_store.list_agent_runs(session_id)}


def _assistant_message(session_id: str, message_id: int) -> dict:
    """读取并校验会话内的助手消息，所有消息级操作共用这一权限边界。"""
    message = _main().session_store.get_message(session_id, message_id)
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    if str(message.get("role") or "") != "assistant":
        raise HTTPException(status_code=400, detail="Only assistant messages support this operation")
    return message


def _report_title(question: str, answer: str) -> str:
    """从原问题和可见回答提取稳定标题，不为导出再次调用模型。"""
    years = list(dict.fromkeys(re.findall(r"(20\d{2})年", question)))
    months = [int(value) for value in re.findall(r"(?<!\d)(\d{1,2})月", question) if 1 <= int(value) <= 12]
    if len(years) == 1 and len(months) >= 2:
        return f"{years[0]}年{months[0]}月至{months[-1]}月气象灾害分析报告"
    if len(years) == 1 and len(months) == 1:
        return f"{years[0]}年{months[0]}月气象灾害分析报告"
    if len(years) == 1:
        return f"{years[0]}年气象灾害分析报告"
    heading = re.search(r"^#{1,3}\s+(.+)$", answer, flags=re.MULTILINE)
    if heading:
        value = re.sub(r"[*_`#]", "", heading.group(1)).strip(" ：:。")
        if value:
            return value if value.endswith("报告") else f"{value}报告"
    compact = re.sub(r"[\s，。！？：；,.!?:;]+", "", question)
    return f"{compact[:24]}报告" if compact else "气象灾害分析报告"


@router.post("/api/sessions/{session_id}/messages/{message_id}/export-pdf")
def export_message_pdf(session_id: str, message_id: int):
    """仅把指定助手消息的可见正文排版为 PDF，不重新检索或调用分析模型。"""
    message = _assistant_message(session_id, message_id)
    source_user = _main().session_store.previous_user_message(session_id, message_id)
    answer = str(message.get("content") or "").strip()
    if not answer:
        raise HTTPException(status_code=400, detail="Assistant message has no visible content")
    question = str((source_user or {}).get("content") or "")
    title = _report_title(question, answer)
    report_id = f"message_report_{uuid4().hex}"
    try:
        from backend.app.services.agent.case_multidim_search.router import _agent

        report_path = _agent().export_pdf_from_previous_result(
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
        # 消息级导出是一次性下载，不返回缩略图；正式多维报告仍保留自己的报告卡缩略图。
        "download_only": True,
        "report_mode": "message_export",
        "source_message_id": int(message_id),
    }
    metadata = dict(message.get("metadata") or {})
    # 兼容旧会话里可能存在的损坏报告项，避免导出成功后因元数据格式异常而返回失败。
    reports = [
        item
        for item in (metadata.get("reports") or [])
        if isinstance(item, dict) and item.get("report_id") != report_id
    ]
    reports.append(report)
    _main().session_store.update_message_metadata(session_id, message_id, {"reports": reports})
    return report


async def _normalize_feedback_in_background(
    main,
    guidance_id: str,
    original_question: str,
    visible_answer: str,
    raw_feedback: str,
) -> None:
    """后台归一化用户反馈，网络或模型异常只更新状态，不影响主聊天流。"""
    try:
        result = await asyncio.to_thread(
            normalize_with_model,
            getattr(main, "llm_client", None),
            original_question,
            visible_answer,
            raw_feedback,
        )
        stored = main.session_store.update_feedback_guidance(guidance_id, result)
        if stored and stored.get("status") == "active":
            session_id = str(stored.get("session_id") or "")
            current_key = guidance_fingerprint(str(stored.get("guidance_prompt") or ""))
            duplicate = next(
                (
                    item
                    for item in main.session_store.list_feedback_guidances(session_id, statuses=("active",), limit=5)
                    if str(item.get("guidance_id") or "") != guidance_id
                    and current_key
                    and guidance_fingerprint(str(item.get("guidance_prompt") or "")) == current_key
                ),
                None,
            )
            if duplicate:
                main.session_store.set_feedback_guidance_status(guidance_id, "duplicate")
            else:
                main.session_store.expire_old_feedback_guidances(session_id, keep=5)
    except Exception as exc:  # pragma: no cover - 仅记录部署环境异常
        LOGGER.warning("[会话反馈] 异步归一化失败 guidance_id=%s error=%s", guidance_id, exc)
        try:
            main.session_store.update_feedback_guidance(
                guidance_id,
                {"status": "failed", "error": str(exc)[:240]},
            )
        except Exception:
            LOGGER.exception("[会话反馈] 归一化失败状态写入失败 guidance_id=%s", guidance_id)


def _schedule_feedback_normalization(
    main,
    guidance_id: str,
    original_question: str,
    visible_answer: str,
    raw_feedback: str,
) -> None:
    """登记受引用保护的异步任务，避免请求返回后任务被垃圾回收。"""
    tasks = getattr(main, "_feedback_guidance_tasks", None)
    if tasks is None:
        tasks = set()
        setattr(main, "_feedback_guidance_tasks", tasks)
    task = asyncio.create_task(
        _normalize_feedback_in_background(
            main,
            guidance_id,
            original_question,
            visible_answer,
            raw_feedback,
        )
    )
    tasks.add(task)
    task.add_done_callback(tasks.discard)


@router.post("/api/sessions/{session_id}/messages/{message_id}/feedback")
async def save_message_feedback(session_id: str, message_id: int, payload: MessageFeedbackRequest):
    """保存互斥反馈，并异步生成当前会话的临时回答要求。"""
    main = _main()
    message = _assistant_message(session_id, message_id)
    result = main.session_store.set_message_feedback(
        session_id, message_id, payload.client_id, payload.rating, payload.reason,
    )
    if payload.rating == "needs_improvement":
        source_user = main.session_store.previous_user_message(session_id, message_id) or {}
        guidance = main.session_store.create_feedback_guidance(
            session_id,
            message_id,
            payload.client_id,
            payload.reason,
        )
        _schedule_feedback_normalization(
            main,
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
def report_message_issue(session_id: str, message_id: int, payload: MessageIssueRequest):
    """记录回答问题分类和用户说明，不把上报内容混入聊天上下文。"""
    _assistant_message(session_id, message_id)
    return _main().session_store.add_message_issue(
        session_id, message_id, payload.client_id, payload.category, payload.description,
    )


@router.post("/api/sessions/{session_id}/messages/{message_id}/share")
def share_message(session_id: str, message_id: int, request: Request):
    """创建只含当前问题和可见回答的不可变共享快照。"""
    message = _assistant_message(session_id, message_id)
    source_user = _main().session_store.previous_user_message(session_id, message_id)
    question = str((source_user or {}).get("content") or "")
    answer = str(message.get("content") or "")
    title = _report_title(question, answer).removesuffix("报告")
    snapshot = _main().session_store.create_shared_snapshot(
        session_id, message_id, question, answer, title,
    )
    snapshot["share_url"] = f"{str(request.base_url).rstrip('/')}/share/{snapshot['token']}"
    return snapshot


@router.get("/api/shared-messages/{token}")
def get_shared_message(token: str):
    """返回公开快照，不包含原消息元数据和会话内其他内容。"""
    snapshot = _main().session_store.get_shared_snapshot(token)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Shared message not found")
    return snapshot


@router.get("/share/{token}", response_class=HTMLResponse)
def shared_message_page(token: str):
    """提供最小只读共享页，所有用户文本均先做 HTML 转义。"""
    snapshot = _main().session_store.get_shared_snapshot(token)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Shared message not found")
    title = html.escape(str(snapshot.get("title") or "气象灾害问答"))
    question = html.escape(str(snapshot.get("question") or ""))
    answer = html.escape(str(snapshot.get("answer") or ""))
    return HTMLResponse(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>{title}</title><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<style>body{margin:0;background:#fbfbfa;color:#292927;font:15px/1.75 system-ui,sans-serif}"
        "main{max-width:780px;margin:auto;padding:48px 22px}h1{font-size:22px}"
        ".q{background:#eeeeec;border-radius:14px;padding:10px 14px;margin:22px 0;white-space:pre-wrap}"
        ".a{white-space:pre-wrap;overflow-wrap:anywhere}</style></head><body><main>"
        f"<h1>{title}</h1><div class='q'>{question}</div><div class='a'>{answer}</div>"
        "</main></body></html>"
    )
