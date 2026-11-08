"""失败运行重放服务：读取 failed 运行并按原会话重放。"""
from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from backend.app.api.schemas import StreamQueryRequest


async def list_failed_runs(
    session_store: Any,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """列出全部 failed 状态运行，供运维脚本批量重放。

    只返回运行 ID、会话 ID、问题与创建时间，不返回大块证据。
    """
    runs: list[dict[str, Any]] = []
    for session in session_store.list_sessions():
        session_id = (
            str(session.get("session_id") or "")
            if isinstance(session, dict)
            else str(session)
        )
        if not session_id:
            continue
        for run in session_store.list_agent_runs(session_id):
            if str(run.get("status") or "") != "failed":
                continue
            runs.append(
                {
                    "run_id": str(run.get("run_id") or ""),
                    "session_id": str(run.get("session_id") or ""),
                    "question": str(run.get("question") or ""),
                    "created_at": str(run.get("created_at") or ""),
                }
            )
            if len(runs) >= max(1, int(limit)):
                return runs
    return runs


async def replay_failed_run(
    session_store: Any,
    run_id: str,
    *,
    stream_fn: Callable[
        [StreamQueryRequest, Callable[[], Awaitable[bool]] | None],
        AsyncIterator[dict[str, Any]],
    ],
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
) -> dict[str, Any]:
    """按运行 ID 重放一次失败运行，复用原会话且不重复创建。

    返回重放摘要：原运行 ID、会话 ID、事件数、最终状态与错误信息；
    运行不存在、会话不存在或状态不是 failed 时返回 skipped 摘要。
    """
    run = session_store.get_agent_run(run_id)
    if run is None:
        return {"run_id": run_id, "skipped": True, "reason": "运行不存在"}
    if str(run.get("status") or "") != "failed":
        return {
            "run_id": run_id,
            "skipped": True,
            "reason": f"状态不是 failed：{run.get('status')}",
        }
    session_id = str(run.get("session_id") or "")
    if not session_id or session_store.get_session(session_id) is None:
        return {"run_id": run_id, "skipped": True, "reason": "会话不存在"}
    request = StreamQueryRequest(
        question=str(run.get("question") or ""),
        session_id=session_id,
    )
    events: list[dict[str, Any]] = []
    error = ""
    try:
        async for event in stream_fn(request, is_disconnected):
            events.append(event)
    except Exception as exc:
        error = str(exc)
    done_events = [
        event
        for event in events
        if isinstance(event, dict) and event.get("type") == "done"
    ]
    status = (
        str(done_events[-1].get("status") or "completed")
        if done_events
        else ("failed" if error else "interrupted")
    )
    return {
        "run_id": run_id,
        "session_id": session_id,
        "skipped": False,
        "events": len(events),
        "status": status,
        "error": error,
    }
