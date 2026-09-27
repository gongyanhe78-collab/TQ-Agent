"""保存多维检索请求的轻量进度状态。"""

from __future__ import annotations

from copy import deepcopy
from threading import Lock
from time import time


_LOCK = Lock()
_STATES: dict[str, dict] = {}
_MAX_AGE_SECONDS = 3600


def start(progress_id: str) -> None:
    """初始化检索进度，并顺便清理一小时前的历史状态。"""
    if not progress_id:
        return
    now = time()
    with _LOCK:
        expired = [key for key, value in _STATES.items() if now - value["updated_at"] > _MAX_AGE_SECONDS]
        for key in expired:
            _STATES.pop(key, None)
        _STATES[progress_id] = {
            "progress_id": progress_id,
            "status": "running",
            "stage": "json_filter",
            "message": "正在筛选标准化个例",
            "percent": 5,
            "completed_cases": 0,
            "total_cases": 0,
            "updated_at": now,
        }


def update(
    progress_id: str,
    *,
    stage: str,
    message: str,
    percent: int,
    completed_cases: int | None = None,
    total_cases: int | None = None,
) -> None:
    """更新阶段、百分比和逐例完成数量。"""
    if not progress_id:
        return
    with _LOCK:
        state = _STATES.setdefault(progress_id, {"progress_id": progress_id})
        state.update(
            {
                "status": "running",
                "stage": stage,
                "message": message,
                "percent": min(99, max(0, int(percent))),
                "updated_at": time(),
            }
        )
        if completed_cases is not None:
            state["completed_cases"] = max(0, int(completed_cases))
        if total_cases is not None:
            state["total_cases"] = max(0, int(total_cases))


def complete(progress_id: str, message: str = "检索分析完成") -> None:
    """标记检索、图表和结果缓存全部完成。"""
    if not progress_id:
        return
    with _LOCK:
        state = _STATES.setdefault(progress_id, {"progress_id": progress_id})
        total = int(state.get("total_cases") or 0)
        state.update(
            {
                "status": "completed",
                "stage": "completed",
                "message": message or "检索分析完成",
                "percent": 100,
                "completed_cases": total,
                "updated_at": time(),
            }
        )


def fail(progress_id: str, message: str) -> None:
    """记录失败状态，错误信息只保留适合前端展示的简短文本。"""
    if not progress_id:
        return
    with _LOCK:
        state = _STATES.setdefault(progress_id, {"progress_id": progress_id})
        state.update(
            {
                "status": "failed",
                "stage": "failed",
                "message": str(message or "检索失败")[:200],
                "updated_at": time(),
            }
        )


def get(progress_id: str) -> dict | None:
    """返回状态副本，避免前端读取期间被工作线程原地修改。"""
    with _LOCK:
        state = _STATES.get(progress_id)
        return deepcopy(state) if state is not None else None
