"""相似个例匹配任务的进程内轻量进度状态。"""
from __future__ import annotations

from copy import deepcopy
from threading import Lock
from time import monotonic


_LOCK = Lock()
_STATES: dict[str, dict] = {}
# 客户端进度号只作为查询别名，真实状态始终使用服务端唯一 run_id 保存。
_ALIASES: dict[str, str] = {}
_TERMINAL_STATE_TTL_SECONDS = 2 * 60 * 60
_RUNNING_STATE_TTL_SECONDS = 24 * 60 * 60
_MAX_TERMINAL_STATES = 2000


def _purge_terminal_states_locked(now: float | None = None) -> None:
    """清理已结束且过期的进度，运行中的任务不会因 TTL 被中途删除。"""
    current = now if now is not None else monotonic()
    expired = [
        progress_id
        for progress_id, state in _STATES.items()
        if (
            state.get("status") in {"completed", "failed"}
            and current - float(state.get("_updated_monotonic") or current) >= _TERMINAL_STATE_TTL_SECONDS
        )
        or (
            state.get("status") == "running"
            and current - float(state.get("_updated_monotonic") or current) >= _RUNNING_STATE_TTL_SECONDS
        )
    ]
    for progress_id in expired:
        _STATES.pop(progress_id, None)
    # 任务状态删除后同步清理别名，避免长期运行时积累失效映射。
    stale_aliases = [alias for alias, run_id in _ALIASES.items() if run_id not in _STATES]
    for alias in stale_aliases:
        _ALIASES.pop(alias, None)
    terminal_ids = [
        (progress_id, float(state.get("_updated_monotonic") or current))
        for progress_id, state in _STATES.items()
        if state.get("status") in {"completed", "failed"}
    ]
    if len(terminal_ids) > _MAX_TERMINAL_STATES:
        terminal_ids.sort(key=lambda item: item[1])
        for progress_id, _ in terminal_ids[: len(terminal_ids) - _MAX_TERMINAL_STATES]:
            _STATES.pop(progress_id, None)
        stale_aliases = [alias for alias, run_id in _ALIASES.items() if run_id not in _STATES]
        for alias in stale_aliases:
            _ALIASES.pop(alias, None)


def start(progress_id: str, alias: str = "") -> None:
    """使用唯一运行号初始化任务，并可注册一个向后兼容的客户端查询别名。"""
    if not progress_id:
        return
    with _LOCK:
        _purge_terminal_states_locked()
        _STATES[progress_id] = {
            "progress_id": progress_id,
            "status": "running",
            "stage": "starting",
            "message": "正在准备匹配",
            "percent": 1,
            "_updated_monotonic": monotonic(),
        }
        normalized_alias = str(alias or "").strip()
        if normalized_alias and normalized_alias != progress_id:
            # 同一客户端别名重复提交时仅更新查询指向，不会覆盖两个任务各自的真实状态。
            _ALIASES[normalized_alias] = progress_id


def update(progress_id: str, stage: str, message: str, percent: int) -> None:
    """更新任务阶段，不保存业务正文和模型输入。"""
    if not progress_id:
        return
    with _LOCK:
        _purge_terminal_states_locked()
        state = _STATES.setdefault(progress_id, {"progress_id": progress_id})
        state.update({"status": "running", "stage": stage, "message": message, "percent": max(0, min(100, int(percent))), "_updated_monotonic": monotonic()})


def complete(progress_id: str) -> None:
    """标记任务执行完成。"""
    if not progress_id:
        return
    with _LOCK:
        _purge_terminal_states_locked()
        state = _STATES.setdefault(progress_id, {"progress_id": progress_id})
        state.update({"status": "completed", "stage": "completed", "message": "相似个例匹配完成", "percent": 100, "_updated_monotonic": monotonic()})


def fail(progress_id: str, message: str) -> None:
    """记录对用户可读的失败信息。"""
    if not progress_id:
        return
    with _LOCK:
        _purge_terminal_states_locked()
        state = _STATES.setdefault(progress_id, {"progress_id": progress_id})
        state.update({"status": "failed", "stage": "failed", "message": str(message), "percent": 100, "_updated_monotonic": monotonic()})


def get(progress_id: str) -> dict | None:
    """返回进度快照，避免外部修改共享状态。"""
    with _LOCK:
        _purge_terminal_states_locked()
        # 真实 run_id 优先；旧页面传客户端 progress_id 时再解析到最近注册的唯一任务。
        resolved_id = progress_id if progress_id in _STATES else _ALIASES.get(progress_id, progress_id)
        state = _STATES.get(resolved_id)
        if not state:
            return None
        snapshot = deepcopy(state)
        snapshot.pop("_updated_monotonic", None)
        return snapshot
