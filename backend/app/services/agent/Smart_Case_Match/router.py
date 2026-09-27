"""相似个例智能匹配体的 FastAPI 路由。"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from .workflow import progress
from .agent import SmartCaseMatchAgent
from .schemas import NaturalLanguageMatchRequest, SmartCaseMatchRequest, SmartCaseMatchResponse
from .infrastructure.stream_events import encode_sse


logger = logging.getLogger("uvicorn.error")


class _SmartCaseRuntime:
    """在线程安全的前提下持有可迁移的 Agent 单例和关闭状态。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self._agent: SmartCaseMatchAgent | None = None
        self._accepting = True

    def start(self) -> None:
        """允许新的应用生命周期按需创建 Agent。"""
        with self._lock:
            self._accepting = True

    def get_agent(self) -> SmartCaseMatchAgent:
        """首次访问时创建一次 Agent，并让后续请求复用同一连接池。"""
        with self._lock:
            if not self._accepting:
                raise RuntimeError("相似个例服务正在关闭")
            # 初始化过程放在锁内，避免并发首个请求创建多个 Agent 和连接池。
            if self._agent is None:
                self._agent = SmartCaseMatchAgent()
            return self._agent

    def close(self) -> None:
        """停止分发新实例并关闭已创建的 Agent；从未使用时保持空操作。"""
        with self._lock:
            if not self._accepting and self._agent is None:
                return
            self._accepting = False
            agent = self._agent
            # 先摘除单例，确保关闭后的线程池不会再次返回给后续调用方。
            self._agent = None
        if agent is not None:
            agent.close()


_runtime = _SmartCaseRuntime()


@asynccontextmanager
async def _router_lifespan(_: Any):
    """让本目录随 Router 一起迁移，并在 Ctrl+C 关闭时自动释放自有资源。"""
    _runtime.start()
    try:
        yield
    finally:
        try:
            _runtime.close()
            logger.info("[SmartCaseMatch] Agent 与共享连接池已关闭")
        except Exception as exc:
            # 关闭异常只能记录，不能阻止 FastAPI 继续完成应用退出。
            logger.exception("[SmartCaseMatch] 关闭 Agent 资源失败 error=%s", exc)


router = APIRouter(
    prefix="/api/smart-case-match",
    tags=["smart-case-match"],
    lifespan=_router_lifespan,
)


def _agent() -> SmartCaseMatchAgent:
    """从本目录自己的运行时获取长期复用的 Agent。"""
    return _runtime.get_agent()


@router.get("/health")
def health():
    """检查本地资料、Embedding、LLM 和 LangGraph 状态。"""
    try:
        return _agent().health()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"相似个例智能体健康检查失败：{exc}") from exc


@router.get("/chat")
def chat_page():
    """返回独立的相似个例匹配工作台。"""
    return FileResponse(Path(__file__).parent / "pages" / "chat_page.html", media_type="text/html; charset=utf-8")


@router.get("/conversation")
def conversation_page():
    """返回自然语言输入的流式相似个例聊天页面。"""
    return FileResponse(Path(__file__).parent / "pages" / "conversation_page.html", media_type="text/html; charset=utf-8")


@router.get("/progress/{progress_id}")
def match_progress(progress_id: str):
    """返回轻量任务进度，不包含业务正文和模型输入。"""
    state = progress.get(progress_id)
    if state is None:
        raise HTTPException(status_code=404, detail="匹配进度不存在")
    return state


@router.post("/search", response_model=SmartCaseMatchResponse)
def search(payload: SmartCaseMatchRequest):
    """执行结构化召回、语义补充、重排、逐例提炼和综合提示生成。"""
    try:
        return _agent().run(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"相似个例匹配失败：{exc}") from exc


@router.post("/conversation/stream")
async def stream_conversation(payload: NaturalLanguageMatchRequest, request: Request):
    """按 SSE 事件依次返回自然语言解析、匹配个例和综合研判。"""
    async def generate():
        async for event, data in _agent().stream_natural_match(payload, request.is_disconnected):
            yield encode_sse(event, data)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/assets/chunks/{chunk_id}")
def chunk_asset(chunk_id: str):
    """按 chunk_id 返回文字证据正文，供页面弹窗核验。"""
    try:
        return _agent().chunk_detail(chunk_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/assets/images/{image_id}")
def image_asset(image_id: str):
    """按 image_id 读取本地图片，避免向页面暴露磁盘路径。"""
    try:
        return FileResponse(_agent().image_path(image_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

