"""供前端、Postman 和主项目调用的个例多维检索 FastAPI 接口。"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.services.agent.case_multidim_search.agent import CaseMultidimSearchAgent
from backend.app.config import settings
from backend.app.services.agent.Smart_Case_Match.infrastructure.data_store import LocalCaseDataStore
from backend.app.services.agent.case_multidim_search.integrations.local_stores import (
    LocalDocumentChunkStore,
    LocalImageEvidenceStore,
    LocalStandardCaseStore,
)
from backend.app.services.agent.case_multidim_search.core import progress as search_progress
from backend.app.services.agent.case_multidim_search.core.natural_query import (
    NaturalCaseQueryParser,
    NaturalConversationStore,
)
from backend.app.services.agent.case_multidim_search.integrations.non_thinking_llm import ensure_non_thinking_client
from backend.app.services.agent.case_multidim_search.reporting.report_preview import (
    ensure_report_thumbnail,
)
from backend.app.services.agent.case_multidim_search.schemas import (
    CaseAnalysisResponse,
    CasePdfExportRequest,
    NaturalCaseQueryParseRequest,
    NaturalCaseQueryParseResponse,
    NaturalCaseSearchRequest,
    NaturalConversationResetRequest,
    StructuredCaseSearchRequest,
)
from backend.app.services.model_client import get_llm_client


router = APIRouter(prefix="/api/case-multidim", tags=["case-multidim-search"])


RUNTIME_DIR_ENV = "CASE_MULTIDIM_RUNTIME_DIR"
_NATURAL_STORE: NaturalConversationStore | None = None


def _agent() -> CaseMultidimSearchAgent:
    """创建使用共享本地知识库的多维个例检索智能体。"""
    # 三类存储共享一个只读加载器，保证两个 Agent 使用同一份本地资料。
    data_store = LocalCaseDataStore(settings.smart_case_data_dir)
    runtime_dir = _runtime_dir()
    return CaseMultidimSearchAgent(
        standard_case_store=LocalStandardCaseStore(data_store),
        document_store=LocalDocumentChunkStore(data_store),
        image_store=LocalImageEvidenceStore(data_store),
        output_dir=runtime_dir / "reports",
        llm_client=ensure_non_thinking_client(get_llm_client()),
    )


def _runtime_dir() -> Path:
    """返回运行时输出目录；这里不是知识库，只保存报告、图表和远程图片缓存。"""
    configured = os.getenv(RUNTIME_DIR_ENV, "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent / "runtime"


def _natural_store(session_store=None) -> NaturalConversationStore:
    """独立页使用文件状态，主聊天页使用 sessions.sqlite3 中的会话状态。"""
    if session_store is not None:
        return NaturalConversationStore(session_store=session_store)
    global _NATURAL_STORE
    if _NATURAL_STORE is None:
        _NATURAL_STORE = NaturalConversationStore(_runtime_dir() / "natural_conversations")
    return _NATURAL_STORE


@router.get("/health")
def health():
    """检查共享本地知识库、文档片段、图片和中文字体状态。"""
    return _agent().health()


@router.get("/chat")
def chat_page():
    """返回个例多维检索智能体的结构化检索工作台页面。"""
    page_path = Path(__file__).parent / "pages" / "chat_page.html"
    return FileResponse(page_path, media_type="text/html; charset=utf-8")


@router.get("/natural-chat")
def natural_chat_page():
    """返回纯自然语言输入的多维个例检索聊天页面。"""
    page_path = Path(__file__).parent / "pages" / "natural_chat_page.html"
    return FileResponse(page_path, media_type="text/html; charset=utf-8")


@router.post("/natural/parse", response_model=NaturalCaseQueryParseResponse)
def natural_query_parse(payload: NaturalCaseQueryParseRequest):
    """解析自然语言并返回待确认条件，不在此阶段执行检索或污染历史上下文。"""
    try:
        parser = NaturalCaseQueryParser(ensure_non_thinking_client(get_llm_client()))
        return _natural_store().propose(payload.conversation_id, payload.message, parser)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"自然语言条件解析失败：{exc}") from exc


@router.post("/natural/search", response_model=CaseAnalysisResponse)
def natural_search(payload: NaturalCaseSearchRequest):
    """确认当前条件提案后，完整复用现有结构化检索与报告生成流程。"""
    try:
        request, original_message = _natural_store().confirm(
            payload.conversation_id,
            payload.proposal_id,
            payload.progress_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    search_progress.start(request.progress_id)
    try:
        result = _agent().analyze_structured(request)
        # 仅追加聊天入口审计信息，报告正文和 PDF 继续复用原结构化响应。
        result.question = original_message or result.question
        result.audit.update(
            {
                "input_mode": "natural_language_confirmed",
                "conversation_id": payload.conversation_id,
                "proposal_id": payload.proposal_id,
            }
        )
        result.search_response.audit.update(
            {
                "input_mode": "natural_language_confirmed",
                "conversation_id": payload.conversation_id,
            }
        )
        search_progress.complete(request.progress_id)
        return result
    except Exception as exc:
        search_progress.fail(request.progress_id, f"检索失败：{exc}")
        raise HTTPException(status_code=500, detail=f"自然语言多维个例检索失败：{exc}") from exc


@router.post("/natural/reset")
def natural_conversation_reset(payload: NaturalConversationResetRequest):
    """清空当前已确认上下文并创建全新聊天会话。"""
    return {"conversation_id": _natural_store().reset(payload.conversation_id)}


@router.get("/search/progress/{progress_id}")
def structured_search_progress(progress_id: str):
    """返回当前检索请求的轻量进度，不包含个例正文或模型输出。"""
    state = search_progress.get(progress_id)
    if state is None:
        raise HTTPException(status_code=404, detail="检索进度不存在")
    return state


@router.post("/search", response_model=CaseAnalysisResponse)
def structured_search(payload: StructuredCaseSearchRequest):
    """按结构化表单条件执行全量检索分析，并持续更新逐例处理进度。"""
    search_progress.start(payload.progress_id)
    try:
        result = _agent().analyze_structured(payload)
        search_progress.complete(payload.progress_id)
        return result
    except Exception as exc:
        search_progress.fail(payload.progress_id, f"检索失败：{exc}")
        raise HTTPException(status_code=500, detail=f"结构化个例多维检索失败：{exc}") from exc


@router.post("/search/export/csv")
def structured_export_csv(payload: StructuredCaseSearchRequest):
    """按结构化条件导出完整命中个例 CSV 表格。"""
    try:
        path = _agent().export_structured_csv(payload)
        return FileResponse(path, media_type="text/csv", filename="case-search.csv")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"结构化 CSV 导出失败：{exc}") from exc


@router.get("/export-pdf/progress/{progress_id}")
def export_pdf_progress(progress_id: str):
    """读取 PDF 导出的实时进度。"""
    state = search_progress.get(progress_id)
    if state is None:
        raise HTTPException(status_code=404, detail="PDF 导出进度不存在")
    return state


@router.post("/export-pdf")
def export_pdf(payload: CasePdfExportRequest):
    """根据已缓存的 answer_id 导出 PDF，并回传导出进度。"""
    progress_id = payload.progress_id or ""
    if progress_id:
        search_progress.start(progress_id)
        search_progress.update(progress_id, stage="pdf_prepare", message="正在准备 PDF 导出", percent=5)
    try:
        filename = payload.filename or "case-search-report.pdf"

        def update_pdf_progress(percent: int, message: str) -> None:
            """把 PDF 构建阶段进度同步给前端。"""
            search_progress.update(progress_id, stage="pdf_export", message=message, percent=percent)

        path = _agent().export_pdf_from_answer(
            payload.answer_id,
            filename=filename,
            report_title=payload.report_title,
            progress_callback=update_pdf_progress if progress_id else None,
        )
        if progress_id:
            search_progress.complete(progress_id, message="PDF 导出完成")
        return FileResponse(path, media_type="application/pdf", filename=Path(filename).name)
    except FileNotFoundError as exc:
        search_progress.fail(progress_id, f"PDF 导出失败：{exc}")
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        search_progress.fail(progress_id, f"PDF 导出失败：{exc}")
        raise HTTPException(status_code=500, detail=f"PDF 导出失败：{exc}") from exc


@router.get("/assets/images/{image_id}")
def get_image_asset(image_id: str):
    """读取共享本地知识库图片，供网页预览和 PDF 复用。"""
    try:
        return FileResponse(_agent().image_asset_path(image_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/reports/{filename}")
def get_report(filename: str):
    """安全读取统一聊天页生成的 PDF 报告。"""
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="报告文件名不合法")
    # 文件读取不初始化模型客户端，报告目录与 Agent 输出目录保持同一配置来源。
    path = _runtime_dir() / "reports" / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="报告不存在")
    return FileResponse(path, media_type="application/pdf", filename=safe_name)


@router.get("/report-thumbnails/{filename}")
def get_report_thumbnail(filename: str):
    """返回持久化的 PDF 第一页缩略图，并兼容自动补齐历史报告。"""
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.lower().endswith(".png"):
        raise HTTPException(status_code=400, detail="缩略图文件名不合法")
    reports_dir = _runtime_dir() / "reports"
    suffix = ".thumbnail.png"
    if not safe_name.endswith(suffix):
        raise HTTPException(status_code=404, detail="报告缩略图不存在")
    pdf_path = reports_dir / f"{safe_name[:-len(suffix)]}.pdf"
    try:
        # 每次都经过按需生成函数，以便 PDF 被同名更新后自动刷新旧缩略图。
        thumbnail_path = ensure_report_thumbnail(pdf_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="对应 PDF 报告不存在") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"报告缩略图生成失败：{exc}") from exc
    return FileResponse(thumbnail_path, media_type="image/png")


@router.get("/assets/{filename}")
def get_asset(filename: str):
    """读取分析阶段生成的图表资源，供 Postman 或前端预览。"""
    safe_name = Path(filename).name
    path = _agent().output_dir / "charts" / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="图表资源不存在")
    return FileResponse(path)

