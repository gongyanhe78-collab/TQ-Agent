"""个例多维检索智能体的 HTTP 路由。"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.agents.case_multidim_search.core import progress as search_progress
from backend.app.agents.case_multidim_search.reporting.report_preview import (
    ensure_report_thumbnail,
)
from backend.app.agents.case_multidim_search.schemas import (
    CaseAnalysisResponse,
    CasePdfExportRequest,
    NaturalCaseQueryParseRequest,
    NaturalCaseQueryParseResponse,
    NaturalCaseSearchRequest,
    NaturalConversationResetRequest,
    StructuredCaseSearchRequest,
)
from backend.app.api.dependencies import (
    CurrentUserDep,
    MultidimAgentDep,
    MultidimPageStoreDep,
    ServicesDep,
)

router = APIRouter(prefix="/api/case-multidim", tags=["case-multidim-search"])
CASE_AGENT_DIR = Path(__file__).resolve().parents[2] / "agents" / "case_multidim_search"


@router.get("/health")
def health(services: ServicesDep):
    """检查共享知识库、文档片段、图片和中文字体状态。"""
    return services.multidim_agent.health()


@router.get("/chat")
def chat_page():
    """返回个例多维检索智能体的结构化检索工作台页面。"""
    page_path = CASE_AGENT_DIR / "pages" / "chat_page.html"
    return FileResponse(page_path, media_type="text/html; charset=utf-8")


@router.get("/natural-chat")
def natural_chat_page():
    """返回纯自然语言输入的多维个例检索聊天页面。"""
    page_path = CASE_AGENT_DIR / "pages" / "natural_chat_page.html"
    return FileResponse(page_path, media_type="text/html; charset=utf-8")


@router.post("/natural/parse", response_model=NaturalCaseQueryParseResponse)
def natural_query_parse(
    payload: NaturalCaseQueryParseRequest,
    services: ServicesDep,
    page_store: MultidimPageStoreDep,
):
    """解析自然语言并返回待确认条件，不执行检索。"""
    try:
        return page_store.propose(
            payload.conversation_id,
            payload.message,
            services.natural_query_parser,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"自然语言条件解析失败：{exc}",
        ) from exc


@router.post("/natural/search", response_model=CaseAnalysisResponse)
def natural_search(
    payload: NaturalCaseSearchRequest,
    user: CurrentUserDep,
    agent: MultidimAgentDep,
    page_store: MultidimPageStoreDep,
):
    """确认当前条件提案后，复用结构化检索与报告生成流程。"""
    try:
        request, original_message = page_store.confirm(
            payload.conversation_id,
            payload.proposal_id,
            payload.progress_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    search_progress.start(
        request.progress_id,
        owner_id=str(user["user_id"]),
    )
    try:
        result = agent.analyze_structured(request)
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
        raise HTTPException(
            status_code=500,
            detail=f"自然语言多维个例检索失败：{exc}",
        ) from exc


@router.post("/natural/reset")
def natural_conversation_reset(
    payload: NaturalConversationResetRequest,
    page_store: MultidimPageStoreDep,
):
    """清空当前已确认上下文并创建全新聊天会话。"""
    return {
        "conversation_id": page_store.reset(payload.conversation_id)
    }


@router.get("/search/progress/{progress_id}")
def structured_search_progress(progress_id: str, user: CurrentUserDep):
    """返回当前检索请求的轻量进度。"""
    state = search_progress.get(progress_id, owner_id=str(user["user_id"]))
    if state is None:
        raise HTTPException(status_code=404, detail="检索进度不存在")
    return state


@router.post("/search", response_model=CaseAnalysisResponse)
def structured_search(
    payload: StructuredCaseSearchRequest,
    user: CurrentUserDep,
    agent: MultidimAgentDep,
):
    """按结构化表单条件执行全量检索分析。"""
    search_progress.start(
        payload.progress_id,
        owner_id=str(user["user_id"]),
    )
    try:
        result = agent.analyze_structured(payload)
        search_progress.complete(payload.progress_id)
        return result
    except Exception as exc:
        search_progress.fail(payload.progress_id, f"检索失败：{exc}")
        raise HTTPException(
            status_code=500,
            detail=f"结构化个例多维检索失败：{exc}",
        ) from exc


@router.post("/search/export/csv")
def structured_export_csv(
    payload: StructuredCaseSearchRequest,
    user: CurrentUserDep,
    agent: MultidimAgentDep,
):
    """按结构化条件导出完整命中个例 CSV 表格。"""
    try:
        path = agent.export_structured_csv(payload)
        return FileResponse(path, media_type="text/csv", filename="case-search.csv")
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"结构化 CSV 导出失败：{exc}",
        ) from exc


@router.get("/export-pdf/progress/{progress_id}")
def export_pdf_progress(progress_id: str, user: CurrentUserDep):
    """读取 PDF 导出的实时进度。"""
    state = search_progress.get(progress_id, owner_id=str(user["user_id"]))
    if state is None:
        raise HTTPException(status_code=404, detail="PDF 导出进度不存在")
    return state


@router.post("/export-pdf")
def export_pdf(
    payload: CasePdfExportRequest,
    user: CurrentUserDep,
    agent: MultidimAgentDep,
):
    """根据已缓存的 answer_id 导出 PDF 并回传导出进度。"""
    progress_id = payload.progress_id or ""
    if progress_id:
        search_progress.start(progress_id, owner_id=str(user["user_id"]))
        search_progress.update(
            progress_id,
            stage="pdf_prepare",
            message="正在准备 PDF 导出",
            percent=5,
        )
    try:
        filename = payload.filename or "case-search-report.pdf"

        def update_pdf_progress(percent: int, message: str) -> None:
            """把 PDF 构建阶段进度同步给前端。"""
            search_progress.update(
                progress_id,
                stage="pdf_export",
                message=message,
                percent=percent,
            )

        path = agent.export_pdf_from_answer(
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
def get_image_asset(
    image_id: str,
    services: ServicesDep,
    user: CurrentUserDep,
):
    """读取共享知识库图片，供网页预览和 PDF 复用。"""
    try:
        return FileResponse(services.multidim_agent.image_asset_path(image_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/reports/{filename}")
def get_report(
    filename: str,
    user: CurrentUserDep,
    agent: MultidimAgentDep,
):
    """安全读取统一聊天页生成的 PDF 报告。"""
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="报告文件名不合法")
    path = agent.output_dir / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="报告不存在")
    return FileResponse(path, media_type="application/pdf", filename=safe_name)


@router.get("/report-thumbnails/{filename}")
def get_report_thumbnail(
    filename: str,
    user: CurrentUserDep,
    agent: MultidimAgentDep,
):
    """返回持久化的 PDF 第一页缩略图，并兼容历史报告。"""
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.lower().endswith(".png"):
        raise HTTPException(status_code=400, detail="缩略图文件名不合法")
    reports_dir = agent.output_dir
    suffix = ".thumbnail.png"
    if not safe_name.endswith(suffix):
        raise HTTPException(status_code=404, detail="报告缩略图不存在")
    pdf_path = reports_dir / f"{safe_name[:-len(suffix)]}.pdf"
    try:
        thumbnail_path = ensure_report_thumbnail(pdf_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="对应 PDF 报告不存在") from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"报告缩略图生成失败：{exc}",
        ) from exc
    return FileResponse(thumbnail_path, media_type="image/png")


@router.get("/assets/{filename}")
def get_asset(
    filename: str,
    user: CurrentUserDep,
    agent: MultidimAgentDep,
):
    """读取分析阶段生成的图表资源。"""
    safe_name = Path(filename).name
    path = agent.output_dir / "charts" / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="图表资源不存在")
    return FileResponse(path)
