"""知识库重建与聊天图片证据访问接口。"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.api.dependencies import AdminUserDep, ServicesDep


router = APIRouter()


@router.post("/api/documents/index")
def index_document_pdfs(services: ServicesDep, user: AdminUserDep):
    """用统一自然段方案重建主向量库、图片索引和精简标准个例。"""
    try:
        return services.knowledge_service.rebuild_documents()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document indexing failed: {exc}",
        ) from exc


@router.get("/api/image-evidence/{image_id}")
def get_image_evidence_file(image_id: str, services: ServicesDep):
    """返回已登记的图片证据文件。"""
    image = services.image_store.get_image(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image evidence not found")
    root = services.settings.document_images_dir.resolve()
    raw_path = Path(image.image_path)
    candidates = [
        raw_path,
        root / raw_path.parent.name / raw_path.name,
        root / Path(image.source_pdf).stem / raw_path.name,
        root / raw_path.name,
    ]
    # 元数据可能保存旧机器绝对路径，只允许重定位到当前受控图片根目录。
    image_path = next(
        (candidate.resolve() for candidate in candidates if candidate.is_file()),
        raw_path.resolve(),
    )
    if root not in image_path.parents:
        raise HTTPException(
            status_code=403,
            detail="Image evidence path is outside document_images",
        )
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image evidence file not found")
    return FileResponse(image_path)
