"""主知识库重建与聊天图片证据访问接口。"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

# API 路由实例
router = APIRouter()


def _main():
    """
    延迟加载 main 模块中的服务实例
    避免循环导入问题，在请求实际执行时才获取向量库服务。
    """
    from backend.app import main

    return main


@router.post("/api/documents/index")
def index_document_pdfs():
    """用统一自然段方案重建主向量库、图片索引和精简标准个例。"""
    main = _main()
    try:
        return main._index_document_pdfs()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document indexing failed: {exc}") from exc


@router.get("/api/image-evidence/{image_id}")
def get_image_evidence_file(image_id: str):
    """返回已登记的图片证据文件。"""
    main = _main()
    image = main.image_evidence_store.get_image(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image evidence not found")
    root = main.settings.document_images_dir.resolve()
    raw_path = Path(image.image_path)
    candidates = [
        raw_path,
        root / raw_path.parent.name / raw_path.name,
        root / Path(image.source_pdf).stem / raw_path.name,
        root / raw_path.name,
    ]
    # 元数据可能保存旧机器绝对路径；只允许重定位到当前受控图片根目录。
    image_path = next(
        (candidate.resolve() for candidate in candidates if candidate.is_file()),
        raw_path.resolve(),
    )
    if root not in image_path.parents:
        raise HTTPException(status_code=403, detail="Image evidence path is outside document_images")
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image evidence file not found")
    return FileResponse(image_path)
