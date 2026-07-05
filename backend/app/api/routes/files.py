from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from backend.app.config import settings


router = APIRouter()


def _main():
    """请求执行时获取 main 模块中的文件处理函数和服务实例。"""
    from backend.app import main

    return main


@router.post("/api/extract")
def extract_all_pdfs():
    """批量提取 resource 目录下所有 PDF 个例，保存为 txt 文件。"""
    main = _main()
    written_files = main._extract_all_pdfs()
    return {"written_files": written_files, "count": len(written_files)}


@router.post("/api/refresh")
def refresh_case_library():
    """仅对新增或变更的 PDF 执行增量提取和索引。"""
    return _main()._refresh_incremental_case_library()


@router.post("/api/materials")
async def upload_material(request: Request):
    """上传一个 PDF 文件并仅对该文件执行增量处理。"""
    main = _main()
    filename = main._safe_pdf_filename(request.headers.get("x-filename", ""))
    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty")
    settings.resource_dir.mkdir(parents=True, exist_ok=True)
    target_path = settings.resource_dir / filename
    target_path.write_bytes(payload)
    return {
        "uploaded_file": filename,
        **main._index_document_pdfs([target_path]),
    }


@router.post("/api/materials/batch")
async def upload_materials_batch(files: list[UploadFile] = File(...)):
    """批量上传 PDF 后，只对本次上传文件执行增量抽取和入库。"""
    main = _main()
    settings.resource_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for upload in files:
        filename = main._safe_pdf_filename(upload.filename or "")
        content = await upload.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"Uploaded PDF is empty: {filename}")
        target_path = settings.resource_dir / filename
        target_path.write_bytes(content)
        saved_paths.append(target_path)
    return {
        "uploaded_files": [path.name for path in saved_paths],
        **main._index_document_pdfs(saved_paths),
    }
