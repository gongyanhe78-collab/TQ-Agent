from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.api.schemas import DeleteVectorsRequest


router = APIRouter()


def _main():
    """请求执行时获取 main 模块中的向量库服务，兼容现有测试 patch。"""
    from backend.app import main

    return main


@router.post("/api/index")
def index_all_cases():
    """读取所有 txt 个例，生成向量并重建原个例向量库。"""
    main = _main()
    try:
        return main._index_all_txt_cases()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc


@router.get("/api/cases")
def list_cases():
    """列出原个例向量库中的所有案例 ID。"""
    main = _main()
    case_ids = main.case_store.list_case_ids()
    return {
        "case_ids": case_ids,
        "count": len(case_ids),
        **main.case_store.collection_info(),
    }


@router.get("/api/vectors/keys")
def list_vector_keys():
    """只读取原个例向量库中的 key，不触发 PDF 扫描、抽取或入库。"""
    main = _main()
    case_ids = main.case_store.list_case_ids()
    vector_info = main.case_store.collection_info()
    return {
        "case_ids": case_ids,
        "count": len(case_ids),
        "dimension": vector_info["dimension"],
        "source": vector_info["source"],
        "vector_count": vector_info["count"],
    }


@router.post("/api/documents/index")
def index_document_pdfs():
    """把 resource 目录下 PDF 文档化、切 chunk，并写入独立文档向量库。"""
    main = _main()
    try:
        return main._index_document_pdfs()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document indexing failed: {exc}") from exc


@router.get("/api/documents/keys")
def list_document_chunk_keys():
    """只读取新文档向量库中的 chunk key。"""
    main = _main()
    chunk_ids = main.document_store.list_chunk_ids()
    vector_info = main.document_store.collection_info()
    return {
        "chunk_ids": chunk_ids,
        "count": len(chunk_ids),
        "dimension": vector_info["dimension"],
        "source": vector_info["source"],
        "vector_count": vector_info["count"],
    }


@router.get("/api/documents/chunks/{chunk_id}")
def get_document_chunk(chunk_id: str):
    """获取单个文档 chunk 详情。"""
    main = _main()
    chunk = main.document_store.get_chunk(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="Document chunk not found")
    return main._document_chunk_to_response(chunk)


@router.get("/api/image-evidence/{image_id}")
def get_image_evidence_file(image_id: str):
    """返回已登记的图片证据文件。"""
    main = _main()
    image = main.image_evidence_store.get_image(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image evidence not found")
    image_path = Path(image.image_path).resolve()
    root = main.settings.document_images_dir.resolve()
    if root not in image_path.parents:
        raise HTTPException(status_code=403, detail="Image evidence path is outside document_images")
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image evidence file not found")
    return FileResponse(image_path)


@router.get("/api/cases/{case_id}")
def get_case(case_id: str):
    """获取单个原个例详情。"""
    main = _main()
    case = main.case_store.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return main._case_to_response(case)


@router.post("/api/vectors/delete")
def delete_vectors(payload: DeleteVectorsRequest):
    """按向量 key 批量删除原个例向量库中的内容。"""
    main = _main()
    keys = [key.strip() for key in payload.keys if key and key.strip()]
    if not keys:
        raise HTTPException(status_code=400, detail="At least one vector key is required")
    result = main.case_store.delete_cases(keys)
    vector_info = main.case_store.collection_info()
    return {
        **result,
        "vector_count": vector_info["count"],
        "vector_dimension": vector_info["dimension"],
        "vector_source": vector_info["source"],
    }
