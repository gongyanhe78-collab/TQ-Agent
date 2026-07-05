from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query


router = APIRouter()


def _main():
    """请求执行时获取 main 模块中的标准化个例服务实例。"""
    from backend.app import main

    return main


@router.post("/api/standard-cases/build")
def build_standard_cases(use_llm: bool = Query(default=True)):
    """从 document_index 的 chunk 构建第一期标准化个例层。"""
    main = _main()
    try:
        return main._build_standard_cases(use_llm=use_llm)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Standard case build failed: {exc}") from exc


@router.get("/api/standard-cases")
def list_standard_cases(
    q: str | None = None,
    date: str | None = None,
    disaster_type: str | None = None,
    area: str | None = None,
    image_type: str | None = None,
    data_category: str | None = None,
    source_pdf: str | None = None,
):
    """列出标准化个例，支持字段、图片类型和自然语言组合检索。"""
    main = _main()
    cases = main._search_standard_cases(
        q=q,
        date=date,
        disaster_type=disaster_type,
        area=area,
        image_type=image_type,
        data_category=data_category,
        source_pdf=source_pdf,
    )
    inferred = main._parse_standard_case_query(q or "")
    response_image_type = image_type or inferred.get("image_type")
    response_data_category = data_category or inferred.get("data_category")
    return {
        "count": len(cases),
        "cases": [
            main._standard_case_to_response(
                case,
                image_type=response_image_type,
                data_category=response_data_category,
            )
            for case in cases
        ],
    }


@router.get("/api/standard-cases/similar")
def match_similar_standard_cases(
    q: str | None = None,
    date: str | None = None,
    disaster_type: str | None = None,
    area: str | None = None,
    image_type: str | None = None,
    data_category: str | None = None,
    source_pdf: str | None = None,
    top_n: int = Query(default=5, ge=1, le=5),
):
    """相似个例智能匹配，综合结构化字段、图像证据、时空和灾种相似度排序。"""
    main = _main()
    matches = main._match_similar_standard_cases(
        q=q,
        date=date,
        disaster_type=disaster_type,
        area=area,
        image_type=image_type,
        data_category=data_category,
        source_pdf=source_pdf,
        top_n=top_n,
    )
    return {
        "count": len(matches),
        "matches": [main._similar_case_match_to_response(match) for match in matches],
    }


@router.get("/api/standard-cases/{case_id}")
def get_standard_case(case_id: str):
    """获取单条标准化个例详情。"""
    main = _main()
    case = main.standard_case_store.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Standard case not found")
    return main._standard_case_to_response(case)
