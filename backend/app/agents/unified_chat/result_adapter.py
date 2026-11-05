"""把三个 Agent 的结果统一转换为主页面可消费的字段。"""
from __future__ import annotations

from typing import Any


def confirmation_answer(proposal) -> str:
    """把多维检索条件提案转换为同一聊天页面中的确认消息。"""
    lines = list(proposal.display_lines or [])
    warnings = list(proposal.warnings or [])
    parts = ["已解析出以下检索条件："]
    parts.extend(f"- {line}" for line in lines)
    if warnings:
        parts.append("")
        parts.extend(f"提示：{warning}" for warning in warnings)
    if proposal.can_confirm:
        parts.extend(["", "请回复“确认”开始检索分析；也可以直接补充或修改条件。"])
    return "\n".join(parts)


def adapt_multidim_result(result, document_store, reports: list[dict] | None = None) -> dict[str, Any]:
    """将多维检索分析对象转换为主页面统一结果。"""
    search = result.search_response
    evidence_cases = []
    chunk_ids: list[str] = []
    for hit in search.results:
        case = dict(hit.case)
        chunk_ids.extend(str(item) for item in case.get("source_chunk_ids") or [])
        evidence_cases.append(
            {
                **case,
                "matched_fields": list(hit.matched_fields or []),
                "score": float(hit.score or 0.0),
                "analysis": hit.analysis,
                "intensity_metrics": [item.model_dump() for item in hit.intensity_metrics],
            }
        )
    images = [_visual_to_image(item) for item in [*result.images, *result.charts]]
    agent_result = {
        "answer_id": result.answer_id,
        "title": result.title,
        "query_conditions": search.parsed_query.model_dump(),
        "analysis": dict(search.analysis or {}),
        "evidence_cases": evidence_cases,
        "images": images,
        "reports": list(reports or []),
    }
    return {
        "agent_type": "case_multidim_search",
        "run_id": result.answer_id,
        "question": result.question,
        "answer": result.answer,
        "retrieval_mode": "agent:case_multidim_search",
        "hit_count": int(search.result_count),
        "hits": [],
        "evidence_cases": evidence_cases,
        "evidence_chunks": _chunks_from_ids(document_store, chunk_ids),
        "images": images,
        "visuals": [],
        "reports": list(reports or []),
        "analysis": dict(search.analysis or {}),
        "query_conditions": search.parsed_query.model_dump(),
        "intent_trace": {"route": "case_multidim_search"},
        "execution_plan": {"agent": "case_multidim_search", "flow_preserved": True},
        "audit": dict(result.audit or {}),
        "agent_result": agent_result,
        "llm_used": True,
    }


def adapt_smart_result(result: dict[str, Any], document_store) -> dict[str, Any]:
    """将 Smart 最终协议转换为主页面统一结果。"""
    matched_cases = list(result.get("matched_cases") or [])
    chunk_ids = [
        str(chunk_id)
        for case in matched_cases
        for chunk_id in case.get("source_chunk_ids") or []
    ]
    images = _dedupe_images(
        [
            image
            for case in matched_cases
            for image in [*(case.get("evidence_images") or []), *(case.get("supplemental_images") or [])]
        ]
    )
    agent_result = {
        "run_id": str(result.get("run_id") or ""),
        "status": str(result.get("status") or "completed"),
        "query_summary": dict(result.get("query_summary") or {}),
        "matched_cases": matched_cases,
        "forecast_summary": dict(result.get("forecast_summary") or {}),
        "forecast_tips": list(result.get("forecast_tips") or []),
        "warnings": list(result.get("warnings") or []),
    }
    return {
        "agent_type": "smart_case_match",
        "run_id": str(result.get("run_id") or ""),
        "question": "",
        "answer": _smart_answer(result),
        "retrieval_mode": "agent:smart_case_match",
        "hit_count": len(matched_cases),
        "hits": [],
        "evidence_cases": matched_cases,
        "evidence_chunks": _chunks_from_ids(document_store, chunk_ids),
        "images": images[:12],
        "visuals": [],
        "reports": [],
        "analysis": {
            "forecast_summary": dict(result.get("forecast_summary") or {}),
            "forecast_tips": list(result.get("forecast_tips") or []),
        },
        "query_conditions": dict(result.get("query_summary") or {}),
        "intent_trace": {"route": "smart_case_match"},
        "execution_plan": {"agent": "smart_case_match", "flow_preserved": True},
        "audit": dict(result.get("audit") or {}),
        "agent_result": agent_result,
        "llm_used": True,
    }


def _chunks_from_ids(document_store, chunk_ids: list[str], limit: int = 24) -> list[dict]:
    """从统一主库读取少量可追溯正文，避免把向量写入会话数据库。"""
    result = []
    for chunk_id in dict.fromkeys(chunk_ids):
        chunk = document_store.get_chunk(chunk_id)
        if chunk is None:
            continue
        result.append(
            {
                "chunk_id": chunk.chunk_id,
                "chunk_no": chunk.chunk_no,
                "source_pdf": chunk.source_pdf,
                "content": chunk.content,
            }
        )
        if len(result) >= limit:
            break
    return result


def _visual_to_image(visual) -> dict:
    """统一数据库图片和生成图表的页面展示字段。"""
    return {
        "image_id": visual.visual_id,
        "url": visual.url,
        "caption": visual.title,
        "page_no": 0,
        "source_pdf": visual.source,
        "visual_type": visual.visual_type,
    }


def _dedupe_images(images: list[dict]) -> list[dict]:
    """按图片编号去重并过滤空资源。"""
    result = []
    seen = set()
    for image in images:
        image_id = str(image.get("image_id") or "")
        if not image_id or image_id in seen:
            continue
        seen.add(image_id)
        result.append(dict(image))
    return result


def _smart_answer(result: dict[str, Any]) -> str:
    """从 Smart 的结构化输出生成主聊天气泡中的可读答案。"""
    cases = list(result.get("matched_cases") or [])
    summary = dict(result.get("forecast_summary") or {})
    tips = list(result.get("forecast_tips") or [])
    lines = [f"共找到 {len(cases)} 个相似历史过程。"]
    for case in cases:
        score = float(case.get("retrieval_score") or 0.0)
        lines.append(f"{int(case.get('rank') or len(lines))}. {case.get('title') or case.get('case_id')}（匹配度 {score:.3f}）")
        reasons = list(case.get("match_reasons") or [])[:2]
        if reasons:
            lines.append("   " + "；".join(str(item) for item in reasons))
    assessment = str(summary.get("similarity_assessment") or "").strip()
    main_risk = str(summary.get("main_risk") or "").strip()
    if assessment:
        lines.extend(["", f"综合研判：{assessment}"])
    if main_risk:
        lines.append(f"主要风险：{main_risk}")
    tip_texts = [str(item.get("text") or "").strip() for item in tips if str(item.get("text") or "").strip()]
    if tip_texts:
        lines.extend(["", "参考经验："])
        lines.extend(f"- {text}" for text in tip_texts)
    warnings = [str(item) for item in result.get("warnings") or [] if str(item)]
    if warnings:
        lines.extend(["", "注意：" + "；".join(warnings)])
    return "\n".join(lines)
