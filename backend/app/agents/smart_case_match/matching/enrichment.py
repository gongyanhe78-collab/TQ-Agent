"""入选个例的正文丰富、证据图片准备和历史预警抽取。"""
from __future__ import annotations

import re
from datetime import date as date_type
from typing import Any

from ..infrastructure.data_store import LocalCaseDataStore


FOCUS_TERMS = ("天气实况", "实况", "环流形势", "影响系统", "演变", "降水", "大风", "温度", "预警", "服务")
WARNING_TERMS = ("预警", "警报", "预报服务", "服务材料", "应急响应")
# 先扩大候选图片范围，最终主证据和补充资料仍由结果组装节点分别限量。
IMAGE_CANDIDATE_LIMIT = 18


def enrich_selected_cases(
    query: dict[str, Any],
    selected_cases: list[dict[str, Any]],
    store: LocalCaseDataStore,
) -> list[dict[str, Any]]:
    """为每个入选个例准备精选正文、历史预警和图片证据。"""
    result = []
    for case in selected_cases:
        chunks = store.get_case_chunks(case)
        selected_chunks = _select_relevant_chunks(query, case, chunks, limit=6)
        context = "\n\n".join(str(item.get("content") or "") for item in selected_chunks)
        images = store.get_case_images(case, limit=IMAGE_CANDIDATE_LIMIT) if query.get("include_images") else []
        result.append({
            **case,
            "relevant_chunks": [_public_chunk(item) for item in selected_chunks],
            "case_context": context[:12000],
            "historical_warnings": _extract_prior_warnings(case, chunks),
            "evidence_images": [_public_image(item) for item in images],
        })
    return result


def _select_relevant_chunks(
    query: dict[str, Any],
    case: dict[str, Any],
    chunks: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """按查询词、业务章节词和语义命中 ID 选择少量正文。"""
    semantic_ids = set(case.get("semantic_chunk_ids") or [])
    terms = set(query.get("disaster_types") or []) | set(query.get("affected_areas") or []) | set(FOCUS_TERMS)
    scored = []
    for chunk in chunks:
        text = str(chunk.get("content") or "")
        score = sum(1 for term in terms if term and term in text)
        if str(chunk.get("chunk_id")) in semantic_ids:
            score += 5
        if re.search(r"\d+(?:\.\d+)?\s*(?:mm|毫米|℃|m/s|米/秒|级|站)", text, flags=re.IGNORECASE):
            score += 2
        scored.append((score, int(chunk.get("chunk_no") or 0), chunk))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    chosen = [item[2] for item in scored[:limit]]
    return sorted(chosen, key=lambda item: int(item.get("chunk_no") or 0))


def _extract_prior_warnings(case: dict[str, Any], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """只从个例关联正文中提取明确早于过程开始时间的预警句。"""
    case_start = _case_start_date(case)
    if case_start is None:
        return []
    records = []
    seen = set()
    for chunk in chunks:
        content = str(chunk.get("content") or "")
        for sentence in re.split(r"(?<=[。；！？])", content):
            text = sentence.strip()
            if not text or not any(term in text for term in WARNING_TERMS):
                continue
            issue_date = _date_from_sentence(text, case_start.year)
            if issue_date is None or issue_date >= case_start:
                continue
            normalized = re.sub(r"\s+", "", text)
            if normalized in seen:
                continue
            seen.add(normalized)
            records.append({
                "issue_date": issue_date.isoformat(),
                "text": text[:500],
                "source_chunk_id": str(chunk.get("chunk_id") or ""),
            })
    records.sort(key=lambda item: item["issue_date"], reverse=True)
    return records[:5]


def _case_start_date(case: dict[str, Any]) -> date_type | None:
    """从标准日期、标题或来源文件中恢复个例开始日期。"""
    text = " ".join([str(case.get("date_range") or ""), str(case.get("title") or ""), str(case.get("source_pdf") or "")])
    year_match = re.search(r"(20\d{2})", text)
    start_match = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*(?:日|号|[-~～至到])", text)
    if not (year_match and start_match):
        return None
    try:
        return date_type(int(year_match.group(1)), int(start_match.group(1)), int(start_match.group(2)))
    except ValueError:
        return None


def _date_from_sentence(text: str, default_year: int) -> date_type | None:
    """从预警句中解析发布日期；缺少年份时沿用个例年份。"""
    match = re.search(r"(?:(20\d{2})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if not match:
        return None
    try:
        return date_type(int(match.group(1) or default_year), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _public_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """移除向量和本地绝对路径，仅保留生成与溯源需要的字段。"""
    return {
        "chunk_id": str(chunk.get("chunk_id") or ""),
        "chunk_no": int(chunk.get("chunk_no") or 0),
        "source_pdf": str(chunk.get("source_pdf") or ""),
        "content": str(chunk.get("content") or ""),
    }


def _public_image(image: dict[str, Any]) -> dict[str, Any]:
    """生成页面可用的图片描述，真实文件由图片接口按 ID 提供。"""
    image_id = str(image.get("image_id") or "")
    return {
        "image_id": image_id,
        "url": f"/api/smart-case-match/assets/images/{image_id}",
        "caption": str(image.get("caption") or ""),
        "page_no": int(image.get("page_no") or 0),
        "extraction_type": str(image.get("extraction_type") or ""),
    }

