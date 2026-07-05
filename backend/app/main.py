"""
FastAPI 主应用模块，定义所有 API 接口，整合各服务组件提供完整的问答系统功能。"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.agent import router as agent_router
from backend.app.api.routes.files import router as files_router
from backend.app.api.routes.sessions import router as sessions_router
from backend.app.api.routes.standard_cases import router as standard_cases_router
from backend.app.api.routes.system import router as system_router
from backend.app.api.routes.vectors import router as vectors_router
from backend.app.api.schemas import (
    AgentQueryRequest,
    CreateSessionRequest,
    DeleteVectorsRequest,
    QueryRequest,
    StreamQueryRequest,
)
from backend.app.config import settings
from backend.app.models import DocumentChunk, DocumentRetrievalHit, ExtractedCase
from backend.app.services.build_status import BuildStatusStore
from backend.app.services.document_indexing import DocumentChunkIndexer
from backend.app.services.document_store import ChromaDocumentChunkStore
from backend.app.services.embedding_client import DashScopeEmbeddingClient
from backend.app.services.extraction_pipeline import ExtractionPipeline
from backend.app.services.extraction_service import CaseExtractionService
from backend.app.services.image_extraction import ImageEvidenceStore
from backend.app.services.llm_client import DashScopeChatClient
from backend.app.services.processed_files import ProcessedFileStore
from backend.app.services.rerank_client import DashScopeRerankClient, ModelFirstReranker
from backend.app.services.retrieval import KeywordReranker, RagService, SimpleAnswerGenerator
from backend.app.services.session_store import SessionStore
from backend.app.services.similar_case_matcher import SimilarCaseMatcher, SimilarCaseQuery
from backend.app.services.standard_case_builder import StandardCaseBuilder
from backend.app.services.standard_case_store import JsonStandardCaseStore
from backend.app.services.vector_store import ChromaCaseStore


app = FastAPI(title="Weather Case RAG System", version="0.1.0")
# 配置 CORS 跨域支持（允许所有来源访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化核心服务组件。
embedding_client = DashScopeEmbeddingClient()
llm_client = DashScopeChatClient()
keyword_reranker = KeywordReranker()
rerank_client = DashScopeRerankClient()
case_store = ChromaCaseStore(settings.index_dir, settings.collection_name)
document_store = ChromaDocumentChunkStore(
    settings.document_index_dir,
    settings.document_collection_name,
)
image_evidence_store = ImageEvidenceStore(settings.image_metadata_path)
standard_case_store = JsonStandardCaseStore(settings.standard_cases_path)
build_status_store = BuildStatusStore(settings.build_status_path)
processed_file_store = ProcessedFileStore(settings.processed_files_path)
session_store = SessionStore(settings.session_db_path)
rag_service = RagService(
    embedding_client=embedding_client,
    case_store=case_store,
    reranker=ModelFirstReranker(
        model_reranker=rerank_client,
        fallback_reranker=keyword_reranker,
    ),
    answer_generator=SimpleAnswerGenerator(llm_client=llm_client),
)
# 案例提取服务和流水线
extractor = CaseExtractionService(llm_client=llm_client)
pipeline = ExtractionPipeline(extractor=extractor, output_dir=settings.extracted_dir)
document_indexer = DocumentChunkIndexer(embedding_client=embedding_client)
standard_case_builder = StandardCaseBuilder(
    image_store=image_evidence_store,
    llm_client=llm_client,
)
similar_case_matcher = SimilarCaseMatcher(
    image_store=image_evidence_store,
    image_filter=lambda image, image_type, data_category: _image_matches_filter(
        image,
        image_type=image_type,
        data_category=data_category,
    ),
)


def _case_to_response(case: ExtractedCase) -> dict:
    """
    将案例对象转换为 API 响应格式（内部函数）

    Args:
        case: ExtractedCase 案例对象

    Returns:
        案例字典
    """
    return {
        "source_pdf": case.source_pdf,
        "case_id": case.case_id,
        "case_no": case.case_no,
        "title": case.title,
        "date_range": case.date_range,
        "content": case.content,
        "file_path": case.file_path,
    }


def _hit_to_response(hit) -> dict:
    """
    将搜索命中结果转换为 API 响应格式（内部函数）

    Args:
        hit: RetrievalHit 命中对象

    Returns:
        命中字典（包含得分和案例）
    """
    return {
        "score": hit.score,
        "case": _case_to_response(hit.case),
    }


def _document_chunk_to_response(chunk: DocumentChunk) -> dict:
    """将文档 chunk 转换为前端证据片段响应。"""
    return {
        "source_pdf": chunk.source_pdf,
        "chunk_id": chunk.chunk_id,
        "chunk_no": chunk.chunk_no,
        "content": chunk.content,
        "file_path": chunk.file_path,
    }


def _document_hit_to_response(hit: DocumentRetrievalHit) -> dict:
    """将文档 chunk 检索命中转换为 API 响应。"""
    return {
        "score": hit.score,
        "chunk": _document_chunk_to_response(hit.chunk),
        "images": [_image_evidence_to_response(image) for image in _images_for_document_hit(hit)],
    }


def _images_for_document_hit(hit: DocumentRetrievalHit, limit: int = 3):
    """返回 chunk 关联的可展示图片证据。"""
    return image_evidence_store.list_by_chunk_id(hit.chunk.chunk_id, limit=limit)


def _image_evidence_to_response(image) -> dict:
    """将图片证据元数据转换为前端响应。"""
    image_type, data_category = image_evidence_store.classify(image)
    return {
        "image_id": image.image_id,
        "source_pdf": image.source_pdf,
        "page_no": image.page_no,
        "image_no": image.image_no,
        "extraction_type": image.extraction_type,
        "image_type": image_type,
        "data_category": data_category,
        "caption": image.caption,
        "nearby_text": image.nearby_text,
        "related_chunk_ids": image.related_chunk_ids,
        "width": image.width,
        "height": image.height,
        "url": f"/api/image-evidence/{image.image_id}",
    }


def _standard_case_to_response(
    case,
    *,
    image_type: str | None = None,
    data_category: str | None = None,
) -> dict:
    """将标准化个例转换为 API 响应。"""
    payload = case.to_dict()
    images = image_evidence_store.list_by_image_ids(case.evidence_image_ids)
    if image_type or data_category:
        images = [
            image
            for image in images
            if _image_matches_filter(image, image_type=image_type, data_category=data_category)
        ]
    payload["evidence_images"] = [
        _image_evidence_to_response(image)
        for image in images
    ]
    return payload


def _similar_case_match_to_response(match) -> dict:
    """将相似个例匹配结果转换为 API 响应。"""
    payload = _standard_case_to_response(match.case)
    payload["similarity_score"] = match.score
    payload["score_breakdown"] = match.score_breakdown
    payload["match_reasons"] = match.reasons
    payload["forecast_tips"] = match.forecast_tips
    payload["evidence_images"] = [
        _image_evidence_to_response(image)
        for image in match.evidence_images
    ]
    return payload


def _match_similar_standard_cases(
    *,
    q: str | None = None,
    date: str | None = None,
    disaster_type: str | None = None,
    area: str | None = None,
    image_type: str | None = None,
    data_category: str | None = None,
    source_pdf: str | None = None,
    top_n: int = 5,
) -> list:
    """按结构化字段、图片证据和文本线索综合匹配历史标准个例。"""
    query = SimilarCaseQuery(
        q=q or "",
        date=date or "",
        disaster_type=disaster_type or "",
        area=area or "",
        image_type=image_type or "",
        data_category=data_category or "",
        source_pdf=source_pdf or "",
        top_n=top_n,
    )
    return similar_case_matcher.match(query, standard_case_store.list_cases())


def _search_standard_cases(
    *,
    q: str | None = None,
    date: str | None = None,
    disaster_type: str | None = None,
    area: str | None = None,
    source_pdf: str | None = None,
    image_type: str | None = None,
    data_category: str | None = None,
) -> list:
    """按结构化字段、图片类型和自然语言查询标准化个例。"""
    inferred = _parse_standard_case_query(q or "")
    date = date or inferred.get("date")
    disaster_type = disaster_type or inferred.get("disaster_type")
    area = area or inferred.get("area")
    image_type = image_type or inferred.get("image_type")
    data_category = data_category or inferred.get("data_category")
    cases = standard_case_store.search_cases(
        date=date,
        source_pdf=source_pdf,
    )
    if disaster_type:
        cases = [case for case in cases if _standard_case_matches_disaster(case, disaster_type)]
    if area:
        cases = [case for case in cases if _standard_case_matches_area(case, area)]
    if image_type or data_category:
        cases = [
            case for case in cases
            if _standard_case_has_image(case, image_type=image_type, data_category=data_category)
        ]
    return cases


def _parse_standard_case_query(query: str) -> dict[str, str]:
    """从类似“2025年5月山西北部雷暴大风，有雷达图的个例”中提取轻量过滤条件。"""
    parsed: dict[str, str] = {}
    month_day = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日?", query)
    month_only = re.search(r"(?:\d{4}\s*年\s*)?(\d{1,2})\s*月", query)
    if month_day:
        parsed["date"] = f"{month_day.group(1)}月{month_day.group(2)}日"
    elif month_only:
        parsed["date"] = f"{month_only.group(1)}月"

    for term in ("雷暴大风", "强对流", "暴雨", "大暴雨", "强降水", "雷暴", "大风", "冰雹", "暴雪", "雨雪", "降雪", "寒潮", "高温", "低温", "沙尘", "雾"):
        if term in query:
            parsed["disaster_type"] = term
            break

    for area_term in ("山西北部", "山西中部", "山西南部", "北部", "中部", "南部", "太原", "大同", "朔州", "忻州", "阳泉", "晋中", "吕梁", "长治", "晋城", "临汾", "运城", "山西"):
        if area_term in query:
            parsed["area"] = area_term
            break

    image_keywords = {
        "radar": ("雷达", "回波", "组合反射率", "风雷"),
        "satellite": ("卫星", "云图", "红外", "可见光"),
        "precipitation": ("降水图", "雨量图", "累计降水", "降水"),
        "wind": ("大风图", "风速", "阵风"),
        "sounding": ("探空", "TlnP", "TInP"),
        "synoptic": ("环流", "形势图", "海平面气压"),
        "temperature": ("高温图", "气温图", "温度图"),
        "warning": ("预警", "风险图"),
    }
    for value, keywords in image_keywords.items():
        if any(keyword in query for keyword in keywords):
            parsed["image_type"] = value
            break
    if "图" in query and "image_type" not in parsed:
        parsed["data_category"] = "图片"
    return parsed


AREA_ALIASES = {
    "山西北部": ("山西北部", "北部", "大同", "朔州", "忻州", "山西"),
    "北部": ("山西北部", "北部", "大同", "朔州", "忻州", "山西"),
    "山西中部": ("山西中部", "中部", "太原", "阳泉", "晋中", "吕梁", "山西"),
    "中部": ("山西中部", "中部", "太原", "阳泉", "晋中", "吕梁", "山西"),
    "山西南部": ("山西南部", "南部", "长治", "晋城", "临汾", "运城", "山西"),
    "南部": ("山西南部", "南部", "长治", "晋城", "临汾", "运城", "山西"),
}


def _standard_case_matches_disaster(case, disaster_type: str) -> bool:
    """支持“雷暴大风”等组合灾种匹配标准个例。"""
    terms = _disaster_terms(disaster_type)
    if not terms:
        return True
    haystack = " ".join(
        [
            case.title,
            case.summary,
            case.weather_facts,
            case.forecast_focus,
            " ".join(case.disaster_types),
        ]
    )
    return all(term in haystack for term in terms)


def _disaster_terms(disaster_type: str) -> list[str]:
    value = disaster_type.strip()
    if not value:
        return []
    if value == "雷暴大风":
        return ["雷暴", "大风"]
    if value == "雨雪":
        return ["雨", "雪"]
    return [value]


def _standard_case_matches_area(case, area: str) -> bool:
    """支持省级、地市和“山西北部/中部/南部”等区域别名匹配。"""
    candidates = AREA_ALIASES.get(area.strip(), (area.strip(),))
    haystack = " ".join(
        [
            case.title,
            case.summary,
            case.weather_facts,
            case.forecast_focus,
            " ".join(case.affected_areas),
        ]
    )
    return any(candidate and candidate in haystack for candidate in candidates)


def _standard_case_has_image(case, *, image_type: str | None = None, data_category: str | None = None) -> bool:
    for image in image_evidence_store.list_by_image_ids(case.evidence_image_ids):
        if _image_matches_filter(image, image_type=image_type, data_category=data_category):
            return True
    return False


def _image_matches_filter(image, *, image_type: str | None = None, data_category: str | None = None) -> bool:
    current_type, current_category = image_evidence_store.classify(image)
    if image_type and current_type != image_type:
        return False
    if data_category and data_category not in current_category and data_category not in image.caption:
        return False
    return True


def _build_standard_cases(use_llm: bool = True) -> dict:
    """从 document_index 的 chunk 构建标准化个例层。"""
    builder = StandardCaseBuilder(
        image_store=image_evidence_store,
        llm_client=llm_client,
        use_llm=use_llm,
    )
    chunks = document_store.list_chunks()
    cases = builder.build(chunks)
    standard_case_store.replace_cases(cases)
    return {
        "case_count": len(cases),
        "case_ids": [case.case_id for case in cases],
        "source_chunk_count": len(chunks),
        "storage_path": str(settings.standard_cases_path),
        "llm_used": use_llm and llm_client.is_available(),
    }


def _context_blocks_from_hits(hits) -> list[str]:
    """
    从搜索命中结果构建 LLM 上下文块（内部函数）

    Args:
        hits: RetrievalHit 对象列表

    Returns:
        上下文字符串列表
    """
    return [
        (
            f"个例ID：{hit.case.case_id}\n"
            f"标题：{hit.case.title}\n"
            f"时段：{hit.case.date_range}\n"
            f"来源：{hit.case.source_pdf}\n"
            f"相似度：{hit.score}\n"
            f"正文：{hit.case.content}"
        )
        for hit in hits
    ]


def _context_blocks_from_document_hits(hits: list[DocumentRetrievalHit]) -> list[str]:
    """从文档 chunk 命中构建 LLM 上下文块。"""
    blocks = []
    for hit in hits:
        images = _images_for_document_hit(hit)
        image_lines = [
            (
                f"- {image.caption or f'第{image.page_no}页图片'}"
                f"（图片ID：{image.image_id}，来源：{image.source_pdf} 第{image.page_no}页）"
            )
            for image in images
        ]
        image_block = "\n相关图像证据：\n" + "\n".join(image_lines) if image_lines else ""
        blocks.append(
            f"片段ID：{hit.chunk.chunk_id}\n"
            f"来源：{hit.chunk.source_pdf}\n"
            f"片段序号：{hit.chunk.chunk_no}\n"
            f"相似度：{hit.score}\n"
            f"正文：{hit.chunk.content}"
            f"{image_block}"
        )
    return blocks


def _retrieve_document_chunks(question: str, top_k: int = 5, top_n: int = 3):
    """将问题向量化，并从新文档 chunk 向量库召回相似片段。"""
    retrieval_mode = "document_vector"
    try:
        _ensure_document_index_compatible()
        query_embedding = embedding_client.embed_query(question)
        hits = document_store.query(query_embedding, top_k=top_k)
    except Exception as exc:
        retrieval_mode = f"document_vector_error_fallback: {exc}"
        hits = _lexical_recall_document_chunks(question, top_k=top_k)
    return retrieval_mode, hits[:top_n]


def _ensure_document_index_compatible() -> None:
    """检查文档向量库是否可用且维度匹配。"""
    if not embedding_client.is_available():
        raise RuntimeError("embedding client is unavailable")
    expected = embedding_client.expected_dimension()
    if expected is None:
        return
    actual = document_store.collection_info().get("dimension")
    if actual != expected:
        raise RuntimeError(
            f"document vector index dimension mismatch: expected {expected}, got {actual}; rebuild document_index"
        )


def _lexical_recall_document_chunks(question: str, top_k: int) -> list[DocumentRetrievalHit]:
    """向量召回不可用时，用简单关键词匹配文档 chunk。"""
    tokens = [part for part in re.split(r"[\s，。！？、；：,.!?;:（）()]+", question) if part]
    hits = []
    for chunk in document_store.list_chunks():
        normalized = chunk.content.replace(" ", "")
        score = sum(max(1, len(token)) for token in tokens if token and token.replace(" ", "") in normalized)
        hits.append(DocumentRetrievalHit(chunk=chunk, score=float(score)))
    hits.sort(key=lambda item: item.score, reverse=True)
    return hits[:top_k]


def _ask_document_chunks(question: str, top_k: int = 5, top_n: int = 3) -> dict:
    """基于新文档 chunk 向量库执行非流式 RAG。"""
    retrieval_mode, hits = _retrieve_document_chunks(question, top_k=top_k, top_n=top_n)
    if not hits:
        return {
            "question": question,
            "answer": "未检索到相关证据片段。",
            "retrieval_mode": retrieval_mode,
            "llm_used": False,
            "llm_status": "skipped_no_hits",
            "hits": hits,
        }
    if llm_client.is_available():
        try:
            answer = llm_client.answer_with_context(question, _context_blocks_from_document_hits(hits))
            return {
                "question": question,
                "answer": answer,
                "retrieval_mode": retrieval_mode,
                "llm_used": True,
                "llm_status": "called",
                "hits": hits,
            }
        except Exception as exc:
            return {
                "question": question,
                "answer": f"大模型调用失败，以下为检索到的证据片段：\n{_document_fallback_summary(hits)}",
                "retrieval_mode": retrieval_mode,
                "llm_used": False,
                "llm_status": f"failed: {exc}",
                "hits": hits,
            }
    return {
        "question": question,
        "answer": f"根据命中的证据片段，优先参考以下内容：\n{_document_fallback_summary(hits)}",
        "retrieval_mode": retrieval_mode,
        "llm_used": False,
        "llm_status": "skipped_no_api_key",
        "hits": hits,
    }


def _document_fallback_summary(hits: list[DocumentRetrievalHit]) -> str:
    """生成无 LLM 或 LLM 失败时的证据片段摘要。"""
    return "\n".join(f"{hit.chunk.chunk_id}: {hit.chunk.content[:120]}" for hit in hits)


def _fast_document_answer(question: str, hits: list[DocumentRetrievalHit]) -> str:
    """生成可立即返回的本地首屏回答，降低等待 LLM 首 token 的体感延迟。"""
    if not hits:
        return "未检索到相关证据片段。"
    lines = ["先给你一个基于本地检索的快速结论："]
    for index, hit in enumerate(hits[:3], start=1):
        text = re.sub(r"\s+", " ", hit.chunk.content).strip()[:140]
        images = _images_for_document_hit(hit, limit=2)
        image_note = ""
        if images:
            captions = "；".join(image.caption or f"第{image.page_no}页图片" for image in images)
            image_note = f" 可参考图像证据：{captions}。"
        lines.append(f"{index}. {hit.chunk.source_pdf} 的 {hit.chunk.chunk_id} 提到：{text}。{image_note}")
    if llm_client.is_available():
        lines.append("下面继续补充模型综合分析。")
    return "\n".join(lines) + "\n\n"


def _case_from_txt(txt_path: Path) -> ExtractedCase:
    """读取一个已提取的 TXT 文件并将其转换为 ExtractedCase 对象。"""
    raw = txt_path.read_text(encoding="utf-8")
    header, _, content = raw.partition("\n\n")
    metadata = {}
    for line in header.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    case_id = metadata.get("case_id", txt_path.stem)
    case_no_match = re.search(r"(\d+)$", case_id)
    return ExtractedCase(
        source_pdf=metadata.get("source_pdf", ""),
        case_id=case_id,
        case_no=int(case_no_match.group(1)) if case_no_match else 0,
        title=metadata.get("title", ""),
        date_range=metadata.get("date_range", ""),
        content=content.strip(),
        file_path=str(txt_path),
    )


def _extract_all_pdfs() -> list[str]:
    """
    提取资源目录下所有 PDF 文件的案例（内部函数）
    Returns:
        生成的 txt 文件路径列表
    """
    settings.resource_dir.mkdir(parents=True, exist_ok=True)
    written_files = []
    # 遍历所有 PDF 文件
    for pdf_path in sorted(settings.resource_dir.glob("*.pdf")):
        written_files.extend(str(path) for path in pipeline.process_pdf(pdf_path))
    return written_files


def _index_all_txt_cases() -> dict:
    """
    将所有 txt 案例向量化并存入向量数据库（内部函数）
    Returns:
        索引结果信息字典
    """
    # 读取样本目录下所有 txt 文件
    txt_files = sorted(settings.samples_dir.glob("*.txt"))
    cases = [_case_from_txt(txt_path) for txt_path in txt_files]
    # 如果有案例，批量生成向量
    if cases:
        embeddings = embedding_client.embed_documents(
            [f"{case.title}\n{case.date_range}\n{case.content}" for case in cases]
        )
        # 将向量赋值给对应案例
        for case, embedding in zip(cases, embeddings):
            case.embedding = embedding
    # 替换向量数据库中的所有案例（重建索引）
    case_store.replace_cases(cases)
    vector_info = case_store.collection_info()
    return {
        "indexed": len(cases),
        "case_ids": [case.case_id for case in cases],
        "vector_dimension": vector_info["dimension"],
        "vector_count": vector_info["count"],
        "vector_source": vector_info["source"],
    }


def _refresh_case_library() -> dict:
    """
    刷新整个案例库：提取 PDF 案例 + 重建向量索引（内部函数）

    Returns:
        刷新结果信息字典
    """
    written_files = _extract_all_pdfs()
    index_result = _index_all_txt_cases()
    return {
        "written_files": written_files,
        "count": len(written_files),
        **index_result,
    }


def _embed_and_upsert_cases(cases: list[ExtractedCase]) -> dict:
    """
    将新增个例批量向量化，并以 upsert 方式写入现有向量库。
    这个方法只处理传入的新增个例，不会清空已有 ChromaDB 集合。
    """
    if cases:
        embeddings = embedding_client.embed_documents(
            [f"{case.title}\n{case.date_range}\n{case.content}" for case in cases]
        )
        for case, embedding in zip(cases, embeddings):
            case.embedding = embedding
        case_store.upsert_cases(cases)
    vector_info = case_store.collection_info()
    return {
        "indexed": len(cases),
        "case_ids": [case.case_id for case in cases],
        "vector_dimension": vector_info["dimension"],
        "vector_count": vector_info["count"],
        "vector_source": vector_info["source"],
    }


def _process_pdf_paths_incrementally(pdf_paths: list[Path]) -> dict:
    """
    对指定 PDF 列表执行增量处理：抽取个例、写 TXT、向量化并追加入库。
    成功后会把 PDF 指纹记录为已处理；失败时会写入构建状态错误信息。
    """
    build_status_store.mark_building()
    try:
        written_files: list[str] = []
        for pdf_path in pdf_paths:
            written = pipeline.process_pdf(pdf_path)
            written_files.extend(str(path) for path in written)
            processed_file_store.mark_processed(pdf_path)
        cases = [_case_from_txt(Path(path)) for path in written_files]
        index_result = _embed_and_upsert_cases(cases)
        result = {
            "processed_pdfs": [Path(path).name for path in pdf_paths],
            "written_files": written_files,
            "count": len(written_files),
            **index_result,
        }
        build_status_store.mark_success(result)
        return result
    except Exception as exc:
        build_status_store.mark_failed(exc)
        raise


def _refresh_incremental_case_library() -> dict:
    """
    扫描 resource 目录，只挑出未处理或内容变化的 PDF 执行增量入库。
    这是刷新按钮对应的后端逻辑，不再对全部 PDF 做全量重建。
    """
    settings.resource_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = [
        pdf_path
        for pdf_path in sorted(settings.resource_dir.glob("*.pdf"))
        if not processed_file_store.is_processed(pdf_path)
    ]
    return _process_pdf_paths_incrementally(pdf_paths)


def _index_document_pdfs(pdf_paths: list[Path] | None = None) -> dict:
    """扫描 PDF 并按“文档化 + chunk 切分”方式写入独立文档向量库。"""
    settings.resource_dir.mkdir(parents=True, exist_ok=True)
    is_incremental = pdf_paths is not None
    target_paths = pdf_paths if is_incremental else sorted(settings.resource_dir.glob("*.pdf"))
    chunks = document_indexer.index_pdfs([Path(path) for path in target_paths])
    if is_incremental:
        document_store.upsert_chunks(chunks)
    else:
        document_store.replace_chunks(chunks)
    vector_info = document_store.collection_info()
    return {
        "processed_pdfs": [Path(path).name for path in target_paths],
        "indexed": len(chunks),
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
        "vector_dimension": vector_info["dimension"],
        "vector_count": vector_info["count"],
        "vector_source": vector_info["source"],
    }


def _safe_pdf_filename(raw_filename: str) -> str:
    """
    安全处理上传的 PDF 文件名，防止路径遍历攻击（内部函数）

    Args:
        raw_filename: 原始文件名字符串

    Returns:
        安全的文件名（仅保留文件名部分）

    Raises:
        HTTPException: 如果不是 PDF 文件
    """
    # 解码 URL 编码并仅取文件名部分（去除路径）
    filename = Path(unquote(raw_filename or "")).name
    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    return filename


app.include_router(system_router)
app.include_router(files_router)
app.include_router(vectors_router)
app.include_router(sessions_router)
app.include_router(agent_router)
app.include_router(standard_cases_router)
