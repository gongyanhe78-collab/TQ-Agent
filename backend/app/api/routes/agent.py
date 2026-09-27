"""
智能体问答 API 路由模块
提供统一流式问答和系统集成问答接口。
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.app.api.schemas import AgentQueryRequest, QueryRequest, StreamQueryRequest
from backend.app.config import settings
from backend.app.services.agent.unified_chat.orchestrator import UnifiedChatOrchestrator
from backend.app.services.agent.query_intent_compat import QueryIntentAgent


# API 路由实例
router = APIRouter()
# 统一聊天复用同一个意图识别实例，避免每次请求重复初始化知识范围读取器和模型客户端。
_INTENT_AGENT = QueryIntentAgent(data_dir=settings.smart_case_data_dir)


def _main():
    """
    延迟加载 main 模块中的服务实例
    避免循环导入问题，在请求实际执行时才获取 RAG、LLM 和会话服务。
    """
    from backend.app import main

    return main


def _images_from_hit_payloads(hit_payloads: list[dict], limit: int = 6) -> list[dict]:
    """从检索命中响应中抽取去重后的图片证据，用于会话恢复。"""
    images = []
    seen = set()
    for hit in hit_payloads:
        for image in hit.get("images", []):
            image_id = image.get("image_id")
            if not image_id or image_id in seen:
                continue
            seen.add(image_id)
            images.append(image)
            if len(images) >= limit:
                return images
    return images


def _answer_images_from_result(result: dict, hit_payloads: list[dict] | None = None) -> list[dict]:
    """返回结构化图片，缺失时从 chunk 命中里抽取图片。"""
    if result.get("images"):
        return result["images"][:6]
    return _images_from_hit_payloads(hit_payloads or [])


def _answer_visuals_from_result(result: dict) -> list[dict]:
    """返回可写入会话记忆的结构化可视化结果。"""
    return result.get("visuals", [])[:6]


def _assistant_memory_metadata(
    *,
    question: str,
    answer: str,
    retrieval_mode: str,
    context_case_ids: list[str],
    hit_payloads: list[dict] | None = None,
    evidence_cases: list[dict] | None = None,
    evidence_chunks: list[dict] | None = None,
    images: list[dict] | None = None,
    visuals: list[dict] | None = None,
    intent_trace: dict | None = None,
    execution_plan: dict | None = None,
    audit: dict | None = None,
) -> dict:
    """构建会话内隔离的统一记忆包，并保留顶层字段兼容旧解析逻辑。"""
    chunk_memory = evidence_chunks if evidence_chunks is not None else (hit_payloads or [])
    memory = {
        "version": 1,
        "question": question,
        "answer": answer,
        "retrieval_mode": retrieval_mode,
        "context_case_ids": context_case_ids,
        "evidence_cases": evidence_cases or [],
        "evidence_chunks": chunk_memory,
        "images": images or [],
        "visuals": visuals or [],
        "intent_trace": intent_trace or {},
        "execution_plan": execution_plan or {},
        "audit": audit or {},
    }
    return {
        "memory": memory,
        "images": memory["images"],
        "visuals": memory["visuals"],
        "evidence_cases": memory["evidence_cases"],
        "evidence_chunks": memory["evidence_chunks"],
        "context_case_ids": memory["context_case_ids"],
        "intent_trace": memory["intent_trace"],
        "execution_plan": memory["execution_plan"],
        "audit": memory["audit"],
        "retrieval_mode": memory["retrieval_mode"],
    }


@router.post("/api/query")
def query_cases(payload: QueryRequest):
    """为前端执行非流式 RAG 查询。"""
    main = _main()
    structured = main._answer_structured_question(payload.question)
    if structured is not None:
        if main._structured_result_needs_metric_documents(structured):
            metric_document = main._metric_document_result_from_structured(
                payload.question,
                top_k=payload.top_k,
                top_n=payload.top_n,
            )
            if metric_document is not None:
                return metric_document
        return structured
    result = main._ask_document_chunks(payload.question, top_k=payload.top_k, top_n=payload.top_n)
    return {
        "question": result["question"],
        "answer": result["answer"],
        "retrieval_mode": result["retrieval_mode"],
        "llm_used": result["llm_used"],
        "llm_status": result["llm_status"],
        "hit_count": len(result["hits"]),
        "hits": [main._document_hit_to_response(hit) for hit in result["hits"]],
    }


@router.post("/api/agent/query")
def agent_query(payload: AgentQueryRequest):
    """面向其他智能体项目的结构化 RAG 问答接口。"""
    main = _main()
    if payload.session_id and main.session_store.get_session(payload.session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    context_case_ids = []
    conversation_history = []
    if payload.session_id:
        conversation_history = main.session_store.list_messages(payload.session_id)
        context_case_ids = main._context_case_ids_from_messages(
            conversation_history,
            payload.question,
        )
    if payload.session_id:
        main.session_store.add_message(payload.session_id, "user", payload.question)
    result = main._answer_agent_question(
        payload.question,
        top_k=payload.top_k,
        top_n=payload.top_n,
        context_case_ids=context_case_ids,
        conversation_history=conversation_history,
    )
    hits = result.get("hit_payloads", [])
    if payload.session_id:
        answer_images = _answer_images_from_result(result, hits)
        answer_visuals = _answer_visuals_from_result(result)
        main.session_store.add_message(
            payload.session_id,
            "assistant",
            result["answer"],
            metadata=_assistant_memory_metadata(
                question=payload.question,
                answer=result["answer"],
                retrieval_mode=result["retrieval_mode"],
                context_case_ids=result.get("context_case_ids", context_case_ids),
                hit_payloads=hits,
                evidence_cases=result.get("evidence_cases", []),
                evidence_chunks=result.get("evidence_chunks", hits),
                images=answer_images,
                visuals=answer_visuals,
                intent_trace=result.get("intent_trace", {}),
                execution_plan=result.get("execution_plan", {}),
                audit=result.get("audit", {}),
            ),
        )
    return {
        "question": result["question"],
        "answer": result["answer"],
        "session_id": payload.session_id,
        "hit_count": result.get("hit_count", len(hits)),
        "hits": hits if payload.return_context else [],
        "intent_trace": result.get("intent_trace", {}),
        "execution_plan": result.get("execution_plan", {}),
        "evidence_cases": result.get("evidence_cases", []),
        "evidence_chunks": result.get("evidence_chunks", []),
        "evidence_images": result.get("evidence_images", []),
        "visuals": result.get("visuals", []),
        "audit": result.get("audit", {}),
        "retrieval": {
            "mode": result["retrieval_mode"],
            "embedding_model": settings.embedding_model,
            "rerank_model": settings.rerank_model,
            "top_k": payload.top_k,
            "top_n": payload.top_n,
        },
        "llm": {
            "model": settings.chat_model,
            "used": result["llm_used"],
            "status": result["llm_status"],
        },
        "metadata": payload.metadata or {},
    }


@router.post("/api/query/stream")
async def unified_stream_query_cases(payload: StreamQueryRequest, request: Request):
    """主页面统一流式入口，按意图选择 RAG、多维检索或 Smart。"""
    main = _main()
    # 延迟获取两个 Agent，保持 Router 注册阶段没有循环依赖。
    from backend.app.services.agent.case_multidim_search.router import _agent as multidim_agent
    from backend.app.services.agent.case_multidim_search.router import _natural_store as multidim_store
    from backend.app.services.agent.Smart_Case_Match.router import _agent as smart_agent

    orchestrator = UnifiedChatOrchestrator(
        main,
        multidim_agent(),
        multidim_store(main.session_store),
        smart_agent(),
        _INTENT_AGENT,
    )

    async def event_stream():
        async for event in orchestrator.stream(payload, request.is_disconnected):
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Keep-Alive": "timeout=60",
        },
    )
