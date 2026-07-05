from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.app.api.schemas import AgentQueryRequest, QueryRequest, StreamQueryRequest
from backend.app.config import settings
from backend.app.services.retrieval import SimpleAnswerGenerator


router = APIRouter()


def _main():
    """请求执行时获取 main 模块中的 RAG、LLM 和会话服务实例。"""
    from backend.app import main

    return main


@router.post("/api/query")
def query_cases(payload: QueryRequest):
    """为前端执行非流式 RAG 查询。"""
    main = _main()
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
    if payload.session_id:
        main.session_store.add_message(payload.session_id, "user", payload.question)
    result = main._ask_document_chunks(payload.question, top_k=payload.top_k, top_n=payload.top_n)
    if payload.session_id:
        main.session_store.add_message(payload.session_id, "assistant", result["answer"])
    hits = [main._document_hit_to_response(hit) for hit in result["hits"]]
    return {
        "question": result["question"],
        "answer": result["answer"],
        "session_id": payload.session_id,
        "hit_count": len(result["hits"]),
        "hits": hits if payload.return_context else [],
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
def stream_query_cases(payload: StreamQueryRequest):
    """流式问答接口，使用 SSE 返回 metadata、delta、done 和 error 事件。"""
    main = _main()

    def event_stream():
        try:
            if payload.session_id:
                if main.session_store.get_session(payload.session_id) is None:
                    raise HTTPException(status_code=404, detail="Session not found")
                main.session_store.add_message(payload.session_id, "user", payload.question)

            answer_parts: list[str] = []
            retrieval_mode, hits = main._retrieve_document_chunks(
                payload.question,
                top_k=payload.top_k,
                top_n=payload.top_n,
            )
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "metadata",
                        "question": payload.question,
                        "retrieval_mode": retrieval_mode,
                        "hit_count": len(hits),
                        "hits": [main._document_hit_to_response(hit) for hit in hits],
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )

            context_blocks = main._context_blocks_from_document_hits(hits)
            if not hits:
                text = "未检索到相关证据片段。"
                answer_parts.append(text)
                yield "data: " + json.dumps({"type": "delta", "text": text}, ensure_ascii=False) + "\n\n"
            elif main.llm_client.is_available():
                quick_answer = main._fast_document_answer(payload.question, hits)
                answer_parts.append(quick_answer)
                yield "data: " + json.dumps({"type": "delta", "text": quick_answer}, ensure_ascii=False) + "\n\n"
                for delta in main.llm_client.stream_answer_with_context(payload.question, context_blocks):
                    answer_parts.append(delta)
                    yield "data: " + json.dumps({"type": "delta", "text": delta}, ensure_ascii=False) + "\n\n"
            else:
                fallback_answer = (
                    "根据命中的证据片段，优先参考以下内容：\n"
                    + main._document_fallback_summary(hits)
                )
                answer_parts.append(fallback_answer)
                yield "data: " + json.dumps({"type": "delta", "text": fallback_answer}, ensure_ascii=False) + "\n\n"

            if payload.session_id:
                main.session_store.add_message(payload.session_id, "assistant", "".join(answer_parts))
            yield "data: " + json.dumps({"type": "done", "llm_used": main.llm_client.is_available()}, ensure_ascii=False) + "\n\n"
        except Exception as exc:
            yield "data: " + json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/api/eval/questions")
def eval_questions():
    """从 weather_qa_results.json 读取预设问答对，用于效果评估。"""
    json_path = settings.project_root / "weather_qa_results.json"
    if not json_path.exists():
        return {"questions": []}
    payload = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
    questions = []
    for group in payload:
        for sample in group.get("samples", []):
            questions.append(
                {
                    "pdf_filename": group.get("pdf_filename", ""),
                    "question": sample.get("question", ""),
                    "answer": sample.get("answer", ""),
                }
            )
    return {"questions": questions}
