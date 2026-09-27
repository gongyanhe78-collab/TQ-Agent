"""
FastAPI 主应用模块，定义所有 API 接口，整合各服务组件提供完整的问答系统功能。"""
from __future__ import annotations

import re
import importlib
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.agent import router as agent_router
from backend.app.api.routes.sessions import router as sessions_router
from backend.app.services.agent.case_multidim_search.router import router as case_multidim_router
from backend.app.services.agent.Smart_Case_Match.router import router as smart_case_match_router
from backend.app.api.routes.system import router as system_router
from backend.app.api.routes.vectors import router as vectors_router
from backend.app.api.schemas import (
    AgentQueryRequest,
    CreateSessionRequest,
    QueryRequest,
    StreamQueryRequest,
)
from backend.app.config import settings
from backend.app.models import DocumentChunk, DocumentRetrievalHit
from backend.app.services.document_store import ChromaDocumentChunkStore
from backend.app.services.model_client.embedding import get_embedding_client
from backend.app.services.image_extraction import ImageEvidenceStore
from backend.app.services.model_client.chat import get_llm_client
from backend.app.services.session_store import SessionStore
from backend.app.services.agent.intent_analyzer.analyzers import AgentIntentAnalyzer
from backend.app.services.agent.synthesizer.answer_synthesizer import AnswerSynthesizer
from backend.app.services.agent.reviewer.final_reviewer import AgentFinalReviewer
from backend.app.services.agent.models import EvidenceChunk, IntentType
from backend.app.services.agent.orchestrator.orchestrator import AgentOrchestrator
from backend.app.services.agent.planner.planner import AgentPlanner
from backend.app.services.similar_case_matcher import SimilarCaseMatcher
from backend.app.services.standard_case_store import JsonStandardCaseStore
from backend.app.services.structured_qa import StructuredQuestionAnswerer


# 意图 Agent 位于带连字符的目录，只能通过 importlib 注册其独立诊断接口。
query_intent_router = importlib.import_module(
    "backend.app.services.agent.query-intent-agent.router"
).router


app = FastAPI(title="Weather Case RAG System", version="0.1.0")
# 配置 CORS 跨域支持（允许所有来源访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化核心服务组件，Chat、Embedding 和 Rerank 均使用云端 API。
embedding_client = get_embedding_client()
llm_client = get_llm_client()
document_store = ChromaDocumentChunkStore(
    settings.document_index_dir,
    settings.document_collection_name,
)
image_evidence_store = ImageEvidenceStore(settings.image_metadata_path)
standard_case_store = JsonStandardCaseStore(settings.standard_cases_path)
session_store = SessionStore(settings.session_db_path)
similar_case_matcher = SimilarCaseMatcher(
    image_store=image_evidence_store,
    image_filter=lambda image, image_type, data_category: _image_matches_filter(
        image,
        image_type=image_type,
        data_category=data_category,
    ),
)
structured_question_answerer = StructuredQuestionAnswerer(
    image_store=image_evidence_store,
    image_to_response=lambda image: _image_evidence_to_response(image),
)


def _make_structured_question_answerer() -> StructuredQuestionAnswerer:
    """创建绑定当前图片证据库的结构化问答器。"""
    return StructuredQuestionAnswerer(
        image_store=image_evidence_store,
        image_to_response=lambda image: _image_evidence_to_response(image),
    )


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


def _answer_structured_question(question: str, context_case_ids: list[str] | None = None) -> dict | None:
    """优先用标准化个例库回答统计、证据清单和对比类问题。"""
    if _question_uses_context_reference(question) and not context_case_ids:
        return _missing_context_reference_answer(question)
    answerer = _make_structured_question_answerer()
    cases = _cases_scoped_by_context(standard_case_store.list_cases(), context_case_ids)
    answer = answerer.answer(question, cases)
    if answer is None:
        return None
    analysis = AgentIntentAnalyzer().analyze(question)
    plan = AgentPlanner().plan(question, analysis)
    evidence_cases = [_standard_case_to_response(item.case) for item in answer.cases]
    return {
        "question": question,
        "answer": answer.answer,
        "retrieval_mode": f"structured_cases:{answer.intent.intent}",
        "llm_used": False,
        "llm_status": "structured_answer",
        "hit_count": len(answer.cases),
        "hits": [],
        "evidence_cases": evidence_cases,
        "evidence_chunks": [],
        "images": answerer.response_images(answer),
        "visuals": answer.visuals,
        "intent": answer.intent.intent,
        "context_case_ids": context_case_ids or [],
        "intent_trace": {
            "intents": [intent.value for intent in plan.intents],
            "required_tools": [step.tool.value for step in plan.steps],
            "forbidden_tools": [tool.value for tool in plan.forbidden_tools],
            "risk_notes": plan.risk_notes,
        },
        "execution_plan": plan.to_dict(),
        "audit": {
            "agent_runs": [
                {
                    "agent": "intent_analyzer",
                    "status": "ok",
                    "intents": [intent.value for intent in plan.intents],
                    "reason": "结构化问答已完成意图识别",
                },
                {
                    "agent": "structured_answerer",
                    "status": "ok",
                    "reason": f"标准化个例链路返回 {len(answer.cases)} 个证据个例",
                },
            ],
            "executed_tools": [step.tool.value for step in plan.steps],
            "skipped_tools": [tool.value for tool in plan.forbidden_tools],
            "failures": [],
        },
    }


def _structured_result_needs_metric_documents(result: dict | None) -> bool:
    if not result:
        return False
    plan = result.get("execution_plan") or {}
    slots = plan.get("slots") or {}
    answer = str(result.get("answer") or "")
    return bool(slots.get("metrics")) and "没有可用" in answer and "结构化数据" in answer


def _metric_document_result_from_structured(
    question: str,
    *,
    top_k: int = 5,
    top_n: int = 3,
    context_case_ids: list[str] | None = None,
) -> dict | None:
    retrieval_mode, hits = _retrieve_document_chunks(
        question,
        top_k=top_k,
        top_n=top_n,
        context_case_ids=context_case_ids,
    )
    if not hits:
        return None
    analysis = AgentIntentAnalyzer().analyze(question)
    plan = AgentPlanner().plan(question, analysis)
    hit_payloads = [_document_hit_to_response(hit) for hit in hits]
    answer = AnswerSynthesizer().metric_comparison_from_chunks(question, _document_hits_to_evidence_chunks(hits))
    return {
        "question": question,
        "answer": answer,
        "retrieval_mode": f"document_metric:{retrieval_mode}",
        "llm_used": False,
        "llm_status": "document_metric_summary",
        "hit_count": len(hits),
        "hits": hit_payloads,
        "hit_payloads": hit_payloads,
        "evidence_cases": [],
        "evidence_chunks": hit_payloads,
        "evidence_images": _images_from_document_hit_payloads(hit_payloads),
        "images": _images_from_document_hit_payloads(hit_payloads),
        "visuals": _metric_document_visuals(hits),
        "context_case_ids": context_case_ids or [],
        "intent_trace": {
            "intents": [intent.value for intent in plan.intents],
            "required_tools": [step.tool.value for step in plan.steps],
            "forbidden_tools": [tool.value for tool in plan.forbidden_tools],
            "risk_notes": plan.risk_notes,
        },
        "execution_plan": plan.to_dict(),
        "audit": {
            "agent_runs": [
                {
                    "agent": "intent_analyzer",
                    "status": "ok",
                    "intents": [intent.value for intent in analysis.intent_types],
                    "reason": "指标型问题已完成意图识别",
                },
                {
                    "agent": "metric_data_guard",
                    "status": "missing_structured_metric_data",
                    "reason": "结构化指标缺失，沿 missing_metric_data 条件边转入文档证据检索",
                },
                {
                    "agent": "document_retriever",
                    "status": "ok",
                    "reason": f"文档证据检索返回 {len(hits)} 条",
                },
                {
                    "agent": "answer_synthesizer",
                    "status": "ok",
                    "reason": "已从零散文档片段抽取指标值并汇总比较",
                },
            ],
            "executed_tools": [step.tool.value for step in plan.steps],
            "skipped_tools": [],
            "failures": [],
        },
    }


def _metric_document_visuals(hits: list[DocumentRetrievalHit]) -> list[dict]:
    if not hits:
        return []
    return [
        {
            "type": "table",
            "title": "文档指标片段",
            "columns": [
                {"key": "source", "label": "来源"},
                {"key": "chunk_id", "label": "片段"},
                {"key": "content", "label": "内容摘要"},
            ],
            "rows": [
                {
                    "source": hit.chunk.source_pdf,
                    "chunk_id": hit.chunk.chunk_id,
                    "content": re.sub(r"\s+", " ", hit.chunk.content).strip()[:120],
                }
                for hit in hits[:6]
            ],
        }
    ]


def _missing_context_reference_answer(question: str) -> dict:
    analysis = AgentIntentAnalyzer().analyze(question)
    plan = AgentPlanner().plan(question, analysis)
    answer = "\n".join(
        [
            "结论：我没有从上文解析出明确的个例范围，不能直接替你展开。",
            "说明：这类追问需要上一轮回答中有可追溯的个例表或明确的个例标题。",
            "建议：请重新指定时间、数量或个例名称，例如“详细讲讲1月这三个个例”。",
        ]
    )
    return {
        "question": question,
        "answer": answer,
        "retrieval_mode": "structured_cases:context_missing",
        "llm_used": False,
        "llm_status": "context_missing",
        "hit_count": 0,
        "hits": [],
        "evidence_cases": [],
        "evidence_chunks": [],
        "images": [],
        "visuals": [],
        "intent": "context_missing",
        "context_case_ids": [],
        "intent_trace": {
            "intents": [intent.value for intent in plan.intents],
            "required_tools": [step.tool.value for step in plan.steps],
            "forbidden_tools": [tool.value for tool in plan.forbidden_tools],
            "risk_notes": plan.risk_notes,
        },
        "execution_plan": plan.to_dict(),
        "audit": {
            "agent_runs": [
                {
                    "agent": "context_resolver",
                    "status": "failed",
                    "reason": "问题包含上文指代，但没有解析到明确个例范围",
                }
            ],
            "executed_tools": [],
            "skipped_tools": [tool.value for tool in plan.forbidden_tools],
            "failures": ["context_reference_unresolved"],
        },
    }


def _answer_agent_question(
    question: str,
    top_k: int = 5,
    top_n: int = 3,
    context_case_ids: list[str] | None = None,
    conversation_history: list[dict] | None = None,
) -> dict:
    """运行只读智能体路由，返回回答、计划、链路和证据。"""
    result = _answer_agent_question_once(
        question,
        top_k=top_k,
        top_n=top_n,
        context_case_ids=context_case_ids,
    )
    reviewer = AgentFinalReviewer(llm_client=llm_client)
    review = reviewer.review(question, result, conversation_history or [])
    result = _apply_final_review(result, review)
    if review.get("should_retry") and review.get("normalized_question") and review["normalized_question"] != question:
        retry_question = str(review["normalized_question"])
        retried = _answer_agent_question_once(
            retry_question,
            top_k=top_k,
            top_n=top_n,
            context_case_ids=context_case_ids,
        )
        retry_review = reviewer.review(question, retried, conversation_history or [])
        result = _apply_final_review(
            retried,
            {
                **retry_review,
                "status": "retry",
                "reason": f"已使用规范化问题重新执行：{retry_question}",
                "normalized_question": retry_question,
            },
        )
        result["question"] = question
    return result


def _answer_agent_question_once(
    question: str,
    top_k: int = 5,
    top_n: int = 3,
    context_case_ids: list[str] | None = None,
) -> dict:
    """执行一次意图识别、工具调用和初始答案生成，不做最终审查。"""
    analysis = AgentIntentAnalyzer().analyze(question)
    if analysis.intent_types == [IntentType.GENERAL_RAG]:
        plan = AgentPlanner().plan(question, analysis)
        result = _ask_document_chunks(
            question,
            top_k=top_k,
            top_n=top_n,
            context_case_ids=context_case_ids,
        )
        hit_payloads = [_document_hit_to_response(hit) for hit in result.get("hits", [])]
        return {
            **result,
            "hit_count": len(result.get("hits", [])),
            "hit_payloads": hit_payloads,
            "evidence_cases": [],
            "evidence_chunks": hit_payloads,
            "evidence_images": _images_from_document_hit_payloads(hit_payloads),
            "intent_trace": {
                "intents": [intent.value for intent in plan.intents],
                "required_tools": [step.tool.value for step in plan.steps],
                "forbidden_tools": [tool.value for tool in plan.forbidden_tools],
                "risk_notes": plan.risk_notes,
            },
            "execution_plan": plan.to_dict(),
            "audit": {
                "agent_runs": [
                    {
                        "agent": "intent_analyzer",
                        "status": "ok",
                        "intents": [intent.value for intent in analysis.intent_types],
                        "reason": "普通问答未命中结构化业务意图",
                    },
                    {
                        "agent": "document_retriever",
                        "status": "ok",
                        "reason": f"文档证据检索返回 {len(result.get('hits', []))} 条",
                    },
                ],
                "executed_tools": [step.tool.value for step in plan.steps],
                "skipped_tools": [tool.value for tool in plan.forbidden_tools],
                "failures": [],
            },
        }
    agent = AgentOrchestrator(
        cases_provider=lambda: _cases_scoped_by_context(standard_case_store.list_cases(), context_case_ids),
        structured_answerer=_make_structured_question_answerer(),
        document_retriever=lambda query, k, n: _retrieve_document_chunks(
            query,
            top_k=k,
            top_n=n,
            context_case_ids=context_case_ids,
        ),
        document_hit_to_response=lambda hit: _document_hit_to_response(hit),
        similar_matcher=similar_case_matcher,
        standard_case_to_response=lambda case: _standard_case_to_response(case),
    )
    similar_case_matcher.image_store = image_evidence_store
    result = agent.answer(question, top_k=top_k, top_n=top_n)
    result["context_case_ids"] = context_case_ids or []
    return result


def _apply_final_review(result: dict, review: dict) -> dict:
    reviewed = dict(result)
    audit = dict(reviewed.get("audit") or {})
    agent_runs = list(audit.get("agent_runs") or [])
    agent_runs.append(
        {
            "agent": "final_reviewer",
            "status": review.get("status", "ok"),
            "reason": review.get("reason", ""),
            "normalized_question": review.get("normalized_question", reviewed.get("question", "")),
            "llm_used": bool(review.get("llm_used", False)),
        }
    )
    audit["agent_runs"] = agent_runs
    reviewed["audit"] = audit
    final_answer = str(review.get("final_answer") or "")
    if final_answer:
        reviewed["answer"] = final_answer
        reviewed["llm_used"] = True
        reviewed["llm_status"] = "final_review"
    return reviewed


def _cases_scoped_by_context(cases: list, context_case_ids: list[str] | None = None) -> list:
    if not context_case_ids:
        return cases
    wanted = set(context_case_ids)
    scoped = [case for case in cases if case.case_id in wanted]
    return scoped or cases


def _context_case_ids_from_messages(messages: list[dict], question: str) -> list[str]:
    if not _question_uses_context_reference(question):
        return []
    candidates = []
    referenced_count = _referenced_count_from_question(question)
    indexed_messages = list(enumerate(messages))
    for recency_index, (message_index, message) in enumerate(reversed(indexed_messages)):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "")
        metadata = message.get("metadata", {}) or {}
        memory = metadata.get("memory", {}) if isinstance(metadata.get("memory", {}), dict) else {}
        previous_user_text = _previous_user_text(messages, message_index)
        memory_text = f"{previous_user_text}\n{content}".strip()
        stored_context_ids = [
            item
            for item in (metadata.get("context_case_ids") or memory.get("context_case_ids") or [])
            if isinstance(item, str) and item
        ]
        if stored_context_ids:
            candidates.append(
                _context_case_candidate(
                    stored_context_ids,
                    question,
                    memory_text,
                    "stored_context",
                    recency_index,
                    referenced_count,
                )
            )
        evidence_cases = (
            message.get("evidence_cases")
            or message.get("metadata", {}).get("evidence_cases")
            or memory.get("evidence_cases")
            or []
        )
        case_ids = [
            item.get("case_id")
            for item in evidence_cases
            if isinstance(item, dict) and item.get("case_id")
        ]
        if case_ids:
            candidates.append(
                _context_case_candidate(
                    case_ids,
                    question,
                    memory_text,
                    "evidence_cases",
                    recency_index,
                    referenced_count,
                )
            )
        visual_case_ids = _case_ids_from_visuals(metadata.get("visuals") or memory.get("visuals") or [])
        if visual_case_ids:
            candidates.append(
                _context_case_candidate(
                    visual_case_ids,
                    question,
                    memory_text,
                    "visuals",
                    recency_index,
                    referenced_count,
                )
            )
        mentioned_case_ids = _case_ids_mentioned_in_text(message.get("content", ""))
        if mentioned_case_ids:
            candidates.append(
                _context_case_candidate(
                    mentioned_case_ids,
                    question,
                    memory_text,
                    "text",
                    recency_index,
                    referenced_count,
                )
            )
    valid_candidates = [candidate for candidate in candidates if candidate["ids"]]
    if not valid_candidates:
        return []
    if referenced_count:
        exact_candidates = [
            candidate
            for candidate in valid_candidates
            if len(candidate["ids"]) == referenced_count
        ]
        if exact_candidates:
            best = max(exact_candidates, key=lambda candidate: (candidate["score"], -candidate["recency_index"]))
            return best["ids"]
        return []
    best = max(valid_candidates, key=lambda candidate: (candidate["score"], -candidate["recency_index"]))
    return _limit_context_case_ids(best["ids"], question)


def _previous_user_text(messages: list[dict], before_index: int) -> str:
    for message in reversed(messages[:before_index]):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _context_case_candidate(
    case_ids: list[str],
    question: str,
    memory_text: str,
    source: str,
    recency_index: int,
    referenced_count: int | None,
) -> dict:
    deduped = _refine_context_case_ids(_dedupe_case_ids(case_ids), memory_text, referenced_count)
    return {
        "ids": deduped,
        "score": _score_context_case_candidate(
            deduped,
            question,
            memory_text,
            source,
            referenced_count,
        ),
        "recency_index": recency_index,
        "source": source,
    }


def _refine_context_case_ids(case_ids: list[str], memory_text: str, referenced_count: int | None) -> list[str]:
    if not case_ids:
        return []
    filtered = _case_ids_matching_memory(case_ids, memory_text)
    if referenced_count:
        if len(filtered) >= referenced_count:
            return filtered[:referenced_count]
        if len(case_ids) == referenced_count:
            return case_ids
        return []
    return filtered or case_ids


def _case_ids_matching_memory(case_ids: list[str], memory_text: str) -> list[str]:
    months = _month_tokens_from_text(memory_text)
    if not months:
        return []
    cases_by_id = {case.case_id: case for case in standard_case_store.list_cases()}
    matched = []
    for case_id in case_ids:
        case = cases_by_id.get(case_id)
        if case and _case_matches_month_tokens(case, months):
            matched.append(case_id)
    return matched


def _case_matches_month_tokens(case, month_tokens: list[str]) -> bool:
    case_text_value = " ".join([case.title, case.date_range, case.source_pdf])
    return any(token in case_text_value for token in month_tokens)


def _score_context_case_candidate(
    case_ids: list[str],
    question: str,
    content: str,
    source: str,
    referenced_count: int | None,
) -> int:
    if not case_ids:
        return -1000
    score = 0
    if source == "stored_context":
        score += 18
    elif source == "visuals":
        score += 14
    elif source == "evidence_cases":
        score += 8
    elif source == "text":
        score += 4
    if referenced_count:
        if len(case_ids) == referenced_count:
            score += 100
        elif len(case_ids) > referenced_count:
            score -= min(60, (len(case_ids) - referenced_count) * 8)
        else:
            score -= 25
        if _content_mentions_count(content, referenced_count):
            score += 22
        if _content_mentions_full_library_stat(content, referenced_count):
            score -= 45
    month_tokens = _month_tokens_from_text(question)
    if month_tokens and any(token in content for token in month_tokens):
        score += 12
    return score


def _dedupe_case_ids(case_ids: list[str]) -> list[str]:
    deduped = []
    for case_id in case_ids:
        if case_id and case_id not in deduped:
            deduped.append(case_id)
    return deduped


def _content_mentions_count(content: str, count: int) -> bool:
    chinese = {
        1: "一",
        2: "二",
        3: "三",
        4: "四",
        5: "五",
        6: "六",
        7: "七",
        8: "八",
        9: "九",
        10: "十",
    }.get(count, "")
    patterns = [rf"{count}\s*(?:次|个|条)"]
    if chinese:
        patterns.append(rf"{chinese}\s*(?:次|个|条)")
    return any(re.search(pattern, content) for pattern in patterns)


def _content_mentions_full_library_stat(content: str, referenced_count: int) -> bool:
    if not re.search(r"(全库|标准化个例库|共筛选到|按月份|按灾种|统计)", content):
        return False
    match = re.search(r"共筛选到\s*(\d+)\s*个", content)
    if match and int(match.group(1)) != referenced_count:
        return True
    return "按月份" in content or "按灾种" in content


def _month_tokens_from_text(text: str) -> list[str]:
    tokens = []
    for match in re.finditer(r"(\d{1,2})\s*月", text):
        token = f"{int(match.group(1))}月"
        if token not in tokens:
            tokens.append(token)
    return tokens


def _question_uses_context_reference(question: str) -> bool:
    if any(term in question for term in (
        "上述", "上面", "上文", "前面", "这些", "那些", "这两个", "那两个", "这几个", "那几个",
        "这几次", "那几次", "它们", "他们", "继续分析", "接着分析", "进一步分析", "分别分析",
    )):
        return True
    if re.search(r"([一二两三四五六七八九十]|\d+)(次|个).{0,6}(灾害|过程|个例)", question):
        return True
    return bool(re.search(r"(这|那|该).{0,3}([一二两三四五六七八九十\d]+|几)(次|个).{0,6}(灾害|过程|个例)", question))


def _limit_context_case_ids(case_ids: list[str], question: str) -> list[str]:
    deduped = []
    for case_id in case_ids:
        if case_id not in deduped:
            deduped.append(case_id)
    referenced_count = _referenced_count_from_question(question)
    if referenced_count and len(deduped) > referenced_count:
        return deduped[:referenced_count]
    return deduped


def _referenced_count_from_question(question: str) -> int | None:
    match = re.search(r"([一二两三四五六七八九十]|\d+)(?:次|个)", question)
    if not match:
        return None
    value = match.group(1)
    if value.isdigit():
        return int(value)
    return {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }.get(value)


def _case_ids_from_visuals(visuals: list[dict]) -> list[str]:
    case_ids = []
    for visual in visuals or []:
        if not isinstance(visual, dict) or visual.get("type") != "table":
            continue
        for row in visual.get("rows", []):
            case_id = _case_id_from_visual_row(row)
            if case_id and case_id not in case_ids:
                case_ids.append(case_id)
    return case_ids


def _case_id_from_visual_row(row: dict) -> str:
    if not isinstance(row, dict):
        return ""
    if row.get("case_id"):
        return row["case_id"]
    row_title = str(row.get("title") or row.get("process") or "").strip()
    row_date = str(row.get("date") or row.get("date_range") or "").strip()
    row_source = str(row.get("source") or row.get("source_pdf") or "").strip()
    normalized_title = _normalize_case_reference(row_title)
    for case in standard_case_store.list_cases():
        if row_source and case.source_pdf != row_source:
            continue
        if row_title and (case.title == row_title or _normalize_case_reference(case.title) == normalized_title):
            return case.case_id
        if row_date and case.date_range == row_date:
            return case.case_id
    return ""


def _case_ids_mentioned_in_text(text: str) -> list[str]:
    if not text:
        return []
    case_ids = []
    normalized_text = _normalize_case_reference(text)
    for case in standard_case_store.list_cases():
        normalized_title = _normalize_case_reference(case.title)
        if case.case_id in text or case.title in text or (normalized_title and normalized_title in normalized_text):
            case_ids.append(case.case_id)
            continue
        if case.date_range and case.date_range in text:
            disaster_hit = any(disaster in text for disaster in case.disaster_types)
            title_words_hit = any(part and part in text for part in re.split(r"[~\-—\s]+", case.title) if len(part) >= 4)
            if disaster_hit or title_words_hit:
                case_ids.append(case.case_id)
    return _limit_context_case_ids(case_ids, "")


def _normalize_case_reference(text: str) -> str:
    value = re.sub(r"\s+", "", text)
    value = value.replace("~", "-").replace("－", "-").replace("—", "-")
    for suffix in ("天气过程", "过程", "天气"):
        value = value.replace(suffix, "")
    return value


def _images_from_document_hit_payloads(hit_payloads: list[dict], limit: int = 6) -> list[dict]:
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


def _image_matches_filter(image, *, image_type: str | None = None, data_category: str | None = None) -> bool:
    current_type, current_category = image_evidence_store.classify(image)
    if image_type and current_type != image_type:
        return False
    if data_category and data_category not in current_category and data_category not in image.caption:
        return False
    return True


def _build_standard_cases(use_llm: bool = True) -> dict:
    """从主自然段 chunk 构建固定 8 字段的标准个例 JSON。"""
    from backend.app.services.knowledge_builder.build_standard_cases import build

    return build(use_llm=use_llm)


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


def _retrieve_document_chunks(
    question: str,
    top_k: int = 5,
    top_n: int = 3,
    context_case_ids: list[str] | None = None,
):
    """将问题向量化，并从新文档 chunk 向量库召回相似片段。"""
    retrieval_mode = "document_vector"
    scoped_chunks = _document_chunks_for_context_cases(context_case_ids)
    if scoped_chunks:
        return "document_context_cases", _rank_context_document_chunks(
            question,
            scoped_chunks,
            context_case_ids or [],
            top_n,
        )
    try:
        _ensure_document_index_compatible()
        query_embedding = embedding_client.embed_query(question)
        hits = document_store.query(query_embedding, top_k=top_k)
    except Exception as exc:
        retrieval_mode = f"document_vector_error_fallback: {exc}"
        hits = _lexical_recall_document_chunks(question, top_k=top_k)
    return retrieval_mode, hits[:top_n]


def _document_chunks_for_context_cases(context_case_ids: list[str] | None = None) -> list:
    """严格按明确个例范围读取正文，非法 case_id 不得退化为全库。"""
    if not context_case_ids:
        return []
    wanted_case_ids = set(context_case_ids)
    context_cases = [case for case in standard_case_store.list_cases() if case.case_id in wanted_case_ids]
    if not context_cases:
        return []
    wanted_chunk_ids = {
        chunk_id
        for case in context_cases
        for chunk_id in case.source_chunk_ids
        if chunk_id
    }
    wanted_sources = {case.source_pdf for case in context_cases if case.source_pdf}
    chunks = []
    for chunk in document_store.list_chunks():
        if wanted_chunk_ids and chunk.chunk_id in wanted_chunk_ids:
            chunks.append(chunk)
        elif not wanted_chunk_ids and chunk.source_pdf in wanted_sources:
            chunks.append(chunk)
    return chunks


def _rank_context_document_chunks(
    question: str,
    chunks: list,
    context_case_ids: list[str],
    top_n: int,
) -> list[DocumentRetrievalHit]:
    """一至两个明确个例读取完整正文，多个个例按个例均衡选取证据。"""
    cases_by_id = {
        case.case_id: case
        for case in standard_case_store.list_cases()
        if case.case_id in set(context_case_ids)
    }
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    tokens = [part for part in re.split(r"[\s，。！？、；：,.!?;:（）()]+", question) if part]

    def make_hit(chunk) -> DocumentRetrievalHit:
        """用词面分数保留可解释的上下文命中分值。"""
        normalized = chunk.content.replace(" ", "")
        score = sum(max(1, len(token)) for token in tokens if token.replace(" ", "") in normalized)
        return DocumentRetrievalHit(chunk=chunk, score=float(score))

    ordered_groups: list[list] = []
    seen_chunk_ids: set[str] = set()
    for case_id in context_case_ids:
        case = cases_by_id.get(case_id)
        if case is None:
            continue
        group = []
        for chunk_id in case.source_chunk_ids:
            chunk = chunk_by_id.get(chunk_id)
            if chunk is not None and chunk.chunk_id not in seen_chunk_ids:
                group.append(chunk)
                seen_chunk_ids.add(chunk.chunk_id)
        group.sort(key=lambda item: (item.source_pdf, item.chunk_no, item.chunk_id))
        if group:
            ordered_groups.append(group)

    # 两个以内的明确个例按业务章节做覆盖式压缩，减少长个例造成的模型首字等待。
    if len(cases_by_id) <= 2:
        ungrouped = [
            chunk
            for chunk in sorted(chunks, key=lambda item: (item.source_pdf, item.chunk_no, item.chunk_id))
            if chunk.chunk_id not in seen_chunk_ids
        ]
        compression_groups = list(ordered_groups)
        if ungrouped:
            compression_groups.append(ungrouped)
        selected = [
            chunk
            for group in compression_groups
            for chunk in _compress_context_chunk_group(question, group, limit=12)
        ]
        return [make_hit(chunk) for chunk in selected]

    # 多个个例限制总量，并轮询各组，防止单个长个例占满模型上下文。
    per_group = [sorted(group, key=lambda item: make_hit(item).score, reverse=True) for group in ordered_groups]
    limit = min(30, max(int(top_n), int(top_n) * 3))
    selected = []
    index = 0
    while len(selected) < limit and any(index < len(group) for group in per_group):
        for group in per_group:
            if index < len(group):
                selected.append(group[index])
                if len(selected) >= limit:
                    break
        index += 1
    return [make_hit(chunk) for chunk in selected]


_CONTEXT_EVIDENCE_SECTIONS: tuple[tuple[str, ...], ...] = (
    (
        "天气实况", "实况特征", "过程降水", "累计降水", "小时雨强", "最大风速", "极大风速",
        "最高气温", "最低气温", "降温幅度", "积雪深度", "能见度", "国家站", "区域站",
    ),
    (
        "环流形势", "演变特征", "天气成因", "成因分析", "500hPa", "700hPa", "850hPa",
        "高压脊", "冷涡", "低涡", "切变线", "冷锋", "副热带高压", "水汽", "垂直运动",
    ),
    (
        "影响", "灾情", "农业", "交通", "电力", "人体健康", "服务", "风险", "预警区域",
    ),
    (
        "数值预报", "模式预报", "预报产品", "预报着眼点", "预报效果", "预报偏差",
        "EC模式", "CMA模式", "GFS", "预警", "漏报", "空报", "订正",
    ),
)


def _compress_context_chunk_group(question: str, chunks: list[DocumentChunk], limit: int = 12) -> list[DocumentChunk]:
    """覆盖关键业务章节并保留相邻上下文，压缩已知个例的模型输入。"""
    ordered = sorted(chunks, key=lambda item: (item.source_pdf, item.chunk_no, item.chunk_id))
    if len(ordered) <= limit:
        return ordered

    selected_indices = {0, len(ordered) - 1}

    def section_score(index: int, terms: tuple[str, ...]) -> int:
        """按关键词出现次数评价片段对某个业务章节的覆盖度。"""
        content = ordered[index].content
        return sum(content.count(term) * max(1, len(term)) for term in terms)

    # 每类最多保留两个高信息片段，保证实况、成因、影响和预报复盘均有证据。
    for terms in _CONTEXT_EVIDENCE_SECTIONS:
        ranked = sorted(
            range(len(ordered)),
            key=lambda index: (section_score(index, terms), -index),
            reverse=True,
        )
        for index in [item for item in ranked if section_score(item, terms) > 0][:2]:
            selected_indices.add(index)

    # 用户明确追问某类内容时，再优先补足该类最相关的片段。
    focus_terms = tuple(
        term
        for terms in _CONTEXT_EVIDENCE_SECTIONS
        for term in terms
        if term in question
    )
    if focus_terms:
        focused = sorted(
            range(len(ordered)),
            key=lambda index: (section_score(index, focus_terms), -index),
            reverse=True,
        )
        selected_indices.update([item for item in focused if section_score(item, focus_terms) > 0][:2])

    # 为已选章节补充紧邻的后续片段，避免自然段切分正好截断事实或分析句。
    for index in sorted(tuple(selected_indices)):
        if len(selected_indices) >= limit:
            break
        if index + 1 < len(ordered):
            selected_indices.add(index + 1)

    all_terms = tuple(term for terms in _CONTEXT_EVIDENCE_SECTIONS for term in terms)
    remaining = sorted(
        (index for index in range(len(ordered)) if index not in selected_indices),
        key=lambda index: (section_score(index, all_terms), -index),
        reverse=True,
    )
    selected_indices.update(remaining[: max(0, limit - len(selected_indices))])
    return [ordered[index] for index in sorted(selected_indices)[:limit]]


def _rank_document_chunks_by_question(question: str, chunks: list, top_n: int) -> list[DocumentRetrievalHit]:
    tokens = [part for part in re.split(r"[\s，。！？、；：,.!?;:（）()]+", question) if part]
    hits = []
    for chunk in chunks:
        normalized = chunk.content.replace(" ", "")
        score = sum(max(1, len(token)) for token in tokens if token and token.replace(" ", "") in normalized)
        hits.append(DocumentRetrievalHit(chunk=chunk, score=float(score)))
    hits.sort(key=lambda item: item.score, reverse=True)
    return hits[:top_n]


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


def _ask_document_chunks(
    question: str,
    top_k: int = 5,
    top_n: int = 3,
    context_case_ids: list[str] | None = None,
) -> dict:
    """基于新文档 chunk 向量库执行非流式 RAG。"""
    retrieval_mode, hits = _retrieve_document_chunks(
        question,
        top_k=top_k,
        top_n=top_n,
        context_case_ids=context_case_ids,
    )
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
                "answer": _document_answer_fallback(question, hits),
                "retrieval_mode": retrieval_mode,
                "llm_used": False,
                "llm_status": f"failed: {exc}",
                "hits": hits,
            }
    return {
        "question": question,
        "answer": _document_answer_fallback(question, hits),
        "retrieval_mode": retrieval_mode,
        "llm_used": False,
        "llm_status": "skipped_no_api_key",
        "hits": hits,
    }


def _document_fallback_summary(hits: list[DocumentRetrievalHit]) -> str:
    """生成无 LLM 或 LLM 失败时的证据片段摘要。"""
    return "\n".join(f"{hit.chunk.chunk_id}: {hit.chunk.content[:120]}" for hit in hits)


def _document_answer_fallback(question: str, hits: list[DocumentRetrievalHit]) -> str:
    """用文档命中结果生成优先回答问题的本地兜底答案。"""
    evidence = _document_hits_to_evidence_chunks(hits)
    if _question_needs_metric_document_summary(question):
        return AnswerSynthesizer().metric_comparison_from_chunks(question, evidence)
    return AnswerSynthesizer().document_fallback(question, evidence)


def _document_hits_to_evidence_chunks(hits: list[DocumentRetrievalHit]) -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            chunk_id=hit.chunk.chunk_id,
            source_pdf=hit.chunk.source_pdf,
            content=hit.chunk.content,
        )
        for hit in hits
    ]


def _question_needs_metric_document_summary(question: str) -> bool:
    return any(term in question for term in ("降水量", "雨量", "平均降水", "累计降水")) and any(
        term in question for term in ("对比", "相比", "比较", "差异", "高于", "低于")
    )


def _fast_document_answer(question: str, hits: list[DocumentRetrievalHit]) -> str:
    """生成可立即返回的本地首屏回答，降低等待 LLM 首 token 的体感延迟。"""
    if not hits:
        return "未检索到相关证据片段。"
    lines = [_document_answer_fallback(question, hits)]
    if llm_client.is_available():
        lines.append("模型综合分析会继续补充。")
    return "\n".join(lines) + "\n\n"


def _index_document_pdfs(pdf_paths: list[Path] | None = None) -> dict:
    """用唯一的自然段方案全量重建向量、图片索引和标准个例。"""
    from backend.app.services.knowledge_builder.build_paragraph_vectors import build as build_vectors

    # 上传入口传入文件列表仅用于回显；为保持图片和个例关联一致，始终重建整个 resource 数据集。
    requested_files = [Path(path).name for path in (pdf_paths or [])]
    vector_result = build_vectors(resource_dir=settings.resource_dir)
    case_result = _build_standard_cases(use_llm=False)
    return {
        "processed_pdfs": requested_files or [path.name for path in sorted(settings.resource_dir.glob("*.pdf"))],
        "indexed": vector_result["chunk_count"],
        "chunk_count": vector_result["chunk_count"],
        "image_count": vector_result["image_count"],
        "case_count": case_result["case_count"],
        "vector_dimension": vector_result["document_collection"].get("dimension"),
        "vector_count": vector_result["document_collection"].get("count"),
        "vector_source": vector_result["document_collection"].get("source"),
    }


app.include_router(system_router)
app.include_router(vectors_router)
app.include_router(sessions_router)
app.include_router(agent_router)
app.include_router(case_multidim_router)
app.include_router(smart_case_match_router)
app.include_router(query_intent_router)
