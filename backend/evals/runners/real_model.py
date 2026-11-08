"""真实模型评测执行器：调用真实云端模型与真实检索链路，不使用离线夹具。"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from backend.app.agents.case_multidim_search.core.natural_query import NaturalCaseQueryParser
from backend.app.agents.case_multidim_search.integrations.non_thinking_llm import (
    ensure_non_thinking_client,
)
from backend.app.agents.case_multidim_search.retrieval.structured_retriever import (
    StructuredCaseRetriever,
)
from backend.app.agents.query_intent import QueryIntentAgent
from backend.app.agents.query_intent.schemas import IntentRouteRequest
from backend.app.core.config import Settings, get_settings
from backend.app.services.image_extraction import ImageEvidenceStore
from backend.app.services.model_client import (
    get_embedding_client,
    get_intent_llm_client,
    get_llm_client,
    get_rerank_client,
)
from backend.app.services.query.document_answer import DocumentQuestionAnswerer
from backend.app.services.query.document_retrieval import DocumentRetriever
from backend.app.services.query.result_presenters import QueryResultPresenter
from backend.app.services.standard_case_store import JsonStandardCaseStore
from backend.app.services.vector.factory import build_document_vector_store
from backend.evals.schemas.dataset import EvaluationCase, EvaluationCaseResult

# 回答中出现这些短语时视为明确拒答；只用于拒答判断，不影响回答正文。
拒答短语 = (
    "未检索到",
    "没有找到",
    "未找到",
    "无法确定",
    "无法回答",
    "无法提供",
    "暂无可",
    "证据不足",
)

# 结构化检索条件中的字段名，用于从解析结果中读取已确认条件。
_检索条件字段 = (
    "start_date",
    "end_date",
    "years",
    "months",
    "disaster_types",
    "cities",
    "areas",
)


def _去重(items: list[str]) -> list[str]:
    """按首次出现顺序去重，保持证据顺序稳定。"""

    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _构建记忆上下文(conversation_setup: list[dict[str, Any]]) -> dict[str, Any]:
    """把黄金数据集的多轮预设转换为意图 Agent 可识别的记忆上下文。"""

    turns = []
    for item in conversation_setup or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role and content:
            turns.append({"role": role, "content": content})
    if not turns:
        return {}
    return {
        "recent_turns": turns,
        "previous_turn": dict(turns[-1]),
    }


def _提取引用证据(answer: str, evidence_ids: list[str]) -> list[str]:
    """从回答正文中提取实际出现的证据 ID，作为引用支撑率输入。"""

    cited = [evidence_id for evidence_id in evidence_ids if evidence_id and evidence_id in answer]
    return _去重(cited)


def _判断拒答(answer: str, route: str, no_result: bool) -> bool:
    """结合路由、检索结果和回答措辞判断是否属于明确拒答。"""

    if route == "clarify":
        return True
    if not no_result:
        return False
    return any(marker in answer for marker in 拒答短语)


class 真实模型执行器:
    """可注入 EvaluationHarness 的真实模型样例执行器。

    执行器复用生产链路中的意图 Agent、文档检索、结构化个例检索和回答生成，
    只省略会话持久化，保证评测结果反映真实模型与真实知识库的表现。
    """

    def __init__(
        self,
        *,
        top_k: int = 30,
        top_n: int = 5,
        enable_rerank: bool = True,
        output_dir: Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        """构造真实模型客户端、检索组件和意图 Agent，不在此阶段触网。

        top_k 控制文档向量召回的候选数量，top_n 控制重排后最终保留的证据数量；
        宽召回搭配大模型重排可以显著提升正确切片进入前五名的概率。
        """

        self.settings = settings or get_settings()
        self.top_k = max(int(top_k), 1)
        self.top_n = max(int(top_n), 1)
        self.enable_rerank = bool(enable_rerank)
        self.output_dir = Path(output_dir or ".tmp/eval-real-runtime")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_client = get_embedding_client()
        self.llm_client = get_llm_client()
        self.intent_llm_client = get_intent_llm_client()
        self.rerank_client = get_rerank_client()

        self.document_store = build_document_vector_store(
            self.settings,
            embedding_client=self.embedding_client,
        )
        self.standard_case_store = JsonStandardCaseStore(self.settings.standard_cases_path)
        self.standard_cases = self.standard_case_store.list_cases()
        # chunk 到 case 的映射用于把检索命中的片段回溯到所属标准个例。
        self.块到个例 = {
            chunk_id: case.case_id
            for case in self.standard_cases
            for chunk_id in case.source_chunk_ids
        }

        image_store = ImageEvidenceStore(self.settings.image_metadata_path)
        presenter = QueryResultPresenter(image_store)
        self.retriever = DocumentRetriever(
            embedding_client=self.embedding_client,
            document_store=self.document_store,
            standard_case_store=self.standard_case_store,
        )
        self.answerer = DocumentQuestionAnswerer(
            retriever=self.retriever,
            presenter=presenter,
            llm_client=self.llm_client,
        )
        self.intent_agent = QueryIntentAgent(
            data_dir=self.settings.smart_case_data_dir,
            llm_client=self.intent_llm_client,
            audit_path=self.output_dir / "intent_audit.jsonl",
            case_provider=self.standard_case_store.list_cases,
        )
        self.query_parser = NaturalCaseQueryParser(
            llm_client=ensure_non_thinking_client(self.llm_client),
        )
        self.structured_retriever = StructuredCaseRetriever()

    def __call__(self, case: EvaluationCase, seed: int = 0) -> EvaluationCaseResult:
        """执行单条样例并返回真实路由、证据、回答和延迟。"""

        started = time.perf_counter()
        memory = _构建记忆上下文(case.conversation_setup)
        actual_route, normalized_question = self._执行路由(case, memory)

        chunk_ids, case_ids = self._执行检索(
            case,
            question=normalized_question or case.question,
            route=actual_route,
            memory=memory,
        )
        answer = self._生成回答(
            case.question,
            chunk_ids=chunk_ids,
            case_ids=case_ids,
            route=actual_route,
        )
        no_result = not chunk_ids and not case_ids
        evidence_ids = _去重([*case_ids, *chunk_ids])
        latency_ms = int((time.perf_counter() - started) * 1000)

        return EvaluationCaseResult(
            case_id=case.case_id,
            category=str(case.category),
            question=case.question,
            actual_route=actual_route,
            actual_case_ids=case_ids,
            actual_chunk_ids=chunk_ids,
            answer=answer,
            cited_evidence_ids=_提取引用证据(answer, evidence_ids),
            no_result=no_result,
            abstained=_判断拒答(answer, actual_route, no_result),
            latency_ms=latency_ms,
            offline_fixture=False,
        )

    def close(self) -> None:
        """释放向量存储持有的连接。"""

        try:
            self.document_store.close()
        except Exception:
            pass

    def _执行路由(
        self,
        case: EvaluationCase,
        memory: dict[str, Any],
    ) -> tuple[str, str]:
        """调用真实意图 Agent 得到路由和补全后的独立问题。"""

        try:
            decision = self.intent_agent.route(
                IntentRouteRequest(message=case.question, memory_context=memory),
            )
            return str(decision.intent), str(decision.normalized_message or "")
        except Exception:
            # 意图模型不可用时保持与生产一致的保守澄清，不伪造路由。
            return "clarify", case.question

    def _执行检索(
        self,
        case: EvaluationCase,
        *,
        question: str,
        route: str,
        memory: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        """执行文档检索和结构化个例检索，按路由决定证据优先级。"""

        chunk_ids = self._文档检索(question)
        # 个例 ID 只来自显式条件解析，不把“切片所属个例”自动计入检索证据。
        # RAG 只有在唯一定位到一个个例时才附带该 ID，多候选时以切片为证据，
        # 避免个例 ID 挤占 chunk 的前排位置导致 hit@k 口径失真。
        structured_ids = self._结构化检索(case, memory)
        if route in {"multidim_search", "similar_case_match"}:
            case_ids = structured_ids
        else:
            case_ids = structured_ids if len(structured_ids) == 1 else []
        return chunk_ids, case_ids

    def _文档检索(self, question: str) -> list[str]:
        """调用生产文档检索并按 Rerank 结果重排。"""

        try:
            # 先做宽召回拿到候选集，再交给大模型重排，避免正确切片在向量阶段被截断。
            _mode, candidates = self.retriever.retrieve(
                question,
                top_k=self.top_k,
                top_n=self.top_k,
            )
        except Exception:
            return []
        hits = list(candidates)
        if self.enable_rerank and hits and self.rerank_client.is_available():
            try:
                hits = self.rerank_client.rerank(question, hits, self.top_n)
            except Exception:
                # Rerank 失败时保留向量检索顺序，符合生产降级策略。
                hits = hits[: self.top_n]
        else:
            hits = hits[: self.top_n]
        return _去重([str(getattr(hit.chunk, "chunk_id", "")) for hit in hits])

    def _结构化检索(self, case: EvaluationCase, memory: dict[str, Any]) -> list[str]:
        """用自然语言解析加确定性过滤得到标准个例 ID。"""

        try:
            committed = self._回放历史条件(case.conversation_setup)
            _op, request, _display, _changes, _warnings, _status, _ok = self.query_parser.parse(
                case.question,
                committed,
            )
            matches = self.structured_retriever.search(self.standard_cases, request.to_query())
        except Exception:
            return []
        return _去重([match.case.case_id for match in matches])

    def _回放历史条件(self, conversation_setup: list[dict[str, Any]]) -> Any:
        """按顺序回放历史用户消息，恢复多轮场景下已确认的检索条件。"""

        committed = None
        for item in conversation_setup or []:
            if not isinstance(item, dict) or item.get("role") != "user":
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            try:
                _op, committed, *_rest = self.query_parser.parse(content, committed)
            except Exception:
                continue
        return committed

    def _生成回答(
        self,
        question: str,
        *,
        chunk_ids: list[str],
        case_ids: list[str],
        route: str,
    ) -> str:
        """基于真实证据生成回答；模型失败时退回可核验的证据摘要。"""

        blocks = self._证据上下文(chunk_ids, case_ids, route=route)
        if not blocks:
            return "未检索到与该问题相关的权威证据，无法据此给出结论。"
        try:
            answer = self.llm_client.answer_with_context(
                question,
                blocks,
                max_tokens=800,
            ).strip()
            if answer:
                return answer
        except Exception:
            pass
        return self._证据摘要(blocks, route)

    def _证据上下文(
        self,
        chunk_ids: list[str],
        case_ids: list[str],
        *,
        route: str,
    ) -> list[str]:
        """按路由整理证据块：RAG 优先原文，结构化检索优先标准个例。"""

        blocks: list[str] = []
        cases_by_id = {case.case_id: case for case in self.standard_cases}
        prefer_cases = route in {"multidim_search", "similar_case_match"}

        if prefer_cases:
            blocks.extend(self._个例证据块(case_ids, cases_by_id))
            if not blocks:
                blocks.extend(self._切片证据块(chunk_ids))
        else:
            blocks.extend(self._切片证据块(chunk_ids))
            if not blocks:
                blocks.extend(self._个例证据块(case_ids, cases_by_id))
        return blocks

    def _个例证据块(self, case_ids: list[str], cases_by_id: dict[str, Any]) -> list[str]:
        """把命中的标准个例转换为模型可读证据块。"""

        blocks: list[str] = []
        for case_id in case_ids[: self.top_n]:
            case = cases_by_id.get(case_id)
            if case is None:
                continue
            blocks.append(
                "\n".join(
                    (
                        f"个例ID：{case.case_id}",
                        f"标题：{case.title}",
                        f"时间：{case.date_range}",
                        f"灾种：{'、'.join(case.disaster_types)}",
                        f"影响区域：{'、'.join(case.affected_areas)}",
                        f"摘要：{case.summary or case.weather_facts}",
                    )
                )
            )
        return blocks

    def _切片证据块(self, chunk_ids: list[str]) -> list[str]:
        """把命中的文档切片转换为模型可读证据块。"""

        blocks: list[str] = []
        for chunk_id in chunk_ids[: self.top_n]:
            try:
                chunk = self.document_store.get_chunk(chunk_id)
            except Exception:
                chunk = None
            if chunk is None:
                continue
            blocks.append(
                "\n".join(
                    (
                        f"切片ID：{chunk.chunk_id}",
                        f"来源：{chunk.source_pdf}",
                        f"正文：{chunk.content}",
                    )
                )
            )
        return blocks

    @staticmethod
    def _证据摘要(blocks: list[str], route: str) -> str:
        """模型不可用时返回不超出证据范围的本地摘要。"""

        label = "标准个例" if route in {"multidim_search", "similar_case_match"} else "原文证据"
        return f"模型暂不可用，以下为检索到的{label}摘要：\n" + "\n\n".join(blocks[:3])
