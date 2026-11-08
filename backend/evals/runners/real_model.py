"""真实模型评测执行器：调用真实云端模型与真实检索链路，不使用离线夹具。"""
from __future__ import annotations

import re
import time
from calendar import monthrange
from datetime import date
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
from backend.app.agents.smart_case_match.agent import SmartCaseMatchAgent
from backend.app.agents.smart_case_match.schemas import SmartCaseMatchRequest
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
from backend.app.services.query.evidence_window import (
    diversify_evidence_by_case,
    dynamic_evidence_window,
)
from backend.app.services.query.document_retrieval import DocumentRetriever
from backend.app.services.query.result_presenters import QueryResultPresenter
from backend.app.services.query.structured_facts import (
    extract_structured_facts,
    render_structured_fact_block,
)
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

# 库内比较仍由主 RAG 回答，但需要相似 Agent 给出更可靠的个例排序。
_相似比较表达 = (
    "比较",
    "对比",
    "相似",
    "差异",
    "相近",
    "相同点",
    "不同点",
)


def _需要相似度补充(question: str) -> bool:
    """判断 RAG 问题是否需要相似个例 Agent 提供补充排序。"""

    return any(term in str(question or "") for term in _相似比较表达)


def _是否模型降级演练(question: str) -> bool:
    """识别明确要求验证聊天模型不可用降级的样例。"""

    compact = re.sub(r"\s+", "", str(question or ""))
    return "聊天模型不可用" in compact or "模型服务不可用" in compact


def _应沿用上一轮结果(question: str) -> bool:
    """判断当前问句是否明确要求在当前候选集合内继续操作。"""

    terms = (
        "刚才", "刚刚", "上一轮", "上述", "上面", "这些结果", "那些结果",
        "当前结果", "现有结果", "其中", "第一个", "第二个", "第三个",
        "第一个个例", "第二个个例", "继续比较", "继续按", "再查一次", "再补充",
    )
    return any(term in str(question or "") for term in terms)


def _查询条件窗口(query: Any) -> tuple[date | None, date | None]:
    """从结构化查询中计算可比较的起止日期窗口。"""

    if query is None:
        return None, None
    data = query.model_dump() if hasattr(query, "model_dump") else dict(query)
    start: date | None = None
    end: date | None = None
    years = [int(value) for value in (data.get("years") or []) if str(value).isdigit()]
    months = [int(value) for value in (data.get("months") or []) if str(value).isdigit()]
    if years:
        if months:
            start = date(min(years), min(months), 1)
            last_year = max(years)
            last_month = max(months)
            end = date(last_year, last_month, monthrange(last_year, last_month)[1])
        else:
            start = date(min(years), 1, 1)
            end = date(max(years), 12, 31)
    try:
        if data.get("start_date"):
            start = date.fromisoformat(str(data["start_date"]))
        if data.get("end_date"):
            end = date.fromisoformat(str(data["end_date"]))
    except ValueError:
        pass
    return start, end


def _窗口超出知识范围(
    query: Any,
    knowledge_start: str,
    knowledge_end: str,
) -> bool:
    """判断显式查询窗口是否完全落在当前知识库时间范围之外。"""

    start, end = _查询条件窗口(query)
    if start is None and end is None:
        return False
    try:
        kb_start = date.fromisoformat(str(knowledge_start))
        kb_end = date.fromisoformat(str(knowledge_end))
    except ValueError:
        return False
    query_start = start or end
    query_end = end or start
    return bool(query_start and query_end and (query_end < kb_start or query_start > kb_end))


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


def _日期线索(text: str) -> list[dict[str, int | None]]:
    """按原文顺序提取年份、月份和日期范围线索。"""

    pattern = re.compile(
        r"(?:(?P<year>20\d{2})\s*年\s*)?"
        r"(?P<month>1[0-2]|0?[1-9])\s*月"
        r"(?:\s*(?P<start>3[01]|[12]\d|0?[1-9])\s*日?"
        r"(?:\s*(?:至|到|~|～|-|—)\s*(?P<end>3[01]|[12]\d|0?[1-9])\s*日?)?)?"
    )
    clues: list[dict[str, int | None]] = []
    for match in pattern.finditer(str(text or "")):
        year = int(match.group("year")) if match.group("year") else None
        month = int(match.group("month"))
        start = int(match.group("start")) if match.group("start") else None
        end = int(match.group("end")) if match.group("end") else start
        clue = {"year": year, "month": month, "start": start, "end": end}
        if clue not in clues:
            clues.append(clue)
    return clues


def _个例日期线索(case: Any) -> dict[str, int | None] | None:
    """从标准个例的日期范围中提取可比较的年月日。"""

    clues = _日期线索(str(getattr(case, "date_range", "") or ""))
    return clues[0] if clues else None


def _时间线索匹配(case_clue: dict[str, int | None], query_clue: dict[str, int | None]) -> bool:
    """判断个例日期是否落在用户提到的年份、月份或日期范围内。"""

    if query_clue.get("year") and case_clue.get("year") != query_clue["year"]:
        return False
    if case_clue.get("month") != query_clue.get("month"):
        return False
    query_start = query_clue.get("start")
    if query_start is None:
        return True
    case_start = case_clue.get("start") or case_clue.get("month") or 1
    case_end = case_clue.get("end") or case_start
    query_end = query_clue.get("end") or query_start
    return max(case_start, query_start) <= min(case_end, query_end)


def _按日期线索匹配个例(
    standard_cases: list[Any],
    clues: list[dict[str, int | None]],
) -> list[Any]:
    """按用户原文中的日期出现顺序返回标准个例，并保持稳定去重。"""

    matches: list[Any] = []
    seen: set[str] = set()
    for clue in clues:
        matching_cases = []
        exact_cases = []
        for case in standard_cases:
            case_id = str(getattr(case, "case_id", "") or "")
            case_clue = _个例日期线索(case)
            if not case_id or not case_clue or case_id in seen:
                continue
            if _时间线索匹配(case_clue, clue):
                matching_cases.append(case)
                if (
                    clue.get("start") is not None
                    and case_clue.get("start") == clue.get("start")
                    and case_clue.get("end") == clue.get("end")
                ):
                    exact_cases.append(case)
        # 日期范围完全一致时优先精确个例，避免范围中的单日过程抢占。
        for case in exact_cases or matching_cases:
            case_id = str(getattr(case, "case_id", "") or "")
            if case_id and case_id not in seen:
                matches.append(case)
                seen.add(case_id)
    return matches


def _个例引用(case: Any) -> dict[str, Any]:
    """生成意图 Agent 可校验、回答阶段可复用的轻量个例引用。"""

    return {
        "case_id": str(getattr(case, "case_id", "") or ""),
        "title": str(getattr(case, "title", "") or ""),
        "date_range": str(getattr(case, "date_range", "") or ""),
        "disaster_types": list(getattr(case, "disaster_types", []) or []),
        "affected_areas": list(getattr(case, "affected_areas", []) or []),
    }


def _构建记忆上下文(
    conversation_setup: list[dict[str, Any]],
    standard_cases: list[Any] | None = None,
) -> dict[str, Any]:
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

    previous_question = next(
        (item["content"] for item in reversed(turns) if item["role"] == "user"),
        "",
    )
    previous_answer = next(
        (item["content"] for item in reversed(turns) if item["role"] == "assistant"),
        "",
    )
    question_clues = _日期线索(previous_question)
    answer_clues = _日期线索(previous_answer)
    # 问题只给出月份、而上一轮回答明确到具体日期时，以回答中的个例为准；
    # 否则保留问题里的顺序，支持“第一个/第二个”等后续指代。
    if (question_clues and all(item.get("start") is None for item in question_clues)
            and any(item.get("start") is not None for item in answer_clues)):
        reference_clues = answer_clues
    elif question_clues:
        reference_clues = question_clues
    else:
        reference_clues = answer_clues

    refs = _按日期线索匹配个例(list(standard_cases or []), reference_clues)
    case_refs = [_个例引用(case) for case in refs]
    query_conditions: dict[str, Any] = {}
    if previous_question:
        try:
            parser = NaturalCaseQueryParser(llm_client=None)
            _operation, committed, *_rest = parser.parse(previous_question, None)
            if committed is not None and hasattr(committed, "to_query"):
                query_conditions = dict(committed.to_query())
        except Exception:
            # 条件是可选增强项，解析失败不能阻断历史引用。
            query_conditions = {}

    previous_turn = {
        "role": "assistant" if previous_answer else "user",
        "content": previous_answer or previous_question,
        "question": previous_question,
        "answer_summary": previous_answer,
        "case_refs": case_refs,
        "query_conditions": query_conditions,
    }
    return {
        "recent_turns": turns,
        "previous_turn": previous_turn,
        "available_case_refs": case_refs,
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
        self.smart_match_agent = SmartCaseMatchAgent(
            data_dir=self.settings.smart_case_data_dir,
            embedding_client=self.embedding_client,
            rerank_client=self.rerank_client,
            llm_client=self.llm_client,
        )

    def __call__(self, case: EvaluationCase, seed: int = 0) -> EvaluationCaseResult:
        """执行单条样例并返回真实路由、证据、回答和延迟。"""

        started = time.perf_counter()
        memory = _构建记忆上下文(case.conversation_setup, self.standard_cases)
        actual_route, normalized_question, referenced_case_ids, clarification = self._执行路由(case, memory)

        if actual_route == "clarify":
            # 澄清路由不执行检索，避免把近似片段误当成可回答证据。
            chunk_ids: list[str] = []
            case_ids: list[str] = []
            answer = clarification or "当前条件不足或请求超出权限，无法给出可靠结论。请补充明确的个例、时间、区域或灾种。"
        else:
            chunk_ids, case_ids, answer_case_ids = self._执行检索(
                case,
                question=normalized_question or case.question,
                route=actual_route,
                memory=memory,
                referenced_case_ids=referenced_case_ids,
            )
            answer = self._生成回答(
                case.question,
                chunk_ids=chunk_ids,
                case_ids=answer_case_ids,
                route=actual_route,
                memory=memory,
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
        """释放相似 Agent 和向量存储持有的连接。"""

        try:
            self.smart_match_agent.close()
        except Exception:
            pass
        try:
            self.document_store.close()
        except Exception:
            pass

    def _执行路由(
        self,
        case: EvaluationCase,
        memory: dict[str, Any],
    ) -> tuple[str, str, list[str], str]:
        """调用真实意图 Agent 得到路由、独立问题、历史个例和澄清提示。"""

        try:
            decision = self.intent_agent.route(
                IntentRouteRequest(message=case.question, memory_context=memory),
            )
            return (
                str(decision.intent),
                str(decision.normalized_message or ""),
                list(decision.referenced_case_ids or []),
                str(decision.clarification_question or ""),
            )
        except Exception:
            # 意图模型不可用时保持与生产一致的保守澄清，不伪造路由。
            return "clarify", case.question, [], "当前请求无法安全路由，请补充明确的时间、区域和灾种。"

    def _执行检索(
        self,
        case: EvaluationCase,
        *,
        question: str,
        route: str,
        memory: dict[str, Any],
        referenced_case_ids: list[str] | None = None,
    ) -> tuple[list[str], list[str], list[str]]:
        """执行检索，返回排序证据、实际证据个例和回答上下文个例。"""

        # 个例 ID 只来自显式条件解析，不把“切片所属个例”自动映射成证据；
        # 但明确解析出的多个标准个例本身就是有效结构化证据，RAG 回答必须
        # 同时看到完整候选集，否则容易只依据局部切片漏掉部分过程。
        resolver = getattr(self, "_解析当前条件", None)
        query = resolver(case) if callable(resolver) else None
        if route == "rag" and query is not None:
            knowledge = self.intent_agent.knowledge_range()
            if _窗口超出知识范围(query, knowledge.start_date, knowledge.end_date):
                # 明确超出知识库范围时返回空结果，不能继续做向量近邻从而制造伪命中。
                return [], [], []
        structured_ids = (
            self._结构化检索(case, memory)
            if query is None
            else self._结构化检索(case, memory, query=query)
        )
        follow_up_ids = (
            _去重([value for value in (referenced_case_ids or []) if value])
            if _应沿用上一轮结果(case.question)
            else []
        )
        if follow_up_ids:
            intersection = [case_id for case_id in follow_up_ids if case_id in structured_ids]
            if intersection:
                # 保留上一轮展示顺序，不用检索器返回的逆序覆盖。
                structured_ids = intersection
            elif not _日期线索(case.question):
                # 用户只修改地区、灾种或展示方式时，日期条件仍沿用上一轮；
                # 没有新日期就不应被当前结构化解析结果带偏。
                structured_ids = follow_up_ids
        evidence_window = dynamic_evidence_window(
            question,
            self.top_n,
            matched_case_count=len(structured_ids),
        )
        chunk_ids = self._文档检索(
            question,
            top_n=evidence_window,
            diversify=len(structured_ids) > 1 or evidence_window > self.top_n,
        )
        if follow_up_ids:
            # 多轮承接必须在该轮已展示的个例集合内回答，不能重新扩展到全库。
            case_ids = _去重(structured_ids)
            answer_case_ids = case_ids
        elif route == "similar_case_match":
            # 相似匹配必须使用真实 Agent 的融合排序和 LLM 重排结果；
            # 结构化条件仅作为 Agent 故障时的保守降级。
            case_ids = self._相似匹配检索(question) or structured_ids
            answer_case_ids = case_ids
        elif route == "rag" and _需要相似度补充(question):
            # 路由契约仍是主 RAG，Agent 只补充个例排序，不接管回答。
            case_ids = _去重([*self._相似匹配检索(question), *structured_ids])
            answer_case_ids = case_ids
        elif route == "multidim_search":
            case_ids = structured_ids
            answer_case_ids = case_ids
        else:
            # 普通 RAG 只有唯一定位时才把个例计入实际证据排名，避免挤占切片；
            # 多候选个例仍进入回答上下文，保证回答能覆盖完整候选集。
            case_ids = structured_ids if len(structured_ids) == 1 else []
            answer_case_ids = structured_ids
        return chunk_ids, case_ids, answer_case_ids

    def _相似匹配检索(self, question: str) -> list[str]:
        """调用相似个例 Agent，并按排名返回真实匹配到的标准个例 ID。"""

        try:
            response = self.smart_match_agent.run(
                SmartCaseMatchRequest(
                    raw_query=question,
                    top_n=max(3, min(int(self.top_n), 5)),
                    include_images=False,
                )
            )
        except Exception:
            return []
        return _去重(
            [
                str(item.get("case_id") or "")
                for item in (getattr(response, "matched_cases", None) or [])
                if isinstance(item, dict)
            ]
        )

    def _文档检索(
        self,
        question: str,
        *,
        top_n: int | None = None,
        diversify: bool = False,
    ) -> list[str]:
        """调用生产文档检索、动态限流，并在多案例问题中做证据多样化。"""

        window = max(int(top_n or self.top_n), 1)
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
                hits = self.rerank_client.rerank(question, hits, window)
            except Exception:
                # Rerank 失败时保留向量检索顺序，符合生产降级策略。
                hits = hits[:window]
        else:
            hits = hits[:window]
        if diversify:
            hits = diversify_evidence_by_case(
                hits,
                self.块到个例,
                limit=window,
            )
        return _去重([str(getattr(hit.chunk, "chunk_id", "")) for hit in hits])

    def _解析当前条件(self, case: EvaluationCase) -> Any:
        """结合历史条件解析当前查询，供检索和知识范围判断复用。"""

        try:
            committed = self._回放历史条件(case.conversation_setup)
            _op, request, _display, _changes, _warnings, _status, _ok = self.query_parser.parse(
                case.question,
                committed,
            )
            return request.to_query()
        except Exception:
            return None

    def _结构化检索(
        self,
        case: EvaluationCase,
        memory: dict[str, Any],
        *,
        query: Any = None,
    ) -> list[str]:
        """用自然语言解析加确定性过滤得到标准个例 ID。"""

        try:
            query = query if query is not None else self._解析当前条件(case)
            if query is None:
                return []
            matches = self.structured_retriever.search(self.standard_cases, query)
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
        memory: dict[str, Any] | None = None,
    ) -> str:
        """基于真实证据生成回答；模型失败时退回可核验的证据摘要。"""

        blocks = self._证据上下文(chunk_ids, case_ids, route=route)
        memory_block = self._记忆上下文块(memory or {})
        if memory_block:
            blocks.insert(0, memory_block)
        if not blocks:
            return "未找到与该问题相关的权威证据，无法据此给出结论。"
        if _是否模型降级演练(question):
            # 故障演练必须展示真实的证据降级路径，不能调用当前可用模型伪造结果。
            return self._证据摘要(blocks, route)
        try:
            answer = self.llm_client.answer_with_context(
                question,
                blocks,
                max_tokens=2048,
            ).strip()
            if answer:
                return answer
        except Exception:
            pass
        return self._证据摘要(blocks, route)

    @staticmethod
    def _记忆上下文块(memory: dict[str, Any]) -> str:
        """把上一轮问题、摘要和个例 ID 作为低权限承接信息提供给回答模型。"""

        previous = memory.get("previous_turn") if isinstance(memory, dict) else None
        if not isinstance(previous, dict) or not previous:
            return ""
        case_refs = [
            item for item in (previous.get("case_refs") or [])
            if isinstance(item, dict) and item.get("case_id")
        ]
        lines = ["会话承接上下文（只用于理解指代，事实必须由本轮证据支持）："]
        question = str(previous.get("question") or "").strip()
        answer_summary = str(previous.get("answer_summary") or "").strip()
        if question:
            lines.append(f"上一轮问题：{question[:500]}")
        if answer_summary:
            lines.append(f"上一轮结果摘要：{answer_summary[:800]}")
        if case_refs:
            lines.append(
                "上一轮个例："
                + "；".join(
                    f"{item['case_id']}《{str(item.get('title') or '')[:80]}》"
                    for item in case_refs[:20]
                )
            )
        return "\n".join(lines)
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
        fact_block = self._结构化事实块(
            chunk_ids,
            case_ids,
            cases_by_id,
        )
        if fact_block:
            blocks.append(fact_block)
        return blocks

    def _结构化事实块(
        self,
        chunk_ids: list[str],
        case_ids: list[str],
        cases_by_id: dict[str, Any],
    ) -> str:
        """补充标准个例元数据和来自原文的逐字事实，减少回答遗漏。"""

        metadata_lines: list[str] = []
        source_chunk_ids: list[str] = []
        for case_id in case_ids[:10]:
            case = cases_by_id.get(case_id)
            if case is None:
                continue
            metadata_lines.append(
                f"- 个例ID：{case.case_id}；标题：{case.title}；时间：{case.date_range}；"
                f"灾种：{'、'.join(case.disaster_types)}；影响区域：{'、'.join(case.affected_areas)}"
            )
            source_chunk_ids.extend(list(getattr(case, "source_chunk_ids", []) or []))

        facts = []
        for evidence_id in _去重([*source_chunk_ids, *chunk_ids])[:16]:
            try:
                chunk = self.document_store.get_chunk(evidence_id)
            except Exception:
                chunk = None
            if chunk is None:
                continue
            facts.extend(
                extract_structured_facts(
                    chunk.content,
                    evidence_id=chunk.chunk_id,
                    limit=8,
                )
            )
        fact_block = render_structured_fact_block(facts, max_facts=32)
        parts: list[str] = []
        if metadata_lines:
            parts.append("标准个例元数据：\n" + "\n".join(metadata_lines))
        if fact_block:
            parts.append(fact_block)
        return "\n\n".join(parts)

    def _个例证据块(self, case_ids: list[str], cases_by_id: dict[str, Any]) -> list[str]:
        """把命中的标准个例转换为模型可读证据块。"""

        blocks: list[str] = []
        for case_id in case_ids[:10]:
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
        for chunk_id in chunk_ids[:10]:
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
