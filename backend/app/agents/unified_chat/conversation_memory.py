"""统一聊天的会话内分层记忆管理。"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from typing import Any


MEMORY_AGENT_TYPE = "conversation_memory"


class ConversationMemoryManager:
    """从原始消息生成可控长度的分层记忆，并按 session_id 持久化。"""

    def __init__(
        self,
        session_store,
        *,
        token_budget: int = 6000,
        reserve_ratio: float = 0.30,
        recent_turn_limit: int = 6,
        case_provider: Callable[[], list[Any]] | None = None,
    ) -> None:
        self.session_store = session_store
        self.token_budget = max(1200, int(token_budget))
        self.reserve_ratio = min(0.6, max(0.1, float(reserve_ratio)))
        self.recent_turn_limit = max(2, int(recent_turn_limit))
        self.case_provider = case_provider

    def build_context(self, session_id: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """同步当前会话的摘要状态并返回提供给意图模型的记忆包。"""
        if not session_id:
            return self._empty_packet("")
        state = self.session_store.get_agent_state(session_id, MEMORY_AGENT_TYPE) or self._new_state()
        knowledge_cases, knowledge_version = self._knowledge_catalog()
        # 标准个例库更新后从原始聊天记录重新生成派生记忆，避免旧版虚假 ID 继续污染后续路由。
        if knowledge_version and state.get("knowledge_version") != knowledge_version:
            state = self._new_state()
        self._sync_messages(state, messages)
        self._validate_state_case_refs(state, knowledge_cases)
        self._compress_state(state)
        packet = self._packet(session_id, state)
        packet["knowledge_version"] = knowledge_version
        state["knowledge_version"] = knowledge_version
        state["estimated_context_tokens"] = packet["budget"]["estimated_tokens"]
        self.session_store.save_agent_state(session_id, MEMORY_AGENT_TYPE, state)
        return packet

    def load_context(self, session_id: str) -> dict[str, Any]:
        """优先读取已保存记忆；后台尚未覆盖上一轮时只增量补齐。"""
        if not session_id:
            return self._empty_packet("")
        state = self.session_store.get_agent_state(session_id, MEMORY_AGENT_TYPE) or self._new_state()
        covered = int(state.get("covered_message_id") or 0)
        pending_messages = self.session_store.list_messages_after(session_id, covered)
        if any(str(item.get("role") or "") == "assistant" for item in pending_messages):
            # 正常情况下后台任务已经更新完成；这里只处理用户连续快速提问时的竞态窗口。
            return self.build_context(session_id, pending_messages)
        return self._packet(session_id, state)

    def refresh(self, session_id: str) -> dict[str, Any]:
        """一轮回答落库后立即更新摘要，保证下一轮可直接读取。"""
        if not session_id:
            return self._empty_packet("")
        state = self.session_store.get_agent_state(session_id, MEMORY_AGENT_TYPE) or self._new_state()
        covered = int(state.get("covered_message_id") or 0)
        return self.build_context(session_id, self.session_store.list_messages_after(session_id, covered))

    def context_before_message(self, session_id: str, message_id: int) -> dict[str, Any]:
        """临时构造目标消息之前的记忆，不覆盖当前会话已经保存的摘要状态。"""
        if not session_id:
            return self._empty_packet("")
        state = self._new_state()
        knowledge_cases, knowledge_version = self._knowledge_catalog()
        messages = self.session_store.list_messages_before(session_id, int(message_id))
        self._sync_messages(state, messages)
        self._validate_state_case_refs(state, knowledge_cases)
        self._compress_state(state)
        packet = self._packet(session_id, state)
        packet["knowledge_version"] = knowledge_version
        return packet

    def _sync_messages(self, state: dict[str, Any], messages: list[dict[str, Any]]) -> None:
        """仅为尚未覆盖的助手消息生成轮次摘要，避免重复写入。"""
        covered = int(state.get("covered_message_id") or 0)
        latest_user: dict[str, Any] | None = None
        recent_turns = list(state.get("recent_turns") or [])
        for message in messages:
            role = str(message.get("role") or "")
            message_id = int(message.get("message_id") or 0)
            if role == "user":
                latest_user = message
                continue
            if role != "assistant" or message_id <= covered:
                continue
            recent_turns.append(self._turn_digest(latest_user, message))
            covered = max(covered, message_id)
        state["recent_turns"] = recent_turns
        state["covered_message_id"] = covered

    def _turn_digest(self, user_message: dict[str, Any] | None, assistant_message: dict[str, Any]) -> dict[str, Any]:
        """保留指代消解需要的业务实体，并压缩自然语言正文。"""
        metadata = assistant_message.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        memory = metadata.get("memory") if isinstance(metadata.get("memory"), dict) else {}
        evidence_cases = metadata.get("evidence_cases") or memory.get("evidence_cases") or []
        agent_result = metadata.get("agent_result") or memory.get("agent_result") or {}
        matched_cases = agent_result.get("matched_cases") if isinstance(agent_result, dict) else []
        case_refs = self._case_refs([*(evidence_cases or []), *(matched_cases or [])])
        agent_type = str(metadata.get("agent_type") or memory.get("agent_type") or "")
        return {
            "user_message_id": int((user_message or {}).get("message_id") or 0),
            "assistant_message_id": int(assistant_message.get("message_id") or 0),
            "question": self._compact_text(str((user_message or {}).get("content") or ""), 800),
            "answer_summary": self._compact_text(str(assistant_message.get("content") or ""), 700),
            "agent_type": agent_type,
            "intent": self._intent_from_agent(agent_type, metadata),
            "retrieval_mode": str(metadata.get("retrieval_mode") or memory.get("retrieval_mode") or ""),
            "case_refs": case_refs,
            "conditions": metadata.get("query_conditions") or memory.get("query_conditions") or {},
            "run_id": str(metadata.get("run_id") or memory.get("run_id") or ""),
            "reports": [
                {
                    "report_id": str(item.get("report_id") or ""),
                    "title": str(item.get("title") or item.get("filename") or ""),
                }
                for item in (metadata.get("reports") or memory.get("reports") or [])[:4]
                if isinstance(item, dict)
            ],
        }

    def _compress_state(self, state: dict[str, Any]) -> None:
        """旧轮次逐步并入长期摘要，始终为当前问题和模型输出预留 30% 空间。"""
        recent_turns = list(state.get("recent_turns") or [])
        while len(recent_turns) > self.recent_turn_limit:
            self._merge_long_term(state, recent_turns.pop(0))
        state["recent_turns"] = recent_turns

        history_budget = int(self.token_budget * (1.0 - self.reserve_ratio))
        while len(recent_turns) > 1 and self._estimate_tokens(self._packet("", state)) > history_budget:
            self._merge_long_term(state, recent_turns.pop(0))
            state["recent_turns"] = recent_turns

        if self._estimate_tokens(self._packet("", state)) > history_budget and recent_turns:
            recent_turns[-1]["answer_summary"] = self._compact_text(recent_turns[-1].get("answer_summary", ""), 360)
        state["recent_turns"] = recent_turns

    def _merge_long_term(self, state: dict[str, Any], turn: dict[str, Any]) -> None:
        """将较早轮次压缩为主题摘要，同时单独保存不可丢失的个例标识。"""
        case_titles = "、".join(item.get("title") or item.get("case_id") or "" for item in turn.get("case_refs") or [])
        line = f"问题：{self._compact_text(turn.get('question', ''), 120)}；结论：{self._compact_text(turn.get('answer_summary', ''), 180)}"
        if case_titles:
            line += f"；涉及个例：{self._compact_text(case_titles, 180)}"
        reference_type = str(turn.get("reference_type") or "none")
        if reference_type != "none":
            line += f"；实体来源：{reference_type}"
            state["long_term_reference_types"] = list(dict.fromkeys([
                *(state.get("long_term_reference_types") or []),
                reference_type,
            ]))
        existing = str(state.get("long_term_summary") or "")
        combined = "\n".join(item for item in (existing, line) if item)
        state["long_term_summary"] = self._progressive_compact(combined, 2200)
        state["long_term_case_refs"] = self._case_refs(
            [*(state.get("long_term_case_refs") or []), *(turn.get("case_refs") or [])],
            limit=40,
        )

    def _packet(self, session_id: str, state: dict[str, Any]) -> dict[str, Any]:
        """构建“长期摘要 + 较近轮次 + 上一轮详细摘要”的模型输入。"""
        turns = list(state.get("recent_turns") or [])
        previous_turn = dict(turns[-1]) if turns else {}
        medium_turns = [self._medium_turn(turn) for turn in turns[:-1][-4:]]
        available_refs = self._case_refs(
            [
                *(previous_turn.get("case_refs") or []),
                *(item for turn in reversed(medium_turns) for item in (turn.get("case_refs") or [])),
                *(state.get("long_term_case_refs") or []),
            ],
            limit=40,
        )
        packet = {
            "version": 2,
            "session_id": session_id,
            "long_term_summary": str(state.get("long_term_summary") or ""),
            "recent_turns": medium_turns,
            "previous_turn": previous_turn,
            "available_case_refs": available_refs,
            "long_term_reference_types": list(state.get("long_term_reference_types") or []),
            "previous_intent": str(previous_turn.get("intent") or "") or None,
            "covered_message_id": int(state.get("covered_message_id") or 0),
        }
        estimated = self._estimate_tokens(packet)
        packet["budget"] = {
            "token_budget": self.token_budget,
            "reserve_ratio": self.reserve_ratio,
            "history_limit": int(self.token_budget * (1.0 - self.reserve_ratio)),
            "estimated_tokens": estimated,
            "compressed": bool(state.get("long_term_summary")),
        }
        return packet

    def _medium_turn(self, turn: dict[str, Any]) -> dict[str, Any]:
        """较近但非上一轮的内容使用中等精度摘要。"""
        return {
            "user_message_id": turn.get("user_message_id", 0),
            "assistant_message_id": turn.get("assistant_message_id", 0),
            "question": self._compact_text(turn.get("question", ""), 320),
            "answer_summary": self._compact_text(turn.get("answer_summary", ""), 260),
            "agent_type": turn.get("agent_type", ""),
            "intent": turn.get("intent", ""),
            "retrieval_mode": turn.get("retrieval_mode", ""),
            "reference_type": turn.get("reference_type", "none"),
            "case_refs": turn.get("case_refs") or [],
            "conditions": turn.get("conditions") or {},
        }

    def _case_refs(self, items: list[Any], limit: int = 20) -> list[dict[str, Any]]:
        """按 case_id 去重，保留指代和本地检索所需的最小字段。"""
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            case_id = str(item.get("case_id") or "").strip()
            if not case_id or case_id in seen:
                continue
            seen.add(case_id)
            result.append({
                "case_id": case_id,
                "title": str(item.get("title") or item.get("case_title") or ""),
                "date_range": str(item.get("date_range") or ""),
                "disaster_types": [str(value) for value in (item.get("disaster_types") or [])[:8]],
                "affected_areas": [str(value) for value in (item.get("affected_areas") or [])[:8]],
                "source_pdf": str(item.get("source_pdf") or ""),
                "reference_type": str(item.get("reference_type") or "knowledge_case"),
            })
            if len(result) >= limit:
                break
        return result

    def _knowledge_catalog(self) -> tuple[dict[str, dict[str, Any]] | None, str]:
        """读取当前标准个例并生成稳定版本号；读取失败时不破坏已有会话记忆。"""
        if self.case_provider is None:
            return None, ""
        try:
            cases = list(self.case_provider() or [])
        except Exception:
            return None, ""
        catalog: dict[str, dict[str, Any]] = {}
        version_rows: list[str] = []
        for case in cases:
            if isinstance(case, dict):
                item = case
            else:
                item = {
                    key: getattr(case, key, None)
                    for key in ("case_id", "title", "date_range", "disaster_types", "affected_areas", "source_pdf")
                }
            case_id = str(item.get("case_id") or "").strip()
            if not case_id:
                continue
            canonical = {
                "case_id": case_id,
                "title": str(item.get("title") or ""),
                "date_range": str(item.get("date_range") or ""),
                "disaster_types": [str(value) for value in (item.get("disaster_types") or [])[:8]],
                "affected_areas": [str(value) for value in (item.get("affected_areas") or [])[:8]],
                "source_pdf": str(item.get("source_pdf") or ""),
                "reference_type": "knowledge_case",
            }
            catalog[case_id] = canonical
            version_rows.append("|".join((case_id, canonical["title"], canonical["date_range"], canonical["source_pdf"])))
        digest = hashlib.sha256("\n".join(sorted(version_rows)).encode("utf-8")).hexdigest()
        return catalog, digest

    def _validate_state_case_refs(
        self,
        state: dict[str, Any],
        catalog: dict[str, dict[str, Any]] | None,
    ) -> None:
        """仅保留当前标准个例库真实存在的引用，并用当前库字段覆盖旧摘要字段。"""
        if catalog is None:
            return

        def valid_refs(items: list[Any]) -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = []
            seen: set[str] = set()
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                case_id = str(item.get("case_id") or "").strip()
                if case_id in catalog and case_id not in seen:
                    seen.add(case_id)
                    result.append(dict(catalog[case_id]))
            return result

        for turn in state.get("recent_turns") or []:
            if isinstance(turn, dict):
                turn["case_refs"] = valid_refs(turn.get("case_refs") or [])
                turn_source = self._turn_reference_type(turn)
                turn["reference_type"] = turn_source if turn_source != "none" else (
                    "knowledge_case" if turn["case_refs"] else "none"
                )
        state["long_term_case_refs"] = valid_refs(state.get("long_term_case_refs") or [])
        state["long_term_reference_types"] = [
            value
            for value in dict.fromkeys(str(item) for item in state.get("long_term_reference_types") or [])
            if value in {"knowledge_case", "external_process", "aggregate_result"}
        ]

    @staticmethod
    def _turn_reference_type(turn: dict[str, Any]) -> str:
        """区分库内个例、Smart 外部过程和普通聚合结果，供意图模型理解实体来源。"""
        agent_type = str(turn.get("agent_type") or "")
        if agent_type == "smart_case_match":
            return "external_process"
        if str(turn.get("retrieval_mode") or "").startswith("structured_cases") and not turn.get("case_refs"):
            return "aggregate_result"
        return "none"

    @staticmethod
    def _intent_from_agent(agent_type: str, metadata: dict[str, Any]) -> str:
        """把实际执行的 Agent 映射回意图白名单。"""
        if agent_type == "case_multidim_search":
            return "multidim_search"
        if agent_type == "smart_case_match":
            return "similar_case_match"
        if agent_type == "document_rag":
            return "rag"
        route = (metadata.get("intent_trace") or {}).get("route") if isinstance(metadata.get("intent_trace"), dict) else ""
        return str(route or "")

    @staticmethod
    def _compact_text(value: Any, limit: int) -> str:
        """按句子边界压缩文本，避免在普通词语中间生硬截断。"""
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) <= limit:
            return text
        prefix = text[:limit]
        boundary = max(prefix.rfind("。"), prefix.rfind("；"), prefix.rfind("！"), prefix.rfind("？"))
        return (prefix[: boundary + 1] if boundary >= int(limit * 0.55) else prefix.rstrip("，；、 ") + "…")

    @staticmethod
    def _progressive_compact(value: str, limit: int) -> str:
        """更早内容保留极简开头和较新的详细尾部，实现随时间逐级压缩。"""
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        head = ConversationMemoryManager._compact_text(text[:700], 360)
        tail = text[-(limit - len(head) - 18):].lstrip()
        return f"早期摘要：{head}\n……\n较近摘要：{tail}"

    @staticmethod
    def _estimate_tokens(value: Any) -> int:
        """用中日韩字符和其他字符的比例估算模型 token，占用只用于触发压缩。"""
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":")) if not isinstance(value, str) else value
        cjk = len(re.findall(r"[\u3400-\u9fff]", text))
        other = len(re.sub(r"[\s\u3400-\u9fff]", "", text))
        return cjk + max(1, other // 4)

    @staticmethod
    def _new_state() -> dict[str, Any]:
        """返回新的分层记忆状态。"""
        return {
            "version": 2,
            "covered_message_id": 0,
            "recent_turns": [],
            "long_term_summary": "",
            "long_term_case_refs": [],
            "long_term_reference_types": [],
            "knowledge_version": "",
            "estimated_context_tokens": 0,
        }

    def _empty_packet(self, session_id: str) -> dict[str, Any]:
        """无会话时返回结构稳定的空记忆包。"""
        return self._packet(session_id, self._new_state())
