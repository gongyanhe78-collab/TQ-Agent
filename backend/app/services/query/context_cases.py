"""根据会话消息解析用户追问中引用的标准个例范围。"""
from __future__ import annotations

import re
from typing import Any

CONTEXT_REFERENCE_TERMS = (
    "上述",
    "上面",
    "上文",
    "前面",
    "这些",
    "那些",
    "这两个",
    "那两个",
    "这几个",
    "那几个",
    "这几次",
    "那几次",
    "它们",
    "他们",
    "继续分析",
    "接着分析",
    "进一步分析",
    "分别分析",
)


class ContextCaseResolver:
    """从最近一轮回答的证据和会话记忆中恢复可追溯的个例。"""

    def __init__(self, standard_case_store: Any) -> None:
        self.standard_case_store = standard_case_store

    def from_messages(self, messages: list[dict], question: str) -> list[str]:
        """返回与当前追问最匹配的个例 ID。"""
        if not self.question_uses_context_reference(question):
            return []
        candidates = []
        referenced_count = self.referenced_count_from_question(question)
        indexed_messages = list(enumerate(messages))
        for recency_index, (message_index, message) in enumerate(reversed(indexed_messages)):
            if message.get("role") != "assistant":
                continue
            content = str(message.get("content") or "")
            metadata = message.get("metadata", {}) or {}
            memory = metadata.get("memory", {}) if isinstance(metadata.get("memory", {}), dict) else {}
            previous_user_text = self._previous_user_text(messages, message_index)
            memory_text = f"{previous_user_text}\n{content}".strip()
            stored_context_ids = [
                item
                for item in (metadata.get("context_case_ids") or memory.get("context_case_ids") or [])
                if isinstance(item, str) and item
            ]
            if stored_context_ids:
                candidates.append(
                    self._candidate(
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
                    self._candidate(
                        case_ids,
                        question,
                        memory_text,
                        "evidence_cases",
                        recency_index,
                        referenced_count,
                    )
                )
            visual_case_ids = self.case_ids_from_visuals(
                metadata.get("visuals") or memory.get("visuals") or []
            )
            if visual_case_ids:
                candidates.append(
                    self._candidate(
                        visual_case_ids,
                        question,
                        memory_text,
                        "visuals",
                        recency_index,
                        referenced_count,
                    )
                )
            mentioned_case_ids = self.case_ids_mentioned_in_text(message.get("content", ""))
            if mentioned_case_ids:
                candidates.append(
                    self._candidate(
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
                best = max(
                    exact_candidates,
                    key=lambda candidate: (candidate["score"], -candidate["recency_index"]),
                )
                return best["ids"]
            return []
        best = max(
            valid_candidates,
            key=lambda candidate: (candidate["score"], -candidate["recency_index"]),
        )
        return self.limit_case_ids(best["ids"], question)

    @staticmethod
    def cases_scoped_by_context(cases: list[Any], context_case_ids: list[str] | None = None) -> list[Any]:
        """按明确个例范围过滤；范围无匹配时保留原集合以兼容旧行为。"""
        if not context_case_ids:
            return cases
        wanted = set(context_case_ids)
        scoped = [case for case in cases if case.case_id in wanted]
        return scoped or cases

    @staticmethod
    def question_uses_context_reference(question: str) -> bool:
        """判断问题是否依赖上一轮个例上下文。"""
        if any(term in question for term in CONTEXT_REFERENCE_TERMS):
            return True
        if re.search(r"([一二两三四五六七八九十]|\d+)(次|个).{0,6}(灾害|过程|个例)", question):
            return True
        return bool(
            re.search(
                r"(这|那|该).{0,3}([一二两三四五六七八九十\d]+|几)(次|个).{0,6}(灾害|过程|个例)",
                question,
            )
        )

    @staticmethod
    def referenced_count_from_question(question: str) -> int | None:
        """提取问题中明确引用的个例数量。"""
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

    def case_ids_from_visuals(self, visuals: list[dict]) -> list[str]:
        """从表格可视化行中提取个例 ID。"""
        case_ids = []
        for visual in visuals or []:
            if not isinstance(visual, dict) or visual.get("type") != "table":
                continue
            for row in visual.get("rows", []):
                case_id = self._case_id_from_visual_row(row)
                if case_id and case_id not in case_ids:
                    case_ids.append(case_id)
        return case_ids

    def case_ids_mentioned_in_text(self, text: str) -> list[str]:
        """识别回答正文中明确出现的标准个例。"""
        if not text:
            return []
        case_ids = []
        normalized_text = self._normalize_case_reference(text)
        for case in self.standard_case_store.list_cases():
            normalized_title = self._normalize_case_reference(case.title)
            if case.case_id in text or case.title in text or (
                normalized_title and normalized_title in normalized_text
            ):
                case_ids.append(case.case_id)
                continue
            if case.date_range and case.date_range in text:
                disaster_hit = any(disaster in text for disaster in case.disaster_types)
                title_words_hit = any(
                    part and part in text
                    for part in re.split(r"[~\-—\s]+", case.title)
                    if len(part) >= 4
                )
                if disaster_hit or title_words_hit:
                    case_ids.append(case.case_id)
        return self.limit_case_ids(case_ids, "")

    @staticmethod
    def limit_case_ids(case_ids: list[str], question: str) -> list[str]:
        """按问题声明的数量限制候选个例。"""
        deduped = []
        for case_id in case_ids:
            if case_id not in deduped:
                deduped.append(case_id)
        referenced_count = ContextCaseResolver.referenced_count_from_question(question)
        if referenced_count and len(deduped) > referenced_count:
            return deduped[:referenced_count]
        return deduped

    def _candidate(
        self,
        case_ids: list[str],
        question: str,
        memory_text: str,
        source: str,
        recency_index: int,
        referenced_count: int | None,
    ) -> dict[str, Any]:
        deduped = self._refine_case_ids(self._dedupe_case_ids(case_ids), memory_text, referenced_count)
        return {
            "ids": deduped,
            "score": self._score_candidate(
                deduped,
                question,
                memory_text,
                source,
                referenced_count,
            ),
            "recency_index": recency_index,
            "source": source,
        }

    def _refine_case_ids(
        self,
        case_ids: list[str],
        memory_text: str,
        referenced_count: int | None,
    ) -> list[str]:
        if not case_ids:
            return []
        filtered = self._case_ids_matching_memory(case_ids, memory_text)
        if referenced_count:
            if len(filtered) >= referenced_count:
                return filtered[:referenced_count]
            if len(case_ids) == referenced_count:
                return case_ids
            return []
        return filtered or case_ids

    def _case_ids_matching_memory(self, case_ids: list[str], memory_text: str) -> list[str]:
        months = self._month_tokens_from_text(memory_text)
        if not months:
            return []
        cases_by_id = {case.case_id: case for case in self.standard_case_store.list_cases()}
        matched = []
        for case_id in case_ids:
            case = cases_by_id.get(case_id)
            if case and any(
                token in " ".join([case.title, case.date_range, case.source_pdf])
                for token in months
            ):
                matched.append(case_id)
        return matched

    @staticmethod
    def _score_candidate(
        case_ids: list[str],
        question: str,
        content: str,
        source: str,
        referenced_count: int | None,
    ) -> int:
        if not case_ids:
            return -1000
        score = {
            "stored_context": 18,
            "visuals": 14,
            "evidence_cases": 8,
            "text": 4,
        }.get(source, 0)
        if referenced_count:
            if len(case_ids) == referenced_count:
                score += 100
            elif len(case_ids) > referenced_count:
                score -= min(60, (len(case_ids) - referenced_count) * 8)
            else:
                score -= 25
            if ContextCaseResolver._content_mentions_count(content, referenced_count):
                score += 22
            if ContextCaseResolver._content_mentions_full_library_stat(content, referenced_count):
                score -= 45
        month_tokens = ContextCaseResolver._month_tokens_from_text(question)
        if month_tokens and any(token in content for token in month_tokens):
            score += 12
        return score

    @staticmethod
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

    @staticmethod
    def _content_mentions_full_library_stat(content: str, referenced_count: int) -> bool:
        if not re.search(r"(全库|标准化个例库|共筛选到|按月份|按灾种|统计)", content):
            return False
        match = re.search(r"共筛选到\s*(\d+)\s*个", content)
        if match and int(match.group(1)) != referenced_count:
            return True
        return "按月份" in content or "按灾种" in content

    @staticmethod
    def _month_tokens_from_text(text: str) -> list[str]:
        tokens = []
        for match in re.finditer(r"(\d{1,2})\s*月", text):
            token = f"{int(match.group(1))}月"
            if token not in tokens:
                tokens.append(token)
        return tokens

    @staticmethod
    def _previous_user_text(messages: list[dict], before_index: int) -> str:
        for message in reversed(messages[:before_index]):
            if message.get("role") == "user":
                return str(message.get("content") or "")
        return ""

    @staticmethod
    def _dedupe_case_ids(case_ids: list[str]) -> list[str]:
        deduped = []
        for case_id in case_ids:
            if case_id and case_id not in deduped:
                deduped.append(case_id)
        return deduped

    def _case_id_from_visual_row(self, row: dict) -> str:
        if not isinstance(row, dict):
            return ""
        if row.get("case_id"):
            return row["case_id"]
        row_title = str(row.get("title") or row.get("process") or "").strip()
        row_date = str(row.get("date") or row.get("date_range") or "").strip()
        row_source = str(row.get("source") or row.get("source_pdf") or "").strip()
        normalized_title = self._normalize_case_reference(row_title)
        for case in self.standard_case_store.list_cases():
            if row_source and case.source_pdf != row_source:
                continue
            if row_title and (
                case.title == row_title
                or self._normalize_case_reference(case.title) == normalized_title
            ):
                return case.case_id
            if row_date and case.date_range == row_date:
                return case.case_id
        return ""

    @staticmethod
    def _normalize_case_reference(text: str) -> str:
        value = re.sub(r"\s+", "", text)
        value = value.replace("~", "-").replace("－", "-").replace("—", "-")
        for suffix in ("天气过程", "过程", "天气"):
            value = value.replace(suffix, "")
        return value
