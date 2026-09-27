"""
最终审查模块
对工具返回结果进行最终一致性判定，检查月份约束等边界条件，
在必要时对问题归一化并重新执行，支持确定性规则和 LLM 辅助审查。
"""
from __future__ import annotations

import re
from typing import Any


# 中文数字月份映射表
CHINESE_MONTHS = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
}


class AgentFinalReviewer:
    """
    智能体最终审查器
    对工具链路返回结果进行一致性判定，检查约束条件（如月份范围），
    在必要时执行问题归一化并重试，确保答案质量。
    """

    def __init__(self, llm_client: Any | None = None):
        self.llm_client = llm_client

    def review(self, question: str, result: dict, conversation_history: list[dict] | None = None) -> dict:
        """
        执行最终审查

        Args:
            question: 用户原始问题
            result: 工具链路返回的结果字典
            conversation_history: 对话历史列表

        Returns:
            审查结果字典，包含状态、是否重试、归一化问题和最终答案
        """
        deterministic = self._deterministic_review(question, result)
        if deterministic["should_retry"]:
            return deterministic
        llm_review = self._llm_review(question, result, conversation_history or [])
        if llm_review:
            if llm_review.get("_review_error"):
                return {
                    **deterministic,
                    "reason": f"{deterministic['reason']}；大模型最终审查降级：{llm_review['_review_error']}",
                }
            return {
                "agent": "final_reviewer",
                "status": "ok" if llm_review.get("is_aligned", True) else "misaligned",
                "reason": str(llm_review.get("reason") or "大模型已完成最终汇总判定"),
                "should_retry": bool(llm_review.get("should_retry", False)),
                "normalized_question": str(llm_review.get("normalized_question") or question),
                "final_answer": str(llm_review.get("final_answer") or result.get("answer") or ""),
                "llm_used": True,
            }
        return deterministic

    def _deterministic_review(self, question: str, result: dict) -> dict:
        requested_months = _month_constraints(question)
        evidence_cases = result.get("evidence_cases") or []
        if requested_months and evidence_cases:
            requested = set(requested_months)
            outside_cases = [
                case
                for case in evidence_cases
                if _case_months(case) and not (set(_case_months(case)) & requested)
            ]
            if outside_cases:
                normalized = normalize_question_constraints(question)
                return {
                    "agent": "final_reviewer",
                    "status": "retry",
                    "reason": "工具结果包含不符合问题月份约束的个例，回退重新进行意图识别",
                    "should_retry": normalized != question,
                    "normalized_question": normalized,
                    "final_answer": "",
                    "llm_used": False,
                }
        return {
            "agent": "final_reviewer",
            "status": "ok",
            "reason": "工具结果与问题的显式约束一致",
            "should_retry": False,
            "normalized_question": question,
            "final_answer": "",
            "llm_used": False,
        }

    def _llm_review(self, question: str, result: dict, conversation_history: list[dict]) -> dict | None:
        if not conversation_history:
            return None
        if self.llm_client is None or not hasattr(self.llm_client, "review_answer_alignment"):
            return None
        if hasattr(self.llm_client, "is_available") and not self.llm_client.is_available():
            return None
        try:
            return self.llm_client.review_answer_alignment(
                question=question,
                result=result,
                conversation_history=_sanitize_history(conversation_history),
            )
        except Exception as exc:
            return {"_review_error": str(exc)}


def normalize_question_constraints(question: str) -> str:
    normalized = question
    for text, month in sorted(CHINESE_MONTHS.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = re.sub(fr"(?<!\d){text}月", f"{month}月", normalized)
    return normalized


def _month_constraints(text: str) -> list[int]:
    normalized = normalize_question_constraints(text)
    months = set()
    for month in re.findall(r"(\d{1,2})\s*月", normalized):
        value = int(month)
        if 1 <= value <= 12:
            months.add(value)
    return sorted(months)


def _case_months(case: dict) -> list[int]:
    text = " ".join(
        str(case.get(key) or "")
        for key in ("title", "date_range", "source_pdf")
    )
    return _month_constraints(text)


def _sanitize_history(messages: list[dict], limit: int = 10) -> list[dict]:
    cleaned = []
    for message in messages[-limit:]:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned
