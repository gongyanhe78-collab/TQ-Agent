"""统一意图路由、历史实体校验和承接式报告判断。"""
from __future__ import annotations

import re
from typing import Any

from backend.app.agents.query_intent import IntentRouteRequest
from backend.app.agents.unified_chat.orchestration_models import (
    UnifiedRouteDecision,
)


class RoutingMixin:
    """把统一意图模型输出转换为内部路由决策。"""

    def _route(
        self,
        question: str,
        has_pending: bool,
        memory_context: dict[str, Any],
    ) -> UnifiedRouteDecision:
        """调用意图 Agent，并把待确认操作固定留在多维链路。"""
        if has_pending and (
            self._is_confirmation(question)
            or self._is_cancellation(question)
        ):
            return UnifiedRouteDecision(
                "multidim_search",
                "structured",
                question,
            )
        try:
            result = self.intent_agent.route(
                IntentRouteRequest(
                    message=question,
                    previous_intent=memory_context.get("previous_intent"),
                    memory_context=memory_context,
                )
            )
            # 待确认时，明确修改词或短条件表达继续交给条件解析器。
            modifies_pending = any(
                term in question
                for term in (
                    "改",
                    "换",
                    "调整",
                    "不要",
                    "去掉",
                    "不限",
                    "增加",
                    "补充",
                )
            )
            short_condition = (
                bool(result.conditions)
                and len(str(question).strip()) <= 30
            )
            if has_pending and (modifies_pending or short_condition):
                return UnifiedRouteDecision(
                    "multidim_search",
                    "structured",
                    question,
                )
            trace = {
                "audit_id": result.audit_id,
                "context_related": result.context_related,
                "relation_type": result.relation_type,
                "context_confidence": result.context_confidence,
                "context_resolution_source": (
                    result.context_resolution_source
                ),
                "context_message_ids": result.context_message_ids,
                "reference_type": result.reference_type,
                "reference_clues": result.reference_clues,
                "routing_source": result.routing_source,
                "reason": result.reason,
            }
            tasks = [
                item.model_dump()
                for item in getattr(result, "tasks", [])
            ]
            trace["task_plan"] = tasks
            referenced_case_ids = self._valid_knowledge_case_ids(
                result.referenced_case_ids
            )
            intent = str(result.intent)
            rag_strategy = str(result.rag_strategy or "vector")
            # 已锁定库内历史个例时绝不调用 Smart。
            if referenced_case_ids and intent == "similar_case_match":
                intent = "rag"
                rag_strategy = "vector"
                trace["orchestrator_guard"] = (
                    "knowledge_case_forced_to_rag"
                )
            trace["validated_knowledge_case_ids"] = referenced_case_ids
            return UnifiedRouteDecision(
                intent,
                rag_strategy,
                str(result.normalized_message or question),
                str(result.clarification_question or ""),
                referenced_case_ids,
                trace,
                tasks,
            )
        except Exception:
            # 意图 Agent 暂不可用时保留规则兜底。
            text = str(question or "")
            if "报告" in text and any(
                term in text for term in ("生成", "导出", "制作")
            ):
                return UnifiedRouteDecision(
                    "multidim_search",
                    "vector",
                    text,
                )
            if any(
                term in text
                for term in ("相似个例", "相似过程", "参考经验")
            ):
                return UnifiedRouteDecision(
                    "similar_case_match",
                    "vector",
                    text,
                )
            return UnifiedRouteDecision("rag", "vector", text)

    def _standard_cases(self) -> list[Any]:
        """向记忆和意图模块提供主项目当前标准个例。"""
        store = self.dependencies.standard_case_store
        return list(store.list_cases()) if store is not None else []

    def _valid_knowledge_case_ids(
        self,
        case_ids: list[str] | None,
    ) -> list[str]:
        """过滤和去重模型返回的非法个例 ID。"""
        valid_ids = {
            str(case.case_id)
            for case in self._standard_cases()
            if getattr(case, "case_id", None)
        }
        result: list[str] = []
        for case_id in case_ids or []:
            value = str(case_id)
            if value in valid_ids and value not in result:
                result.append(value)
        return result

    @staticmethod
    def _is_previous_result_report_request(
        question: str,
        memory_context: dict[str, Any],
    ) -> bool:
        """判断报告动作是否明确承接当前会话。"""
        text = re.sub(r"\s+", "", str(question or "")).lower()
        report_action = any(
            term in text for term in ("报告", "pdf", "导出", "整理")
        )
        context_cues = (
            "上面",
            "上一轮",
            "刚才",
            "前面",
            "上述",
            "这些",
            "这几个",
            "这8",
            "这八",
            "它们",
            "前述",
            "刚刚",
        )
        # 匹配完整动作结构，避免单独出现“这个”时误判。
        result_phrases = (
            "这个分析",
            "该分析",
            "这个结果",
            "该结果",
            "这个汇总",
            "该汇总",
            "这份分析",
            "这份结果",
            "这个回答",
            "这个答案",
            "以上分析",
            "以上结果",
        )
        structured_reuse = any(
            phrase in text for phrase in result_phrases
        ) or bool(
            re.search(
                r"(?:把|将).{0,16}"
                r"(?:分析|结果|汇总|答案|回答).{0,16}"
                r"(?:成|为|整理|导出|生成).{0,12}"
                r"(?:pdf|报告)",
                text,
            )
        )
        return report_action and (
            any(cue in text for cue in context_cues)
            or structured_reuse
        )

    @staticmethod
    def _is_report_action(question: str) -> bool:
        """判断历史用户消息是否曾要求生成报告。"""
        text = re.sub(r"\s+", "", str(question or "")).lower()
        return any(
            term in text
            for term in ("报告", "pdf", "导出", "整理成", "汇总成")
        )

    def _previous_reusable_result(
        self,
        session_id: str,
        question: str,
        memory_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """从原始消息恢复上一轮完整结果，不依赖压缩记忆正文。"""
        if not session_id:
            return None
        messages = self.session_store.list_messages(session_id)
        # 本轮用户消息尚未写入，倒序首条可复用回答就是上一轮结果。
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if str(message.get("role") or "") != "assistant":
                continue
            metadata = (
                message.get("metadata")
                if isinstance(message.get("metadata"), dict)
                else {}
            )
            merged = dict(metadata)
            for key, value in message.items():
                if key not in {
                    "metadata",
                    "content",
                    "role",
                    "message_id",
                    "session_id",
                    "created_at",
                }:
                    merged.setdefault(key, value)
            status = str(merged.get("status") or "")
            if status in {
                "awaiting_confirmation",
                "needs_clarification",
                "cancelled",
            }:
                continue
            # 要求 PDF 却没有报告的回答是失败回退，不能遮蔽更早结果。
            source_question = str(
                merged.get("original_question")
                or merged.get("question")
                or (merged.get("memory") or {}).get("question")
                or ""
            )
            if not source_question:
                for prior in reversed(messages[:index]):
                    if str(prior.get("role") or "") == "user":
                        source_question = str(
                            prior.get("content") or ""
                        )
                        break
            reports = list(merged.get("reports") or [])
            if (
                self._is_report_action(source_question)
                and not reports
            ):
                continue
            if (
                str(merged.get("agent_type") or "")
                == "conversation_report"
                and not reports
            ):
                continue
            cases = list(merged.get("evidence_cases") or [])
            agent_result = merged.get("agent_result")
            if not cases and isinstance(agent_result, dict):
                cases = list(
                    agent_result.get("evidence_cases")
                    or agent_result.get("matched_cases")
                    or []
                )
            answer = str(
                merged.get("answer") or message.get("content") or ""
            ).strip()
            if (
                not cases
                and not answer
                and not merged.get("analysis")
            ):
                continue
            return {
                "source_message_id": int(
                    message.get("message_id") or 0
                ),
                "answer": answer,
                "agent_type": str(merged.get("agent_type") or ""),
                "run_id": str(merged.get("run_id") or ""),
                "evidence_cases": cases,
                "evidence_chunks": list(
                    merged.get("evidence_chunks") or []
                ),
                "images": list(merged.get("images") or []),
                "analysis": merged.get("analysis") or {},
                "task_plan": list(merged.get("task_plan") or []),
                "agent_result": (
                    agent_result
                    if isinstance(agent_result, dict)
                    else {}
                ),
                "question": str(merged.get("question") or ""),
            }
        return None

    @staticmethod
    def _conversation_report_title(
        previous_result: dict[str, Any],
    ) -> str:
        """依据上一轮问题和回答生成简洁标题，不调用模型。"""
        result = (
            previous_result
            if isinstance(previous_result, dict)
            else {}
        )
        question = re.sub(
            r"\s+",
            "",
            str(result.get("question") or ""),
        )
        answer = str(result.get("answer") or "")
        years = list(
            dict.fromkeys(re.findall(r"(20\d{2})年", question))
        )
        months = [
            int(value)
            for value in dict.fromkeys(
                re.findall(r"(?<!\d)(\d{1,2})月", question)
            )
            if 1 <= int(value) <= 12
        ]
        if len(years) == 1 and len(months) >= 2:
            return (
                f"{years[0]}年{months[0]}月至{months[-1]}月"
                "气象灾害分析报告"
            )
        if len(years) == 1 and len(months) == 1:
            return f"{years[0]}年{months[0]}月气象灾害分析报告"
        if len(years) == 1:
            return f"{years[0]}年气象灾害分析报告"
        heading = re.search(
            r"^#{1,3}\s+(.+)$",
            answer,
            flags=re.MULTILINE,
        )
        if heading:
            title = re.sub(
                r"[*_`#]",
                "",
                heading.group(1),
            ).strip(" ：:。")
            if title and title not in {
                "核心结论",
                "统计结论",
                "分析结论",
                "主要过程",
            }:
                return (
                    title
                    if title.endswith("报告")
                    else f"{title}报告"
                )
        return "气象灾害分析报告"

    @staticmethod
    def _is_confirmation(question: str) -> bool:
        text = str(question or "").strip()
        return text in {
            "确认",
            "确定",
            "确认条件",
            "确定条件",
            "开始检索",
            "执行检索",
            "按此检索",
            "生成吧",
        } or text.startswith(("确认", "确定"))

    @staticmethod
    def _is_cancellation(question: str) -> bool:
        return str(question or "").strip() in {
            "取消",
            "取消检索",
            "放弃",
            "重新输入",
        }
