"""统一聊天结果的会话持久化与证据元数据整理。"""
from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from backend.app.agents.unified_chat.performance import PerformanceTrace


LOGGER = logging.getLogger("uvicorn.error")


class ResultPersistenceMixin:
    """把统一结果写入 Agent 运行表、消息元数据和证据字段。"""

    def _start_persisted_run(
        self,
        run_id: str,
        session_id: str,
        agent_type: str,
        question: str,
        *,
        status: str = "running",
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        """创建运行记录，并登记当前请求正在持久化的运行编号。"""
        enriched = self._merge_runtime_metadata(metadata)
        run = self.session_store.start_agent_run(
            run_id,
            session_id,
            agent_type,
            question,
            status=status,
            metadata=enriched,
        )
        self._active_run_id = run_id
        self._active_run_session_id = session_id
        return run

    def _finish_persisted_run(
        self,
        run_id: str,
        *,
        status: str,
        answer: str = "",
        evidence_cases: list[dict] | None = None,
        evidence_chunks: list[dict] | None = None,
        images: list[dict] | None = None,
        reports: list[dict] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        """完成运行记录，并在成功后清除当前运行编号。"""
        enriched = self._merge_runtime_metadata(metadata)
        run = self.session_store.finish_agent_run(
            run_id,
            status=status,
            answer=answer,
            evidence_cases=evidence_cases,
            evidence_chunks=evidence_chunks,
            images=images,
            reports=reports,
            metadata=enriched,
        )
        if getattr(self, "_active_run_id", "") == run_id:
            self._active_run_id = ""
            self._active_run_session_id = ""
        return run

    def _finish_active_run(
        self,
        *,
        status: str,
        error: str = "",
    ) -> dict:
        """终止尚未完成的当前运行，避免异常后留下 running 记录。"""
        metadata = (
            self._runtime_failure_metadata(error)
            if error
            else self._merge_runtime_metadata()
        )
        run_id = str(getattr(self, "_active_run_id", "") or "")
        if not run_id:
            return {}
        return self._finish_persisted_run(
            run_id,
            status=status,
            metadata=metadata,
        )

    def _persist_complete(
        self,
        session_id: str,
        common: dict,
        *,
        status: str,
        already_started: bool = False,
        performance: PerformanceTrace | None = None,
    ) -> dict:
        """把统一结果同时写入 Agent 运行表和消息元数据。"""
        # 完成快照必须先收口当前路由子任务，再写入运行表。
        self._finalize_runtime_status(status)
        if performance:
            performance.start("persist_ms")
            common["performance"] = performance.snapshot()
        run_id = str(common.get("run_id") or f"agent_run_{uuid4().hex}")
        if not already_started:
            self._start_persisted_run(
                run_id,
                session_id,
                str(common.get("agent_type") or "unknown"),
                str(common.get("question") or ""),
                status=status,
                metadata=self._runtime_metadata(),
            )
        common.update(self._regeneration_context)
        if not common.get("source_user_message_id"):
            # 普通回答以本轮用户消息作为稳定版本组，重生成只追加版本。
            source_user = next(
                (
                    item
                    for item in reversed(
                        self.session_store.list_messages(session_id)
                    )
                    if item.get("role") == "user"
                ),
                None,
            )
            source_user_message_id = int(
                (source_user or {}).get("message_id") or 0
            )
            common["source_user_message_id"] = source_user_message_id or None
            common["version_group_id"] = (
                f"user-{source_user_message_id}"
                if source_user_message_id
                else ""
            )
        metadata = self._merge_runtime_metadata(
            self._message_metadata(common, status)
        )
        self._finish_persisted_run(
            run_id,
            status=status,
            answer=str(common.get("answer") or ""),
            evidence_cases=list(common.get("evidence_cases") or []),
            evidence_chunks=list(common.get("evidence_chunks") or []),
            images=list(common.get("images") or []),
            reports=list(common.get("reports") or []),
            metadata=metadata,
        )
        saved_message = self.session_store.add_message(
            session_id,
            "assistant",
            str(common.get("answer") or ""),
            metadata=metadata,
        )
        common["message_id"] = int(
            saved_message.get("message_id") or 0
        ) or None
        if performance:
            performance.stop("persist_ms")
            common["performance"] = performance.snapshot()
            LOGGER.info(
                "[统一聊天性能] run_id=%s timings=%s",
                run_id,
                common["performance"],
            )
        return saved_message

    def _message_metadata(self, common: dict, status: str) -> dict:
        """构建会话恢复需要的顶层字段和统一记忆包。"""
        fields = {
            key: common.get(key, default)
            for key, default in (
                ("agent_type", ""),
                ("run_id", ""),
                # 待确认提案编号必须落库，跨请求恢复时才能精确确认。
                ("proposal_id", ""),
                ("proposal_operation", ""),
                ("proposal_question", ""),
                ("retrieval_mode", ""),
                ("evidence_cases", []),
                ("evidence_chunks", []),
                ("images", []),
                ("image_display_mode", "hidden"),
                ("visuals", []),
                ("reports", []),
                ("analysis", {}),
                ("query_conditions", {}),
                ("intent_trace", {}),
                ("execution_plan", {}),
                ("task_plan", []),
                ("audit", {}),
                ("agent_result", {}),
                ("original_question", ""),
                ("resolved_question", ""),
                ("context_case_ids", []),
                ("memory_trace", {}),
                ("report_mode", "none"),
                ("source_message_id", None),
                ("source_run_id", ""),
                ("llm_used", False),
                ("llm_status", ""),
                ("answer_renderer", ""),
                ("performance", {}),
                ("request_id", ""),
                ("trace_id", ""),
                ("termination_reason", ""),
                ("steps", 0),
                ("model_calls", 0),
                ("subtask_status", []),
                ("budget", {}),
                ("token_stats", {}),
                ("token_usage_source", ""),
                ("follow_up_suggestions", []),
                ("follow_up_status", "none"),
                ("regenerated_from_message_id", None),
                ("regeneration_instruction", ""),
                ("source_user_message_id", None),
                ("version_group_id", ""),
                ("version", 1),
            )
        }
        fields["status"] = status
        fields["memory"] = {
            "version": 2,
            "question": common.get("question", ""),
            "answer": common.get("answer", ""),
            **fields,
        }
        return fields

    @staticmethod
    def _attach_memory_trace(
        common: dict[str, Any],
        original_question: str,
        resolved_question: str,
        context_case_ids: list[str],
        route_trace: dict[str, Any] | None,
    ) -> None:
        """把问题改写和历史引用链写入统一元数据。"""
        common["original_question"] = original_question
        common["resolved_question"] = resolved_question
        common["context_case_ids"] = list(context_case_ids)
        common["memory_trace"] = dict(route_trace or {})

    def _context_evidence_cases(
        self,
        case_ids: list[str],
    ) -> list[dict[str, Any]]:
        """为指代检索补回标准个例实体，保证后续轮次仍能继续引用。"""
        store = self.dependencies.standard_case_store
        converter = self.dependencies.presenter.standard_case_to_response
        if not case_ids or store is None or converter is None:
            return []
        wanted = set(case_ids)
        return [
            converter(case)
            for case in store.list_cases()
            if case.case_id in wanted
        ]

    def _evidence_cases_from_document_hits(
        self,
        hits: list[dict[str, Any]],
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """把普通向量命中的 chunk 反向关联标准个例。"""
        store = self.dependencies.standard_case_store
        converter = self.dependencies.presenter.standard_case_to_response
        if store is None or converter is None:
            return []
        cases = store.list_cases()
        result = []
        seen = set()
        for hit in hits:
            chunk = hit.get("chunk") if isinstance(hit, dict) else {}
            chunk_id = str((chunk or {}).get("chunk_id") or "")
            source_pdf = str((chunk or {}).get("source_pdf") or "")
            for case in cases:
                if case.case_id in seen:
                    continue
                if (
                    chunk_id in case.source_chunk_ids
                    or (
                        not case.source_chunk_ids
                        and source_pdf == case.source_pdf
                    )
                ):
                    seen.add(case.case_id)
                    result.append(converter(case))
                    if len(result) >= limit:
                        return result
        return result

    @staticmethod
    def _images_from_hits(
        hits: list[dict],
        limit: int = 6,
    ) -> list[dict]:
        """从普通 RAG 命中中提取去重图片。"""
        images = []
        seen = set()
        for hit in hits:
            for image in hit.get("images") or []:
                image_id = str(image.get("image_id") or "")
                if not image_id or image_id in seen:
                    continue
                seen.add(image_id)
                images.append(image)
                if len(images) >= limit:
                    return images
        return images
