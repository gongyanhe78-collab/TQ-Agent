"""为每次相似个例匹配生成可人工复核的独立 JSON 审计文件。"""
from __future__ import annotations

import json
import hashlib
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


logger = logging.getLogger("uvicorn.error")
# 新增客户端关联号字段，版本递增便于后续审计工具区分结构。
AUDIT_SCHEMA_VERSION = "1.1"
MIN_STRUCTURED_SCORE = 0.30


class MatchAuditLogWriter:
    """原子写入单次请求审计；写入失败只记后台日志，不影响业务结果。"""

    def __init__(self, base_dir: Path | None = None) -> None:
        # 审计日志与 data、pages 同级保存在 Agent 根目录，不跟随基础设施子包迁移。
        self.base_dir = Path(base_dir or Path(__file__).resolve().parents[1] / "logs")

    def write(self, state: dict[str, Any], status: str, error: str = "") -> Path | None:
        """按日期和 run_id 保存审计快照，返回文件路径；任何异常均在本层隔离。"""
        try:
            now = datetime.now().astimezone()
            run_id = self._safe_run_id(state.get("run_id"))
            day_dir = self.base_dir / now.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            target = day_dir / f"{run_id}.json"
            temporary = day_dir / f".{run_id}.{uuid4().hex}.tmp"
            payload = self._build_payload(state, status, error, now.isoformat())
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            os.replace(temporary, target)
            logger.info("[SmartCaseMatch][Audit] 已写入匹配审计 run_id=%s path=%s", run_id, target)
            return target
        except Exception as exc:
            # 审计属于旁路能力，磁盘只读、空间不足等情况不能让主匹配链路失败。
            logger.exception("[SmartCaseMatch][Audit] 审计写入失败 run_id=%s error=%s", state.get("run_id"), exc)
            return None

    def _build_payload(
        self,
        state: dict[str, Any],
        status: str,
        error: str,
        created_at: str,
    ) -> dict[str, Any]:
        candidate_cases = list(state.get("candidate_cases") or [])
        scored_cases = list(state.get("scored_cases") or [])
        selected_cases = list(state.get("selected_cases") or [])
        rank_audit = dict(state.get("candidate_rank_audit") or {})
        scored_map = {str(item.get("case_id") or ""): item for item in scored_cases}
        post_rank = {str(item.get("case_id") or ""): index for index, item in enumerate(scored_cases, start=1)}
        selected_ids = {str(item.get("case_id") or "") for item in selected_cases}
        llm_ids = set(rank_audit.get("llm_candidate_ids") or [])
        llm_order = {
            str(case_id): index
            for index, case_id in enumerate(rank_audit.get("llm_ordered_case_ids") or [], start=1)
        }
        assessments = rank_audit.get("assessments") or {}
        candidates = []
        for structured_rank, candidate in enumerate(candidate_cases, start=1):
            case_id = str(candidate.get("case_id") or "")
            scored = scored_map.get(case_id, candidate)
            structured_score = float(candidate.get("structured_score") or 0.0)
            rejection_reasons = []
            if not candidate.get("disaster_gate_passed", True):
                rejection_reasons.append("灾种标题准入未通过")
            if structured_score < MIN_STRUCTURED_SCORE:
                rejection_reasons.append(f"结构化分低于{MIN_STRUCTURED_SCORE:.2f}")
            if case_id not in llm_ids:
                rejection_reasons.append("未进入候选LLM前15名")
            if scored.get("dimension_coverage") is not None and float(scored.get("dimension_coverage") or 0.0) < 0.6:
                rejection_reasons.append("动态维度证据覆盖不足60%")
            decision = "selected" if case_id in selected_ids else "rejected"
            if decision == "rejected" and not rejection_reasons:
                rejection_reasons.append("最终排序或多样性选择未进入目标数量")
            assessment = dict(assessments.get(case_id) or {})
            candidates.append({
                "case_id": case_id,
                "title": str(candidate.get("title") or ""),
                "date_range": str(candidate.get("date_range") or ""),
                "source_pdf": str(candidate.get("source_pdf") or ""),
                "disaster_types": list(candidate.get("disaster_types") or []),
                "affected_areas": list(candidate.get("affected_areas") or []),
                "structured_rank": structured_rank,
                "post_llm_rank": post_rank.get(case_id),
                "llm_order_rank": llm_order.get(case_id),
                "entered_llm_top15": case_id in llm_ids,
                "structured_score": structured_score,
                "semantic_score": float(candidate.get("semantic_score") or 0.0),
                "semantic_rerank_score": candidate.get("semantic_rerank_score"),
                "fused_score_before_llm": (rank_audit.get("fused_scores_before_llm") or {}).get(case_id),
                "final_retrieval_score": float(scored.get("retrieval_score") or 0.0),
                "score_breakdown": dict(scored.get("score_breakdown") or candidate.get("score_breakdown") or {}),
                "disaster_gate_passed": bool(candidate.get("disaster_gate_passed", True)),
                "dimension_scores": dict(scored.get("dimension_scores") or assessment.get("dimension_scores") or {}),
                "metric_scores": dict(scored.get("metric_scores") or assessment.get("metric_scores") or {}),
                "missing_dimensions": list(scored.get("missing_dimensions") or assessment.get("missing_dimensions") or []),
                "missing_metrics": list(scored.get("missing_metrics") or assessment.get("missing_metrics") or []),
                "dimension_raw_compatibility": scored.get("dimension_raw_compatibility"),
                "dimension_compatibility": scored.get("dimension_compatibility"),
                "dimension_coverage": scored.get("dimension_coverage"),
                "assessment_source": str(scored.get("dimension_assessment_source") or assessment.get("assessment_source") or ""),
                "semantic_chunk_ids": list(candidate.get("semantic_chunk_ids") or []),
                "semantic_evidence_previews": [
                    {
                        "chunk_id": str(item.get("chunk_id") or ""),
                        "role": str(item.get("evidence_role") or ""),
                        "preview": re.sub(r"\s+", " ", str(item.get("content") or ""))[:180],
                    }
                    for item in scored.get("semantic_evidence") or []
                ],
                "decision": decision,
                "rejection_reasons": rejection_reasons,
            })
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "run_id": str(state.get("run_id") or ""),
            "client_progress_id": str(state.get("client_progress_id") or ""),
            "created_at": created_at,
            "status": status,
            "error": str(error or ""),
            "input_mode": str((state.get("audit") or {}).get("input_mode") or "structured"),
            "request": dict(state.get("request") or {}),
            "normalized_query": self._clean_query(state.get("query") or {}),
            "pipeline_audit": dict(state.get("audit") or {}),
            "candidate_llm": {
                "status": rank_audit.get("llm_status"),
                "candidate_ids": list(rank_audit.get("llm_candidate_ids") or []),
                "ordered_case_ids": list(rank_audit.get("llm_ordered_case_ids") or []),
            },
            # 单独记录逐例提炼来源，人工复核时可以确认综合研判是否使用过规则降级内容。
            "reference_generation": [
                {
                    "case_id": str(item.get("case_id") or ""),
                    "source": str(item.get("reference_source") or ""),
                    "status": str(item.get("reference_status") or ""),
                    "reference_point_count": len(item.get("reference_points") or []),
                }
                for item in state.get("case_references") or []
            ],
            "candidate_count": len(candidates),
            "candidates": candidates,
            "selected_case_ids": [str(item.get("case_id") or "") for item in selected_cases],
            "warnings": list(state.get("warnings") or []),
        }

    @staticmethod
    def _clean_query(query: dict[str, Any]) -> dict[str, Any]:
        """保留人工复核所需查询画像，删除重复拼接的内部 query_text。"""
        return {key: value for key, value in query.items() if key != "query_text"}

    @staticmethod
    def _safe_run_id(value: Any) -> str:
        """限制文件名字符，并用原值摘要区分清理后同名的不同运行号。"""
        source = str(value or "match")
        cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", source) or "match"
        if cleaned != source or len(cleaned) > 120 or cleaned in {".", ".."}:
            # 哈希只用于消除 ab/c 与 ab_c 之类的映射碰撞，不在文件名中暴露原始非法字符。
            digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:10]
            cleaned = f"{cleaned[:109]}-{digest}"
        return cleaned[:120]
