"""单次匹配 JSON 审计日志的完整性和安全性测试。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ..infrastructure.audit_log import MatchAuditLogWriter


class MatchAuditLogTests(unittest.TestCase):
    """验证审计日志可供人工复核，同时保持旁路失败隔离。"""

    @staticmethod
    def _state(run_id: str = "audit-run") -> dict:
        """构造十五个候选的完整状态，覆盖最终选择和淘汰原因。"""
        candidates = []
        scored = []
        for index in range(15):
            case_id = f"case-{index:02d}"
            candidate = {
                "case_id": case_id,
                "title": f"历史过程{index}",
                "structured_score": round(0.90 - index * 0.03, 4),
                "semantic_score": round(0.80 - index * 0.02, 4),
                "disaster_gate_passed": index < 12,
                "score_breakdown": {
                    "disaster": 0.9,
                    "disaster_coverage": 0.75,
                    "area": 0.8,
                    "temporal": 0.7,
                },
                "semantic_chunk_ids": [f"chunk-{index:02d}"],
            }
            candidates.append(candidate)
            scored.append({
                **candidate,
                "retrieval_score": round(0.88 - index * 0.025, 4),
                "dimension_scores": {"hazard_match": 0.9, "intensity": 0.8},
                "metric_scores": {"最大阵风": 0.78},
                "missing_metrics": ["风力演变"],
                "dimension_coverage": 0.72,
                "dimension_assessment_source": "llm",
            })
        return {
            "run_id": run_id,
            "request": {"raw_query": "复合天气过程"},
            "query": {
                "disaster_types": ["沙尘", "大风"],
                "disaster_importance": {
                    "沙尘": {"weight": 0.65, "role": "primary"},
                    "大风": {"weight": 0.35, "role": "secondary"},
                },
                "query_text": "内部重复文本",
            },
            "candidate_cases": candidates,
            "scored_cases": scored,
            "selected_cases": scored[:3],
            "candidate_rank_audit": {
                "llm_status": "called",
                "llm_candidate_ids": [item["case_id"] for item in candidates],
                "llm_ordered_case_ids": [item["case_id"] for item in scored],
                "fused_scores_before_llm": {
                    item["case_id"]: item["retrieval_score"] for item in scored
                },
            },
            "warnings": [],
            "audit": {"input_mode": "natural_language_stream", "timings_ms": {"rank_and_select": 12.3}},
        }

    def test_write_records_fifteen_candidates_and_rejection_reasons(self):
        """日志必须保留十五个候选、最终选择和未入选原因，支持逐例人工复核。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            target = MatchAuditLogWriter(Path(temp_dir)).write(self._state(), "completed")
            self.assertIsNotNone(target)
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["candidate_count"], 15)
            self.assertEqual(payload["selected_case_ids"], ["case-00", "case-01", "case-02"])
            self.assertNotIn("query_text", payload["normalized_query"])
            rejected = next(item for item in payload["candidates"] if item["case_id"] == "case-14")
            self.assertEqual(rejected["decision"], "rejected")
            self.assertTrue(rejected["rejection_reasons"])
            self.assertEqual(rejected["metric_scores"]["最大阵风"], 0.78)

    def test_run_id_cannot_escape_dated_directory(self):
        """外部 progress_id 即使带路径字符，也只能生成日期目录内的安全文件名。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            target = MatchAuditLogWriter(base_dir).write(self._state("../../outside\\audit"), "completed")
            self.assertIsNotNone(target)
            self.assertTrue(str(target.resolve()).startswith(str(base_dir.resolve())))
            self.assertEqual(target.parent.name, __import__("datetime").date.today().isoformat())

    def test_sanitized_run_ids_do_not_overwrite_each_other(self):
        """不同运行号即使清理后的主体相同，也必须生成不同审计文件。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = MatchAuditLogWriter(Path(temp_dir))
            first = writer.write(self._state("ab/c"), "completed")
            second = writer.write(self._state("ab_c"), "completed")

            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertNotEqual(first.name, second.name)
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())

    def test_write_failure_is_isolated(self):
        """日志目录不可写等异常必须返回空结果，不能向匹配主链路抛出异常。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_base = Path(temp_dir) / "occupied"
            invalid_base.write_text("not-a-directory", encoding="utf-8")
            result = MatchAuditLogWriter(invalid_base).write(self._state(), "completed")
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
