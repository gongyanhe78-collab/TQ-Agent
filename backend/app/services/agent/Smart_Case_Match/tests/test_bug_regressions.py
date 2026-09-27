"""2026-08-18 问题清单的针对性回归测试。"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from ..workflow import progress
from ..infrastructure.data_store import LocalCaseDataStore
from ..matching.enrichment import _case_start_date
from ..matching.matching import _duration_from_text, _parse_range_end
from ..matching.natural_query import _normalize_dates, normalize_natural_extraction
from ..workflow.nodes import SmartCaseGraphNodes


class SmartCaseBugRegressionTests(unittest.TestCase):
    """确保边界修复只补齐错误，不改变正常业务链路。"""

    def test_relative_date_range_keeps_both_endpoints(self):
        """今天到后天不能因先命中今天而退化成单日。"""
        result = _normalize_dates("今天到后天有强对流", today=date(2026, 12, 31))
        self.assertEqual(result["start_date"], "2026-12-31")
        self.assertEqual(result["end_date"], "2027-01-02")

    def test_past_relative_date_range_keeps_both_endpoints(self):
        """昨天到今天必须从业务日期向前计算一天，不能只命中今天。"""
        result = _normalize_dates("昨天到今天出现持续性暴雨", today=date(2026, 8, 18))
        self.assertEqual(result["start_date"], "2026-08-17")
        self.assertEqual(result["end_date"], "2026-08-18")
        self.assertEqual(result["date"], "昨天到今天")

    def test_all_supported_relative_offsets_use_rule_dates(self):
        """前天、昨天、今天、明天和后天均由规则按统一业务日期换算。"""
        self.assertEqual(
            _normalize_dates("前天到后天有降水", today=date(2026, 8, 18)),
            {"start_date": "2026-08-16", "end_date": "2026-08-20", "date": "前天到后天"},
        )
        self.assertEqual(
            _normalize_dates("昨日有大风", today=date(2026, 8, 18))["start_date"],
            "2026-08-17",
        )

    def test_rule_date_wins_and_model_conflict_is_audited(self):
        """模型给出错误精确日期时，规则结果覆盖模型并记录冲突。"""
        audit = {}
        with patch("backend.app.services.agent.Smart_Case_Match.matching.natural_query._business_today", return_value=date(2026, 8, 18)):
            result = normalize_natural_extraction(
                "昨天到今天太原出现暴雨",
                {
                    "start_date": "2025-08-17",
                    "end_date": "2025-08-18",
                    "date_expression": "昨天到今天",
                    "date_evidence": "昨天到今天太原出现暴雨",
                },
                audit,
            )
        self.assertEqual(result["start_date"], "2026-08-17")
        self.assertEqual(result["end_date"], "2026-08-18")
        self.assertEqual(audit["source"], "rule")
        self.assertEqual(audit["business_timezone"], "Asia/Shanghai")
        self.assertTrue(audit["conflict"])

    def test_fuzzy_date_never_accepts_model_generated_exact_dates(self):
        """前段时间等模糊表达只能保留原文，不能采用模型猜测的精确日期。"""
        audit = {}
        with patch("backend.app.services.agent.Smart_Case_Match.matching.natural_query._business_today", return_value=date(2026, 8, 18)):
            result = normalize_natural_extraction(
                "前段时间太原出现强降水",
                {
                    "start_date": "2026-08-01",
                    "end_date": "2026-08-05",
                    "date_expression": "前段时间",
                    "date_evidence": "前段时间太原出现强降水",
                },
                audit,
            )
        self.assertEqual(result["start_date"], "")
        self.assertEqual(result["end_date"], "")
        self.assertEqual(result["date"], "前段时间")
        self.assertTrue(audit["conflict"])

    def test_conversation_page_displays_full_date_range(self):
        """自然语言解析结果应同时展示起止日期，避免范围被视觉上误认为单日。"""
        page = (Path(__file__).resolve().parents[1] / "pages" / "conversation_page.html").read_text(encoding="utf-8")
        self.assertIn("`${query.start_date} 至 ${query.end_date}`", page)

    def test_cross_year_range_end_is_next_year(self):
        """12月29日至1月3日的结束日期应落在下一年。"""
        start = date(2026, 12, 29)
        self.assertEqual(_parse_range_end("12月29日至1月3日", start), date(2027, 1, 3))

    def test_cross_month_duration_is_not_one_day(self):
        """7月31日至8月2日应计算为3天，跨年范围也应保留完整时长。"""
        self.assertEqual(_duration_from_text("2025年7月31日-8月2日"), 3)
        self.assertEqual(_duration_from_text("2025年12月29日-1月3日"), 6)
        self.assertEqual(_duration_from_text("2025年12月29日-2026年1月3日"), 6)

    def test_cross_year_image_date_is_kept(self):
        """跨年个例中的1月图片不能被12月29日-1月3日范围判断误删。"""
        case = {"date_range": "2025年12月29日-2026年1月3日", "title": "跨年过程"}
        self.assertTrue(SmartCaseGraphNodes._image_date_matches_case("图1 2026年1月1日形势", case))
        self.assertFalse(SmartCaseGraphNodes._image_date_matches_case("图2 2026年2月1日形势", case))

    def test_case_start_date_does_not_use_end_day_after_to(self):
        """只有“持续到5日”时不能把结束日误当成过程开始日。"""
        self.assertIsNone(_case_start_date({"date_range": "2025年3月持续到5日"}))
        self.assertEqual(_case_start_date({"date_range": "2025年3月3-5日"}), date(2025, 3, 3))

    def test_audit_keeps_per_node_details(self):
        """同名审计计数即使平铺覆盖，node_details 仍保留每个节点的原始值。"""
        nodes = SmartCaseGraphNodes.__new__(SmartCaseGraphNodes)
        first = nodes._audit({"audit": {}}, "first", 0.0, candidate_count=15)
        second = nodes._audit({"audit": first}, "second", 0.0, candidate_count=3)
        self.assertEqual(second["node_details"]["first"]["candidate_count"], 15)
        self.assertEqual(second["node_details"]["second"]["candidate_count"], 3)

    def test_progress_terminal_state_is_expired(self):
        """已完成进度会按 TTL 清理，运行中的进度不受清理影响。"""
        old_id = "regression-progress-old"
        new_id = "regression-progress-new"
        with patch.object(progress, "_TERMINAL_STATE_TTL_SECONDS", 0):
            progress.start(old_id)
            progress.complete(old_id)
            progress.start(new_id)
            self.assertIsNone(progress.get(old_id))
            self.assertIsNotNone(progress.get(new_id))
            progress.complete(new_id)

    def test_client_progress_id_is_only_a_query_alias(self):
        """重复客户端进度号不能覆盖两个服务端运行状态。"""
        alias = "shared-client-progress"
        first_run = "server-run-first"
        second_run = "server-run-second"
        progress.start(first_run, alias=alias)
        progress.update(first_run, "first", "第一个任务", 30)
        progress.start(second_run, alias=alias)
        progress.update(second_run, "second", "第二个任务", 40)

        self.assertEqual(progress.get(first_run)["message"], "第一个任务")
        self.assertEqual(progress.get(second_run)["message"], "第二个任务")
        self.assertEqual(progress.get(alias)["progress_id"], second_run)
        progress.complete(first_run)
        progress.complete(second_run)

    def test_data_store_reports_invalid_top_level_without_silent_ok(self):
        """错误的数据顶层结构不再静默伪装成正常本地资料。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            # Smart 已统一读取主库 standard_cases1.json，回归夹具必须覆盖当前真实数据源。
            (root / "standard_cases1.json").write_text(json.dumps({"case": 1}), encoding="utf-8")
            (root / "document_index").mkdir()
            (root / "document_index" / "chunks.json").write_text("[]", encoding="utf-8")
            (root / "image_metadata.json").write_text("[]", encoding="utf-8")
            health = LocalCaseDataStore(root).health()
        self.assertEqual(health["status"], "degraded")
        self.assertTrue(health["data_load_errors"])


if __name__ == "__main__":
    unittest.main()
