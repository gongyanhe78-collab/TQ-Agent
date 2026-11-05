"""通用指标聚合、地域层级和依赖结果传递测试。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.models import DocumentChunk, StandardCase
from backend.app.services.agent.case_multidim_search.retrieval.structured_retriever import StructuredCaseRetriever
from backend.app.services.agent.case_multidim_search.schemas import CaseSearchQuery
from backend.app.services.agent.unified_chat.metric_aggregation import aggregate_metric_facts


class MetricAggregationTests(unittest.TestCase):
    """验证同一执行器可以处理不同指标与聚合方向。"""

    def test_minimum_temperature_keeps_time_location_and_case(self) -> None:
        chunk = DocumentChunk(
            "山西省2025年1月天气过程总结.pdf",
            "chunk-016",
            16,
            "1月14日08时至15日08时，全省各地最低气温介于-25.6℃（新荣）～-2.7℃（永济）之间。",
        )
        result = aggregate_metric_facts(
            [chunk],
            metric="air_temperature.minimum",
            operator="min",
            chunk_case_ids={"chunk-016": ["case-jan"]},
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["winner"]["normalized_value"], -25.6)
        self.assertEqual(result["winner"]["location"], "新荣")
        self.assertIn("1月14日08时", result["winner"]["time"])
        self.assertEqual(result["selected_case_ids"], ["case-jan"])

    def test_other_metrics_reuse_the_same_aggregation_contract(self) -> None:
        chunks = [
            DocumentChunk(
                "m.pdf", "rain", 1,
                "2025 年 7 月 23-26 日，过程累计降水量最高达\n303mm，出现在大同市天镇李二口气象观测站。",
            ),
            DocumentChunk("m.pdf", "wind", 2, "6月11日14时，山区最大风速达到43.4m/s（五台山）。"),
            DocumentChunk("m.pdf", "visibility", 3, "3月28日07时，过程最低能见度为0.6km（太原）。"),
        ]
        cases = {item.chunk_id: [f"case-{item.chunk_id}"] for item in chunks}
        rain = aggregate_metric_facts(chunks, metric="precipitation.maximum", operator="max", chunk_case_ids=cases)
        wind = aggregate_metric_facts(chunks, metric="wind_speed.maximum", operator="max", chunk_case_ids=cases)
        visibility = aggregate_metric_facts(chunks, metric="visibility.minimum", operator="min", chunk_case_ids=cases)
        self.assertEqual(rain["winner"]["normalized_value"], 303)
        self.assertEqual(rain["winner"]["location"], "大同市天镇李二口气象观测站")
        self.assertEqual(rain["winner"]["time"], "2025年7月23-26日")
        self.assertEqual(rain["selected_case_ids"], ["case-rain"])
        self.assertEqual(wind["winner"]["normalized_value"], 43.4)
        self.assertEqual(wind["winner"]["location"], "五台山")
        self.assertEqual(wind["winner"]["time"], "6月11日14时")
        self.assertEqual(wind["selected_case_ids"], ["case-wind"])
        self.assertEqual(visibility["winner"]["normalized_value"], 600)
        self.assertEqual(visibility["winner"]["location"], "太原")
        self.assertEqual(visibility["winner"]["time"], "3月28日07时")
        self.assertEqual(visibility["selected_case_ids"], ["case-visibility"])

    def test_province_query_matches_provincewide_case(self) -> None:
        case = StandardCase(
            "case-jan", "寒潮过程", "2025年1月14-15日",
            affected_areas=["全省"], year=2025, months=[1],
            start_date="2025-01-14", end_date="2025-01-15",
        )
        query = CaseSearchQuery(years=[2025], months=[1], areas=["山西"])
        self.assertEqual([item.case.case_id for item in StructuredCaseRetriever().search([case], query)], ["case-jan"])

    def test_iso_range_matches_legacy_case_date_range(self) -> None:
        """标准日期字段为空时，ISO 起止范围仍应匹配 date_range 中的年月。"""
        case = StandardCase(
            "case-jul", "7月23-26日北部暴雨过程", "2025年7月23-26日",
            source_pdf="FST2025-7.pdf",
        )
        query = CaseSearchQuery(start_date="2025-01-01", end_date="2025-07-31")
        matches = StructuredCaseRetriever().search([case], query)
        self.assertEqual([item.case.case_id for item in matches], ["case-jul"])

    def test_wind_extreme_ignores_upper_air_speed_core(self) -> None:
        """最大风速应采用地面观测实况，不把高空环流风速核当作站点极值。"""
        chunks = [
            DocumentChunk(
                "FST2025-4.pdf", "surface", 1,
                "区域站最大阵风风速达39.7 m/s（13级），出现在神池义井[B5511]，"
                "发生于2025年4月11日17:05；",
            ),
            DocumentChunk(
                "FST2025-4.pdf", "aloft", 2,
                "第一阶段（11日08时至12日08时），500hPa随着≥40m/s的偏北风大风速核南移；",
            ),
        ]
        result = aggregate_metric_facts(
            chunks,
            metric="wind_speed.maximum",
            operator="max",
            chunk_case_ids={"surface": ["case-surface"], "aloft": ["case-aloft"]},
        )
        self.assertEqual(result["winner"]["normalized_value"], 39.7)
        self.assertEqual(result["winner"]["location"], "神池义井")
        self.assertEqual(result["winner"]["time"], "2025年4月11日17:05")
        self.assertEqual(result["selected_case_ids"], ["case-surface"])

    def test_visibility_sequence_binds_each_value_to_nearest_day(self) -> None:
        """逐日能见度序列应把最小值绑定到邻近日号，并从上下文补全年月。"""
        chunk = DocumentChunk(
            "FST2025-3.pdf", "visibility-sequence", 1,
            "沙尘过程最低能见度25日1.3km，26日0.6km，"
            "27日中部最小水平能见度2.86km，28日南部最小水平能见度为0.96km。",
        )
        result = aggregate_metric_facts(
            [chunk],
            metric="visibility.minimum",
            operator="min",
            chunk_case_ids={"visibility-sequence": ["case-dust"]},
        )
        self.assertEqual(result["winner"]["normalized_value"], 600)
        self.assertEqual(result["winner"]["time"], "2025年3月26日")
        self.assertEqual(result["winner"]["location"], "")
        self.assertEqual(result["selected_case_ids"], ["case-dust"])

    def test_surface_temperature_ignores_upper_air_cold_core_in_joined_text(self) -> None:
        """PDF 段落粘连时，500hPa 冷中心温度不能覆盖地面最低气温实况。"""
        chunks = [
            DocumentChunk(
                "jan.pdf", "surface-temp", 1,
                "1月14日08时至15日08时，全省最低气温介于-25.6℃（新荣）至-2.7℃（永济）之间。",
            ),
            DocumentChunk(
                "jan.pdf", "joined-aloft", 2,
                "图7最低气温实况图2.环流形势演变特征，1月13日20时，500hPa"
                "贝加尔湖以东上空有一冷涡，冷中心强度达到-48℃。",
            ),
        ]
        result = aggregate_metric_facts(
            chunks,
            metric="air_temperature.minimum",
            operator="min",
            chunk_case_ids={"surface-temp": ["case-cold"], "joined-aloft": ["case-cold"]},
        )
        self.assertEqual(result["winner"]["normalized_value"], -25.6)
        self.assertEqual(result["winner"]["location"], "新荣")
        self.assertEqual(result["selected_case_ids"], ["case-cold"])

    def test_temperature_sequence_extracts_location_before_parenthesized_value(self) -> None:
        """“地点（数值）”格式应同时提取邻近日号、地点和最低气温。"""
        chunk = DocumentChunk(
            "山西省2025年1月天气过程总结.pdf", "late-cold", 1,
            "26日最低气温出现在新荣（-29.8℃），27日最低气温出现在新荣（-30.7℃）；",
        )
        result = aggregate_metric_facts(
            [chunk], metric="air_temperature.minimum", operator="min",
            chunk_case_ids={"late-cold": ["case-late-cold"]},
        )
        self.assertEqual(result["winner"]["normalized_value"], -30.7)
        self.assertEqual(result["winner"]["time"], "2025年1月27日")
        self.assertEqual(result["winner"]["location"], "新荣")


if __name__ == "__main__":
    unittest.main()
