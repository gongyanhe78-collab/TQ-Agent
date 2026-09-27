"""相似个例结构化评分的边界测试。"""
from __future__ import annotations

import unittest

from ..infrastructure.data_store import LocalCaseDataStore
from ..matching.matching import normalize_query, structured_recall


class StructuredScoringTests(unittest.TestCase):
    """验证标题灾种优先级和跨年季节距离。"""

    def test_title_disaster_restrains_noisy_tags(self):
        """标题明确为高温时，正文误抽出的暴雨标签不能得到满分。"""
        query = normalize_query({"disaster_types": ["暴雨"], "start_date": "2026-07-10", "top_n": 3})
        cases = [
            {"case_id": "rain", "title": "7月暴雨过程", "date_range": "2020年7月", "disaster_types": ["暴雨"], "affected_areas": []},
            {"case_id": "heat", "title": "7月高温过程", "date_range": "2025年7月", "disaster_types": ["高温", "暴雨"], "affected_areas": []},
        ]
        ranked = structured_recall(query, cases)
        self.assertEqual(ranked[0]["case_id"], "rain")
        heat = next(item for item in ranked if item["case_id"] == "heat")
        self.assertLessEqual(heat["score_breakdown"]["disaster"], 0.35)

    def test_broad_precipitation_title_passes_rain_family_gate(self):
        """暴雨查询应接纳标题为宽泛“降水过程”的真实同家族个例。"""
        query = normalize_query({
            "disaster_types": ["暴雨"],
            "start_date": "2026-06-13",
            "end_date": "2026-06-14",
            "affected_areas": ["全省"],
            "top_n": 3,
        })
        case = next(
            item for item in LocalCaseDataStore().cases
            if item["case_id"] == "FST2025-6-std-case-001"
        )
        result = structured_recall(query, [case])[0]

        self.assertTrue(result["disaster_gate_passed"])
        self.assertGreater(result["structured_score"], 0.80)

    def test_broad_precipitation_title_does_not_admit_snow_query(self):
        """宽泛“降水”只属于降雨家族，不能放松雨雪查询的标题门禁。"""
        query = normalize_query({"disaster_types": ["暴雪"], "start_date": "2026-06-13", "top_n": 3})
        case = {
            "case_id": "broad-rain",
            "title": "6月大范围降水过程",
            "date_range": "2025年6月13-14日",
            "disaster_types": ["暴雪"],
            "affected_areas": [],
        }
        result = structured_recall(query, [case])[0]

        self.assertFalse(result["disaster_gate_passed"])

    def test_december_and_january_are_adjacent(self):
        """月份距离应按环形计算，12 月和 1 月为相邻月份。"""
        query = normalize_query({"start_date": "2026-01-10", "top_n": 3})
        cases = [
            {"case_id": "december", "title": "12月寒潮过程", "date_range": "2024年12月10日", "disaster_types": ["寒潮"], "affected_areas": []},
            {"case_id": "july", "title": "7月过程", "date_range": "2025年7月10日", "disaster_types": [], "affected_areas": []},
        ]
        ranked = structured_recall(query, cases)
        self.assertEqual(ranked[0]["case_id"], "december")

    def test_snow_query_prioritizes_explicit_snow_cases_over_warm_season_rain(self):
        """雪类查询必须优先标题明确的雨雪个例，不能被暖季降水标签抬高。"""
        query = normalize_query({
            "date": "1月23-26日",
            "disaster_types": ["暴雪", "雨雪", "降雪"],
            "affected_areas": ["全省"],
            "top_n": 3,
        })
        cases = [
            {
                "case_id": "snow",
                "title": "1月雨雪寒潮天气过程",
                "date_range": "2025年1月23-26日",
                "disaster_types": ["雨雪", "降雪", "暴雪"],
                "affected_areas": ["全省"],
            },
            {
                "case_id": "warm-rain",
                "title": "4月降水大风天气过程",
                "date_range": "2025年4月20-21日",
                "disaster_types": ["雨雪", "寒潮", "大风"],
                "affected_areas": ["全省"],
            },
        ]
        ranked = structured_recall(query, cases)
        self.assertEqual(ranked[0]["case_id"], "snow")
        warm_rain = next(item for item in ranked if item["case_id"] == "warm-rain")
        self.assertFalse(warm_rain["disaster_gate_passed"])
        self.assertLess(warm_rain["structured_score"], 0.30)

    def test_compound_hazard_uses_mentions_and_coverage_without_snow_preference(self):
        """复合过程应由原文重要性和灾种覆盖率排序，局地弱降雪不能压过主要沙尘过程。"""
        raw_query = (
            "3月25日到28日出现沙尘与强降温复合过程。25到26日受强冷空气和蒙古气旋影响，"
            "全省出现大风沙尘天气，平均风力5到7级、阵风8到10级，中北部能见度明显降低；"
            "27到28日继续降温，局地伴随弱降雪。"
        )
        query = normalize_query({
            "process_name": "3月大风沙尘与强降温复合过程",
            "date": "3月25日到28日",
            "disaster_types": ["沙尘", "大风", "寒潮", "降雪"],
            "affected_areas": ["全省"],
            "raw_query": raw_query,
        })
        self.assertEqual(query["primary_disaster_types"], ["沙尘"])
        self.assertEqual(query["disaster_importance"]["降雪"]["role"], "secondary")

        ranked = structured_recall(query, LocalCaseDataStore().cases)
        self.assertGreaterEqual(len(ranked), 15)
        target = next(item for item in ranked if item["case_id"] == "FST2025-4-std-case-002")
        self.assertTrue(target["disaster_gate_passed"])
        self.assertGreater(target["score_breakdown"]["disaster_coverage"], 0.70)
        self.assertLessEqual(ranked.index(target), 2)


if __name__ == "__main__":
    unittest.main()
