"""验证灾种动态指标配置不会退化为单一雨雪规则。"""
from __future__ import annotations

import unittest
import json

from ..matching.dimension_profiles import (
    calculate_dimension_compatibility,
    calculate_dimension_coverage,
    effective_dimension_compatibility,
    resolve_dimension_profile,
)
from ..llm.llm_service import SmartCaseLlmService
from ..matching.matching import structured_recall
from ..matching.selection import apply_llm_tie_break


class DynamicDimensionTests(unittest.TestCase):
    def test_wind_profile_does_not_use_snow_only_metrics(self):
        """大风查询应使用阵风和气压梯度等指标，而不是积雪深度。"""
        profile = resolve_dimension_profile({"disaster_types": ["大风"]})
        keys = {item["key"] for item in profile["dimensions"]}
        self.assertIn("intensity", keys)
        self.assertIn("mechanism", keys)
        self.assertNotIn("snow_depth", keys)
        self.assertIn("最大阵风", profile["metrics"])

    def test_compound_profile_keeps_secondary_hazard_metrics(self):
        """复合过程要比较全部已识别灾种指标，主次划分不能隐藏伴随灾种的关键量级。"""
        profile = resolve_dimension_profile({
            "disaster_types": ["沙尘", "大风", "寒潮", "降雪"],
            "primary_disaster_types": ["沙尘"],
        })
        self.assertIn("sand", profile["profile_id"])
        self.assertIn("wind", profile["profile_id"])
        self.assertIn("cold", profile["profile_id"])
        self.assertIn("snow", profile["profile_id"])
        self.assertIn("能见度", profile["metrics"])
        self.assertIn("最大阵风", profile["metrics"])
        self.assertIn("降温幅度", profile["metrics"])
        self.assertIn("相态演变", profile["metrics"])

    def test_missing_dimensions_are_renormalized(self):
        """历史材料缺失某项指标时，只按有证据的维度计算，不把缺失当零分。"""
        profile = resolve_dimension_profile({"disaster_types": ["高温"]})
        score, used = calculate_dimension_compatibility(
            {"dimension_scores": {"hazard_match": 0.9, "intensity": 0.8}},
            profile,
        )
        self.assertIsNotNone(score)
        self.assertEqual(set(used), {"hazard_match", "intensity"})
        self.assertGreater(score, 0.8)

    def test_missing_high_weight_dimensions_reduce_effective_compatibility(self):
        """只有基础标签分时保留原始分，但用于排序的有效兼容度必须受覆盖度约束。"""
        profile = resolve_dimension_profile({"disaster_types": ["大风"]})
        assessment = {"dimension_scores": {"hazard_match": 1.0, "temporal": 1.0}}
        raw, _ = calculate_dimension_compatibility(assessment, profile)
        effective, _, coverage = effective_dimension_compatibility(assessment, profile)
        self.assertAlmostEqual(coverage, calculate_dimension_coverage(assessment, profile))
        self.assertLess(coverage, 0.6)
        self.assertLess(effective, raw)

    def test_mechanism_complete_candidate_beats_season_only_candidate(self):
        """跨灾种通用规则应让机制、强度和演变证据完整的候选超过只靠时段的候选。"""
        profile = resolve_dimension_profile({"disaster_types": ["雨雪"]})
        ranked = apply_llm_tie_break(
            [
                {"case_id": "season-only", "retrieval_score": 0.92, "score_breakdown": {}},
                {"case_id": "mechanism-complete", "retrieval_score": 0.89, "score_breakdown": {}},
            ],
            # 即使模型顺序把季节候选放前，也不能推翻明显的动态兼容度差距。
            ["season-only", "mechanism-complete"],
            {
                "season-only": {"dimension_scores": {"hazard_match": 1.0, "temporal": 1.0, "area": 0.8}},
                "mechanism-complete": {"dimension_scores": {
                    "hazard_match": 1.0,
                    "mechanism": 0.95,
                    "intensity": 0.85,
                    "temporal": 0.7,
                    "area": 0.8,
                    "evolution": 0.9,
                }},
            },
            dimension_profile=profile,
        )
        self.assertEqual(ranked[0]["case_id"], "mechanism-complete")
        season = next(item for item in ranked if item["case_id"] == "season-only")
        self.assertLess(season["dimension_coverage"], 1.0)

    def test_structured_recall_uses_profile_weights(self):
        """同样的落区和灾种分数下，大风配置应按照灾种权重稳定计算。"""
        query = {
            "disaster_types": ["大风"],
            "primary_disaster_types": ["大风"],
            "secondary_disaster_types": [],
            "affected_areas": ["太原"],
            "months": [4],
            "query_text": "灾种：大风 区域：太原",
        }
        cases = [{
            "case_id": "wind-case",
            "disaster_types": ["大风"],
            "affected_areas": ["太原"],
            "date_range": "2025年4月20-21日",
            "title": "大风天气过程",
        }]
        result = structured_recall(query, cases)[0]
        self.assertAlmostEqual(result["score_breakdown"]["disaster"], 1.0)
        self.assertEqual(result["dimension_profile"]["profile_id"], "wind")
        self.assertEqual(result["dimension_profile"]["structured_weights"]["disaster"], 0.55)

    def test_candidate_protocol_accepts_profile_dimensions(self):
        """候选协议应能接收大风等非雨雪灾种的动态维度。"""
        class WindClient:
            model = "dynamic-test"

            def is_available(self):
                return True

            def answer_with_context(self, question, context_blocks, max_tokens=1800):
                return json.dumps({
                    "ordered_case_ids": ["wind-case"],
                    "candidate_assessments": [{
                        "case_id": "wind-case",
                        "dimension_scores": {
                            "hazard_match": 0.95,
                            "intensity": 0.8,
                            "mechanism": 0.7,
                        },
                        "metric_scores": {"最大阵风": 0.82, "风力演变": 0.74},
                        "missing_dimensions": ["temporal"],
                    }],
                }, ensure_ascii=False)

        ordered, assessments, status = SmartCaseLlmService(WindClient()).rerank(
            {"disaster_types": ["大风"]},
            [{"case_id": "wind-case"}],
        )
        self.assertEqual(status, "called")
        self.assertEqual(ordered, ["wind-case"])
        self.assertEqual(assessments["wind-case"]["dimension_scores"]["intensity"], 0.8)
        self.assertEqual(assessments["wind-case"]["metric_scores"]["最大阵风"], 0.82)

    def test_partial_candidate_assessment_uses_structured_completion(self):
        """模型只评估部分候选时应保留有效结果，并用确定性分数补齐其余候选。"""
        class PartialClient:
            model = "partial-test"

            def is_available(self):
                return True

            def answer_with_context(self, question, context_blocks, max_tokens=1800):
                return json.dumps({
                    "ordered_case_ids": ["case-1"],
                    "candidate_assessments": [{
                        "case_id": "case-1",
                        "dimension_scores": {"hazard_match": 0.9, "intensity": 0.8},
                        "missing_dimensions": [],
                    }],
                }, ensure_ascii=False)

        candidates = [
            {"case_id": "case-1", "score_breakdown": {"disaster": 1.0, "area": 0.8, "temporal": 0.7}},
            {"case_id": "case-2", "score_breakdown": {"disaster": 0.9, "area": 0.7, "temporal": 0.6}},
        ]
        _, assessments, status = SmartCaseLlmService(PartialClient()).rerank(
            {"disaster_types": ["大风"]},
            candidates,
        )
        self.assertEqual(status, "called_partial")
        self.assertEqual(assessments["case-1"]["assessment_source"], "llm")
        self.assertEqual(assessments["case-1"]["dimension_scores"]["area"], 0.8)
        self.assertEqual(assessments["case-2"]["assessment_source"], "structured_fallback")
        self.assertEqual(assessments["case-2"]["dimension_scores"]["hazard_match"], 0.9)


if __name__ == "__main__":
    unittest.main()
