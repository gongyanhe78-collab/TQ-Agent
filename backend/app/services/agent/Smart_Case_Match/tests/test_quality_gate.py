"""业务质量门槛测试。"""
from __future__ import annotations

import unittest

from ..llm.llm_service import _clean_summary, _prioritize_forecast_tips
from ..matching.matching import normalize_query, structured_recall
from ..llm.text_quality import (
    build_action_tip,
    clean_reference_text,
    filter_specific_match_reasons,
    make_transferable_reference,
    normalize_action_text,
    normalize_business_text,
    normalize_soft_text,
)


class QualityGateTests(unittest.TestCase):
    """验证标题灾种冲突不会被宽泛标签抬高。"""

    def test_conflicting_title_falls_below_selection_threshold(self):
        """查询暴雨时，标题为高温的污染个例必须低于选择门槛。"""
        query = normalize_query({
            "start_date": "2026-07-10",
            "disaster_types": ["暴雨"],
            "affected_areas": ["太原"],
            "top_n": 3,
        })
        cases = [{
            "case_id": "polluted-heat",
            "title": "7月持续性高温天气过程",
            "date_range": "2025年7月10-12日",
            "disaster_types": ["高温", "暴雨", "强降水"],
            "affected_areas": ["太原"],
        }]
        result = structured_recall(query, cases)[0]
        self.assertLess(result["structured_score"], 0.30)

    def test_action_tip_prefers_short_structured_fields(self):
        """综合提示应使用三个短字段，不能继续采用模型返回的长段落。"""
        tip = build_action_tip({
            "focus_object": "12日夜间晋北雨雪转换及850hPa -4℃线",
            "possible_bias": "模式可能低估降雪范围",
            "suggested_action": "结合雷达和自动站实况订正晋北降雪落区",
            "text": "这是一段不应优先展示的冗长模型原文。",
        })
        self.assertIn("关注：12日夜间晋北", tip)
        self.assertIn("偏差：模式可能低估", tip)
        self.assertIn("建议：结合雷达和自动站", tip)
        self.assertNotIn("冗长模型原文", tip)

    def test_reference_text_removes_current_forecast_advice(self):
        """参考经验仅保留历史事实，删除面向当前过程的预报建议。"""
        text = clean_reference_text(
            "历史个例中700hPa切变于夜间东移。对本次相态预报具有参考价值，建议加强主观订正。"
        )
        self.assertEqual(text, "历史个例中700hPa切变于夜间东移")

    def test_reference_text_removes_indirect_current_process_value(self):
        """间接表达的当前过程参考价值也不能混入历史经验。"""
        text = clean_reference_text(
            "历史个例中地面冷垫维持12小时。该特征可作为当前相态研判依据。"
        )
        self.assertEqual(text, "历史个例中地面冷垫维持12小时")

    def test_reference_description_is_rewritten_as_cautious_transferable_experience(self):
        """一次性历史描述应改成“可表现为”，不能直接冒充普遍规律。"""
        text = make_transferable_reference("该次过程单日站数较少，最大降水量为5.4mm。", 90)
        self.assertTrue(text.startswith("该历史个例表明，在相近配置下可表现为"))
        self.assertNotIn("通常", text)

    def test_match_reason_rejects_season_and_hazard_template(self):
        """只写同月份、同灾种和宽泛相态的理由应被过滤。"""
        reasons = filter_specific_match_reasons(
            [
                "同属2月山西雨雪过程，均出现雨雪相态转换",
                "双方均有850hPa冷垫和低层切变配置，主导机制相近",
            ],
            "当前与历史正文均包含850hPa冷垫、低层切变和相态转换",
            ["冷垫", "切变", "相态转换"],
        )
        self.assertEqual(reasons, ["双方均有850hPa冷垫和低层切变配置，主导机制相近"])

    def test_wind_reason_uses_wind_signals_without_snow_assumptions(self):
        """同一质量门槛应接受大风机制信号，证明规则不是雨雪专用。"""
        reasons = filter_specific_match_reasons(
            ["气压梯度和冷平流配置相近，最大阵风量级接近"],
            "当前与历史过程均有气压梯度、冷平流和最大阵风记录",
            ["气压梯度", "冷平流", "阵风"],
        )
        self.assertEqual(len(reasons), 1)

    def test_action_text_adds_explicit_business_action(self):
        """模型只返回分析表述时，建议字段应补成明确核查动作。"""
        text = normalize_action_text("28日夜间晋北相态分界线变化", 45)
        self.assertTrue(text.startswith("核查"))

    def test_action_text_keeps_complete_sentence_over_target_length(self):
        """建议略超目标长度时保留完整句，不能在动作对象中间硬截断。"""
        text = normalize_action_text(
            "结合自动站、雷达和卫星资料核查沙尘影响范围，及时调整重点区域预报结论。",
            45,
        )
        self.assertTrue(text.endswith("。"))
        self.assertIn("调整重点区域预报结论", text)
        self.assertLessEqual(len(text), 100)

    def test_soft_text_uses_ellipsis_when_no_sentence_boundary(self):
        """没有句末边界的异常长输出才使用省略号，不伪造句号。"""
        text = normalize_soft_text("连续输出但没有完整句末标点" * 20, 20, 70)
        self.assertTrue(text.endswith("…"))
        self.assertLessEqual(len(text), 70)

    def test_forecast_tips_keep_three_to_four_core_choices(self):
        """证据充足时应主展示三至四条，低置信度不再减少可选数量。"""
        tips = [
            {"tip_id": f"old-{index}", "priority": index, "confidence": confidence}
            for index, confidence in enumerate([0.88, 0.82, 0.76, 0.72, 0.65, 0.60], start=1)
        ]
        selected = _prioritize_forecast_tips(tips)
        self.assertEqual(
            [item["display_level"] for item in selected],
            ["core", "core", "core", "core"],
        )
        self.assertEqual([item["confidence"] for item in selected], [0.88, 0.82, 0.76, 0.72])

        low_confidence = _prioritize_forecast_tips([
            {"priority": index, "confidence": confidence}
            for index, confidence in enumerate([0.85, 0.65, 0.60], start=1)
        ])
        self.assertEqual(len(low_confidence), 3)
        self.assertTrue(all(item["display_level"] == "core" for item in low_confidence))
    def test_summary_is_short_and_conclusion_oriented(self):
        """摘要只保留两条依据，并在更高上限内避免无意义的过短截断。"""
        summary = _clean_summary({
            "similarity_assessment": "当前过程主要参照锋后回流型雨雪过程" * 4,
            "core_features": ["东路冷空气渗透形成冷垫" * 5, "短波槽配合低层倒槽" * 5, "不应保留"],
            "main_risk": "相态转换时间偏差可能扩大道路结冰落区" * 4,
            "confidence": 0.8,
        }, [])
        self.assertLessEqual(len(summary["similarity_assessment"]), 200)
        self.assertEqual(len(summary["core_features"]), 2)
        self.assertTrue(all(len(item) <= 180 for item in summary["core_features"]))
        self.assertLessEqual(len(summary["main_risk"]), 200)

    def test_summary_uses_case_name_and_keeps_complete_business_phrase(self):
        """综合正文不能泄露内部ID，且不能把“系统移动”截成半个短语。"""
        case_id = "FST2025-7-std-case-001"
        summary = _clean_summary({
            "similarity_assessment": (
                f"当前过程与{case_id}在环流机制（低槽+急流）和强度指标上最为相似，"
                "但系统移动缓慢，降水效率较高。"
            ),
            "core_features": [f"{case_id}低槽和急流配置相近。"],
            "main_risk": f"{case_id}的降水持续性对当前落区判断有参考价值。",
            "confidence": 0.85,
            "support_case_ids": [case_id],
        }, [{
            "case_id": case_id,
            "case_title": "7月2-3日分散性暴雨天气过程",
            "date_range": "2025年7月2-3日",
            "reference_points": [{"evidence_chunk_ids": ["FST2025-7-chunk-008"]}],
            "dimension_compatibility": 0.9,
            "retrieval_score": 0.9,
        }])
        self.assertNotIn(case_id, summary["similarity_assessment"])
        self.assertIn("2025年7月2-3日分散性暴雨天气过程", summary["similarity_assessment"])
        self.assertIn("系统移动缓慢", summary["similarity_assessment"])
        self.assertEqual(summary["support_case_ids"], [case_id])
    def test_business_text_limit_applies_after_replacement(self):
        """业务词替换扩展字符后，最终文本仍不得超过字段上限。"""
        text = normalize_business_text("这次过程的关注内容需要进一步核对", 8)
        self.assertLessEqual(len(text), 8)


if __name__ == "__main__":
    unittest.main()
