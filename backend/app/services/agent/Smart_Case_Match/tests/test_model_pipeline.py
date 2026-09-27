"""模型精排、证据传递和调用状态的单元测试。"""
from __future__ import annotations

import json
import unittest

from ..infrastructure.data_store import LocalCaseDataStore
from ..matching.dimension_profiles import terminology_profile_context
from ..llm.llm_service import (
    SmartCaseLlmService,
    _find_terminology_violations,
    _is_data_inspection_error,
    _payload_diagnostics,
)
from ..matching.semantic_rerank import rerank_semantic_hits
from ..matching.selection import apply_llm_tie_break


class RecordingLlmClient:
    """记录提示内容，验证向量证据确实进入 LLM 候选重排。"""

    model = "unit-test-llm"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def is_available(self) -> bool:
        return True

    def answer_with_context(self, question, context_blocks, max_tokens=1800):
        self.calls.append({"question": question, "context_blocks": context_blocks, "max_tokens": max_tokens})
        return json.dumps({
            "ordered_case_ids": ["case-1"],
            "candidate_assessments": [{
                "case_id": "case-1",
                "mechanism_score": 0.8,
                "intensity_score": 0.75,
            }],
        }, ensure_ascii=False)


class RecordingRerankClient:
    """模拟专用 Rerank 模型并记录收到的 chunk。"""

    model = "unit-test-rerank"

    def __init__(self) -> None:
        self.calls = []

    def is_available(self) -> bool:
        return True

    def rerank(self, question, hits, top_n):
        self.calls.append({"question": question, "hits": hits, "top_n": top_n})
        return list(reversed(hits))[:top_n]


class NaturalQueryCountingClient:
    """返回合法自然语言解析协议并记录真实模型调用次数。"""

    model = "unit-test-natural-query"

    def __init__(self) -> None:
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def answer_with_context(self, _question, _context_blocks, max_tokens=1800):
        self.calls += 1
        return json.dumps({
            "process_name": "当前暴雨过程",
            "date_expression": "",
            "date_evidence": "",
            "start_date": "",
            "end_date": "",
            "date": "",
            "disaster_types": [{"value": "暴雨", "evidence": "暴雨"}],
            "affected_areas": [],
            "observation_description": "当前出现暴雨",
            "circulation_description": "",
            "intensity_description": "",
            "metric_descriptions": {},
            "raw_query": "",
        }, ensure_ascii=False)


class ModelPipelineTests(unittest.TestCase):
    """验证专用 Rerank 与候选 LLM 重排都使用向量命中正文。"""

    def test_validated_smart_stage_result_is_cached(self):
        """完全相同的 Smart 阶段输入应复用已通过协议校验的结果。"""
        client = NaturalQueryCountingClient()
        service = SmartCaseLlmService(client)
        try:
            first, first_status = service.parse_natural_query("当前出现暴雨")
            second, second_status = service.parse_natural_query("当前出现暴雨")
        finally:
            service.close()

        self.assertEqual(first, second)
        self.assertEqual(first_status, "called")
        self.assertEqual(second_status, "called")
        self.assertEqual(client.calls, 1)

    def test_input_inspection_diagnostics_keep_source_ids_and_short_previews(self):
        """审核诊断应能定位候选个例和 chunk，但不能把完整正文写入日志。"""
        summary, records = _payload_diagnostics({
            "当前过程": {"raw_query": "雨雪过程"},
            "候选个例": [{
                "case_id": "case-1",
                "semantic_evidence": [{
                    "chunk_id": "chunk-1",
                    "content": "历史正文" * 200,
                }],
            }],
        })
        self.assertEqual(summary["case_ids"], ["case-1"])
        self.assertEqual(summary["chunk_ids"], ["chunk-1"])
        content_record = next(item for item in records if item["path"].endswith("content"))
        self.assertEqual(content_record["case_id"], "case-1")
        self.assertEqual(content_record["chunk_id"], "chunk-1")
        self.assertLessEqual(len(content_record["preview"]), 160)
        self.assertTrue(_is_data_inspection_error(RuntimeError("data_inspection_failed")))
        self.assertFalse(_is_data_inspection_error(RuntimeError("timeout")))

    def test_input_diagnostics_still_collects_chunk_ids_after_preview_limit(self):
        """文本预览达到120项后仍应继续扫描来源ID，避免正文chunk在复杂配置中显示为空。"""
        payload = {f"field_{index}": f"value-{index}" for index in range(130)}
        payload["历史个例"] = {
            "case_id": "case-after-limit",
            "正文片段": [{"chunk_id": "chunk-after-limit", "content": "真实正文"}],
        }
        summary, records = _payload_diagnostics(payload)
        self.assertEqual(len(records), 120)
        self.assertIn("case-after-limit", summary["case_ids"])
        self.assertIn("chunk-after-limit", summary["chunk_ids"])

    def test_case_reference_normalizes_common_model_field_drift(self):
        """额外字段、空数组和明确别名应安全归一化，不能让整份有效参考经验降级。"""
        class DriftedReferenceClient:
            model = "reference-drift-test"

            def is_available(self):
                return True

            def answer_with_context(self, question, context_blocks, max_tokens=1800):
                return json.dumps({
                    "case_id": "case-1",
                    "match_reasons": [{"reason": "双方均受低层切变和水汽输送影响"}],
                    "reference_points": [{
                        "reference": "历史个例中低层切变维持并伴随水汽输送。",
                        "chunk_ids": "chunk-1",
                        "confidence": 0.8,
                    }],
                    "similarities": None,
                    "differences": [{"difference": "历史过程持续时间更短"}],
                    "warning_references": None,
                }, ensure_ascii=False)

        result, status = SmartCaseLlmService(DriftedReferenceClient()).extract_case_reference(
            {"disaster_types": ["暴雨"], "circulation_description": "低层切变和水汽输送"},
            {
                "case_id": "case-1",
                "title": "历史暴雨过程",
                "disaster_types": ["暴雨"],
                "relevant_chunks": [{"chunk_id": "chunk-1", "content": "低层切变维持并伴随水汽输送。"}],
            },
        )
        self.assertEqual(status, "called")
        self.assertEqual(result["reference_source"], "llm_valid")
        self.assertEqual(result["reference_points"][0]["evidence_chunk_ids"], ["chunk-1"])
        self.assertEqual(result["differences"], ["历史过程持续时间更短"])

    def test_case_reference_retries_once_when_evidence_id_is_invalid(self):
        """首次返回编造chunk ID时应执行极简重提炼，修复成功后不得产生规则降级。"""
        class EvidenceRepairClient:
            model = "reference-evidence-repair-test"

            def __init__(self):
                self.calls = 0

            def is_available(self):
                return True

            def answer_with_context(self, question, context_blocks, max_tokens=1800):
                self.calls += 1
                chunk_id = "missing-chunk" if self.calls == 1 else "chunk-1"
                return json.dumps({
                    "match_reasons": ["低层切变配置相近"],
                    "reference_points": [{
                        "text": "历史个例表明低层切变维持时降水可以持续。",
                        "evidence_chunk_ids": [chunk_id],
                    }],
                    "similarities": ["低层切变维持"],
                    "differences": ["历史过程持续时间更短"],
                    "warning_references": [],
                }, ensure_ascii=False)

        client = EvidenceRepairClient()
        result, status = SmartCaseLlmService(client).extract_case_reference(
            {"disaster_types": ["暴雨"], "circulation_description": "低层切变"},
            {
                "case_id": "case-1",
                "title": "历史暴雨过程",
                "disaster_types": ["暴雨"],
                "relevant_chunks": [{"chunk_id": "chunk-1", "content": "低层切变维持时降水持续。"}],
            },
        )
        self.assertEqual(client.calls, 2)
        self.assertEqual(status, "called_repaired")
        self.assertEqual(result["reference_source"], "llm_repaired")
        self.assertEqual(result["reference_points"][0]["evidence_chunk_ids"], ["chunk-1"])

    def test_llm_rerank_receives_semantic_evidence(self):
        """LLM 候选重排的上下文必须包含真实 chunk ID 和正文。"""
        client = RecordingLlmClient()
        service = SmartCaseLlmService(client)
        ordered, reasons, status = service.rerank(
            {"query_text": "晋南暴雨和低层切变"},
            [{
                "case_id": "case-1",
                "title": "历史暴雨过程",
                "retrieval_score": 0.88,
                "semantic_evidence": [{"chunk_id": "chunk-1", "content": "低层切变维持并伴随水汽输送"}],
            }],
        )
        self.assertEqual(status, "called")
        self.assertEqual(ordered, ["case-1"])
        self.assertEqual(reasons["case-1"]["mechanism_score"], 0.8)
        serialized_context = "".join(client.calls[0]["context_blocks"])
        self.assertIn("chunk-1", serialized_context)
        self.assertIn("低层切变", serialized_context)

    def test_candidate_mechanism_and_intensity_scores_only_lower_weak_case(self):
        """机制和强度明显不符时应压低候选分，不能被原始落区分抬到第一位。"""
        ranked = apply_llm_tie_break(
            [
                {"case_id": "weak", "retrieval_score": 0.93, "score_breakdown": {}},
                {"case_id": "strong", "retrieval_score": 0.90, "score_breakdown": {}},
            ],
            ["weak", "strong"],
            {
                "weak": {"mechanism_score": 0.25, "intensity_score": 0.30},
                "strong": {"mechanism_score": 0.90, "intensity_score": 0.90},
            },
        )
        self.assertEqual(ranked[0]["case_id"], "strong")
        weak = next(item for item in ranked if item["case_id"] == "weak")
        self.assertLess(weak["retrieval_score"], 0.93)
        self.assertEqual(weak["score_breakdown"]["mechanism"], 0.25)

    def test_dedicated_rerank_updates_semantic_representatives(self):
        """专用 Rerank 应真实接收向量命中的 chunk 并返回调用状态。"""
        store = LocalCaseDataStore()
        first, second = store.chunks[:2]
        client = RecordingRerankClient()
        candidates = [{
            "case_id": "case-1",
            "semantic_score": 0.8,
            "semantic_available": True,
            "semantic_hits": [
                {"chunk_id": first["chunk_id"], "score": 0.9},
                {"chunk_id": second["chunk_id"], "score": 0.8},
            ],
        }]
        updated, status, warnings = rerank_semantic_hits(
            {"query_text": "暴雨环流形势"},
            candidates,
            store,
            client,
        )
        self.assertEqual(status, "called")
        self.assertFalse(warnings)
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(updated[0]["semantic_rerank_available"])
        self.assertEqual(updated[0]["semantic_chunk_ids"][0], second["chunk_id"])

    def test_invalid_llm_numeric_fields_degrade_instead_of_raising(self):
        """非法置信度必须触发协议降级，不能在 float 转换处击穿整条图。"""
        class InvalidSynthesisClient:
            model = "invalid-schema-test"

            def is_available(self):
                return True

            def answer_with_context(self, question, context_blocks, max_tokens=1800):
                return json.dumps({
                    "forecast_summary": {
                        "similarity_assessment": "相似",
                        "core_features": ["切变"],
                        "main_risk": "落区偏差",
                        "confidence": "较高",
                    },
                    "forecast_tips": [],
                }, ensure_ascii=False)

        references = [{
            "case_id": "case-1",
            "reference_points": [{"text": "历史事实", "evidence_chunk_ids": ["chunk-1"]}],
        }]
        result, status = SmartCaseLlmService(InvalidSynthesisClient()).synthesize({}, references)
        self.assertEqual(status, "invalid_schema")
        self.assertEqual(result["forecast_summary"]["confidence"], 0.5)

    def test_synthesis_removes_case_without_its_own_evidence(self):
        """支持个例必须至少拥有一条被该提示引用的真实 chunk。"""
        class CrossCaseClient:
            model = "cross-case-test"

            def is_available(self):
                return True

            def answer_with_context(self, question, context_blocks, max_tokens=1800):
                return json.dumps({
                    "forecast_summary": {
                        "similarity_assessment": "较相似",
                        "core_features": ["切变"],
                        "main_risk": "落区偏差",
                        "confidence": 0.8,
                    },
                    "forecast_tips": [{
                        "priority": 1,
                        "focus_object": "降水落区",
                        "possible_bias": "可能偏北",
                        "suggested_action": "核查雷达并订正落区",
                        "support_case_ids": ["case-1", "case-2"],
                        "evidence_chunk_ids": ["chunk-1"],
                        "consensus_level": "多数个例共同支持",
                        "confidence": 0.8,
                    }],
                }, ensure_ascii=False)

        references = [
            {"case_id": "case-1", "reference_points": [{"text": "事实1", "evidence_chunk_ids": ["chunk-1"]}]},
            {"case_id": "case-2", "reference_points": [{"text": "事实2", "evidence_chunk_ids": ["chunk-2"]}]},
        ]
        result, status = SmartCaseLlmService(CrossCaseClient()).synthesize({}, references)
        self.assertEqual(status, "called")
        self.assertEqual(result["forecast_tips"][0]["support_case_ids"], ["case-1"])
        self.assertEqual(result["forecast_tips"][0]["consensus_level"], "单个例提示")

    def test_cross_profile_term_triggers_targeted_repair(self):
        """暴雨综合中出现雨雪专属术语时只触发一次术语修复，不把错误概念展示出去。"""
        class TerminologyRepairClient:
            model = "terminology-repair-test"

            def __init__(self):
                self.calls = 0

            def is_available(self):
                return True

            def answer_with_context(self, question, context_blocks, max_tokens=1800):
                self.calls += 1
                if self.calls == 1:
                    summary_assessment = "当前过程与历史过程均有冷垫配置，暴雨强度接近。"
                    suggested_action = "监测冷垫厚度并结合地面温度调整结论"
                else:
                    summary_assessment = "当前过程与历史过程均有水汽输送和低层急流，小时雨强接近。"
                    suggested_action = "核查雷达回波和小时雨强并订正落区"
                return json.dumps({
                    "forecast_summary": {
                        "similarity_assessment": summary_assessment,
                        "core_features": ["水汽输送和低层急流配置相近。"],
                        "main_risk": "小时雨强偏差可能影响局地落区判断。",
                        "confidence": 0.8,
                        "support_case_ids": ["case-1"],
                    },
                    "forecast_tips": [{
                        "priority": 1,
                        "focus_object": "小时雨强和雷达回波",
                        "possible_bias": "局地强度可能偏弱",
                        "suggested_action": suggested_action,
                        "support_case_ids": ["case-1"],
                        "evidence_chunk_ids": ["chunk-1"],
                        "consensus_level": "单个例提示",
                        "confidence": 0.8,
                    }],
                }, ensure_ascii=False)

        query = {
            "disaster_types": ["暴雨"],
            "months": [7],
            "raw_query": "7月暴雨过程，关注水汽输送、低层急流和小时雨强",
        }
        references = [{
            "case_id": "case-1",
            "case_title": "7月暴雨过程",
            "date_range": "2025年7月2-3日",
            "reference_points": [{
                "text": "低层急流输送水汽，雷达回波和小时雨强均较大。",
                "evidence_chunk_ids": ["chunk-1"],
            }],
        }]
        context = terminology_profile_context(query)
        self.assertIn("冷垫", context["foreign_terms"])
        self.assertNotIn("冷垫", context["active_terms"])
        client = TerminologyRepairClient()
        result, status = SmartCaseLlmService(client).synthesize(query, references)
        self.assertEqual(status, "called")
        self.assertEqual(client.calls, 2)
        self.assertNotIn("冷垫", result["forecast_summary"]["similarity_assessment"])
        self.assertIn("水汽输送", result["forecast_summary"]["similarity_assessment"])
        self.assertEqual(result["forecast_tips"][0]["support_case_ids"], ["case-1"])

    def test_term_validator_allows_plausible_profile_signal_without_exact_word(self):
        """当前灾种下语义合理的指标可以保留，不因缺少逐字证据而误触发修复。"""
        query = {"disaster_types": ["暴雨"], "raw_query": "7月暴雨，关注水汽输送"}
        context = terminology_profile_context(query)
        output = {
            "forecast_summary": {
                "similarity_assessment": "当前过程需关注降水演变。",
                "core_features": [],
                "main_risk": "",
                "support_case_ids": ["case-1"],
            },
            "forecast_tips": [],
        }
        violations = _find_terminology_violations(query, [{
            "case_id": "case-1",
            "reference_points": [{"text": "水汽输送明显。", "evidence_chunk_ids": ["chunk-1"]}],
        }], output, context)
        self.assertNotIn("降水演变", [item["term"] for item in violations])

    def test_term_validator_accepts_cross_profile_visibility_and_wind_terms(self):
        """沙尘综合可使用能见度演变、最大阵风等伴随指标，不应被误判。"""
        query = {
            "disaster_types": ["沙尘"],
            "raw_query": "沙尘过程，关注能见度和地面风速变化",
        }
        context = terminology_profile_context(query)
        output = {
            "forecast_summary": {
                "similarity_assessment": "需关注能见度演变和最大阵风变化。",
                "core_features": [],
                "main_risk": "",
                "support_case_ids": ["case-1"],
            },
            "forecast_tips": [],
        }
        violations = _find_terminology_violations(query, [{
            "case_id": "case-1",
            "reference_points": [{"text": "地面风速增强，能见度下降。", "evidence_chunk_ids": ["chunk-1"]}],
        }], output, context)
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
