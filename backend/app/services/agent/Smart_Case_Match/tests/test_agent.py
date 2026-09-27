"""相似个例智能匹配体的本地单元测试。"""
from __future__ import annotations

import threading
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ..agent import SmartCaseMatchAgent
from ..infrastructure.data_store import LocalCaseDataStore
from ..matching.matching import normalize_query, structured_recall
from ..workflow.nodes import SmartCaseGraphNodes
from ..schemas import SmartCaseMatchRequest


class UnavailableEmbeddingClient:
    """模拟不可用的 Embedding，验证结构化降级链路。"""

    def is_available(self) -> bool:
        return False


class UnavailableLlmClient:
    """模拟不可用的聊天模型，验证规则提炼和综合降级链路。"""

    def is_available(self) -> bool:
        return False


class ConcurrentReferenceLlmService:
    """记录逐例提炼的并发数，并返回可识别的个例结果。"""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def extract_case_reference(self, query, case):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        return {"case_id": case["case_id"], "reference_points": []}, "called"


class RecordingRerankLlmService:
    """记录候选重排调用次数，用于验证每个请求固定调用一次模型。"""

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query, candidates, deadline=None):
        self.calls += 1
        return [str(item.get("case_id")) for item in candidates], {}, "called"


class RecordingSynthesisService:
    """记录综合节点收到的参考来源，验证规则降级内容不会进入跨个例共识。"""

    def __init__(self) -> None:
        self.references = []

    def synthesize(self, query, references, deadline=None):
        self.references = list(references)
        return {
            "forecast_summary": {
                "similarity_assessment": "有效参考可用于综合",
                "core_features": ["低层切变"],
                "main_risk": "落区偏差",
                "confidence": 0.8,
            },
            "forecast_tips": [],
        }, "called"


class SmartCaseMatchTests(unittest.TestCase):
    """覆盖本地数据完整性、季节排序和端到端输出协议。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.store = LocalCaseDataStore()

    def test_local_data_relations(self):
        """标准个例引用的 chunk 应全部存在，共享 chunk 应保留多个 case_id。"""
        self.assertGreaterEqual(len(self.store.cases), 20)
        self.assertGreaterEqual(len(self.store.chunks), 300)
        for case in self.store.cases:
            for chunk_id in case.get("source_chunk_ids") or []:
                self.assertIsNotNone(self.store.get_chunk(chunk_id))
        self.assertTrue(any(len(self.store.case_ids_for_chunk(chunk_id)) > 1 for chunk_id in self.store._case_ids_by_chunk))

    def test_specific_city_is_not_expanded_to_central_region(self):
        """晋中作为地市输入时不能自动扩展成太原、阳泉和吕梁。"""
        query = normalize_query({"affected_areas": ["晋中"], "top_n": 3})
        cases = [
            {"case_id": "jinzhong", "title": "晋中过程", "affected_areas": ["晋中"], "disaster_types": [], "date_range": ""},
            {"case_id": "taiyuan", "title": "太原过程", "affected_areas": ["太原"], "disaster_types": [], "date_range": ""},
        ]
        ranked = structured_recall(query, cases)
        self.assertEqual(ranked[0]["case_id"], "jinzhong")
        self.assertEqual(ranked[1]["score_breakdown"]["area"], 0.0)

    def test_default_case_count_is_three(self):
        """API 请求和内部标准化在未指定数量时都应默认处理三个个例。"""
        request = SmartCaseMatchRequest(disaster_types=["暴雨"])
        self.assertEqual(request.top_n, 3)
        self.assertEqual(normalize_query({"disaster_types": ["暴雨"]})["top_n"], 3)

    def test_rank_node_always_calls_candidate_llm_once(self):
        """即使融合分层次清楚，候选重排模型也必须固定调用一次。"""
        llm_service = RecordingRerankLlmService()
        nodes = SmartCaseGraphNodes(self.store, None, None, llm_service)
        candidates = []
        for index, score in enumerate((0.90, 0.76, 0.62, 0.45)):
            candidates.append({
                "case_id": f"case-{index}",
                "title": "暴雨过程",
                "source_pdf": f"source-{index}.pdf",
                "structured_score": score,
                "semantic_score": score,
                "semantic_available": True,
                "score_breakdown": {"disaster": 1.0, "area": 1.0, "temporal": 1.0},
                "structured_reasons": ["灾害类型高度相关"],
            })
        result = nodes.rank_and_select({
            "run_id": "",
            "query": {"disaster_types": ["暴雨"], "affected_areas": ["太原"], "top_n": 3, "diversity_mode": "moderate"},
            "candidate_cases": candidates,
            "warnings": [],
            "audit": {},
        })

        self.assertEqual(llm_service.calls, 1)
        self.assertEqual(result["audit"]["llm_rerank_status"], "called")
        self.assertFalse(result["warnings"])

    def test_case_reference_extraction_uses_two_workers_and_keeps_order(self):
        """逐例提炼最多并发两个任务，结果仍保持入选个例的原始顺序。"""
        llm_service = ConcurrentReferenceLlmService()
        nodes = SmartCaseGraphNodes(self.store, None, None, llm_service)
        # 本测试只验证并发行为，关闭最终重试退避以避免无证据测试数据增加等待时间。
        with patch("backend.app.services.agent.Smart_Case_Match.workflow.nodes.CASE_REFERENCE_FINAL_RETRY_DELAY_SECONDS", 0):
            result = nodes.extract_case_references({
                "run_id": "",
                "query": {},
                "enriched_cases": [{"case_id": f"case-{index}"} for index in range(3)],
            })

        self.assertEqual(llm_service.max_active, 2)
        self.assertEqual(
            [item["case_id"] for item in result["case_references"]],
            ["case-0", "case-1", "case-2"],
        )
        self.assertEqual(result["audit"]["reference_max_workers"], 2)

    def test_valid_case_reference_keeps_called_status(self):
        """模型提炼结果通过证据校验后必须保持成功状态，不能产生虚假降级警告。"""
        class ValidReferenceService:
            def extract_case_reference(self, query, case, deadline=None):
                return {
                    "case_id": case["case_id"],
                    "reference_points": [{"text": "有效参考经验", "evidence_chunk_ids": ["chunk-1"]}],
                }, "called"

        nodes = SmartCaseGraphNodes(self.store, None, None, ValidReferenceService())
        # 本测试只验证状态迁移，证据主题和数值校验已有独立测试覆盖。
        nodes._validate_reference_evidence = lambda query, reference, case: reference
        result = nodes.extract_case_references({
            "run_id": "",
            "query": {},
            "enriched_cases": [{"case_id": "case-1"}],
            "warnings": [],
            "audit": {},
        })

        self.assertEqual(result["audit"]["reference_statuses"], {"called": 1})
        self.assertFalse(result["warnings"])

    def test_repaired_case_reference_is_not_counted_as_degraded(self):
        """协议修复成功属于有效LLM提炼，不能继续显示未完成提炼警告。"""
        class RepairedReferenceService:
            def extract_case_reference(self, query, case, deadline=None):
                return {
                    "case_id": case["case_id"],
                    "reference_source": "llm_repaired",
                    "reference_points": [{"text": "有效参考经验", "evidence_chunk_ids": ["chunk-1"]}],
                }, "called_repaired"

        nodes = SmartCaseGraphNodes(self.store, None, None, RepairedReferenceService())
        nodes._validate_reference_evidence = lambda query, reference, case: reference
        result = nodes.extract_case_references({
            "run_id": "",
            "query": {},
            "enriched_cases": [{"case_id": "case-1"}],
            "warnings": [],
            "audit": {},
        })
        self.assertEqual(result["audit"]["reference_statuses"], {"called_repaired": 1})
        self.assertFalse(result["warnings"])

    def test_final_case_reference_retry_recovers_before_synthesis(self):
        """首次逐例提炼无效时应完整重试一次，成功后不得生成提炼或综合排除警告。"""
        class RetryReferenceService:
            def __init__(self):
                self.calls = 0

            def extract_case_reference(self, query, case, deadline=None):
                self.calls += 1
                if self.calls == 1:
                    return {"case_id": case["case_id"], "reference_points": []}, "invalid_evidence"
                return {
                    "case_id": case["case_id"],
                    "reference_source": "llm_repaired",
                    "reference_points": [{"text": "重试后有效经验", "evidence_chunk_ids": ["chunk-1"]}],
                }, "called_repaired"

        service = RetryReferenceService()
        nodes = SmartCaseGraphNodes(self.store, None, None, service)
        nodes._validate_reference_evidence = lambda query, reference, case: reference
        with patch("backend.app.services.agent.Smart_Case_Match.workflow.nodes.CASE_REFERENCE_FINAL_RETRY_DELAY_SECONDS", 0):
            result = nodes.extract_case_references({
                "run_id": "",
                "query": {},
                "enriched_cases": [{"case_id": "case-1"}],
                "warnings": [],
                "audit": {},
            })
        self.assertEqual(service.calls, 2)
        self.assertEqual(result["audit"]["reference_statuses"], {"called_repaired": 1})
        self.assertEqual(result["case_references"][0]["reference_source"], "llm_repaired")
        self.assertFalse(result["warnings"])

    def test_synthesis_excludes_rule_fallback_references(self):
        """规则摘要可以展示，但综合模型只能接收有效或已修复的LLM参考。"""
        llm_service = RecordingSynthesisService()
        nodes = SmartCaseGraphNodes(self.store, None, None, llm_service)
        result = nodes.synthesize_forecast_tips({
            "run_id": "",
            "query": {},
            "case_references": [
                {
                    "case_id": "valid-case",
                    "reference_source": "llm_valid",
                    "reference_points": [{"text": "有效经验", "evidence_chunk_ids": ["chunk-1"]}],
                },
                {
                    "case_id": "fallback-case",
                    "reference_source": "rule_fallback",
                    "reference_points": [{"text": "规则摘要", "evidence_chunk_ids": ["chunk-2"]}],
                },
            ],
            "warnings": [],
            "audit": {},
        })
        self.assertEqual([item["case_id"] for item in llm_service.references], ["valid-case"])
        self.assertEqual(result["audit"]["excluded_fallback_reference_count"], 1)
        self.assertTrue(any("已排除 1 个规则降级参考" in item for item in result["warnings"]))

    def test_season_has_priority_and_year_is_not_scored(self):
        """同月旧个例应排在跨季节的新个例之前。"""
        query = normalize_query({"start_date": "2026-07-10", "affected_areas": ["太原"], "top_n": 3})
        cases = [
            {"case_id": "old-same-season", "title": "暴雨过程", "affected_areas": ["太原"], "disaster_types": ["暴雨"], "date_range": "2020年7月10日"},
            {"case_id": "new-other-season", "title": "暴雨过程", "affected_areas": ["太原"], "disaster_types": ["暴雨"], "date_range": "2025年1月10日"},
        ]
        ranked = structured_recall(query, cases)
        self.assertEqual(ranked[0]["case_id"], "old-same-season")

    def test_image_numbering_matches_reference_labels(self):
        """主证据应按引用编号，未引用图片应进入不编号的补充资料。"""
        nodes = SmartCaseGraphNodes(self.store, None, None, None)
        points, images, supplemental = nodes._organize_case_evidence({"disaster_types": ["暴雨"]}, {}, [
            {"text": "历史个例累计降水分布"},
            {"text": "700hPa切变位置东移"},
        ], [
            {"image_id": "image-1", "caption": "图 2025年7月降水实况", "extraction_type": "embedded"},
            {"image_id": "image-2", "caption": "图2 700hPa环流形势", "extraction_type": "embedded"},
            {"image_id": "image-3", "caption": "图1 卫星云图", "extraction_type": "embedded"},
            {"image_id": "image-4", "caption": "图4 雷达组合反射率", "extraction_type": "embedded"},
            {"image_id": "image-5", "caption": "图5 地面气压场", "extraction_type": "embedded"},
            {"image_id": "image-6", "caption": "图6 能见度实况", "extraction_type": "embedded"},
        ])

        self.assertEqual([item["display_no"] for item in images], ["图1", "图2"])
        self.assertEqual(images[0]["caption"], "2025年7月降水实况")
        self.assertEqual(images[1]["caption"], "700hPa环流形势")
        self.assertEqual(points[0]["evidence_image_refs"][0]["label"], "图1")
        self.assertEqual(points[0]["evidence_image_refs"][0]["image_id"], "image-1")
        self.assertEqual(points[1]["evidence_image_refs"][0]["label"], "图2")
        self.assertEqual(points[1]["evidence_image_refs"][0]["image_id"], "image-2")
        self.assertEqual(len(supplemental), 4)
        self.assertTrue(all("display_no" not in item for item in supplemental))

    def test_circulation_reference_rejects_precipitation_image_from_same_chunk(self):
        """同一 chunk 的降水图不能替代真正支持环流经验的环流图。"""
        nodes = SmartCaseGraphNodes(self.store, None, None, None)
        point = {
            "text": "乌拉尔山阻塞高压形成，横槽转竖引导冷空气南下",
            "evidence_chunk_ids": ["FST2025-3-chunk-013"],
        }
        points, images, _ = nodes._organize_case_evidence(
            {"disaster_types": ["雨雪", "降雪"], "circulation_description": "阻塞高压和横槽共同影响"},
            {"title": "3月25-28日雨雪天气过程", "date_range": "2025年3月25-28日"},
            [point],
            [
                {"image_id": "FST2025-3-page-008-image-001", "caption": "图15 累计降水实况分布", "extraction_type": "embedded"},
                {"image_id": "FST2025-3-page-009-image-001", "caption": "图16 环流形势与海平面气压场演变", "extraction_type": "embedded"},
            ],
        )
        self.assertEqual([item["image_id"] for item in images], ["FST2025-3-page-009-image-001"])
        self.assertEqual(points[0]["evidence_image_refs"][0]["image_id"], "FST2025-3-page-009-image-001")

    def test_reference_point_rejects_chunk_without_matching_topic(self):
        """文字证据 ID 虽然合法，但正文不支持参考经验主题时必须删除该引用。"""
        nodes = SmartCaseGraphNodes(self.store, None, None, None)
        case = {
            "relevant_chunks": [{"chunk_id": "FST2025-3-chunk-013"}],
        }
        reference = {
            "case_id": "case-1",
            "reference_points": [{
                "text": "雷达回波呈现明显弓状结构",
                "evidence_chunk_ids": ["FST2025-3-chunk-013"],
            }],
        }
        validated = nodes._validate_reference_evidence(
            {"disaster_types": ["强对流"]},
            reference,
            case,
        )
        self.assertEqual(validated["reference_points"], [])

    def test_snow_query_rejects_generic_precipitation_image(self):
        """雪类查询只保留查询、个例和经验的交集证据，普通降水量图不能作为雪经验配图。"""
        nodes = SmartCaseGraphNodes(self.store, None, None, None)
        point = {
            "text": "过程受低层切变和东路冷空气共同影响，降水相态由雨转雪，北部出现积雪",
            "evidence_chunk_ids": ["FST2025-3-chunk-003"],
        }
        points, images, _ = nodes._organize_case_evidence(
            {
                "disaster_types": ["雨雪", "降雪"],
                "observation_description": "北中部出现降雪和积雪",
                "circulation_description": "低层切变配合东路冷空气",
            },
            {
                "title": "1-3日雨雪天气过程",
                "date_range": "2025年3月1-3日",
                "disaster_types": ["大风", "雨雪"],
            },
            [point],
            [{
                "image_id": "FST2025-3-page-003-image-001",
                "caption": "图1 3月1日20时-2日20时、2日20时-4日08时降水量分布图",
                "extraction_type": "embedded",
            }],
        )
        self.assertEqual(images, [])
        self.assertNotIn("evidence_image_refs", points[0])

    def test_text_evidence_must_belong_to_query_case_intersection(self):
        """雪类测试中的普通降水经验即使来自合法 chunk，也不能冒充三方交集参考经验。"""
        nodes = SmartCaseGraphNodes(self.store, None, None, None)
        case = {
            "title": "1-3日雨雪天气过程",
            "disaster_types": ["大风", "雨雪"],
            "relevant_chunks": [{"chunk_id": "FST2025-3-chunk-003"}],
        }
        reference = {
            "case_id": "FST2025-3-std-case-001",
            "reference_points": [{
                "text": "历史过程24小时累计降水量主要为0.1-17.9mm",
                "evidence_chunk_ids": ["FST2025-3-chunk-003"],
            }],
        }
        validated = nodes._validate_reference_evidence(
            {"disaster_types": ["雨雪", "降雪"]},
            reference,
            case,
        )
        self.assertEqual(validated["reference_points"], [])

    def test_chunk_detail_is_public_and_traceable(self):
        """chunk 弹窗接口只应返回正文溯源字段，不得包含向量或本地路径。"""
        agent = SmartCaseMatchAgent.__new__(SmartCaseMatchAgent)
        agent.store = self.store
        chunk_id = str(self.store.chunks[0]["chunk_id"])
        detail = agent.chunk_detail(chunk_id)
        self.assertEqual(detail["chunk_id"], chunk_id)
        self.assertTrue(detail["content"])
        self.assertNotIn("embedding", detail)
        self.assertNotIn("file_path", detail)
        with self.assertRaises(FileNotFoundError):
            agent.chunk_detail("missing-chunk")

    def test_agent_degrades_without_models(self):
        """模型全部不可用时仍应完成图执行，并且不暴露本地图片路径。"""
        client_progress_id = "unit_test_smart_case"
        # 端到端测试仍验证审计写入，但使用临时目录隔离真实运行日志。
        with tempfile.TemporaryDirectory() as audit_dir:
            agent = SmartCaseMatchAgent(
                embedding_client=UnavailableEmbeddingClient(),
                llm_client=UnavailableLlmClient(),
                audit_log_dir=Path(audit_dir),
            )
            response = agent.run(SmartCaseMatchRequest(
                process_name="7月太原强对流过程",
                start_date="2026-07-08",
                end_date="2026-07-09",
                disaster_types=["强对流", "雷暴大风"],
                affected_areas=["太原"],
                observation_description="太原出现雷暴大风和短时强降水",
                circulation_description="低层切变和暖湿气流共同影响",
                top_n=3,
                progress_id=client_progress_id,
            ))
        self.assertIn(response.status, {"completed", "degraded"})
        # 客户端关联号不能再成为运行主键，否则重复提交仍会覆盖进度和审计文件。
        self.assertNotEqual(response.run_id, client_progress_id)
        self.assertTrue(response.run_id.startswith("smart_case_"))
        self.assertLessEqual(len(response.matched_cases), 3)
        self.assertGreaterEqual(len(response.matched_cases), 1)
        serialized = response.model_dump_json()
        self.assertNotIn("resolved_path", serialized)
        self.assertNotIn("case_multidim_search\\data\\document_images", serialized)


if __name__ == "__main__":
    unittest.main()
