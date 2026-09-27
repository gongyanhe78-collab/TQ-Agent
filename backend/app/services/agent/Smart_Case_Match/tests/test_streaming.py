"""自然语言解析和分阶段流式输出测试。"""
from __future__ import annotations

import json
import asyncio
import tempfile
import unittest
from pathlib import Path

from ..agent import SmartCaseMatchAgent
from ..infrastructure.data_store import LocalCaseDataStore
from ..matching.matching import normalize_query, structured_recall
from ..matching.natural_query import normalize_natural_extraction
from ..schemas import NaturalLanguageMatchRequest
from ..infrastructure.stream_events import encode_sse


class UnavailableModelClient:
    """模拟不可用的 Embedding 或 Rerank 服务。"""

    def is_available(self) -> bool:
        return False


class StreamingLlmClient:
    """按不同提示返回可核验 JSON，避免单元测试调用外部模型。"""

    model = "stream-unit-test"

    def is_available(self) -> bool:
        return True

    def answer_with_context(self, question, context_blocks, max_tokens=1800):
        context = context_blocks[0]
        if context.startswith("DATA_BEGIN\n") and context.endswith("\nDATA_END"):
            context = context[len("DATA_BEGIN\n"):-len("\nDATA_END")]
        payload = json.loads(context)
        if "查询解析" in question:
            return json.dumps({
                "process_name": "太原7月暴雨过程",
                "start_date": "",
                "end_date": "",
                "date": "7月",
                "disaster_types": ["暴雨"],
                "affected_areas": ["太原"],
                "observation_description": "太原出现短时强降水",
                "circulation_description": "低层切变维持",
                "intensity_description": "最大小时雨强较大",
                "raw_query": "",
            }, ensure_ascii=False)
        if "重排" in question:
            case_ids = [str(item["case_id"]) for item in payload["候选个例"]]
            metrics = list(payload.get("匹配配置", {}).get("metrics") or [])
            return json.dumps({
                "ordered_case_ids": case_ids,
                "candidate_assessments": [
                    {
                        "case_id": case_id,
                        "mechanism_score": 0.8,
                        "intensity_score": 0.75,
                        # 模拟模型对业务细分指标逐项给分，验证后端不会只保留通用维度。
                        "metric_scores": {metric: 0.7 for metric in metrics[:2]},
                    }
                    for case_id in case_ids
                ],
            }, ensure_ascii=False)
        if "参考提炼" in question:
            case = payload["历史个例"]
            chunks = case.get("正文片段") or []
            chunk_id = str(chunks[0]["chunk_id"]) if chunks else ""
            return json.dumps({
                "match_reasons": ["低层切变位置接近，对降水落区判断具有参考意义"],
                "reference_points": [{"text": "历史过程低层切变维持并伴有水汽输送。", "evidence_chunk_ids": [chunk_id]}],
                "similarities": ["低层切变位置接近"],
                "differences": ["历史过程降水持续时间不同"],
                "warning_references": [],
            }, ensure_ascii=False)
        if "综合研判" in question:
            references = payload["逐例参考"]
            case_ids = [str(item["case_id"]) for item in references]
            chunk_ids = [
                str(point["evidence_chunk_ids"][0])
                for item in references
                for point in item.get("reference_points") or []
                if point.get("evidence_chunk_ids")
            ]
            return json.dumps({
                "forecast_summary": {
                    "similarity_assessment": "当前过程与历史暴雨过程较为相似。",
                    "core_features": ["低层切变维持"],
                    "main_risk": "短时强降水落区偏差。",
                    "confidence": 0.8,
                },
                "forecast_tips": [{
                    "priority": 1,
                    "focus_object": "太原短时强降水",
                    "possible_bias": "局地雨强可能低估",
                    "suggested_action": "结合雷达和自动站订正雨强",
                    "support_case_ids": case_ids,
                    "evidence_chunk_ids": chunk_ids,
                    "consensus_level": "多数个例共同支持",
                    "confidence": 0.8,
                }],
            }, ensure_ascii=False)
        return "{}"


class StreamingMatchTests(unittest.TestCase):
    """验证聊天入口复用原图，并按业务依赖顺序发送结果。"""

    @classmethod
    def setUpClass(cls) -> None:
        # 测试日志写入临时目录，避免自动化测试污染真实的人工复核审计目录。
        cls.audit_temp_dir = tempfile.TemporaryDirectory()
        cls.agent = SmartCaseMatchAgent(
            embedding_client=UnavailableModelClient(),
            rerank_client=UnavailableModelClient(),
            llm_client=StreamingLlmClient(),
            audit_log_dir=Path(cls.audit_temp_dir.name),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        # 测试结束显式关闭线程池，验证脚本场景不依赖进程退出回收资源。
        cls.agent.close()
        cls.audit_temp_dir.cleanup()

    def test_stream_returns_cases_before_forecast(self):
        """匹配个例必须先于综合研判发送，且参考经验保留文字证据。"""
        async def collect_events():
            return [
                event
                async for event in self.agent.stream_natural_match(NaturalLanguageMatchRequest(
                    message="7月太原出现暴雨，低层切变维持，最大小时雨强较大。",
                ))
            ]

        events = asyncio.run(collect_events())
        event_names = [name for name, _ in events]
        case_indexes = [index for index, name in enumerate(event_names) if name == "matched_cases"]
        forecast_index = event_names.index("forecast")

        self.assertEqual(event_names[0], "accepted")
        self.assertEqual(event_names[1], "parsed_query")
        # 个例卡片只在逐例理由和参考经验完成后发送一次，加载期间只显示阶段状态。
        self.assertEqual(len(case_indexes), 1)
        self.assertLess(case_indexes[-1], forecast_index)
        referenced = events[case_indexes[-1]][1]
        self.assertEqual(referenced["phase"], "referenced")
        self.assertTrue(referenced["matched_cases"])
        self.assertIn("低层切变位置接近", referenced["matched_cases"][0]["match_reasons"][0])
        self.assertTrue(referenced["matched_cases"][0]["key_references"][0]["evidence_chunk_ids"])
        self.assertTrue(referenced["matched_cases"][0]["candidate_assessment_available"])
        self.assertIn("mechanism", referenced["matched_cases"][0]["score_breakdown"])
        self.assertTrue(referenced["matched_cases"][0]["metric_scores"])
        self.assertEqual(events[-1][0], "completed")

    def test_selected_phase_hides_rule_reasons_behind_loading_state(self):
        """逐例LLM尚未完成时，页面应显示提取中，不能先展示容易误解的结构化通用理由。"""
        page = (Path(__file__).resolve().parents[1] / "pages" / "conversation_page.html").read_text(encoding="utf-8")
        self.assertIn("正在提取匹配理由", page)
        self.assertIn("<h4>匹配理由</h4>${matchReasons}", page)
        self.assertIn("匹配理由与参考经验生成中", page)

    def test_forecast_support_ids_are_mapped_to_case_names_in_page(self):
        """综合研判页面必须显示历史个例名称，不能直接暴露内部case_id。"""
        page = (Path(__file__).resolve().parents[1] / "pages" / "conversation_page.html").read_text(encoding="utf-8")
        self.assertIn("turn.caseDisplayNames", page)
        self.assertIn("supportNames.join", page)
        self.assertIn('|| "相关历史个例"', page)
        self.assertIn("可核验文字证据 ${evidenceCount} 条", page)

    def test_snow_query_recovers_standard_types_and_province_area(self):
        """模型误抽宽泛标签时，原文规则仍应恢复雨雪灾种和全省城市范围。"""
        message = (
            "2月21日至23日山西出现雨雪过程，北部基本为纯雪，中部先雨夹雪后转雪，"
            "南部前期以雨为主、后期转为雨夹雪，东部山区积雪较深。"
        )
        normalized = normalize_natural_extraction(message, {
            "process_name": "山西雨雪过程",
            "date": "2月21日至23日",
            "disaster_types": ["降水", "道路结冰"],
            "affected_areas": ["山西东部", "北部", "中部", "南部", "东部山区"],
            "raw_query": "",
        })

        self.assertEqual(normalized["disaster_types"], ["雨雪", "降雪"])
        self.assertEqual(normalized["affected_areas"][0], "全省")
        self.assertIn("北部", normalized["affected_areas"])
        self.assertIn("太原", normalized["affected_areas"])
        self.assertIn("运城", normalized["affected_areas"])
        self.assertEqual(normalized["raw_query"], message)

    def test_missing_year_uses_current_year_and_rejects_model_year(self):
        """用户未说年份时必须按当前年解析，不能采用模型幻觉年份。"""
        normalized = normalize_natural_extraction("1月14日太原有降雪", {
            "start_date": "2024-01-14",
            "end_date": "2024-01-14",
            "date": "",
            "disaster_types": ["降雪"],
            "affected_areas": ["太原"],
        })
        current_year = __import__("datetime").date.today().year
        self.assertEqual(normalized["start_date"], f"{current_year}-01-14")
        self.assertEqual(normalized["end_date"], f"{current_year}-01-14")

    def test_streaming_agent_emits_parsed_current_year(self):
        """异步SSE链路展示的解析日期也必须采用当前年。"""
        async def collect_parsed():
            async for event, data in self.agent.stream_natural_match(NaturalLanguageMatchRequest(
                message="1月14日太原有暴雨过程。",
            )):
                if event == "parsed_query":
                    return data
            return {}

        parsed = asyncio.run(collect_parsed())
        current_year = __import__("datetime").date.today().year
        self.assertEqual(parsed["query"]["start_date"], f"{current_year}-01-14")

    def test_natural_snow_query_has_same_qualified_count_as_structured_input(self):
        """同一雨雪过程的自然语言和结构化输入应至少得到相同数量的合格候选。"""
        message = "2月21日至23日，北部为纯雪，中部雨夹雪后转雪，南部由雨转雨夹雪。"
        natural = normalize_natural_extraction(message, {
            "date": "2月21日至23日",
            "disaster_types": [],
            "affected_areas": ["北部", "中部", "南部"],
            "raw_query": "",
        })
        structured = {
            "date": "2月21日至23日",
            "disaster_types": ["雨雪", "降雪"],
            "affected_areas": ["全省"],
        }
        store = LocalCaseDataStore()
        natural_ranked = structured_recall(normalize_query(natural), store.cases)
        structured_ranked = structured_recall(normalize_query(structured), store.cases)
        natural_ids = [item["case_id"] for item in natural_ranked if item["structured_score"] >= 0.30]
        structured_ids = [item["case_id"] for item in structured_ranked if item["structured_score"] >= 0.30]

        self.assertGreaterEqual(len(natural_ids), 3)
        self.assertEqual(natural_ids[:3], structured_ids[:3])

    def test_sse_encoder_keeps_chinese_content(self):
        """SSE 编码必须保留中文并形成完整事件边界。"""
        encoded = encode_sse("stage", {"message": "正在检索历史个例"})
        self.assertTrue(encoded.startswith("event: stage\n"))
        self.assertIn("正在检索历史个例", encoded)
        self.assertTrue(encoded.endswith("\n\n"))

    def test_conversation_page_only_uses_clickable_image_references(self):
        """聊天页面不能平铺证据图片或补充图片区，但必须保留图片点击入口。"""
        html = (Path(__file__).resolve().parents[1] / "pages" / "conversation_page.html").read_text(encoding="utf-8")
        self.assertIn('data-image-id', html)
        self.assertNotIn("补充资料图片（", html)
        self.assertNotIn("<h4>证据图片</h4>", html)


if __name__ == "__main__":
    unittest.main()
