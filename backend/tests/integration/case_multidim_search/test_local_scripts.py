"""公司知识库模式下的多维个例检索轻量回归测试。"""
from __future__ import annotations

import tempfile
import time
import sys
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from backend.app.models import DocumentChunk, StandardCase
from backend.app.services.agent.case_multidim_search import router
from backend.app.services.agent.case_multidim_search.agent import CaseMultidimSearchAgent
from backend.app.services.agent.case_multidim_search.reporting.chart_tool import ChartTool
from backend.app.services.agent.case_multidim_search.retrieval.case_chunk_analyzer import CaseChunkAnalyzer
from backend.app.services.agent.case_multidim_search.analysis.llm_intensity_extractor import LlmIntensityExtractor
from backend.app.services.agent.case_multidim_search.analysis.intensity import IntensityExtractor
from backend.app.services.agent.case_multidim_search.analysis.case_metric_pipeline import CaseMetricPipeline
from backend.app.services.agent.case_multidim_search.core.natural_query import (
    NaturalCaseQueryParser,
    NaturalConversationStore,
)
from backend.app.services.agent.case_multidim_search.integrations.non_thinking_llm import NonThinkingLlmClient
from backend.app.services.agent.case_multidim_search.analysis.report_analyzer import ReportAnalyzer
from backend.app.services.agent.case_multidim_search.analysis.report_llm_enhancer import ReportLlmEnhancer
from backend.app.services.agent.case_multidim_search.integrations.company_stores import (
    CompanyDocumentChunkStore,
    CompanyImageEvidenceStore,
    CompanyStandardCaseStore,
)
from backend.app.services.agent.case_multidim_search.schemas import (
    CaseSearchHit,
    CaseSearchQuery,
    IntensityMetric,
    StructuredCaseSearchRequest,
)
from backend.app.services.agent.case_multidim_search.retrieval.structured_retriever import StructuredCaseRetriever


class FakeCompanyKbClient:
    """模拟公司知识库接口，避免单元测试访问真实网络。"""

    base_url = "http://company.example"

    def __init__(self):
        self.cases = [
            {
                "case_id": "case-001",
                "title": "5月16日雷暴大风天气过程",
                "date_range": "2025年5月16日",
                "disaster_types": ["雷暴大风", "强对流"],
                "affected_areas": ["太原", "大同"],
                "source_pdf": "FST2025-5.docx",
                "source_chunk_ids": ["chunk-001"],
                "evidence_image_ids": ["fig-001"],
            }
        ]
        self.chunks = {
            "chunk-001": {
                "chunk_id": "chunk-001",
                "text": "太原和大同出现雷暴大风，极大风速达到30.7m/s。",
                "metadata": {"source_file": "FST2025-5.docx", "chunk_index": 12},
            }
        }
        self.images = [
            {
                "image_id": "fig-001",
                "source_pdf": "FST2025-5.docx",
                "page_no": 3,
                "caption": "图 1 雷暴大风实况分布",
                "image_url": "/assets/figures/fig-001.png",
                "related_chunk_ids": ["chunk-001"],
            }
        ]

    def list_standard_cases(self, params=None):
        """返回模拟标准个例。"""
        return self.cases

    def get_chunk(self, chunk_id):
        """返回模拟 chunk。"""
        return self.chunks.get(chunk_id)

    def list_image_metadata(self, params=None):
        """返回模拟图片元数据。"""
        return self.images

    def status(self):
        """返回模拟知识库状态。"""
        return {"total_chunks": len(self.chunks), "index_ready": True}

    def absolute_url(self, value):
        """拼接模拟图片 URL。"""
        return self.base_url + value if value.startswith("/") else value




class FakeLlmClient:
    """模拟大模型结构化补抽返回，避免测试依赖真实本地模型。"""

    def is_available(self):
        """模拟模型服务可用。"""
        return True

    def answer_with_context(self, question, context_blocks, max_tokens=1200):
        """返回强对流降水指标 JSON。"""
        return (
            '{"metrics": ['
            '{"metric_name":"最大小时雨强","value":38.7,"unit":"mm/h","location":"阳泉",'
            '"relation":"最大","source_chunk_id":"chunk-rain","source_text":"阳泉站最大小时雨强达38.7mm/h。","confidence":0.92},'
            '{"metric_name":"过程最大降水量","value":85.2,"unit":"mm","location":"大同",'
            '"relation":"过程最大","source_chunk_id":"chunk-rain","source_text":"大同过程最大降水量为85.2mm。","confidence":0.91}'
            ']}'
        )


class FakeContextLlmClient:
    """模拟结合已确认条件判断上下文关系的大模型。"""

    def is_available(self):
        """模拟上下文分类模型可用。"""
        return True

    def answer_with_context(self, question, context_blocks, max_tokens=260):
        """根据本轮自然语言返回高置信上下文操作。"""
        context = "\n".join(context_blocks)
        if "太原和晋中也需要纳入范围" in context:
            return '{"operation":"restrict","confidence":0.94,"reason":"承接上一轮并收紧地区"}'
        if "冰雹相关的留下" in context:
            return '{"operation":"scope","confidence":0.91,"reason":"收缩灾种范围"}'
        return '{"operation":"new","confidence":0.88,"reason":"独立完整查询"}'


class CaseMultidimCompanyApiTests(unittest.TestCase):
    """验证当前模块只依赖公司 API 适配层仍能保持原业务流程。"""

    def test_router_only_exposes_structured_search_entries(self):
        """接口层同时暴露结构化入口和复用该流程的自然语言入口。"""
        paths = {route.path for route in router.router.routes}

        self.assertIn("/api/case-multidim/search", paths)
        self.assertIn("/api/case-multidim/natural-chat", paths)
        self.assertIn("/api/case-multidim/natural/parse", paths)
        self.assertIn("/api/case-multidim/natural/search", paths)
        self.assertNotIn("/api/case-multidim/query", paths)
        self.assertNotIn("/api/case-multidim/analyze", paths)

    def test_natural_query_context_requires_confirmation_and_explicit_operation(self):
        """只有确认后的条件可继承，普通新问题不得隐式沿用历史。"""
        parser = NaturalCaseQueryParser(None)
        with tempfile.TemporaryDirectory() as temp_dir:
            store = NaturalConversationStore(Path(temp_dir))
            first = store.propose(
                "",
                "检索2025年1月至5月山西强对流、暴雨和冰雹过程，重点关注太原、晋中和吕梁",
                parser,
            )
            self.assertTrue(first.can_confirm)
            self.assertEqual([2025], first.parsed_request.years)
            self.assertEqual([1, 2, 3, 4, 5], first.parsed_request.months)
            self.assertEqual(["强对流", "暴雨", "冰雹"], first.parsed_request.disaster_types)
            self.assertEqual(["太原", "晋中", "吕梁"], first.parsed_request.cities)

            committed, _ = store.confirm(first.conversation_id, first.proposal_id, "progress-first")
            self.assertEqual("progress-first", committed.progress_id)

            pending = store.propose(first.conversation_id, "再限定太原和晋中", parser)
            self.assertEqual("restrict", pending.operation)
            self.assertEqual(["太原", "晋中"], pending.parsed_request.cities)
            self.assertEqual(["强对流", "暴雨", "冰雹"], pending.parsed_request.disaster_types)

            # 上一条限定尚未确认，本轮必须继续从 first 的已确认条件派生。
            changed_time = store.propose(first.conversation_id, "把时间改成5月至7月", parser)
            self.assertEqual("replace", changed_time.operation)
            self.assertEqual([2025], changed_time.parsed_request.years)
            self.assertEqual([5, 6, 7], changed_time.parsed_request.months)
            self.assertEqual(["太原", "晋中", "吕梁"], changed_time.parsed_request.cities)

            fresh = store.propose(first.conversation_id, "查询6月大风过程", parser)
            self.assertEqual("new", fresh.operation)
            self.assertEqual([6], fresh.parsed_request.months)
            self.assertEqual(["大风"], fresh.parsed_request.disaster_types)
            self.assertEqual([], fresh.parsed_request.cities)
            self.assertFalse(any("未继承" in warning for warning in fresh.warnings))

    def test_natural_query_uses_semantic_context_without_fixed_operation_words(self):
        """自然语言追问不含固定词时，仍可结合已确认条件识别补充和收缩。"""
        parser = NaturalCaseQueryParser(FakeContextLlmClient())
        with tempfile.TemporaryDirectory() as temp_dir:
            store = NaturalConversationStore(Path(temp_dir))
            first = store.propose("", "检索2025年5月至7月强对流、暴雨和冰雹过程，关注吕梁", parser)
            store.confirm(first.conversation_id, first.proposal_id, "progress-first")

            region_followup = store.propose(first.conversation_id, "太原和晋中也需要纳入范围", parser)
            self.assertEqual("restrict", region_followup.operation)
            self.assertEqual(["太原", "晋中"], region_followup.parsed_request.cities)
            self.assertEqual(["强对流", "暴雨", "冰雹"], region_followup.parsed_request.disaster_types)

            disaster_followup = store.propose(first.conversation_id, "冰雹相关的留下，其他先不看", parser)
            self.assertEqual("scope", disaster_followup.operation)
            self.assertEqual(["冰雹"], disaster_followup.parsed_request.disaster_types)
            self.assertEqual(["吕梁"], disaster_followup.parsed_request.cities)

    def test_natural_query_scope_and_stale_proposal_protection(self):
        """“只看”收缩灾种，且新提案生成后旧确认卡必须失效。"""
        parser = NaturalCaseQueryParser(None)
        with tempfile.TemporaryDirectory() as temp_dir:
            store = NaturalConversationStore(Path(temp_dir))
            first = store.propose("", "检索2025年5月强对流和暴雨过程，关注太原", parser)
            store.confirm(first.conversation_id, first.proposal_id, "progress-first")

            stale = store.propose(first.conversation_id, "只看其中有冰雹的过程", parser)
            self.assertEqual("scope", stale.operation)
            self.assertEqual(["冰雹"], stale.parsed_request.disaster_types)
            self.assertEqual([5], stale.parsed_request.months)
            self.assertEqual(["太原"], stale.parsed_request.cities)

            current = store.propose(first.conversation_id, "把时间改成5月至7月", parser)
            with self.assertRaisesRegex(ValueError, "失效"):
                store.confirm(first.conversation_id, stale.proposal_id, "progress-stale")
            confirmed, _ = store.confirm(first.conversation_id, current.proposal_id, "progress-current")
            self.assertEqual([2025], confirmed.years)
            self.assertEqual([5, 6, 7], confirmed.months)

    def test_natural_query_clears_previous_region_when_city_is_unlimited(self):
        """“地市不限”必须清空上轮区域，不能留下隐藏的硬过滤条件。"""
        parser = NaturalCaseQueryParser(None)
        with tempfile.TemporaryDirectory() as temp_dir:
            store = NaturalConversationStore(Path(temp_dir))
            first = store.propose("", "检索2025年3月山西北部暴雪过程", parser)
            store.confirm(first.conversation_id, first.proposal_id, "progress-first")

            revised = store.propose(first.conversation_id, "改成2025年3月暴雪，地市不限", parser)

            self.assertEqual("replace", revised.operation)
            self.assertEqual([2025], revised.parsed_request.years)
            self.assertEqual([3], revised.parsed_request.months)
            self.assertEqual(["暴雪"], revised.parsed_request.disaster_types)
            self.assertEqual([], revised.parsed_request.cities)
            self.assertEqual([], revised.parsed_request.areas)

    def test_company_standard_cases_keep_structured_filter_semantics(self):
        """公司标准个例进入本地筛选后，灾种和区域多选仍保持 OR 语义。"""
        store = CompanyStandardCaseStore(FakeCompanyKbClient())
        cases = store.list_cases()
        self.assertIsInstance(cases[0], StandardCase)

        query = CaseSearchQuery(months=[5], disaster_types=["强降水", "雷暴大风"], cities=["阳泉", "大同"])
        matches = StructuredCaseRetriever().search(cases, query)

        self.assertEqual(1, len(matches))
        self.assertIn("disaster", matches[0].matched_fields)
        self.assertIn("region", matches[0].matched_fields)

    def test_company_chunk_store_reads_chunks_by_source_ids(self):
        """文档 store 按 source_chunk_ids 精确读取公司接口 chunk。"""
        chunks = CompanyDocumentChunkStore(FakeCompanyKbClient()).get_chunks(["chunk-001"])

        self.assertEqual(1, len(chunks))
        self.assertEqual("chunk-001", chunks[0].chunk_id)
        self.assertIn("30.7", chunks[0].content)

    def test_case_chunks_trim_shared_boundaries_before_metric_extraction(self):
        """共享首尾 chunk 必须只保留当前标题到下一标题之间的正文。"""
        class ChunkStore:
            def __init__(self, chunks):
                self.chunks = {chunk.chunk_id: chunk for chunk in chunks}

            def get_chunks(self, chunk_ids):
                return [self.chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunks]

        class CaseStore:
            def __init__(self, cases):
                self.cases = cases

            def list_cases(self):
                return list(self.cases)

        previous = StandardCase.from_dict({
            "case_id": "case-previous",
            "title": "7月6日暴雨过程",
            "date_range": "2025年7月6日",
            "source_pdf": "case.pdf",
            "source_chunk_ids": ["chunk-1"],
        })
        current = StandardCase.from_dict({
            "case_id": "case-current",
            "title": "7月7-8日暴雨过程",
            "date_range": "2025年7月7-8日",
            "source_pdf": "case.pdf",
            "source_chunk_ids": ["chunk-1", "chunk-2", "chunk-3"],
        })
        following = StandardCase.from_dict({
            "case_id": "case-following",
            "title": "7月9日高温天气过程",
            "date_range": "2025年7月9日",
            "source_pdf": "case.pdf",
            "source_chunk_ids": ["chunk-3"],
        })
        chunks = [
            DocumentChunk(
                source_pdf="case.pdf",
                chunk_id="chunk-1",
                chunk_no=1,
                content="上一例过程最大降水量303mm。二、7月7～8日暴雨过程1 实况：当前过程最大降水量101.6mm。",
            ),
            DocumentChunk(
                source_pdf="case.pdf",
                chunk_id="chunk-2",
                chunk_no=2,
                content="当前过程最大小时雨强38.7mm/h。",
            ),
            DocumentChunk(
                source_pdf="case.pdf",
                chunk_id="chunk-3",
                chunk_no=3,
                content="当前过程服务复盘结束。三、7月9日高温天气过程1 实况：最高气温46.8℃。",
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = CaseChunkAnalyzer(
                ChunkStore(chunks),
                None,
                Path(temp_dir),
                standard_case_store=CaseStore([previous, current, following]),
            )
            sliced = analyzer.load_case_chunks(current)

        content = "".join(chunk.content for chunk in sliced)
        self.assertIn("7月7～8日暴雨过程", content)
        self.assertIn("101.6mm", content)
        self.assertIn("38.7mm/h", content)
        self.assertIn("当前过程服务复盘结束", content)
        self.assertNotIn("303mm", content)
        self.assertNotIn("7月9日高温天气过程", content)
        self.assertNotIn("46.8℃", content)

    def test_case_chunk_boundary_prefers_chapter_title_over_monthly_catalog(self):
        """目录中的同名过程不能截断正式章节正文，必须优先识别带章节编号的标题。"""
        class ChunkStore:
            def __init__(self, chunks):
                self.chunks = {chunk.chunk_id: chunk for chunk in chunks}

            def get_chunks(self, chunk_ids):
                return [self.chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunks]

        class CaseStore:
            def __init__(self, cases):
                self.cases = cases

            def list_cases(self):
                return list(self.cases)

        current = StandardCase.from_dict({
            "case_id": "case-current", "title": "4月11-13日极端大风过程", "date_range": "2025年4月11-13日",
            "source_pdf": "case.pdf", "source_chunk_ids": ["chunk-1", "chunk-2"],
        })
        following = StandardCase.from_dict({
            "case_id": "case-following", "title": "4月16-18日大风过程", "date_range": "2025年4月16-18日",
            "source_pdf": "case.pdf", "source_chunk_ids": ["chunk-2"],
        })
        chunks = [
            DocumentChunk(source_pdf="case.pdf", chunk_id="chunk-1", chunk_no=1, content="本月过程包括4月11-13日极端大风过程、4月16-18日大风过程。一、4月11-13日极端大风过程实况最大阵风39.7m/s（13级）。"),
            DocumentChunk(source_pdf="case.pdf", chunk_id="chunk-2", chunk_no=2, content="过程服务结束。二、4月16-18日大风过程实况最大风速34.5m/s。"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = CaseChunkAnalyzer(
                ChunkStore(chunks), None, Path(temp_dir), standard_case_store=CaseStore([current, following])
            )
            sliced = analyzer.load_case_chunks(current)

        content = "".join(chunk.content for chunk in sliced)
        self.assertIn("39.7m/s", content)
        self.assertIn("过程服务结束", content)
        self.assertNotIn("34.5m/s", content)

    def test_company_image_store_keeps_image_id_order(self):
        """图片 store 按 evidence_image_ids 返回图片元数据并保持顺序。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = CompanyImageEvidenceStore(FakeCompanyKbClient(), Path(temp_dir))
            images = store.list_by_image_ids(["fig-001"])

        self.assertEqual(1, len(images))
        self.assertEqual("fig-001", images[0].image_id)
        self.assertIn("雷暴大风", images[0].caption)

    def test_case_focus_disaster_keeps_matches_and_uses_title_for_single_focus(self):
        """检索命中的灾种全部保留，单个分析视角按检索顺序稳定选择。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = CaseMultidimSearchAgent(None, None, CompanyImageEvidenceStore(FakeCompanyKbClient(), Path(temp_dir)), Path(temp_dir))
            case = StandardCase.from_dict(
                {
                    "case_id": "case-rainstorm",
                    "title": "5月21~22日暴雨天气过程",
                    "date_range": "2025年5月21日至22日",
                    "disaster_types": ["强对流", "暴雨", "强降水", "短时强降水", "大风"],
                }
            )
            query = CaseSearchQuery(disaster_types=["强对流", "暴雨", "强降水"])

            self.assertEqual(["强对流", "暴雨", "强降水"], agent._case_matched_disasters(case, query))
            self.assertEqual("暴雨", agent._case_focus_disaster(case, query))

    def test_case_analysis_images_match_parallel_subfigures(self):
        """正文并列引用图15(a)、图15(b)时，两张子图都应展示。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = CaseMultidimSearchAgent(None, None, CompanyImageEvidenceStore(FakeCompanyKbClient(), Path(temp_dir)), Path(temp_dir))
            images = [
                {"image_id": "fig-15a", "caption": "图 15（a） 24日20时雷达组合反射率"},
                {"image_id": "fig-15b", "caption": "图 15（b） 25日02时雷达组合反射率"},
                {"image_id": "fig-16", "caption": "图 16 地面大风实况"},
            ]
            analysis = "本次过程需结合图15(a)、图15(b)判断回波组织演变。"

            selected = agent._images_referenced_by_analysis(analysis, images, max_count=8)

            self.assertEqual(["fig-15a", "fig-15b"], [item["image_id"] for item in selected])
    def test_case_focus_disaster_auto_assigns_when_query_has_no_disaster(self):
        """未勾选灾种时，个例仍按标题优先自动归属到一个标准灾种。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = CaseMultidimSearchAgent(None, None, CompanyImageEvidenceStore(FakeCompanyKbClient(), Path(temp_dir)), Path(temp_dir))
            case = StandardCase.from_dict(
                {
                    "case_id": "case-snow-cold",
                    "title": "1月23-26日雨雪寒潮大风天气过程",
                    "date_range": "2025年1月23日至26日",
                    "disaster_types": ["强降水", "大风", "雨雪", "降雪", "暴雪", "寒潮", "低温", "雾"],
                }
            )

            self.assertEqual("雨雪", agent._case_focus_disaster(case, CaseSearchQuery()))

    def test_chart_specs_use_horizontal_frequency_and_low_temperature_bars(self):
        """灾种频次和最低气温图使用水平柱状图，共现热力图按频次重排。"""
        chart_tool = ChartTool()
        aggregations = {
            "case_count": 3,
            "disaster_counts": {"雾": 1, "大风": 3, "降雪": 2, "寒潮": 2},
            "disaster_cooccurrence": {
                "labels": ["雾", "大风", "降雪", "寒潮"],
                "matrix": [[1, 1, 1, 1], [1, 3, 2, 2], [1, 2, 2, 1], [1, 2, 1, 2]],
            },
            "intensity_case_values": {
                "最低气温": [
                    {"case_id": "one", "title": "1月23-26日雨雪寒潮大风过程", "value": -30.7, "unit": "℃"},
                    {"case_id": "two", "title": "1月14-15日寒潮天气过程", "value": -25.6, "unit": "℃"},
                    {"case_id": "three", "title": "1月11日降雪天气过程", "value": -24.3, "unit": "℃"},
                ],
            },
        }

        specs = chart_tool.select_specs(aggregations, CaseSearchQuery(disaster_types=["低温"]))
        frequency = next(spec for spec in specs if spec.chart_key == "disaster_frequency")
        heatmap = next(spec for spec in specs if spec.chart_key == "disaster_cooccurrence")
        low_temperature = next(spec for spec in specs if spec.chart_key == "intensity_low_temperature")

        self.assertEqual("horizontal_bar", frequency.chart_type)
        self.assertEqual(["大风", "寒潮", "降雪", "雾"], frequency.labels)
        self.assertEqual(["大风", "寒潮", "降雪", "雾"], heatmap.row_labels)
        self.assertEqual("horizontal_bar", low_temperature.chart_type)






    def test_llm_intensity_extractor_enriches_missing_rain_metrics(self):
        """LLM 补抽能把规则漏掉的 5 月强对流雨强和过程雨量纳入结构化指标。"""
        case = StandardCase.from_dict(
            {
                "case_id": "case-rain",
                "title": "5月21~22日暴雨天气过程",
                "date_range": "2025年5月21日至22日",
                "disaster_types": ["强对流", "大暴雨", "强降水"],
                "source_chunk_ids": ["chunk-rain"],
            }
        )
        chunks = [
            DocumentChunk(
                source_pdf="rain.pdf",
                chunk_id="chunk-rain",
                chunk_no=1,
                content="阳泉站最大小时雨强达38.7mm/h。大同过程最大降水量为85.2mm。",
            )
        ]

        metrics = LlmIntensityExtractor(FakeLlmClient()).enrich(
            case,
            chunks,
            [],
            disaster_names=["强对流", "大暴雨", "强降水"],
            enabled=True,
        )

        values = {(metric.metric_name, metric.value, metric.unit) for metric in metrics}
        self.assertIn(("最大小时雨强", 38.7, "mm/h"), values)
        self.assertIn(("过程最大降水量", 85.2, "mm"), values)

    def test_rule_intensity_uses_regional_station_hourly_precip_extreme(self):
        """同个例同时出现国家站和区域站小时雨强时，汇总表应取更大的区域站极值。"""
        text = (
            "\u533a\u57df\u7ad9 45 \u7ad9\u66b4\u96e8\uff0c\u6700\u5927\u6c81\u6c34\u5409\u5bb685.2mm\u3002"
            "\u8fc7\u7a0b\u671f\u95f4\u6709\u77ed\u65f6\u5f3a\u964d\u6c34\uff0c\u5176\u4e2d\u56fd\u5bb6\u7ad9 6 \u7ad9\u3001\u533a\u57df\u7ad9 123 \u7ad9\u3002"
            "\u56fd\u5bb6\u7ad9\u6700\u5927\u8944\u6c7e 27.2mm/h\uff1b\u533a\u57df\u7ad9\u6700\u5927\u76d0\u6e56\u51e4\u51f0\u8c3738.7mm/h\uff1b"
        )
        case = StandardCase.from_dict(
            {
                "case_id": "case-rain-extreme",
                "title": "5\u670821~22\u65e5\u66b4\u96e8\u5929\u6c14\u8fc7\u7a0b",
                "date_range": "2025\u5e745\u670821\u65e5\u81f322\u65e5",
                "disaster_types": ["\u5f3a\u5bf9\u6d41", "\u66b4\u96e8", "\u77ed\u65f6\u5f3a\u964d\u6c34"],
                "source_chunk_ids": ["chunk-rain-extreme"],
            }
        )
        metrics = IntensityExtractor().extract(
            case,
            [DocumentChunk(source_pdf="rain.pdf", chunk_id="chunk-rain-extreme", chunk_no=1, content=text)],
        )

        metric = ReportAnalyzer()._case_metric_extreme(metrics, {"\u6700\u5927\u5c0f\u65f6\u96e8\u5f3a"})

        self.assertIsNotNone(metric)
        self.assertEqual(38.7, metric.value)

    def test_monthly_precip_overview_is_not_used_as_case_metric(self):
        """月度气象概况中的全省降水范围不能串入单个个例强度汇总。"""
        case = StandardCase.from_dict(
            {
                "case_id": "case-monthly-overview",
                "title": "5\u670821~22\u65e5\u66b4\u96e8\u5929\u6c14\u8fc7\u7a0b",
                "date_range": "2025\u5e745\u670821\u65e5\u81f322\u65e5",
                "disaster_types": ["\u66b4\u96e8"],
                "source_chunk_ids": ["chunk-monthly-overview"],
            }
        )
        text = "2025\u5e745\u6708\uff0c\u5c71\u897f\u7701\u964d\u6c34\u91cf\u4ecb\u4e8e9.6\uff5e143.6\u6beb\u7c73\uff0c\u8f83\u5e38\u5e74\u540c\u671f\u504f\u591a\u3002"

        metrics = IntensityExtractor().extract(
            case,
            [DocumentChunk(source_pdf="overview.pdf", chunk_id="chunk-monthly-overview", chunk_no=1, content=text)],
        )

        self.assertEqual([], metrics)

    def test_precip_process_without_title_hazard_uses_first_query_match(self):
        """标题未明确灾种时，从实际命中的检索灾种中按查询顺序选择唯一主导灾种。"""
        case = StandardCase.from_dict(
            {
                "case_id": "case-wide-precip",
                "title": "6\u670813-14\u65e5\u5927\u8303\u56f4\u964d\u6c34\u8fc7\u7a0b",
                "date_range": "2025\u5e746\u670813\u65e5\u81f314\u65e5",
                "disaster_types": ["\u5f3a\u5bf9\u6d41", "\u77ed\u65f6\u5f3a\u964d\u6c34", "\u5f3a\u964d\u6c34"],
            }
        )
        query = CaseSearchQuery(disaster_types=["\u5f3a\u5bf9\u6d41", "\u77ed\u65f6\u5f3a\u964d\u6c34", "\u5f3a\u964d\u6c34"])

        self.assertEqual("\u5f3a\u5bf9\u6d41", ReportAnalyzer()._primary_disaster_text(case, query))


    def test_rule_intensity_uses_range_extremes_and_ignores_wind_as_visibility(self):
        """范围实况应提取正确极值，且 m/s 不能误解为能见度。"""
        case = StandardCase.from_dict({"case_id": "case-range-extremes", "date_range": "2025-07-28 to 31", "title": "7月28日至31日高温、沙尘过程", "disaster_types": ["高温", "沙尘"], "source_chunk_ids": ["chunk-range-extremes"]})
        content = "最高气温介于35-46.8℃之间。25-28日大风、沙尘天气持续四天，最大五台山18.4m/s，最低能见度25日1.3km，26日0.6km，27日2.86km，28日0.96km。"
        metrics = IntensityExtractor().extract(case, [DocumentChunk(source_pdf="range.pdf", chunk_id="chunk-range-extremes", chunk_no=1, content=content)])
        analyzer = ReportAnalyzer()

        self.assertEqual(46.8, analyzer._case_metric_extreme(metrics, {"最高气温"}).value)
        visibility = analyzer._case_metric_extreme(metrics, {"最低能见度"}, prefer_min=True)
        self.assertEqual((0.6, "km"), (visibility.value, visibility.unit))

    def test_rule_intensity_uses_precip_and_snow_depth_upper_bounds(self):
        """过程降水与积雪深度应取当个例原文中的最大值。"""
        case = StandardCase.from_dict({"case_id": "case-precip-snow", "date_range": "2025-07-02 to 03", "title": "7月2-3日暴雨和降雪过程", "disaster_types": ["暴雨", "降雪"], "source_chunk_ids": ["chunk-precip-snow"]})
        content = "全省累计降水量为0.1～150.4毫米，25日08时观测积雪深度≧5cm的有8站，最大为10cm（平陆）。"
        metrics = IntensityExtractor().extract(case, [DocumentChunk(source_pdf="precip-snow.pdf", chunk_id="chunk-precip-snow", chunk_no=1, content=content)])
        analyzer = ReportAnalyzer()

        self.assertEqual(150.4, analyzer._case_metric_extreme(metrics, {"过程最大降水量"}).value)
        self.assertEqual(10, analyzer._case_metric_extreme(metrics, {"最大积雪深度"}).value)

    def test_case_analysis_images_require_text_reference_and_follow_text_order(self):
        """只展示正文明确引用的图，并按正文引用顺序排列。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = CaseMultidimSearchAgent(None, None, CompanyImageEvidenceStore(FakeCompanyKbClient(), Path(temp_dir)), Path(temp_dir))
            images = [{"image_id": "fig-1", "caption": "图 1 高温实况"}, {"image_id": "fig-3", "caption": "图 3 雷达回波演变"}, {"image_id": "fig-17", "caption": "图 17 探空曲线"}]
            analysis = "图3显示强对流回波演变，图1显示高温实况。"
            _, selected = agent._align_case_analysis_images(analysis, images, cited_image_ids=["fig-17"])

        self.assertEqual(["fig-3", "fig-1"], [item["image_id"] for item in selected])

    def test_case_analysis_recovers_json_wrapped_text(self):
        """模型返回 JSON 协议时，页面只显示 analysis 正文。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = CaseChunkAnalyzer(None, None, Path(temp_dir))
            result = analyzer._extract_analysis_text('{"analysis":"寒潮过程最低气温达-15℃。","cited_image_ids":[]}')

        self.assertEqual("寒潮过程最低气温达-15℃。", result)


    def test_case_analysis_removes_unmatched_figure_reference(self):
        """正文虚构候选列表外图号时，页面不保留该句或展示错误图片。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = CaseMultidimSearchAgent(
                None,
                None,
                CompanyImageEvidenceStore(FakeCompanyKbClient(), Path(temp_dir)),
                Path(temp_dir),
            )
            analysis, selected = agent._align_case_analysis_images(
                "图34显示错误过程。图14显示降雪实况。",
                [{"image_id": "fig-14", "caption": "图14 降雪实况"}],
            )

        self.assertNotIn("图34", analysis)
        self.assertEqual(["fig-14"], [item["image_id"] for item in selected])

    def test_case_analysis_removes_open_ended_single_figure_reference(self):
        """“可参考图N...”没有完整图文关系时，应删除引用并且不再展示该图。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = CaseMultidimSearchAgent(None, None, None, Path(temp_dir))
            analysis, selected = agent._align_case_analysis_images(
                "可参考图14...高空槽东移，冷空气随之南下。",
                [{"image_id": "fig-14", "caption": "图14 高空环流形势"}],
                cited_image_ids=["fig-14"],
            )

        self.assertEqual("高空槽东移，冷空气随之南下。", analysis)
        self.assertEqual([], selected)

    def test_case_analysis_keeps_only_explicit_figures_before_open_ended_list(self):
        """并列图号后的省略项必须删除，能够明确匹配的图号和图片仍然保留。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = CaseMultidimSearchAgent(None, None, None, Path(temp_dir))
            analysis, selected = agent._align_case_analysis_images(
                "图14、图15、...共同显示降雪落区由北向南扩展。",
                [
                    {"image_id": "fig-14", "caption": "图14 前期降雪实况"},
                    {"image_id": "fig-15", "caption": "图15 后期降雪实况"},
                ],
            )

        self.assertEqual("图14、图15共同显示降雪落区由北向南扩展。", analysis)
        self.assertEqual(["fig-14", "fig-15"], [item["image_id"] for item in selected])

    def test_case_analysis_keeps_complete_figure_reference(self):
        """有明确支撑关系的正常图号引用不能被开放式引用清理误删。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = CaseMultidimSearchAgent(None, None, None, Path(temp_dir))
            analysis, selected = agent._align_case_analysis_images(
                "图14显示北部降雪中心与积雪高值区基本一致。",
                [{"image_id": "fig-14", "caption": "图14 降雪实况"}],
            )

        self.assertEqual("图14显示北部降雪中心与积雪高值区基本一致。", analysis)
        self.assertEqual(["fig-14"], [item["image_id"] for item in selected])

    def test_intensity_ignores_image_nearby_text(self):
        """图片附近文字即使包含数值，也不能污染当前个例的强度汇总。"""
        case = StandardCase.from_dict(
            {
                "case_id": "case-source-isolation",
                "title": "5月2日大风过程",
                "date_range": "2025年5月2日",
                "source_chunk_ids": ["chunk-source"],
            }
        )
        metrics = IntensityExtractor().extract(
            case,
            [
                DocumentChunk(
                    source_pdf="case.pdf",
                    chunk_id="chunk-source",
                    chunk_no=1,
                    content="最大风速为27.4m/s。",
                ),
                DocumentChunk(
                    source_pdf="other-case.pdf",
                    chunk_id="image-nearby::foreign-image",
                    chunk_no=2,
                    content="过程最大降水量为77.9mm。",
                ),
            ],
        )

        values = {(metric.metric_name, metric.value) for metric in metrics}
        self.assertIn(("最大风速", 27.4), values)
        self.assertNotIn(("过程最大降水量", 77.9), values)

    def test_llm_intensity_rejects_metric_without_matching_source_text(self):
        """大模型返回的数值必须能在指定个例原文中逐字核验。"""
        extractor = LlmIntensityExtractor(None)
        chunks = [
            DocumentChunk(
                source_pdf="case.pdf",
                chunk_id="chunk-source",
                chunk_no=1,
                content="局部出现大风，最大值27.4m/s。",
            )
        ]
        metric = extractor._metric_from_payload(
            {
                "metric_name": "最大风速",
                "value": 43.4,
                "unit": "m/s",
                "source_chunk_id": "chunk-source",
                "source_text": "最大风速为43.4m/s。",
            },
            {"chunk-source"},
            chunks,
        )

        self.assertIsNone(metric)

    def test_pdf_subtitle_uses_separate_filter_lines(self):
        """PDF 副标题按月份、灾种和地市分行展示检索条件。"""
        from types import SimpleNamespace

        from backend.app.services.agent.case_multidim_search.reporting.pdf_report import PdfReportBuilder

        response = SimpleNamespace(
            parsed_query=CaseSearchQuery(
                months=[1, 2, 5, 6],
                disaster_types=["强对流", "大暴雨", "雷暴", "暴雪"],
                cities=["太原", "大同", "忻州", "阳泉"],
            ),
            question="",
        )

        self.assertEqual(
            "检索条件：\n月份：1月、2月、5月、6月\n灾种：强对流、大暴雨、雷暴、暴雪\n地市：太原、大同、忻州、阳泉",
            PdfReportBuilder()._report_subtitle(response),
        )

    def test_wind_level_uses_maximum_speed_conversion(self):
        """阵风等级始终按最大风速折算，并覆盖原文中的较低门槛等级。"""
        analyzer = ReportAnalyzer()
        metrics = [
            IntensityMetric(metric_name="最大风速", value=43.4, unit="m/s", source_text="最大值43.4m/s。"),
            IntensityMetric(metric_name="阵风风力", value=12, unit="级", source_text="12级以上有6站。"),
        ]

        level = analyzer._case_metric_extreme(metrics, {"阵风风力"})

        self.assertEqual(14, level.value)
        self.assertEqual(13, analyzer._wind_level_from_speed(37.4))
        self.assertEqual(10, analyzer._wind_level_from_speed(27.4))
        self.assertEqual(10, analyzer._wind_level_from_speed(25.7))

    def test_split_lines_keep_case_intensity_extremes(self):
        """PDF 行内断行不能丢失区域站雨强、最大风速和降温站点极值。"""
        content = (
            "雨强介于20～40mm。国家站最大襄汾27.2mm/h；区域站最大盐湖凤凰谷\n38.7mm/h；"
            "局部出现8级以上大风，最\n大值出现在中阳南垣，达到27.4m/s。"
            "最低气温出现降温介于1.4-13.3℃之间，降温最大值出现在宁武。"
            "最低气\n温降温大于等于12℃的有5站：宁武13.3、朔城13.1、中阳12.6、神池12.5、岢岚12.3；"
        )
        case = StandardCase.from_dict(
            {
                "case_id": "case-line-breaks",
                "title": "复合天气过程",
                "date_range": "2025年5月",
                "source_chunk_ids": ["chunk-line-breaks"],
            }
        )
        metrics = IntensityExtractor().extract(
            case,
            [DocumentChunk(source_pdf="case.pdf", chunk_id="chunk-line-breaks", chunk_no=1, content=content)],
        )
        analyzer = ReportAnalyzer()

        self.assertEqual(38.7, analyzer._case_metric_extreme(metrics, {"最大小时雨强"}).value)
        self.assertEqual(27.4, analyzer._case_metric_extreme(metrics, {"最大风速"}).value)
        self.assertEqual(13.3, analyzer._case_metric_extreme(metrics, {"过程降温幅度"}).value)

    def test_case_analysis_output_defaults_and_truncation_guard(self):
        """逐例分析默认使用充足输出上限，并移除模型最后一段残句。"""
        self.assertEqual(1536, StructuredCaseSearchRequest().case_max_output_tokens)
        self.assertEqual(4, StructuredCaseSearchRequest().case_analysis_concurrency)
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = CaseChunkAnalyzer(None, None, Path(temp_dir))
            cleaned = analyzer._complete_analysis_text("第一句完整。第二句完整。最后一句被截断")

        self.assertEqual("第一句完整。第二句完整。", cleaned)

    def test_numeric_evidence_collects_all_chunks_without_deciding_extreme(self):
        """全部正文阶段只收集完整数字证据和承接句，不预先生成强度结论。"""
        chunks = [
            DocumentChunk(
                source_pdf="case.pdf",
                chunk_id="c1",
                chunk_no=1,
                content="前期最高气温超过40℃。\n其他说明。",
            ),
            DocumentChunk(
                source_pdf="case.pdf",
                chunk_id="c2",
                chunk_no=2,
                content="最高气温介于35-46.1℃之间。最大值出现在永济雪花山。",
            ),
        ]

        evidence = LlmIntensityExtractor().collect_numeric_evidence(chunks)
        texts = [item["text"] for item in evidence]

        self.assertIn("前期最高气温超过40℃。", texts)
        self.assertIn("最高气温介于35-46.1℃之间。", texts)
        self.assertIn("最大值出现在永济雪花山。", texts)
        self.assertTrue(all("metric_name" not in item for item in evidence))

    def test_diagnostic_slots_use_verified_metrics_and_compose_three_paragraphs(self):
        """正文模型只填四个诊断槽位，后端必须把已核验强度事实写入第一段。"""
        class DiagnosticLlm:
            def __init__(self):
                self.question = ""
                self.context = ""

            def answer_with_context(self, question, context_blocks, max_tokens=1536):
                self.question = question
                self.context = "\n".join(context_blocks)
                import json
                return json.dumps(
                    {
                        "process_profile": "午后对流云团由西向东发展，阳泉一带雨强明显增强，强降水落区与短时风险时段对应。",
                        "hazard_chain": "材料中的切变线和低层水汽输送共同维持上升运动，地形抬升使局地强降水在傍晚前后加强。",
                        "compound_risk": "短时强降水与雷暴大风在同一时段出现，山区沟谷和城市低洼区域的积涝、落石及交通风险同步上升。",
                        "transferable_insight": "当切变线维持且迎风坡回波持续发展时，应把雨强跃增区与山区交通敏感点作为同一短临服务单元跟踪。",
                        "cited_image_ids": [],
                    },
                    ensure_ascii=False,
                )

        case = StandardCase.from_dict({
            "case_id": "case-diagnostic-slots",
            "title": "强对流过程",
            "date_range": "2025年5月16日",
            "disaster_types": ["强对流", "暴雨"],
            "affected_areas": ["阳泉"],
            "source_chunk_ids": ["c1"],
        })
        chunks = [DocumentChunk(
            source_pdf="case.pdf",
            chunk_id="c1",
            chunk_no=1,
            content="切变线影响下，阳泉午后出现短时强降水和雷暴大风，预警服务覆盖山区旅游点。",
        )]
        metrics = [IntensityMetric(
            metric_name="最大小时雨强",
            value=38.7,
            unit="mm/h",
            location="阳泉",
            source_chunk_id="c1",
            source_text="阳泉站最大小时雨强达38.7mm/h。",
        )]
        client = DiagnosticLlm()
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = CaseChunkAnalyzer(None, client, Path(temp_dir))
            analysis, status, image_ids, returned_metrics = analyzer.analyze_case_with_metrics(
                case,
                chunks,
                CaseSearchQuery(disaster_types=["强对流"]),
                "规则兜底。",
                True,
                6000,
                1536,
                extract_metrics=False,
                verified_metrics=metrics,
            )

        self.assertEqual("llm_generated", status)
        self.assertEqual([], image_ids)
        self.assertEqual([], returned_metrics)
        self.assertEqual(3, len(analysis.split("\n\n")))
        self.assertIn("最大小时雨强38.7mm/h（阳泉）", analysis)
        self.assertIn("已核验强度事实", client.context)
        self.assertIn("来源句：阳泉站最大小时雨强达38.7mm/h。", client.context)
        self.assertIn('"process_profile"', client.question)
        self.assertNotIn('"forecast_service_review"', client.question)
        self.assertNotIn('"metrics"', client.question)

    def test_diagnostic_analysis_merges_missing_evidence_boundaries(self):
        """缺少机理和复合风险证据时，只保留一次合并边界且不再输出服务复盘模板。"""
        case = StandardCase.from_dict({
            "case_id": "case-diagnostic-boundary",
            "title": "大风过程",
            "date_range": "2025年4月20日",
            "disaster_types": ["大风"],
            "affected_areas": ["山西省"],
        })
        metrics = [IntensityMetric(
            metric_name="最大风速",
            value=29.2,
            unit="m/s",
            location="五台山",
            source_chunk_id="c1",
            source_text="五台山最大风速为29.2m/s。",
        )]
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = CaseChunkAnalyzer(None, None, Path(temp_dir))
            analysis = analyzer._compose_diagnostic_analysis(case, metrics, {
                "process_profile": "大风实况主要出现在五台山一带。",
                "hazard_chain": "",
                "compound_risk": "",
                "transferable_insight": "",
            })

        self.assertEqual(3, len(analysis.split("\n\n")))
        self.assertEqual(1, analysis.count("当前精选材料未形成可核验"))
        self.assertNotIn("预报、预警或服务记录", analysis)
        self.assertNotIn("加强监测", analysis)

    def test_analysis_role_coverage_replaces_duplicate_role_with_missing_role(self):
        """六个精选片段应覆盖过程、机理和影响，不再强制为服务材料预留名额。"""
        chunks = [
            DocumentChunk(source_pdf="case.pdf", chunk_id="overview-1", chunk_no=1, content="天气过程概况和天气实况。"),
            DocumentChunk(source_pdf="case.pdf", chunk_id="overview-2", chunk_no=2, content="主要天气过程及过程特征。"),
            DocumentChunk(source_pdf="case.pdf", chunk_id="overview-3", chunk_no=3, content="天气实况持续发展。"),
            DocumentChunk(source_pdf="case.pdf", chunk_id="circulation", chunk_no=4, content="高空槽和冷锋共同影响。"),
            DocumentChunk(source_pdf="case.pdf", chunk_id="forecast", chunk_no=5, content="预报服务发布预警信息。"),
            DocumentChunk(source_pdf="case.pdf", chunk_id="impact", chunk_no=6, content="道路交通风险明显增大。"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = CaseChunkAnalyzer(None, None, Path(temp_dir))
            selected = analyzer._ensure_analysis_role_coverage(chunks[:3], chunks, 3, 1000)

        selected_roles = {analyzer._chunk_role(chunk) for chunk in selected}
        self.assertLessEqual(len(selected), 3)
        self.assertIn("overview", selected_roles)
        self.assertIn("circulation", selected_roles)
        self.assertIn("impact", selected_roles)

    def test_four_slot_analysis_receives_all_captions_and_displays_only_cited_image(self):
        """全部个例图注应随六 chunk 进入模型，最终只展示正文明确引用且返回 ID 的图片。"""
        class CaptionAwareLlm:
            def __init__(self):
                self.context = ""

            def answer_with_context(self, question, context_blocks, max_tokens=1536):
                self.context = "\n".join(context_blocks)
                import json
                return json.dumps(
                    {
                        "process_profile": "图14显示北部降雪中心与积雪高值区基本重合，降雪由北向南扩展。过程实况表明夜间降温后雨雪相态发生转换。",
                        "hazard_chain": "低层冷垫与暖湿气流对峙使迎风坡抬升增强，强降雪集中在冷暖交汇区域。回波演变与地面降温时段相互对应。",
                        "compound_risk": "降雪和降温共同促进路面积雪结冰，高海拔道路风险高于平原区域。该判断只对应材料明确记录的影响区域。",
                        "transferable_insight": "相似过程中应同步跟踪冷垫加深、相态转换和迎风坡回波增强信号。北部高海拔道路应作为风险研判的优先区域。",
                        "cited_image_ids": ["fig-14"],
                    },
                    ensure_ascii=False,
                )

        case = StandardCase.from_dict({
            "case_id": "case-caption-selection",
            "title": "雨雪天气过程",
            "date_range": "2025年3月14-15日",
            "disaster_types": ["雨雪", "暴雪"],
            "affected_areas": ["山西北部"],
            "source_chunk_ids": ["c1"],
        })
        chunks = [DocumentChunk(
            source_pdf="case.pdf",
            chunk_id="c1",
            chunk_no=1,
            content="冷暖空气交汇，北部山区出现雨转雪和道路结冰。",
        )]
        images = [
            {"image_id": "fig-14", "caption": "图14 降雪实况与积雪深度"},
            {"image_id": "fig-15", "caption": "图15 高空环流形势"},
        ]
        client = CaptionAwareLlm()
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = CaseChunkAnalyzer(None, client, Path(temp_dir))
            analysis, status, cited_ids, _ = analyzer.analyze_case_with_metrics(
                case,
                chunks,
                CaseSearchQuery(disaster_types=["雨雪"]),
                "规则兜底。",
                True,
                6000,
                1536,
                displayed_images=images,
                extract_metrics=False,
            )
            agent = CaseMultidimSearchAgent(None, None, None, Path(temp_dir))
            _, selected = agent._align_case_analysis_images(
                analysis,
                images,
                cited_image_ids=cited_ids,
            )

        self.assertEqual("llm_generated", status)
        self.assertIn("图14 降雪实况与积雪深度", client.context)
        self.assertIn("图15 高空环流形势", client.context)
        self.assertEqual(["fig-14"], cited_ids)
        self.assertEqual(["fig-14"], [image["image_id"] for image in selected])

    def test_one_case_llm_call_returns_metrics_analysis_and_images_together(self):
        """单个个例的一次模型调用必须同时返回指标、正文和图片引用。"""
        class OneCallLlm:
            def __init__(self):
                self.calls = 0

            def is_available(self):
                return True

            def answer_with_context(self, question, context_blocks, max_tokens=1536):
                self.calls += 1
                paragraph = "过程演变与主导灾种对应明确，实况峰值及影响区域均有当前材料支撑。天气系统配置与地形作用共同调制了峰值时段，业务复盘可据此核对过程发展。"
                analysis = "\n\n".join([paragraph * 2, paragraph * 2, paragraph * 2])
                import json
                return json.dumps(
                    {
                        "metrics": [{
                            "metric_name": "最高气温",
                            "value": 46.1,
                            "unit": "℃",
                            "location": "永济雪花山",
                            "relation": "最大",
                            "source_chunk_id": "c2",
                            "source_text": "最高气温介于35-46.1℃之间。",
                            "confidence": 0.96,
                        }],
                        "analysis": analysis,
                        "cited_image_ids": [],
                    },
                    ensure_ascii=False,
                )

        case = StandardCase.from_dict({
            "case_id": "case-once",
            "title": "高温过程",
            "date_range": "2025年7月",
            "disaster_types": ["高温"],
            "source_chunk_ids": ["c1", "c2"],
        })
        chunks = [
            DocumentChunk(source_pdf="case.pdf", chunk_id="c1", chunk_no=1, content="前期最高气温超过40℃。"),
            DocumentChunk(source_pdf="case.pdf", chunk_id="c2", chunk_no=2, content="最高气温介于35-46.1℃之间。最大值出现在永济雪花山。"),
        ]
        client = OneCallLlm()
        extractor = LlmIntensityExtractor()
        evidence = extractor.collect_numeric_evidence(chunks)
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = CaseChunkAnalyzer(None, client, Path(temp_dir), metric_extractor=extractor)
            analysis, status, image_ids, metrics = analyzer.analyze_case_with_metrics(
                case,
                chunks,
                CaseSearchQuery(disaster_types=["高温"]),
                "规则兜底。",
                True,
                6000,
                1536,
                metric_evidence=evidence,
                all_chunks=chunks,
            )

        self.assertEqual(1, client.calls)
        self.assertEqual("llm_generated", status)
        self.assertEqual([], image_ids)
        self.assertEqual(46.1, metrics[0].value)
        self.assertEqual(3, len(analysis.split("\n\n")))

    def test_metric_facts_scan_all_chunks_in_six_chunk_batches(self):
        """第七个 chunk 的指标也必须进入独立事实库，不能受正文精选上限影响。"""
        case = StandardCase.from_dict({
            "case_id": "case-all-chunks",
            "title": "暴雪天气过程",
            "date_range": "2025年3月",
            "disaster_types": ["暴雪"],
            "source_chunk_ids": [f"chunk-{index}" for index in range(1, 8)],
        })
        chunks = [
            DocumentChunk(source_pdf="case.pdf", chunk_id=f"chunk-{index}", chunk_no=index, content="过程记录正常。")
            for index in range(1, 7)
        ] + [
            DocumentChunk(
                source_pdf="case.pdf",
                chunk_id="chunk-7",
                chunk_no=7,
                content="最大积雪深度达到21厘米。过程最大降水量为26.6毫米。",
            )
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = CaseMetricPipeline(
                IntensityExtractor(),
                LlmIntensityExtractor(),
                Path(temp_dir),
            )
            facts = pipeline.extract(case, chunks, llm_enabled=False)

        values = {(metric.metric_name, metric.value, metric.unit) for metric in facts.metrics}
        self.assertEqual(2, facts.batch_count)
        self.assertIn(("最大积雪深度", 21.0, "cm"), values)
        self.assertIn(("过程最大降水量", 26.6, "mm"), values)
        self.assertEqual("confirmed", facts.status_by_metric["最大积雪深度"])
        self.assertEqual("confirmed", facts.status_by_metric["过程最大降水量"])

    def test_metric_facts_extracts_24_hour_accumulated_precip_range(self):
        """24小时累计降水量省略连接词时仍需识别范围上限。"""
        case = StandardCase.from_dict({
            "case_id": "case-24h-precip",
            "title": "暴雪天气过程",
            "date_range": "2025年3月",
            "disaster_types": ["暴雪"],
            "source_chunk_ids": ["chunk-precip"],
        })
        chunks = [DocumentChunk(
            source_pdf="case.pdf",
            chunk_id="chunk-precip",
            chunk_no=1,
            content="24小时累计降水量 0.1-26.6mm（神池），10mm以上降水有14站。",
        )]
        with tempfile.TemporaryDirectory() as temp_dir:
            facts = CaseMetricPipeline(
                IntensityExtractor(),
                LlmIntensityExtractor(),
                Path(temp_dir),
            ).extract(case, chunks, llm_enabled=False)

        self.assertIn(
            ("过程最大降水量", 26.6, "mm"),
            {(metric.metric_name, metric.value, metric.unit) for metric in facts.metrics},
        )

    def test_metric_fact_llm_never_receives_evidence_from_more_than_six_chunks(self):
        """指标补充模型每次只能看到一个六 chunk 批次的数字证据。"""
        class BatchLlm:
            def __init__(self):
                self.contexts = []

            def answer_with_context(self, question, context_blocks, max_tokens=640):
                self.contexts.append("\n".join(context_blocks))
                return '{"metrics":[]}'

        case = StandardCase.from_dict({
            "case_id": "case-batch-limit",
            "title": "大风天气过程",
            "date_range": "2025年4月",
            "disaster_types": ["大风"],
            "source_chunk_ids": [f"chunk-{index}" for index in range(1, 8)],
        })
        chunks = [
            DocumentChunk(
                source_pdf="case.pdf",
                chunk_id=f"chunk-{index}",
                chunk_no=index,
                content=f"第{index}段过程记录，风力有变化。",
            )
            for index in range(1, 8)
        ]
        client = BatchLlm()
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = CaseMetricPipeline(
                IntensityExtractor(),
                LlmIntensityExtractor(llm_client=client),
                Path(temp_dir),
            )
            pipeline.extract(case, chunks, llm_enabled=True)

        self.assertEqual(2, len(client.contexts))
        self.assertIn("[chunk-1|句0]", client.contexts[0])
        self.assertIn("[chunk-6|句0]", client.contexts[0])
        self.assertNotIn("[chunk-7|句0]", client.contexts[0])
        self.assertIn("[chunk-7|句0]", client.contexts[1])

    def test_company_chunk_store_shares_concurrent_requests(self):
        """并发个例读取相同 chunk 时必须复用一个远程请求并保持顺序。"""
        class CountingClient(FakeCompanyKbClient):
            def __init__(self):
                super().__init__()
                self.chunk_calls = 0

            def get_chunk(self, chunk_id):
                self.chunk_calls += 1
                time.sleep(0.03)
                return super().get_chunk(chunk_id)

        client = CountingClient()
        store = CompanyDocumentChunkStore(client)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: store.get_chunks(["chunk-001"]), range(2)))

        self.assertEqual(1, client.chunk_calls)
        self.assertEqual(["chunk-001"], [item.chunk_id for item in results[0]])
        self.assertEqual(["chunk-001"], [item.chunk_id for item in results[1]])

    def test_non_thinking_adapter_sends_qwen_template_flag(self):
        """OpenAI 兼容请求必须真正传递 Qwen3 关闭思考模板参数。"""
        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                message = types.SimpleNamespace(content="完成")
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = types.SimpleNamespace(completions=FakeCompletions())

        class RawOpenAiClient:
            api_key = "EMPTY"
            base_url = "http://model.example/v1"
            model = "qwen3-14b"

            def _answer_messages(self, question, context_blocks):
                return [{"role": "user", "content": question}]

        fake_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
        with patch.dict(sys.modules, {"openai": fake_module}):
            answer = NonThinkingLlmClient(RawOpenAiClient()).answer_with_context("问题", ["材料"], 512)

        self.assertEqual("完成", answer)
        self.assertEqual(
            {"chat_template_kwargs": {"enable_thinking": False}},
            captured["extra_body"],
        )

    def test_image_metadata_stage_does_not_download_candidate_files(self):
        """逐例模型选图前只读取图注元数据，不触发候选图片下载。"""
        class NoDownloadClient(FakeCompanyKbClient):
            def __init__(self):
                super().__init__()
                self.download_calls = 0

            def download_asset(self, *args, **kwargs):
                self.download_calls += 1
                raise AssertionError("元数据阶段不应下载图片")

        client = NoDownloadClient()
        case = StandardCase.from_dict(client.cases[0])
        with tempfile.TemporaryDirectory() as temp_dir:
            image_store = CompanyImageEvidenceStore(client, Path(temp_dir))
            agent = CaseMultidimSearchAgent(None, None, image_store, Path(temp_dir))
            images = agent._images_for_case(case, limit=6, resolve_paths=False)

        self.assertEqual(0, client.download_calls)
        self.assertEqual(1, len(images))
        self.assertFalse(images[0]["path_resolved"])

    def test_case_pipeline_uses_two_llm_slots_and_keeps_original_order(self):
        """逐例生成可两路重叠，但最终结果必须按检索顺序归位且每例只调用一次。"""
        cases = [
            StandardCase.from_dict({
                "case_id": f"case-{index}",
                "title": f"7月{index}日高温过程",
                "date_range": f"2025年7月{index}日",
                "year": 2025,
                "months": [7],
                "disaster_types": ["高温"],
                "source_chunk_ids": [f"chunk-{index}"],
            })
            for index in (1, 2)
        ]

        class CaseStore:
            def list_cases(self):
                return cases

        class ChunkStore:
            def get_chunks(self, chunk_ids):
                index = chunk_ids[0].split("-")[-1]
                return [DocumentChunk(
                    source_pdf="case.pdf",
                    chunk_id=chunk_ids[0],
                    chunk_no=int(index),
                    content=f"最高气温达到4{index}.1℃，最大值出现在测试站。",
                )]

        class EmptyImageStore:
            def list_by_image_ids(self, image_ids):
                return []

            def list_by_chunk_id(self, chunk_id, limit=6):
                return []

        class PipelineLlm:
            def __init__(self):
                self.case_calls = 0
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def is_available(self):
                return True

            def answer_with_context(self, question, context_blocks, max_tokens=1536):
                import json
                if "请只分析当前这一个" not in question:
                    return "{}"
                with self.lock:
                    self.case_calls += 1
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    time.sleep(0.04)
                    context = "\n".join(context_blocks)
                    index = 1 if "case-1" in context else 2
                    paragraph = "过程演变、影响范围与高温实况对应清楚，站点峰值可由当前个例材料核验。环流背景和地形差异共同影响高温发展，业务复盘应结合时段变化检查预报偏差。"
                    return json.dumps({
                        "metrics": [{
                            "metric_name": "最高气温",
                            "value": float(f"4{index}.1"),
                            "unit": "℃",
                            "location": "测试站",
                            "relation": "最大",
                            "source_chunk_id": f"chunk-{index}",
                            "source_text": f"最高气温达到4{index}.1℃，最大值出现在测试站。",
                            "confidence": 0.95,
                        }],
                        "analysis": "\n\n".join([paragraph * 2, paragraph * 2, paragraph * 2]),
                        "cited_image_ids": [],
                    }, ensure_ascii=False)
                finally:
                    with self.lock:
                        self.active -= 1

        client = PipelineLlm()
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = CaseMultidimSearchAgent(
                CaseStore(),
                ChunkStore(),
                EmptyImageStore(),
                Path(temp_dir),
                llm_client=client,
            )
            request = StructuredCaseSearchRequest(
                years=[2025],
                months=[7],
                disaster_types=["高温"],
                include_images=False,
                case_analysis_concurrency=2,
            )
            expected_order = [item.case.case_id for item in agent.structured_retriever.search(cases, request.to_query())]
            execution = agent._execute_structured(request)

        self.assertEqual(2, client.case_calls)
        self.assertEqual(2, client.max_active)
        self.assertEqual(expected_order, [hit.case["case_id"] for hit in execution.response.results])
        values_by_case = {hit.case["case_id"]: hit.intensity_metrics[0].value for hit in execution.response.results}
        self.assertEqual({"case-1": 41.1, "case-2": 42.1}, values_by_case)
    def test_report_enhancer_rejects_generation_traces_and_incomplete_advice(self):
        """报告增强不得用生成痕迹结论或不完整建议覆盖规则兜底稿。"""
        enhancer = ReportLlmEnhancer(None)
        fallback = {
            "conclusion": "规则生成的完整综合结论。",
            "recommendations": ["规则建议一。", "规则建议二。", "规则建议三。", "规则建议四。"],
        }
        bad_conclusion = (
            "原始材料显示，过程具有明显阶段性和区域差异。" * 16
        )
        bad_recommendations = [
            "建议建立短临触发信号清单并逐次复盘预警提前量，明确雷达回波、自动站和闪电资料的协同判据。",
            "建议针对吕梁山和五台山迎风坡开展模式降水及温度偏差订正，形成分区阈值并滚动检验订正效果。",
            "建议完善大风、降水和低能见度叠加风险研判流程，按影响对象形成交通与城市运行服务清单。",
            "建议统一个例正文、图像和指标的关联规范，补齐来源标识与质量审核，避免跨个例数据混入",
        ]

        merged = enhancer._merge_analysis(
            fallback,
            {"conclusion": bad_conclusion, "recommendations": bad_recommendations},
        )

        self.assertEqual(fallback["conclusion"], merged["conclusion"])
        self.assertEqual(fallback["recommendations"], merged["recommendations"])

    def test_report_conclusion_does_not_take_over_recommendation_actions(self):
        """综合结论不得使用建议句式，业务建议则必须包含明确且审慎的行动表达。"""
        enhancer = ReportLlmEnhancer(None)
        action_conclusion = "建议探索建立分区订正和短临联动流程，以解决复杂地形下的预警问题。" * 15
        recommendation = (
            "建议探索在吕梁山和五台山迎风坡试点建立模式偏差档案，针对降水落区和风速预报偏差开展滚动检验，"
            "可参考历史过程筛选稳定因子，在完成本地验证后再逐步用于短临会商和服务产品。"
        )

        self.assertFalse(enhancer._valid_conclusion(action_conclusion))
        self.assertTrue(enhancer._valid_recommendation(recommendation))
    def test_report_enhancer_paragraphizes_valid_conclusion(self):
        """合格综合结论应稳定整理成三个自然段供网页和 PDF 分段展示。"""
        enhancer = ReportLlmEnhancer(None)
        conclusion = (
            "检索样本在时间演变上呈现阶段集中与季节转换并存的特征，灾种组合受冷暖空气活动和地形共同影响，北部山区、吕梁山沿线及晋中盆地表现出不同的风险侧重。"
            "各类过程并非孤立发生，大风、降水、降温和低能见度常在相近时段叠加，业务研判需要从单一灾种识别转向过程链分析。"
            "现有复盘反映出短临触发信号使用不够统一，复杂地形下模式落区和强度偏差仍较突出，部分复合影响缺少面向交通、能源和城市运行的定量评估。"
            "尤其在迎风坡、山口和盆地边缘，观测代表性与预警阈值之间仍需建立更稳定的对应关系，以减少局地高影响天气的漏报和迟报。"
            "后续应把监测识别、订正检验、风险研判和服务复盘串成闭环，按区域建立可检验的指标体系，并将每次过程的预警提前量、落区偏差及服务反馈纳入常态评估。"
            "同时需要统一个例正文、图片和量化指标的关联规则，通过来源核验和质量审核提高材料一致性，为山西复杂地形下的精细化预报预警提供可靠依据。"
        )

        formatted = enhancer._paragraphize_conclusion(conclusion)

        self.assertTrue(enhancer._valid_conclusion(conclusion))
        self.assertEqual(3, len(formatted.split("\n\n")))

    def test_report_enhancer_parses_thinking_and_repairs_missing_field_comma(self):
        """报告模型带思考链且顶层字段漏逗号时仍应解析最终 JSON。"""
        enhancer = ReportLlmEnhancer(None)
        response = """<think>这里是不会进入报告的思考过程。</think>
{
  "executive_summary": "总体概况。",
  "sections": {
    "temporal": "时间特征。",
    "disaster": "灾种特征。",
    "spatial": "空间特征。",
    "intensity": "强度特征。"
  },
  "conclusion": "综合结论。"
  "recommendations": ["建议一。", "建议二。", "建议三。", "建议四。"]
}"""

        parsed = enhancer._parse_json(response)

        self.assertEqual("总体概况。", parsed["executive_summary"])
        self.assertEqual("综合结论。", parsed["conclusion"])
        self.assertEqual(4, len(parsed["recommendations"]))

    def test_main_areas_use_full_case_text_frequency_and_province_shortcut(self):
        """主要影响区域应扫描完整正文，地市按频次选前五，全省过程只显示全省。"""
        analyzer = ReportAnalyzer()
        case = StandardCase(case_id="case-area", title="大风过程", date_range="4月16-18日")
        city_text = (
            "4月16日08时至17日08时，太原市、吕梁市、晋中市、长治市、晋城市、运城市、临汾市的局部，"
            "忻州市、朔州市、大同市、阳泉市的部分出现大风天气。次日太原市、吕梁市再次出现大风。"
        )

        self.assertEqual("太原、吕梁、晋中、长治、晋城等", analyzer._short_area_text(case, [], city_text))
        self.assertEqual("全省", analyzer._short_area_text(case, [], "全省共2186站出现降水，过程影响范围广。"))

    def test_process_extremes_compare_all_values_and_keep_metric_semantics(self):
        """过程雨量、最高气温和风力等级应按完整正文极值提取，小时雨量不得污染累计雨量。"""
        content = (
            "小时最大降水量为67.8mm/h，累计降水量为0.1～150.4\n毫米，最大值出现在榆次小张义。"
            "过程累计降水量最高达303mm，国家站最大累计降水量为205.5mm。"
            "最高气温大于40℃，最高气温介于35-46.1℃之间，最大值出现在永济雪花山。"
            "26日北中部局地还出现了8-10级的雷暴大风。"
        )
        case = StandardCase.from_dict(
            {"case_id": "case-extremes", "title": "复合天气过程", "date_range": "7月", "source_chunk_ids": ["chunk-extremes"]}
        )
        metrics = IntensityExtractor().extract(
            case,
            [DocumentChunk(source_pdf="case.pdf", chunk_id="chunk-extremes", chunk_no=1, content=content)],
        )
        analyzer = ReportAnalyzer()

        self.assertEqual(303, analyzer._case_metric_extreme(metrics, {"过程最大降水量", "最大降水量"}).value)
        self.assertEqual(46.1, analyzer._case_metric_extreme(metrics, {"最高气温"}).value)
        self.assertEqual(10, analyzer._case_metric_extreme(metrics, {"阵风风力"}).value)
        self.assertNotIn(
            67.8,
            [metric.value for metric in metrics if metric.metric_name in {"过程最大降水量", "最大降水量"}],
        )

    def test_case_order_and_figure_numbers_are_shared_by_web_and_pdf(self):
        """逐例结果先稳定分组，再统一改写正文引用与图注编号，供网页和PDF共用。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = CaseMultidimSearchAgent(None, None, CompanyImageEvidenceStore(FakeCompanyKbClient(), Path(temp_dir)), Path(temp_dir))
            hits = [
                CaseSearchHit(
                    case={"case_id": "a1", "matched_disaster": "暴雨"},
                    score=1,
                    analysis="图22（左上）显示降水落区。",
                    evidence_images=[{"image_id": "img-a1", "caption": "图22（a）降水实况"}],
                ),
                CaseSearchHit(
                    case={"case_id": "b1", "matched_disaster": "大风"},
                    score=1,
                    analysis="图9显示大风实况。",
                    evidence_images=[{"image_id": "img-b1", "caption": "图9大风实况"}],
                ),
                CaseSearchHit(
                    case={"case_id": "a2", "matched_disaster": "暴雨"},
                    score=1,
                    analysis="图7显示雷达回波。",
                    evidence_images=[{"image_id": "img-a2", "caption": "图7雷达回波"}],
                ),
            ]
            ordered = agent._group_hits_for_display(hits)
            agent._renumber_case_images(ordered)

        self.assertEqual(["a1", "a2", "b1"], [hit.case["case_id"] for hit in ordered])
        self.assertEqual([1, 2, 3], [hit.evidence_images[0]["display_number"] for hit in ordered])
        self.assertIn("图 1", ordered[0].analysis)
        self.assertIn("图 2", ordered[1].analysis)
        self.assertIn("图 3", ordered[2].analysis)
        self.assertEqual("图 2 雷达回波", ordered[1].evidence_images[0]["display_caption"])

    def test_pdf_uses_backend_display_caption_and_number(self):
        """PDF逐例图片必须复用后端统一图注和编号，不再按PDF内部位置重新编号。"""
        from types import SimpleNamespace

        from backend.app.services.agent.case_multidim_search.reporting.pdf_report import PdfReportBuilder

        class FakeDocument:
            """记录PDF写入参数，避免单元测试真实渲染文件。"""

            def __init__(self):
                self.figures = []

            def add_section(self, _title):
                return None

            def add_subsection(self, _title):
                return None

            def add_paragraph(self, _text, **_kwargs):
                return None

            def add_figure(self, path, caption, **kwargs):
                self.figures.append((str(path), caption, kwargs))
                return True

        document = FakeDocument()
        response = SimpleNamespace(
            results=[
                CaseSearchHit(
                    case={"case_id": "case-pdf", "title": "暴雨过程"},
                    score=1,
                    analysis="图 2显示降水实况。",
                    evidence_images=[
                        {
                            "image_id": "img-pdf",
                            "image_path": "figure.png",
                            "caption": "图22原始图题",
                            "display_caption": "图 2 降水实况",
                            "display_number": 2,
                        }
                    ],
                )
            ]
        )

        PdfReportBuilder()._write_cases(document, response)

        self.assertEqual("图 2 降水实况", document.figures[0][1])
        self.assertEqual(2, document.figures[0][2]["figure_number"])
    def test_cited_image_id_requires_a_matching_text_reference(self):
        """图片ID只用于消除图号歧义，正文没有引用时不得额外展示。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = CaseMultidimSearchAgent(None, None, CompanyImageEvidenceStore(FakeCompanyKbClient(), Path(temp_dir)), Path(temp_dir))
            images = [
                {"image_id": "fig-22a", "caption": "图22（a）降水实况"},
                {"image_id": "fig-17", "caption": "图17探空曲线"},
            ]
            _, selected = agent._align_case_analysis_images(
                "图22（左上）显示降水实况。",
                images,
                cited_image_ids=["fig-22a", "fig-17"],
            )

        self.assertEqual(["fig-22a"], [image["image_id"] for image in selected])
    def test_primary_disaster_follows_cold_wave_query_not_title_first_word(self):
        """标题先写大风时，只要检索实际命中寒潮，汇总表仍应填写寒潮。"""
        case = StandardCase.from_dict(
            {
                "case_id": "case-wind-cold",
                "title": "4月11-13日极端大风和寒潮、沙尘、霜冻、弱降水过程",
                "date_range": "2025年4月11日至13日",
                "disaster_types": ["大风", "寒潮", "沙尘", "霜冻"],
            }
        )
        analyzer = ReportAnalyzer()

        self.assertEqual("寒潮", analyzer._primary_disaster_text(case, CaseSearchQuery(disaster_types=["寒潮"])))
        self.assertEqual(
            "大风",
            analyzer._primary_disaster_text(case, CaseSearchQuery(disaster_types=["大风", "寒潮"])),
        )

    def test_primary_disaster_routes_rain_case_out_of_convection_table(self):
        """并发灾种相同的个例按标题中的检索命中灾种分流，暴雨过程不得落入强对流表。"""
        analyzer = ReportAnalyzer()
        query = CaseSearchQuery(disaster_types=["强对流", "暴雨", "短时强降水"])
        rain_case = StandardCase.from_dict(
            {
                "case_id": "case-rain-table",
                "title": "5月21-22日暴雨天气过程",
                "date_range": "2025年5月21日至22日",
                "disaster_types": ["强对流", "暴雨", "短时强降水"],
            }
        )
        convection_case = StandardCase.from_dict(
            {
                "case_id": "case-convection-table",
                "title": "6月27-28日强对流天气过程",
                "date_range": "2025年6月27日至28日",
                "disaster_types": ["强对流", "暴雨", "短时强降水"],
            }
        )

        rain_primary = analyzer._primary_disaster_text(rain_case, query)
        convection_primary = analyzer._primary_disaster_text(convection_case, query)
        self.assertEqual("暴雨", rain_primary)
        self.assertEqual("rain", analyzer._overview_group_for_case(rain_case, rain_primary)["key"])
        self.assertEqual("强对流", convection_primary)
        self.assertEqual("convection_hail", analyzer._overview_group_for_case(convection_case, convection_primary)["key"])

    def test_overview_tables_split_cold_wind_rain_snow_snowfall_and_blizzard(self):
        """寒潮、大风、雨雪、降雪和暴雪主导过程必须进入各自独立的汇总表。"""
        analyzer = ReportAnalyzer()
        cases = [
            StandardCase.from_dict({"case_id": "cold", "title": "2月寒潮天气过程", "date_range": "2025年2月", "disaster_types": ["寒潮", "大风"]}),
            StandardCase.from_dict({"case_id": "wind", "title": "3月大风天气过程", "date_range": "2025年3月", "disaster_types": ["大风"]}),
            StandardCase.from_dict({"case_id": "rain-snow", "title": "3月雨雪天气过程", "date_range": "2025年3月", "disaster_types": ["雨雪", "寒潮"]}),
            StandardCase.from_dict({"case_id": "snow", "title": "1月降雪天气过程", "date_range": "2025年1月", "disaster_types": ["降雪"]}),
            StandardCase.from_dict({"case_id": "blizzard", "title": "2月暴雪天气过程", "date_range": "2025年2月", "disaster_types": ["暴雪"]}),
        ]
        metrics = {
            "cold": [IntensityMetric(metric_name="最低气温", value=-25.6, unit="℃", source_text="最低气温-25.6℃。")],
            "wind": [IntensityMetric(metric_name="最大风速", value=22.1, unit="m/s", source_text="最大风速22.1m/s。")],
            "rain-snow": [IntensityMetric(metric_name="过程最大降水量", value=23.4, unit="mm", source_text="过程最大降水量23.4mm。")],
            "snow": [IntensityMetric(metric_name="过程最大降雪量", value=3.2, unit="mm", source_text="过程最大降雪量3.2mm。")],
            "blizzard": [IntensityMetric(metric_name="最大积雪深度", value=18, unit="cm", source_text="最大积雪深度18cm。")],
        }

        tables = analyzer._overview_intensity_tables(cases, metrics, CaseSearchQuery(), [], {})

        self.assertEqual(
            ["wind", "cold_wave", "rain_snow", "snowfall", "blizzard"],
            [table["key"] for table in tables],
        )
        self.assertEqual(
            {
                "cold_wave": ["cold"],
                "wind": ["wind"],
                "rain_snow": ["rain-snow"],
                "snowfall": ["snow"],
                "blizzard": ["blizzard"],
            },
            {table["key"]: [row["case_id"] for row in table["rows"]] for table in tables},
        )

    def test_overview_group_columns_do_not_include_partially_empty_metrics(self):
        """同一主导灾种表优先只显示所有个例都有值的动态指标列。"""
        analyzer = ReportAnalyzer()
        cases = [
            StandardCase.from_dict({"case_id": "wind-1", "title": "3月大风过程", "date_range": "2025年3月", "disaster_types": ["大风"]}),
            StandardCase.from_dict({"case_id": "wind-2", "title": "4月大风过程", "date_range": "2025年4月", "disaster_types": ["大风"]}),
        ]
        metrics = {
            "wind-1": [
                IntensityMetric(metric_name="最大风速", value=22.1, unit="m/s", source_text="最大风速22.1m/s。"),
                IntensityMetric(metric_name="最低能见度", value=0.8, unit="km", source_text="最低能见度0.8km。"),
            ],
            "wind-2": [IntensityMetric(metric_name="最大风速", value=18.4, unit="m/s", source_text="最大风速18.4m/s。")],
        }

        tables = analyzer._overview_intensity_tables(cases, metrics, CaseSearchQuery(), [], {})

        self.assertEqual(["max_wind", "max_wind_level"], [column["key"] for column in tables[0]["columns"]])

    def test_wind_table_uses_observed_scope_and_rejects_warning_wind_level(self):
        """大风表固定六列，范围来自实况，预警中的预计风级不得覆盖实测值。"""
        analyzer = ReportAnalyzer()
        cases = [
            StandardCase.from_dict({"case_id": "wind-11", "title": "4月11日大风过程", "date_range": "2025年4月11日", "disaster_types": ["大风"]}),
            StandardCase.from_dict({"case_id": "wind-16", "title": "4月16日大风过程", "date_range": "2025年4月16日", "disaster_types": ["大风"]}),
            StandardCase.from_dict({"case_id": "wind-20", "title": "4月20日大风过程", "date_range": "2025年4月20日", "disaster_types": ["大风"]}),
        ]
        metrics = {
            "wind-11": [IntensityMetric(metric_name="极大风速", value=39.7, unit="m/s", source_text="区域站最大阵风风速达39.7m/s（13级）。")],
            "wind-16": [IntensityMetric(metric_name="最大风速", value=34.5, unit="m/s", source_text="最大值达到34.5m/s，风力等级达12级。")],
            "wind-20": [IntensityMetric(metric_name="最大风速", value=29.2, unit="m/s", source_text="最大值达到29.2m/s（11级）。")],
        }
        case_texts = {
            "wind-11": "8级以上阵风影响我省98%。",
            "wind-16": "共429站出现8级以上大风天气。",
            "wind-20": "共390个站出现8级以上大风天气。",
        }

        tables = analyzer._overview_intensity_tables(cases, metrics, CaseSearchQuery(), [], case_texts)

        self.assertEqual(["max_wind", "max_wind_level", "wind_impact_scope"], [item["key"] for item in tables[0]["columns"]])
        rows = {row["case_id"]: row for row in tables[0]["rows"]}
        self.assertEqual("39.7 m/s", rows["wind-11"]["metrics"]["max_wind"])
        self.assertEqual("13 级", rows["wind-11"]["metrics"]["max_wind_level"])
        self.assertEqual("全省98%", rows["wind-11"]["metrics"]["wind_impact_scope"])
        self.assertEqual("29.2 m/s", rows["wind-20"]["metrics"]["max_wind"])
        self.assertEqual("11 级", rows["wind-20"]["metrics"]["max_wind_level"])
        self.assertEqual("390站", rows["wind-20"]["metrics"]["wind_impact_scope"])
        self.assertTrue(
            LlmIntensityExtractor()._should_skip_metric(
                "阵风风力", 13, "预警区域预计未来24小时阵风可达11-13级。"
            )
        )

    def test_overview_table_lists_only_concurrent_disasters_after_primary(self):
        """强度汇总第三列展示并发灾害，并排除当前主导灾种。"""
        analyzer = ReportAnalyzer()
        case = StandardCase.from_dict(
            {
                "case_id": "case-concurrent",
                "title": "6月强对流和暴雨过程",
                "date_range": "2025年6月1日",
                "disaster_types": ["强对流", "暴雨", "短时强降水", "雷暴大风"],
            }
        )
        rows = analyzer._overview_intensity_table(
            [case],
            {case.case_id: []},
            [],
            CaseSearchQuery(disaster_types=["暴雨"]),
            primary_by_case={case.case_id: "暴雨"},
        )

        self.assertEqual("暴雨", rows[0]["main_disasters"])
        self.assertEqual("强对流、短时强降水、雷暴大风", rows[0]["concurrent_disasters"])

    def test_large_rainstorm_query_reports_below_threshold(self):
        """大暴雨检索命中但过程雨量低于100毫米时，概况必须明确区分标签和强度达标。"""
        metrics = {
            "case-rain": [
                IntensityMetric(
                    metric_name="过程最大降水量",
                    value=92.9,
                    unit="mm",
                    source_text="过程最大降水量为92.9毫米。",
                )
            ]
        }
        assessments = ReportAnalyzer()._query_strength_assessments(
            CaseSearchQuery(disaster_types=["大暴雨"]),
            metrics,
        )

        self.assertEqual(1, len(assessments))
        self.assertIn("92.9 mm", assessments[0])
        self.assertIn("低于24小时100 mm", assessments[0])
        self.assertIn("不能表述为强度达标", assessments[0])

    def test_spatial_chart_uses_case_count_ranking_not_pie_share(self):
        """空间图按地市命中个例数排名，同一个例涉及多地时不计算互斥占比。"""
        specs = ChartTool().select_specs(
            {"case_count": 6, "city_counts": {"太原": 6, "晋中": 4, "大同": 2}},
            CaseSearchQuery(),
        )
        spatial = next(spec for spec in specs if spec.chart_key == "spatial")

        self.assertEqual("horizontal_bar", spatial.chart_type)
        self.assertEqual("命中个例数", spatial.x_label)
        self.assertEqual(6, spatial.sample_size)
        self.assertNotIn("占比", spatial.interpretation)
        self.assertNotIn("%", spatial.interpretation)

    def test_small_case_keeps_all_chunks_before_context_truncation(self):
        """关联片段不超过六个时全部保留，字符限制不再误删片段。"""
        case = StandardCase.from_dict(
            {
                "case_id": "case-four-chunks",
                "title": "1月降雪过程",
                "date_range": "2025年1月11日",
                "source_chunk_ids": ["c1", "c2", "c3", "c4"],
            }
        )
        chunks = [
            DocumentChunk(source_pdf="case.pdf", chunk_id=f"c{index}", chunk_no=index, content=f"片段{index}的正文")
            for index in range(1, 5)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = CaseChunkAnalyzer(None, None, Path(temp_dir))
            selected = analyzer.select_relevant_chunks(case, chunks, CaseSearchQuery(), limit=6, char_limit=5)

        self.assertEqual(["c1", "c2", "c3", "c4"], [chunk.chunk_id for chunk in selected])

        seven_chunks = chunks + [DocumentChunk(source_pdf="case.pdf", chunk_id="c5", chunk_no=5, content="片段5的正文"), DocumentChunk(source_pdf="case.pdf", chunk_id="c6", chunk_no=6, content="片段6的正文"), DocumentChunk(source_pdf="case.pdf", chunk_id="c7", chunk_no=7, content="片段7的正文")]
        analyzer._select_by_rerank_model = lambda *_args: []
        selected_seven = analyzer.select_relevant_chunks(case, seven_chunks, CaseSearchQuery(), limit=8, char_limit=10000)
        self.assertEqual(6, len(selected_seven))

    def test_short_unbroken_analysis_is_balanced_before_completeness_check(self):
        """模型返回约四百字但没有换行时，按完整句整理后不再直接降级。"""
        text = "".join(
            [
                "过程概况说明天气演变、主导灾种、主要影响区域以及降水风温的变化特点。",
                "实况强度与当前检索灾种相互对应，站点观测能够支撑过程峰值和影响范围判断。",
                "环流背景说明冷空气、地形抬升和水汽输送共同作用，并对应到天气发展阶段。",
                "峰值时段与站点观测变化基本一致，局地差异还反映山地、盆地之间的下垫面调制。",
                "预报服务需要结合预警发布时间、落区预报偏差和实况演变过程进行业务复盘。",
                "重点区域应关注交通、城市运行和能源保障风险，并明确短临监测与服务响应动作。",
            ] * 2
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = CaseChunkAnalyzer(None, None, Path(temp_dir))
            normalized = analyzer._normalize_analysis_paragraphs(text)

        self.assertEqual(3, len(normalized.split("\n\n")))
        self.assertTrue(analyzer._analysis_is_complete(normalized))
if __name__ == "__main__":
    unittest.main()
