"""统一聊天入口的三路路由、SSE 协议和会话落库测试。"""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.app.models import DocumentChunk, StandardCase
from backend.app.services.agent.case_multidim_search.core.natural_query import NaturalConversationStore
from backend.app.services.agent.case_multidim_search.schemas import CaseSearchQuery
from backend.app.services.agent.unified_chat.orchestrator import UnifiedChatOrchestrator, UnifiedRouteDecision
from backend.app.services.agent.query_intent_compat import QueryIntentAgent
from backend.app.services.session_store import SessionStore


class _UnavailableLlm:
    """模拟未配置 API Key 的普通 RAG 客户端。"""

    def is_available(self) -> bool:
        return False


class _SuggestingLlm(_UnavailableLlm):
    """模拟正文完成后的轻量追问生成，不参与正文内容生成。"""

    def is_available(self) -> bool:
        return True

    def suggest_follow_ups(self, question: str, answer: str, agent_type: str):
        self.last_suggestion_input = {
            "question": question,
            "answer": answer,
            "agent_type": agent_type,
        }
        return [{"label": "继续分析影响", "question": "请继续分析这次过程的影响范围"}]


class _StreamingLlm:
    """模拟有首字间隔的云端流，用于确认统一入口没有缓存完整答案。"""

    def __init__(self) -> None:
        self.last_question = ""

    def is_available(self) -> bool:
        return True

    def stream_answer_with_context(self, _question, _context_blocks):
        self.last_context_blocks = list(_context_blocks)
        self.last_question = _question
        yield "第一段"
        time.sleep(0.15)
        yield "第二段"


class _GuidedStreamingLlm(_StreamingLlm):
    """记录会话级临时要求，验证它只进入回答生成而不改变检索问题。"""

    def __init__(self) -> None:
        super().__init__()
        self.guidance_prompt = ""

    def stream_answer_with_context(
        self,
        question,
        context_blocks,
        *,
        session_guidance_prompt="",
    ):
        self.guidance_prompt = session_guidance_prompt
        yield from super().stream_answer_with_context(question, context_blocks)


class _IntentAnalyzer:
    """提供普通 RAG 生成执行计划所需的最小分析结果。"""

    def analyze(self, _question: str):
        return SimpleNamespace(intent_types=[])


class _Planner:
    """提供普通 RAG 元数据需要的最小执行计划。"""

    def plan(self, _question: str, _analysis):
        return SimpleNamespace(intents=[], to_dict=lambda: {"agent": "document_rag"})


class _DocumentStore:
    """模拟主文档库并保留可追溯 chunk。"""

    def __init__(self) -> None:
        self.chunk = DocumentChunk(
            source_pdf="sample.pdf",
            chunk_id="chunk-001",
            chunk_no=1,
            content="山西出现一次降水天气过程。",
            file_path="resource/sample.pdf",
            embedding=[],
        )
        self.warning_chunk = DocumentChunk(
            source_pdf="sample.pdf",
            chunk_id="chunk-warning",
            chunk_no=2,
            content="2025年3月1日08时发布暴雨黄色预警，预警区域为山西中部。",
            file_path="resource/sample.pdf",
            embedding=[],
        )
        # 三个个例使用互不相同的正文 chunk，验证多案例分析不会只拿到一个共享片段。
        self.march_chunks = [
            DocumentChunk("FST2025-3.pdf", "march-chunk-1", 1, "1-3日过程：累计降水、雨转雪和大风降温。"),
            DocumentChunk("FST2025-3.pdf", "march-chunk-2", 2, "14-15日暴雪：积雪深度、风力和降温特征。"),
            DocumentChunk("FST2025-3.pdf", "march-chunk-3", 3, "25-28日沙尘：大风、能见度和降温特征。"),
        ]
        self.temperature_chunk = DocumentChunk(
            "山西省2025年1月天气过程总结.pdf",
            "jan-temperature-chunk",
            16,
            "1月14日08时至15日08时，全省各地最低气温介于-25.6℃（新荣）～-2.7℃（永济）之间。",
        )

    def get_chunk(self, chunk_id: str):
        return {self.chunk.chunk_id: self.chunk, self.warning_chunk.chunk_id: self.warning_chunk}.get(chunk_id)

    def list_chunks(self) -> list[DocumentChunk]:
        """返回主体与跨章节预警证据，模拟同一 PDF 内的扩展召回。"""
        return [self.chunk, self.warning_chunk, *self.march_chunks, self.temperature_chunk]


class _StandardCaseStore:
    """提供统一编排测试使用的主标准个例集合。"""

    def __init__(self) -> None:
        self.cases = [
            StandardCase(
                case_id="case-001",
                title="历史暴雨过程",
                date_range="2025年3月1日",
                source_pdf="sample.pdf",
                source_chunk_ids=["chunk-001"],
            ),
            StandardCase(case_id="case-jan-0", title="1月11日降雪天气过程", date_range="2025年1月11日"),
            StandardCase(
                case_id="case-jan-1",
                title="1月14日-15日寒潮天气过程",
                date_range="2025年1月14-15日",
                disaster_types=["寒潮"],
                affected_areas=["全省"],
                source_pdf="山西省2025年1月天气过程总结.pdf",
                source_chunk_ids=["jan-temperature-chunk"],
                year=2025,
                months=[1],
                start_date="2025-01-14",
                end_date="2025-01-15",
            ),
            StandardCase(case_id="case-jan-2", title="1月23-26日雨雪寒潮大风天气过程", date_range="2025年1月23-26日"),
            StandardCase(case_id="hot-may", title="5月19-21日高温天气", date_range="2025年5月19-21日", disaster_types=["高温"], source_pdf="FST2025-5.pdf"),
            StandardCase(case_id="hot-july-1", title="7月12-18日持续性高温天气过程", date_range="2025年7月12-18日", disaster_types=["高温"], source_pdf="FST2025-7.pdf"),
            StandardCase(case_id="hot-july-2", title="7月28-31日高温天气过程", date_range="2025年7月28-31日", disaster_types=["高温"], source_pdf="FST2025-7.pdf"),
        ]

    def list_cases(self) -> list[StandardCase]:
        return list(self.cases)


class _MainModule:
    """模拟主应用依赖，避免测试调用外部模型和真实知识库。"""

    AgentIntentAnalyzer = _IntentAnalyzer
    AgentPlanner = _Planner

    def __init__(self, session_store: SessionStore, document_store: _DocumentStore) -> None:
        self.session_store = session_store
        self.document_store = document_store
        self.standard_case_store = _StandardCaseStore()
        self.llm_client = _UnavailableLlm()
        self.followup_llm_client = _UnavailableLlm()
        self.last_retrieval = {}

    def _context_case_ids_from_messages(self, _history, _question):
        return []

    def _retrieve_document_chunks(self, _question, **_kwargs):
        self.last_retrieval = {"question": _question, **_kwargs}
        context_case_ids = list(_kwargs.get("context_case_ids") or [])
        if context_case_ids:
            cases = {case.case_id: case for case in self.standard_case_store.list_cases()}
            chunks = {chunk.chunk_id: chunk for chunk in self.document_store.list_chunks()}
            scoped = []
            for case_id in context_case_ids:
                for chunk_id in cases.get(case_id, StandardCase("", "", "")).source_chunk_ids:
                    if chunk_id in chunks:
                        scoped.append(SimpleNamespace(chunk=chunks[chunk_id], score=0.91))
            if scoped:
                return "document_context_cases", scoped
        hit = SimpleNamespace(chunk=self.document_store.chunk, score=0.91)
        return "document_vector", [hit]

    def _document_hit_to_response(self, hit):
        return {
            "score": hit.score,
            "chunk": {
                "chunk_id": hit.chunk.chunk_id,
                "chunk_no": hit.chunk.chunk_no,
                "source_pdf": hit.chunk.source_pdf,
                "content": hit.chunk.content,
            },
            "images": [],
        }

    def _document_answer_fallback(self, _question, _hits):
        return "普通 RAG 回答"

    def _context_blocks_from_document_hits(self, hits):
        """返回普通 RAG 流式测试所需的最小证据上下文。"""
        return [f"片段ID：{hit.chunk.chunk_id}\n{hit.chunk.content}" for hit in hits]

    @staticmethod
    def _standard_case_to_response(case: StandardCase) -> dict:
        """模拟主应用的标准个例响应转换。"""
        return case.to_dict()

    def _answer_structured_question(self, question, context_case_ids=None):
        """模拟标准个例聚合，验证简单统计不会进入重型 Agent。"""
        if "几个" not in question:
            return None
        return {
            "answer": "3月份共有3个灾害天气个例。",
            "retrieval_mode": "structured_cases:statistics",
            "hit_count": 3,
            "evidence_cases": [],
            "evidence_chunks": [],
            "images": [],
            "visuals": [],
            "llm_used": False,
        }

    def _structured_result_needs_metric_documents(self, _result):
        return False


class _MultidimAgent:
    """模拟多维 Agent 最终对象，不改变编排器对原流程的调用方式。"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.analysis_calls = 0
        self.reuse_export_calls = 0
        self.last_reuse_report_title = ""
        self.last_reuse_result = {}

    def analyze_structured(self, request):
        self.analysis_calls += 1
        case = {
            "case_id": "case-001",
            "title": "2025年3月暴雨过程",
            "source_chunk_ids": ["chunk-001"],
            "evidence_image_ids": [],
        }
        hit = SimpleNamespace(
            case=case,
            matched_fields=["years", "months"],
            score=1.0,
            analysis="个例分析",
            intensity_metrics=[],
        )
        search = SimpleNamespace(
            results=[hit],
            result_count=1,
            analysis={"count": 1},
            parsed_query=CaseSearchQuery(years=request.years, months=request.months),
        )
        return SimpleNamespace(
            answer_id="answer-001",
            title="多维检索报告",
            question="结构化检索",
            answer="多维检索回答",
            search_response=search,
            images=[],
            charts=[],
            audit={"flow": "preserved"},
        )

    def export_pdf_from_answer(self, _answer_id, filename="", report_title="", progress_callback=None):
        """模拟新版 PDF 导出签名。"""
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4\n%%EOF")
        return path

    def export_pdf_from_previous_result(self, _previous_result, filename="", report_title="", progress_callback=None):
        """模拟只排版上一轮结果的简易报告导出。"""
        self.reuse_export_calls += 1
        self.last_reuse_report_title = report_title
        self.last_reuse_result = dict(_previous_result)
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4\n%%EOF")
        return path


class _SmartAgent:
    """模拟 Smart 原始事件流，供统一 SSE 适配测试使用。"""

    def __init__(self) -> None:
        self.call_count = 0

    async def stream_natural_match(self, _request, _is_disconnected=None):
        self.call_count += 1
        yield "accepted", {"run_id": "smart-run-001", "stage": "parse", "percent": 2}
        yield "stage", {"run_id": "smart-run-001", "stage": "matching", "percent": 50}
        yield "completed", {
            "run_id": "smart-run-001",
            "status": "completed",
            "query_summary": {"disaster_types": ["暴雨"]},
            "matched_cases": [{
                "case_id": "case-001",
                "title": "历史暴雨过程",
                "rank": 1,
                "retrieval_score": 0.9,
                "source_chunk_ids": ["chunk-001"],
                "match_reasons": ["降水落区相近"],
                "key_references": [],
                "evidence_images": [],
                "supplemental_images": [],
            }],
            "forecast_summary": {"main_risk": "短时强降水"},
            "forecast_tips": [],
            "warnings": [],
            "audit": {"flow": "preserved"},
        }


class _MisroutingIntentAgent:
    """模拟意图模块误把有效库内个例送往 Smart，验证编排器最终保护。"""

    case_provider = None

    @staticmethod
    def route(_request):
        return SimpleNamespace(
            intent="similar_case_match",
            rag_strategy="vector",
            normalized_message="分析历史暴雨过程的预报效果",
            clarification_question="",
            referenced_case_ids=["case-001", "missing-case"],
            conditions={},
            audit_id="intent-misroute",
            context_related=True,
            relation_type="follow_up_reference",
            reference_type="knowledge_case",
            reference_clues=["历史暴雨过程"],
            context_confidence=0.99,
            context_resolution_source="llm",
            context_message_ids=[1, 2],
            routing_source="llm",
            reason="模拟误路由",
        )


class UnifiedChatOrchestratorTests(unittest.IsolatedAsyncioTestCase):
    """验证三类问题在同一会话和统一协议下正确运行。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.session_store = SessionStore(root / "sessions.sqlite3")
        self.document_store = _DocumentStore()
        self.main = _MainModule(self.session_store, self.document_store)
        self.multidim_store = NaturalConversationStore(session_store=self.session_store)
        self.smart_agent = _SmartAgent()
        self.orchestrator = UnifiedChatOrchestrator(
            self.main,
            _MultidimAgent(root / "reports"),
            self.multidim_store,
            self.smart_agent,
            QueryIntentAgent(
                llm_client=_UnavailableLlm(),
                audit_path=root / "intent-audit.jsonl",
                case_provider=self.main.standard_case_store.list_cases,
            ),
        )

    async def asyncTearDown(self) -> None:
        await self.orchestrator.wait_for_background_tasks()
        self.temp_dir.cleanup()

    async def _collect(self, question: str, session_id: str) -> list[dict]:
        """收集一轮统一事件，并验证没有泄漏旧事件名。"""
        payload = SimpleNamespace(question=question, session_id=session_id, top_k=10, top_n=5)
        events = [event async for event in self.orchestrator.stream(payload)]
        # 业务断言前等待后台增强完成；生产 SSE 的 done 不会等待这里。
        await self.orchestrator.wait_for_background_tasks()
        self.assertTrue(events)
        self.assertTrue({event["type"] for event in events} <= {"metadata", "delta", "done", "error"})
        return events

    def _new_orchestrator(self) -> UnifiedChatOrchestrator:
        """用相同主会话存储创建新的编排器，模拟服务重载或多请求实例。"""
        return UnifiedChatOrchestrator(
            self.main,
            _MultidimAgent(Path(self.temp_dir.name) / "reports-reloaded"),
            NaturalConversationStore(session_store=self.session_store),
            self.smart_agent,
            QueryIntentAgent(
                llm_client=_UnavailableLlm(),
                audit_path=Path(self.temp_dir.name) / "intent-audit-reloaded.jsonl",
                case_provider=self.main.standard_case_store.list_cases,
            ),
        )

    @staticmethod
    def _agent_metadata(events: list[dict], agent_type: str) -> dict:
        """跳过统一入口进度，定位包含业务结果的 Agent 元数据。"""
        return next(
            event
            for event in events
            if event.get("type") == "metadata"
            and event.get("agent_type") == agent_type
            and ("retrieval_mode" in event or "agent_result" in event)
        )

    def test_metric_result_selects_case_for_dependent_analysis(self):
        """极值任务必须把获胜个例而非原始候选集合传给下游分析。"""
        tasks = [
            {
                "id": "task_1", "type": "case_identification", "route": "rag",
                "question": "定位时间范围内候选个例", "depends_on": [],
                "evidence_scope": "structured_cases",
                "conditions": {"years": [2025], "months": [1], "areas": ["山西"]},
            },
            {
                "id": "task_2", "type": "metric_aggregation", "route": "rag",
                "question": "找出最低气温及时间地点", "depends_on": ["task_1"],
                "evidence_scope": "document_metric_facts",
                "conditions": {
                    "years": [2025], "months": [1], "areas": ["山西"],
                    "metric": "air_temperature.minimum", "operator": "min",
                },
            },
            {
                "id": "task_3", "type": "case_analysis", "route": "rag",
                "question": "分析极值所属过程", "depends_on": ["task_2", "task_1"],
                "evidence_scope": "case_chunks", "conditions": {},
            },
        ]
        _, _, context, results, _ = self.orchestrator._retrieve_rag_task_plan(
            tasks, [], top_k=10, top_n=5,
        )
        metric = next(item for item in results if item["type"] == "metric_aggregation")
        analysis = next(item for item in results if item["type"] == "case_analysis")
        self.assertEqual(metric["structured_result"]["winner"]["value_text"], "-25.6℃")
        self.assertEqual(metric["resolved_case_ids"], ["case-jan-1"])
        self.assertEqual(analysis["resolved_case_ids"], ["case-jan-1"])
        self.assertTrue(any("新荣" in block and "-25.6℃" in block for block in context))

    async def test_document_rag_uses_unified_sse_and_persists_result(self):
        """普通问题应走主 RAG，并保存答案、证据和运行编号。"""
        session_id = self.session_store.create_session()["session_id"]
        events = await self._collect("介绍一下山西天气过程", session_id)

        self.assertEqual(events[0]["progress"]["message"], "正在理解问题")
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(self._agent_metadata(events, "document_rag")["agent_type"], "document_rag")
        runs = self.session_store.list_agent_runs(session_id)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["answer"], "普通 RAG 回答")
        self.assertEqual(runs[0]["evidence_chunks"][0]["chunk"]["chunk_id"], "chunk-001")

    async def test_completed_answer_does_not_start_follow_up_model(self):
        """回答完成后不再启动独立后续建议模型，避免额外请求和后台告警。"""
        self.main.followup_llm_client = _SuggestingLlm()
        self.orchestrator.followup_llm_client = self.main.followup_llm_client
        session_id = self.session_store.create_session()["session_id"]

        events = await self._collect("介绍一下山西天气过程", session_id)

        done = events[-1]
        saved = self.session_store.get_message(session_id, done["message_id"])
        self.assertEqual(done["status"], "completed")
        self.assertIsInstance(done["message_id"], int)
        self.assertEqual(done["follow_up_suggestions"], [])
        self.assertEqual(done["follow_up_status"], "none")
        self.assertEqual(saved["follow_up_suggestions"], [])
        self.assertEqual(saved["follow_up_status"], "none")
        self.assertFalse(hasattr(self.main.followup_llm_client, "last_suggestion_input"))
        completion_metadata = events[-2]
        self.assertEqual(completion_metadata["type"], "metadata")
        self.assertEqual(completion_metadata["status"], "completed")
        self.assertEqual(completion_metadata["message_id"], done["message_id"])
        self.assertEqual(completion_metadata["follow_up_suggestions"], [])

    async def test_regenerate_reuses_original_question_without_duplicate_user_message(self):
        """重新生成应排除旧回答，不重复插入原用户消息，并保存版本来源。"""
        session_id = self.session_store.create_session()["session_id"]
        source_user = self.session_store.add_message(session_id, "user", "介绍历史暴雨过程")
        old_answer = self.session_store.add_message(
            session_id,
            "assistant",
            "需要被排除的旧回答",
            metadata={"agent_type": "document_rag", "status": "completed", "version": 1},
        )
        self.session_store.add_message(session_id, "user", "这是旧回答之后的问题")
        self.session_store.add_message(
            session_id,
            "assistant",
            "这是旧回答之后的内容",
            metadata={"agent_type": "document_rag", "status": "completed"},
        )
        payload = SimpleNamespace(
            question="重新生成原回答",
            session_id=session_id,
            top_k=5,
            top_n=3,
            regenerate_from_message_id=old_answer["message_id"],
        )

        events = [event async for event in self.orchestrator.stream(payload)]
        await self.orchestrator.wait_for_background_tasks()

        messages = self.session_store.list_messages(session_id)
        done = events[-1]
        self.assertEqual(sum(item["role"] == "user" for item in messages), 2)
        self.assertEqual(len(messages), 5)
        self.assertEqual(done["regenerated_from_message_id"], old_answer["message_id"])
        self.assertEqual(done["version"], 2)
        self.assertEqual(done["version_group_id"], f"user-{source_user['message_id']}")
        self.assertEqual(messages[-1]["regenerated_from_message_id"], old_answer["message_id"])
        self.assertEqual(messages[-1]["version"], 2)
        self.assertEqual(messages[-1]["version_group_id"], done["version_group_id"])
        regenerated_run = self.session_store.list_agent_runs(session_id)[0]
        self.assertEqual(regenerated_run["question"], "介绍历史暴雨过程")

        # 从第二版再次生成时，后面即使已有其他用户消息，也必须回到最初问题并得到版本 3。
        second_payload = SimpleNamespace(
            question="重新生成原回答",
            session_id=session_id,
            top_k=5,
            top_n=3,
            regenerate_from_message_id=messages[-1]["message_id"],
            regeneration_instruction="补充影响范围",
        )
        second_events = [event async for event in self.orchestrator.stream(second_payload)]
        await self.orchestrator.wait_for_background_tasks()
        second_done = second_events[-1]
        self.assertEqual(second_done["version"], 3)
        self.assertEqual(second_done["version_group_id"], done["version_group_id"])
        self.assertEqual(sum(item["role"] == "user" for item in self.session_store.list_messages(session_id)), 2)
        self.assertEqual(self.session_store.list_agent_runs(session_id)[0]["question"], "介绍历史暴雨过程")

    async def test_feedback_instruction_only_changes_generation_prompt(self):
        """负向反馈约束只进入回答生成，不得污染检索问题和运行审计中的原问题。"""
        session_id = self.session_store.create_session()["session_id"]
        self.session_store.add_message(session_id, "user", "介绍历史暴雨过程")
        old_answer = self.session_store.add_message(
            session_id,
            "assistant",
            "旧回答遗漏了影响范围",
            metadata={"agent_type": "document_rag", "status": "completed", "version": 1},
        )
        llm = _StreamingLlm()
        self.main.llm_client = llm
        payload = SimpleNamespace(
            question="重新生成原回答",
            session_id=session_id,
            top_k=5,
            top_n=3,
            regenerate_from_message_id=old_answer["message_id"],
            regeneration_instruction="遗漏关键信息：补充影响范围",
        )

        events = [event async for event in self.orchestrator.stream(payload)]
        await self.orchestrator.wait_for_background_tasks()

        self.assertEqual(self.main.last_retrieval["question"], "介绍历史暴雨过程")
        self.assertIn("遗漏关键信息：补充影响范围", llm.last_question)
        self.assertEqual(events[-1]["regeneration_instruction"], "遗漏关键信息：补充影响范围")
        regenerated_run = self.session_store.list_agent_runs(session_id)[0]
        self.assertEqual(regenerated_run["question"], "介绍历史暴雨过程")

    async def test_session_feedback_guidance_applies_and_isolated_from_new_session(self):
        """仅当前会话的后续主 RAG 回答读取临时要求，新会话不得继承。"""
        session_id = self.session_store.create_session()["session_id"]
        source_user = self.session_store.add_message(session_id, "user", "介绍历史暴雨过程")
        source_answer = self.session_store.add_message(
            session_id,
            "assistant",
            "上一条回答遗漏了影响范围。",
            metadata={"agent_type": "document_rag", "status": "completed"},
        )
        guidance = self.session_store.create_feedback_guidance(
            session_id,
            source_answer["message_id"],
            "browser-1",
            "遗漏关键信息：请补充影响范围",
        )
        self.session_store.update_feedback_guidance(
            guidance["guidance_id"],
            {
                "status": "active",
                "guidance_prompt": "回答灾害过程时补充已有证据支持的影响范围。",
                "guidance_type": ["answer_completeness"],
                "requested_dimensions": ["影响范围"],
                "needs_same_scope_retrieval": True,
                "scope_change": False,
                "confidence": 0.95,
            },
        )
        llm = _GuidedStreamingLlm()
        self.main.llm_client = llm

        events = await self._collect("介绍历史暴雨过程", session_id)

        self.assertTrue(any(event.get("type") == "done" for event in events))
        self.assertIn("补充已有证据支持的影响范围", llm.guidance_prompt)
        # 临时要求不拼进原始查询，检索仍然只使用用户本轮问题。
        self.assertEqual(self.main.last_retrieval["question"], "介绍历史暴雨过程")
        metadata = self._agent_metadata(events, "document_rag")
        self.assertTrue(metadata["audit"]["session_feedback_guidance"]["applied"])

        new_session = self.session_store.create_session()["session_id"]
        llm.guidance_prompt = "残留值"
        await self._collect("介绍历史暴雨过程", new_session)
        self.assertEqual(llm.guidance_prompt, "")
        self.assertEqual(
            self.session_store.list_feedback_guidances(new_session),
            [],
        )

    async def test_structured_answer_also_uses_final_renderer_and_feedback(self):
        """结构化快速查询不能绕过最终渲染模型和当前会话的临时要求。"""
        session_id = self.session_store.create_session()["session_id"]
        self.session_store.add_message(session_id, "user", "上一次统计回答")
        source_answer = self.session_store.add_message(
            session_id,
            "assistant",
            "希望下一次用表格展示。",
            metadata={"agent_type": "document_rag", "status": "completed"},
        )
        guidance = self.session_store.create_feedback_guidance(
            session_id,
            source_answer["message_id"],
            "browser-1",
            "请使用表格展示统计结果",
        )
        self.session_store.update_feedback_guidance(
            guidance["guidance_id"],
            {
                "status": "active",
                "guidance_prompt": "统计结果优先使用表格展示，并保留每个过程的日期和名称。",
                "guidance_type": ["answer_style"],
                "requested_dimensions": ["表格展示"],
                "needs_same_scope_retrieval": False,
                "scope_change": False,
                "confidence": 0.96,
            },
        )
        llm = _GuidedStreamingLlm()
        self.main.llm_client = llm
        decision = UnifiedRouteDecision(
            "rag",
            "structured",
            "三月份有几个灾害？",
            tasks=[],
        )

        with patch.object(self.orchestrator, "_route", return_value=decision):
            events = await self._collect("三月份有几个灾害？", session_id)

        self.assertIn("<session_feedback_guidance>", llm.guidance_prompt)
        self.assertIn("已核验的结构化查询结果", "\n".join(llm.last_context_blocks))
        self.assertEqual(
            "".join(event.get("text", "") for event in events if event.get("type") == "delta"),
            "第一段第二段",
        )
        saved = self.session_store.get_message(session_id, events[-1]["message_id"])
        self.assertEqual(saved["content"], "第一段第二段")
        saved_metadata = saved["metadata"]
        self.assertEqual(saved_metadata["answer_renderer"], "llm")
        self.assertEqual(saved_metadata["llm_status"], "called_with_structured_facts")
        self.assertTrue(saved_metadata["audit"]["session_feedback_guidance"]["applied"])

    async def test_simple_count_uses_structured_main_rag(self):
        """简单月份计数应通过共享个例集合完成计数和清单，不调用多维或 Smart。"""
        session_id = self.session_store.create_session()["session_id"]
        events = await self._collect("三月份有几个灾害？", session_id)

        metadata = self._agent_metadata(events, "document_rag")
        self.assertEqual(metadata["retrieval_mode"], "document_task_plan")
        self.assertTrue(any("命中个例总数：1" in event.get("text", "") for event in events))
        self.assertEqual(
            [item["type"] for item in metadata["task_plan"]],
            ["case_identification", "aggregate_statistics", "case_listing"],
        )

    def test_annual_statistics_task_graph_reuses_one_structured_case_set(self):
        """计数和清单只消费上游个例集合，不加载仅属于某个月份的向量 chunk。"""
        conditions = {
            "years": [2025],
            "disaster_types": ["高温"],
            "time_scope_type": "calendar_year",
            "output_fields": ["count", "title", "date_range"],
        }
        tasks = [
            {"id": "task_1", "type": "case_identification", "route": "rag", "question": "筛选全年高温个例", "depends_on": [], "evidence_scope": "structured_cases", "conditions": conditions},
            {"id": "task_2", "type": "aggregate_statistics", "route": "rag", "question": "统计个例数量", "depends_on": ["task_1"], "evidence_scope": "structured_cases", "conditions": conditions},
            {"id": "task_3", "type": "case_listing", "route": "rag", "question": "列出名称和时间", "depends_on": ["task_1"], "evidence_scope": "structured_cases", "conditions": conditions},
        ]

        mode, hits, blocks, results, cases = self.orchestrator._retrieve_rag_task_plan(
            tasks, [], top_k=10, top_n=5,
        )

        expected_ids = {"hot-may", "hot-july-1", "hot-july-2"}
        self.assertEqual(mode, "document_task_plan")
        self.assertEqual(hits, [])
        self.assertEqual(set(results[0]["resolved_case_ids"]), expected_ids)
        self.assertEqual(set(results[1]["resolved_case_ids"]), expected_ids)
        self.assertEqual(set(results[2]["resolved_case_ids"]), expected_ids)
        self.assertEqual({item["case_id"] for item in cases}, expected_ids)
        self.assertTrue(any("命中个例总数：3" in block for block in blocks))
        self.assertTrue(any("5月19-21日高温天气" in block for block in blocks))

    def test_case_analysis_retrieves_chunks_for_each_identified_case(self):
        """三个标准个例的特征分析必须各自读取关联正文，并进入统一回答上下文。"""
        # 仅本测试临时加入三月夹具，避免改变其他统计测试的固定命中数量。
        self.main.standard_case_store.cases.extend([
            StandardCase(case_id="march-1", title="1-3日雨雪天气过程", date_range="2025年3月1-3日", disaster_types=["雨雪"], source_pdf="FST2025-3.pdf", source_chunk_ids=["march-chunk-1"]),
            StandardCase(case_id="march-2", title="14-15日暴雪天气过程", date_range="2025年3月14-15日", disaster_types=["暴雪"], source_pdf="FST2025-3.pdf", source_chunk_ids=["march-chunk-2"]),
            StandardCase(case_id="march-3", title="25-28日沙尘天气过程", date_range="2025年3月25-28日", disaster_types=["沙尘"], source_pdf="FST2025-3.pdf", source_chunk_ids=["march-chunk-3"]),
        ])
        tasks = [
            {
                "id": "task_1",
                "type": "case_identification",
                "route": "rag",
                "question": "筛选2025年3月灾害个例",
                "depends_on": [],
                "evidence_scope": "structured_cases",
                "conditions": {"years": [2025], "months": [3], "disaster_types": ["雨雪", "暴雪", "沙尘"]},
            },
            {
                "id": "task_2",
                "type": "case_analysis",
                "route": "rag",
                "question": "分别分析每个个例的天气特征、强度和影响",
                "depends_on": ["task_1"],
                "evidence_scope": "case_chunks",
                "conditions": {"analysis_fields": ["features", "intensity_metrics", "impact"]},
            },
        ]

        mode, hits, blocks, results, cases = self.orchestrator._retrieve_rag_task_plan(
            tasks, [], top_k=10, top_n=5,
        )

        analysis = next(item for item in results if item["type"] == "case_analysis")
        expected_chunks = {"march-chunk-1", "march-chunk-2", "march-chunk-3"}
        self.assertEqual(mode, "document_task_plan")
        self.assertEqual({item["case_id"] for item in analysis["case_evidence"]}, {"march-1", "march-2", "march-3"})
        self.assertEqual(
            {chunk_id for item in analysis["case_evidence"] for chunk_id in item["evidence_chunk_ids"]},
            expected_chunks,
        )
        self.assertTrue(expected_chunks.issubset(set(analysis["evidence_chunk_ids"])))
        self.assertTrue(all(any(chunk_id in block for block in blocks) for chunk_id in expected_chunks))
        self.assertEqual({item["case_id"] for item in cases}, {"march-1", "march-2", "march-3"})

    def test_optional_no_hit_task_stays_in_audit_only(self):
        """附属任务未命中时保留任务状态，但不能把失败说明送入回答上下文。"""
        tasks = [{
            "id": "task_optional",
            "type": "aggregate_statistics",
            "route": "rag",
            "question": "统计2099年未收录过程",
            "depends_on": [],
            "evidence_scope": "structured_cases",
            "conditions": {"years": [2099], "output_fields": ["count"]},
        }]

        _mode, hits, blocks, results, cases = self.orchestrator._retrieve_rag_task_plan(
            tasks,
            [],
            top_k=10,
            top_n=5,
        )

        self.assertEqual(hits, [])
        self.assertEqual(blocks, [])
        self.assertEqual(cases, [])
        self.assertEqual(results[0]["status"], "no_hits")
        self.assertNotIn("task_optional", "\n".join(blocks))

    async def test_multidim_confirmation_reuses_session_and_persists_report(self):
        """多维条件提案和确认应复用主 session_id，并保存 PDF 地址。"""
        session_id = self.session_store.create_session()["session_id"]
        proposal_events = await self._collect("统计2025年3月个例并生成报告", session_id)

        self.assertEqual(proposal_events[-1]["status"], "awaiting_confirmation")
        self.assertIsNotNone(self.multidim_store.pending_proposal(session_id))
        # 直接核对 SQLite 状态，防止后续改动重新引入独立会话文件。
        saved_state = self.session_store.get_agent_state(session_id, "case_multidim")
        self.assertEqual(saved_state["conversation_id"], session_id)
        result_events = await self._collect("确认", session_id)
        self.assertEqual(result_events[-1]["agent_type"], "case_multidim_search")
        self.assertEqual(result_events[-1]["status"], "completed")
        self.assertTrue(result_events[-1]["reports"][0]["url"].endswith(".pdf"))

        runs = self.session_store.list_agent_runs(session_id)
        completed = next(run for run in runs if run["status"] == "completed")
        self.assertEqual(completed["evidence_cases"][0]["case_id"], "case-001")
        self.assertTrue(completed["reports"])
        self.assertIsNone(self.multidim_store.pending_proposal(session_id))
        messages = self.session_store.list_messages(session_id)
        self.assertEqual(messages[-1]["metadata"]["agent_result"]["evidence_cases"][0]["case_id"], "case-001")

    async def test_previous_result_report_reuses_evidence_without_multidim_analysis(self):
        """承接式 PDF 请求只复用上一轮八个个例，不重新检索或调用多维分析。"""
        session_id = self.session_store.create_session()["session_id"]
        cases = [
            {
                "case_id": f"case-{index}",
                "title": f"历史灾害过程 {index}",
                "date_range": f"2025年3月{index}日",
                "source_pdf": "FST2025-3.pdf",
                "analysis": f"第 {index} 个例的已完成分析。",
            }
            for index in range(1, 9)
        ]
        self.session_store.add_message(session_id, "user", "统计并分析八次灾害")
        self.session_store.add_message(
            session_id,
            "assistant",
            "上一轮已完成八个个例分析。",
            metadata={
                "agent_type": "document_rag",
                "run_id": "previous-rag-run",
                "evidence_cases": cases,
                "evidence_chunks": [{"chunk_id": "chunk-001", "content": "真实证据"}],
                "analysis": {"executive_summary": "八个个例的历史分析汇总。"},
            },
        )

        events = await self._collect("把上面这8次灾害弄成一个pdf的报告给我", session_id)

        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["agent_type"], "conversation_report")
        self.assertTrue(events[-1]["reports"][0]["url"].endswith(".pdf"))
        self.assertEqual(self.orchestrator.multidim_agent.reuse_export_calls, 1)
        self.assertEqual(self.orchestrator.multidim_agent.analysis_calls, 0)
        runs = self.session_store.list_agent_runs(session_id)
        report_run = next(run for run in runs if run["agent_type"] == "conversation_report")
        self.assertEqual(len(report_run["evidence_cases"]), 8)
        self.assertEqual(report_run["metadata"]["report_mode"], "from_previous_result")

    async def test_result_postprocessing_phrase_reuses_analysis_after_failed_pdf_fallback(self):
        """“把这个分析汇总成 PDF”应复用真实分析，跳过失败的 PDF 回退消息。"""
        session_id = self.session_store.create_session()["session_id"]
        source_cases = [{
            "case_id": "case-001",
            "title": "历史暴雨过程",
            "date_range": "2025年5月1日",
            "source_pdf": "sample.pdf",
            "analysis": "已完成的个例特征分析。",
        }]
        self.session_store.add_message(session_id, "user", "2025年5月到7月发生了几次灾害，分别是什么灾害，简单分析一下")
        self.session_store.add_message(
            session_id,
            "assistant",
            "已完成三个灾害过程的统计和分析。",
            metadata={
                "agent_type": "document_rag",
                "run_id": "source-analysis-run",
                "question": "2025年5月到7月发生了几次灾害，分别是什么灾害，简单分析一下",
                "evidence_cases": source_cases,
                "evidence_chunks": [{"chunk_id": "chunk-001", "content": "真实分析证据"}],
                "analysis": {"executive_summary": "三个过程分析"},
                "reports": [],
            },
        )
        # 模拟旧版本把第二轮 PDF 请求错误送入主 RAG 后形成的失败回答。
        self.session_store.add_message(session_id, "user", "把这个分析汇总成pdf给我")
        self.session_store.add_message(
            session_id,
            "assistant",
            "作为人工智能助手，我无法直接生成 PDF 文件。",
            metadata={
                "agent_type": "document_rag",
                "run_id": "failed-pdf-fallback",
                "question": "把这个分析汇总成pdf给我",
                "evidence_cases": [{"case_id": "wrong-fallback", "title": "不应被复用的回退结果"}],
                "reports": [],
            },
        )

        events = await self._collect("把这个分析汇总成pdf给我", session_id)

        self.assertEqual(events[-1]["agent_type"], "conversation_report")
        self.assertEqual(events[-1]["reports"][0]["report_mode"], "from_previous_result")
        self.assertEqual(self.orchestrator.multidim_agent.reuse_export_calls, 1)
        self.assertEqual(self.orchestrator.multidim_agent.analysis_calls, 0)
        self.assertEqual(self.orchestrator.multidim_agent.last_reuse_result["answer"], "已完成三个灾害过程的统计和分析。")
        self.assertEqual(self.orchestrator.multidim_agent.last_reuse_report_title, "2025年5月至7月气象灾害分析报告")
        self.assertEqual(events[-1]["reports"][0]["title"], "2025年5月至7月气象灾害分析报告")
        report_run = next(
            run for run in self.session_store.list_agent_runs(session_id)
            if run["agent_type"] == "conversation_report"
        )
        self.assertEqual(report_run["metadata"]["source_run_id"], "source-analysis-run")
        self.assertEqual([item["case_id"] for item in report_run["evidence_cases"]], ["case-001"])

    def test_result_postprocessing_phrase_is_not_only_single_word_cue(self):
        """结果后处理识别采用语义短语，避免单独的“这个”触发新报告。"""
        self.assertTrue(
            self.orchestrator._is_previous_result_report_request("把这个结果整理成PDF", {})
        )
        self.assertTrue(
            self.orchestrator._is_previous_result_report_request("刚才的分析导出成报告", {})
        )
        self.assertFalse(
            self.orchestrator._is_previous_result_report_request("生成一个新的PDF报告", {})
        )

    async def test_multidim_confirmation_survives_new_orchestrator_instance(self):
        """服务重建后“确认”仍读取同一 session_id 的待确认提案。"""
        session_id = self.session_store.create_session()["session_id"]
        proposal_events = await self._collect(
            "统计2025年3月至5月发生了几次灾害，分别是哪几次，有什么特征，并生成PDF",
            session_id,
        )
        self.assertEqual(proposal_events[-1]["status"], "awaiting_confirmation")
        proposal = self.session_store.get_agent_state(session_id, "case_multidim")
        self.assertTrue(proposal and proposal.get("pending", {}).get("proposal_id"))

        reloaded = self._new_orchestrator()
        try:
            events = [
                event
                async for event in reloaded.stream(
                    SimpleNamespace(question="确定", session_id=session_id, top_k=10, top_n=5)
                )
            ]
            await reloaded.wait_for_background_tasks()
        finally:
            await reloaded.wait_for_background_tasks()
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["status"], "completed")
        self.assertTrue(any(event.get("type") == "delta" for event in events))
        self.assertIsNone(self.multidim_store.pending_proposal(session_id))
        completed = [run for run in self.session_store.list_agent_runs(session_id) if run["status"] == "completed"]
        self.assertTrue(completed and completed[0]["reports"])

    async def test_multidim_confirmation_recovers_from_message_when_state_missing(self):
        """Agent 状态被清理时，最近确认卡元数据仍可恢复提案。"""
        session_id = self.session_store.create_session()["session_id"]
        proposal_events = await self._collect("统计2025年3月个例并生成报告", session_id)
        proposal_metadata = next(
            event for event in proposal_events
            if event.get("type") == "metadata" and event.get("status") == "awaiting_confirmation"
        )
        proposal_id = proposal_metadata["proposal_id"]
        self.session_store.delete_agent_state(session_id, "case_multidim")
        self.assertIsNone(self.session_store.get_agent_state(session_id, "case_multidim"))

        reloaded = self._new_orchestrator()
        try:
            recovered = reloaded.multidim_store.pending_proposal(session_id)
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered["proposal_id"], proposal_id)
            events = [
                event
                async for event in reloaded.stream(
                    SimpleNamespace(question="确认", session_id=session_id, top_k=10, top_n=5)
                )
            ]
            await reloaded.wait_for_background_tasks()
        finally:
            await reloaded.wait_for_background_tasks()
        self.assertEqual(events[-1]["status"], "completed")
        self.assertTrue(any(event.get("type") == "metadata" and event.get("status") == "running" for event in events))
        self.assertIsNone(reloaded.multidim_store.pending_proposal(session_id))

    async def test_smart_match_uses_unified_sse_and_persists_evidence(self):
        """相似过程问题应走 Smart，并把证据和研判写入主会话。"""
        session_id = self.session_store.create_session()["session_id"]
        events = await self._collect("当前过程与哪些历史过程相似，给出参考经验", session_id)

        self.assertTrue(any(event.get("agent_type") == "smart_case_match" for event in events))
        self.assertEqual(events[-1]["run_id"], "smart-run-001")
        runs = self.session_store.list_agent_runs(session_id)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["evidence_cases"][0]["case_id"], "case-001")
        self.assertEqual(runs[0]["evidence_chunks"][0]["chunk_id"], "chunk-001")
        messages = self.session_store.list_messages(session_id)
        self.assertEqual(messages[-1]["metadata"]["agent_result"]["matched_cases"][0]["case_id"], "case-001")

    async def test_follow_up_uses_resolved_question_and_previous_case_ids(self):
        """承接问题应先消解为完整问题，再限定上一轮证据个例执行本地检索。"""
        session_id = self.session_store.create_session()["session_id"]
        self.session_store.add_message(session_id, "user", "列出寒潮个例")
        self.session_store.add_message(
            session_id,
            "assistant",
            "一月份有两个寒潮个例。",
            metadata={
                "agent_type": "document_rag",
                "run_id": "previous-run",
                "evidence_cases": [
                    {"case_id": "case-jan-1", "title": "1月14日-15日寒潮天气过程", "date_range": "2025-01-14 to 15"},
                    {"case_id": "case-jan-2", "title": "1月23-26日雨雪寒潮大风天气过程", "date_range": "2025-01-23 to 26"},
                ],
            },
        )

        events = await self._collect("详细分析一下1月份的这两个", session_id)

        metadata = self._agent_metadata(events, "document_rag")
        self.assertEqual(metadata["context_case_ids"], ["case-jan-1", "case-jan-2"])
        self.assertIn("1月14日-15日寒潮天气过程", metadata["resolved_question"])
        self.assertEqual(self.main.last_retrieval["context_case_ids"], ["case-jan-1", "case-jan-2"])
        self.assertEqual(self.session_store.get_agent_state(session_id, "conversation_memory")["covered_message_id"], 4)

    async def test_single_january_case_follow_up_does_not_call_smart(self):
        """上一轮三个一月个例中追问 1 月 23 日时，只检索该库内个例的正文。"""
        session_id = self.session_store.create_session()["session_id"]
        self.session_store.add_message(session_id, "user", "列出一月份的三个天气过程")
        self.session_store.add_message(
            session_id,
            "assistant",
            "一月有三个天气过程。",
            metadata={
                "agent_type": "document_rag",
                "evidence_cases": [case.to_dict() for case in self.main.standard_case_store.list_cases() if "case-jan" in case.case_id],
            },
        )

        events = await self._collect("帮我详细分析一下1月23号的这个灾害情况", session_id)

        metadata = self._agent_metadata(events, "document_rag")
        self.assertEqual(metadata["context_case_ids"], ["case-jan-2"])
        self.assertEqual(self.main.last_retrieval["context_case_ids"], ["case-jan-2"])
        self.assertEqual(self.smart_agent.call_count, 0)

    async def test_compound_rag_question_retrieves_evidence_per_task(self):
        """复合问题应分别检索主体和同 PDF 预警证据，再统一生成回答。"""
        session_id = self.session_store.create_session()["session_id"]
        self.session_store.add_message(session_id, "user", "列出三月暴雨个例")
        self.session_store.add_message(
            session_id,
            "assistant",
            "历史暴雨过程。",
            metadata={
                "agent_type": "document_rag",
                "evidence_cases": [self.main.standard_case_store.list_cases()[0].to_dict()],
            },
        )

        events = await self._collect("分析这个历史过程，并说明发生之前有没有预报预警", session_id)

        metadata = self._agent_metadata(events, "document_rag")
        self.assertEqual(metadata["retrieval_mode"], "document_task_plan")
        task_by_type = {item["type"]: item for item in metadata["task_plan"]}
        self.assertIn("case_analysis", task_by_type)
        self.assertIn("forecast_warning_lookup", task_by_type)
        self.assertIn("chunk-001", task_by_type["case_analysis"]["evidence_chunk_ids"])
        self.assertIn("chunk-warning", task_by_type["forecast_warning_lookup"]["evidence_chunk_ids"])
        saved = self.session_store.list_messages(session_id)[-1]["metadata"]
        self.assertEqual(len(saved["task_plan"]), 2)
        self.assertIn("chunk-warning", saved["task_plan"][1]["evidence_chunk_ids"])
        self.assertEqual(self.smart_agent.call_count, 0)

    def test_warning_table_recall_keeps_header_date_rows_and_continuation(self):
        """同 PDF 预警检索应保留完整表格段，且排除更早章节中的同日期噪声。"""
        chunks = [
            DocumentChunk("sample.pdf", "old", 1, "前文提到2025年7月28日预报效果。"),
            DocumentChunk("sample.pdf", "header", 2, "预警服务情况：本月及时发布气象灾害预警。"),
            DocumentChunk("sample.pdf", "row-time", 3, "07月28日15时25分 暴雨黄色预警。"),
            DocumentChunk("sample.pdf", "row-area", 4, "预警区域：大同、朔州、忻州。"),
            DocumentChunk("sample.pdf", "row-next", 5, "07月28日21时33分 雷暴大风黄色预警。"),
            DocumentChunk("sample.pdf", "summary", 6, "小结：预报预警服务及时。"),
        ]

        hits = self.orchestrator._rank_related_evidence_chunks(
            "查询2025年7月28日过程发生前及过程中的预报预警", chunks, 8,
        )

        self.assertEqual(
            [hit.chunk.chunk_id for hit in hits],
            ["header", "row-time", "row-area", "row-next", "summary"],
        )

    async def test_smart_result_follow_up_on_history_case_returns_to_rag(self):
        """Smart 给出的库内历史匹配结果被继续追问时，应回到限定个例的主 RAG。"""
        session_id = self.session_store.create_session()["session_id"]
        await self._collect("请分析2026年8月新发生的暴雨过程风险", session_id)
        self.assertEqual(self.smart_agent.call_count, 1)

        events = await self._collect("详细分析第一个历史个例", session_id)

        metadata = self._agent_metadata(events, "document_rag")
        self.assertEqual(metadata["context_case_ids"], ["case-001"])
        self.assertEqual(self.smart_agent.call_count, 1)

    async def test_rag_forwards_first_delta_before_stream_finishes(self):
        """主 RAG 应在第二段生成完成前就把第一段交给 SSE 消费者。"""
        self.main.llm_client = _StreamingLlm()
        session_id = self.session_store.create_session()["session_id"]
        payload = SimpleNamespace(question="介绍天气过程", session_id=session_id, top_k=5, top_n=3)
        started = time.perf_counter()
        delta_arrivals = []
        events = []
        async for event in self.orchestrator.stream(payload):
            events.append(event)
            if event.get("type") == "delta":
                delta_arrivals.append((event.get("text", ""), time.perf_counter() - started))
        self.assertEqual(len(delta_arrivals), 2)
        # 直接检查两个模型片段的到达间隔；如果入口先收集完整答案，两段会几乎同时到达。
        self.assertGreaterEqual(delta_arrivals[1][1] - delta_arrivals[0][1], 0.12)
        self.assertEqual("".join(event.get("text", "") for event in events), "第一段第二段")

    async def test_orchestrator_guard_overrides_smart_for_valid_knowledge_case(self):
        """即使意图模块异常返回 Smart，编排器也必须按主库实体二次校验后改回 RAG。"""
        self.orchestrator.intent_agent = _MisroutingIntentAgent()

        decision = self.orchestrator._route("分析这个历史过程的预报效果", False, {})

        self.assertEqual(decision.intent, "rag")
        self.assertEqual(decision.referenced_case_ids, ["case-001"])
        self.assertEqual(decision.trace["orchestrator_guard"], "knowledge_case_forced_to_rag")


if __name__ == "__main__":
    unittest.main()
