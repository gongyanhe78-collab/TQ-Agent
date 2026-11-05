"""应用服务组合根，集中构造查询链路所需的存储、客户端和 Agent。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.agents.case_multidim_search.agent import CaseMultidimSearchAgent
from backend.app.agents.case_multidim_search.core.natural_query import NaturalCaseQueryParser
from backend.app.agents.case_multidim_search.core.natural_query import NaturalConversationStore
from backend.app.agents.case_multidim_search.integrations.local_stores import (
    LocalDocumentChunkStore,
    LocalImageEvidenceStore,
    LocalStandardCaseStore,
)
from backend.app.agents.case_multidim_search.integrations.non_thinking_llm import (
    ensure_non_thinking_client,
)
from backend.app.agents.query_intent import QueryIntentAgent
from backend.app.agents.smart_case_match.agent import SmartCaseMatchAgent
from backend.app.agents.smart_case_match.infrastructure.data_store import LocalCaseDataStore
from backend.app.agents.unified_chat.orchestrator import (
    OrchestratorDependencies,
    UnifiedChatOrchestrator,
)
from backend.app.core.config import Settings
from backend.app.services.document_store import ChromaDocumentChunkStore
from backend.app.services.image_extraction import ImageEvidenceStore
from backend.app.services.model_client import (
    get_embedding_client,
    get_intent_llm_client,
    get_llm_client,
    get_rerank_client,
)
from backend.app.services.query.agent_query_service import AgentQueryService
from backend.app.services.query.chat_service import ChatService
from backend.app.services.query.context_cases import ContextCaseResolver
from backend.app.services.query.document_answer import DocumentQuestionAnswerer
from backend.app.services.query.document_qa import DocumentQA
from backend.app.services.query.document_retrieval import DocumentRetriever
from backend.app.services.query.knowledge_service import KnowledgeService
from backend.app.services.query.result_presenters import QueryResultPresenter
from backend.app.services.query.structured_query import StructuredQueryService
from backend.app.services.feedback_tasks import FeedbackTaskService
from backend.app.services.session_store import SessionStore
from backend.app.services.similar_case_matcher import SimilarCaseMatcher
from backend.app.services.standard_case_store import JsonStandardCaseStore
from backend.app.services.structured_qa import StructuredQuestionAnswerer


@dataclass(slots=True)
class AppServices:
    """应用级依赖容器，只在 FastAPI 生命周期内持有共享服务。"""

    settings: Settings
    embedding_client: Any
    llm_client: Any
    rerank_client: Any
    document_store: ChromaDocumentChunkStore
    image_store: ImageEvidenceStore
    standard_case_store: JsonStandardCaseStore
    session_store: SessionStore
    similar_case_matcher: SimilarCaseMatcher
    structured_answerer: StructuredQuestionAnswerer
    presenter: QueryResultPresenter
    document_qa: DocumentQA
    structured_query: StructuredQueryService
    context_cases: ContextCaseResolver
    agent_query: AgentQueryService
    knowledge_service: KnowledgeService
    chat_service: ChatService
    natural_query_parser: NaturalCaseQueryParser
    multidim_agent: CaseMultidimSearchAgent
    multidim_store: NaturalConversationStore
    multidim_page_store: NaturalConversationStore
    smart_agent: SmartCaseMatchAgent
    intent_agent: QueryIntentAgent
    feedback_tasks: FeedbackTaskService
    _closed: bool = False

    def close(self) -> None:
        """释放支持显式关闭的 Agent；重复关闭保持空操作。"""
        if self._closed:
            return
        self._closed = True
        self.feedback_tasks.close()
        self.smart_agent.close()


def _build_multidim_agent(
    settings: Settings,
    llm_client: Any,
) -> CaseMultidimSearchAgent:
    """使用共享本地知识库构造多维检索 Agent。"""
    data_store = LocalCaseDataStore(settings.smart_case_data_dir)
    return CaseMultidimSearchAgent(
        standard_case_store=LocalStandardCaseStore(data_store),
        document_store=LocalDocumentChunkStore(data_store),
        image_store=LocalImageEvidenceStore(data_store),
        output_dir=settings.case_multidim_runtime_dir / "reports",
        llm_client=ensure_non_thinking_client(llm_client),
    )


def _build_query_services(
    *,
    settings: Settings,
    embedding_client: Any,
    llm_client: Any,
    document_store: ChromaDocumentChunkStore,
    image_store: ImageEvidenceStore,
    standard_case_store: JsonStandardCaseStore,
    session_store: SessionStore,
) -> tuple[
    SimilarCaseMatcher,
    StructuredQuestionAnswerer,
    QueryResultPresenter,
    DocumentQA,
    StructuredQueryService,
    ContextCaseResolver,
    AgentQueryService,
    KnowledgeService,
]:
    """构造非 Agent 的查询、证据和知识库服务。"""
    presenter = QueryResultPresenter(image_store)
    similar_case_matcher = SimilarCaseMatcher(
        image_store=image_store,
        image_filter=lambda image, image_type, data_category: presenter.image_matches_filter(
            image,
            image_type=image_type,
            data_category=data_category,
        ),
    )
    structured_answerer = StructuredQuestionAnswerer(
        image_store=image_store,
        image_to_response=presenter.image_evidence_to_response,
    )
    retriever = DocumentRetriever(
        embedding_client=embedding_client,
        document_store=document_store,
        standard_case_store=standard_case_store,
    )
    document_answerer = DocumentQuestionAnswerer(
        retriever=retriever,
        presenter=presenter,
        llm_client=llm_client,
    )
    document_qa = DocumentQA(retriever=retriever, answerer=document_answerer)
    structured_query = StructuredQueryService(
        standard_case_store=standard_case_store,
        answerer=structured_answerer,
        presenter=presenter,
        document_qa=document_qa,
    )
    context_cases = ContextCaseResolver(standard_case_store)
    agent_query = AgentQueryService(
        standard_case_store=standard_case_store,
        similar_case_matcher=similar_case_matcher,
        presenter=presenter,
        document_qa=document_qa,
        structured_query=structured_query,
        answerer=structured_answerer,
        llm_client=llm_client,
    )
    knowledge_service = KnowledgeService(settings=settings)
    return (
        similar_case_matcher,
        structured_answerer,
        presenter,
        document_qa,
        structured_query,
        context_cases,
        agent_query,
        knowledge_service,
    )


def _orchestrator_factory(
    services: AppServices,
) -> UnifiedChatOrchestrator:
    """为每个流式请求创建独立编排器，隔离重生成和上下文状态。"""
    dependencies = OrchestratorDependencies(
        session_store=services.session_store,
        document_qa=services.document_qa,
        structured_query=services.structured_query,
        context_cases=services.context_cases,
        presenter=services.presenter,
        standard_case_store=services.standard_case_store,
        llm_client=services.llm_client,
        multidim_agent=services.multidim_agent,
        multidim_store=services.multidim_store,
        smart_agent=services.smart_agent,
        intent_agent=services.intent_agent,
    )
    return UnifiedChatOrchestrator(dependencies)


def build_services(settings: Settings) -> AppServices:
    """按依赖方向构造应用服务，构造阶段不发起模型请求。"""
    embedding_client = get_embedding_client()
    llm_client = get_llm_client()
    rerank_client = get_rerank_client()
    document_store = ChromaDocumentChunkStore(
        settings.document_index_dir,
        settings.document_collection_name,
    )
    image_store = ImageEvidenceStore(settings.image_metadata_path)
    standard_case_store = JsonStandardCaseStore(settings.standard_cases_path)
    session_store = SessionStore(settings.session_db_path)
    (
        similar_case_matcher,
        structured_answerer,
        presenter,
        document_qa,
        structured_query,
        context_cases,
        agent_query,
        knowledge_service,
    ) = _build_query_services(
        settings=settings,
        embedding_client=embedding_client,
        llm_client=llm_client,
        document_store=document_store,
        image_store=image_store,
        standard_case_store=standard_case_store,
        session_store=session_store,
    )
    multidim_agent = _build_multidim_agent(settings, llm_client)
    multidim_store = NaturalConversationStore(session_store=session_store)
    multidim_page_store = NaturalConversationStore(
        settings.case_multidim_runtime_dir / "natural_conversations"
    )
    smart_agent = SmartCaseMatchAgent(
        data_dir=settings.smart_case_data_dir,
        embedding_client=embedding_client,
        rerank_client=rerank_client,
        llm_client=llm_client,
    )
    intent_agent = QueryIntentAgent(
        data_dir=settings.smart_case_data_dir,
        llm_client=get_intent_llm_client(),
        case_provider=standard_case_store.list_cases,
    )
    services = AppServices(
        settings=settings,
        embedding_client=embedding_client,
        llm_client=llm_client,
        rerank_client=rerank_client,
        document_store=document_store,
        image_store=image_store,
        standard_case_store=standard_case_store,
        session_store=session_store,
        similar_case_matcher=similar_case_matcher,
        structured_answerer=structured_answerer,
        presenter=presenter,
        document_qa=document_qa,
        structured_query=structured_query,
        context_cases=context_cases,
        agent_query=agent_query,
        knowledge_service=knowledge_service,
        chat_service=ChatService(
            settings=settings,
            session_store=session_store,
            document_qa=document_qa,
            structured_query=structured_query,
            context_cases=context_cases,
            agent_query=agent_query,
            presenter=presenter,
            orchestrator_factory=lambda: _orchestrator_factory(services),
        ),
        natural_query_parser=NaturalCaseQueryParser(
            llm_client=ensure_non_thinking_client(llm_client),
        ),
        multidim_agent=multidim_agent,
        multidim_store=multidim_store,
        multidim_page_store=multidim_page_store,
        smart_agent=smart_agent,
        intent_agent=intent_agent,
        feedback_tasks=FeedbackTaskService(
            session_store=session_store,
            llm_client=llm_client,
        ),
    )
    return services
