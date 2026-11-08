"""应用服务组合根，集中构造查询链路所需的存储、缓存和 Agent。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import Engine

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
from backend.app.agents.unified_chat.runtime import AgentRuntime
from backend.app.cache.context_cache import CachedSessionStore, RedisContextCache
from backend.app.core.config import Settings
from backend.app.db.repositories import AuthRepository
from backend.app.db.session import build_engine
from backend.app.db.session_repository import (
    SharedMessageRepository,
    SqlAlchemySessionStore,
)
from backend.app.repositories.memory_repository import MemoryRepository
from backend.app.services.memory_service import MemoryService
from backend.app.security.tokens import AccessTokenService
from backend.app.services.auth_service import AuthService
from backend.app.services.feedback_tasks import FeedbackTaskService
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
from backend.app.services.similar_case_matcher import SimilarCaseMatcher
from backend.app.services.standard_case_store import JsonStandardCaseStore
from backend.app.services.structured_qa import StructuredQuestionAnswerer
from backend.app.services.vector.contracts import (
    DocumentVectorStore,
    MemoryVectorStore,
)
from backend.app.services.vector.factory import (
    build_document_vector_store,
    build_memory_vector_store,
)


@dataclass(slots=True)
class AppServices:
    """应用级依赖容器，用户级仓储在首次访问时按身份构造。"""

    settings: Settings
    engine: Engine
    auth_service: AuthService
    shared_store: SharedMessageRepository
    context_cache: RedisContextCache
    embedding_client: Any
    llm_client: Any
    rerank_client: Any
    document_store: DocumentVectorStore
    memory_store: MemoryVectorStore
    image_store: ImageEvidenceStore
    standard_case_store: JsonStandardCaseStore
    similar_case_matcher: SimilarCaseMatcher
    structured_answerer: StructuredQuestionAnswerer
    presenter: QueryResultPresenter
    document_qa: DocumentQA
    structured_query: StructuredQueryService
    context_cases: ContextCaseResolver
    agent_query: AgentQueryService
    knowledge_service: KnowledgeService
    natural_query_parser: NaturalCaseQueryParser
    multidim_agent: CaseMultidimSearchAgent
    smart_agent: SmartCaseMatchAgent
    intent_agent: QueryIntentAgent
    _session_stores: dict[str, CachedSessionStore] = field(default_factory=dict)
    _chat_services: dict[str, ChatService] = field(default_factory=dict)
    _feedback_tasks: dict[str, FeedbackTaskService] = field(default_factory=dict)
    _multidim_page_stores: dict[str, NaturalConversationStore] = field(default_factory=dict)
    _multidim_agents: dict[str, CaseMultidimSearchAgent] = field(default_factory=dict)
    _memory_services: dict[str, MemoryService] = field(default_factory=dict)
    _closed: bool = False

    def session_store_for(self, user_id: str) -> CachedSessionStore:
        """返回绑定用户身份、缓存优先且以 PostgreSQL 为权威的会话仓储。"""
        normalized = self._user_id(user_id)
        store = self._session_stores.get(normalized)
        if store is None:
            authority = SqlAlchemySessionStore(self.engine, normalized)
            store = CachedSessionStore(
                authority,
                cache=self.context_cache,
                user_id=normalized,
            )
            self._session_stores[normalized] = store
        return store

    def feedback_tasks_for(self, user_id: str) -> FeedbackTaskService:
        """返回绑定用户会话仓储的反馈后台任务管理器。"""
        normalized = self._user_id(user_id)
        service = self._feedback_tasks.get(normalized)
        if service is None:
            service = FeedbackTaskService(
                session_store=self.session_store_for(normalized),
                llm_client=self.llm_client,
            )
            self._feedback_tasks[normalized] = service
        return service

    def multidim_page_store_for(self, user_id: str) -> NaturalConversationStore:
        """返回仅属于当前用户的多维页面提案存储。"""
        normalized = self._user_id(user_id)
        store = self._multidim_page_stores.get(normalized)
        if store is None:
            directory = (
                self.settings.case_multidim_runtime_dir
                / "users"
                / normalized
                / "natural_conversations"
            )
            store = NaturalConversationStore(directory)
            self._multidim_page_stores[normalized] = store
        return store

    def multidim_agent_for(self, user_id: str) -> CaseMultidimSearchAgent:
        """返回绑定独立输出目录和缓存文件的用户级多维检索 Agent。"""
        normalized = self._user_id(user_id)
        agent = self._multidim_agents.get(normalized)
        if agent is None:
            output_dir = (
                self.settings.case_multidim_runtime_dir
                / "users"
                / normalized
                / "reports"
            )
            agent = _build_multidim_agent(
                self.settings,
                self.llm_client,
                output_dir=output_dir,
            )
            self._multidim_agents[normalized] = agent
        return agent

    def chat_service_for(self, user_id: str) -> ChatService:
        """返回全部查询链路均绑定当前用户的统一查询服务。"""
        normalized = self._user_id(user_id)
        service = self._chat_services.get(normalized)
        if service is None:
            session_store = self.session_store_for(normalized)
            service = ChatService(
                settings=self.settings,
                session_store=session_store,
                document_qa=self.document_qa,
                structured_query=self.structured_query,
                context_cases=self.context_cases,
                agent_query=self.agent_query,
                presenter=self.presenter,
                orchestrator_factory=lambda runtime: self._orchestrator_factory(
                    session_store,
                    runtime,
                ),
            )
            self._chat_services[normalized] = service
        return service

    def memory_service_for(self, user_id: str) -> MemoryService:
        """返回绑定当前用户、以 PostgreSQL 为权威的长期记忆服务。"""
        normalized = self._user_id(user_id)
        service = self._memory_services.get(normalized)
        if service is None:
            service = MemoryService(
                repository=MemoryRepository(self.engine),
                vector_store=self.memory_store,
                cache=self.context_cache,
                embedding_client=self.embedding_client,
            )
            self._memory_services[normalized] = service
        return service

    def close(self) -> None:
        """释放后台任务、Agent 和数据库连接；重复关闭保持空操作。"""
        if self._closed:
            return
        self._closed = True
        for service in self._feedback_tasks.values():
            service.close()
        for agent in (self.smart_agent, self.intent_agent):
            close = getattr(agent, "close", None)
            if callable(close):
                close()
        document_store = getattr(self, "document_store", None)
        if document_store is not None:
            document_store.close()
        memory_store = getattr(self, "memory_store", None)
        if memory_store is not None:
            memory_store.close()
        self.engine.dispose()

    def _orchestrator_factory(
        self,
        session_store: CachedSessionStore,
        runtime: AgentRuntime,
    ) -> UnifiedChatOrchestrator:
        """为当前用户创建请求级编排器，隔离上下文和会话状态。"""
        dependencies = OrchestratorDependencies(
            session_store=session_store,
            document_qa=self.document_qa,
            structured_query=self.structured_query,
            context_cases=self.context_cases,
            presenter=self.presenter,
            standard_case_store=self.standard_case_store,
            llm_client=self.llm_client,
            multidim_agent=self.multidim_agent_for(session_store.user_id),
            multidim_store=NaturalConversationStore(session_store=session_store),
            smart_agent=self.smart_agent,
            intent_agent=self.intent_agent,
            memory_service=self.memory_service_for(session_store.user_id),
        )
        return UnifiedChatOrchestrator(dependencies, runtime=runtime)

    @staticmethod
    def _user_id(user_id: str) -> str:
        """校验用户级服务工厂不能接受空身份。"""
        normalized = str(user_id or "").strip()
        if not normalized:
            raise ValueError("user_id 不能为空")
        return normalized


def _build_multidim_agent(
    settings: Settings,
    llm_client: Any,
    output_dir: Path | None = None,
) -> CaseMultidimSearchAgent:
    """使用共享本地知识库构造多维检索 Agent。"""
    data_store = LocalCaseDataStore(settings.smart_case_data_dir)
    return CaseMultidimSearchAgent(
        standard_case_store=LocalStandardCaseStore(data_store),
        document_store=LocalDocumentChunkStore(data_store),
        image_store=LocalImageEvidenceStore(data_store),
        output_dir=output_dir or settings.case_multidim_runtime_dir / "reports",
        llm_client=ensure_non_thinking_client(llm_client),
    )


def _build_query_services(
    *,
    settings: Settings,
    embedding_client: Any,
    llm_client: Any,
    document_store: DocumentVectorStore,
    image_store: ImageEvidenceStore,
    standard_case_store: JsonStandardCaseStore,
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
    """构造不持有用户身份、可安全共享的查询和知识库服务。"""
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


def build_services(settings: Settings) -> AppServices:
    """按依赖方向构造应用服务，构造阶段不建表也不发起模型请求。"""
    engine = build_engine(settings)
    auth_service = AuthService(
        repository=AuthRepository(engine),
        access_tokens=AccessTokenService(
            secret=settings.auth_jwt_secret,
            ttl_seconds=settings.auth_access_ttl_seconds,
        ),
        refresh_ttl=timedelta(seconds=settings.auth_refresh_ttl_seconds),
    )
    context_cache = RedisContextCache(
        url=settings.redis_url,
        prefix=settings.redis_key_prefix,
        ttl_seconds=settings.redis_context_ttl_seconds,
        enabled=settings.redis_cache_enabled,
    )
    embedding_client = get_embedding_client()
    llm_client = get_llm_client()
    rerank_client = get_rerank_client()
    document_store = build_document_vector_store(
        settings,
        embedding_client=embedding_client,
    )
    memory_store = build_memory_vector_store(settings)
    image_store = ImageEvidenceStore(settings.image_metadata_path)
    standard_case_store = JsonStandardCaseStore(settings.standard_cases_path)
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
    )
    return AppServices(
        settings=settings,
        engine=engine,
        auth_service=auth_service,
        shared_store=SharedMessageRepository(engine),
        context_cache=context_cache,
        embedding_client=embedding_client,
        llm_client=llm_client,
        rerank_client=rerank_client,
        document_store=document_store,
        memory_store=memory_store,
        image_store=image_store,
        standard_case_store=standard_case_store,
        similar_case_matcher=similar_case_matcher,
        structured_answerer=structured_answerer,
        presenter=presenter,
        document_qa=document_qa,
        structured_query=structured_query,
        context_cases=context_cases,
        agent_query=agent_query,
        knowledge_service=knowledge_service,
        natural_query_parser=NaturalCaseQueryParser(
            llm_client=ensure_non_thinking_client(llm_client),
        ),
        multidim_agent=_build_multidim_agent(settings, llm_client),
        smart_agent=SmartCaseMatchAgent(
            data_dir=settings.smart_case_data_dir,
            embedding_client=embedding_client,
            rerank_client=rerank_client,
            llm_client=llm_client,
        ),
        intent_agent=QueryIntentAgent(
            data_dir=settings.smart_case_data_dir,
            llm_client=get_intent_llm_client(),
            case_provider=standard_case_store.list_cases,
        ),
    )
