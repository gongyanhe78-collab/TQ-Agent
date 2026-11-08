"""HTTP 路由的认证、管理员权限和跨用户资源隔离测试。"""
from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from backend.app.api.routes import (
    agent,
    auth,
    case_multidim,
    query_intent,
    sessions,
    smart_case_match,
    vectors,
)
from backend.app.agents.case_multidim_search.core import progress as search_progress
from backend.app.agents.case_multidim_search.schemas import (
    CaseAnalysisResponse,
    CaseSearchQuery,
    CaseSearchResponse,
    StructuredCaseSearchRequest,
)
from backend.app.agents.smart_case_match.workflow import progress as match_progress
from backend.app.db.base import Base
from backend.app.db.repositories import AuthRepository
from backend.app.db.session_repository import (
    SharedMessageRepository,
    SqlAlchemySessionStore,
)
from backend.app.security.tokens import AccessTokenService
from backend.app.services.auth_service import AuthService


class _ChatService:
    """记录查询路由实际绑定的用户，不进入真实检索链路。"""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id

    def query_document(self, payload) -> dict:
        return {"user_id": self.user_id, "question": payload.question}

    def query_agent(self, payload) -> dict:
        return {"user_id": self.user_id, "question": payload.question}

    async def stream(self, _payload, _is_disconnected):
        yield {"type": "delta", "text": self.user_id}


class _FeedbackTasks:
    """为路由测试提供可调用的反馈任务占位。"""

    def schedule(self, *_args, **_kwargs) -> None:
        return None


class _KnowledgeService:
    """记录管理员是否成功触发知识库重建。"""

    def __init__(self) -> None:
        self.calls = 0

    def rebuild_documents(self) -> dict:
        self.calls += 1
        return {"rebuilt": True}


class _UserMultidimPageStore:
    """用会话编号暴露多维路由绑定的用户，避免进入真实检索。"""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id

    def reset(self, _conversation_id: str) -> str:
        return f"new-session-for-{self.user_id}"

    def confirm(self, _conversation_id: str, _proposal_id: str, progress_id: str):
        """返回可执行的固定提案，并保留当前用户生成的进度编号。"""
        return (
            StructuredCaseSearchRequest(progress_id=progress_id),
            f"来自 {self.user_id} 的自然语言问题",
        )


class _MultidimAgent:
    """为隔离测试提供用户目录、缓存编号和固定分析响应。"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.answer_ids: set[str] = set()
        self.analysis_calls = 0

    def analyze_structured(self, _request) -> CaseAnalysisResponse:
        """返回最小合法响应，并记录分析调用次数。"""
        self.analysis_calls += 1
        search_response = CaseSearchResponse(
            question="固定问题",
            parsed_query=CaseSearchQuery(),
            retrieval_mode="structured_full",
            result_count=0,
            displayed_result_count=0,
            answer="固定回答",
            results=[],
            aggregations={},
        )
        return CaseAnalysisResponse(
            answer_id="fixed-answer",
            title="固定报告",
            question=search_response.question,
            answer=search_response.answer,
            search_response=search_response,
            audit={},
        )

    def export_pdf_from_answer(
        self,
        answer_id: str,
        filename: str = "",
        report_title: str = "",
        progress_callback=None,
    ) -> Path:
        """只允许导出当前用户目录中已登记的答案缓存。"""
        if answer_id not in self.answer_ids:
            raise FileNotFoundError("answer_id 不存在")
        path = self.output_dir / Path(filename or f"{answer_id}.pdf").name
        path.parent.mkdir(parents=True, exist_ok=True)
        if callable(progress_callback):
            progress_callback(100, "导出完成")
        path.write_bytes(b"%PDF-1.4\n%%EOF")
        return path

    def image_asset_path(self, _image_id: str) -> Path:
        """共享知识库图片接口不参与本组用户输出目录测试。"""
        raise FileNotFoundError("图片不存在")


class _IsolatedServices:
    """把真实认证和 SQLAlchemy 仓储装配到最小 HTTP 应用。"""

    def __init__(self, engine, auth_service: AuthService, runtime_root: Path) -> None:
        self.engine = engine
        self.auth_service = auth_service
        self.settings = SimpleNamespace(
            auth_cookie_secure=False,
            auth_cookie_samesite="lax",
            auth_refresh_cookie_name="refresh_token",
            case_multidim_runtime_dir=runtime_root,
        )
        self.shared_store = SharedMessageRepository(engine)
        self.knowledge_service = _KnowledgeService()
        self.multidim_agent = _MultidimAgent(runtime_root / "shared-reports")
        self._feedback = _FeedbackTasks()
        self._multidim_agents: dict[str, _MultidimAgent] = {}

    def session_store_for(self, user_id: str) -> SqlAlchemySessionStore:
        return SqlAlchemySessionStore(self.engine, user_id)

    def chat_service_for(self, user_id: str) -> _ChatService:
        return _ChatService(user_id)

    def feedback_tasks_for(self, _user_id: str) -> _FeedbackTasks:
        return self._feedback

    def multidim_page_store_for(
        self,
        user_id: str,
    ) -> _UserMultidimPageStore:
        return _UserMultidimPageStore(user_id)

    def multidim_agent_for(self, user_id: str) -> _MultidimAgent:
        """按用户返回独立输出目录，模拟生产服务组合根。"""
        agent = self._multidim_agents.get(user_id)
        if agent is None:
            agent = _MultidimAgent(
                self.settings.case_multidim_runtime_dir
                / "users"
                / user_id
                / "reports"
            )
            self._multidim_agents[user_id] = agent
        return agent

    def close(self) -> None:
        self.engine.dispose()


class RouteIsolationTests(unittest.TestCase):
    """验证所有业务路由都从访问令牌确定用户身份。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        token_service = AccessTokenService(
            secret="route-isolation-secret-that-is-long-enough",
            ttl_seconds=900,
        )
        self.auth_service = AuthService(
            repository=AuthRepository(self.engine),
            access_tokens=token_service,
            refresh_ttl=timedelta(days=7),
        )
        self.auth_service.register("user-a", "StrongPassword-2026")
        self.auth_service.register("user-b", "StrongPassword-2026")
        self.auth_service.register(
            "admin-user",
            "StrongPassword-2026",
            role="admin",
        )
        self.services = _IsolatedServices(
            self.engine,
            self.auth_service,
            Path(self.temp_dir.name) / "runtime",
        )
        app = FastAPI()
        app.state.services = self.services
        for router in (
            auth.router,
            sessions.router,
            agent.router,
            case_multidim.router,
            smart_case_match.router,
            query_intent.router,
            vectors.router,
        ):
            app.include_router(router)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        self.engine.dispose()
        self.temp_dir.cleanup()

    def _access_token(self, username: str) -> str:
        response = self.client.post(
            "/api/auth/login",
            json={"username": username, "password": "StrongPassword-2026"},
        )
        self.assertEqual(response.status_code, 200)
        return response.json()["access_token"]

    @staticmethod
    def _headers(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def test_session_and_query_routes_require_authentication(self) -> None:
        """匿名访问会话和全部查询入口必须统一返回 401。"""
        requests = (
            ("get", "/api/sessions", None),
            ("post", "/api/query", {"question": "匿名查询"}),
            ("post", "/api/agent/query", {"question": "匿名查询"}),
            ("post", "/api/query/stream", {"question": "匿名查询"}),
            ("post", "/api/query-intent/route", {"message": "匿名意图路由"}),
            ("get", "/api/query-intent/knowledge-range", None),
            ("post", "/api/case-multidim/natural/reset", {}),
            ("get", "/api/case-multidim/search/progress/missing", None),
            ("get", "/api/smart-case-match/progress/missing", None),
        )
        for method, path, payload in requests:
            with self.subTest(path=path):
                request_kwargs = {"json": payload} if payload is not None else {}
                response = getattr(self.client, method)(path, **request_kwargs)
                self.assertEqual(response.status_code, 401)

    def test_users_cannot_read_delete_or_reuse_each_others_resources(self) -> None:
        """B 无法读取、删除或借用 A 的会话、消息、运行和分享 token。"""
        token_a = self._access_token("user-a")
        token_b = self._access_token("user-b")
        session_a = self.client.post(
            "/api/sessions",
            json={"title": "A 的会话"},
            headers=self._headers(token_a),
        ).json()
        session_a_id = session_a["session_id"]
        store_a = self.services.session_store_for(
            self.auth_service.repository.get_user_by_username("user-a")["user_id"]
        )
        request_message = store_a.add_message(
            session_a_id,
            "user",
            "分析历史暴雨过程",
        )
        answer = store_a.add_message(
            session_a_id,
            "assistant",
            "用户 A 的回答",
        )

        listed_b = self.client.get(
            "/api/sessions",
            headers=self._headers(token_b),
        )
        self.assertNotIn(
            session_a_id,
            [item["session_id"] for item in listed_b.json()["sessions"]],
        )
        self.assertEqual(
            self.client.get(
                f"/api/sessions/{session_a_id}",
                headers=self._headers(token_b),
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(
                f"/api/sessions/{session_a_id}/messages",
                headers=self._headers(token_b),
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.post(
                f"/api/sessions/{session_a_id}/messages/{answer['message_id']}/share",
                headers=self._headers(token_b),
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.delete(
                f"/api/sessions/{session_a_id}",
                headers=self._headers(token_b),
            ).status_code,
            404,
        )

        shared = self.client.post(
            f"/api/sessions/{session_a_id}/messages/{answer['message_id']}/share",
            headers=self._headers(token_a),
        )
        self.assertEqual(shared.status_code, 200)
        public = self.client.get(
            f"/api/shared-messages/{shared.json()['token']}",
        )
        self.assertEqual(public.status_code, 200)
        self.assertEqual(public.json()["question"], request_message["content"])
        self.assertEqual(public.json()["answer"], answer["content"])

    def test_only_admin_can_rebuild_the_knowledge_base(self) -> None:
        """普通用户重建知识库返回 403，管理员请求可以进入服务层。"""
        user_token = self._access_token("user-a")
        admin_token = self._access_token("admin-user")

        denied = self.client.post(
            "/api/documents/index",
            headers=self._headers(user_token),
        )
        allowed = self.client.post(
            "/api/documents/index",
            headers=self._headers(admin_token),
        )

        self.assertEqual(denied.status_code, 403)
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(self.services.knowledge_service.calls, 1)

    def test_query_identity_comes_from_token_and_ignores_body_user_id(self) -> None:
        """请求体伪造 user_id 不能改变查询服务绑定的认证用户。"""
        token_a = self._access_token("user-a")
        user_a_id = self.auth_service.repository.get_user_by_username("user-a")["user_id"]

        response = self.client.post(
            "/api/query",
            json={"question": "查询历史个例", "user_id": "forged-user"},
            headers=self._headers(token_a),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user_id"], user_a_id)

    def test_multidim_store_is_bound_to_authenticated_user(self) -> None:
        """多维自然语言会话仓储只能按令牌用户分别构造。"""
        token_a = self._access_token("user-a")
        token_b = self._access_token("user-b")
        user_a_id = self.auth_service.repository.get_user_by_username("user-a")["user_id"]
        user_b_id = self.auth_service.repository.get_user_by_username("user-b")["user_id"]

        response_a = self.client.post(
            "/api/case-multidim/natural/reset",
            json={},
            headers=self._headers(token_a),
        )
        response_b = self.client.post(
            "/api/case-multidim/natural/reset",
            json={},
            headers=self._headers(token_b),
        )

        self.assertEqual(response_a.status_code, 200)
        self.assertEqual(response_b.status_code, 200)
        self.assertEqual(
            response_a.json()["conversation_id"],
            f"new-session-for-{user_a_id}",
        )
        self.assertEqual(
            response_b.json()["conversation_id"],
            f"new-session-for-{user_b_id}",
        )

    def test_natural_search_uses_authenticated_user_agent(self) -> None:
        """自然语言检索必须绑定令牌用户，不能引用未定义的请求用户。"""
        token_a = self._access_token("user-a")
        user_a_id = self.auth_service.repository.get_user_by_username("user-a")["user_id"]

        response = self.client.post(
            "/api/case-multidim/natural/search",
            json={
                "conversation_id": "conversation-a",
                "proposal_id": "proposal-a",
                "progress_id": "natural-progress-a",
            },
            headers=self._headers(token_a),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["question"],
            f"来自 {user_a_id} 的自然语言问题",
        )
        self.assertEqual(
            self.services.multidim_agent_for(user_a_id).analysis_calls,
            1,
        )

    def test_multidim_reports_and_thumbnails_are_private(self) -> None:
        """B 不能读取 A 的同名 PDF 报告和报告缩略图。"""
        token_a = self._access_token("user-a")
        token_b = self._access_token("user-b")
        user_a_id = self.auth_service.repository.get_user_by_username("user-a")["user_id"]
        agent_a = self.services.multidim_agent_for(user_a_id)
        report_path = agent_a.output_dir / "private-report.pdf"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(b"%PDF-1.4\n%%EOF")
        thumbnail_path = report_path.with_suffix(".thumbnail.png")
        thumbnail_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        self.assertEqual(
            self.client.get(
                "/api/case-multidim/reports/private-report.pdf",
                headers=self._headers(token_a),
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client.get(
                "/api/case-multidim/reports/private-report.pdf",
                headers=self._headers(token_b),
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(
                "/api/case-multidim/report-thumbnails/private-report.thumbnail.png",
                headers=self._headers(token_a),
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client.get(
                "/api/case-multidim/report-thumbnails/private-report.thumbnail.png",
                headers=self._headers(token_b),
            ).status_code,
            404,
        )

    def test_multidim_answer_cache_and_chart_assets_are_private(self) -> None:
        """B 不能使用 A 的 answer_id 导出 PDF，也不能读取 A 的图表。"""
        token_a = self._access_token("user-a")
        token_b = self._access_token("user-b")
        user_a_id = self.auth_service.repository.get_user_by_username("user-a")["user_id"]
        agent_a = self.services.multidim_agent_for(user_a_id)
        agent_a.answer_ids.add("answer-a")
        chart_path = agent_a.output_dir / "charts" / "private-chart.png"
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        chart_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        allowed_export = self.client.post(
            "/api/case-multidim/export-pdf",
            json={"answer_id": "answer-a", "filename": "answer-a.pdf"},
            headers=self._headers(token_a),
        )
        denied_export = self.client.post(
            "/api/case-multidim/export-pdf",
            json={"answer_id": "answer-a", "filename": "answer-a.pdf"},
            headers=self._headers(token_b),
        )
        allowed_chart = self.client.get(
            "/api/case-multidim/assets/private-chart.png",
            headers=self._headers(token_a),
        )
        denied_chart = self.client.get(
            "/api/case-multidim/assets/private-chart.png",
            headers=self._headers(token_b),
        )

        self.assertEqual(allowed_export.status_code, 200)
        self.assertEqual(denied_export.status_code, 404)
        self.assertEqual(allowed_chart.status_code, 200)
        self.assertEqual(denied_chart.status_code, 404)

    def test_progress_can_only_be_read_by_its_owner(self) -> None:
        """用户 B 即使知道用户 A 的进度编号也不能读取任务阶段。"""
        token_a = self._access_token("user-a")
        token_b = self._access_token("user-b")
        user_a_id = self.auth_service.repository.get_user_by_username("user-a")["user_id"]

        search_progress_id = "route-isolation-search-progress"
        match_alias = "route-isolation-match-alias"
        search_progress.start(search_progress_id, owner_id=user_a_id)
        match_progress.start(
            "route-isolation-match-run",
            alias=match_alias,
            owner_id=user_a_id,
        )
        self.addCleanup(search_progress.fail, search_progress_id, "测试结束")
        self.addCleanup(match_progress.fail, "route-isolation-match-run", "测试结束")

        allowed_search = self.client.get(
            f"/api/case-multidim/search/progress/{search_progress_id}",
            headers=self._headers(token_a),
        )
        denied_search = self.client.get(
            f"/api/case-multidim/search/progress/{search_progress_id}",
            headers=self._headers(token_b),
        )
        allowed_match = self.client.get(
            f"/api/smart-case-match/progress/{match_alias}",
            headers=self._headers(token_a),
        )
        denied_match = self.client.get(
            f"/api/smart-case-match/progress/{match_alias}",
            headers=self._headers(token_b),
        )

        self.assertEqual(allowed_search.status_code, 200)
        self.assertEqual(denied_search.status_code, 404)
        self.assertEqual(allowed_match.status_code, 200)
        self.assertEqual(denied_match.status_code, 404)


if __name__ == "__main__":
    unittest.main()
