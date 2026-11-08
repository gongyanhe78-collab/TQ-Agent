"""公开分享快照的过期、撤销与存在性边界测试。"""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from backend.app.api.routes import sessions
from backend.app.db.base import Base
from backend.app.db.session_repository import (
    SharedMessageRepository,
    SqlAlchemySessionStore,
)
from backend.app.services.session_store import SessionStore


class _ShareServices:
    """只暴露分享接口所需门面的最小服务容器。"""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.shared_store = SharedMessageRepository(engine)
        self.settings = SimpleNamespace(share_ttl_days=30)
        self.auth_service = self

    def get_current_user(self, token: str) -> dict:
        return {"user_id": "share-user", "role": "user", "token": token}

    def session_store_for(self, user_id: str) -> SqlAlchemySessionStore:
        return SqlAlchemySessionStore(self.engine, user_id)


class _ShareHttpApp:
    """构建绑定内存 PostgreSQL 语义仓储的测试应用。"""

    def __init__(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.store = SqlAlchemySessionStore(self.engine, "share-user")
        self.services = _ShareServices(self.engine)
        self.session_id = self.store.create_session("分享会话")["session_id"]
        self.user_message = self.store.add_message(
            self.session_id,
            "user",
            "请分析2025年5月暴雨过程",
        )
        self.assistant_message = self.store.add_message(
            self.session_id,
            "assistant",
            "公开分享的只读回答。",
            metadata={"agent_type": "document_rag", "status": "completed"},
        )
        app = FastAPI()
        app.state.services = self.services
        app.include_router(sessions.router)
        self.client = TestClient(
            app,
            headers={"Authorization": "Bearer share-token"},
        )

    def close(self) -> None:
        self.engine.dispose()


class ShareExpiryApiTests(unittest.TestCase):
    """验证分享创建、读取、撤销与过期在 HTTP 层的契约。"""

    def setUp(self) -> None:
        self.app_ctx = _ShareHttpApp()
        self.session_id = self.app_ctx.session_id
        self.message_id = self.app_ctx.assistant_message["message_id"]

    def tearDown(self) -> None:
        self.app_ctx.close()

    def test_create_share_returns_token_and_expires_at(self) -> None:
        response = self.app_ctx.client.post(
            f"/api/sessions/{self.session_id}/messages/{self.message_id}/share"
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["token"])
        self.assertIn("expires_at", payload)
        self.assertIn("share_url", payload)

    def test_shared_message_is_readable_before_expiry(self) -> None:
        token = self._create_share()
        response = self.app_ctx.client.get(f"/api/shared-messages/{token}")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "公开分享的只读回答。")
        self.assertNotIn("user_id", body)
        self.assertNotIn("session_id", body)

    def test_revoked_share_returns_403_and_second_revoke_returns_404(self) -> None:
        token = self._create_share()
        revoke = self.app_ctx.client.delete(
            f"/api/sessions/{self.session_id}/messages/{self.message_id}/share"
        )
        self.assertEqual(revoke.status_code, 200)
        self.assertEqual(revoke.json()["revoked"], True)
        after = self.app_ctx.client.get(f"/api/shared-messages/{token}")
        self.assertEqual(after.status_code, 403)
        second = self.app_ctx.client.delete(
            f"/api/sessions/{self.session_id}/messages/{self.message_id}/share"
        )
        self.assertEqual(second.status_code, 404)

    def test_revoke_is_session_scoped(self) -> None:
        other_session = self.app_ctx.store.create_session("其他会话")["session_id"]
        response = self.app_ctx.client.delete(
            f"/api/sessions/{other_session}/messages/{self.message_id}/share"
        )
        self.assertEqual(response.status_code, 404)

    def test_expired_share_returns_403_while_unknown_token_returns_404(self) -> None:
        expired = self.app_ctx.store.create_shared_snapshot(
            self.session_id,
            self.message_id,
            "问题",
            "回答",
            "标题",
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        expired_response = self.app_ctx.client.get(
            f"/api/shared-messages/{expired['token']}"
        )
        self.assertEqual(expired_response.status_code, 403)
        missing = self.app_ctx.client.get("/api/shared-messages/no-such-token")
        self.assertEqual(missing.status_code, 404)

    def _create_share(self) -> str:
        response = self.app_ctx.client.post(
            f"/api/sessions/{self.session_id}/messages/{self.message_id}/share"
        )
        self.assertEqual(response.status_code, 200)
        return response.json()["token"]


class SqliteShareStoreTests(unittest.TestCase):
    """验证 SQLite 旧仓储的分享过期与撤销逻辑。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = SessionStore(Path(self.temp_dir.name) / "sessions.sqlite3")
        self.session_id = self.store.create_session("分享会话")["session_id"]
        self.user_message = self.store.add_message(
            self.session_id,
            "user",
            "请分析2025年5月暴雨过程",
        )
        self.assistant_message = self.store.add_message(
            self.session_id,
            "assistant",
            "SQLite 只读回答。",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_future_expiry_is_readable_and_past_expiry_is_not(self) -> None:
        future = self.store.create_shared_snapshot(
            self.session_id,
            self.assistant_message["message_id"],
            "问题",
            "回答",
            "标题",
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )
        self.assertIsNotNone(self.store.get_shared_snapshot(future["token"]))
        past = self.store.create_shared_snapshot(
            self.session_id,
            self.assistant_message["message_id"],
            "问题",
            "回答",
            "标题",
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        self.assertIsNone(self.store.get_shared_snapshot(past["token"]))

    def test_no_expiry_means_never_expires(self) -> None:
        snapshot = self.store.create_shared_snapshot(
            self.session_id,
            self.assistant_message["message_id"],
            "问题",
            "回答",
            "标题",
        )
        self.assertEqual(snapshot["expires_at"], "")
        self.assertIsNotNone(self.store.get_shared_snapshot(snapshot["token"]))

    def test_meta_distinguishes_missing_from_expired(self) -> None:
        past = self.store.create_shared_snapshot(
            self.session_id,
            self.assistant_message["message_id"],
            "问题",
            "回答",
            "标题",
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        self.assertIsNotNone(self.store.shared_snapshot_meta(past["token"]))
        self.assertIsNone(self.store.shared_snapshot_meta("no-such-token"))

    def test_revoke_marks_snapshot_unreadable(self) -> None:
        snapshot = self.store.create_shared_snapshot(
            self.session_id,
            self.assistant_message["message_id"],
            "问题",
            "回答",
            "标题",
        )
        revoked = self.store.revoke_shared_snapshot(
            self.session_id,
            self.assistant_message["message_id"],
        )
        self.assertTrue(revoked)
        self.assertIsNone(self.store.get_shared_snapshot(snapshot["token"]))
        second = self.store.revoke_shared_snapshot(
            self.session_id,
            self.assistant_message["message_id"],
        )
        self.assertFalse(second)


if __name__ == "__main__":
    unittest.main()
