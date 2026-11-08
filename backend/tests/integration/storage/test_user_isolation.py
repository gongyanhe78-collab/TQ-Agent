"""PostgreSQL 兼容仓储的用户数据隔离测试。"""
from __future__ import annotations

import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.db.base import Base
from backend.app.db.models import User
from backend.app.db.session_repository import SqlAlchemySessionStore


class SqlAlchemySessionStoreIsolationTests(unittest.TestCase):
    """同名会话、消息和分享操作必须始终按用户隔离。"""

    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        with Session(self.engine) as session:
            for user_id in ("user-a", "user-b"):
                session.add(
                    User(
                        user_id=user_id,
                        username=user_id,
                        password_hash="not-used",
                        role="user",
                        status="active",
                    )
                )
            session.commit()
        self.store_a = SqlAlchemySessionStore(self.engine, "user-a")
        self.store_b = SqlAlchemySessionStore(self.engine, "user-b")

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_lists_and_resource_lookups_are_user_scoped(self) -> None:
        """用户 B 不能看到、读取、更新或删除用户 A 的会话。"""
        session_a = self.store_a.create_session("A 的会话")
        session_b = self.store_b.create_session("B 的会话")

        self.assertEqual(
            [item["session_id"] for item in self.store_a.list_sessions()],
            [session_a["session_id"]],
        )
        self.assertEqual(
            [item["session_id"] for item in self.store_b.list_sessions()],
            [session_b["session_id"]],
        )
        self.assertIsNone(self.store_b.get_session(session_a["session_id"]))
        with self.assertRaises(KeyError):
            self.store_b.update_session_title(session_a["session_id"], "越权")
        self.assertFalse(self.store_b.delete_session(session_a["session_id"]))
        self.assertIsNotNone(self.store_a.get_session(session_a["session_id"]))

    def test_message_feedback_run_and_share_are_user_scoped(self) -> None:
        """消息、反馈、运行和分享读取都不能跨用户访问。"""
        session_a = self.store_a.create_session("A 的会话")
        user_message = self.store_a.add_message(
            session_a["session_id"],
            "user",
            "分析暴雨过程",
        )
        assistant_message = self.store_a.add_message(
            session_a["session_id"],
            "assistant",
            "回答正文",
        )
        message_id = assistant_message["message_id"]
        run = self.store_a.start_agent_run(
            "run-a",
            session_a["session_id"],
            "document_rag",
            "分析暴雨过程",
        )
        snapshot = self.store_a.create_shared_snapshot(
            session_a["session_id"],
            message_id,
            user_message["content"],
            assistant_message["content"],
            "暴雨过程",
        )

        self.assertIsNone(self.store_b.get_message(session_a["session_id"], message_id))
        self.assertIsNone(self.store_b.get_agent_run(run["run_id"]))
        self.assertEqual(self.store_b.list_agent_runs(session_a["session_id"]), [])
        session_b = self.store_b.create_session("B 的会话")
        with self.assertRaises(KeyError):
            self.store_b.start_agent_run(
                run["run_id"],
                session_b["session_id"],
                "document_rag",
                "尝试复用运行编号",
            )
        self.assertEqual(
            self.store_b.get_message_feedback(session_a["session_id"], message_id, "browser"),
            {"rating": "none", "reason": ""},
        )
        with self.assertRaises(KeyError):
            self.store_b.set_message_feedback(
                session_a["session_id"],
                message_id,
                "browser",
                "love",
            )
        self.assertIsNotNone(self.store_a.get_shared_snapshot(snapshot["token"]))


if __name__ == "__main__":
    unittest.main()
