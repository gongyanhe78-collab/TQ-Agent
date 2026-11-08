"""Alembic 初始迁移和 SQLite 历史数据迁移的集成测试。"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from backend.app.db import models
from backend.app.db.base import Base
from backend.app.db.models import (
    AgentRun,
    MemoryItem,
    MemorySummary,
    Message,
    MessageFeedback,
    MessageIssueReport,
    SessionAgentState,
    SessionFeedbackGuidance,
    SharedMessageSnapshot,
    User,
    UserProfile,
)
from backend.app.services.session_store import SessionStore
from scripts.migrate_sqlite_to_postgres import MigrationOptions, run_migration


PROJECT_ROOT = Path(__file__).resolve().parents[4]
ALEMBIC_INI = PROJECT_ROOT / "backend" / "alembic.ini"


class AlembicInitialMigrationTests(unittest.TestCase):
    """初始迁移必须完整覆盖 ORM 结构且可重复执行。"""

    def test_upgrade_head_covers_all_model_constraints_and_is_repeatable(self) -> None:
        """空库升级后应具备全部表、索引、外键和唯一约束。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "target.sqlite3"
            database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
            config = self._alembic_config(database_url)

            command.upgrade(config, "head")
            command.upgrade(config, "head")

            engine = create_engine(database_url)
            try:
                inspector = inspect(engine)
                actual_tables = set(inspector.get_table_names()) - {"alembic_version"}
                self.assertEqual(actual_tables, set(Base.metadata.tables))
                self._assert_indexes_and_uniques(inspector)
                self._assert_foreign_keys(inspector)
            finally:
                engine.dispose()

    def test_project_root_alembic_command_upgrades_empty_database(self) -> None:
        """项目根目录的验收命令应无需切换目录即可完成空库升级。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "root-command.sqlite3"
            database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
            environment = os.environ.copy()
            environment["DATABASE_URL"] = database_url

            result = subprocess.run(
                [sys.executable, "-m", "alembic", "upgrade", "head"],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            engine = create_engine(database_url)
            try:
                inspector = inspect(engine)
                self.assertIn("users", inspector.get_table_names())
                self.assertIn("sessions", inspector.get_table_names())
                self.assertIn("messages", inspector.get_table_names())
            finally:
                engine.dispose()

    def _assert_indexes_and_uniques(self, inspector) -> None:
        """逐表核对 ORM 声明的索引和唯一约束。"""
        for table_name, table in Base.metadata.tables.items():
            with self.subTest(table=table_name, kind="索引"):
                expected_indexes = {
                    (
                        index.name,
                        tuple(column.name for column in index.columns),
                        bool(index.unique),
                    )
                    for index in table.indexes
                }
                actual_indexes = {
                    (
                        index["name"],
                        tuple(index["column_names"]),
                        bool(index["unique"]),
                    )
                    for index in inspector.get_indexes(table_name)
                }
                self.assertTrue(
                    expected_indexes.issubset(actual_indexes),
                    f"{table_name} 缺少索引：{expected_indexes - actual_indexes}",
                )

            expected_uniques = {
                tuple(column.name for column in constraint.columns)
                for constraint in table.constraints
                if constraint.__class__.__name__ == "UniqueConstraint"
            }
            if expected_uniques:
                with self.subTest(table=table_name, kind="唯一约束"):
                    actual_uniques = {
                        tuple(item["column_names"])
                        for item in inspector.get_unique_constraints(table_name)
                    }
                    self.assertTrue(
                        expected_uniques.issubset(actual_uniques),
                        f"{table_name} 缺少唯一约束：{expected_uniques - actual_uniques}",
                    )

    def _assert_foreign_keys(self, inspector) -> None:
        """外键目标列和级联删除策略必须与 ORM 一致。"""
        for table_name, table in Base.metadata.tables.items():
            expected = {
                (
                    foreign_key.parent.name,
                    foreign_key.column.table.name,
                    foreign_key.column.name,
                    (foreign_key.ondelete or "").upper(),
                )
                for foreign_key in table.foreign_keys
            }
            if not expected:
                continue
            with self.subTest(table=table_name, kind="外键"):
                actual = {
                    (
                        tuple(item["constrained_columns"])[0],
                        item["referred_table"],
                        tuple(item["referred_columns"])[0],
                        (item.get("options", {}).get("ondelete") or "").upper(),
                    )
                    for item in inspector.get_foreign_keys(table_name)
                }
                self.assertEqual(actual, expected)

    @staticmethod
    def _alembic_config(database_url: str) -> Config:
        """构造不依赖真实环境文件或 PostgreSQL 连通的 Alembic 配置。"""
        config = Config(str(ALEMBIC_INI))
        config.set_main_option("script_location", str(ALEMBIC_INI.parent / "alembic"))
        config.set_main_option("sqlalchemy.url", database_url)
        return config


class SqliteMigrationTests(unittest.TestCase):
    """旧 SQLite 数据应可重复迁移到已升级的目标库。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source_path = self.root / "legacy.sqlite3"
        self.target_path = self.root / "target.sqlite3"
        self.target_url = f"sqlite+pysqlite:///{self.target_path.as_posix()}"
        self.source = SessionStore(self.source_path)
        self._seed_legacy_store()
        self._seed_optional_legacy_tables()
        config = AlembicInitialMigrationTests._alembic_config(self.target_url)
        command.upgrade(config, "head")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_dry_run_reconciles_in_transaction_without_commit(self) -> None:
        """dry-run 应输出完整对账结果且不在目标库留下数据。"""
        options = MigrationOptions(
            source=self.source_path,
            target_url=self.target_url,
            legacy_user_id="legacy_user",
            legacy_username="legacy_user",
            legacy_password="local-test-password",
            dry_run=True,
        )

        report = run_migration(options)

        self.assertTrue(report.dry_run)
        self.assertEqual(report.counts["sessions"].source, 3)
        self.assertEqual(report.counts["messages"].source, 7)
        self.assertEqual(report.counts["agent_runs"].source, 3)
        self.assertEqual(report.reconcile_mismatches(), [])
        engine = create_engine(self.target_url)
        try:
            with Session(engine) as session:
                self.assertIsNone(session.get(User, "legacy_user"))
                self.assertEqual(session.scalar(select(models.ChatSession.session_id)), None)
        finally:
            engine.dispose()

    def test_migration_maps_legacy_data_and_preserves_identifiers(self) -> None:
        """历史数据应绑定 legacy_user，并保留业务标识和序列化内容。"""
        source_digest = self._sha256(self.source_path)
        expected_message_ids = [item["message_id"] for item in self.source.list_messages("legacy-session-1")]
        expected_share = self.source.get_shared_snapshot("legacy-share-token")
        expected_runs = self.source.list_agent_runs("legacy-session-1")

        report = self._run_real_migration()

        self.assertFalse(report.dry_run)
        self.assertEqual(report.reconcile_mismatches(), [])
        self.assertEqual(self._sha256(self.source_path), source_digest)
        engine = create_engine(self.target_url)
        try:
            with Session(engine) as session:
                legacy_user = session.get(User, "legacy_user")
                self.assertIsNotNone(legacy_user)
                self.assertEqual(legacy_user.username, "legacy_user")
                self.assertEqual(legacy_user.role, "user")

                profile = session.get(UserProfile, "legacy_user")
                self.assertIsNotNone(profile)
                self.assertEqual(profile.display_name, "历史用户")
                self.assertEqual(profile.preferences, {"focus": ["暴雨", "大风"]})

                legacy_sessions = list(
                    session.scalars(
                        select(models.ChatSession)
                        .where(models.ChatSession.user_id == "legacy_user")
                        .order_by(models.ChatSession.session_id)
                    )
                )
                self.assertEqual(len(legacy_sessions), 3)

                messages = list(
                    session.scalars(
                        select(Message)
                        .where(Message.session_id == "legacy-session-1")
                        .order_by(Message.message_id)
                    )
                )
                self.assertEqual([item.message_id for item in messages], expected_message_ids)
                self.assertEqual(messages[1].message_metadata["report"], "暴雨分析报告")

                agent_run = session.get(AgentRun, expected_runs[0]["run_id"])
                self.assertEqual(agent_run.trace_id, "trace-legacy-1")
                self.assertEqual(agent_run.evidence_cases[0]["case_id"], "case-001")
                self.assertEqual(agent_run.reports[0]["title"], "历史报告 1")

                state = session.get(SessionAgentState, ("legacy-session-1", "case_multidim"))
                self.assertEqual(state.state["pending_confirmation"], True)

                feedback = session.get(
                    MessageFeedback,
                    ("legacy-session-1", expected_message_ids[1], "browser-a"),
                )
                self.assertEqual(feedback.user_id, "legacy_user")
                self.assertEqual(feedback.rating, "needs_improvement")

                guidance = session.scalar(
                    select(SessionFeedbackGuidance).where(
                        SessionFeedbackGuidance.session_id == "legacy-session-1"
                    )
                )
                self.assertEqual(guidance.guidance_type, ["evidence"])
                self.assertTrue(guidance.needs_same_scope_retrieval)
                self.assertTrue(guidance.scope_change)

                issue = session.scalar(
                    select(MessageIssueReport).where(
                        MessageIssueReport.session_id == "legacy-session-1"
                    )
                )
                self.assertIsNotNone(issue)
                self.assertEqual(issue.category, "evidence_missing")
                self.assertEqual(issue.user_id, "legacy_user")

                snapshot = session.get(SharedMessageSnapshot, "legacy-share-token")
                self.assertIsNotNone(snapshot)
                self.assertEqual(snapshot.owner_user_id, "legacy_user")
                self.assertEqual(snapshot.question, expected_share["question"])
                self.assertIsNone(snapshot.revoked_at)
                self.assertIsNone(snapshot.expires_at)
                self.assertEqual(
                    self._source_datetime(snapshot.created_at),
                    datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc),
                )

                memories = list(
                    session.scalars(
                        select(MemoryItem).where(MemoryItem.user_id == "legacy_user")
                    )
                )
                self.assertEqual(len(memories), 1)
                self.assertEqual(memories[0].content, "偏好关注重庆西部暴雨")
                summaries = list(
                    session.scalars(
                        select(MemorySummary).where(MemorySummary.user_id == "legacy_user")
                    )
                )
                self.assertEqual(len(summaries), 1)
                self.assertEqual(summaries[0].covered_message_id, expected_message_ids[-1])
        finally:
            engine.dispose()

    def test_second_run_does_not_duplicate_or_overwrite_data(self) -> None:
        """重复执行只应跳过已存在记录，行数和稳定标识保持不变。"""
        first_report = self._run_real_migration()
        counts_after_first = first_report.counts

        second_report = self._run_real_migration()

        self.assertEqual(second_report.reconcile_mismatches(), [])
        for table_name, first in counts_after_first.items():
            second = second_report.counts[table_name]
            self.assertEqual(second.source, first.source)
            self.assertEqual(second.target, first.target)
            self.assertEqual(second.inserted, 0)
            self.assertEqual(second.skipped, first.source)
        engine = create_engine(self.target_url)
        try:
            with Session(engine) as session:
                self.assertEqual(
                    session.scalar(select(models.ChatSession).where(
                        models.ChatSession.session_id == "legacy-session-1"
                    )).session_id,
                    "legacy-session-1",
                )
                self.assertEqual(
                    len(
                        list(
                            session.scalars(
                                select(Message).where(Message.session_id == "legacy-session-1")
                            )
                        )
                    ),
                    3,
                )
        finally:
            engine.dispose()

    def test_failed_migration_rolls_back_every_target_write(self) -> None:
        """任一记录映射失败时，已刷新的用户和会话也必须一并回滚。"""
        with self._sqlite_writer(self.source_path) as connection:
            second_message_id = connection.execute(
                "SELECT message_id FROM messages ORDER BY message_id LIMIT 1 OFFSET 1"
            ).fetchone()[0]
            connection.execute(
                "UPDATE messages SET metadata = '{' WHERE message_id = ?",
                (second_message_id,),
            )

        with self.assertRaises(ValueError):
            self._run_real_migration()

        engine = create_engine(self.target_url)
        try:
            with Session(engine) as session:
                self.assertIsNone(session.get(User, "legacy_user"))
                self.assertEqual(
                    session.scalar(select(models.ChatSession.session_id)),
                    None,
                )
                self.assertEqual(
                    session.scalar(select(models.Message.message_id)),
                    None,
                )
        finally:
            engine.dispose()

    def test_missing_optional_profile_table_does_not_break_reconciliation(self) -> None:
        """旧库没有画像表时只创建基线用户，不应伪造其他业务数据。"""
        with self._sqlite_writer(self.source_path) as connection:
            connection.execute("DROP TABLE user_profiles")

        report = self._run_real_migration()

        self.assertEqual(report.reconcile_mismatches(), [])
        self.assertEqual(report.counts["user_profiles"].source, 0)
        self.assertEqual(report.counts["user_profiles"].target, 0)
        self.assertEqual(report.counts["user_profiles"].inserted, 0)
        engine = create_engine(self.target_url)
        try:
            with Session(engine) as session:
                self.assertIsNotNone(session.get(User, "legacy_user"))
                self.assertIsNone(session.get(UserProfile, "legacy_user"))
        finally:
            engine.dispose()

    def _run_real_migration(self):
        """使用固定基线参数执行一次正式迁移。"""
        return run_migration(
            MigrationOptions(
                source=self.source_path,
                target_url=self.target_url,
                legacy_user_id="legacy_user",
                legacy_username="legacy_user",
                legacy_password="local-test-password",
                dry_run=False,
            )
        )

    def _seed_legacy_store(self) -> None:
        """通过旧 SessionStore 的真实接口写入代表性历史数据。"""
        for index in range(1, 4):
            session_id = f"legacy-session-{index}"
            created = self.source.create_session(f"历史会话 {index}")
            with self._sqlite_writer(self.source_path) as connection:
                connection.execute(
                    "UPDATE sessions SET session_id = ?, created_at = ?, updated_at = ? WHERE session_id = ?",
                    (
                        session_id,
                        f"2026-10-0{index}T08:30:00+00:00",
                        f"2026-10-0{index}T09:30:00+00:00",
                        created["session_id"],
                    ),
                )
                connection.execute(
                    "UPDATE messages SET session_id = ? WHERE session_id = ?",
                    (session_id, created["session_id"]),
                )

        question = self.source.add_message(
            "legacy-session-1",
            "user",
            "分析重庆西部暴雨过程",
        )
        answer = self.source.add_message(
            "legacy-session-1",
            "assistant",
            "应关注短时强降水和地质灾害。",
            {"report": "暴雨分析报告", "image": "evidence-1.png"},
        )
        third = self.source.add_message(
            "legacy-session-1",
            "user",
            "补充关注大风",
            {"trace_id": "trace-legacy-1"},
        )
        for index in (2, 3):
            self.source.add_message(f"legacy-session-{index}", "user", f"历史问题 {index}")
            self.source.add_message(f"legacy-session-{index}", "assistant", f"历史回答 {index}")

        for index in (1, 2, 3):
            run_id = f"legacy-run-{index}"
            self.source.start_agent_run(
                run_id,
                f"legacy-session-{index}",
                "document_rag",
                f"历史问题 {index}",
                metadata={"trace_id": f"trace-legacy-{index}"},
            )
            self.source.finish_agent_run(
                run_id,
                status="completed",
                answer=f"历史回答 {index}",
                evidence_cases=[{"case_id": f"case-00{index}"}],
                evidence_chunks=[{"chunk_id": f"chunk-00{index}"}],
                images=[{"name": f"image-{index}.png"}],
                reports=[{"title": f"历史报告 {index}"}],
                metadata={"trace_id": f"trace-legacy-{index}", "model": "qwen-plus"},
            )

        self.source.save_agent_state(
            "legacy-session-1",
            "case_multidim",
            {"pending_confirmation": True, "proposal": {"region": "重庆西部"}},
        )
        self.source.set_message_feedback(
            "legacy-session-1",
            answer["message_id"],
            "browser-a",
            "needs_improvement",
            "证据不足",
        )
        guidance = self.source.create_feedback_guidance(
            "legacy-session-1",
            answer["message_id"],
            "browser-a",
            "希望增加站点实况",
        )
        self.source.update_feedback_guidance(
            guidance["guidance_id"],
            {
                "status": "active",
                "guidance_prompt": "优先补充站点证据",
                "guidance_type": ["evidence"],
                "requested_dimensions": ["station"],
                "needs_same_scope_retrieval": True,
                "scope_change": True,
                "confidence": 0.91,
            },
        )
        self.source.add_message_issue(
            "legacy-session-1",
            answer["message_id"],
            "browser-a",
            "evidence_missing",
            "未展示自动站数据",
        )

        with self._sqlite_writer(self.source_path) as connection:
            connection.execute(
                (
                    "INSERT INTO shared_message_snapshots "
                    "(token, session_id, message_id, title, question, answer, created_at, revoked_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                (
                    "legacy-share-token",
                    "legacy-session-1",
                    answer["message_id"],
                    "暴雨分析",
                    question["content"],
                    answer["content"],
                    "2026-10-01T10:00:00+00:00",
                    "",
                ),
            )
            connection.execute(
                "UPDATE messages SET message_id = 101 WHERE message_id = ?",
                (third["message_id"],),
            )

    def _seed_optional_legacy_tables(self) -> None:
        """模拟旧库中可能已经存在的画像和长期记忆表。"""
        with self._sqlite_writer(self.source_path) as connection:
            connection.executescript(
                """
                CREATE TABLE user_profiles (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    preferences TEXT NOT NULL,
                    profile_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE memory_items (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    sensitivity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_session_id TEXT,
                    expires_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE memory_summaries (
                    summary_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    covered_message_id INTEGER NOT NULL,
                    version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                (
                    "INSERT INTO user_profiles VALUES (?, ?, ?, ?, ?, ?)"
                ),
                (
                    "old-local-user",
                    "历史用户",
                    '{"focus": ["暴雨", "大风"]}',
                    3,
                    "2026-10-01T08:00:00+00:00",
                    "2026-10-01T09:00:00+00:00",
                ),
            )
            connection.execute(
                (
                    "INSERT INTO memory_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                (
                    "memory-1",
                    "old-local-user",
                    "preference",
                    "偏好关注重庆西部暴雨",
                    0.87,
                    "normal",
                    "active",
                    "legacy-session-1",
                    "",
                    "2026-10-01T08:00:00+00:00",
                    "2026-10-01T08:00:00+00:00",
                ),
            )
            connection.execute(
                (
                    "INSERT INTO memory_summaries VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                (
                    "summary-1",
                    "old-local-user",
                    "legacy-session-1",
                    "用户关注重庆西部暴雨历史过程。",
                    101,
                    1,
                    "2026-10-01T08:00:00+00:00",
                    "2026-10-01T09:00:00+00:00",
                ),
            )

    def test_migration_cli_can_run_as_direct_script(self) -> None:
        """项目根目录直接执行迁移脚本时也应能导入后端模型。"""
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUTF8"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "migrate_sqlite_to_postgres.py"),
                "--help",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=environment,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dry-run", result.stdout)

    @staticmethod
    def _sha256(path: Path) -> str:
        """计算源库摘要，验证迁移过程没有写回 SQLite。"""
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    @contextmanager
    def _sqlite_writer(path: Path):
        """提供会自动提交并关闭的测试用 SQLite 写连接。"""
        connection = sqlite3.connect(path)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _source_datetime(value: datetime) -> datetime:
        """把 SQLite 返回的无时区时间按 UTC 解释。"""
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


if __name__ == "__main__":
    unittest.main()
