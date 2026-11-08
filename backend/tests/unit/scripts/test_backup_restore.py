"""备份恢复脚本的 URL 解析与行数校验逻辑单元测试。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path as FilePath
from unittest.mock import patch

from scripts.backup_postgres import _backup_file, _pg_dump_command
from scripts.restore_postgres import _latest_backup, _psql_command
from scripts.verify_backup_counts import (
    COUNT_TABLES,
    count_table_sql,
    parse_database_url,
    verify_counts,
)


class ParseDatabaseUrlTests(unittest.TestCase):
    """验证 PostgreSQL URL 解析不泄露密码到输出字段之外。"""

    def test_parses_psycopg_url(self) -> None:
        parsed = parse_database_url(
            "postgresql+psycopg://risk:secret@postgres:5432/risk"
        )
        self.assertEqual(parsed["host"], "postgres")
        self.assertEqual(parsed["port"], "5432")
        self.assertEqual(parsed["user"], "risk")
        self.assertEqual(parsed["password"], "secret")
        self.assertEqual(parsed["database"], "risk")

    def test_parses_plain_postgres_url(self) -> None:
        parsed = parse_database_url(
            "postgresql://risk@127.0.0.1:5433/risk_db"
        )
        self.assertEqual(parsed["port"], "5433")
        self.assertEqual(parsed["database"], "risk_db")

    def test_rejects_sqlite_url(self) -> None:
        with self.assertRaises(ValueError):
            parse_database_url("sqlite+pysqlite:///data/risk.sqlite3")

    def test_rejects_empty_url(self) -> None:
        with self.assertRaises(ValueError):
            parse_database_url("")


class CountVerificationTests(unittest.TestCase):
    """验证行数统计 SQL 与校验执行器。"""

    def test_count_sql_uses_fixed_tables(self) -> None:
        self.assertEqual(
            count_table_sql("sessions"),
            "SELECT COUNT(*) FROM sessions",
        )

    def test_count_tables_cover_core_business(self) -> None:
        for table in (
            "users",
            "sessions",
            "messages",
            "agent_runs",
            "shared_message_snapshots",
        ):
            self.assertIn(table, COUNT_TABLES)

    def test_verify_counts_calls_runner_per_table(self) -> None:
        called: list[str] = []

        def runner(sql: str) -> int:
            """记录调用并按表返回计数。"""
            called.append(sql)
            return 3 if "sessions" in sql else 1

        counts = verify_counts(runner, tables=["sessions", "users"])
        self.assertEqual(counts, {"sessions": 3, "users": 1})
        self.assertEqual(len(called), 2)


class BackupCommandTests(unittest.TestCase):
    """验证 pg_dump/psql 命令行构造。"""

    def test_container_pg_dump_command(self) -> None:
        parsed = {
            "host": "postgres",
            "port": "5432",
            "user": "risk",
            "password": "secret",
            "database": "risk",
        }
        command = _pg_dump_command(parsed, container=True)
        self.assertIn("docker", command)
        self.assertIn("pg_dump", command)
        self.assertIn("-U", command)
        self.assertIn("risk", command)
        # 容器模式不把密码拼进命令行。
        self.assertNotIn("secret", command)

    def test_local_psql_command_contains_url_without_password_leak(self) -> None:
        parsed = {
            "host": "postgres",
            "port": "5432",
            "user": "risk",
            "password": "secret",
            "database": "risk",
        }
        command = _psql_command(parsed, container=False)
        self.assertIn("psql", command)
        url = next(item for item in command if item.startswith("postgresql://"))
        # 本地模式需要密码连接，但脚本输出摘要时不打印该 URL。
        self.assertIn("secret", url)

    def test_backup_file_has_risk_prefix_and_sql_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "scripts.backup_postgres.BACKUP_DIR",
                FilePath(tmp),
            ):
                target = _backup_file()
        self.assertTrue(target.name.startswith("risk-"))
        self.assertTrue(target.name.endswith(".sql"))


class LatestBackupTests(unittest.TestCase):
    """验证最新备份选择逻辑。"""

    def test_missing_backup_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "scripts.restore_postgres.BACKUP_DIR",
                FilePath(tmp),
            ):
                with self.assertRaises(FileNotFoundError):
                    _latest_backup()
