"""备份前后数据库行数校验，确保恢复后数据一致。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 恢复演练关注的核心业务表，覆盖用户、会话、消息、运行与分享。
COUNT_TABLES: list[str] = [
    "users",
    "sessions",
    "messages",
    "agent_runs",
    "shared_message_snapshots",
]


def parse_database_url(url: str) -> dict[str, str]:
    """把 SQLAlchemy/psycopg 的 PostgreSQL URL 解析为连接参数。"""
    value = str(url or "").strip()
    if not value:
        raise ValueError("数据库地址不能为空")
    normalized = value
    if normalized.startswith("postgresql+psycopg://"):
        normalized = normalized.replace(
            "postgresql+psycopg://",
            "postgresql://",
            1,
        )
    parts = urlsplit(normalized)
    if parts.scheme not in {"postgresql", "postgres"}:
        raise ValueError(f"不支持的数据库类型：{parts.scheme}")
    return {
        "host": parts.hostname or "127.0.0.1",
        "port": str(parts.port or 5432),
        "user": parts.username or "",
        "password": parts.password or "",
        "database": (parts.path or "/").lstrip("/") or "",
    }


def count_table_sql(table: str) -> str:
    """生成单表行数统计 SQL，表名由固定清单提供。"""
    return f"SELECT COUNT(*) FROM {table}"


def verify_counts(
    runner: Callable[[str], Any],
    tables: list[str] | None = None,
) -> dict[str, int]:
    """按表统计行数，runner 接收 SQL 并返回标量计数。"""
    targets = list(tables or COUNT_TABLES)
    return {
        table: int(runner(count_table_sql(table)))
        for table in targets
    }


def build_count_runner(
    url: str,
    *,
    container: bool,
) -> Callable[[str], int]:
    """构造行数统计执行器；容器模式通过 psql，本地模式通过 psycopg。"""
    parsed = parse_database_url(url)
    if container:
        import subprocess

        def container_runner(sql: str) -> int:
            """在 postgres 容器内执行计数 SQL。"""
            command = [
                "docker",
                "compose",
                "exec",
                "-T",
                "postgres",
                "psql",
                "-U",
                parsed["user"],
                "-d",
                parsed["database"],
                "-t",
                "-A",
                "-c",
                sql,
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )
            return int(result.stdout.strip() or 0)

        return container_runner
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "未安装 psycopg，无法执行行数校验"
        ) from exc
    normalized = url.replace(
        "postgresql+psycopg://",
        "postgresql://",
        1,
    )
    connection = psycopg.connect(normalized)
    try:
        with connection.cursor() as cursor:
            def psycopg_runner(sql: str) -> int:
                """执行计数 SQL 并返回整数结果。"""
                cursor.execute(sql)
                row = cursor.fetchone()
                return int(row[0] if row else 0)

            return psycopg_runner
    finally:
        connection.close()


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数，密钥只通过环境变量传入。"""
    parser = argparse.ArgumentParser(
        description="统计核心业务表的行数，用于备份恢复一致性校验。"
    )
    parser.add_argument(
        "--url",
        default=os.getenv("DATABASE_URL", ""),
        help="PostgreSQL URL；未传入时读取 DATABASE_URL",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出行数结果",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口，只输出行数摘要，不输出密码或消息正文。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("必须提供 --url 或 DATABASE_URL")
    counts = verify_counts(
        build_count_runner(args.url, container=False)
    )
    if args.json:
        print(json.dumps(counts, ensure_ascii=False))
    else:
        for table, count in counts.items():
            print(f"{table}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
