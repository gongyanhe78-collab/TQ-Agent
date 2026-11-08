"""Alembic 在线和离线迁移环境。"""
from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, event, pool
from sqlalchemy.engine import Connection

from backend.app.db import models
from backend.app.db.base import Base


config = context.config
target_metadata = Base.metadata

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)


def _normalize_database_url(value: str) -> str:
    """把常见 PostgreSQL 地址转换为已安装的 psycopg 驱动。"""
    url = str(value or "").strip()
    if not url:
        raise RuntimeError("迁移需要 DATABASE_URL 或 sqlalchemy.url 配置")
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def _database_url() -> str:
    """只读取进程环境或 Alembic 配置，不隐式加载 .env 文件。"""
    return _normalize_database_url(
        os.getenv("DATABASE_URL", "") or config.get_main_option("sqlalchemy.url", "")
    )


def _configure_context(
    connection: Connection | None = None,
    *,
    url: str | None = None,
) -> None:
    """使用统一元数据配置在线或离线迁移上下文。"""
    options = {
        "target_metadata": target_metadata,
        "compare_type": True,
        "include_schemas": False,
    }
    if connection is None:
        context.configure(
            url=url,
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
            **options,
        )
        return
    options["connection"] = connection
    options["render_as_batch"] = connection.dialect.name == "sqlite"
    context.configure(**options)


def run_migrations_offline() -> None:
    """不连接数据库，生成可审计的迁移 SQL。"""
    _configure_context(url=_database_url())
    with context.begin_transaction():
        context.run_migrations()


def _run_online_migrations(connection: Connection) -> None:
    """在调用方提供的连接事务内执行迁移。"""
    _configure_context(connection)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """连接目标库并执行结构迁移。"""
    supplied_connection = config.attributes.get("connection")
    if supplied_connection is not None:
        _run_online_migrations(supplied_connection)
        return

    database_url = _database_url()
    options: dict = {"pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        options["poolclass"] = pool.NullPool
    engine = create_engine(database_url, **options)
    if database_url.startswith("sqlite"):
        event.listen(
            engine,
            "connect",
            lambda dbapi_connection, _record: dbapi_connection.execute(
                "PRAGMA foreign_keys=ON"
            ),
        )
    with engine.connect() as connection:
        _run_online_migrations(connection)
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
