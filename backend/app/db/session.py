"""SQLAlchemy 引擎构造与数据库连通性检查。"""
from __future__ import annotations

from sqlalchemy import Engine, create_engine, text

from backend.app.core.config import Settings


def build_engine(settings: Settings) -> Engine:
    """按环境配置创建数据库引擎，导入阶段不连接数据库。"""
    url = str(settings.database_url or "").strip()
    if not url:
        raise ValueError("DATABASE_URL 不能为空")
    options: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **options)


def check_database(engine: Engine) -> None:
    """通过轻量查询验证数据库连接。"""
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
