"""SQLAlchemy 声明基类和统一时间工具。"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import DeclarativeBase


def utc_now() -> datetime:
    """返回带 UTC 时区的当前时间，避免本地时区写入事实表。"""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """全部数据库模型的声明基类。"""
