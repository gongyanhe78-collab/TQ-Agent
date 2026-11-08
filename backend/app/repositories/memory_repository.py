"""长期记忆权威仓储：所有读、删、重建来源都强制绑定 user_id。"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from backend.app.db.base import utc_now
from backend.app.db.models import MemoryItem


def _as_utc(value: datetime | None) -> datetime | None:
    """把数据库返回的无时区时间按 UTC 解释。"""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _iso(value: datetime | None) -> str | None:
    """把时间统一序列化为带时区的 ISO 字符串。"""
    normalized = _as_utc(value)
    return normalized.isoformat() if normalized else None


class MemoryRepository:
    """通过 SQLAlchemy 访问长期记忆事实，不依赖 FastAPI。"""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def create(
        self,
        *,
        user_id: str,
        content: str,
        memory_type: str = "preference",
        confidence: float = 0.0,
        sensitivity: str = "normal",
        status: str = "active",
        source_session_id: str | None = None,
        source_message_id: int | None = None,
        version: int = 1,
        metadata: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """创建一条记忆事实并返回安全字段。"""
        memory_id = f"memory-{uuid4().hex[:16]}"
        created_at = now or utc_now()
        with Session(self.engine) as session:
            row = MemoryItem(
                memory_id=memory_id,
                user_id=str(user_id),
                memory_type=str(memory_type),
                content=str(content),
                confidence=max(0.0, min(1.0, float(confidence))),
                sensitivity=str(sensitivity),
                status=str(status),
                source_session_id=source_session_id,
                source_message_id=source_message_id,
                version=max(1, int(version)),
                memory_metadata=dict(metadata or {}),
                created_at=created_at,
                updated_at=created_at,
            )
            session.add(row)
            session.commit()
            return self._memory_dict(row)

    def get(self, user_id: str, memory_id: str) -> dict[str, Any] | None:
        """按用户读取单条记忆，跨用户访问返回 None。"""
        with Session(self.engine) as session:
            row = self._memory(session, user_id, memory_id)
            return self._memory_dict(row) if row else None

    def list_active(
        self,
        user_id: str,
        limit: int = 20,
        memory_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """列出当前用户的 active 记忆，排除已删除和已过期。"""
        with Session(self.engine) as session:
            query = (
                select(MemoryItem)
                .where(
                    MemoryItem.user_id == str(user_id),
                    MemoryItem.status == "active",
                    MemoryItem.deleted_at.is_(None),
                )
                .order_by(MemoryItem.updated_at.desc())
            )
            if memory_type:
                query = query.where(MemoryItem.memory_type == memory_type)
            rows = session.scalars(query.limit(max(1, int(limit))))
            return [self._memory_dict(row) for row in rows]

    def list_candidates(
        self,
        user_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """列出当前用户的候选记忆，供异步确认流程使用。"""
        with Session(self.engine) as session:
            rows = session.scalars(
                select(MemoryItem)
                .where(
                    MemoryItem.user_id == str(user_id),
                    MemoryItem.status == "candidate",
                    MemoryItem.deleted_at.is_(None),
                )
                .order_by(MemoryItem.created_at.desc())
                .limit(max(1, int(limit)))
            )
            return [self._memory_dict(row) for row in rows]

    def soft_delete(self, user_id: str, memory_id: str) -> bool:
        """软删除当前用户记忆并保留最小审计信息。"""
        now = utc_now()
        with Session(self.engine) as session:
            row = self._memory(session, user_id, memory_id)
            if row is None:
                return False
            row.status = "deleted"
            row.deleted_at = now
            row.updated_at = now
            session.commit()
            return True

    def update_status(
        self,
        user_id: str,
        memory_id: str,
        status: str,
        now: datetime | None = None,
    ) -> bool:
        """更新当前用户记忆状态，跨用户访问返回 False。"""
        updated_at = now or utc_now()
        with Session(self.engine) as session:
            row = self._memory(session, user_id, memory_id)
            if row is None:
                return False
            row.status = str(status)
            row.updated_at = updated_at
            session.commit()
            return True

    def rebuildable_active(
        self,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """返回可重建记忆索引的全部 active 记忆，供向量重建使用。"""
        with Session(self.engine) as session:
            query = select(MemoryItem).where(
                MemoryItem.status == "active",
                MemoryItem.deleted_at.is_(None),
            )
            if user_id:
                query = query.where(MemoryItem.user_id == str(user_id))
            rows = session.scalars(query.order_by(MemoryItem.updated_at.desc()))
            return [self._memory_dict(row) for row in rows]

    def _memory(
        self,
        session: Session,
        user_id: str,
        memory_id: str,
    ) -> MemoryItem | None:
        """读取归属于当前用户的记忆行，边界在 SQL 层生效。"""
        return session.scalar(
            select(MemoryItem).where(
                MemoryItem.user_id == str(user_id),
                MemoryItem.memory_id == str(memory_id),
            )
        )

    def _memory_dict(self, row: MemoryItem) -> dict[str, Any]:
        """把记忆行投影为不暴露内部实现的安全字典。"""
        return {
            "memory_id": row.memory_id,
            "user_id": row.user_id,
            "memory_type": row.memory_type,
            "content": row.content,
            "confidence": float(row.confidence or 0.0),
            "sensitivity": row.sensitivity,
            "status": row.status,
            "source_session_id": row.source_session_id,
            "source_message_id": row.source_message_id,
            "version": int(row.version or 1),
            "metadata": dict(row.memory_metadata or {}),
            "expires_at": _iso(row.expires_at),
            "deleted_at": _iso(row.deleted_at),
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
        }
