"""认证和用户事实数据的 SQLAlchemy 仓储。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from backend.app.db.models import RefreshToken, User


@dataclass(frozen=True, slots=True)
class RefreshTokenRecord:
    """认证服务需要的刷新令牌只读记录。"""

    token_id: str
    family_id: str
    user_id: str
    token_hash: str
    expires_at: datetime
    revoked_at: datetime | None


class AuthRepository:
    """只通过 SQLAlchemy 访问用户和刷新令牌，不依赖 FastAPI。"""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def create_user(
        self,
        *,
        user_id: str,
        username: str,
        password_hash: str,
        role: str,
        now: datetime,
    ) -> dict[str, Any]:
        """创建用户并返回安全字段，不返回密码哈希。"""
        with Session(self.engine) as session:
            user = User(
                user_id=user_id,
                username=username,
                password_hash=password_hash,
                role=role,
                status="active",
                created_at=now,
                updated_at=now,
            )
            session.add(user)
            session.commit()
            return self._user_dict(user, include_password=True)

    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        """按规范化用户名读取认证所需字段。"""
        with Session(self.engine) as session:
            user = session.scalar(select(User).where(User.username == username))
            return self._user_dict(user, include_password=True) if user else None

    def get_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        """按用户编号读取安全用户信息。"""
        with Session(self.engine) as session:
            user = session.get(User, user_id)
            return self._user_dict(user, include_password=False) if user else None

    def create_refresh_token(
        self,
        *,
        token_id: str,
        family_id: str,
        user_id: str,
        token_hash: str,
        expires_at: datetime,
        now: datetime,
    ) -> None:
        """保存刷新令牌哈希，不保存可用于重建令牌的明文。"""
        with Session(self.engine) as session:
            session.add(
                RefreshToken(
                    token_id=token_id,
                    family_id=family_id,
                    user_id=user_id,
                    token_hash=token_hash,
                    expires_at=expires_at,
                    created_at=now,
                )
            )
            session.commit()

    def get_refresh_token(self, token_id: str) -> RefreshTokenRecord | None:
        """读取刷新令牌状态。"""
        with Session(self.engine) as session:
            row = session.get(RefreshToken, token_id)
            if row is None:
                return None
            return RefreshTokenRecord(
                token_id=row.token_id,
                family_id=row.family_id,
                user_id=row.user_id,
                token_hash=row.token_hash,
                expires_at=row.expires_at,
                revoked_at=row.revoked_at,
            )

    def revoke_refresh_token(self, token_id: str, now: datetime) -> None:
        """撤销单个刷新令牌。"""
        with Session(self.engine) as session:
            row = session.get(RefreshToken, token_id)
            if row is not None and row.revoked_at is None:
                row.revoked_at = now
                session.commit()

    def revoke_token_family(self, family_id: str, now: datetime) -> None:
        """撤销同一令牌族内的全部有效刷新令牌。"""
        with Session(self.engine) as session:
            rows = session.scalars(
                select(RefreshToken).where(
                    RefreshToken.family_id == family_id,
                    RefreshToken.revoked_at.is_(None),
                )
            )
            changed = False
            for row in rows:
                row.revoked_at = now
                changed = True
            if changed:
                session.commit()

    @staticmethod
    def _user_dict(user: User, *, include_password: bool) -> dict[str, Any]:
        """把 ORM 用户转换为服务层字典。"""
        result = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role,
            "status": user.status,
            "created_at": user.created_at.isoformat(),
            "updated_at": user.updated_at.isoformat(),
        }
        if include_password:
            result["password_hash"] = user.password_hash
        return result
