"""短期访问令牌的签发和校验。"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import jwt
from jwt import InvalidTokenError


class InvalidAccessToken(ValueError):
    """访问令牌无效、被篡改或已过期。"""


class AccessTokenService:
    """使用 HS256 签发只在前端内存保存的短期访问令牌。"""

    def __init__(
        self,
        *,
        secret: str,
        ttl_seconds: int = 900,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if len(secret) < 32:
            raise ValueError("JWT 密钥长度不能少于 32 个字符")
        self.secret = secret
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def issue(self, user_id: str, role: str) -> str:
        """签发包含用户、角色、jti 和有效期的访问令牌。"""
        issued_at = self._now().timestamp()
        claims = {
            "sub": str(user_id),
            "role": str(role),
            "jti": uuid4().hex,
            "iat": int(issued_at),
            "exp": int(issued_at) + self.ttl_seconds,
        }
        return jwt.encode(claims, self.secret, algorithm="HS256")

    def decode(self, token: str) -> dict[str, Any]:
        """校验签名、有效期和必要声明，失败时抛出统一异常。"""
        try:
            claims = jwt.decode(
                token,
                self.secret,
                algorithms=["HS256"],
                options={"verify_exp": False, "verify_iat": False},
            )
        except InvalidTokenError as exc:
            raise InvalidAccessToken("访问令牌无效") from exc
        required = ("sub", "role", "jti", "iat", "exp")
        if any(not claims.get(name) for name in required):
            raise InvalidAccessToken("访问令牌缺少必要声明")
        try:
            expires_at = int(claims["exp"])
            issued_at = int(claims["iat"])
        except (TypeError, ValueError) as exc:
            raise InvalidAccessToken("访问令牌时间声明无效") from exc
        now = int(self._now().timestamp())
        if expires_at <= now or issued_at > now + 60:
            raise InvalidAccessToken("访问令牌已过期")
        return claims

    def _now(self) -> datetime:
        """返回统一时区的当前时间，便于测试注入时钟。"""
        value = self.clock()
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
