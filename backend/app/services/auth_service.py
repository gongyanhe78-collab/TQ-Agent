"""本地账号注册、登录、刷新和登出业务。"""
from __future__ import annotations

import hashlib
import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from backend.app.db.repositories import AuthRepository, RefreshTokenRecord
from backend.app.security.passwords import hash_password, verify_password
from backend.app.security.tokens import AccessTokenService, InvalidAccessToken

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,63}$")


class AuthError(ValueError):
    """认证业务错误基类。"""


class InvalidCredentials(AuthError):
    """账号、密码或刷新令牌无效。"""


class UserAlreadyExists(AuthError):
    """注册用户名已存在。"""


class InvalidRegistration(AuthError):
    """注册字段不满足本地账号规则。"""


@dataclass(frozen=True, slots=True)
class AuthTokens:
    """登录或刷新成功后返回的令牌集合。"""

    access_token: str
    refresh_token: str
    expires_in: int
    user: dict


class AuthService:
    """实现账号生命周期和刷新令牌轮换。"""

    def __init__(
        self,
        *,
        repository: AuthRepository,
        access_tokens: AccessTokenService,
        refresh_ttl: timedelta = timedelta(days=7),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.access_tokens = access_tokens
        self.refresh_ttl = refresh_ttl
        self.clock = clock or (lambda: datetime.now(UTC))

    def register(
        self,
        username: str,
        password: str,
        *,
        role: str = "user",
    ) -> dict:
        """注册本地账号，用户名唯一且角色只允许 user/admin。"""
        normalized = self._validate_registration(username, password, role)
        if self.repository.get_user_by_username(normalized):
            raise UserAlreadyExists("用户名已存在")
        user_id = str(uuid4())
        self.repository.create_user(
            user_id=user_id,
            username=normalized,
            password_hash=hash_password(password),
            role=role,
            now=self._now(),
        )
        user = self.repository.get_user_by_id(user_id)
        if user is None:
            raise RuntimeError("账号创建后无法读取")
        return user

    def login(self, username: str, password: str) -> AuthTokens:
        """校验账号密码并签发一组新令牌。"""
        normalized = str(username or "").strip()
        user = self.repository.get_user_by_username(normalized)
        if user is None or user.get("status") != "active":
            verify_password("$argon2id$invalid", str(password or ""))
            raise InvalidCredentials("用户名或密码错误")
        if not verify_password(str(user.get("password_hash") or ""), str(password or "")):
            raise InvalidCredentials("用户名或密码错误")
        return self._issue_tokens(user, family_id=str(uuid4()))

    def refresh(self, refresh_token: str) -> AuthTokens:
        """轮换刷新令牌；旧令牌复用会撤销整个令牌族。"""
        token_id, secret = self._parse_refresh_token(refresh_token)
        record = self.repository.get_refresh_token(token_id)
        if record is None:
            raise InvalidCredentials("刷新令牌无效")
        if record.revoked_at is not None:
            self.repository.revoke_token_family(record.family_id, self._now())
            raise InvalidCredentials("刷新令牌已撤销")
        if self._normalize(record.expires_at) <= self._now():
            self.repository.revoke_token_family(record.family_id, self._now())
            raise InvalidCredentials("刷新令牌已过期")
        if not self._token_matches(record, secret):
            self.repository.revoke_token_family(record.family_id, self._now())
            raise InvalidCredentials("刷新令牌无效")
        user = self.repository.get_user_by_id(record.user_id)
        if user is None or user.get("status") != "active":
            self.repository.revoke_token_family(record.family_id, self._now())
            raise InvalidCredentials("账号不可用")
        self.repository.revoke_refresh_token(record.token_id, self._now())
        return self._issue_tokens(user, family_id=record.family_id)

    def logout(self, refresh_token: str) -> None:
        """撤销当前刷新令牌所属令牌族；令牌已失效时保持幂等。"""
        try:
            token_id, _secret = self._parse_refresh_token(refresh_token)
        except InvalidCredentials:
            return
        record = self.repository.get_refresh_token(token_id)
        if record is not None:
            self.repository.revoke_token_family(record.family_id, self._now())

    def get_current_user(self, access_token: str) -> dict:
        """校验访问令牌并返回仍处于启用状态的用户。"""
        try:
            claims = self.access_tokens.decode(access_token)
        except InvalidAccessToken as exc:
            raise InvalidCredentials("访问令牌无效") from exc
        user = self.repository.get_user_by_id(str(claims.get("sub") or ""))
        if (
            user is None
            or user.get("status") != "active"
            or user.get("role") != claims.get("role")
        ):
            raise InvalidCredentials("账号不可用")
        return user

    def _issue_tokens(self, user: dict, *, family_id: str) -> AuthTokens:
        """签发访问令牌和只存哈希的新刷新令牌。"""
        access_token = self.access_tokens.issue(user["user_id"], user["role"])
        refresh_token, token_id, token_hash = self._new_refresh_token()
        now = self._now()
        self.repository.create_refresh_token(
            token_id=token_id,
            family_id=family_id,
            user_id=user["user_id"],
            token_hash=token_hash,
            expires_at=now + self.refresh_ttl,
            now=now,
        )
        safe_user = {key: value for key, value in user.items() if key != "password_hash"}
        return AuthTokens(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.access_tokens.ttl_seconds,
            user=safe_user,
        )

    def _new_refresh_token(self) -> tuple[str, str, str]:
        """生成随机刷新令牌及可存储的 SHA-256 哈希。"""
        token_id = str(uuid4())
        secret = secrets.token_urlsafe(48)
        raw = f"{token_id}.{secret}"
        return raw, token_id, self._hash(secret)

    @staticmethod
    def _parse_refresh_token(value: str) -> tuple[str, str]:
        """拆分刷新令牌，格式错误时不泄露内部原因。"""
        token_id, separator, secret = str(value or "").partition(".")
        if not separator or not token_id or not secret:
            raise InvalidCredentials("刷新令牌无效")
        return token_id, secret

    @staticmethod
    def _token_matches(record: RefreshTokenRecord, secret: str) -> bool:
        """使用常量时间比较刷新令牌哈希。"""
        return secrets.compare_digest(record.token_hash, AuthService._hash(secret))

    @staticmethod
    def _hash(value: str) -> str:
        """刷新令牌只保存 SHA-256 哈希。"""
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_registration(username: str, password: str, role: str) -> str:
        """校验注册字段并返回规范化用户名。"""
        normalized = str(username or "").strip()
        if not USERNAME_PATTERN.fullmatch(normalized):
            raise InvalidRegistration("用户名需为 3-64 位字母、数字、点、横线或下划线")
        if len(str(password or "")) < 8 or len(str(password)) > 128:
            raise InvalidRegistration("密码长度需为 8-128 位")
        if role not in {"user", "admin"}:
            raise InvalidRegistration("角色不合法")
        return normalized

    def _now(self) -> datetime:
        """返回带 UTC 时区的当前时间。"""
        value = self.clock()
        return self._normalize(value)

    @staticmethod
    def _normalize(value: datetime) -> datetime:
        """把数据库读出的无时区时间按 UTC 处理。"""
        return value if value.tzinfo else value.replace(tzinfo=UTC)
